---
sidebar_label: 执行流水线
---

# 从 Tensor 到机器码

大多数 ML 框架中，计算是立即发生的。在 PyTorch 里写 `a + b`，它*马上*就执行了——GPU 在你能查看结果之前就已经算完了。这种即时求值方式容易理解，但会错过很多优化机会。编译器怎么优化一个还没看到的完整计算呢？

Morok 走的是相反的路线：**惰性求值**。当你写 `a.try_add(&b)?` 时，什么都不会被计算。Morok 构建的是一个描述*做什么*的图，而不是*何时做*。关键在于调用 `realize()`——这个方法触发整个编译流水线，从高层张量操作一路到 JIT 编译的机器码。

本章追踪这一过程。

```text
tensor.realize()
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  LAZY GRAPH                                             │
│  Tensor ops build UOp DAG (no computation yet)          │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  RANGEIFY                                               │
│  Movement ops → explicit RANGE loops                    │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  KERNEL SPLITTING                                       │
│  Split at STORE boundaries → multiple KERNELs          │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  OPTIMIZATION & CODEGEN                                 │
│  Heuristics/beam → LLVM IR → JIT compile               │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  EXECUTION                                              │
│  Parallel kernel launch → result buffer                │
└─────────────────────────────────────────────────────────┘
```

每个框都是一个独立的阶段。逐一讲解。

---

## 惰性求值：构建计算图

Morok 中的 `Tensor` 非常轻量：

```rust
pub struct Tensor {
    entry: Arc<TensorEntry>,      // Computation graph
    buffer: Option<Arc<Buffer>>,  // Materialized data (if any)
}
```

`entry` 持有包含 UOp 图的 `TensorEntry`——即这个张量所代表的计算。`buffer` 是可选的：惰性张量没有 buffer，只有 realize 后的张量才有。

### 三种创建张量的方式

**1. 输入张量** — 立即分配 buffer：

```rust
let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
// `a.buffer` = Some(Arc<Buffer>) with actual data
```

从数据创建张量时，Morok 分配设备内存并拷贝字节。UOp 图中包含一个指向此分配的 `BUFFER` 节点。

**2. 惰性操作** — 没有 buffer，只有图：

```rust
let b = a.try_add(&a)?;   // b.buffer = None
let c = b.try_mul(&a)?;   // c.buffer = None
```

算术操作不执行任何计算。它们构建 UOp 图：`Binary(Add, a.uop, a.uop)`。张量纯粹作为未来工作的描述而存在。

**3. 变换操作** — 共享原始 buffer：

```rust
let d = a.try_reshape(&[1, 3])?;  // d.buffer = same as a.buffer
```

Reshape、permute 等操作创建的是现有数据的新*视图*。buffer 是共享的；只有 UOp 图会改变以描述新的索引方式。

### 全局注册表

Morok 维护着三个全局映射（无锁，线程安全）：

| 映射 | 键 → 值 | 用途 |
|-----|-------------|---------|
| `TENSORS` | tensor_id → `Weak<TensorEntry>` | 跟踪所有张量，用于图替换 |
| `BUFFERS` | uop_id → `Arc<Buffer>` | 在调度阶段查找 buffer |
| `UOP_TO_TENSOR` | uop_id → tensor_id | 用于查找的二级索引 |

这个注册表支撑了一个关键特性：**全局图替换**。当优化变换了某个 UOp 时，所有引用该 UOp 的张量都会自动看到更新后的版本。不会有过时的引用，也不需要手动更新。

### Hash Consing 实战

由于 UOp 使用 hash consing（基于内容的去重），相同的计算共享内存：

```rust
let x = a.try_add(&b)?;
let y = a.try_add(&b)?;
// x.uop() and y.uop() point to the SAME Arc<UOp>
```

这对缓存很重要：编译 kernel 时按 UOp ID 缓存。Hash consing 意味着相同的计算会自动命中缓存，即使是分别构造的。

---

## Rangeify：让循环显式化

当你写 `tensor.reshape([2, 3]).expand([4, 2, 3]).sum(axis=0)` 时，那些变换操作（reshape、expand）是高层描述。要生成实际的循环，我们需要显式的迭代结构。

**Rangeify** 将变换操作转换为 `RANGE` 循环和 `INDEX` 算术运算。入口点是 `schedule/src/rangeify/transforms.rs` 中的 `rangeify()`。

### 八步流水线

Rangeify 不是单一变换——而是八个协调的 pass：

| Pass | 用途 |
|------|---------|
| **1. 范围分配** | 为每个张量维度创建 RANGE UOp |
| **2. 前置重写** | 移除 DETACH，清理无意义的 RESHAPE |
| **3. 大 reduce 拆分** | 对超大数组的两阶段 reduce（比率 > 32768） |
| **4. 核心 Rangeify** | ReduceAxis → REDUCE，buffer 化，变换操作移除 |
| **5. Buffer 折叠** | 通过 buffer 表达式进行常量传播 |
| **6. 死轴移除** | 过滤不影响输出的范围 |
| **7. 基于代价的 Buffer 移除** | 在有利时内联 buffer（PContig 优化） |
| **8. Reduce 简化** | 将与范围无关的代码提升到 reduce 外部 |

每个 pass 都使用基于模式的重写（参见[基于模式的优化](./optimizations)章节）。模式持续触发直到没有更多匹配，然后开始下一个 pass。

### 变换前后对比

考虑这个张量表达式：

```text
Before: BUFFER.reshape([2, 3]).expand([4, 2, 3]).sum(axis=0)
```

经过 rangeify，变换操作变成显式的索引计算：

```text
After:
STORE
├── INDEX[RANGE(0..2), RANGE(0..3)]
└── REDUCE(Add)
    ├── LOAD
    │   └── INDEX[RANGE(0..4), RANGE(0..2), RANGE(0..3)]
    └── RANGE(0..4, Reduce)
```

`EXPAND` 变成了一个不影响 buffer 索引的 `RANGE(0..4)`——即广播。`RESHAPE` 变成了不同的索引算术。`SUM` 变成了 `REDUCE(Add)`，其中第一个范围标记为 `Reduce` 类型。

### 变换 → 索引算术

每种变换操作有其特定的转换方式：

| 操作 | 转换方式 |
|-----------|----------------|
| **RESHAPE** | 展平/反展平索引表达式 |
| **PERMUTE** | 重新排列 INDEX 中的维度 |
| **EXPAND** | 索引变为 0（或范围不影响索引） |
| **PAD** | WHERE(in_bounds, LOAD, pad_value) |
| **SHRINK** | INDEX 中的偏移调整 |
| **FLIP** | `size - 1 - index` |

经过 rangeify 后，不再有变换操作——只有对索引的算术运算。

---

## Kernel 拆分：寻找边界

一个计算图可能有多个输出，或者需要物化的中间值。**Kernel 拆分**识别这些边界并创建独立的 kernel。

入口点是 `schedule/src/rangeify/kernel.rs` 中的 `run_kernel_split_pipeline()`。

### 两阶段变换

**阶段 1：BUFFERIZE → STORE**

`BUFFERIZE` 节点标记值应该物化的位置。阶段 1 将它们转换为显式的 `STORE` 操作：

```text
Before: BUFFERIZE(computation, ranges)
After:  END(STORE(buffer, INDEX(...), computation), ranges)
```

`END` 包装器捕获哪些范围限定了这个 store 的作用域。buffer 在此阶段被分配并赋予 ID。

**阶段 2：STORE → KERNEL**

每个 `STORE` 变成独立的 kernel：

```text
Before: END(STORE(...), ranges)
After:  KERNEL(SINK(STORE(...)), ranges, buffer_list)
```

`KERNEL` 节点封装了所有内容：计算（作为 `SINK`）、迭代范围，以及该 kernel 读写的 buffer 列表。

### 依赖追踪

当一个 kernel 的输出作为另一个 kernel 的输入时，我们需要依赖追踪：

1. `fix_assign()` 将每个 buffer_id 映射到写入它的 kernel
2. 当 kernel B 读取由 kernel A 写入的 buffer 时，B 依赖于 A
3. `resolve_kernel_dependencies()` 构建依赖图

依赖关系以 `AFTER` 节点出现在 IR 中，确保 kernel 按有效顺序执行。

### Buffer 重编号

每个 kernel 按特定顺序看到 buffer（输出优先，然后是输入）。`renumber_define_globals()` 重新映射 buffer ID 以匹配此顺序：

```text
Original: buffer_3, buffer_1, buffer_7
Kernel view: buffer_0 (output), buffer_1, buffer_2 (inputs)
```

这简化了代码生成——buffer `N` 始终是参数 `N`。

---

## 调度创建：准备执行

kernel 拆分完成后，需要**调度**它们：确定执行顺序、分配 buffer，并准备编译。

`tensor/src/schedule.rs` 中的 `create_schedule()` 生成 `Vec<ScheduleItem>`：

```rust
pub struct ScheduleItem {
    pub kernel: Arc<UOp>,              // KERNEL wrapper
    pub ast: Arc<UOp>,                 // Inner computation (for codegen)
    pub buffers: Vec<Buffer>,          // Device buffers
    pub dependencies: Vec<u64>,        // Producer kernel IDs
    pub fixedvars: HashMap<String, i64>,  // Bound iteration variables
}
```

### Buffer 分配策略

- **输入 buffer**：已经分配好（来自 `Tensor::from_slice`）
- **中间 buffer**：在调度阶段分配（用于 kernel 之间传递的输出）
- **输出 buffer**：分配后注册到最终张量

### 并行组分析

并非所有 kernel 都需要顺序执行。无依赖的 kernel 可以并行运行：

```text
Kernel A (writes buf0)
Kernel B (writes buf1)  ─── no dependency ─── can run in parallel
Kernel C (reads buf0, buf1)  ─── depends on A and B
```

调度器使用 **Kahn 算法** 寻找并行组：

1. 构建 kernel 依赖 DAG
2. 找出所有没有入边的 kernel → 第 1 组
3. 移除第 1 组，重复 → 第 2 组，以此类推

每组 kernel 并行执行，然后开始下一组。

---

## 代码生成：从 UOp 到 LLVM IR

kernel 调度完成后，开始生成实际代码。Morok 目前支持 LLVM 后端：

| 后端 | 编译速度 | 输出质量 | 使用场景 |
|---------|---------------|----------------|----------|
| **LLVM** | 较慢 | 高度优化 | 生产环境 |

`Renderer` trait 抽象了代码生成：

```rust
pub trait Renderer {
    fn render(&self, uop: &Arc<UOp>, name: Option<&str>) -> Result<RenderedKernel>;
}
```

### LLVM CPU 渲染器

LLVM 渲染器（`codegen/src/llvm/cpu/`）遍历 UOp 图并生成 LLVM IR：

```llvm
define void @kernel_0(ptr noalias align 32 %buf0, ptr noalias align 32 %buf1) #0 {
entry:
  br label %loop_0

loop_0:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop_0 ]
  ; ... computation ...
  %i.next = add nsw i32 %i, 1
  %cond = icmp slt i32 %i.next, 128
  br i1 %cond, label %loop_0, label %exit

exit:
  ret void
}
```

每个 buffer 都是直接的 `ptr noalias align 32` 参数——不通过 args 数组间接访问。符号变量（用于动态 shape）和线程 ID 作为额外的类型化参数传递（例如 `i32 %N`）。

### 后优化 Pass

在代码生成之前，13+ 个基于模式的 pass 清理 IR：

| Pass | 用途 |
|------|---------|
| `pm_add_loads` | 将 INDEX 操作包装为 LOAD |
| `pre_expand` | 将 UNROLL/UPCAST 范围转换为显式操作 |
| `devectorize` | 组合连续内存访问 |
| `pm_reduce_devectorize` | 处理向量 reduce（K-vec、bool、水平） |
| `pm_fma_decomposition` | 将 `a*b+c` 转换为融合乘加 |
| `bool_storage_patterns` | 在内存操作中转换 bool ↔ uint8 |

这些 pass 将优化后的 AST 转换为适合代码生成的形式。结果是干净的、向量化的代码，具有正确的内存访问模式。

---

## 执行：运行 Kernel

代码生成产生 LLVM IR 字符串。执行阶段涉及 JIT 编译和 kernel 启动。

### ExecutionPlan

`prepare_execution_plan()` 构建 `ExecutionPlan`：

```rust
pub struct ExecutionPlan {
    kernels: Vec<PreparedKernel>,       // Compiled kernels
    parallel_groups: Vec<ParallelGroup>,
    buffers: Vec<Buffer>,
    output_buffer_idx: usize,
}
```

这个计划是**可复用的**：编译一次，可以用不同的数据多次执行。

### JIT 编译

LLVM 运行时（`runtime/src/llvm.rs`）将 IR 编译为机器码：

1. **解析** LLVM IR 字符串为 module
2. **验证** module 格式正确
3. **优化**，使用 LLVM 的 O3 pass pipeline
4. **JIT 编译**为原生机器码
5. **缓存**，按 (AST ID, device) 复用

```rust
// Simplified JIT flow
let module = Module::parse_ir(context, ir_string)?;
module.verify()?;
pass_manager.run(&module);  // O3 optimization
let function = execution_engine.get_function::<KernelFn>(&name)?;
// Cache: (ast_id, device) → function
```

### 并行执行

kernel 编译完成后，按并行组执行：

```rust
for group in &plan.parallel_groups {
    if group.kernel_indices.len() == 1 {
        // Single kernel: direct call
        execute_kernel(&kernels[group.kernel_indices[0]]);
    } else {
        // Multiple kernels: parallel execution
        rayon::scope(|s| {
            for &idx in &group.kernel_indices {
                s.spawn(|_| execute_kernel(&kernels[idx]));
            }
        });
    }
}
```

无依赖的 kernel 通过 Rayon 的工作窃取调度器并行运行。

### Kernel 缓存

Hash consing 使 kernel 缓存非常高效：

- **键**：`(UOp ID, device string)`
- **存储**：无锁 HashMap（papaya crate）
- **命中率**：高，因为相同的计算共享 UOp ID

当你计算同一个表达式两次时，第二次会命中缓存——无需重新编译。

---

## 完整示例：矩阵乘法

追踪 `C = A @ B` 通过整个流水线。假设 4×4 矩阵。

### 阶段 1：惰性图构建

```rust
let a = Tensor::from_slice(a_data).try_reshape(&[4, 4])?;  // Input buffer allocated
let b = Tensor::from_slice(b_data).try_reshape(&[4, 4])?;  // Input buffer allocated
let c = a.matmul(&b)?;                           // Graph built, no computation
```

此时，`c` 是一个惰性张量，具有如下 UOp 图：

```text
REDUCE_AXIS(Add, axis=2)
└── MUL
    ├── EXPAND(A, [4, 4, 4])    — A: [4, 4] → [4, 1, 4] → [4, 4, 4]
    └── EXPAND(B, [4, 4, 4])    — B: [4, 4] → [1, 4, 4] → [4, 4, 4]
```

### 阶段 2：Rangeify

变换操作变成显式循环：

```text
STORE
├── BUFFER(C)
├── INDEX[RANGE(i, 0..4), RANGE(j, 0..4)]
└── REDUCE(Add)
    ├── MUL
    │   ├── LOAD(A)
    │   │   └── INDEX[RANGE(i), RANGE(k, 0..4, Reduce)]
    │   └── LOAD(B)
    │       └── INDEX[RANGE(k), RANGE(j)]
    └── RANGE(k, Reduce)
```

`i` 和 `j` 范围是输出维度。`k` 范围是规约（收缩）维度。

### 阶段 3：Kernel 拆分

单个 STORE → 单个 KERNEL：

```text
KERNEL
├── SINK(STORE(...))
├── ranges: [i: 0..4, j: 0..4]
└── buffers: [C (output), A (input), B (input)]
```

### 阶段 4：调度

一个 `ScheduleItem`，包含：
- `kernel`：KERNEL UOp
- `ast`：内部的 SINK/STORE
- `buffers`：[C, A, B]
- `dependencies`：[]（没有前置 kernel）

### 阶段 5：优化

启发式优化器应用：
- 向量化：将 j 维度 UPCAST 4 倍
- 循环顺序：确保良好的缓存行为

### 阶段 6：代码生成

生成的 LLVM IR（简化版）：

```llvm
define void @matmul(ptr noalias align 32 %C, ptr noalias align 32 %A, ptr noalias align 32 %B) #0 {
entry:
  br label %loop_i

loop_i:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop_i.end ]
  br label %loop_j

loop_j:
  %j = phi i64 [ 0, %loop_i ], [ %j.next, %loop_k.end ]
  %acc = ... ; initialize accumulator
  br label %loop_k

loop_k:
  %k = phi i64 [ 0, %loop_j ], [ %k.next, %loop_k ]
  %a_val = load float, ptr ...  ; A[i, k]
  %b_val = load float, ptr ...  ; B[k, j]
  %prod = fmul float %a_val, %b_val
  %acc.new = fadd float %acc, %prod
  %k.next = add i64 %k, 1
  %k.cond = icmp slt i64 %k.next, 4
  br i1 %k.cond, label %loop_k, label %loop_k.end

loop_k.end:
  store float %acc.new, ptr ...  ; C[i, j]
  ; ... continue j, i loops
}
```

### 阶段 7：执行

1. JIT 编译 LLVM IR
2. 执行：`kernel([C_ptr, A_ptr, B_ptr], [])`
3. 结果存入 C buffer

总计：一次函数调用，结果就绪。

---

## 对比：其他框架如何执行

| 方面 | PyTorch | JAX | TVM | **Morok** |
|--------|---------|-----|-----|-----------|
| **求值方式** | 即时（立即） | 追踪（jit 装饰器） | 惰性（te.compute） | 惰性（realize） |
| **图捕获** | torch.compile | jax.jit trace | 显式 schedule | 通过操作隐式 |
| **编译** | TorchInductor | XLA 后端 | Auto-scheduler | 模式 + beam |
| **缓存** | 按图哈希 | 按 trace | 按 schedule | 按 AST（hash consing） |
| **并行** | DataParallel/DDP | pmap/pjit | Parallel schedule | 并行组 |

**PyTorch**：默认即时求值，torch.compile 用于优化。TorchInductor 生成 Triton 或 C++ 代码。

**JAX**：函数式变换（jit、grad、vmap）追踪计算。XLA 编译为优化 kernel。

**TVM**：将计算和调度显式分离。Auto-scheduler 搜索好的调度方案。

**Morok**：完全惰性——在 `realize()` 之前什么都不执行。Hash consing 提供自动缓存。基于模式的优化，可选 beam 搜索以获得生产级质量。

---

## 更深层的洞察

流水线体现了几个设计原则：

**惰性求值实现全局优化。** 通过延迟计算，我们在生成代码前看到完整的图。局部决策不会限制全局优化。

**显式循环实现硬件特定的调度。** 变换操作是方便的抽象，但 GPU 需要循环。Rangeify 在两者之间架起了桥梁。

**Hash consing 使缓存自动化。** 相同的计算共享指针，所以缓存键很简单。不需要复杂的图哈希。

**关注点分离使每个阶段保持简单。** Rangeify 不知道 LLVM 的存在。代码生成不知道张量语义。每个阶段只做好一件事。

结果是：一个既强大又可维护的编译流水线。从 `tensor.realize()` 到机器码，每一步都是可见的、可调试的、可扩展的。
