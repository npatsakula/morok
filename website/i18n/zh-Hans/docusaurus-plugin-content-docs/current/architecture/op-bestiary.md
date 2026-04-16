---
sidebar_label: 操作图鉴
---

# 操作图鉴：UOp 操作速查手册

调试 Morok IR 输出时，你会遇到一些从名字上看不太直观的操作。本章记录了那些需要解释的操作，包括签名、字段说明和示例。

**涵盖内容：** 需要解释的操作——循环控制、规约、内存操作、内核结构、向量化、张量核心。

**不涵盖：** 简单的 ALU 操作（`Add`、`Mul`、`Sqrt` 等），它们的行为完全符合直觉。

---

## 循环控制：RANGE 和 END

### RANGE — 循环作用域开启

```rust
Range {
    end: Arc<UOp>,           // loop bound (exclusive)
    axis_id: AxisId,         // identifier for deduplication
    axis_type: AxisType,     // scheduling behavior
    deps: SmallVec<[Arc<UOp>; 2]>,  // range dependencies
}
```

**字段：**

| 字段 | 类型 | 用途 |
|------|------|------|
| `end` | `Arc<UOp>` | 上界（不包含），通常是一个 `CONST` |
| `axis_id` | `AxisId` | 内核分割前为 `Unrenumbered(n)`，之后为 `Renumbered(n)` |
| `axis_type` | `AxisType` | 决定循环的调度方式（见下表） |
| `deps` | `SmallVec<[Arc<UOp>; 2]>` | 该 range 依赖的其他 range |

**AxisType 层级：**

| 类型 | 优先级 | GPU 映射 | 用途 |
|------|--------|----------|------|
| `Outer` | -2 | — | 内核边界标记 |
| `Loop` | -1 | `for` 循环 | 顺序迭代 |
| `Global` | 0 | `blockIdx` | 网格并行 |
| `Thread` | 0 | 线程池 | CPU 并行 |
| `Warp` | 1 | warp/wavefront | 子组并行 |
| `Local` | 2 | `threadIdx` | 工作组并行 |
| `GroupReduce` | 2 | 共享内存 | 两阶段规约 |
| `Upcast` | 3 | SIMD | 向量化 |
| `Reduce` | 4 | 累加器 | 规约维度 |
| `Unroll` | 5 | 展开 | 循环展开 |

优先级决定循环嵌套顺序——值越小越在外层。

**示例：**
```text
RANGE(end=128, axis_id=R0, type=Global)
└── CONST(128) : Index
```

### END — 循环作用域关闭

```rust
End {
    computation: Arc<UOp>,              // value computed inside loop
    ranges: SmallVec<[Arc<UOp>; 4]>,    // ranges being closed
}
```

END 关闭一个或多个 RANGE 作用域，并将其从活跃集合中移除。可以同时关闭多个 range。

**示例：**
```text
END
├── STORE(...)           — computation
├── RANGE(R0, Global)    — first range closed
└── RANGE(R1, Local)     — second range closed
```

---

## 规约：REDUCE 与 REDUCE_AXIS

两个名字相似的操作，用途不同。

### REDUCE_AXIS — 张量维度规约（高层）

```rust
ReduceAxis {
    src: Arc<UOp>,           // input tensor
    reduce_op: ReduceOp,     // Add, Mul, Max, Min
    axes: Vec<usize>,        // axes to reduce
}
```

用于 rangeify **之前**。对张量维度进行操作，类似于 NumPy 的 `.sum(axis=0)`。

**示例：**
```text
REDUCE_AXIS(Add, axes=[1])
└── BUFFER[10, 20] : Float32
```

将一个 `[10, 20]` 的张量沿 axis 1 求和，规约为 `[10]`。

### REDUCE — Range 迭代规约（底层）

```rust
Reduce {
    src: Arc<UOp>,                      // value to accumulate
    ranges: SmallVec<[Arc<UOp>; 4]>,    // ranges being reduced
    reduce_op: ReduceOp,                // Add, Mul, Max, Min
}
```

用于 rangeify **之后**。在 RANGE 迭代中累加值，并关闭指定的 range。

**ReduceOp 变体：**

| 操作 | 单位元 | 运算 | Tinygrad |
|------|--------|------|----------|
| `Add` | 0 | `acc + value` | ✓ |
| `Mul` | 1 | `acc * value` | ✓ |
| `Max` | -∞ | `max(acc, value)` | ✓ |
| `Min` | +∞ | `min(acc, value)` | 仅 Morok |

> **兼容性：** Tinygrad 的规范将 REDUCE_AXIS 限制为 `{Add, Mul, Max}`。Morok 额外支持 `Min`。

**示例：**
```text
REDUCE(Add)
├── MUL                      — value to accumulate
│   ├── LOAD(A, ...)
│   └── LOAD(B, ...)
└── RANGE(R2, Reduce)        — range being reduced
    └── CONST(64)
```

### ALLREDUCE — 跨设备规约

```rust
AllReduce {
    src: Arc<UOp>,           // local partial result
    device: Arc<UOp>,        // device specification
    reduce_op: ReduceOp,     // reduction operation
}
```

在多个设备之间执行分布式规约，用于多 GPU 训练。

---

## Buffer 操作

### BUFFER — Buffer 声明

```rust
Buffer {
    unique: Arc<UOp>,        // UNIQUE op for identity
    device: Arc<UOp>,        // DEVICE op
    size: usize,             // total element count
}
```

声明一个用于张量存储的 buffer。`unique` 字段确保即使 size/device 相同，不同 buffer 也能区分。

### BUFFERIZE — 物化标记

```rust
Bufferize {
    compute: Arc<UOp>,                  // computation to materialize
    ranges: SmallVec<[Arc<UOp>; 4]>,    // output dimensions
    opts: BufferizeOpts,                // address space, device
}
```

标记计算结果应物化到内存的位置，触发内核分割。

**BufferizeOpts：**

| 字段 | 类型 | 用途 |
|------|------|------|
| `device` | `Option<DeviceSpec>` | 目标设备，`None` 表示本地 |
| `addrspace` | `AddrSpace` | `Global`（设备）或 `Local`（共享） |

**示例：**
```text
BUFFERIZE(opts={addrspace=Global})
├── REDUCE(Add, ...)         — computation
├── RANGE(R0, Global)        — output dim 0
└── RANGE(R1, Global)        — output dim 1
```

### INDEX — 多维 Buffer 访问

```rust
Index {
    buffer: Arc<UOp>,                   // BUFFER or PARAM
    indices: SmallVec<[Arc<UOp>; 4]>,   // index per dimension
    gate: Option<Arc<UOp>>,             // optional predicate
}
```

从多维索引计算内存地址。返回元素 dtype（不是指针）。

**示例：**
```text
INDEX : Float32
├── PARAM(0)
├── RANGE(R0, Global)        — index for dim 0
├── RANGE(R1, Loop)          — index for dim 1
└── MUL(...)                 — index for dim 2
```

### POINTER_INDEX — 底层指针算术

```rust
PointerIndex {
    ptr: Arc<UOp>,           // base pointer
    offset: Arc<UOp>,        // byte offset
}
```

直接指针算术。在线性化后、索引被展平时使用。

> **兼容性：** Tinygrad 使用 `INDEX` 加 `ptr=True` 标志，而不是独立的操作。

### LOAD — 内存读取

```rust
Load {
    buffer: Arc<UOp>,        // buffer or pointer
    index: Arc<UOp>,         // INDEX op
    alt: Option<Arc<UOp>>,   // alternative value for gated loads
}
```

从 buffer 的指定索引处读取值。对于门控加载，`alt` 字段提供当 INDEX 的 `gate` 为 false 时的替代值（完全跳过内存访问）。

**示例：**
```text
LOAD : Float32
├── PARAM(1)
└── INDEX
    ├── PARAM(1)
    ├── RANGE(R0)
    └── RANGE(R2)
```

### STORE — 内存写入

```rust
Store {
    index: Arc<UOp>,                    // INDEX op (buffer accessed via index.src[0])
    value: Arc<UOp>,                    // value to write
    ranges: SmallVec<[Arc<UOp>; 4]>,    // ranges being closed
}
```

向 buffer 写入值。Buffer 通过 INDEX 节点（`index.src[0]`）访问，而不是单独的字段。STORE 关闭指定的 range，这些 range 代表输出迭代维度。ranges 字段用于输出 upcast：当包含 `Range(Upcast)` 时，展开阶段会将其变为 `UNROLL`，再通过 `CONTRACT` 收缩。

对于门控写入，使用带 gate 的 INDEX（INDEX 有一个可选的 `gate` 字段）。

> **兼容性：** Morok 的 STORE 没有单独的 `buffer` 字段——源为：index=0, value=1, ranges=2+（range_start=2）。Tinygrad 的布局类似。

**示例：**
```text
STORE
├── INDEX[R0, R1]            — write address (buffer via index.src[0])
├── REDUCE(Add, ...)         — value
├── RANGE(R0, Global)        — output dim 0 (closed)
└── RANGE(R1, Global)        — output dim 1 (closed)
```

---

## 内核结构

### KERNEL — 内核包装器

```rust
Kernel {
    sources: SmallVec<[Arc<UOp>; 4]>,   // arguments
    ast: Arc<UOp>,                       // computation (usually SINK)
}
```

封装一个完整的内核用于代码生成。sources 是内核参数（`Param`、`DefineLocal`、`DefineVar`）。注意：在 batching_support PR 中，`Param` 替代了 `DefineGlobal`，通过擦除 buffer 身份来实现内核去重。

**示例：**
```text
KERNEL
├── PARAM(slot=0, size=1024) — output buffer arg
├── PARAM(slot=1, size=1024) — input A arg
├── PARAM(slot=2, size=1024) — input B arg
└── SINK                     — computation
    └── STORE(...)
```

### SINK — 多根收集器

```rust
Sink {
    sources: SmallVec<[Arc<UOp>; 4]>,
}
```

将多个输出收集到一个根节点。每个内核的 `ast` 通常是一个包含 STORE 操作的 SINK。

**示例：**
```text
SINK
├── STORE(output_0, ...)
├── STORE(output_1, ...)
└── STORE(output_2, ...)
```

### AFTER — 依赖标记

```rust
After {
    passthrough: Arc<UOp>,              // value that flows through
    deps: SmallVec<[Arc<UOp>; 4]>,      // operations that must complete
}
```

表达内核之间的执行依赖，不涉及数据依赖。`passthrough` 值原样返回，但必须在所有 `deps` 完成后才执行。

**示例：**
```text
SINK
├── AFTER
│   ├── PARAM(0)     — passthrough (buffer reference)
│   └── KERNEL(...)          — must complete first
└── KERNEL(...)              — can use buffer after AFTER
```

### BARRIER — 同步栅栏

```rust
Barrier {
    src: Arc<UOp>,                      // value passing through
    deps: SmallVec<[Arc<UOp>; 4]>,      // operations to wait for
}
```

GPU 工作组同步。确保工作组中的所有线程到达栅栏后才继续执行。

---

## 向量操作

### VECTORIZE — 从标量创建向量

```rust
Vectorize {
    elements: SmallVec<[Arc<UOp>; 4]>,
}
```

将 N 个标量值组合成一个大小为 N 的向量。所有元素必须具有相同的基础 dtype。

**示例：**
```text
VECTORIZE : <4 x Float32>
├── CONST(1.0)
├── CONST(2.0)
├── CONST(3.0)
└── CONST(4.0)
```

### GEP — Get Element Pointer（向量元素提取）

```rust
Gep {
    vector: Arc<UOp>,        // source vector
    indices: Vec<usize>,     // positions to extract
}
```

从向量中提取元素：
- 单个索引 → 标量
- 多个索引 → 更小的向量

**示例：**
```text
GEP([0, 2]) : <2 x Float32>
└── VECTORIZE : <4 x Float32>
    └── ...
```

### VConst — 向量常量

```rust
VConst {
    values: Vec<ConstValue>,
}
```

编译期常量向量。比用 `CONST` 节点构建 `VECTORIZE` 更高效。

### CAT — 向量拼接

```rust
Cat {
    sources: SmallVec<[Arc<UOp>; 4]>,
}
```

将多个向量拼接成更大的向量。输出的 `vcount` = 各输入 `vcount` 之和。

**示例：**
```text
CAT : <8 x Float32>
├── VECTORIZE : <4 x Float32>
└── VECTORIZE : <4 x Float32>
```

### PtrCat — 指针拼接

```rust
PtrCat {
    sources: SmallVec<[Arc<UOp>; 4]>,
}
```

将内存访问分组以实现向量化 load/store。由 devectorizer pass 使用。

---

## 展开：UNROLL 和 CONTRACT

### UNROLL — 跨迭代展开计算

```rust
Unroll {
    src: Arc<UOp>,                       // computation to expand
    unroll_axes: Vec<(usize, usize)>,    // (axis_index, factor) pairs
}
```

为不同的迭代值创建计算的多个副本，用于循环展开优化。

**示例：** `UNROLL(unroll_axes=[(0, 4)])` 将计算展开 4 次，使用不同的索引值。

### CONTRACT — 将展开的值收缩为向量

```rust
Contract {
    src: Arc<UOp>,                       // unrolled computation
    upcast_ranges: Vec<(usize, usize)>,  // (axis_index, factor) pairs
}
```

UNROLL 的逆操作——将展开的标量值收集成一个向量。输出向量大小 = 各 factor 之积。

**示例：**
```text
CONTRACT(upcast_ranges=[(0, 4)]) : <4 x Float32>
└── UNROLL(unroll_axes=[(0, 4)])
    └── LOAD(...)
```

这个模式实现了 load 的向量化：展开 4 次迭代，然后将结果打包成 4 元素向量。

---

## 张量核心：WMMA

### WMMA — Warp 矩阵乘累加

```rust
Wmma {
    a: Arc<UOp>,             // matrix A fragment
    b: Arc<UOp>,             // matrix B fragment
    c: Arc<UOp>,             // accumulator C fragment
    metadata: WmmaMetadata,  // hardware configuration
}
```

硬件张量核心操作：`D = A × B + C`。需要特定的矩阵形状和数据布局。

**WmmaMetadata 字段：**

| 字段 | 类型 | 用途 |
|------|------|------|
| `name` | `String` | 指令名称（如 `"__hmma..."`） |
| `dims` | `(N, M, K)` | 矩阵维度（如 `(16, 16, 16)`） |
| `dtype_in` | `DType` | 输入矩阵精度（如 `Float16`） |
| `dtype_out` | `DType` | 输出精度（如 `Float32`） |
| `device` | `String` | 目标设备字符串 |
| `threads` | `usize` | 每个 warp 的线程数（通常 32） |
| `upcast_axes` | `WmmaUpcastAxes` | 各操作数的向量化信息（字段：`a`、`b`、`c`） |
| `reduce_axes` | `Vec<(usize, usize)>` | 收缩轴 |
| `tile_grid` | `(usize, usize)` | 多 FMA 批处理网格（默认 (1,1)） |

**示例：**
```text
WMMA(dims=(16, 16, 16), dtype_in=Float16, dtype_out=Float32)
├── A fragment : <8 x Float16>
├── B fragment : <8 x Float16>
└── C accumulator : <8 x Float32>
```

---

## 控制流

### IF / ENDIF — 条件执行

```rust
If {
    condition: Arc<UOp>,                // boolean predicate
    body: SmallVec<[Arc<UOp>; 4]>,      // operations to execute
}

EndIf {
    if_op: Arc<UOp>,         // corresponding IF op
}
```

仅在条件为真时执行 body。用于边界检查和稀疏操作。

**示例：**
```text
IF
├── LT(idx, bound)           — condition (src[0])
├── STORE(...)               — body[0]
└── STORE(...)               — body[1]

ENDIF
└── IF(...)                  — references IF op
```

---

## 定义操作

### PARAM — Buffer 参数

```rust
Param { slot: usize, size: usize, device: Option<Arc<UOp>> }
```

归一化的 buffer 参数——对输入/输出 buffer 的位置引用。
由预调度归一化（BUFFER→PARAM）创建，通过擦除 buffer 身份，
实现对不同 buffer 上相同计算的结构性去重。
`slot` 是内核参数列表中的位置，`size` 是元素数量。

### DEFINE_LOCAL — 共享内存分配

```rust
DefineLocal(usize)           // local memory index
```

GPU 共享内存（LDS）分配。在工作组内可见。

### DEFINE_VAR — 符号运行时变量

```rust
DefineVar {
    name: String,            // variable name
    min_val: i64,            // minimum bound
    max_val: i64,            // maximum bound
}
```

带已知范围的运行时变量。用于已知边界的动态 shape。

**示例：**
```text
DEFINE_VAR(name="batch_size", min=1, max=128) : Index
```

### DEFINE_REG — 寄存器分配

```rust
DefineReg {
    size: usize,             // register size
    id: usize,               // unique accumulator ID
}
```

分配一个寄存器用于中间存储。`id` 字段用于区分相同 dtype 的寄存器——没有它的话，两个相同 dtype 的 reduce 会因为 hash consing 共享同一个 DEFINE_REG。用于代码生成。

### BIND — 变量绑定

```rust
Bind {
    var: Arc<UOp>,           // DEFINE_VAR
    value: Arc<UOp>,         // concrete value
}
```

在运行时将符号变量绑定到具体值。

---

## 特殊操作

### SPECIAL — 硬件提供的值

```rust
Special {
    end: Arc<UOp>,           // upper bound for this dimension
    name: String,            // e.g., "blockIdx.x", "threadIdx.y"
}
```

访问硬件提供的值（线程/块索引）。不是循环——硬件直接提供该值。

**示例：**
```text
SPECIAL(name="blockIdx.x", end=128) : Index
└── CONST(128)
```

### UNIQUE — 标识标记

```rust
Unique(usize)                // unique identifier
```

为 buffer 消歧创建唯一标识。具有不同 UNIQUE 值的两个 buffer 即使其他属性完全相同也是不同的。

### DEVICE — 设备规格

```rust
Device(DeviceSpec)           // device specification
```

指定计算的目标设备。

---

## 移动操作

高层张量 shape 变换。在 rangeify 阶段会被转换为显式的 INDEX 操作。

| 操作 | 签名 | 用途 |
|------|------|------|
| `Reshape` | `{ src, new_shape }` | 改变 shape，元素不变 |
| `Permute` | `{ src, axes: Vec<usize> }` | 转置/重排轴 |
| `Expand` | `{ src, new_shape }` | 广播到更大的 shape |
| `Pad` | `{ src, begin_pads, end_pads }` | 添加填充 |
| `Shrink` | `{ src, begins, ends }` | 提取子区域 |
| `Flip` | `{ src, axes: Vec<bool> }` | 沿轴翻转 |

**示例：** RESHAPE
```text
RESHAPE(new_shape=[6, 4]) : Shape[6, 4]
├── BUFFER[2, 3, 4] : Float32
└── CONST([6, 4]) : Shape
```

---

## 其他操作

以下操作存在于 `Op` 枚举中，但它们要么是内部实现，要么在调试中很少遇到：

| 操作 | 用途 |
|------|------|
| `Copy` | 显式复制值 |
| `BufferView` | 带 offset/stride 的 buffer 视图 |
| `MStack` | 内存栈分配 |
| `MSelect` | 内存选择（条件内存访问） |
| `Multi` | 多输出操作 |
| `Assign` | 变量赋值 |
| `Group` | 用于调度的操作分组 |
| `Detach` | 从图中分离（阻止优化穿透） |
| `Contiguous` | 标记数据连续的提示 |
| `ContiguousBackward` | contiguous 提示的反向传播 |
| `Precast` | 类型转换的预转型 |
| `Custom` / `CustomI` | 自定义操作扩展 |

---

## 速查表

### 按类别

| 类别 | 操作 |
|------|------|
| **循环控制** | `RANGE`, `END` |
| **规约** | `REDUCE_AXIS`, `REDUCE`, `ALLREDUCE` |
| **内存** | `BUFFER`, `BUFFERIZE`, `INDEX`, `POINTER_INDEX`, `LOAD`, `STORE` |
| **内核** | `KERNEL`, `SINK`, `AFTER`, `BARRIER` |
| **向量** | `VECTORIZE`, `GEP`, `VCONST`, `CAT`, `PTRCAT` |
| **展开** | `UNROLL`, `CONTRACT` |
| **硬件** | `WMMA`, `SPECIAL` |
| **控制** | `IF`, `ENDIF` |
| **定义** | `PARAM`, `DEFINE_LOCAL`, `DEFINE_VAR`, `DEFINE_REG`, `BIND`, `UNIQUE`, `DEVICE` |
| **移动** | `RESHAPE`, `PERMUTE`, `EXPAND`, `PAD`, `SHRINK`, `FLIP` |
| **ALU** | `Unary(...)`, `Binary(...)`, `Ternary(...)`, `Cast`, `BitCast` |

### Range 终止操作

关闭 RANGE 作用域（从活跃集合中移除 range）的操作：

| 操作 | Range 起始索引 |
|------|----------------|
| `BUFFERIZE` | 1 (compute=0, ranges=1+) |
| `REDUCE` | 1 (src=0, ranges=1+) |
| `STORE` | 2 (index=0, value=1, ranges=2+) |
| `WMMA` | 3 (a=0, b=1, c=2) |
| `END` | 1 (computation=0, ranges=1+) |

### 可展开操作

通过计算图传播 UNROLL 的操作：

- ALU：`Unary`、`Binary`、`Ternary`
- 类型：`Cast`、`BitCast`
- 向量：`Gep`、`Vectorize`
- 内存：`Load`、`Store`、`Index`、`PointerIndex`
- 控制：`Reduce`、`End`、`After`
- Buffer：`Bufferize`
- 硬件：`Wmma`
