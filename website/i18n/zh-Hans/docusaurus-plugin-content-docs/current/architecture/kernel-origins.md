---
sidebar_label: 内核来源
---

# 内核来源

profile 报告 `r_128_3_32_4_2_2_2_4_4_192_2` 耗时 100 ms，说明的只是内核的形状，而不是它归谁所有。来源回答的正是后一个问题：凡是 dispatch 出去的内核，都知道自己是为哪个模块路径、哪个调用点、哪个 ONNX 节点而构建，profiler 也就能沿着这条路径把时间汇总起来——按层、按块、按阶段。

本页是使用指南：如何开启、如何给模型加标注、如何读懂输出。其机制——每个节点上一个参与 hash-cons 的字段，到内核切分处再被剥除——文末有简要说明，完整文档见 [IR 设计](./ir-design)与[操作图鉴](./op-bestiary)两页。

---

## 开启

捕获默认关闭，关闭时零开销：节点不携带来源，哈希与不启用该特性的构建逐字节相同。两个开关：

| 开关 | 作用 |
|--------|--------|
| `SVOD_ORIGIN=1` | 对进程的每个线程开启捕获 |
| `SVOD_ORIGIN_DEPTH=<n>` | 汇总保留前 `n` 个路径段（未设置或 `0` = 完整路径） |

```bash
SVOD_DEVICE=AMD:0 SVOD_ORIGIN=1 cargo run --release -p svod-model --example gigaam_infer -- \
    audio.wav --profile --origin-depth 3 --profile-json profile.json
```

测试里只为当前线程切换捕获，好让并行的测试各自保持图身份：

```rust
let _capture = svod_ir::origin::capture_for_thread(true); // restored on drop
```

---

## 来源从哪里来

来源是一条帧路径，根在最前。每个帧是下列之一：

| 帧 | 渲染为 | 由谁打开 |
|-------|-------------|-----------|
| `Module` | `encoder.layers.3.ffn1` | 模型代码，每个模块一段 |
| `Label` | `ctc_head`、`initializer` | 流水线阶段、ONNX 导入器、embedder |
| `Onnx` | `/encoder/Conv` 或 `#12:MatMul` | ONNX 导入器，每个节点和子图分支各一个 |
| `Call` | `@ matmul model/src/gigaam/encoder.rs:262` | 每个公开的 `Tensor` 操作，自动打开 |

`Call` 帧位于模块路径之下，是扁平的一层 file:line。公开操作在入口处打开它，以最外层为准，因此建立在其他操作之上的操作（`linear` 之于 `matmul`）只记下用户那一行，绝不会记成 svod 自己的源码。它上面的模块层则由模型代码添加。

### 为 Rust 模型加标注

在 `forward` 里为每个模块打开一个作用域，名字照着它的 state-dict 前缀来取。模型 crate 里就有做这件事的辅助函数：

```rust
use svod_ir::origin::OriginScope;
use crate::state::{scoped, scoped_index};

fn forward(&self, x: &Tensor) -> Result<Tensor> {
    let x = scoped("subsampling", || self.subsampling.forward(x))?;
    let mut x = x;
    for (i, layer) in self.layers.iter().enumerate() {
        x = scoped_index("layers", i, || layer.forward(&x))?;   // layers.0, layers.1, …
    }
    scoped("final_norm", || self.final_norm.forward(&x))
}
```

每个模块只打开属于自己的那一段，嵌套会把完整路径重新拼起来，因此 profile 打印的路径正好等于它所触及权重的 state-dict 键前缀。GigaAM 与 Whisper 都是这样标注的，还有一个测试断言两组路径彼此一致。

流水线阶段是根部的标签：

```rust
let _stage = OriginScope::label("ctc_head");
let plan = model.prepare_with_config(&config)?;   // every kernel below is ctc_head.…
```

在任何作用域之外构建的东西都落到 `<unattributed>` 行。

### ONNX 图

无需任何操作。导入器为每个节点打开一个 `Onnx` 帧（索引、名称、操作类型、域、opset），并在拥有子图的那个节点之下为每个子图分支（`then_branch`、`else_branch`）打开一个 `Label`，于是一个 `If` 的主体读作 `#7:If.then_branch.#0:Add`。初始化器与图输入位于 `initializer` 和 `input` 之下。

### 手写内核

`tk` 内核的归属取自构建它时处于活动状态的作用域——与图内核同一条规则。调度器看不到它的主体，因此由内核构造器自行收集并剥除来源；两个层启动同一个手写内核，仍然共享同一份编译好的程序。

---

## 读懂输出

开启捕获后，`--profile` 先打印常规的逐内核表格，再附上两份汇总。样例取自 GigaAM v3 encoder，f16，gfx1151 上的一个 60 秒窗口，在深度 3 处截断：

```
519 dispatches (519 GPU-stamped), total 444.237 ms
  total ms  count    mean µs      %  name
   103.183     16     6448.9   23.2  r_128_3_32_4_2_2_2_4_4_192_2n1
   100.305     16     6269.1   22.6  r_128_3_32_4_2_2_2_4_4_192_2
    80.530     32     2516.6   18.1  r_128_12_32_4_2_2_2_4_4_48_2
    …
origin rollup (depth 3, exclusive; rows sum to the total):
  total ms  count    mean µs      %  origin path
    27.833     32      869.8    6.3  ctc_head.GigaAmCtcJit.layers.3
    27.678     32      864.9    6.2  ctc_head.GigaAmCtcJit.layers.9
    27.620     32      863.1    6.2  ctc_head.GigaAmCtcJit.layers.0
    …
    23.334      2    11666.8    5.3  ctc_head.GigaAmCtcJit.subsampling
     0.661      4      165.2    0.1  ctc_head.GigaAmCtcJit.head
     0.131      1      131.0    0.0  ctc_head.GigaAmCtcJit
     0.007      1        6.6    0.0  <unattributed>
origin rollup (depth 3, inclusive; parents contain children, rows overlap):
  total ms  count    mean µs      %  origin path
   444.237    519      855.9  100.0  ctc_head
   444.237    519      855.9  100.0  ctc_head.GigaAmCtcJit
    27.833     32      869.8    6.3  ctc_head.GigaAmCtcJit.layers.3
    …
```

读法：

- **Exclusive** 把每次 dispatch 只计一次，记到它的*主要*来源上，也就是产生该内核所存储值的那个作用域。各行把总量恰好切分完，因此十六行 `layers.N` 加上 `subsampling`、`head` 以及余下的 `GigaAmCtcJit` 行，合起来就是 444 ms。十六个层、每层 32 次 dispatch，正好是整个 encoder；层与层之间的差距（25.3 到 27.8 ms）是真实的，也是首先该看的地方。
- **Inclusive** 把一次 dispatch 记到融合进它的每个来源的所有祖先上。父行包含子行，所以 `ctc_head` 是 100 %，各行互相重叠。用它可以看出某个块有多少时间藏在跨模块边界融合的内核里。
- **深度**是保留的路径段数。这里深度 3 给出逐层的行；深度 4 会把一层拆成 `ffn1`、`mhsa`、`conv`、`ffn2`、`final_norm`；叶子保留完整路径。`Call` 帧从不构成汇总键——它们只是内核行和 JSON 里的细节。
- 融合了两个模块的内核，exclusive 只记给它所存储值所属的那个模块（残差加法落在层上，而不是 `ffn2`），inclusive 则两个都记。

`Whisper` 通过 `render_table()` 打印同样的段落；任何 `RunProfile` 都可以。

### JSON

`--profile-json out.json`（或 `RunProfile::to_json()`）为每次运行写出一个文档：

```json
{
  "origin_depth": 3,
  "stages": [{
    "name": "ctc_head", "wall_ms": 463.8, "gpu_ms": 444.2, "dispatches": 519,
    "kernels": [{
      "name": "r_128_3_32_4_2_2_2_4_4_192_2", "count": 1, "total_ms": 6.3,
      "origin": "ctc_head.GigaAmCtcJit.layers.3 @ add model/src/gigaam/encoder.rs:746",
      "origin_id": 41, "origins": ["…"], "origin_ids": [41, 39]
    }],
    "origins_exclusive": [{ "path": "ctc_head.GigaAmCtcJit.layers.3", "count": 32, "total_ms": 27.8, "percent": 6.3, "kernels": [] }],
    "origins_inclusive": []
  }],
  "origins": [{ "id": 41, "parent": 40, "frame": { "Module": { "name": "layers.3" } } }]
}
```

内核行以入口点*和*主要来源共同作键，因此同一个程序在每个 dispatch 过它的作用域下各出现一次。`origins` 只保存本次运行引用到的那些帧，并按 `parent` 关系闭合，因此脱离写出该文件的进程也能解析这些 id。

---

## 线程

捕获状态是每线程的：开关、当前作用域，以及该作用域是不是一个调用帧。作用域不会跟着工作跑到别的线程上；作用域 guard 是 `!Send` 的，只在打开它的那个线程上做恢复。由此推出几条规则：

- 在打开了作用域的那个线程上构建图。GigaAM 与 Whisper 就是这么做的；围绕 `prepare_with_config` 打开的阶段标签，会覆盖其中构建的一切。
- 调度与编译脱离作用域运行（`OriginScope::suspend`），调用方和 rayon worker 一视同仁，因此外围作用域绝不会渗进内核主体；到那时归属早已收集到 CALL 上。
- 要把作用域带到你自己 spawn 的 worker 上，就捕获 `origin::current()`，再在那边用 `origin::install(id)` 装回去。worker 和其他线程一样，按 `SVOD_ORIGIN` 初始化自己的开关。
- BEAM 搜索在子进程中针对不带来源的内核主体运行，从不接触任何作用域。
- **异步代码：**作用域必须嵌套，所以不要跨 `.await` 持有作用域。先打开作用域，同步把图构建完，drop 掉，再去 await。guard 是 `!Send` 的，跨 await 让 guard 存活的 future 无法在多线程执行器上 spawn；同一线程上两个任务交错使用作用域时（一个 guard 在更晚打开的那个仍处于活动状态时被 drop），debug 构建会 panic。svod 的图构建本来就是同步的，代码的自然写法已经满足这一点。

---

## 代价与取舍

- **关闭时：**没有任何代价。每个节点一次 thread-local 读取，不分配，哈希不变。
- **开启时：**每进入一次作用域做一次 interning（arena 上的一把互斥锁，每次前向数百次），为调用帧在每个公开操作上做一次 thread-local 写入，以及在切分处为每个内核做一次拓扑排序来收集并集。GigaAM 的 dispatch 次数与 GPU 时间在捕获开启和关闭时完全一致。
- **节点身份会变。**来源是节点身份的一部分，因此不同作用域下构建的两个相同表达式，在切分把来源剥掉之前是两个节点。内核程序不受影响——剥除会恢复去重——但那种在每个调用点重建同一表达式的辅助函数（一次 mask clamp、一次 table cast、一次 input copy）会让它在每个作用域下各物化一遍。这类辅助函数要么放在 `OriginScope::suspend()` 下运行，要么让副本继承其生产者的来源；`custom_kernel` 对自己的输入已经是后一种做法。出于同样的道理，常量、缓冲区和参数从不携带来源。
- **依赖结构身份的测试**（两个手工构建、预期会 hash-cons 成同一个节点的图）应当以 `capture_for_thread(false)` 运行。

---

## 一段话讲清原理

每个在作用域活动期间构建的 `UOp` 都会存下该作用域 4 字节的 `OriginId`，并把它折进自己的内容哈希，因此来自不同作用域的相同子图直到 rangeify 都保持彼此有别。在内核切分处，`split_store` 把主体遍历一遍，取所存储值的来源作为主要来源、取并集作为集合，把两者一并盖到内核 `CALL` 的 `CallInfo` 上，再重建一份清空了来源的主体。切分之后的一切——优化器、BEAM、代码生成、每一层内核缓存——看到的都是不带来源的 AST。plan 把 CALL 的归属复制到每个 prepared op 上，profiler 复制到每个 `KernelProfile` 上，汇总则把父链截断到所要求的深度。
