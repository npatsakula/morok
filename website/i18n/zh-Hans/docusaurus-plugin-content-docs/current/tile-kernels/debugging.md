---
sidebar_label: 调试
---

# 调试与验证内核

一个手写内核有多可信，取决于你检查它的能力有多强。[Flash Attention](./flash-attention) 的演练展示了什么样的内核值得手写；本章则讲你如何一步步把它信下来。USE 面孔交给你的是一个融入大图的惰性 `Tensor`，方便归方便，可要在这里问「这一个内核对不对、有多快」却很别扭。`tk` 的 **DEBUG 面孔**正是为此而生：让单个内核针对具体缓冲区运行，把结果读回来，给它计时，并证明一次重构没改变它的行为。

---

## 直接派发：跑一个内核，把字节看个清楚

直接启动 API（`tk/src/launch.rs`）完全绕开张量调度器。你给它一个完成的 `Kernel` 和真实的输入缓冲区，它便渲染、编译、派发，把结果写进一个你能读回的输出缓冲区：

```rust
// The DEBUG face from tk/src/lib.rs. `outs` are written in place.
run_kernel("tile_add", [1, 1, 1], block, &mut [&mut out], &[&input_a, &input_b], build)?;
let values = out.as_vec::<f32>()?;   // read the GPU result straight back
assert_eq!(values, expected);
```

因为这跳过了调度、融合和依赖跟踪，你测到的*就只是你的内核*，而不是一张恰好包含它的图。这份隔离正是要点：数字一旦错了，你想确切知道它就错在*这里*，而不是错在某条融合流水线里的某个角落。

关于这条路径多说一句：跳过*调度器*并不等于跳过*优化器*。`compile` 仍会在你的 `SINK` 上跑生产用的 `optimize_kernel_with_config`——它对一个手工降级的函数体不施加任何调度优化（这正是 `opts_to_apply: Some(vec![])` 这个标记换来的），但依然会执行每个内核在渲染前都需要的那些共享重写，其中就包括索引 dtype 的降级。于是你不靠调度器也拿到了正确的代码。

---

## 在真实硬件上计时

做性能工作时，`CompiledLaunch`（来自 `compile` / `compile_kernel`）暴露的是硬件时间戳，而非挂钟上的估摸：

```rust
// Render + compile once …
let launch = compile_kernel("matmul", grid, block, &mut [&mut c], &[&a, &b], build)?;
// … then dispatch in a loop, outside the timed region.
// SAFETY: the bound buffers stay allocated for `launch`'s lifetime.
unsafe { launch.dispatch(true) }?;
let ns = launch.dispatch_gpu_ns()?;   // Option<u64>: device-measured dispatch time
```

`dispatch_gpu_ns()` 在派发前后读取 GPU 自己的时间戳计数器，所以你测到的是设备上的时间，而不是启动它那一来一回的延迟。criterion 基准则在上一层、经由 `plan.profile` 拿到同一批设备时间戳，用来把一个 `tk` 内核与图原生基线作比较。而这同一批基准在 `cargo bench --profile-time` 下还能做得更多：每个受测的 plan 都会送进完整的分层 profiler——设备时间、roofline、占用率和硬件计数器——按每内核最小值累积，并写成一张表。各层级、环境变量，以及如何接入 criterion，详见 [剖析与基准测试](./profiling)。

:::tip[面向 GPU 专家]
`KernelFingerprint` 是 `SINK` 的 UOp 图的一个*结构化*哈希，它捕捉的是形状（操作、dtype、边）而与实例 ID 无关，因此在不同运行和进程间都稳定。这正是它能当金标准测试键的原因：一次保留行为的重构会复现出同一个指纹，而对所发射 IR 的任何改动都会让它挪位。`dispatch_gpu_ns` 在派发前后读取设备自己的时间戳计数器，所以它测的是设备上的时间，而非启动延迟。
:::

---

## 指纹：证明一次重构保留了行为

手写内核有个微妙的风险：你「整理」了一下构建器代码，内核照样能编译、照样产出看似合理的数字，但*生成的 IR* 却以某种只在日后某个形状、某个架构上才暴露的方式变了。

`KernelFingerprint`（`tk/src/fingerprint.rs`）就是防这一手的。它对一个内核的 UOp 图算出一个确定性的、结构化的哈希，抓的是 SINK 的形状，而非指针标识。你把这个指纹快照成一个金标准值，那么一次本该纯属润色的重构就必须复现它：

```rust
let fp = kernel_fingerprint(&sink);
assert_eq!(fp.digest, GOLDEN_MATMUL_DIGEST);  // structure unchanged ⇒ behavior unchanged
```

指纹一旦挪位，就说明你改了所发射的 IR（无论有意无意），金标准测试会逼你正视它。`tk/src/test/unit/golden.rs` 里的单元测试正是用这一招锁住了 matmul 和 Flash Attention 的图（摘要*以及*节点数）。

---

## 哪个问题用哪个工具

| 你在问…… | 用 |
|----------------|-----|
| 「这个内核产出的数字对吗？」 | `run_kernel` + `as_vec`，与一份参考作比较 |
| 「它在这块 GPU 上有多快？」 | `compile_kernel` + `dispatch_gpu_ns` |
| 「我的重构改动了所发射的 IR 吗？」 | `KernelFingerprint` 金标准测试 |
| 「是*设备/驱动层*在捣乱吗？」 | [AMD 后端 → 调试](../backends/amd/debugging)、[CUDA 后端 → 调试](../backends/cuda/debugging) |

最后一行很重要：本章讲的是调试*内核*，也就是你编写的 IR 和它产出的数字。当问题落在那一层之下时（队列派发、内存故障、驱动、PTX JIT），各后端自己的章节才是该去的地方：[AMD](../backends/amd/debugging) 与 [CUDA](../backends/cuda/debugging)。

---

## 为什么这很重要

手工编写，是拿优化器的安全网去换控制权。DEBUG 面孔就是你安全地做这笔交换的途径：用隔离来定位正确性 bug，用硬件时间戳来撑起经得起推敲的性能论断，用结构化指纹让「我只是整理了一下代码」不会悄悄变成「我改了内核」。有这三样在手，手写内核就和自动调优内核一样可验证。
