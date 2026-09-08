---
sidebar_label: Flash Attention
---

# 实战范例：Flash Attention

Flash Attention 正是那个为 `tk` 的存在撑起理由的内核，也就是 [概览](./overview) 点名*无法*表达成单一可调度规约的那个，是手工编写面孔之所以存在的全部缘由。本章带你走一遍：它难在哪里，tile 抽象如何应对，以及 [Wave32 与 Wave64](./wave-portability) 的分歧在何处实打实地显现。

我们要讲的是 `tk/src/kernels/fa.rs` 里的前向内核，从 USE 面孔的 `flash_attention(q, k, v)` 进入。它面向 gfx942（CDNA3）、gfx1151（RDNA3.5）与 CUDA `sm_80+` 构建；每 warp 的 `(q_blk, kv_blk)` tile 由 `FaPolicy` 按设备选定——只有当更高的那个 tile 的共享内存缓冲区装得下、且启动网格已经铺满设备的计算单元时才选它，否则退回基线的 `{16, 32}`。

---

## 为什么 attention 无法被自动调优

朴素的 attention 是 `softmax(QKᵀ) · V`。直白地写出来，意思就是：构造完整的 `N×N` 得分矩阵，对它做 softmax，再乘以 `V`。这个得分矩阵大得吓人，而且从来不必一次性全部存在，于是 Flash Attention 流式扫过一块块 key 和 value，*增量地*维护 softmax。

「增量」这个词正是症结。softmax 的归一化依赖于*所有* key 上的最大值与求和，可我们一次只看到一块。于是我们维护一份当前的统计量，边推进边修正结果。这就是**在线 softmax**，而它是一个递推：每一个 KV 块都要读取并更新上一块产出的状态。

优化器能做的动作只是「把这个 `REDUCE` 分块、展开」。可这里根本没有 `REDUCE` 供它分块，有的是一个循环，循环体依赖自己上一轮的迭代。搜索找不出它，你只能亲手写。

---

## 用 tile 表达的算法

内核给每个 wave 分一块查询，再一块块地走过 key/value。对每一个 KV 块，它跑下面这段循环体，全程都是 tile 操作：

```text
for each block of K, V:                          ┌─ everything here is a tile op
    S   = Q · Kᵀ                                 │  (mma into a register accumulator)
    S   = mask(S)                                │  causal + key-padding masks
    m'  = max(m, rowmax(S))                      │  update running max  (cross-lane reduce)
    P   = exp2(S - m')                           │  rescale to the new max (base-2 exp)
    l   = l * exp2(m - m') + rowsum(P)           │  update running sum
    O   = O * exp2(m - m') + P · V               │  rescale accumulator, accumulate
    m   = m'                                     │
O = O / l                                        └─ final normalize
```

每块两次矩阵乘法（`Q·Kᵀ` 和 `P·V`）、两次跨 lane 行规约（最大值和求和），以及每当当前最大值变动时对输出累加器的一次重新缩放。那个 `exp2`（以 2 为底的指数）是刻意为之：温度被预先折进 `Q`，这样就能直接用上硬件的快速 `exp2` 单元。

那几行里的每一行，都是对 tile 的一个 `Group` 操作：乘法用 `mma`，行最大值/求和用一个 `RV`（寄存器向量）规约，重新缩放用一个逐元素的 `exp2`/`mul` 映射。全程看不到一点 lane 算术。

---

## 流式传输：双缓冲的 KV

这是 [FLOPS 藏在哪里](./where-flops-hide) 里瓶颈 2 的实战。当矩阵核心在处理当前 KV 块时，下一块就该已经在通往共享内存的路上了。内核维持**两个** LDS 缓冲区轮流使用（即「双缓冲」/ 软件流水线）：在缓冲区 A 上计算的同时加载缓冲区 B，然后交换。

```text
   load K/V block 0 --> LDS[A]
   ┌─────────────────────────────────────────────────┐
   │ compute on LDS[A]   ║   load block 1 --> LDS[B] │   <- overlap
   │ compute on LDS[B]   ║   load block 2 --> LDS[A] │
   │ ...                                             │
   └─────────────────────────────────────────────────┘
```

共享 tile 自带它们的 XOR swizzle（瓶颈 3），所以协作填充与逐 lane 读取都不存在 bank 冲突。

---

## 布局的细节：两个 matmul 之间的重新布局

这里就是 [Wave32 与 Wave64](./wave-portability) 不再抽象的地方。内核做两次矩阵乘法，第一次的输出（`S = Q·Kᵀ`，经 softmax 后变成 `P`）是第二次（`P·V`）的*输入*。得分累加器能不能直接回喂、当作操作数？

- **在 CDNA 和 CUDA 上**（`acc_reusable_as_input() == true`）：能。CDNA 上 MFMA 累加器*就是*输入片段，而两半式 `mma.sync` 的 f32 累加器把 m16n8 的 C 片段恰好按 A 操作数的寄存器次序持有，所以那是一次寄存器拷贝，很便宜。
- **在 RDNA 上**（`acc_reusable_as_input() == false`）：不能。偶/奇累加器与复制式操作数并不相同，所以 `P` 必须在第二次乘法之前**经 LDS 往返一趟**（也就是策略中 `att_band` 所分配的那条每 warp softmax 带）重新布局。

内核基于 `ArchCaps` 分支，在每种平台上各做正确的事。同一个算法，两种物理实现，正是上一章所说的那笔可移植性税，落在最重要内核的最热循环里。

---

## 掩码

因果掩码（一个查询不能关注一个未来的 key）和 key 填充掩码（忽略一个 batch 中被填充的位置）都在 softmax 之前作用到得分 tile `S` 上。掩码是从 tile 自身的 lane/行坐标导出的，而非从内存加载：每个得分元素的位置，由「是哪个片段、哪个 lane 持有它」隐含给出，所以掩码是算出来的，不是取出来的。

:::tip[面向 GPU 专家]
计算/内存的重叠在 `tk` 里并不像 HipKittens 的内核那样被手工发射成原始调度内建函数。这里的 KV 循环被标注上 `sched::pipeline(SchedKind::Attention, …)`（`tk/src/kernels/fa.rs`），这是一个标记，由 codegen 中线性化之后的一个调度遍来消费，用以交织矩阵、内存和指数这三股指令流。这让内核体保持可读：它表达的是*要重叠什么*，至于具体的指令排序，则交给后续的遍去决定，而不是让作者亲手把原始调度内建函数一根根穿进算法里。
:::

---

## 为什么这很重要

Flash Attention 是整节内容浓缩进一个文件：

- 它之所以存在，是因为**在线 softmax 是一个递推**，而非一个可分块的规约（[概览](./overview)）；
- 它的成败系于**流式传输与重叠**（[FLOPS 藏在哪里](./where-flops-hide)）；
- 它完全用 **tile 和角色**表达，从不用 lane 索引（[什么是分块](./tiling)）；
- 它编译成与其他一切**相同的 UOp IR**，并作为一个 `Op::Call` 融入惰性图（[向 IR 中编写](./lowering)）；
- 而且它的热循环里携带一个显式的**累加器复用分支**，每种片段布局各一条（[Wave32 与 Wave64](./wave-portability)）。

这就是它为什么要手写，以及 `tk` 为什么为写它而存在。要隔离运行它、检验它的数字，见 [调试](./debugging)。
