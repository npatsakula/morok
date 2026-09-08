---
sidebar_label: Wave32 与 Wave64
---

# 让同一个内核在三种架构上保持正确

这里有个在 NVIDIA 上根本不存在的 bug。你写了一个 tile 内核，在 CDNA 数据中心 GPU 上一测，完美无缺。换到一台 RDNA 笔记本 APU 上跑*同一个*内核，结果却是一堆垃圾数字，没崩溃、没报错，就是错。代码看上去没有任何不同。CUDA 躲过了这个特定的陷阱——warp 处处都是 32 个 lane——却躲不过它背后的成因：片段布局依然不同，所以正是同一层间接把内核也带上了 NVIDIA。

[什么是分块](./tiling) 引入了片段和基于角色的选择；本章解释那层间接为什么非有不可。罪魁祸首是**波前大小**，而能否干净地应对它，正是「只能在一种芯片上工作的 tile 库」与「真正可移植的 tile 库」之间的分水岭。

---

## 32 与 64 之分

波前（NVIDIA 叫「warp」）是一群锁步执行的 lane。AMD 上有两种大小，Svod 两者都瞄准，此外还有 NVIDIA 那唯一的一种：

| 架构 | 示例 | 矩阵操作 | 波前 |
|--------------|---------|-----------|------------|
| **CDNA** | gfx942（数据中心） | MFMA | **wave64**，64 个 lane |
| **RDNA** | gfx1151（RDNA3.5） | WMMA | **wave32**，32 个 lane |
| **CUDA** | sm_80+（Ampere 及更新） | `mma.sync` | **warp32**，32 个 lane |

（这张表是 DSL 会为之解析布局的那组架构；在它之上，每个内核还各自声明自己的 `ArchSet`。Flash Attention、matmul 和单查询注意力三者面向全部三种架构构建；k-means 与 k-NN 内核仅限 AMD，在 CUDA 上返回 `Ok(None)`。）

就这么一个数字，却牵动着一切。一个 `16×16` tile 有 256 个元素：摊到 64 个 lane 上，每 lane 4 个；摊到 32 个 lane 上，每 lane 8 个。不同的 lane 持有不同的元素。于是：

- tile 的**寄存器布局**不同，
- 矩阵指令所要的**操作数布局**不同（RDNA 甚至会把一些操作数*跨 lane 复制*），
- 而任何**跨 lane 规约**（softmax 与 layernorm 的核心）都有着不同的步数和不同的兄弟模式。

一个硬编码了「有 64 个 lane，对 lane 16、32、48 做 xor 来规约」的内核，在 32-lane 机器上算出的只是一个*部分*规约，并悄无声息地返回错误的值。

---

## 对策：索要角色，而非形状

`tk` 的答案是加一层间接。内核从不写下「16×16，每 lane 4 个元素」这样一个具体的片段形状。它索要的是一个**角色**，再交由架构能力去解析：

```text
   kernel says:  "I need an accumulator fragment"   (FragRole::Accumulator)
                          │
                          ▼
   ArchCaps::frag(role)   ── on CDNA ──▶  the wave64 16×16 shape (RT_16X16)
                          ├─ on RDNA ──▶  the wave32 16×16 shape (8 ept, replicated operands)
                          └─ on CUDA ──▶  the two-half mma.sync shape (RT_16X16_MMA)
```

这些角色是 `FragRole::{Accumulator, Operand, AccumulatorT}`，解析器则是 `tk/src/arch.rs` 中的 `ArchCaps::frag(role)`（内核通过 `ker.frag(role)` 触到它）。内核作者只管写「accumulator」和「operand」；至于*物理*布局（每 lane 元素数、interleave 映射、复制），则替目标平台填补进去。它有三条分支：CDNA 把每个角色都解析到那唯一的 wave64 形状，RDNA 解析到偶/奇累加器与复制式操作数，CUDA 解析到两半式的 `mma.sync` 形状。凡是 tk 压根没有对应表的地方（Metal、Ampere 之前的 CUDA）则是 `None`，好让矩阵核心内核大声失败，而不是渲染出一个错误的布局。写一次，三者都能跑。

HipKittens 学到的也是这一课（见 [tk、HipKittens 与 CuTile 对比](./comparison)）：它的 tile 类型以单个编译期 `WARP_THREADS` 常量为键（CDNA 构建里是 `64`），所以换一种 wave 宽度就意味着换一份该库的构建。`tk` 则把这一套折叠成一个运行时解析的 `ArchCaps`。

---

## 一个它真抓住过的 bug

这层间接之所以存在，并非纸上谈兵。早期 `tk` 的一个跨 lane 全规约，即用来把一个值在一个 wave 上求和的 `shuffle_xor` 原语，当初是用硬编码的 wave64 规约树写的。在 RDNA 的 32-lane wave 上，它对那些根本不参与的 lane 做规约，对 attention 所依赖的那种 softmax 式规约算出了错误的和。修复办法是改为基于 `caps.wave_size` 和角色解析出的片段来驱动规约，而不是一个常量。`tk/src/group/shuffle.rs` 里的混洗原语如今会读取 wave 大小；这类 bug 从设计上就被消除了。

:::tip[面向 GPU 专家]
承担 wave 相关大部分分量的是两样东西：

- **片段的 `LaneMap`** 承载着折叠方式。规约是从解析出的片段上读取它的树（`tk/src/group/reduce.rs` 里的 `src.base.map.tree(...)`），而不是从常量：wave64 上是一次跨兄弟 lane 的 gather，偏移为 `[16, 32, 48]`（用 `ds_bpermute` 取 lane `L + d` 上的原始部分和），折叠 4 个子片段；RDNA 的 wave32 上是同一种 gather，只有一个偏移 `[16]`；而 CUDA 的 `MmaSync` 布局上是一个跨掩码 `[1, 2]` 的 xor 蝶形。
- **`acc_reusable_as_input()`** 回答的是：「一个矩阵累加器能否直接回喂、当作下一个乘法的操作数？」CDNA（MFMA 累加器 == 输入片段）以及使用 bf16 `mma.sync` 布局的 CUDA 上是 `true`，布局相符，所以那是一次免费的寄存器拷贝；RDNA 上是 `false`，偶/奇的 `<8×f32>` 累加器与复制式的 `<16×in>` 操作数并不相同，于是这个值得经 LDS 往返一趟重新布局。[Flash Attention](./flash-attention) 在它的两个 matmul 之间处理了这一分歧。

`BaseShape` 上的 `ept` 字段（来自 [什么是分块](./tiling)）也出于同样的理由而存在：RDNA 上操作数被跨 lane 复制，所以每线程元素数并不等于 `element_count / wave_size`，必须显式存储。
:::

---

## 为什么这很重要

跨 wave 大小与片段布局的可移植性，是手写内核身上要付的那笔税，也是为什么把一个 NVIDIA tile 库朴素移植到 AMD 上根本跑不通。`tk` 把这笔税一次性付清，付在 `ArchCaps` 这个抽象里，于是各个内核保持可读：它们只用*角色*说话，把那些 lane 的事交给硬件表去摆平。而同一层抽象随后还能把一个内核带*上* NVIDIA——那里 warp 永远是 32，片段布局却是 `mma.sync` 自己的一套——这正是当初把它建起来所换来的回报。[Flash Attention](./flash-attention) 就是你看到这套做法在一个真实内核里得到回报的地方。
