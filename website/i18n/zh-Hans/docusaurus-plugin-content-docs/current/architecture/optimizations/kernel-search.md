---
sidebar_label: 内核搜索
---

# 内核优化搜索

代数化简之后，每个内核需要*调度决策*：如何分块循环、在哪里并行化、是否使用 tensor core。Svod 提供两种策略：快速启发式和彻底的 beam 搜索。

这在[代码生成流水线](../codegen/overview.md)的阶段 7 运行。

Tinygrad 源码：`tinygrad/codegen/opt/`。Svod 源码：`schedule/src/optimizer/`。

---

## 动作空间

优化通过改变轴类型来变换循环结构。每个动作修改一个范围：

| 动作 | 效果 | 硬件目标 |
|--------|--------|-----------------|
| UPCAST(axis, amount) | 向量化一个维度（SIMD） | 全部 |
| UNROLL(axis, amount) | 展开一个循环维度 | 全部 |
| LOCAL(axis, amount) | 使用 GPU 共享内存 | GPU (LDS) / CPU (L1) |
| GROUP(axis, amount) | 两阶段规约 | 全部 |
| GROUPTOP(axis, amount) | tensor core 的分组规约 | GPU |
| THREAD(axis, amount) | CPU 线程并行 | CPU |
| SWAP(axis1, axis2) | 重排全局维度 | 全部 |
| PADTO(axis, amount) | 对齐填充 | 全部 |
| NOLOCALS | 禁用本地内存 | 全部（约束） |
| TC | 启用 tensor core | NVIDIA、AMD、Metal、Intel GPU（WMMA/MFMA） |

`BEAM_ACTIONS` 中有 193 个基础动作（`BEAM_PADTO=1` 时为 200 个）；每个内核最终保留多少，取决于内核结构和可用并行度。NOLOCALS 不在这个列表里——只有设置了 `NOLOCALS`/`SVOD_NOLOCALS` 时，`generate_actions` 才会追加它。

---

## 启发式（默认）

启发式优化器按固定顺序应用优化（简化伪代码）：

```rust
// Pseudocode — simplified from optimizer/heuristics.rs
fn hand_coded_optimizations(scheduler: &mut Scheduler) {
    // 1. Tensor cores (if matmul pattern detected)
    if let Some(tc) = detect_tensor_core_pattern(scheduler) {
        apply_tensor_core(scheduler, tc);
        return;  // TC handles everything
    }

    // 2. Grouped reductions (two-stage for large reductions)
    apply_grouped_reduction_if_needed(scheduler);

    // 3. Vectorization (UPCAST output dimensions)
    apply_upcast(scheduler, 4);

    // 4. GPU local memory (workgroup dimensions)
    apply_local_dims(scheduler);

    // 5. CPU threading
    apply_threading(scheduler);
}
```

**优点**：快（每个内核约 50ms）、可预测、无需硬件测量。

**缺点**：可能错过优化机会，固定启发式不能适应不同工作负载。

---

## Beam 搜索（可选）

对于生产工作负载，beam 搜索通过编译和计时候选方案找到更好的调度（简化伪代码）：

```rust
// Pseudocode — simplified from optimizer/beam.rs
// Actual API: beam_search_cached_remote(scheduler, config, compiler_identity,
//                                       behavior_fingerprint, compile_wave, benchmark)
fn beam_search(scheduler: Scheduler, config: &BeamConfig) -> Scheduler {
    let mut beam = vec![(scheduler, Duration::MAX)];

    loop {
        // EXPAND: every applicable action on every beam member
        let candidates: Vec<Scheduler> = beam.iter()
            .flat_map(|(state, _)| generate_actions(state))
            .collect();

        // COMPILE in helper worker processes, then time config.num_runs each
        let mut timed = vec![];
        for (candidate, compiled) in compile_wave(&candidates) {
            if !seen_binary.insert(compiled.binary_key) { continue; }  // identical code
            if bloated(&mut least_compute_ops, compiled.compute_ops) { continue; }
            timed.push((candidate, benchmark(&compiled)));
        }

        // Keep top K by execution time
        timed.sort_by_key(|(_, time)| *time);
        timed.truncate(config.beam_width);

        // Stop when the best candidate no longer improves by min_progress_ns
        if best(&beam) - best(&timed) < config.min_progress_ns { break; }
        beam = timed;
    }

    beam.into_iter().next().unwrap().0
}
```

**优点**：找到接近最优的调度方案，能适应硬件。

**缺点**：每个内核需要几分钟（但结果按 AST 哈希缓存）。

---

## 配置

```bash
# Disable optimization (debugging)
SVOD_NOOPT=1 cargo run

# Enable beam search with width 8
BEAM=8 cargo run
```

或通过代码配置：

```rust
let config = PrepareConfig::from(
    OptimizerConfig::builder()
        .strategy(OptStrategy::Beam { width: 8 })
        .build()
);

tensor.realize_with(&config)?;
```

---

## 对比：其他编译器如何优化

| 方面 | XLA | TVM/Ansor | Triton | **Svod** |
|--------|-----|-----------|--------|-----------|
| **理念** | 固定启发式 | 基于搜索 | 程序员引导 | 基于模式 |
| **融合** | 保守规则 | Tile-and-fuse | 块级别 | 图重写 |
| **自动调优** | 无 | 进化算法 + 代价模型 | 网格搜索 | Beam 搜索 |
| **调优成本** | 0 | 数小时 | 数分钟 | 数分钟（有缓存） |
| **灵活性** | 低 | 高 | 中 | 高 |
| **透明度** | 低（C++ pass） | 中（Python） | 中（DSL） | 高（声明式模式） |

**XLA** 使用固定启发式做融合决策。安全可预测，但会损失性能。融合规则硬编码在 C++ 中。

**TVM/Ansor** 将*计算什么*和*如何计算*分离。Ansor 使用进化搜索配合学习的代价模型。可以达到业界最佳性能，但每个模型调优需要数小时。

**Triton** 提供一个类 Python 的 DSL 来编写分块算法。在控制和自动化之间取得了良好平衡，但需要 GPU 编程专业知识。

**Svod** 将优化表达为可组合的模式。Beam 搜索在需要时提供自动调优，结果按 AST 哈希缓存以供复用。
