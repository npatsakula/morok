---
sidebar_label: Kernel Search
---

# Kernel Optimization Search

After algebraic simplification, each kernel needs *scheduling decisions*: how to tile loops, where to parallelize, whether to use tensor cores. Svod offers two strategies: fast heuristics and thorough beam search.

This runs at Stage 7 of the [codegen pipeline](../codegen/overview.md).

Tinygrad source: `tinygrad/codegen/opt/`. Svod source: `schedule/src/optimizer/`.

---

## The Action Space

Optimization transforms loop structures by changing axis types. Each action modifies one range:

| Action | Effect | Hardware Target |
|--------|--------|-----------------|
| UPCAST(axis, amount) | Vectorize a dimension (SIMD) | All |
| UNROLL(axis, amount) | Unroll a loop dimension | All |
| LOCAL(axis, amount) | Use GPU shared memory | GPU (LDS) / CPU (L1) |
| GROUP(axis, amount) | Two-stage reduction | All |
| GROUPTOP(axis, amount) | Grouped reduction for tensor cores | GPU |
| THREAD(axis, amount) | CPU thread-based parallelism | CPU |
| SWAP(axis1, axis2) | Reorder global dimensions | All |
| PADTO(axis, amount) | Pad for alignment | All |
| NOLOCALS | Disable local memory | All (constraint) |
| TC | Enable tensor core usage | NVIDIA, AMD, Metal, Intel GPUs (WMMA/MFMA) |

`BEAM_ACTIONS` holds 193 base actions (200 with `BEAM_PADTO=1`); how many survive per kernel varies with kernel structure and available parallelism. NOLOCALS is not in that list — `generate_actions` appends it only when `NOLOCALS`/`SVOD_NOLOCALS` is set.

---

## Heuristics (Default)

The heuristic optimizer applies optimizations in a fixed order (simplified pseudocode):

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

**Pros**: Fast (~50ms per kernel), predictable, no hardware measurement needed.

**Cons**: May miss optimization opportunities, fixed heuristics don't adapt to workload.

---

## Beam Search (Optional)

For production workloads, beam search finds better schedules by compiling and timing candidates (simplified pseudocode):

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

**Pros**: Finds near-optimal schedules, adapts to hardware.

**Cons**: Minutes per kernel (but results are cached by AST hash).

---

## Configuration

```bash
# Disable optimization (debugging)
SVOD_NOOPT=1 cargo run

# Enable beam search with width 8
BEAM=8 cargo run
```

Or programmatically:

```rust
let config = PrepareConfig::from(
    OptimizerConfig::builder()
        .strategy(OptStrategy::Beam { width: 8 })
        .build()
);

tensor.realize_with(&config)?;
```

---

## Comparison: How Other Compilers Optimize

| Aspect | XLA | TVM/Ansor | Triton | **Svod** |
|--------|-----|-----------|--------|-----------|
| **Philosophy** | Fixed heuristics | Search-based | Programmer-guided | Pattern-based |
| **Fusion** | Conservative rules | Tile-and-fuse | Block-level | Graph rewriting |
| **Auto-tuning** | None | Evolutionary + cost model | Grid search | Beam search |
| **Tuning cost** | 0 | Hours | Minutes | Minutes (cached) |
| **Flexibility** | Low | High | Medium | High |
| **Transparency** | Low (C++ passes) | Medium (Python) | Medium (DSL) | High (declarative patterns) |

**XLA** uses fixed heuristics for fusion decisions. Safe and predictable, but leaves performance on the table. Fusion rules are hard-coded in C++.

**TVM/Ansor** separates *what* to compute from *how* to compute it. Ansor uses evolutionary search with a learned cost model. Best-in-class performance, but tuning takes hours per model.

**Triton** exposes a Python-like DSL for blocked algorithms. Good balance of control and automation, but requires GPU programming expertise.

**Svod** expresses optimizations as composable patterns. Beam search adds auto-tuning when needed, with results cached by AST hash for reuse.
