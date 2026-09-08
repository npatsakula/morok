---
sidebar_label: कर्नेल सर्च
---

# कर्नेल ऑप्टिमाइज़ेशन सर्च

Algebraic simplification के बाद, हर कर्नेल को *scheduling decisions* चाहिए: loops कैसे tile करें, कहाँ parallelize करें, tensor cores इस्तेमाल करें या नहीं। Svod दो strategies offer करता है: fast heuristics और thorough beam search।

यह [codegen pipeline](../codegen/overview.md) के Stage 7 में चलता है।

Tinygrad source: `tinygrad/codegen/opt/`। Svod source: `schedule/src/optimizer/`।

---

## Action Space

ऑप्टिमाइज़ेशन loop structures को axis types बदलकर transform करता है। हर action एक range modify करता है:

| Action | Effect | Hardware Target |
|--------|--------|-----------------|
| UPCAST(axis, amount) | Dimension vectorize करे (SIMD) | सभी |
| UNROLL(axis, amount) | Loop dimension unroll करे | सभी |
| LOCAL(axis, amount) | GPU shared memory इस्तेमाल करे | GPU (LDS) / CPU (L1) |
| GROUP(axis, amount) | Two-stage reduction | सभी |
| GROUPTOP(axis, amount) | Tensor cores के लिए grouped reduction | GPU |
| THREAD(axis, amount) | CPU thread-based parallelism | CPU |
| SWAP(axis1, axis2) | Global dimensions reorder करे | सभी |
| PADTO(axis, amount) | Alignment के लिए pad करे | सभी |
| NOLOCALS | Local memory disable करे | सभी (constraint) |
| TC | Tensor core usage enable करे | NVIDIA, AMD, Metal, Intel GPUs (WMMA/MFMA) |

`BEAM_ACTIONS` में 193 base actions हैं (`BEAM_PADTO=1` के साथ 200); हर kernel में इनमें से कितने बचते हैं, यह kernel structure और available parallelism पर निर्भर करता है। NOLOCALS इस list में नहीं है — `generate_actions` उसे तभी जोड़ता है जब `NOLOCALS`/`SVOD_NOLOCALS` सेट हो।

---

## Heuristics (डिफ़ॉल्ट)

Heuristic optimizer एक fixed order में optimizations apply करता है (simplified pseudocode):

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

**फ़ायदे**: तेज़ (~50ms प्रति कर्नेल), predictable, hardware measurement नहीं चाहिए।

**नुकसान**: ऑप्टिमाइज़ेशन के मौके छूट सकते हैं, fixed heuristics workload के हिसाब से adapt नहीं करते।

---

## Beam Search (ऑप्शनल)

प्रोडक्शन workloads के लिए, beam search बेहतर schedules ढूँढता है candidates compile और time करके (simplified pseudocode):

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

**फ़ायदे**: Near-optimal schedules ढूँढता है, hardware के हिसाब से adapt करता है।

**नुकसान**: प्रति कर्नेल मिनटों में (लेकिन results AST hash से cache होते हैं)।

---

## कॉन्फ़िगरेशन

```bash
# Disable optimization (debugging)
SVOD_NOOPT=1 cargo run

# Enable beam search with width 8
BEAM=8 cargo run
```

या प्रोग्रामैटिकली:

```rust
let config = PrepareConfig::from(
    OptimizerConfig::builder()
        .strategy(OptStrategy::Beam { width: 8 })
        .build()
);

tensor.realize_with(&config)?;
```

---

## तुलना: दूसरे कम्पाइलर कैसे ऑप्टिमाइज़ करते हैं

| पहलू | XLA | TVM/Ansor | Triton | **Svod** |
|-------|-----|-----------|--------|-----------|
| **फ़िलॉसफ़ी** | फ़िक्स्ड heuristics | सर्च-आधारित | प्रोग्रामर-गाइडेड | पैटर्न-आधारित |
| **Fusion** | कंज़र्वेटिव नियम | Tile-and-fuse | Block-level | ग्राफ़ रीराइटिंग |
| **Auto-tuning** | कोई नहीं | Evolutionary + cost model | Grid search | Beam search |
| **ट्यूनिंग कॉस्ट** | 0 | घंटों | मिनटों | मिनटों (कैश्ड) |
| **फ़्लेक्सिबिलिटी** | कम | ज़्यादा | मध्यम | ज़्यादा |
| **ट्रांसपैरेंसी** | कम (C++ पासेज़) | मध्यम (Python) | मध्यम (DSL) | ज़्यादा (declarative patterns) |

**XLA** fusion decisions के लिए fixed heuristics इस्तेमाल करता है। Safe और predictable, लेकिन performance table पर छूट जाती है। Fusion rules C++ में hard-coded हैं।

**TVM/Ansor** *क्या* compute करना है और *कैसे* compute करना है को अलग करता है। Ansor learned cost model के साथ evolutionary search इस्तेमाल करता है। Best-in-class performance, लेकिन tuning में प्रति model घंटे लगते हैं।

**Triton** blocked algorithms के लिए Python-जैसा DSL expose करता है। Control और automation का अच्छा balance, लेकिन GPU programming expertise चाहिए।

**Svod** optimizations को composable patterns में express करता है। Beam search ज़रूरत पड़ने पर auto-tuning जोड़ता है, results reuse के लिए AST hash से cache होते हैं।
