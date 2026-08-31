//! Host-only tests for [`ModernBertEmbedder`] over a tiny random-weight backbone
//! (f32, CPU). No real checkpoint, no HF Hub — the model is `ModernBert::empty`
//! with `fan_in_uniform` random token embeddings, enough to exercise the JIT
//! prepare/pack/execute/read path and the fused mask+pool+norm semantics.

use std::sync::{LazyLock, Mutex, MutexGuard};

use svod_arch::pipelines::text::{Embed, Embedding, EncoderHead, Encoding};
use svod_device::{Buffer, BufferSpec, cpu};
use svod_dtype::DType;

use crate::modernbert::packing::{pack_ids_buffer, pack_mask_buffer};
use crate::modernbert::{ModernBert, ModernBertEmbedder};
use crate::test::unit::modernbert::model::tiny_cfg;

/// Canonical prepared sizes for the shared embedder fixture. Compiling the
/// 2-layer CPU JIT graph takes ~60s, so every test in this module runs against
/// a single plan prepared once at these sizes; metadata-only assertions that
/// used other sizes are normalized onto them.
const MAX_BATCH: usize = 4;
const MAX_SEQ: usize = 16;

/// One JIT-compiled embedder shared by every test in the module. Prepared once
/// per process (`LazyLock` init), then borrowed under a `Mutex`: the only shared
/// mutable state is the plan's input/output buffers, which each `embed_batch`
/// fully overwrites before reading back the live `b` rows — so sharing is
/// contamination-free. Every assertion here is a weight-agnostic invariant
/// (shape, finiteness, single-vs-batch consistency, mask invariance), so a
/// single random instance serves them all.
static EMBEDDER: LazyLock<Mutex<ModernBertEmbedder>> = LazyLock::new(|| {
    let model = ModernBert::empty(tiny_cfg());
    Mutex::new(ModernBertEmbedder::new(model, MAX_BATCH, MAX_SEQ).expect("prepare embedder JIT"))
});

/// Borrow the shared prepared embedder. Recovers from poison so a panicking
/// test doesn't cascade failures into its siblings — the buffers are rewritten
/// each call, so the post-poison state is still safe to reuse.
fn embedder() -> MutexGuard<'static, ModernBertEmbedder> {
    EMBEDDER.lock().unwrap_or_else(|p| p.into_inner())
}

/// A consistent encoding: `n` real token ids (1..) with mask all-ones, plus
/// optional trailing pad positions at mask 0.
fn encoding(real_ids: &[u32], n_pad: usize) -> Encoding {
    let mut ids = real_ids.to_vec();
    let mut mask = vec![1u32; real_ids.len()];
    ids.extend(std::iter::repeat_n(&0u32, n_pad).copied());
    mask.extend(std::iter::repeat_n(&0u32, n_pad));
    let l = ids.len();
    Encoding {
        input_ids: ids,
        attention_mask: mask,
        token_type_ids: vec![0; l],
        offsets: (0..l).map(|i| (i, i + 1)).collect(),
        special_tokens_mask: vec![0; l],
    }
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Max elementwise absolute difference between two equal-length slices — the
/// shared comparison primitive for the batch-vs-single and mask-leak checks.
fn max_delta(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

/// `embed_batch` returns one embedding per input, each of length `hidden_size`,
/// and each L2-normalized (norm ≈ 1).
#[test]
#[ignore = "heavy: 2-layer ModernBERT JIT graph compile through the CPU backend"]
fn embed_batch_shapes_and_norms() {
    let mut emb = embedder();
    let hidden = emb.hidden_size();
    let e1 = encoding(&[1, 2, 3], 0);
    let e2 = encoding(&[4, 5], 1);
    let (out, prof) = emb.run_batch(&[&e1, &e2], false).expect("embed_batch");
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].values.len(), hidden);
    assert_eq!(out[1].values.len(), hidden);
    assert!(prof.is_none(), "unprofiled run yields no profile");
    for Embedding { values } in &out {
        let n = l2_norm(values);
        assert!((n - 1.0).abs() < 1e-3, "L2-normalized embedding should have norm ~1, got {n}");
        assert!(values.iter().all(|v| v.is_finite()), "non-finite embedding value");
    }
}

/// The trait-default `embed` (batch-of-one) agrees with `embed_batch` on a
/// single input — the default delegates to the batch path and pops.
#[test]
#[ignore = "heavy: 2-layer ModernBERT JIT graph compile through the CPU backend"]
fn embed_single_matches_batch_of_one() {
    let mut emb = embedder();
    let e = encoding(&[1, 2, 3, 4], 0);
    let single = emb.run(&e, false).expect("embed").0;
    let batch = emb.run_batch(&[&e], false).expect("embed_batch");
    let batch0 = batch.0.into_iter().next().unwrap();
    let max = single.values.iter().zip(&batch0.values).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert_eq!(max, 0.0, "default embed must match embed_batch exactly");
}

/// The padding mask is load-bearing: pooling must ignore pad positions. A
/// sequence `[1,2,3]` (no pad) and `[1,2,3]` + 2 pad tokens (mask 0) must yield
/// the **same** embedding — proving masked mean-pool, not raw mean-pool.
#[test]
#[ignore = "heavy: 2-layer ModernBERT JIT graph compile through the CPU backend"]
fn pooling_ignores_pad_positions() {
    let mut emb = embedder();
    let unpadded = encoding(&[1, 2, 3], 0);
    let padded = encoding(&[1, 2, 3], 2); // two trailing pad positions, mask 0
    let a = emb.run(&unpadded, false).expect("unpadded").0;
    let b = emb.run(&padded, false).expect("padded").0;
    let max = a.values.iter().zip(&b.values).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    assert!(max < 1e-4, "pad positions leaked into the pooled embedding: max |delta| = {max}");
}

/// `hidden_size`/`capacity` report the prepared sizes.
#[test]
#[ignore = "heavy: 2-layer ModernBERT JIT graph compile through the CPU backend"]
fn capacity_reports_prepared_sizes() {
    let emb = embedder();
    let (mb, ms) = emb.capacity();
    assert_eq!(mb, MAX_BATCH);
    assert_eq!(ms, MAX_SEQ);
    assert_eq!(emb.hidden_size(), 32, "matches tiny_cfg hidden_size");
}

/// A batch exceeding `max_batch` is rejected up front (CapacityExceeded), not
/// silently truncated or overflowed.
#[test]
#[ignore = "heavy: 2-layer ModernBERT JIT graph compile through the CPU backend"]
fn batch_over_max_batch_is_rejected() {
    let mut emb = embedder();
    // One more than the prepared MAX_BATCH.
    let encs = (1..=MAX_BATCH + 1).map(|i| encoding(&[i as u32], 0)).collect::<Vec<_>>();
    let refs: Vec<&Encoding> = encs.iter().collect();
    let err = emb.run_batch(&refs, false).unwrap_err();
    assert!(matches!(err, crate::modernbert::HeadError::CapacityExceeded { .. }));
}

/// An encoding longer than the prepared `max_seq` is rejected
/// (SequenceTooLong), not silently truncated — the guard is reachable via the
/// public chunk seam, which lets a caller feed pre-built chunks that bypass the
/// chunker's length bound.
#[test]
#[ignore = "heavy: 2-layer ModernBERT JIT graph compile through the CPU backend"]
fn sequence_over_max_seq_is_rejected() {
    let mut emb = embedder();
    let ids: Vec<u32> = (1..=MAX_SEQ as u32 + 1).collect();
    let e = encoding(&ids, 0);
    let err = emb.run_batch(&[&e], false).unwrap_err();
    assert!(matches!(err, crate::modernbert::HeadError::SequenceTooLong { .. }));
}

/// An empty batch is a no-op returning no embeddings (mirrors the pipeline's
/// zero-chunk guard).
#[test]
#[ignore = "heavy: 2-layer ModernBERT JIT graph compile through the CPU backend"]
fn empty_batch_is_noop() {
    let mut emb = embedder();
    let (out, prof) = emb.run_batch(&[], false).expect("empty batch");
    assert!(out.is_empty());
    assert!(prof.is_none());
}

/// A profiled run yields a profile with an `embed` GPU stage; an unprofiled run
/// on the same embedder yields none (per-call, no rebuild).
#[test]
#[ignore = "heavy: 2-layer ModernBERT JIT graph compile through the CPU backend"]
fn profiled_run_emits_embed_stage() {
    let mut emb = embedder();
    let e = encoding(&[1, 2], 0);
    let prof = emb.run(&e, true).expect("profiled").1.expect("profile present");
    assert!(prof.stages.iter().any(|s| s.name == "embed"), "embed stage present");
    assert!(emb.run(&e, false).expect("unprofiled").1.is_none(), "unprofiled yields no profile");
}

/// Two **distinct** inputs through `embed_batch` must match the same inputs run
/// one-at-a-time via `embed` — a cross-row leakage guard. Row `i`'s output must
/// depend only on row `i`'s ids/mask, never on a sibling row packed into the
/// `[max_batch, max_seq]` buffers. Exact equality: same weights, same row 0
/// computation whether `b` binds to 1 or 2 (the symbolic-batch graph computes
/// only the first `b` rows), so the bits must match.
#[test]
#[ignore = "heavy: 2-layer ModernBERT JIT graph compile through the CPU backend"]
fn embed_batch_rows_match_single_calls() {
    let mut emb = embedder();
    let e1 = encoding(&[1, 2, 3], 0);
    let e2 = encoding(&[10, 20, 30, 40], 0);
    let batch = emb.run_batch(&[&e1, &e2], false).expect("embed_batch").0;
    let s1 = emb.run(&e1, false).expect("embed e1").0;
    let s2 = emb.run(&e2, false).expect("embed e2").0;
    assert_eq!(max_delta(&batch[0].values, &s1.values), 0.0, "batch row 0 leaked from/into e1");
    assert_eq!(max_delta(&batch[1].values, &s2.values), 0.0, "batch row 1 leaked from/into e2");
}

/// On the **batch** path the attention mask is threaded per row: a row whose
/// real tokens are `[1,2,3]` with two trailing pads must pool to the same vector
/// as the same row run unpadded, and must agree with its standalone `embed`. The
/// batch packing must not collapse pooling to a raw mean over the padded length.
/// Tolerance 1e-4 mirrors `pooling_ignores_pad_positions`.
#[test]
#[ignore = "heavy: 2-layer ModernBERT JIT graph compile through the CPU backend"]
fn batch_path_respects_attention_mask() {
    let mut emb = embedder();
    let padded = encoding(&[1, 2, 3], 2);
    let unpadded = encoding(&[1, 2, 3], 0);
    let batch = emb.run_batch(&[&padded, &unpadded], false).expect("embed_batch").0;
    // Pad positions must not move the pooled output within a row.
    let d = max_delta(&batch[0].values, &batch[1].values);
    assert!(d < 1e-4, "pad mask leaked in the batch path: max |delta| = {d}");
    // And the batch path agrees with the single-call path for the unpadded row.
    let alone = emb.run(&unpadded, false).expect("embed unpadded").0;
    let d = max_delta(&batch[1].values, &alone.values);
    assert!(d < 1e-4, "batch row 1 disagrees with its standalone embed: max |delta| = {d}");
}

/// On the throughput path (`embed_batch`, not the batch-of-one `embed` default),
/// a profiled run emits a GPU stage named `embed`; an unprofiled run on the same
/// embedder emits none — the profile switch is per-call, no rebuild.
#[test]
#[ignore = "heavy: 2-layer ModernBERT JIT graph compile through the CPU backend"]
fn embed_batch_profile_emits_stage() {
    let mut emb = embedder();
    let e1 = encoding(&[1, 2], 0);
    let e2 = encoding(&[3, 4], 0);
    let prof = emb.run_batch(&[&e1, &e2], true).expect("profiled embed_batch").1.expect("profile present");
    assert!(prof.stages.iter().any(|s| s.name == "embed"), "embed GPU stage present");
    assert!(
        emb.run_batch(&[&e1, &e2], false).expect("unprofiled embed_batch").1.is_none(),
        "unprofiled yields no profile"
    );
}

// ── buffer packing (fast, no JIT) ───────────────────────────────────────────
//
// `pack_ids_buffer` / `pack_mask_buffer` are the shared host-side packing logic
// for the embedder, classifier, and token-classifier batch paths — the
// load-bearing mask/pad correctness. Exercising them directly keeps that
// coverage in the fast default suite (the JIT end-to-end version lives in the
// `#[ignore = "heavy"]` tier above).

/// Packing honors the per-row `max_seq` stride, zero-fills pad positions and
/// unused rows, and truncates over-length inputs. The mask mirrors the ids'
/// real-token counts. This is the property the JIT mask-invariance tests rely on.
#[test]
fn pack_buffers_pad_stride_and_truncate_correctly() {
    let max_batch = 3;
    let max_seq = 4;
    let n = max_batch * max_seq;
    let alloc = cpu().expect("cpu allocator");
    let mut ids = Buffer::new(alloc.clone(), DType::Int64, vec![n], BufferSpec::default());
    let mut mask = Buffer::new(alloc, DType::Int64, vec![n], BufferSpec::default());

    // e1: 2 real tokens + 1 trailing pad (mask 0). e2: 5 real tokens — one over
    // max_seq, so the last must be truncated. Row 2 left unused.
    let e1 = encoding(&[1, 2], 1);
    let e2 = encoding(&[10, 20, 30, 40, 50], 0);
    pack_ids_buffer(&mut ids, &[&e1, &e2], max_seq).expect("pack ids");
    pack_mask_buffer(&mut mask, &[&e1, &e2], max_seq).expect("pack mask");

    let ids_v = ids.as_array::<i64>().unwrap().as_slice().unwrap().to_vec();
    let mask_v = mask.as_array::<i64>().unwrap().as_slice().unwrap().to_vec();

    // Row 0: [1,2] + pad(0) + fill(0). Row 1: first 4 of 5 (truncated). Row 2: unused.
    assert_eq!(ids_v, vec![1, 2, 0, 0, 10, 20, 30, 40, 0, 0, 0, 0]);
    // Mask mirrors real-token positions only; pad + fill + unused stay 0.
    assert_eq!(mask_v, vec![1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]);
}
