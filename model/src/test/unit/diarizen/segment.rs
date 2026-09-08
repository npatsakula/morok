//! Unit tests for the DiariZen chunk geometry (pure math) + the chunked
//! `DiariZenSegmenter` driver (tiny config, no weights).

use crate::diarizen::{DiariZenSegmentationModel, DiariZenSegmenter, chunk_plan, hop_samples, window_samples};
use test_case::test_case;

#[test_case(16.0, 16_000, 256_000 ; "16s_at_16khz")]
fn window_samples_cases(seconds: f32, sample_rate: u32, expected: usize) {
    assert_eq!(window_samples(seconds, sample_rate), expected);
}

#[test_case(0.1, 256_000, 25_600 ; "step_0p1_of_256k")]
fn hop_samples_cases(step: f32, window: usize, expected: usize) {
    assert_eq!(hop_samples(step, window), expected);
}

// window = 256_000, hop = 25_600 (90 % overlap). Matches `torch.unfold(1, w, h)`
// plus a single zero-padded trailing chunk for the ragged tail.
#[test_case(0, 1 ; "empty_formula")]
#[test_case(1, 1 ; "one_sample")]
#[test_case(100_000, 1 ; "below_window")]
#[test_case(256_000, 1 ; "exact_window")]
#[test_case(256_001, 2 ; "window_plus_one")]
#[test_case(281_600, 2 ; "window_plus_one_hop")]
#[test_case(332_800, 4 ; "window_plus_three_hops")]
#[test_case(333_000, 5 ; "ragged_tail")]
fn chunk_plan_cases(n: usize, expected: usize) {
    assert_eq!(chunk_plan(n, 256_000, 25_600), expected);
}

/// Drives `DiariZenSegmenter` over a waveform spanning 3 chunks with
/// `inference_batch_size = 2`, forcing the reuse loop through 2 batches
/// (b=2 then b=1). Validates chunk count, output shape, plan-reuse
/// determinism, and the sample-rate guard — no weights/parity, just plumbing.
///
/// Ignored by default: preparing the segmenter compiles the full WavLM +
/// Conformer JIT (~100 s in a debug build). Run on demand with
/// `cargo test -p svod-model -- --ignored segmenter`, or rely on the
/// `diarizen_segment` example for end-to-end coverage.
#[test]
#[ignore = "slow: compiles the full segmentation JIT (~100s debug)"]
fn segmenter_runs_multiple_batches() {
    // Tiny WavLM/Conformer + a small window so the JIT compile is cheap.
    let mut cfg = super::model::tiny_cfg();
    cfg.chunk_size_seconds = 0.256; // window = floor(0.256 * 16000) = 4096
    cfg.inference_batch_size = 2;

    let window = cfg.window_samples();
    let hop = cfg.hop_samples();
    assert_eq!(window, 4096);

    let model = DiariZenSegmentationModel::empty(cfg.clone());
    let mut seg = DiariZenSegmenter::new(model).expect("prepare segmenter");

    // 3 chunks → 2 batches at inference_batch_size = 2.
    let n = window + 2 * hop;
    assert_eq!(chunk_plan(n, window, hop), 3);
    let wav: Vec<f32> = (0..n).map(|i| ((i % 23) as f32) * 0.013 - 0.1).collect();

    let mut out = seg.segment(&wav, cfg.sample_rate).expect("segment");
    assert_eq!(out.num_chunks, 3);
    assert!(out.frames_per_chunk > 0);

    let k = cfg.powerset_class_count();
    let shape = out.logits.dims().unwrap();
    assert_eq!(shape, vec![3, out.frames_per_chunk, k]);

    // Reusing the same prepared plan is deterministic (stateless reuse).
    out.logits.realize().expect("realize logits");
    let first = out.logits.as_vec::<f32>().expect("read logits");
    let mut again = seg.segment(&wav, cfg.sample_rate).expect("segment again");
    again.logits.realize().expect("realize again");
    assert_eq!(first, again.logits.as_vec::<f32>().expect("read again"));

    // Sample-rate guard.
    assert!(seg.segment(&wav, 8_000).is_err(), "wrong sample rate must error");
}

/// Pins the threading contract: a downstream pipeline must be able to move the
/// segmenter across threads (gigaam leaves this implicit).
#[test]
fn segmenter_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<DiariZenSegmenter>();
}
