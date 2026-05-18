//! Silero V5 VAD tests.
//!
//! Cheap default tier walks the forward graph on zero inputs and asserts the
//! symbolic output shape — milliseconds, no compile, mirrors mexus's WeSpeaker
//! pattern in `test/unit/wespeaker/tstp.rs`. The `#[ignore]`-gated tier below
//! exercises the full `VadInference` JIT path (prepare → step → output reads)
//! — useful when actively touching VAD wiring but too slow for default CI.

use svod_arch::vad::AudioChunk;
use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::silero_vad::{CONTEXT_SIZE, HIDDEN, NUM_SAMPLES, SileroVad, VadInference, fixed_windows_from_probs};

// ---------------------------------------------------------------------------
// Cheap default tests (no realize): build forward graph, check symbolic shape.
// ---------------------------------------------------------------------------

/// `forward_chunk` returns `[1, 1 + 2*HIDDEN]` — concatenated `[prob, new_h,
/// new_c]`. Catches axis bugs in STFT-conv → 4 convs → LSTM step → final
/// conv without any kernel compile.
#[test]
fn forward_chunk_zero_weights_shape() {
    let vad = SileroVad::with_random_weights();

    let chunk = Tensor::zeros(&[1, CONTEXT_SIZE + NUM_SAMPLES], DType::Float32).unwrap();
    let state_h = Tensor::zeros(&[1, HIDDEN], DType::Float32).unwrap();
    let state_c = Tensor::zeros(&[1, HIDDEN], DType::Float32).unwrap();

    let out = vad.forward_chunk(&chunk, &state_h, &state_c).unwrap();
    let shape: Vec<usize> = out
        .shape()
        .unwrap()
        .iter()
        .map(|s| s.as_const().or_else(|| s.vmax()).expect("concrete or symbolic-max shape"))
        .collect();
    // [B=1, prob(1) + h(HIDDEN) + c(HIDDEN)]
    assert_eq!(shape, vec![1, 1 + 2 * HIDDEN]);
}

#[test]
fn fixed_windows_from_probs_keeps_only_speech_windows() {
    let mut probs = vec![0.0_f32; 55];
    probs[2] = 0.8;
    probs[42] = 0.9;

    let chunks = fixed_windows_from_probs(&probs, 55, 1, 0.5, 1, 20);
    assert_eq!(chunks, vec![AudioChunk::new(0, 20), AudioChunk::new(40, 55)]);
}

#[test]
fn fixed_windows_from_probs_respects_min_speech_probs() {
    let mut probs = vec![0.0_f32; 40];
    probs[2] = 0.8;
    probs[21] = 0.9;
    probs[22] = 0.95;

    let chunks = fixed_windows_from_probs(&probs, 40, 1, 0.5, 2, 20);
    assert_eq!(chunks, vec![AudioChunk::new(20, 40)]);
}

// ---------------------------------------------------------------------------
// Heavy tests (compile + execute): unique signal, gated behind --ignored.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "heavy: full SileroVad JIT compile + execute on random weights"]
fn vad_inference_runs_on_random_weights() {
    let vad = SileroVad::with_random_weights();
    let mut inf = VadInference::new(vad).expect("prepare");
    let waveform = vec![0.0_f32; 4 * NUM_SAMPLES];
    let probs = inf.probs(&waveform).expect("probs");
    assert_eq!(probs.len(), 4, "one prob per {NUM_SAMPLES}-sample chunk");
    for p in &probs {
        assert!(p.is_finite(), "non-finite VAD prob: {p}");
        assert!((0.0..=1.0).contains(p), "VAD prob outside [0, 1]: {p}");
    }
}

#[test]
#[ignore = "heavy: full VAD pipeline + chunker on random weights"]
fn vad_segment_random_weights_ranges_are_well_formed() {
    let vad = SileroVad::with_random_weights();
    let mut inf = VadInference::new(vad).expect("prepare");
    let waveform = vec![0.0_f32; 16_000];
    let ranges = inf.segment(&waveform, 0.5);
    for (start, end) in &ranges {
        assert!(start <= end, "inverted range: ({start}, {end})");
        assert!(*end <= waveform.len(), "range past waveform end: ({start}, {end})");
    }
}
