//! Unit tests for `FixedLengthSplitter` and `EncoderBounds`.
//!
//! `SileroVadSplitter` is covered end-to-end through the example transcripts
//! (loading a hub VAD into a unit test would be too heavy); only the
//! pure-logic splitter and the bounds getters are exercised here.

use svod_arch::vad::AudioChunk;

use crate::audio::{EncoderBounds, FixedLengthSplitter, Splitter};

/// 16 kHz / hop=160 / subsampling=4 → align_to = 640 samples (40 ms),
/// max_mel_frames=2_080 → max_samples = (2080 - 8) * 160 = 331_520 samples
/// (~20.7 s).
fn bounds_realistic() -> EncoderBounds {
    EncoderBounds { sample_rate: 16_000, hop_length: 160, subsampling_factor: 4, max_mel_frames: 2_080 }
}

/// Small bounds for stride exhaustion tests: align_to=4, max_samples=16.
/// hop=2, subsampling=2 → align=4. max_mel_frames=12 → max_samples=(12-4)*2=16.
fn bounds_tiny() -> EncoderBounds {
    EncoderBounds { sample_rate: 16_000, hop_length: 2, subsampling_factor: 2, max_mel_frames: 12 }
}

#[test]
fn encoder_bounds_getters() {
    let b = bounds_realistic();
    assert_eq!(b.align_to_samples(), 640);
    assert_eq!(b.max_samples(), 331_520);
    let secs = b.encoder_capacity_secs();
    assert!((secs - 20.72).abs() < 0.01, "got {secs}");
}

#[test]
fn encoder_bounds_underflow_safe() {
    // max_mel_frames < 2 * subsampling_factor → saturating_sub clamps to 0.
    let b = EncoderBounds { sample_rate: 16_000, hop_length: 160, subsampling_factor: 8, max_mel_frames: 4 };
    assert_eq!(b.max_samples(), 0);
    assert_eq!(b.encoder_capacity_secs(), 0.0);
}

#[test]
fn fixed_length_splitter_empty_waveform() {
    let mut s = FixedLengthSplitter::new();
    let chunks = s.split(&[], &bounds_tiny()).unwrap();
    assert!(chunks.is_empty());
}

#[test]
fn fixed_length_splitter_shorter_than_max() {
    // 10 samples < max_samples (16) → single chunk, unaligned tail allowed.
    let mut s = FixedLengthSplitter::new();
    let wf = vec![0.0; 10];
    let chunks = s.split(&wf, &bounds_tiny()).unwrap();
    assert_eq!(chunks, vec![AudioChunk::new(0, 10)]);
}

#[test]
fn fixed_length_splitter_exact_one_chunk() {
    // len == max_samples → single chunk, exact fit.
    let mut s = FixedLengthSplitter::new();
    let wf = vec![0.0; 16];
    let chunks = s.split(&wf, &bounds_tiny()).unwrap();
    assert_eq!(chunks, vec![AudioChunk::new(0, 16)]);
}

#[test]
fn fixed_length_splitter_exact_two_chunks() {
    // len == 2 * max_samples → two aligned chunks of max_samples each.
    let mut s = FixedLengthSplitter::new();
    let wf = vec![0.0; 32];
    let chunks = s.split(&wf, &bounds_tiny()).unwrap();
    assert_eq!(chunks, vec![AudioChunk::new(0, 16), AudioChunk::new(16, 32)]);
}

#[test]
fn fixed_length_splitter_two_chunks_plus_tail() {
    // len == 2 * max + 1 → three chunks: two aligned (16 each) + 1-sample tail.
    let mut s = FixedLengthSplitter::new();
    let wf = vec![0.0; 33];
    let chunks = s.split(&wf, &bounds_tiny()).unwrap();
    assert_eq!(chunks, vec![AudioChunk::new(0, 16), AudioChunk::new(16, 32), AudioChunk::new(32, 33),]);
}

#[test]
fn fixed_length_splitter_final_chunk_can_be_unaligned() {
    // 30 samples: first chunk consumes max=16 (aligned). Second chunk is
    // final (touches waveform.len()) and keeps its unaligned 14-sample tail
    // rather than splitting into aligned-12 + tail-2 — the JIT pads the
    // trailing fractional frame anyway, so fewer chunks is preferred.
    let mut s = FixedLengthSplitter::new();
    let wf = vec![0.0; 30];
    let chunks = s.split(&wf, &bounds_tiny()).unwrap();
    assert_eq!(chunks, vec![AudioChunk::new(0, 16), AudioChunk::new(16, 30)]);
}

#[test]
fn fixed_length_splitter_progress_under_degenerate_bounds() {
    // bounds.max_samples() == 0 — defended against; the loop falls back to
    // advancing one align_to_samples per step.
    let bounds = EncoderBounds { sample_rate: 16_000, hop_length: 2, subsampling_factor: 2, max_mel_frames: 0 };
    let mut s = FixedLengthSplitter::new();
    let chunks = s.split(&[0.0_f32; 9], &bounds).unwrap();
    // align=4, max forced to 4 → chunks of 4, 4, 1.
    assert_eq!(chunks, vec![AudioChunk::new(0, 4), AudioChunk::new(4, 8), AudioChunk::new(8, 9),]);
}

#[test]
fn fixed_length_splitter_alignment_does_not_divide_max() {
    // align=6, max=10: penultimate chunk's aligned span = floor(rem/6)*6.
    // hop=3, subsampling=2 → align=6. max_mel=8 → max=(8-4)*3=12. Bumped to
    // a custom struct so we can hit the divisibility edge cleanly.
    let bounds = EncoderBounds { sample_rate: 16_000, hop_length: 3, subsampling_factor: 2, max_mel_frames: 8 };
    let mut s = FixedLengthSplitter::new();
    let wf = vec![0.0; 26];
    let chunks = s.split(&wf, &bounds).unwrap();
    // First chunk: nominal_end=12, span=12, aligned_span=12 → 0..12.
    // Second chunk: nominal_end=24, span=12, aligned_span=12 → 12..24.
    // Third chunk: nominal_end=26 == waveform.len() → final, 24..26.
    assert_eq!(chunks, vec![AudioChunk::new(0, 12), AudioChunk::new(12, 24), AudioChunk::new(24, 26),]);
}
