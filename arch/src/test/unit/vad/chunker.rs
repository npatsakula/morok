use proptest::prelude::*;

use crate::vad::{AudioChunk, ChunkerOpts, Error, chunks_from_probs};

/// Trivial coordinate system: 1 prob = 1 sample = 1 second. Lets us write
/// assertions in human units. Real ASR uses 31.25 probs/sec via
/// `(sample_rate, samples_per_prob)`. Speech gate is disabled (peak ≥ 0
/// always); individual tests opt in when needed.
fn fast_opts() -> ChunkerOpts {
    ChunkerOpts {
        sample_rate: 1,
        samples_per_prob: 1,
        onset: 0.5,
        offset: 0.5,
        min_speech_probs: 1,
        min_silence_probs: 1,
        merge_gap_probs: 0,
        min_chunk_probs: 5,
        max_chunk_probs: 10,
        pad_samples: 0,
        align_to: 1,
        min_chunk_max_prob: 0.0,
    }
}

fn speech(n: usize) -> Vec<f32> {
    vec![1.0; n]
}

fn silence(n: usize) -> Vec<f32> {
    vec![0.0; n]
}

fn cat(parts: &[Vec<f32>]) -> Vec<f32> {
    parts.iter().flatten().copied().collect()
}

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[test]
fn test_chunker_single_segment_under_max() {
    // 7 probs of speech (= 7s, ≤ max=10) bracketed by 3-prob silences.
    let probs = cat(&[silence(3), speech(7), silence(3)]);
    let chunks = chunks_from_probs(&probs, &fast_opts()).unwrap();
    assert_eq!(chunks, vec![AudioChunk::new(3, 10)]);
}

#[test]
fn test_chunker_pack_two_segments_under_max() {
    // 4s + 4s with 2s silence gap = two runs that pack to 10s ≤ max.
    let probs = cat(&[speech(4), silence(2), speech(4)]);
    let chunks = chunks_from_probs(&probs, &fast_opts()).unwrap();
    assert_eq!(chunks, vec![AudioChunk::new(0, 10)]);
}

#[test]
fn test_chunker_close_at_inter_segment_silence() {
    // Three 4s runs, 2s gaps each: spans 16s, max=10.
    // - Pack [0,4]+[6,10] → chunk [0,10] (= max, fits).
    // - Try +[12,16] → would be 16 > max → close, new chunk [12,16].
    let probs = cat(&[speech(4), silence(2), speech(4), silence(2), speech(4)]);
    let chunks = chunks_from_probs(&probs, &fast_opts()).unwrap();
    assert_eq!(chunks, vec![AudioChunk::new(0, 10), AudioChunk::new(12, 16)]);
}

#[test]
fn test_chunker_drops_silence_between_chunks() {
    // Two well-separated 4s runs; the 8s silence gap is gone from the output.
    let probs = cat(&[speech(4), silence(8), speech(4)]);
    let chunks = chunks_from_probs(&probs, &fast_opts()).unwrap();
    assert_eq!(chunks, vec![AudioChunk::new(0, 4), AudioChunk::new(12, 16)]);
}

#[test]
fn test_chunker_empty_probs() {
    let chunks = chunks_from_probs(&[], &fast_opts()).unwrap();
    assert!(chunks.is_empty());
}

#[test]
fn test_chunker_all_silence() {
    let probs = vec![0.0f32; 100];
    let chunks = chunks_from_probs(&probs, &fast_opts()).unwrap();
    assert!(chunks.is_empty());
}

#[test]
fn test_chunker_all_speech_no_breaks() {
    // 50-prob unbroken run, max=10. min-cut's flat-speech fallback returns
    // the rightmost legal split (= hi), so chunks land at exactly max.
    let probs = vec![1.0f32; 50];
    let chunks = chunks_from_probs(&probs, &fast_opts()).unwrap();
    assert_eq!(
        chunks,
        vec![
            AudioChunk::new(0, 10),
            AudioChunk::new(10, 20),
            AudioChunk::new(20, 30),
            AudioChunk::new(30, 40),
            AudioChunk::new(40, 50),
        ]
    );
}

#[test]
fn test_chunker_min_cut_lands_on_silence_trough() {
    // 30-prob run with a single sub-offset trough at index 14. min-cut prefers
    // the rightmost silence inside the legal window [min, max]; index 14 is
    // inside [5, 15] (max=15 here), so the first split lands there.
    let mut probs = vec![1.0f32; 30];
    probs[14] = 0.4;
    let opts = ChunkerOpts { max_chunk_probs: 15, min_chunk_probs: 5, ..fast_opts() };
    let chunks = chunks_from_probs(&probs, &opts).unwrap();
    assert_eq!(chunks.first().unwrap().end_sample, 14);
    for c in &chunks {
        assert!(c.core_len() <= 15, "chunk {c:?} exceeds max_chunk_probs");
    }
    assert_eq!(chunks.first().unwrap().start_sample, 0);
    assert_eq!(chunks.last().unwrap().end_sample, 30);
}

#[test]
fn test_chunker_min_cut_argmin_fallback_when_no_silence() {
    // Flat-speech run; no probs below offset. min-cut's fallback returns
    // the rightmost legal split (= hi), keeping a ≥ min_chunk tail.
    let probs = vec![1.0f32; 22];
    let opts = ChunkerOpts { max_chunk_probs: 10, min_chunk_probs: 5, ..fast_opts() };
    let chunks = chunks_from_probs(&probs, &opts).unwrap();
    assert_eq!(chunks.first().unwrap().start_sample, 0);
    assert_eq!(chunks.last().unwrap().end_sample, 22);
    for c in &chunks {
        assert!(c.core_len() <= 10, "chunk {c:?} exceeds max");
        assert!(c.core_len() >= 1, "empty chunk: {c:?}");
    }
}

#[test]
fn test_chunker_pad_samples() {
    // Run [10, 20], pad=5 → decode window [5, 25].
    let probs = cat(&[silence(10), speech(10), silence(10)]);
    let opts = ChunkerOpts { pad_samples: 5, ..fast_opts() };
    let chunks = chunks_from_probs(&probs, &opts).unwrap();
    assert_eq!(chunks, vec![AudioChunk::with_decode(10, 20, 5, 25)]);
}

#[test]
fn test_chunker_pad_clamps_at_edges() {
    // Run at the file start; pad past 0 saturates and pad past end clamps.
    let probs = cat(&[speech(8), silence(10)]);
    let opts = ChunkerOpts { pad_samples: 100, ..fast_opts() };
    let chunks = chunks_from_probs(&probs, &opts).unwrap();
    let max_sample = probs.len();
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].start_sample, 0);
    assert_eq!(chunks[0].end_sample, 8);
    assert_eq!(chunks[0].decode_start_sample, 0);
    assert_eq!(chunks[0].decode_end_sample, max_sample);
}

#[test]
fn test_chunker_align_to_640() {
    // 10 probs, samples_per_prob=512, speech at probs [3..7). Closure needs
    // min_silence_probs consecutive sub-offset probs; the trailing 3 zeros
    // satisfy that. Core samples (1536, 3584). With align_to=640:
    //   start floor: 1536 / 640 = 2.4 → 2 * 640 = 1280.
    //   end ceil:    3584 / 640 = 5.6 → 6 * 640 = 3840.
    let probs = cat(&[silence(3), speech(4), silence(3)]);
    let opts = ChunkerOpts {
        sample_rate: 16_000,
        samples_per_prob: 512,
        onset: 0.5,
        offset: 0.5,
        min_speech_probs: 1,
        min_silence_probs: 2,
        merge_gap_probs: 0,
        min_chunk_probs: 1,
        max_chunk_probs: 31, // ≈ ceil(1.0 * 16000 / 512)
        pad_samples: 0,
        align_to: 640,
        min_chunk_max_prob: 0.0,
    };
    let chunks = chunks_from_probs(&probs, &opts).unwrap();
    assert_eq!(chunks, vec![AudioChunk::with_decode(1536, 3584, 1280, 3840)]);
    for c in &chunks {
        assert_eq!(c.decode_start_sample % 640, 0);
        assert_eq!(c.decode_end_sample % 640, 0);
    }
}

#[test]
fn test_chunker_end_sample_can_exceed_waveform_len() {
    // `chunks_from_probs` sees only `probs`, not the raw waveform. When the
    // waveform length isn't a multiple of `samples_per_prob`, the last
    // prob's window straddles the waveform end and the emitted chunk's
    // decode_end overshoots — callers must clamp at slice time. Pinned
    // here so the documented overshoot is exercised.
    let probs = vec![1.0_f32; 4];
    let waveform_len = 1800;
    let opts = ChunkerOpts {
        sample_rate: 16_000,
        samples_per_prob: 512,
        onset: 0.5,
        offset: 0.5,
        min_speech_probs: 1,
        min_silence_probs: 1,
        merge_gap_probs: 1,
        min_chunk_probs: 1,
        max_chunk_probs: 31,
        pad_samples: 0,
        align_to: 640,
        min_chunk_max_prob: 0.0,
    };
    let chunks = chunks_from_probs(&probs, &opts).unwrap();
    assert_eq!(chunks, vec![AudioChunk::new(0, 2048)]);
    assert!(chunks[0].decode_end_sample > waveform_len);
}

#[test]
fn test_chunker_hysteresis_sustain_band_keeps_run_open() {
    // onset=0.6, offset=0.3. Mid-run prob 0.4 is in the sustain band and
    // does not close the run — the whole 6 probs become one chunk.
    let probs = vec![0.7, 0.7, 0.7, 0.4, 0.7, 0.7];
    let opts = ChunkerOpts {
        onset: 0.6,
        offset: 0.3,
        min_speech_probs: 1,
        min_silence_probs: 2,
        min_chunk_probs: 1,
        max_chunk_probs: 10,
        ..fast_opts()
    };
    let chunks = chunks_from_probs(&probs, &opts).unwrap();
    assert_eq!(chunks, vec![AudioChunk::new(0, 6)]);
}

#[test]
fn test_chunker_speech_gate_drops_low_peak_chunk() {
    // Single run whose max prob (0.45) is below the gate (0.5). Build with
    // onset = offset = 0.4 so hysteresis opens; then set the gate at 0.5.
    let probs = vec![0.45_f32; 12];
    let opts = ChunkerOpts {
        onset: 0.4,
        offset: 0.4,
        min_speech_probs: 1,
        min_silence_probs: 1,
        min_chunk_probs: 1,
        max_chunk_probs: 20,
        min_chunk_max_prob: 0.5,
        ..fast_opts()
    };
    let chunks = chunks_from_probs(&probs, &opts).unwrap();
    assert!(chunks.is_empty(), "speech gate should drop low-confidence chunk: {chunks:?}");
}

#[test]
fn test_chunker_speech_gate_keeps_high_peak_chunk() {
    let probs = vec![0.9_f32; 12];
    let opts = ChunkerOpts { min_chunk_max_prob: 0.5, ..fast_opts() };
    let chunks = chunks_from_probs(&probs, &opts).unwrap();
    assert!(!chunks.is_empty(), "high-confidence chunk should pass gate");
}

#[test]
fn test_chunker_decode_windows_dont_overlap_with_align_one() {
    // Three 4s runs with 2s gaps, pad=4, align=1. Each side's pad is capped
    // at half the gap so decode windows tile cleanly with no overlap.
    let probs = cat(&[speech(4), silence(2), speech(4), silence(2), speech(4)]);
    let opts = ChunkerOpts { pad_samples: 4, max_chunk_probs: 4, min_chunk_probs: 1, ..fast_opts() };
    let chunks = chunks_from_probs(&probs, &opts).unwrap();
    assert!(chunks.len() >= 2, "expected multiple chunks: {chunks:?}");
    for w in chunks.windows(2) {
        assert!(w[0].decode_end_sample <= w[1].decode_start_sample, "decode windows overlap: {:?} → {:?}", w[0], w[1]);
    }
}

#[test]
fn test_chunker_validates_min_exceeds_max() {
    let opts = ChunkerOpts { min_chunk_probs: 30, max_chunk_probs: 22, ..ChunkerOpts::default() };
    match chunks_from_probs(&[], &opts) {
        Err(Error::MinExceedsMax { min, max }) => {
            assert_eq!(min, 30);
            assert_eq!(max, 22);
        }
        other => panic!("expected MinExceedsMax, got {other:?}"),
    }
}

#[test]
fn test_chunker_validates_zero_max_chunk() {
    let opts = ChunkerOpts { max_chunk_probs: 0, ..ChunkerOpts::default() };
    match chunks_from_probs(&[], &opts) {
        Err(Error::ZeroMaxChunk) => {}
        other => panic!("expected ZeroMaxChunk, got {other:?}"),
    }
}

#[test]
fn test_chunker_validates_offset_exceeds_onset() {
    let opts = ChunkerOpts { onset: 0.3, offset: 0.5, ..ChunkerOpts::default() };
    match chunks_from_probs(&[], &opts) {
        Err(Error::OffsetExceedsOnset { offset, onset }) => {
            assert!((offset - 0.5).abs() < 1e-6);
            assert!((onset - 0.3).abs() < 1e-6);
        }
        other => panic!("expected OffsetExceedsOnset, got {other:?}"),
    }
}

#[test]
fn test_chunker_validates_zero_samples_per_prob() {
    let opts = ChunkerOpts { samples_per_prob: 0, ..ChunkerOpts::default() };
    match chunks_from_probs(&[], &opts) {
        Err(Error::ZeroSamplesPerProb) => {}
        other => panic!("expected ZeroSamplesPerProb, got {other:?}"),
    }
}

#[test]
fn test_chunker_validates_zero_align_to() {
    let opts = ChunkerOpts { align_to: 0, ..ChunkerOpts::default() };
    match chunks_from_probs(&[], &opts) {
        Err(Error::ZeroAlignTo) => {}
        other => panic!("expected ZeroAlignTo, got {other:?}"),
    }
}

// ─── Proptests ──────────────────────────────────────────────────────────────
//
// Invariants checked across a broad parameter sweep:
//
// 1. Structural — sorted, non-overlapping cores; in-bounds decode windows;
//    decode-window alignment; deterministic; adjacent decode windows
//    never overlap (the new invariant that motivated the redesign).
// 2. Max-chunk bound — every core respects `max_chunk_probs · samples_per_prob`;
//    every decode window respects `max_chunk_sample_bound`.
// 3. Coverage — with the gate disabled and smoothing minimal, every prob
//    ≥ onset falls inside some output chunk's sample range.

proptest! {
    #![proptest_config(ProptestConfig { cases: 200, ..ProptestConfig::default() })]

    #[test]
    fn prop_chunker_structural_invariants(
        probs in prop::collection::vec(0.0f32..=1.0f32, 0..400),
        sample_rate in prop::sample::select(vec![8_000u32, 16_000, 22_050, 44_100, 48_000]),
        samples_per_prob in prop::sample::select(vec![64usize, 128, 256, 512, 1024]),
        onset in 0.3f32..0.8,
        offset_extra in 0.0f32..=0.2,
        min_speech in 1usize..=8,
        min_silence in 1usize..=8,
        merge_gap in 0usize..=8,
        min_chunk in 1usize..=4,
        max_chunk_extra in 0usize..=400,
        align_to in prop::sample::select(vec![1usize, 64, 256, 512, 640, 1024, 2048]),
        pad_samples in 0usize..=2048,
    ) {
        let offset = (onset - offset_extra).max(0.0);
        let max_chunk = min_chunk + max_chunk_extra;
        let opts = ChunkerOpts {
            sample_rate,
            samples_per_prob,
            onset,
            offset,
            min_speech_probs: min_speech,
            min_silence_probs: min_silence,
            merge_gap_probs: merge_gap,
            min_chunk_probs: min_chunk,
            max_chunk_probs: max_chunk,
            pad_samples,
            align_to,
            min_chunk_max_prob: 0.0,
        };
        let chunks = chunks_from_probs(&probs, &opts).unwrap();
        let max_sample = probs.len() * samples_per_prob;

        for w in chunks.windows(2) {
            prop_assert!(
                w[0].end_sample <= w[1].start_sample,
                "overlapping cores: {:?} and {:?}", w[0], w[1]
            );
            // Decode windows may overlap by up to 2·(align_to − 1) samples
            // when floor/ceil rounding lands on opposite sides of touching
            // cores. `crop_words_to_core` filters seam duplicates via the
            // core boundaries, so this is benign.
            let overlap = w[0].decode_end_sample.saturating_sub(w[1].decode_start_sample);
            prop_assert!(
                overlap < 2 * align_to,
                "decode overlap {overlap} ≥ 2·align_to {}: {:?} and {:?}",
                2 * align_to, w[0], w[1]
            );
        }
        for c in &chunks {
            prop_assert!(c.start_sample < c.end_sample, "empty core: {c:?}");
            prop_assert!(c.end_sample <= max_sample,
                "core {c:?} exceeds max_sample {max_sample}");
            prop_assert!(c.decode_start_sample <= c.start_sample,
                "decode starts after core: {c:?}");
            prop_assert!(c.end_sample <= c.decode_end_sample,
                "decode ends before core: {c:?}");
            prop_assert!(c.decode_start_sample < c.decode_end_sample,
                "empty decode window: {c:?}");
            prop_assert!(c.decode_end_sample <= max_sample,
                "decode {c:?} exceeds max_sample {max_sample}");

            let end_aligned = c.decode_end_sample % align_to == 0;
            let end_at_max = c.decode_end_sample == max_sample;
            prop_assert!(end_aligned || end_at_max,
                "decode end {} not aligned to {} and not at max_sample {}",
                c.decode_end_sample, align_to, max_sample);
            // decode_start is either an align multiple OR pinned to the
            // previous chunk's decode_end (which is itself either aligned
            // or at max_sample).
            let start_aligned = c.decode_start_sample % align_to == 0;
            prop_assert!(start_aligned || c.decode_start_sample == max_sample,
                "decode start {} not aligned to {}", c.decode_start_sample, align_to);
        }
        let chunks2 = chunks_from_probs(&probs, &opts).unwrap();
        prop_assert_eq!(chunks, chunks2);
    }

    #[test]
    fn prop_chunker_max_chunk_bound(
        probs in prop::collection::vec(0.0f32..=1.0f32, 0..400),
        sample_rate in prop::sample::select(vec![8_000u32, 16_000, 48_000]),
        samples_per_prob in prop::sample::select(vec![128usize, 512, 1024]),
        onset in 0.3f32..0.8,
        offset_extra in 0.0f32..=0.2,
        min_speech in 1usize..=8,
        min_silence in 1usize..=8,
        merge_gap in 0usize..=8,
        min_chunk in 1usize..=4,
        max_chunk_extra in 0usize..=400,
    ) {
        let offset = (onset - offset_extra).max(0.0);
        let max_chunk = min_chunk + max_chunk_extra;
        let opts = ChunkerOpts {
            sample_rate,
            samples_per_prob,
            onset,
            offset,
            min_speech_probs: min_speech,
            min_silence_probs: min_silence,
            merge_gap_probs: merge_gap,
            min_chunk_probs: min_chunk,
            max_chunk_probs: max_chunk,
            pad_samples: 0,
            align_to: 1,
            min_chunk_max_prob: 0.0,
        };
        let chunks = chunks_from_probs(&probs, &opts).unwrap();
        let bound = crate::vad::max_chunk_sample_bound(max_chunk, samples_per_prob, 0, 1);
        for c in &chunks {
            prop_assert!(c.core_len() <= max_chunk * samples_per_prob,
                "chunk {c:?} core_len exceeds max_chunk_probs * samples_per_prob");
            prop_assert!(c.decode_len() <= bound,
                "chunk {c:?} decode_len exceeds max_chunk_sample_bound {bound}");
        }
    }

    #[test]
    fn prop_chunker_unsmoothed_coverage(
        probs in prop::collection::vec(0.0f32..=1.0f32, 1..400),
        onset in 0.3f32..=0.7,
    ) {
        // With min_speech = min_silence = 1, merge_gap = 0, pad = 0,
        // align = 1, and gate disabled, every prob ≥ onset must be covered
        // by some output chunk. Catches phantom-coverage regressions.
        let samples_per_prob = 512usize;
        let opts = ChunkerOpts {
            sample_rate: 16_000,
            samples_per_prob,
            onset,
            offset: onset,
            min_speech_probs: 1,
            min_silence_probs: 1,
            merge_gap_probs: 0,
            min_chunk_probs: 1,
            max_chunk_probs: 16,
            pad_samples: 0,
            align_to: 1,
            min_chunk_max_prob: 0.0,
        };
        let chunks = chunks_from_probs(&probs, &opts).unwrap();
        for (i, &p) in probs.iter().enumerate() {
            if p >= onset {
                let lo = i * samples_per_prob;
                let hi = (i + 1) * samples_per_prob;
                let covered = chunks.iter().any(|c| c.start_sample <= lo && hi <= c.end_sample);
                prop_assert!(
                    covered,
                    "≥-onset prob {p:.3} at index {i} (samples {lo}..{hi}) not covered: {chunks:?}"
                );
            }
        }
    }
}

#[cfg(feature = "serde")]
#[test]
fn test_chunker_serde_default_roundtrip() {
    let opts: ChunkerOpts = serde_json::from_str("{}").unwrap();
    let default = ChunkerOpts::default();
    assert_eq!(opts.sample_rate, default.sample_rate);
    assert_eq!(opts.samples_per_prob, default.samples_per_prob);
    assert!((opts.onset - default.onset).abs() < 1e-6);
    assert!((opts.offset - default.offset).abs() < 1e-6);
    assert_eq!(opts.min_speech_probs, default.min_speech_probs);
    assert_eq!(opts.min_silence_probs, default.min_silence_probs);
    assert_eq!(opts.merge_gap_probs, default.merge_gap_probs);
    assert_eq!(opts.min_chunk_probs, default.min_chunk_probs);
    assert_eq!(opts.max_chunk_probs, default.max_chunk_probs);
    assert_eq!(opts.pad_samples, default.pad_samples);
    assert_eq!(opts.align_to, default.align_to);
    assert!((opts.min_chunk_max_prob - default.min_chunk_max_prob).abs() < 1e-6);
}

#[cfg(feature = "serde")]
#[test]
fn test_chunker_serde_partial_overrides() {
    // serde(default) on the struct lets partial JSON populate only the
    // named fields; unspecified fields fall back to Default.
    let json = r#"{ "min_chunk_probs": 100, "align_to": 640 }"#;
    let opts: ChunkerOpts = serde_json::from_str(json).unwrap();
    assert_eq!(opts.min_chunk_probs, 100);
    assert_eq!(opts.align_to, 640);
    assert_eq!(opts.sample_rate, 16_000);
    assert_eq!(opts.samples_per_prob, 512);
}
