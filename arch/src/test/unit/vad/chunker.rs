use proptest::prelude::*;

use crate::vad::{AudioChunk, ChunkerOpts, Error, chunks_from_probs};

/// Trivial coordinate system: 1 prob = 1 sample = 1 second. Lets us write
/// assertions in human units. Real ASR uses 31.25 probs/sec via
/// `(sample_rate, samples_per_prob)`.
fn fast_opts() -> ChunkerOpts {
    ChunkerOpts {
        sample_rate: 1,
        samples_per_prob: 1,
        threshold: 0.5,
        min_duration: 5.0,
        max_duration: 10.0,
        strict_limit_duration: 15.0,
        min_speech_probs: 1,
        min_silence_probs: 2,
        merge_gap_probs: 0,
        trough_search_probs: None,
        pad_samples: 0,
        align_to: 1,
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

#[test]
fn test_chunker_single_segment_under_max() {
    // 7 probs of speech (= 7s, < max=10) bracketed by 3-prob silences.
    let probs = cat(&[silence(3), speech(7), silence(3)]);
    let chunks = chunks_from_probs(&probs, &fast_opts()).unwrap();
    assert_eq!(chunks, vec![AudioChunk { start_sample: 3, end_sample: 10 }]);
}

#[test]
fn test_chunker_pack_two_segments_under_max() {
    // 4s + 4s with 2s silence gap = 10s total → fits one chunk (max=10).
    let probs = cat(&[speech(4), silence(2), speech(4)]);
    let chunks = chunks_from_probs(&probs, &fast_opts()).unwrap();
    assert_eq!(chunks, vec![AudioChunk { start_sample: 0, end_sample: 10 }]);
}

#[test]
fn test_chunker_close_at_inter_segment_silence() {
    // Three 4s segments with 2s gaps: span 16s. min=5, max=10.
    // - First two pack into [0, 10] (= 10s, exactly max).
    // - Third doesn't fit (would push to 16 > max), and cur_len=10 ≥ min.
    //   Close → chunk[0]=[0,10], chunk[1]=[12,16]. Silence [10,12) dropped.
    let probs = cat(&[speech(4), silence(2), speech(4), silence(2), speech(4)]);
    let chunks = chunks_from_probs(&probs, &fast_opts()).unwrap();
    assert_eq!(
        chunks,
        vec![AudioChunk { start_sample: 0, end_sample: 10 }, AudioChunk { start_sample: 12, end_sample: 16 }]
    );
}

#[test]
fn test_chunker_strict_limit_splits_long_run() {
    // One 30-prob unbroken segment with a deliberate prob trough at index 14
    // (value 0.4). With strict=15, n=ceil(30/15)=2 ⇒ one split target at 15.
    // Search radius = min_silence_probs = 2 ⇒ window [13..=17]. argmin lands
    // on index 14 — *not* the geometric target 15. This is the "do better
    // than GigaAM" property: split at the trough, not at strict_limit.
    let mut probs = vec![1.0f32; 30];
    probs[14] = 0.4;
    let chunks = chunks_from_probs(&probs, &fast_opts()).unwrap();
    assert_eq!(
        chunks,
        vec![AudioChunk { start_sample: 0, end_sample: 14 }, AudioChunk { start_sample: 14, end_sample: 30 }]
    );
}

#[test]
fn test_chunker_drops_silence_between_chunks() {
    // Two well-separated 4s segments, each becomes its own chunk; the 8s
    // silence gap between them is gone from the output.
    let probs = cat(&[speech(4), silence(8), speech(4)]);
    let chunks = chunks_from_probs(&probs, &fast_opts()).unwrap();
    assert_eq!(
        chunks,
        vec![AudioChunk { start_sample: 0, end_sample: 4 }, AudioChunk { start_sample: 12, end_sample: 16 }]
    );
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
    // 50-prob unbroken speech, strict=15. Every chunk ≤ strict_limit; total
    // coverage equals input length.
    let probs = vec![1.0f32; 50];
    let opts = fast_opts();
    let chunks = chunks_from_probs(&probs, &opts).unwrap();
    assert!(!chunks.is_empty());
    for c in &chunks {
        assert!(c.end_sample - c.start_sample <= 15, "chunk {c:?} exceeds strict_limit");
    }
    assert_eq!(chunks.first().unwrap().start_sample, 0);
    assert_eq!(chunks.last().unwrap().end_sample, 50);
}

#[test]
fn test_chunker_pad_samples() {
    let probs = cat(&[silence(10), speech(10), silence(10)]);
    let opts = ChunkerOpts { pad_samples: 5, ..fast_opts() };
    let chunks = chunks_from_probs(&probs, &opts).unwrap();
    // Raw chunk would be (10, 20). With pad=5: (5, 25).
    assert_eq!(chunks, vec![AudioChunk { start_sample: 5, end_sample: 25 }]);
}

#[test]
fn test_chunker_pad_clamps_at_edges() {
    // Speech right at the start; padding past 0 saturates.
    let probs = cat(&[speech(8), silence(10)]);
    let opts = ChunkerOpts { pad_samples: 100, ..fast_opts() };
    let chunks = chunks_from_probs(&probs, &opts).unwrap();
    let max_sample = probs.len(); // samples_per_prob = 1
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].start_sample, 0);
    assert_eq!(chunks[0].end_sample, max_sample);
}

#[test]
fn test_chunker_align_to_640() {
    // 10 probs, samples_per_prob=512, one speech segment at probs [3..7).
    // Raw samples: (1536, 3584). With align_to=640:
    //   start floor: 1536 / 640 = 2.4 → 2 * 640 = 1280.
    //   end ceil:    3584 / 640 = 5.6 → 6 * 640 = 3840.
    let probs = cat(&[silence(3), speech(4), silence(3)]);
    let opts = ChunkerOpts {
        sample_rate: 16000,
        samples_per_prob: 512,
        threshold: 0.5,
        min_duration: 0.0,
        max_duration: 1.0,
        strict_limit_duration: 1.0,
        min_speech_probs: 1,
        min_silence_probs: 2,
        merge_gap_probs: 0,
        trough_search_probs: None,
        pad_samples: 0,
        align_to: 640,
    };
    let chunks = chunks_from_probs(&probs, &opts).unwrap();
    assert_eq!(chunks, vec![AudioChunk { start_sample: 1280, end_sample: 3840 }]);
    for c in &chunks {
        assert_eq!(c.start_sample % 640, 0);
        assert_eq!(c.end_sample % 640, 0);
    }
}

#[test]
fn test_chunker_validates_min_exceeds_max() {
    let opts = ChunkerOpts { min_duration: 30.0, max_duration: 22.0, ..ChunkerOpts::default() };
    match chunks_from_probs(&[], &opts) {
        Err(Error::MinExceedsMax { min, max }) => {
            assert_eq!(min, 30.0);
            assert_eq!(max, 22.0);
        }
        other => panic!("expected MinExceedsMax, got {other:?}"),
    }
}

#[test]
fn test_chunker_validates_max_exceeds_strict() {
    let opts = ChunkerOpts { max_duration: 40.0, strict_limit_duration: 30.0, ..ChunkerOpts::default() };
    match chunks_from_probs(&[], &opts) {
        Err(Error::MaxExceedsStrict { max, strict }) => {
            assert_eq!(max, 40.0);
            assert_eq!(strict, 30.0);
        }
        other => panic!("expected MaxExceedsStrict, got {other:?}"),
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

// ─── Proptests ─────────────────────────────────────────────────────────────
//
// These exercise the algorithm across a broad parameter sweep and check
// invariants that hand-rolled unit tests can't easily cover:
//
// 1. Structural — sorted, non-overlapping, in-bounds, alignment-aligned, and
//    deterministic across re-runs of the same input/opts.
// 2. Strict-limit — with `pad=0, align=1`, every output chunk's sample length
//    is bounded by `(strict_limit_probs + 2 * trough_radius) * samples_per_prob`.
//    The `2 * radius` slack accounts for split_long_runs pieces that can
//    exceed strict_limit by up to `radius` on each side; pack_segments's
//    strict-limit guard caps any further growth.
// 3. Coverage — with all smoothing knobs set to 1 (no smoothing) and
//    `pad=0, align=1`, every above-threshold prob in the input falls inside
//    some output chunk's sample range. Catches phantom-coverage regressions
//    where speech regions are silently dropped.

proptest! {
    #![proptest_config(ProptestConfig { cases: 200, ..ProptestConfig::default() })]

    #[test]
    fn prop_chunker_structural_invariants(
        probs in prop::collection::vec(0.0f32..=1.0f32, 0..400),
        sample_rate in prop::sample::select(vec![8_000u32, 16_000, 22_050, 44_100, 48_000]),
        samples_per_prob in prop::sample::select(vec![64usize, 128, 256, 512, 1024]),
        threshold in 0.2f32..0.8,
        min_dur in 0.05f32..=2.0,
        max_extra in 0.0f32..=4.0,
        strict_extra in 0.0f32..=8.0,
        min_speech in 1usize..=8,
        min_silence in 1usize..=8,
        merge_gap in 0usize..=8,
        align_to in prop::sample::select(vec![1usize, 64, 256, 512, 640, 1024, 2048]),
        pad_samples in 0usize..=2048,
    ) {
        let max_dur = min_dur + max_extra;
        let strict_dur = max_dur + strict_extra;
        let opts = ChunkerOpts {
            sample_rate,
            samples_per_prob,
            threshold,
            min_duration: min_dur,
            max_duration: max_dur,
            strict_limit_duration: strict_dur,
            min_speech_probs: min_speech,
            min_silence_probs: min_silence,
            merge_gap_probs: merge_gap,
            trough_search_probs: None,
            pad_samples,
            align_to,
        };
        let chunks = chunks_from_probs(&probs, &opts).unwrap();
        let max_sample = probs.len() * samples_per_prob;

        // Sorted + non-overlapping (touching allowed: pack_segments may
        // cut at silence boundaries that produce shared sample indices).
        for w in chunks.windows(2) {
            prop_assert!(
                w[0].end_sample <= w[1].start_sample,
                "overlapping chunks: {:?} and {:?}", w[0], w[1]
            );
        }
        // Each chunk is non-empty and inside the input extent.
        for c in &chunks {
            prop_assert!(c.start_sample < c.end_sample, "empty chunk: {c:?}");
            prop_assert!(c.end_sample <= max_sample,
                "chunk {c:?} exceeds max_sample {max_sample}");
        }
        // Alignment: start always aligned, end aligned OR clamped to max_sample
        // (which is the only legitimate way an end can be non-aligned).
        for c in &chunks {
            prop_assert_eq!(c.start_sample % align_to, 0,
                "start {} not aligned to {}", c.start_sample, align_to);
            let end_aligned = c.end_sample % align_to == 0;
            let end_at_max = c.end_sample == max_sample;
            prop_assert!(end_aligned || end_at_max,
                "end {} not aligned to {} and not at max_sample {}",
                c.end_sample, align_to, max_sample);
        }
        // Same input, same output.
        let chunks2 = chunks_from_probs(&probs, &opts).unwrap();
        prop_assert_eq!(chunks, chunks2);
    }

    #[test]
    fn prop_chunker_strict_limit_bound(
        probs in prop::collection::vec(0.0f32..=1.0f32, 0..400),
        sample_rate in prop::sample::select(vec![8_000u32, 16_000, 48_000]),
        samples_per_prob in prop::sample::select(vec![128usize, 512, 1024]),
        threshold in 0.2f32..0.8,
        min_dur in 0.1f32..=2.0,
        max_extra in 0.1f32..=5.0,
        strict_extra in 0.0f32..=10.0,
        min_speech in 1usize..=8,
        min_silence in 1usize..=8,
        merge_gap in 0usize..=8,
    ) {
        let max_dur = min_dur + max_extra;
        let strict_dur = max_dur + strict_extra;
        let opts = ChunkerOpts {
            sample_rate,
            samples_per_prob,
            threshold,
            min_duration: min_dur,
            max_duration: max_dur,
            strict_limit_duration: strict_dur,
            min_speech_probs: min_speech,
            min_silence_probs: min_silence,
            merge_gap_probs: merge_gap,
            trough_search_probs: None,
            pad_samples: 0,
            align_to: 1,
        };
        let chunks = chunks_from_probs(&probs, &opts).unwrap();
        let probs_per_sec = sample_rate as f32 / samples_per_prob as f32;
        let strict_limit_probs = (strict_dur * probs_per_sec).ceil() as usize;
        // With `trough_search_probs: None` the radius defaults to min_silence.
        let radius = min_silence;
        let bound_samples = (strict_limit_probs + 2 * radius) * samples_per_prob;
        for c in &chunks {
            let len = c.end_sample - c.start_sample;
            prop_assert!(
                len <= bound_samples,
                "chunk {c:?} length {len} exceeds bound {bound_samples} \
                 (strict_probs={strict_limit_probs}, radius={radius}, spp={samples_per_prob})"
            );
        }
    }

    #[test]
    fn prop_chunker_unsmoothed_coverage(
        probs in prop::collection::vec(0.0f32..=1.0f32, 1..400),
        threshold in 0.3f32..=0.7,
    ) {
        // With min_speech=min_silence=1, merge_gap=0, pad=0, align=1, every
        // above-threshold prob is its own speech run and must be inside
        // some output chunk's sample range. If this ever fires, the
        // chunker is dropping speech.
        let samples_per_prob = 512usize;
        let opts = ChunkerOpts {
            sample_rate: 16_000,
            samples_per_prob,
            threshold,
            min_duration: 0.05,
            max_duration: 0.5,
            strict_limit_duration: 0.5,
            min_speech_probs: 1,
            min_silence_probs: 1,
            merge_gap_probs: 0,
            trough_search_probs: None,
            pad_samples: 0,
            align_to: 1,
        };
        let chunks = chunks_from_probs(&probs, &opts).unwrap();
        for (i, &p) in probs.iter().enumerate() {
            if p >= threshold {
                let lo = i * samples_per_prob;
                let hi = (i + 1) * samples_per_prob;
                let covered = chunks.iter().any(|c| c.start_sample <= lo && hi <= c.end_sample);
                prop_assert!(
                    covered,
                    "above-threshold prob {p:.3} at index {i} (samples {lo}..{hi}) \
                     not covered by any chunk: {chunks:?}"
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
    assert!((opts.threshold - default.threshold).abs() < 1e-6);
    assert!((opts.min_duration - default.min_duration).abs() < 1e-6);
    assert!((opts.max_duration - default.max_duration).abs() < 1e-6);
    assert!((opts.strict_limit_duration - default.strict_limit_duration).abs() < 1e-6);
    assert_eq!(opts.min_speech_probs, default.min_speech_probs);
    assert_eq!(opts.min_silence_probs, default.min_silence_probs);
    assert_eq!(opts.merge_gap_probs, default.merge_gap_probs);
    assert_eq!(opts.pad_samples, default.pad_samples);
    assert_eq!(opts.align_to, default.align_to);
}

#[test]
fn test_chunker_split_long_runs_respects_min_piece_floor() {
    // 25-prob unbroken speech with a deep trough at index 1 and a wide
    // search radius. Without the min_piece floor, the first split would
    // land on probs[1]=0.0, leaving a 1-prob shard. With the floor
    // (min_piece = 25 / (2 * 3) = 4) the split is held back to ≥ index 4.
    let mut probs = vec![1.0f32; 25];
    probs[1] = 0.0;
    let opts = ChunkerOpts {
        sample_rate: 1,
        samples_per_prob: 1,
        threshold: 0.5,
        min_duration: 1.0,
        max_duration: 10.0,
        strict_limit_duration: 10.0,
        min_speech_probs: 1,
        min_silence_probs: 100, // suppress smoothing-driven termination
        merge_gap_probs: 0,
        trough_search_probs: Some(10),
        pad_samples: 0,
        align_to: 1,
    };
    let chunks = chunks_from_probs(&probs, &opts).unwrap();
    assert!(!chunks.is_empty());
    let first_len = chunks[0].end_sample - chunks[0].start_sample;
    assert!(first_len >= 4, "first chunk too small ({first_len} samples): {chunks:?}");
}

#[test]
fn test_chunker_decoupled_trough_search_radius() {
    // Same 30-prob long-run setup as test_chunker_strict_limit_splits_long_run,
    // but with min_silence=2 (tight smoothing) and a deliberately wide
    // trough_search_probs. Verifies the radius knob is independent of
    // min_silence_probs: the wider window still finds the trough at idx 14
    // even though min_silence is small.
    let mut probs = vec![1.0f32; 30];
    probs[14] = 0.4;
    let opts = ChunkerOpts {
        sample_rate: 1,
        samples_per_prob: 1,
        threshold: 0.5,
        min_duration: 1.0,
        max_duration: 10.0,
        strict_limit_duration: 15.0,
        min_speech_probs: 1,
        min_silence_probs: 2,
        merge_gap_probs: 0,
        trough_search_probs: Some(8),
        pad_samples: 0,
        align_to: 1,
    };
    let chunks = chunks_from_probs(&probs, &opts).unwrap();
    assert_eq!(
        chunks,
        vec![AudioChunk { start_sample: 0, end_sample: 14 }, AudioChunk { start_sample: 14, end_sample: 30 }]
    );
}

#[cfg(feature = "serde")]
#[test]
fn test_chunker_serde_partial_overrides() {
    // Confirms that serde(default) on the struct lets partial JSON populate
    // only the named fields; unspecified fields fall back to Default.
    let json = r#"{ "min_duration": 10.0, "align_to": 640 }"#;
    let opts: ChunkerOpts = serde_json::from_str(json).unwrap();
    assert!((opts.min_duration - 10.0).abs() < 1e-6);
    assert_eq!(opts.align_to, 640);
    // Other fields stayed at default.
    assert_eq!(opts.sample_rate, 16_000);
    assert_eq!(opts.samples_per_prob, 512);
}
