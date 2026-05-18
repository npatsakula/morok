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
        cluster_target_duration: None,
        strict_limit_duration: 15.0,
        min_speech_probs: 1,
        min_silence_probs: 2,
        merge_gap_probs: 0,
        trough_search_probs: None,
        trough_threshold: None,
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
    assert_eq!(chunks, vec![AudioChunk::new(3, 10)]);
}

#[test]
fn test_chunker_pack_two_segments_under_max() {
    // 4s + 4s with 2s silence gap = 10s total → fits one chunk (max=10).
    let probs = cat(&[speech(4), silence(2), speech(4)]);
    let chunks = chunks_from_probs(&probs, &fast_opts()).unwrap();
    assert_eq!(chunks, vec![AudioChunk::new(0, 10)]);
}

#[test]
fn test_chunker_close_at_inter_segment_silence() {
    // Three 4s segments with 2s gaps: span 16s. min=5, max=10.
    // - First two pack into [0, 10] (= 10s, exactly max).
    // - Third doesn't fit (would push to 16 > max), and cur_len=10 ≥ min.
    //   Close → chunk[0]=[0,10], chunk[1]=[12,16]. Silence [10,12) dropped.
    let probs = cat(&[speech(4), silence(2), speech(4), silence(2), speech(4)]);
    let chunks = chunks_from_probs(&probs, &fast_opts()).unwrap();
    assert_eq!(chunks, vec![AudioChunk::new(0, 10), AudioChunk::new(12, 16)]);
}

#[test]
fn test_chunker_cluster_target_splits_at_internal_gap() {
    // The coarse packer would keep this as one 20s chunk. With a 8s cluster
    // target, the second pass splits at the real 4s VAD gap nearest that
    // target. The next core starts inside the gap, preserving bounded pre-roll
    // for low-confidence speech that may sit before the next VAD run.
    let probs = cat(&[speech(8), silence(4), speech(8)]);
    let opts = ChunkerOpts {
        max_duration: 30.0,
        strict_limit_duration: 30.0,
        cluster_target_duration: Some(8.0),
        ..fast_opts()
    };
    let chunks = chunks_from_probs(&probs, &opts).unwrap();
    assert_eq!(chunks, vec![AudioChunk::new(0, 8), AudioChunk::new(10, 20)]);
}

#[test]
fn test_chunker_cluster_target_splits_greedy_envelopes_at_midpoint_gaps() {
    // Mirrors a pause-heavy longform layout: many short speech runs should first
    // be packed into model-sized envelopes, then each long envelope split near
    // its midpoint. A global duration-only partition tends to cut after the
    // largest early gaps and separates short runs that belong together.
    let speech_lens = [229, 59, 14, 60, 92, 56, 68, 38, 92, 73, 51, 95, 24, 56];
    let gap_lens = [107, 92, 46, 28, 48, 39, 61, 32, 30, 46, 126, 117, 31];
    let mut parts = Vec::new();
    for (i, &speech_len) in speech_lens.iter().enumerate() {
        parts.push(speech(speech_len));
        if let Some(&gap_len) = gap_lens.get(i) {
            parts.push(silence(gap_len));
        }
    }
    let probs = cat(&parts);
    let opts = ChunkerOpts {
        sample_rate: 1,
        samples_per_prob: 1,
        threshold: 0.5,
        min_duration: 469.0,
        max_duration: 638.0,
        cluster_target_duration: Some(319.0),
        strict_limit_duration: 638.0,
        min_speech_probs: 8,
        min_silence_probs: 4,
        merge_gap_probs: 8,
        trough_search_probs: None,
        trough_threshold: None,
        pad_samples: 0,
        align_to: 1,
    };
    let chunks = chunks_from_probs(&probs, &opts).unwrap();
    assert_eq!(
        chunks,
        vec![
            AudioChunk::new(0, 229),
            AudioChunk::new(233, 607),
            AudioChunk::new(611, 938),
            AudioChunk::new(942, 1264),
            AudioChunk::new(1268, 1582),
            AudioChunk::new(1586, 1810),
        ]
    );
}

#[test]
fn test_chunker_strict_limit_splits_long_run() {
    // One 30-prob unbroken segment with a deliberate prob trough at index 14
    // (value 0.4). With strict=15, n=ceil(30/15)=2 ⇒ one split target at 15.
    // Search radius = min_silence_probs = 2, but index 14 is illegal because
    // it would leave a 16-prob suffix. The hard limit wins over trough quality.
    let mut probs = vec![1.0f32; 30];
    probs[14] = 0.4;
    let chunks = chunks_from_probs(&probs, &fast_opts()).unwrap();
    assert_eq!(chunks, vec![AudioChunk::new(0, 15), AudioChunk::new(15, 30)]);
}

#[test]
fn test_chunker_drops_silence_between_chunks() {
    // Two well-separated 4s segments, each becomes its own chunk; the 8s
    // silence gap between them is gone from the output.
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
    assert_eq!(chunks, vec![AudioChunk::with_decode(10, 20, 5, 25)]);
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
    assert_eq!(chunks[0].end_sample, 8);
    assert_eq!(chunks[0].decode_start_sample, 0);
    assert_eq!(chunks[0].decode_end_sample, max_sample);
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
        cluster_target_duration: None,
        strict_limit_duration: 1.0,
        min_speech_probs: 1,
        min_silence_probs: 2,
        merge_gap_probs: 0,
        trough_search_probs: None,
        trough_threshold: None,
        pad_samples: 0,
        align_to: 640,
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
    // `chunks_from_probs` sees only `probs`, not the raw `waveform`. When the
    // waveform's actual length isn't a multiple of `samples_per_prob`, the
    // last prob's window straddles the waveform end and the emitted chunk's
    // `decode_end_sample` reflects the full window — overshooting the waveform by
    // up to `samples_per_prob - 1` samples. This is the contract documented
    // at `AudioChunk::decode_end_sample`: callers owning the waveform must clamp at
    // slice time. Pinned here so the documented overshoot is exercised by a
    // test, not just a comment.
    let probs = vec![1.0_f32; 4]; // 4 windows × 512 = 2048 samples of coverage
    let waveform_len = 1800; // real waveform ended mid-window
    let opts = ChunkerOpts {
        sample_rate: 16_000,
        samples_per_prob: 512,
        threshold: 0.5,
        min_duration: 0.0,
        max_duration: 1.0,
        cluster_target_duration: None,
        strict_limit_duration: 1.0,
        min_speech_probs: 1,
        min_silence_probs: 1,
        merge_gap_probs: 1,
        trough_search_probs: None,
        trough_threshold: None,
        pad_samples: 0,
        align_to: 640, // GigaAM: hop_length * subsampling_factor
    };
    let chunks = chunks_from_probs(&probs, &opts).unwrap();
    assert_eq!(chunks, vec![AudioChunk::new(0, 2048)]);
    assert!(chunks[0].decode_end_sample > waveform_len);
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
//    is bounded by `strict_limit_probs * samples_per_prob`. Trough search can
//    move split points only inside legal ranges that preserve that hard cap.
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
            cluster_target_duration: None,
            strict_limit_duration: strict_dur,
            min_speech_probs: min_speech,
            min_silence_probs: min_silence,
            merge_gap_probs: merge_gap,
            trough_search_probs: None,
        trough_threshold: None,
            pad_samples,
            align_to,
        };
        let chunks = chunks_from_probs(&probs, &opts).unwrap();
        let max_sample = probs.len() * samples_per_prob;

        // Core output ranges are sorted + non-overlapping (touching allowed).
        for w in chunks.windows(2) {
            prop_assert!(
                w[0].end_sample <= w[1].start_sample,
                "overlapping chunk cores: {:?} and {:?}", w[0], w[1]
            );
        }
        // Each core/decode range is non-empty and inside the input extent.
        for c in &chunks {
            prop_assert!(c.start_sample < c.end_sample, "empty chunk: {c:?}");
            prop_assert!(c.end_sample <= max_sample,
                "chunk core {c:?} exceeds max_sample {max_sample}");
            prop_assert!(c.decode_start_sample <= c.start_sample, "decode starts after core: {c:?}");
            prop_assert!(c.end_sample <= c.decode_end_sample, "decode ends before core: {c:?}");
            prop_assert!(c.decode_start_sample < c.decode_end_sample, "empty decode window: {c:?}");
            prop_assert!(c.decode_end_sample <= max_sample,
                "chunk decode {c:?} exceeds max_sample {max_sample}");
        }
        // Decode-window alignment: start always aligned, end aligned OR
        // clamped to max_sample (the only legitimate non-aligned end).
        for c in &chunks {
            prop_assert_eq!(c.decode_start_sample % align_to, 0,
                "decode start {} not aligned to {}", c.decode_start_sample, align_to);
            let end_aligned = c.decode_end_sample % align_to == 0;
            let end_at_max = c.decode_end_sample == max_sample;
            prop_assert!(end_aligned || end_at_max,
                "decode end {} not aligned to {} and not at max_sample {}",
                c.decode_end_sample, align_to, max_sample);
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
            cluster_target_duration: None,
            strict_limit_duration: strict_dur,
            min_speech_probs: min_speech,
            min_silence_probs: min_silence,
            merge_gap_probs: merge_gap,
            trough_search_probs: None,
        trough_threshold: None,
            pad_samples: 0,
            align_to: 1,
        };
        let chunks = chunks_from_probs(&probs, &opts).unwrap();
        let probs_per_sec = sample_rate as f32 / samples_per_prob as f32;
        let strict_limit_probs = (strict_dur * probs_per_sec).ceil() as usize;
        let bound_samples = crate::vad::strict_chunk_sample_bound(
            strict_limit_probs,
            samples_per_prob,
            opts.pad_samples,
            opts.align_to,
        );
        for c in &chunks {
            let len = c.end_sample - c.start_sample;
            prop_assert!(
                len <= bound_samples,
                "chunk {c:?} length {len} exceeds bound {bound_samples} \
                 (strict_probs={strict_limit_probs}, spp={samples_per_prob})"
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
            cluster_target_duration: None,
            strict_limit_duration: 0.5,
            min_speech_probs: 1,
            min_silence_probs: 1,
            merge_gap_probs: 0,
            trough_search_probs: None,
        trough_threshold: None,
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
    assert_eq!(opts.cluster_target_duration, default.cluster_target_duration);
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
        cluster_target_duration: None,
        strict_limit_duration: 10.0,
        min_speech_probs: 1,
        min_silence_probs: 100, // suppress smoothing-driven termination
        merge_gap_probs: 0,
        trough_search_probs: Some(10),
        trough_threshold: None,
        pad_samples: 0,
        align_to: 1,
    };
    let chunks = chunks_from_probs(&probs, &opts).unwrap();
    assert!(!chunks.is_empty());
    let first_len = chunks[0].end_sample - chunks[0].start_sample;
    assert!(first_len >= 4, "first chunk too small ({first_len} samples): {chunks:?}");
}

#[test]
fn test_chunker_split_long_runs_respects_strict_limit_with_wide_search() {
    // Wide trough search must not pull a split so far toward a low-prob frame
    // that any produced cluster exceeds the hard model-context limit.
    let mut probs = vec![1.0f32; 25];
    probs[18] = 0.0;
    let opts = ChunkerOpts {
        sample_rate: 1,
        samples_per_prob: 1,
        threshold: 0.5,
        min_duration: 1.0,
        max_duration: 10.0,
        cluster_target_duration: None,
        strict_limit_duration: 10.0,
        min_speech_probs: 1,
        min_silence_probs: 100,
        merge_gap_probs: 0,
        trough_search_probs: Some(20),
        trough_threshold: None,
        pad_samples: 0,
        align_to: 1,
    };
    let chunks = chunks_from_probs(&probs, &opts).unwrap();
    assert!(!chunks.is_empty());
    for c in &chunks {
        assert!(c.core_len() <= 10, "chunk exceeds strict limit: {c:?}");
    }
    assert_eq!(chunks.first().unwrap().start_sample, 0);
    assert_eq!(chunks.last().unwrap().end_sample, 25);
}

#[test]
fn test_chunker_decoupled_trough_search_radius() {
    // With min_silence=2 the default search around target 13 would miss the
    // legal trough at index 10. A wider trough_search_probs can choose it while
    // still keeping the remaining pieces within strict=15.
    let mut probs = vec![1.0f32; 40];
    probs[10] = 0.4;
    let opts = ChunkerOpts {
        sample_rate: 1,
        samples_per_prob: 1,
        threshold: 0.5,
        min_duration: 1.0,
        max_duration: 10.0,
        cluster_target_duration: None,
        strict_limit_duration: 15.0,
        min_speech_probs: 1,
        min_silence_probs: 2,
        merge_gap_probs: 0,
        trough_search_probs: Some(8),
        trough_threshold: None,
        pad_samples: 0,
        align_to: 1,
    };
    let chunks = chunks_from_probs(&probs, &opts).unwrap();
    assert_eq!(chunks, vec![AudioChunk::new(0, 10), AudioChunk::new(10, 25), AudioChunk::new(25, 40)]);
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
