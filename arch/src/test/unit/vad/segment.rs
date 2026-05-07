use crate::vad::ChunkerOpts;
use crate::vad::segment::threshold_segments;

fn opts(threshold: f32, min_speech: usize, min_silence: usize, merge_gap: usize) -> ChunkerOpts {
    ChunkerOpts {
        threshold,
        min_speech_probs: min_speech,
        min_silence_probs: min_silence,
        merge_gap_probs: merge_gap,
        ..ChunkerOpts::default()
    }
}

/// Builds a `[0|1]` prob array from a bit-string spec like `"00111111110000111111"`.
fn probs_from(spec: &str) -> Vec<f32> {
    spec.chars()
        .map(|c| match c {
            '0' => 0.0,
            '1' => 1.0,
            other => panic!("unsupported bit char: {other}"),
        })
        .collect()
}

#[test]
fn test_threshold_segments_basic() {
    // Two distinct speech runs, separated by enough silence to avoid
    // both the smoothing-merge and the gap-merge.
    let probs = probs_from("00111111110000111111");
    let segments = threshold_segments(&probs, &opts(0.5, 4, 3, 1));
    assert_eq!(segments, vec![(2, 10), (14, 20)]);
}

#[test]
fn test_threshold_segments_min_speech_filters_short_runs() {
    // 2-frame speech blip is filtered by min_speech_probs=4; only the
    // longer trailing run survives.
    let probs = probs_from("0011000011111111");
    let segments = threshold_segments(&probs, &opts(0.5, 4, 3, 0));
    assert_eq!(segments, vec![(8, 16)]);
}

#[test]
fn test_threshold_segments_min_silence_keeps_short_gaps() {
    // 2-frame silence gap is below min_silence_probs=3, so the two
    // speech runs are kept as a single continuous run.
    let probs = probs_from("11111100111111");
    let segments = threshold_segments(&probs, &opts(0.5, 4, 3, 0));
    assert_eq!(segments, vec![(0, 14)]);
}

#[test]
fn test_threshold_segments_merge_gap_does_not_merge_when_far() {
    // Two runs separated by 10 silence probs; merge_gap=8 keeps them split.
    let probs = probs_from("11111111000000000011111111");
    let segments = threshold_segments(&probs, &opts(0.5, 4, 3, 8));
    assert_eq!(segments, vec![(0, 8), (18, 26)]);
}

#[test]
fn test_threshold_segments_merge_gap_merges_when_close() {
    // Same shape, but merge_gap=10 fuses them into one.
    let probs = probs_from("11111111000000000011111111");
    let segments = threshold_segments(&probs, &opts(0.5, 4, 3, 10));
    assert_eq!(segments, vec![(0, 26)]);
}

#[test]
fn test_threshold_segments_empty() {
    let segments = threshold_segments(&[], &opts(0.5, 4, 3, 0));
    assert!(segments.is_empty());
}

#[test]
fn test_threshold_segments_all_silence() {
    let probs = vec![0.0f32; 50];
    let segments = threshold_segments(&probs, &opts(0.5, 4, 3, 0));
    assert!(segments.is_empty());
}

#[test]
fn test_threshold_segments_all_speech() {
    let probs = vec![1.0f32; 50];
    let segments = threshold_segments(&probs, &opts(0.5, 4, 3, 0));
    assert_eq!(segments, vec![(0, 50)]);
}
