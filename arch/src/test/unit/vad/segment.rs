use crate::vad::segment::{hysteresis_segments, merge_close};

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
fn test_hysteresis_basic_two_runs() {
    // Two distinct speech runs separated by enough silence to close cleanly.
    let probs = probs_from("00111111110000111111");
    let runs = hysteresis_segments(&probs, 0.5, 0.5, 4, 3);
    assert_eq!(runs, vec![(2, 10), (14, 20)]);
}

#[test]
fn test_hysteresis_min_speech_filters_short_runs() {
    // 2-frame blip is filtered by min_speech=4.
    let probs = probs_from("0011000011111111");
    let runs = hysteresis_segments(&probs, 0.5, 0.5, 4, 3);
    assert_eq!(runs, vec![(8, 16)]);
}

#[test]
fn test_hysteresis_min_silence_keeps_short_gaps() {
    // 2-frame silence gap is below min_silence=3, so the two speech runs stay fused.
    let probs = probs_from("11111100111111");
    let runs = hysteresis_segments(&probs, 0.5, 0.5, 4, 3);
    assert_eq!(runs, vec![(0, 14)]);
}

#[test]
fn test_hysteresis_sustain_band_keeps_run_open() {
    // onset=0.6, offset=0.3. A mid-run prob of 0.4 is in the sustain band:
    // it does NOT increment the silence streak even though it's < onset.
    let probs = vec![0.7, 0.7, 0.7, 0.4, 0.7, 0.7];
    let runs = hysteresis_segments(&probs, 0.6, 0.3, 2, 2);
    assert_eq!(runs, vec![(0, 6)]);
}

#[test]
fn test_hysteresis_offset_closes_run_promptly() {
    // Run opens at p>=onset=0.6. Two consecutive p<offset=0.3 then close.
    let probs = vec![0.7, 0.7, 0.7, 0.7, 0.1, 0.1, 0.7, 0.7];
    let runs = hysteresis_segments(&probs, 0.6, 0.3, 2, 2);
    // First run [0,4); after two `<offset` frames, run reopens at index 6.
    // Tail run length 2 == min_speech, retained.
    assert_eq!(runs, vec![(0, 4), (6, 8)]);
}

#[test]
fn test_hysteresis_empty_input() {
    assert!(hysteresis_segments(&[], 0.5, 0.5, 4, 3).is_empty());
}

#[test]
fn test_hysteresis_all_silence() {
    let probs = vec![0.0f32; 50];
    assert!(hysteresis_segments(&probs, 0.5, 0.5, 4, 3).is_empty());
}

#[test]
fn test_hysteresis_all_speech_flushes_tail() {
    let probs = vec![1.0f32; 50];
    let runs = hysteresis_segments(&probs, 0.5, 0.5, 4, 3);
    assert_eq!(runs, vec![(0, 50)]);
}

#[test]
fn test_merge_close_does_not_merge_when_far() {
    // Gap = 10 probs; merge_gap = 8 keeps them split.
    let runs = vec![(0, 8), (18, 26)];
    assert_eq!(merge_close(runs, 8), vec![(0, 8), (18, 26)]);
}

#[test]
fn test_merge_close_merges_when_close() {
    let runs = vec![(0, 8), (18, 26)];
    assert_eq!(merge_close(runs, 10), vec![(0, 26)]);
}

#[test]
fn test_merge_close_zero_gap_disables_merging() {
    // merge_gap=0 means only directly-adjacent runs (gap == 0) merge.
    let runs = vec![(0, 8), (8, 16), (20, 24)];
    assert_eq!(merge_close(runs, 0), vec![(0, 16), (20, 24)]);
}
