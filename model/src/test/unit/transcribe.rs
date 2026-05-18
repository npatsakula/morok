//! Unit tests for the per-chunk decoder helpers in
//! `model/src/gigaam/transcribe.rs`.
//!
//! `HeadDecoder::decode_chunk` itself requires a real CtcHeadJit /
//! RnntStepBackend fixture (i.e. a loaded model + prepared plans) and is
//! covered end-to-end by the `gigaam_infer` example transcripts in CI. The
//! one piece of new pure-logic — `ctc_frames_to_words` — is testable in
//! isolation and lives here.

use svod_arch::rnnt::Word;

use crate::gigaam::TranscribeOpts;
use crate::gigaam::transcribe::{crop_words_to_core, ctc_frames_to_words, words_to_text};

#[test]
fn ctc_frames_to_words_empty() {
    let words = ctc_frames_to_words("", &[], 0.04);
    assert!(words.is_empty());
}

#[test]
fn ctc_frames_to_words_single_word() {
    // "hello" emitted at frames 10..15.
    let words = ctc_frames_to_words("hello", &[10, 11, 12, 13, 14], 0.04);
    assert_eq!(words, vec![Word { text: "hello".to_string(), start: 10.0 * 0.04, end: 15.0 * 0.04 }]);
}

#[test]
fn ctc_frames_to_words_two_words_with_space() {
    // "hi mom" — space at frame 12 should commit "hi" and start "mom".
    let words = ctc_frames_to_words("hi mom", &[5, 6, 12, 20, 21, 22], 0.05);
    assert_eq!(
        words,
        vec![
            Word { text: "hi".to_string(), start: 5.0 * 0.05, end: 7.0 * 0.05 },
            Word { text: "mom".to_string(), start: 20.0 * 0.05, end: 23.0 * 0.05 },
        ]
    );
}

#[test]
fn ctc_frames_to_words_leading_and_trailing_spaces() {
    // " hi " — leading space commits nothing; trailing flushes "hi".
    let words = ctc_frames_to_words(" hi ", &[3, 5, 6, 9], 0.04);
    assert_eq!(words, vec![Word { text: "hi".to_string(), start: 5.0 * 0.04, end: 7.0 * 0.04 }],);
}

#[test]
fn ctc_frames_to_words_consecutive_spaces() {
    // "a  b" — two spaces between, second commits an empty pending and is
    // a no-op.
    let words = ctc_frames_to_words("a  b", &[2, 3, 4, 7], 0.04);
    assert_eq!(
        words,
        vec![
            Word { text: "a".to_string(), start: 2.0 * 0.04, end: 3.0 * 0.04 },
            Word { text: "b".to_string(), start: 7.0 * 0.04, end: 8.0 * 0.04 },
        ]
    );
}

#[test]
fn ctc_frames_to_words_frame_shift_scales_linearly() {
    // Same emissions, half the frame_shift → half the timings.
    let a = ctc_frames_to_words("ok", &[10, 11], 0.04);
    let b = ctc_frames_to_words("ok", &[10, 11], 0.02);
    assert_eq!(a.len(), 1);
    assert_eq!(b.len(), 1);
    assert!((a[0].start - 2.0 * b[0].start).abs() < 1e-6);
    assert!((a[0].end - 2.0 * b[0].end).abs() < 1e-6);
}

#[test]
fn crop_words_to_core_keeps_midpoints_inside_core() {
    let words = vec![
        Word { text: "left".to_string(), start: 0.0, end: 0.25 },
        Word { text: "keep".to_string(), start: 0.5, end: 0.75 },
        Word { text: "right".to_string(), start: 1.25, end: 1.5 },
    ];
    let cropped = crop_words_to_core(words, 0.0, 0.375, 1.0);
    assert_eq!(cropped, vec![Word { text: "keep".to_string(), start: 0.125, end: 0.375 }]);
}

#[test]
fn crop_words_to_core_clamps_boundary_word_times() {
    let words = vec![Word { text: "edge".to_string(), start: 0.25, end: 0.75 }];
    let cropped = crop_words_to_core(words, 0.0, 0.375, 0.625);
    assert_eq!(cropped, vec![Word { text: "edge".to_string(), start: 0.0, end: 0.25 }]);
}

#[test]
fn words_to_text_joins_non_empty_words() {
    let words = vec![
        Word { text: "hello".to_string(), start: 0.0, end: 0.1 },
        Word { text: "".to_string(), start: 0.1, end: 0.2 },
        Word { text: "world".to_string(), start: 0.2, end: 0.3 },
    ];
    assert_eq!(words_to_text(&words), "hello world");
}

// ─── TranscribeOpts builder ──────────────────────────────────────────────
//
// We don't test `from_env()` directly — env-var manipulation isn't safe
// across parallel tests and the equivalence with `builder().build()` is a
// one-liner in the impl. The builder-overrides test below confirms every
// field flows through to the struct correctly.

#[test]
fn transcribe_opts_builder_overrides_all_fields() {
    let opts = TranscribeOpts::builder().word_timestamps(true).beam_decode(true).max_scores_mib(512).build();
    assert!(opts.word_timestamps);
    assert!(opts.beam_decode);
    assert_eq!(opts.max_scores_mib, 512);
}
