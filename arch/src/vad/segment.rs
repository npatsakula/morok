//! Threshold + smoothing pass that turns per-frame speech probabilities into
//! `[start, end)` prob-index ranges of speech.
//!
//! Output is in *prob-grid index* units so downstream code (chunker
//! post-process) can decide on sample/time conversion. Smoothing constants
//! (min run lengths, merge gaps) are read from [`super::ChunkerOpts`].

use super::ChunkerOpts;

/// Returns `[start, end)` ranges (prob-grid indices) of speech runs in
/// `probs`, with smoothing per `opts`:
///
/// - A speech run begins at the first prob ≥ `opts.threshold` after a
///   silence run.
/// - A speech run terminates at the first index where `opts.min_silence_probs`
///   consecutive sub-threshold probs have been seen. The terminator index
///   itself is *exclusive*.
/// - Runs containing fewer than `opts.min_speech_probs` total above-threshold
///   indices are dropped.
/// - Adjacent speech runs separated by ≤ `opts.merge_gap_probs` silence
///   probs are merged.
pub(crate) fn threshold_segments(probs: &[f32], opts: &ChunkerOpts) -> Vec<(usize, usize)> {
    let min_speech = opts.min_speech_probs;
    let min_silence = opts.min_silence_probs;
    let merge_gap = opts.merge_gap_probs;
    let threshold = opts.threshold;

    let mut raw: Vec<(usize, usize)> = Vec::new();
    let mut speech_start: Option<usize> = None;
    let mut silence_count = 0usize;

    for (i, &p) in probs.iter().enumerate() {
        if p >= threshold {
            if speech_start.is_none() {
                speech_start = Some(i);
            }
            silence_count = 0;
        } else if let Some(start) = speech_start {
            silence_count += 1;
            if min_silence == 0 || silence_count >= min_silence {
                // First non-speech index in the trailing silence run; the run
                // started at `i + 1 - silence_count` and the speech segment
                // ends just before it.
                let end = i + 1 - silence_count;
                if end > start && end - start >= min_speech {
                    raw.push((start, end));
                }
                speech_start = None;
                silence_count = 0;
            }
        }
    }

    if let Some(start) = speech_start {
        let end = probs.len();
        if end - start >= min_speech {
            raw.push((start, end));
        }
    }

    // Merge consecutive runs separated by ≤ merge_gap silence probs.
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for seg in raw {
        if let Some(last) = merged.last_mut()
            && seg.0 - last.1 <= merge_gap
        {
            last.1 = seg.1;
            continue;
        }
        merged.push(seg);
    }

    merged
}
