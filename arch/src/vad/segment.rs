//! Hysteresis binarisation + merge-close pass that turns per-frame speech
//! probabilities into `[start, end)` prob-index ranges of speech.
//!
//! Hysteresis (two thresholds, `onset > offset`) suppresses the
//! single-threshold chatter that earlier needed `min_speech` / `min_silence`
//! smoothing to paper over. The smoothing knobs still exist as run-length
//! floors: a run only commits when it lasts at least `min_speech_probs`, and
//! only closes after `min_silence_probs` consecutive below-`offset` probs.

/// Returns `[start, end)` ranges (prob-grid indices) of speech runs in
/// `probs`, using hysteresis binarisation:
///
/// - A speech run opens at the first prob `≥ onset`.
/// - While in speech, probs `≥ offset` keep the run open (the band
///   `[offset, onset)` is "in-speech sustain"). Below `offset` starts a
///   silence streak.
/// - A speech run terminates at the first index where
///   `min_silence_probs` consecutive probs `< offset` have been seen.
///   The terminator index is *exclusive*.
/// - Runs shorter than `min_speech_probs` are dropped.
///
/// `onset` and `offset` are assumed to satisfy `offset ≤ onset`. Pass equal
/// values to recover plain single-threshold behaviour.
pub(crate) fn hysteresis_segments(
    probs: &[f32],
    onset: f32,
    offset: f32,
    min_speech_probs: usize,
    min_silence_probs: usize,
) -> Vec<(usize, usize)> {
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut speech_start: Option<usize> = None;
    let mut silence_count = 0usize;

    for (i, &p) in probs.iter().enumerate() {
        match speech_start {
            None => {
                if p >= onset {
                    speech_start = Some(i);
                    silence_count = 0;
                }
            }
            Some(start) => {
                if p >= offset {
                    silence_count = 0;
                } else {
                    silence_count += 1;
                    if min_silence_probs == 0 || silence_count >= min_silence_probs {
                        let end = i + 1 - silence_count;
                        if end > start && end - start >= min_speech_probs {
                            runs.push((start, end));
                        }
                        speech_start = None;
                        silence_count = 0;
                    }
                }
            }
        }
    }

    if let Some(start) = speech_start {
        let end = probs.len();
        if end > start && end - start >= min_speech_probs {
            runs.push((start, end));
        }
    }

    runs
}

/// Fold consecutive speech runs whose gap is `≤ merge_gap_probs` into one.
/// Pass `0` to disable merging.
pub(crate) fn merge_close(runs: Vec<(usize, usize)>, merge_gap_probs: usize) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(runs.len());
    for seg in runs {
        if let Some(last) = out.last_mut()
            && seg.0.saturating_sub(last.1) <= merge_gap_probs
        {
            last.1 = seg.1;
            continue;
        }
        out.push(seg);
    }
    out
}
