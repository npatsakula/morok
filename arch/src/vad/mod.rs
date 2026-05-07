//! VAD-aware chunker for long-form ASR.
//!
//! Operates on `&[f32]` per-frame speech probabilities — the output of any
//! frame-level VAD — and packs them into bounded-length [`AudioChunk`]s
//! suitable for feeding to an encoder one chunk at a time. Speech-bearing
//! regions of the waveform are preserved; pure-silence regions between
//! chunks are dropped.
//!
//! The chunker is purely algorithmic: no Tensor or model dependency, no
//! coupling to a specific VAD. The output is sample-index ranges that any
//! downstream decoder can consume.
//!
//! # Algorithm
//!
//! ```text
//! 1. threshold + smoothing  → speech runs (prob-grid indices)
//! 2. split runs ≥ strict_limit at internal prob troughs
//! 3. greedy-pack runs into chunks of ~[min_duration, max_duration]
//!    (closing at inter-segment silence rather than mid-speech)
//! 4. convert prob indices → samples, apply pad, align to align_to
//! ```
//!
//! All knobs live in [`ChunkerOpts`]; nothing inside the algorithm hardcodes
//! sample rates, prob granularity, or alignment.

pub(crate) mod segment;

#[cfg(feature = "serde")]
use serde::Deserialize;
use snafu::Snafu;

use segment::threshold_segments;

// ─── Config ───────────────────────────────────────────────────────────────

/// Configuration for [`chunks_from_probs`].
///
/// All `*_duration` fields are wall-clock seconds; the chunker converts to
/// prob-grid indices via `(sample_rate, samples_per_prob)`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct ChunkerOpts {
    /// Sample rate of the source waveform in Hz.
    pub sample_rate: u32,
    /// Number of input samples covered by one entry of the `probs` array.
    /// Match the stride of the upstream frame-level VAD. Required so the
    /// chunker stays VAD-agnostic.
    pub samples_per_prob: usize,
    /// Speech threshold: prob entries `>= threshold` count as speech.
    pub threshold: f32,
    /// Soft minimum chunk duration. The chunker won't voluntarily close a
    /// chunk shorter than this.
    pub min_duration: f32,
    /// Soft maximum chunk duration. Past `min_duration`, the chunk closes
    /// at the next inter-segment silence (or, for a single long run, at a
    /// local prob trough) instead of extending past max.
    pub max_duration: f32,
    /// Hard ceiling. A single VAD segment longer than this is split
    /// internally at prob-trough argmins so no output chunk exceeds it.
    /// Also caps chunk length when an under-min chunk would otherwise
    /// be extended past this.
    pub strict_limit_duration: f32,
    /// Pre-segmentation smoothing: a speech run must contain at least this
    /// many above-threshold probs to be retained.
    pub min_speech_probs: usize,
    /// Pre-segmentation smoothing: a silence gap must span at least this
    /// many below-threshold probs to terminate a speech run.
    pub min_silence_probs: usize,
    /// Two speech runs separated by ≤ this many silence probs are merged
    /// before chunking.
    pub merge_gap_probs: usize,
    /// Window radius (in prob-grid units) for the trough-argmin search when
    /// splitting overlong runs. `None` (default) reuses `min_silence_probs`,
    /// which is fine when smoothing tightness and trough-search width happen
    /// to want the same scale; set explicitly to decouple them.
    pub trough_search_probs: Option<usize>,
    /// Symmetric pad in samples added to each chunk's start/end (clamped at
    /// 0 and the implicit waveform end). Gives the encoder context at chunk
    /// boundaries.
    pub pad_samples: usize,
    /// Snap chunk boundaries to integer multiples of this many samples.
    /// `1` = sample-precise. Set to the encoder's effective frame stride
    /// (e.g. `mel_hop * subsample_factor`) so chunks land on encoder-frame
    /// boundaries. Pathological values (e.g. > min_duration) are the
    /// caller's responsibility — boundaries can shift by up to
    /// `align_to - 1` samples.
    pub align_to: usize,
}

impl Default for ChunkerOpts {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            samples_per_prob: 512,
            threshold: 0.5,
            min_duration: 15.0,
            max_duration: 22.0,
            strict_limit_duration: 30.0,
            min_speech_probs: 8,
            min_silence_probs: 4,
            merge_gap_probs: 8,
            trough_search_probs: None,
            pad_samples: 0,
            align_to: 1,
        }
    }
}

// ─── Output ───────────────────────────────────────────────────────────────

/// A speech-bearing region of the source waveform.
///
/// Sample indices reference the *original* waveform passed to the VAD.
/// `start_sample` is `chunk_index * samples_per_prob` (after pad + align);
/// callers can derive `start_sec = start_sample as f32 / sample_rate as f32`
/// to offset per-chunk transcripts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AudioChunk {
    /// Inclusive start sample index in the original waveform.
    pub start_sample: usize,
    /// Exclusive end sample index. May exceed the waveform length if the
    /// last prob entry covered samples past the waveform end; callers
    /// should clamp at slice time.
    pub end_sample: usize,
}

// ─── Errors ───────────────────────────────────────────────────────────────

#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum Error {
    #[snafu(display("samples_per_prob must be > 0"))]
    ZeroSamplesPerProb,
    #[snafu(display("align_to must be > 0"))]
    ZeroAlignTo,
    #[snafu(display("min_duration ({min}) must be ≤ max_duration ({max})"))]
    MinExceedsMax { min: f32, max: f32 },
    #[snafu(display("max_duration ({max}) must be ≤ strict_limit_duration ({strict})"))]
    MaxExceedsStrict { max: f32, strict: f32 },
}

pub type Result<T> = std::result::Result<T, Error>;

// ─── Public entry point ───────────────────────────────────────────────────

/// Pack VAD speech probabilities into bounded-length chunks.
///
/// Output chunks cover only speech-bearing portions of the waveform; silence
/// between chunks is dropped. Boundaries are padded by `opts.pad_samples` and
/// snapped to `opts.align_to` multiples (start floored, end ceil'd, so
/// coverage is preserved). Adjacent chunks that overlap after padding are
/// merged.
pub fn chunks_from_probs(probs: &[f32], opts: &ChunkerOpts) -> Result<Vec<AudioChunk>> {
    validate(opts)?;
    if probs.is_empty() {
        return Ok(Vec::new());
    }

    let probs_per_sec = opts.sample_rate as f32 / opts.samples_per_prob as f32;
    let strict_limit_probs = (opts.strict_limit_duration * probs_per_sec).ceil() as usize;
    let min_probs = (opts.min_duration * probs_per_sec).ceil() as usize;
    let max_probs = (opts.max_duration * probs_per_sec).ceil() as usize;

    let trough_radius = opts.trough_search_probs.unwrap_or(opts.min_silence_probs);
    let segments = threshold_segments(probs, opts);
    let segments = split_long_runs(segments, probs, trough_radius, strict_limit_probs);
    let chunks = pack_segments(&segments, min_probs, max_probs, strict_limit_probs);

    Ok(post_process(&chunks, probs.len(), opts))
}

// ─── Internals ────────────────────────────────────────────────────────────

fn validate(opts: &ChunkerOpts) -> Result<()> {
    if opts.samples_per_prob == 0 {
        return ZeroSamplesPerProbSnafu.fail();
    }
    if opts.align_to == 0 {
        return ZeroAlignToSnafu.fail();
    }
    if opts.min_duration > opts.max_duration {
        return MinExceedsMaxSnafu { min: opts.min_duration, max: opts.max_duration }.fail();
    }
    if opts.max_duration > opts.strict_limit_duration {
        return MaxExceedsStrictSnafu { max: opts.max_duration, strict: opts.strict_limit_duration }.fail();
    }
    Ok(())
}

/// Break any speech segment whose length exceeds `strict_limit_probs` into
/// `ceil(len / strict_limit)` near-equal pieces, choosing each split point
/// as the prob argmin within ±`search_radius` of the geometric target. Lands
/// on natural pauses inside long unbroken runs instead of hard-cutting at
/// fixed time intervals.
///
/// Each emitted piece is at least `len / (2 * n)` long. Without that floor
/// a wide `search_radius` can let the argmin land arbitrarily close to a
/// split's neighbours and produce 1-prob shards that downstream code has to
/// special-case. With the floor the worst-case shrinkage is half the
/// average piece length.
fn split_long_runs(
    segments: Vec<(usize, usize)>,
    probs: &[f32],
    search_radius: usize,
    strict_limit_probs: usize,
) -> Vec<(usize, usize)> {
    if strict_limit_probs == 0 {
        return segments;
    }
    let mut out = Vec::with_capacity(segments.len());
    for (start, end) in segments {
        let len = end - start;
        if len <= strict_limit_probs {
            out.push((start, end));
            continue;
        }
        let n = len.div_ceil(strict_limit_probs);
        let min_piece = (len / (2 * n)).max(1);
        let mut cur = start;
        for k in 1..n {
            let target = start + (len * k) / n;
            let pieces_left = n - k;
            // Constrain the argmin window so this split is at least
            // min_piece away from cur and from `end - pieces_left * min_piece`
            // (i.e. each remaining piece can still hit min_piece).
            let lo = target.saturating_sub(search_radius).max(cur + min_piece);
            let hi_floor = end.saturating_sub(pieces_left * min_piece);
            let hi = (target + search_radius).min(hi_floor.saturating_sub(1));
            let split = if hi >= lo {
                lo + argmin(&probs[lo..=hi])
            } else {
                // Constraints incompatible (radius wider than the available
                // slack). Fall back to the geometric target, clamped so the
                // remaining pieces are still non-empty.
                target.clamp(cur + min_piece, hi_floor.saturating_sub(1).max(cur + min_piece))
            };
            if split > cur && split < end {
                out.push((cur, split));
                cur = split;
            }
        }
        if cur < end {
            out.push((cur, end));
        }
    }
    out
}

fn argmin(slice: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = slice[0];
    for (i, &v) in slice.iter().enumerate().skip(1) {
        if v < best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

/// Greedy-concat speech segments into bounded-length chunks. Closes a chunk
/// when the next segment would push it past `max_probs` AND either the
/// current chunk has reached `min_probs` *or* extending would exceed
/// `strict_limit_probs` (the hard ceiling).
fn pack_segments(
    segments: &[(usize, usize)],
    min_probs: usize,
    max_probs: usize,
    strict_limit_probs: usize,
) -> Vec<(usize, usize)> {
    let mut chunks = Vec::new();
    let mut cur: Option<(usize, usize)> = None;
    for &(s, e) in segments {
        match cur {
            None => cur = Some((s, e)),
            Some((cs, ce)) => {
                let prospective = e - cs;
                let cur_len = ce - cs;
                if prospective > max_probs && (cur_len >= min_probs || prospective > strict_limit_probs) {
                    chunks.push((cs, ce));
                    cur = Some((s, e));
                } else {
                    cur = Some((cs, e));
                }
            }
        }
    }
    if let Some(c) = cur {
        chunks.push(c);
    }
    chunks
}

/// Convert prob-index ranges to sample ranges, apply padding + alignment,
/// and merge any overlaps introduced by padding.
fn post_process(chunks: &[(usize, usize)], probs_len: usize, opts: &ChunkerOpts) -> Vec<AudioChunk> {
    let max_sample = probs_len * opts.samples_per_prob;
    let pad = opts.pad_samples;
    let align = opts.align_to;

    let mut out: Vec<AudioChunk> = Vec::with_capacity(chunks.len());
    for &(s, e) in chunks {
        let raw_start = s * opts.samples_per_prob;
        let raw_end = e * opts.samples_per_prob;
        let padded_start = raw_start.saturating_sub(pad);
        let padded_end = (raw_end + pad).min(max_sample);
        // Floor start, ceil end (preserves coverage).
        let aligned_start = (padded_start / align) * align;
        let mut aligned_end = padded_end.div_ceil(align) * align;
        if aligned_end > max_sample {
            aligned_end = max_sample;
        }
        if aligned_end <= aligned_start {
            continue;
        }
        // Merge only on *strict* overlap (start < last.end). Two chunks
        // that just touch at a shared sample (start == last.end) come from
        // pack_segments deliberately splitting at a silence — preserve
        // that decision. Padding only triggers a merge when chunks
        // actually grow into one another.
        if let Some(last) = out.last_mut()
            && aligned_start < last.end_sample
        {
            last.end_sample = last.end_sample.max(aligned_end);
            continue;
        }
        out.push(AudioChunk { start_sample: aligned_start, end_sample: aligned_end });
    }
    out
}
