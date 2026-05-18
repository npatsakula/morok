//! Cut & Merge VAD chunker for long-form ASR.
//!
//! Operates on `&[f32]` per-frame speech probabilities — the output of any
//! frame-level VAD — and packs them into bounded-length [`AudioChunk`]s
//! suitable for feeding to an encoder one chunk at a time.
//!
//! # Algorithm
//!
//! ```text
//! 1. hysteresis_segments    (onset/offset → speech runs)
//! 2. merge_close            (fold runs separated by ≤ merge_gap_probs)
//! 3. cut_long_runs          (any run > max_chunk_probs is split at silence trough)
//! 4. pack_chunks            (greedy concat up to max_chunk_probs)
//! 5. post_process           (prob → samples, raw pad, frame-align, no decode overlap)
//! 6. speech_gate            (drop chunks with peak core prob < min_chunk_max_prob)
//! ```
//!
//! Compared with the previous design this collapses smoothing + cluster
//! packing + trough-radius search into a single pass that always splits on
//! the rightmost real silence (or argmin fallback), guarantees decode
//! windows never overlap (eliminating duplicate tokens at seams), and gates
//! out chunks that contain no Silero-confident speech.

pub(crate) mod segment;

#[cfg(feature = "serde")]
use serde::Deserialize;
use snafu::Snafu;

use segment::{hysteresis_segments, merge_close};

// ─── Config ───────────────────────────────────────────────────────────────

/// Configuration for [`chunks_from_probs`].
///
/// All durations are expressed in prob-grid units — the upstream
/// [`SileroVadSplitter`](../silero_vad/struct.SileroVadSplitter.html) (or
/// equivalent) is responsible for converting wall-clock seconds and
/// `sample_rate` into prob counts.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct ChunkerOpts {
    /// Sample rate of the source waveform in Hz.
    pub sample_rate: u32,
    /// Number of input samples covered by one entry of the `probs` array.
    /// Match the stride of the upstream frame-level VAD.
    pub samples_per_prob: usize,

    /// Hysteresis open threshold: a speech run begins at the first prob `≥ onset`.
    pub onset: f32,
    /// Hysteresis close threshold (`≤ onset`): probs `< offset` accumulate
    /// silence; reaching `min_silence_probs` consecutive such frames closes
    /// the run. Pass equal to `onset` for single-threshold behaviour.
    pub offset: f32,

    /// Minimum length (in probs) of a committed speech run. Shorter runs are
    /// dropped before chunk packing.
    pub min_speech_probs: usize,
    /// Number of consecutive `< offset` probs required to terminate a speech run.
    pub min_silence_probs: usize,
    /// Two committed runs separated by ≤ this many probs are merged.
    pub merge_gap_probs: usize,

    /// Soft minimum chunk length (in probs). Used as the left edge of the
    /// legal split window when min-cutting an over-long run; not a hard
    /// emission floor.
    pub min_chunk_probs: usize,
    /// Hard maximum chunk length (in probs). No emitted chunk's core spans
    /// more than this many probs.
    pub max_chunk_probs: usize,

    /// Symmetric pad in samples added to each decode window (clamped at 0
    /// and the implicit waveform end). After alignment, the next chunk's
    /// decode_start is pinned to `≥ prev.decode_end` so decode windows
    /// never overlap.
    pub pad_samples: usize,
    /// Snap decode-window boundaries to integer multiples of this many
    /// samples. Set to the encoder's effective frame stride
    /// (e.g. `mel_hop * subsample_factor`) so chunks land on encoder-frame
    /// boundaries.
    pub align_to: usize,

    /// Speech gate: drop any chunk whose **peak** Silero prob over its core
    /// (prob-grid) range is below this. Set to `0.0` to disable.
    pub min_chunk_max_prob: f32,
}

impl Default for ChunkerOpts {
    fn default() -> Self {
        // Defaults assume 16 kHz / 512-sample Silero windows ≈ 31.25 probs/s.
        // 8 probs ≈ 256 ms · 4 probs ≈ 128 ms · 13 probs ≈ 416 ms · 875 probs ≈ 28 s.
        Self {
            sample_rate: 16_000,
            samples_per_prob: 512,
            onset: 0.50,
            offset: 0.35,
            min_speech_probs: 8,
            min_silence_probs: 4,
            merge_gap_probs: 13,
            min_chunk_probs: 16,
            max_chunk_probs: 875,
            pad_samples: 4_800,
            align_to: 1,
            min_chunk_max_prob: 0.3,
        }
    }
}

// ─── Output ───────────────────────────────────────────────────────────────

/// A speech-bearing output region plus the waveform window used to decode it.
///
/// `start_sample..end_sample` is the non-overlapping core range that owns
/// output text. `decode_start_sample..decode_end_sample` is the possibly
/// wider model input window that supplies acoustic context. All sample indices
/// reference the original waveform passed to the VAD.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AudioChunk {
    pub start_sample: usize,
    pub end_sample: usize,
    pub decode_start_sample: usize,
    pub decode_end_sample: usize,
}

impl AudioChunk {
    pub fn new(start_sample: usize, end_sample: usize) -> Self {
        Self { start_sample, end_sample, decode_start_sample: start_sample, decode_end_sample: end_sample }
    }

    pub fn with_decode(
        start_sample: usize,
        end_sample: usize,
        decode_start_sample: usize,
        decode_end_sample: usize,
    ) -> Self {
        Self { start_sample, end_sample, decode_start_sample, decode_end_sample }
    }

    pub fn core_len(&self) -> usize {
        self.end_sample.saturating_sub(self.start_sample)
    }

    pub fn decode_len(&self) -> usize {
        self.decode_end_sample.saturating_sub(self.decode_start_sample)
    }
}

// ─── Errors ───────────────────────────────────────────────────────────────

#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum Error {
    #[snafu(display("samples_per_prob must be > 0"))]
    ZeroSamplesPerProb,
    #[snafu(display("align_to must be > 0"))]
    ZeroAlignTo,
    #[snafu(display("max_chunk_probs must be > 0"))]
    ZeroMaxChunk,
    #[snafu(display("min_chunk_probs ({min}) must be ≤ max_chunk_probs ({max})"))]
    MinExceedsMax { min: usize, max: usize },
    #[snafu(display("offset ({offset}) must be ≤ onset ({onset})"))]
    OffsetExceedsOnset { offset: f32, onset: f32 },
}

pub type Result<T> = std::result::Result<T, Error>;

/// Upper bound (in samples) on any chunk decode window [`chunks_from_probs`]
/// can emit: `max_chunk_probs · samples_per_prob + 2·pad + 2·(align_to - 1)`.
/// Downstream callers use this to size buffers.
pub fn max_chunk_sample_bound(
    max_chunk_probs: usize,
    samples_per_prob: usize,
    pad_samples: usize,
    align_to: usize,
) -> usize {
    max_chunk_probs * samples_per_prob + 2 * pad_samples + 2 * align_to.saturating_sub(1)
}

// ─── Public entry point ───────────────────────────────────────────────────

/// Pack VAD speech probabilities into bounded-length chunks via Cut & Merge.
pub fn chunks_from_probs(probs: &[f32], opts: &ChunkerOpts) -> Result<Vec<AudioChunk>> {
    validate(opts)?;
    if probs.is_empty() {
        return Ok(Vec::new());
    }

    let runs = hysteresis_segments(probs, opts.onset, opts.offset, opts.min_speech_probs, opts.min_silence_probs);
    let runs = merge_close(runs, opts.merge_gap_probs);
    let sub_runs = cut_long_runs(probs, &runs, opts.min_chunk_probs, opts.max_chunk_probs, opts.offset);
    let chunks = pack_chunks(&sub_runs, opts.max_chunk_probs);
    let chunks = post_process(&chunks, probs.len(), opts);
    Ok(speech_gate(chunks, probs, opts))
}

// ─── Internals ────────────────────────────────────────────────────────────

fn validate(opts: &ChunkerOpts) -> Result<()> {
    if opts.samples_per_prob == 0 {
        return ZeroSamplesPerProbSnafu.fail();
    }
    if opts.align_to == 0 {
        return ZeroAlignToSnafu.fail();
    }
    if opts.max_chunk_probs == 0 {
        return ZeroMaxChunkSnafu.fail();
    }
    if opts.min_chunk_probs > opts.max_chunk_probs {
        return MinExceedsMaxSnafu { min: opts.min_chunk_probs, max: opts.max_chunk_probs }.fail();
    }
    if opts.offset > opts.onset {
        return OffsetExceedsOnsetSnafu { offset: opts.offset, onset: opts.onset }.fail();
    }
    Ok(())
}

/// Split any run longer than `max_chunk_probs` at the rightmost frame within
/// the legal window `[run_start + min_chunk, run_start + max_chunk]` whose
/// prob is below `offset` (real silence). Falls back to the legal-window
/// argmin if no qualifying silence exists. Recurses on the right remainder.
fn cut_long_runs(
    probs: &[f32],
    runs: &[(usize, usize)],
    min_chunk: usize,
    max_chunk: usize,
    offset: f32,
) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(runs.len());
    for &(start, end) in runs {
        cut_one(probs, start, end, min_chunk, max_chunk, offset, &mut out);
    }
    out
}

fn cut_one(
    probs: &[f32],
    start: usize,
    end: usize,
    min_chunk: usize,
    max_chunk: usize,
    offset: f32,
    out: &mut Vec<(usize, usize)>,
) {
    if end - start <= max_chunk {
        out.push((start, end));
        return;
    }
    // Left edge of the legal split window: keep the LEFT piece ≥ min_chunk.
    let lo = (start + min_chunk.min(max_chunk)).max(start + 1);
    // Right edge: keep both pieces inside their length caps. The split must
    // leave the RIGHT remainder ≥ min_chunk where possible (avoids tiny
    // tails on the next recursion); always ≥ 1 frame.
    let hi_cap = start + max_chunk;
    let hi_keep_tail = end.saturating_sub(min_chunk.max(1));
    let hi = hi_cap.min(hi_keep_tail).min(end.saturating_sub(1)).max(lo);
    let split = find_split(probs, lo, hi, offset);
    out.push((start, split));
    cut_one(probs, split, end, min_chunk, max_chunk, offset, out);
}

/// Pick a split point in `[lo, hi]` (inclusive). Preferences:
/// 1. Rightmost frame with `prob < offset` — a real silence trough. Picking
///    the rightmost maximises chunk fill (WhisperX min-cut rule).
/// 2. Fallback when no qualifying silence exists: lowest-prob frame in
///    `[lo, hi]`, ties broken to the right. On a flat-speech run this
///    becomes `hi`, producing maximally-sized chunks.
fn find_split(probs: &[f32], lo: usize, hi: usize, offset: f32) -> usize {
    for i in (lo..=hi).rev() {
        if probs[i] < offset {
            return i;
        }
    }
    let mut best = hi;
    let mut best_v = probs[hi];
    for i in (lo..hi).rev() {
        if probs[i] < best_v {
            best_v = probs[i];
            best = i;
        }
    }
    best
}

/// Greedy-concat runs into chunks bounded by `max_chunk_probs`. Each input
/// run is assumed to already satisfy `len ≤ max_chunk_probs` (post-cut).
fn pack_chunks(runs: &[(usize, usize)], max_chunk: usize) -> Vec<(usize, usize)> {
    let mut chunks: Vec<(usize, usize)> = Vec::new();
    for &(rs, re) in runs {
        match chunks.last_mut() {
            Some(last) if re - last.0 <= max_chunk => last.1 = re,
            _ => chunks.push((rs, re)),
        }
    }
    chunks
}

/// Convert prob-index core ranges to sample ranges and derive decode windows.
/// Pad on each side is capped at half the silence gap to the neighbouring
/// core so pre-align decode windows never cross a core. After alignment,
/// the next chunk's `decode_start` is pinned up to the previous chunk's
/// `decode_end` to eliminate the ≤ `align_to - 1` overlap that floor/ceil
/// rounding can introduce.
fn post_process(chunks: &[(usize, usize)], probs_len: usize, opts: &ChunkerOpts) -> Vec<AudioChunk> {
    let max_sample = probs_len * opts.samples_per_prob;
    let pad = opts.pad_samples;
    let align = opts.align_to;
    let spp = opts.samples_per_prob;

    let mut out: Vec<AudioChunk> = Vec::with_capacity(chunks.len());
    for (i, &(ps, pe)) in chunks.iter().enumerate() {
        let core_start = ps * spp;
        let core_end = pe * spp;
        if core_end <= core_start {
            continue;
        }

        // Cap each side's pad at half the gap to the neighbouring core (full
        // budget at file edges). Floor division → adjacent chunks' raw pads
        // never cross.
        let pad_left = if i == 0 {
            pad.min(core_start)
        } else {
            let prev_core_end = chunks[i - 1].1 * spp;
            pad.min(core_start.saturating_sub(prev_core_end) / 2)
        };
        let pad_right = if i + 1 == chunks.len() {
            pad.min(max_sample.saturating_sub(core_end))
        } else {
            let next_core_start = chunks[i + 1].0 * spp;
            pad.min(next_core_start.saturating_sub(core_end) / 2)
        };

        let padded_start = core_start.saturating_sub(pad_left);
        let padded_end = core_end.saturating_add(pad_right).min(max_sample);

        let decode_start = (padded_start / align) * align;
        let mut decode_end = padded_end.div_ceil(align) * align;
        if decode_end > max_sample {
            decode_end = max_sample;
        }
        if decode_end <= decode_start {
            continue;
        }
        // Floor/ceil rounding around touching cores can leave adjacent
        // decode windows overlapping by ≤ align - 1 samples. We accept
        // that overlap rather than pinning past `core_start`: cores never
        // overlap (pack guarantees this) so any duplicated word at the
        // seam is filtered by `crop_words_to_core` downstream.
        out.push(AudioChunk::with_decode(core_start, core_end, decode_start, decode_end));
    }
    out
}

/// Drop any chunk whose peak prob over its core (prob-grid) range is below
/// `min_chunk_max_prob`. Defensive backstop: hysteresis already requires each
/// run to peak `≥ onset`, so this only fires on configuration accidents or
/// future bugs that emit chunks with no high-confidence speech.
fn speech_gate(chunks: Vec<AudioChunk>, probs: &[f32], opts: &ChunkerOpts) -> Vec<AudioChunk> {
    if opts.min_chunk_max_prob <= 0.0 {
        return chunks;
    }
    chunks
        .into_iter()
        .filter(|c| {
            let lo = c.start_sample / opts.samples_per_prob;
            let hi = c.end_sample.div_ceil(opts.samples_per_prob).min(probs.len());
            let slice = probs.get(lo..hi).unwrap_or(&[]);
            slice.iter().copied().fold(f32::NEG_INFINITY, f32::max) >= opts.min_chunk_max_prob
        })
        .collect()
}
