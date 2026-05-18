//! VAD-aware chunker for long-form ASR.
//!
//! Operates on `&[f32]` per-frame speech probabilities — the output of any
//! frame-level VAD — and packs them into bounded-length [`AudioChunk`]s
//! suitable for feeding to an encoder one chunk at a time. Speech-bearing
//! regions of the waveform are preserved; inter-chunk silence is dropped except
//! for bounded handoff pre-roll in clustered mode.
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
//! 3. greedy-pack runs into bounded envelopes
//! 4. optionally split long envelopes at midpoint-nearest VAD gaps
//! 5. convert prob indices → samples, apply pad, align to align_to
//! ```
//!
//! All knobs live in [`ChunkerOpts`]; nothing inside the algorithm hardcodes
//! sample rates, prob granularity, or alignment.

pub(crate) mod segment;

use std::cmp::Reverse;

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
    /// Optional target duration for gap-aware clustering. When set, chunking
    /// recursively splits greedy VAD envelopes at midpoint-nearest inter-run
    /// gaps, with the hard strict limit still enforced.
    pub cluster_target_duration: Option<f32>,
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
    /// Secondary threshold (typically lower than `threshold`) for
    /// overlong-run splitting. When `Some(t)`, search the full legal split
    /// range for the frame closest to the geometric target with prob
    /// `< t`; fall back to the narrow argmin around the target when no
    /// frame qualifies. `None` always uses narrow argmin.
    pub trough_threshold: Option<f32>,
    /// Symmetric pad in samples added to each decode window (clamped at 0 and
    /// the implicit waveform end). Gives the encoder context at chunk seams.
    pub pad_samples: usize,
    /// Snap decode-window boundaries to integer multiples of this many samples.
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
            cluster_target_duration: None,
            strict_limit_duration: 30.0,
            min_speech_probs: 8,
            min_silence_probs: 4,
            merge_gap_probs: 8,
            trough_search_probs: None,
            trough_threshold: None,
            pad_samples: 0,
            align_to: 1,
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
    /// Inclusive core/output start sample index in the original waveform.
    pub start_sample: usize,
    /// Exclusive core/output end sample index.
    pub end_sample: usize,
    /// Inclusive decode-window start sample index in the original waveform.
    pub decode_start_sample: usize,
    /// Exclusive decode-window end sample index. May exceed the waveform
    /// length if the last prob entry covered samples past the waveform end;
    /// callers owning the waveform should clamp before slicing.
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
    #[snafu(display("min_duration ({min}) must be ≤ max_duration ({max})"))]
    MinExceedsMax { min: f32, max: f32 },
    #[snafu(display("max_duration ({max}) must be ≤ strict_limit_duration ({strict})"))]
    MaxExceedsStrict { max: f32, strict: f32 },
}

pub type Result<T> = std::result::Result<T, Error>;

/// Upper bound (in samples) on any chunk decode window [`chunks_from_probs`]
/// can emit: `strict_limit + 2·pad + 2·(align_to - 1)`. Single source of
/// truth for downstream callers that need to size buffers or assert the
/// contract.
pub fn strict_chunk_sample_bound(
    strict_limit_probs: usize,
    samples_per_prob: usize,
    pad_samples: usize,
    align_to: usize,
) -> usize {
    strict_limit_probs * samples_per_prob + 2 * pad_samples + 2 * align_to.saturating_sub(1)
}

// ─── Public entry point ───────────────────────────────────────────────────

/// Pack VAD speech probabilities into bounded-length chunks.
///
/// Output chunk cores cover speech-bearing portions of the waveform and may
/// retain bounded pre-roll from a split gap when clustering is enabled. Decode
/// windows are expanded by `opts.pad_samples` and snapped to `opts.align_to`
/// multiples (start floored, end ceil'd) so the model receives boundary context
/// without changing output ownership.
pub fn chunks_from_probs(probs: &[f32], opts: &ChunkerOpts) -> Result<Vec<AudioChunk>> {
    validate(opts)?;
    if probs.is_empty() {
        return Ok(Vec::new());
    }

    let probs_per_sec = opts.sample_rate as f32 / opts.samples_per_prob as f32;
    let strict_limit_probs = (opts.strict_limit_duration * probs_per_sec).ceil() as usize;
    let min_probs = (opts.min_duration * probs_per_sec).ceil() as usize;
    let max_probs = (opts.max_duration * probs_per_sec).ceil() as usize;
    let cluster_target_probs = opts
        .cluster_target_duration
        .map(|duration| (duration * probs_per_sec).ceil() as usize)
        .filter(|&probs| probs > 0);

    let trough_radius = opts.trough_search_probs.unwrap_or(opts.min_silence_probs);
    let trough_threshold = opts.trough_threshold;

    // Halve the silence-sensitivity knobs and retry if `threshold_segments`
    // produced any segment exceeding `strict_limit_probs` — gives long-run splitting
    // less work / more silence to cut at. Floor at 2 because a single
    // sub-threshold prob is reliably a VAD micro-dip mid-word, not silence.
    let mut adapted = opts.clone();
    let segments = loop {
        let segs = threshold_segments(probs, &adapted);
        let any_over = segs.iter().any(|&(s, e)| e - s > strict_limit_probs);
        if !any_over || adapted.min_silence_probs <= 2 {
            break segs;
        }
        adapted.min_silence_probs = (adapted.min_silence_probs / 2).max(2);
        adapted.merge_gap_probs = (adapted.merge_gap_probs / 2).max(1);
    };
    let segments = LongRunSplitter::new(probs, trough_radius, trough_threshold, strict_limit_probs).split(segments);
    let chunks = if let Some(target_probs) = cluster_target_probs {
        ClusterPacker::new(&segments, min_probs, max_probs, strict_limit_probs, target_probs, opts.min_silence_probs)
            .pack()
    } else {
        pack_segments(&segments, min_probs, max_probs, strict_limit_probs)
    };

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

/// Splits single VAD runs that are longer than the model can decode.
struct LongRunSplitter<'a> {
    probs: &'a [f32],
    search_radius: usize,
    trough_threshold: Option<f32>,
    strict_limit_probs: usize,
}

#[derive(Clone, Copy)]
struct SplitWindow {
    lo: usize,
    hi: usize,
}

impl<'a> LongRunSplitter<'a> {
    fn new(probs: &'a [f32], search_radius: usize, trough_threshold: Option<f32>, strict_limit_probs: usize) -> Self {
        Self { probs, search_radius, trough_threshold, strict_limit_probs }
    }

    fn split(&self, segments: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
        if self.strict_limit_probs == 0 {
            return segments;
        }

        let mut out = Vec::with_capacity(segments.len());
        for segment in segments {
            self.split_one(segment, &mut out);
        }
        out
    }

    fn split_one(&self, (start, end): (usize, usize), out: &mut Vec<(usize, usize)>) {
        let len = end - start;
        if len <= self.strict_limit_probs {
            out.push((start, end));
            return;
        }

        let pieces = len.div_ceil(self.strict_limit_probs);
        let min_piece = (len / (2 * pieces)).max(1);
        let mut cur = start;
        for piece_idx in 1..pieces {
            let target = start + (len * piece_idx) / pieces;
            let pieces_left = pieces - piece_idx;
            let Some(window) = self.legal_window(cur, end, pieces_left, min_piece) else {
                continue;
            };
            let split = self.choose_split(target, window);
            if split > cur && split < end {
                out.push((cur, split));
                cur = split;
            }
        }

        if cur < end {
            out.push((cur, end));
        }
    }

    fn legal_window(&self, cur: usize, end: usize, pieces_left: usize, min_piece: usize) -> Option<SplitWindow> {
        let lo_fit = end.saturating_sub(pieces_left * self.strict_limit_probs).max(cur + 1);
        let hi_fit = (cur + self.strict_limit_probs).min(end.saturating_sub(pieces_left));
        if hi_fit < lo_fit {
            return None;
        }

        let lo_quality = lo_fit.max(cur + min_piece.min(self.strict_limit_probs));
        let lo = if lo_quality <= hi_fit { lo_quality } else { lo_fit };
        Some(SplitWindow { lo, hi: hi_fit })
    }

    fn choose_split(&self, target: usize, window: SplitWindow) -> usize {
        if let Some(split) = self.threshold_split(target, window) {
            return split;
        }

        let lo = target.saturating_sub(self.search_radius).max(window.lo);
        let hi = (target + self.search_radius).min(window.hi);
        if hi < lo {
            return target.clamp(window.lo, window.hi);
        }
        lo + argmin(&self.probs[lo..=hi])
    }

    fn threshold_split(&self, target: usize, window: SplitWindow) -> Option<usize> {
        let threshold = self.trough_threshold?;
        self.probs[window.lo..=window.hi]
            .iter()
            .enumerate()
            .filter(|&(_, &p)| p < threshold)
            .min_by_key(|(i, _)| (window.lo + i).abs_diff(target))
            .map(|(i, _)| window.lo + i)
    }
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

/// Gap-aware ordered clustering for RNNT-sensitive long-form decoding.
struct ClusterPacker<'a> {
    segments: &'a [(usize, usize)],
    min_probs: usize,
    max_probs: usize,
    strict_limit_probs: usize,
    target_probs: usize,
    handoff_probs: usize,
}

#[derive(Clone, Copy, Debug)]
struct SegmentGroup {
    start_idx: usize,
    end_idx: usize,
}

impl SegmentGroup {
    fn new(start_idx: usize, end_idx: usize) -> Self {
        Self { start_idx, end_idx }
    }
}

impl<'a> ClusterPacker<'a> {
    fn new(
        segments: &'a [(usize, usize)],
        min_probs: usize,
        max_probs: usize,
        strict_limit_probs: usize,
        target_probs: usize,
        handoff_probs: usize,
    ) -> Self {
        Self { segments, min_probs, max_probs, strict_limit_probs, target_probs, handoff_probs }
    }

    fn pack(&self) -> Vec<(usize, usize)> {
        if self.segments.is_empty() || self.target_probs == 0 {
            return Vec::new();
        }

        let mut groups = Vec::new();
        for group in self.greedy_groups() {
            self.split_at_midpoints(group, &mut groups);
        }
        self.materialize(&groups)
    }

    fn greedy_groups(&self) -> Vec<SegmentGroup> {
        let mut groups = Vec::new();
        let mut start_idx = 0usize;
        for idx in 1..self.segments.len() {
            let prospective = self.segments[idx].1 - self.segments[start_idx].0;
            let cur_len = self.segments[idx - 1].1 - self.segments[start_idx].0;
            if prospective > self.max_probs && (cur_len >= self.min_probs || prospective > self.strict_limit_probs) {
                groups.push(SegmentGroup::new(start_idx, idx));
                start_idx = idx;
            }
        }
        groups.push(SegmentGroup::new(start_idx, self.segments.len()));
        groups
    }

    fn split_at_midpoints(&self, group: SegmentGroup, out: &mut Vec<SegmentGroup>) {
        if group.end_idx <= group.start_idx + 1 || self.group_len(group) <= self.split_threshold() {
            out.push(group);
            return;
        }

        let split = self.midpoint_gap(group).expect("non-singleton group has an internal split");
        self.split_at_midpoints(SegmentGroup::new(group.start_idx, split), out);
        self.split_at_midpoints(SegmentGroup::new(split, group.end_idx), out);
    }

    fn materialize(&self, groups: &[SegmentGroup]) -> Vec<(usize, usize)> {
        if groups.is_empty() {
            return Vec::new();
        }

        let mut chunks = Vec::with_capacity(groups.len());
        let mut start = self.segments[groups[0].start_idx].0;

        for (idx, group) in groups.iter().copied().enumerate() {
            let end = self.segments[group.end_idx - 1].1;
            if end > start {
                chunks.push((start.max(end.saturating_sub(self.strict_limit_probs)), end));
            }
            if let Some(next) = groups.get(idx + 1).copied() {
                start = self.handoff_start(end, self.segments[next.start_idx].0);
            }
        }
        chunks
    }

    fn group_len(&self, group: SegmentGroup) -> usize {
        self.segments[group.end_idx - 1].1 - self.group_start(group.start_idx)
    }

    fn group_start(&self, start_idx: usize) -> usize {
        if start_idx == 0 {
            self.segments[0].0
        } else {
            self.handoff_start(self.segments[start_idx - 1].1, self.segments[start_idx].0)
        }
    }

    fn split_threshold(&self) -> usize {
        self.target_probs.saturating_add(self.target_probs / 2)
    }

    fn midpoint_gap(&self, group: SegmentGroup) -> Option<usize> {
        let group_start = self.group_start(group.start_idx);
        let group_end = self.segments[group.end_idx - 1].1;
        let midpoint_twice = group_start + group_end;

        (group.start_idx + 1..group.end_idx).min_by_key(|&idx| self.gap_key(idx, midpoint_twice))
    }

    fn gap_key(&self, split_idx: usize, midpoint_twice: usize) -> (usize, Reverse<usize>) {
        let gap_start = self.segments[split_idx - 1].1;
        let gap_end = self.segments[split_idx].0;
        let distance_twice = if midpoint_twice < 2 * gap_start {
            2 * gap_start - midpoint_twice
        } else {
            midpoint_twice.saturating_sub(2 * gap_end)
        };
        (distance_twice, Reverse(gap_end.saturating_sub(gap_start)))
    }

    fn handoff_start(&self, gap_start: usize, gap_end: usize) -> usize {
        if gap_end <= gap_start {
            return gap_end;
        }

        let gap = gap_end - gap_start;
        let drop_after_prev = self.handoff_probs.min(gap / 2);
        let pre_context = (self.target_probs * 2 / 5).max(1);
        (gap_start + drop_after_prev).max(gap_end.saturating_sub(pre_context)).min(gap_end)
    }
}

/// Convert prob-index core ranges to sample ranges and derive decode windows.
/// Padding is adaptive: each side is capped at half the core gap to the
/// neighbour, so decode windows remain non-overlapping while still exposing
/// local silence/context around each core.
fn post_process(chunks: &[(usize, usize)], probs_len: usize, opts: &ChunkerOpts) -> Vec<AudioChunk> {
    let max_sample = probs_len * opts.samples_per_prob;
    let pad = opts.pad_samples;
    let align = opts.align_to;

    let mut out: Vec<AudioChunk> = Vec::with_capacity(chunks.len());
    for (i, &(s, e)) in chunks.iter().enumerate() {
        let core_start = s * opts.samples_per_prob;
        let core_end = e * opts.samples_per_prob;

        // Cap each side's padding at half the silence gap to the neighbour
        // (or the full margin at the waveform edges). Floor division: total
        // pad consumed by adjacent chunks ≤ gap, so they never overlap.
        let pad_left = if i == 0 {
            pad.min(core_start)
        } else {
            let prev_raw_end = chunks[i - 1].1 * opts.samples_per_prob;
            pad.min(core_start.saturating_sub(prev_raw_end) / 2)
        };
        let pad_right = if i + 1 == chunks.len() {
            pad.min(max_sample.saturating_sub(core_end))
        } else {
            let next_raw_start = chunks[i + 1].0 * opts.samples_per_prob;
            pad.min(next_raw_start.saturating_sub(core_end) / 2)
        };

        let padded_start = core_start - pad_left;
        let padded_end = (core_end + pad_right).min(max_sample);
        let decode_start = (padded_start / align) * align;
        let mut decode_end = padded_end.div_ceil(align) * align;
        if decode_end > max_sample {
            decode_end = max_sample;
        }
        if core_end <= core_start || decode_end <= decode_start {
            continue;
        }
        out.push(AudioChunk::with_decode(core_start, core_end, decode_start, decode_end));
    }
    out
}
