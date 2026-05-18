//! Silero-VAD-driven [`Splitter`](crate::audio::Splitter) implementation.
//!
//! Wraps [`VadInference`] + [`svod_arch::vad::chunks_from_probs`] for
//! long-form ASR. Knobs are exposed in wall-clock seconds and translated to
//! the prob-grid units that [`ChunkerOpts`](svod_arch::vad::ChunkerOpts)
//! expects at split time.

use bon::bon;
use snafu::{ResultExt, Snafu};

use crate::audio::{AudioChunk, EncoderBounds, Splitter, trim_chunks_to_waveform};
use crate::silero_vad::{NUM_SAMPLES, SileroVad, VadInference};

/// 2-second safety margin under the encoder's `max_samples()` so chunks
/// never sit at the JIT capacity ceiling.
const ENCODER_SAFETY_MARGIN_SECS: f32 = 2.0;

/// VAD-driven splitter. Construction: [`from_hub`](Self::from_hub) loads the
/// default model from HF and pulls overrides from `SVOD_VAD_*` env vars. For
/// custom knobs use [`builder`](Self::builder) and supply a pre-loaded
/// [`VadInference`].
pub struct SileroVadSplitter {
    vad: VadInference,
    onset: f32,
    offset: f32,
    min_speech_secs: f32,
    min_silence_secs: f32,
    merge_gap_secs: f32,
    min_chunk_secs: f32,
    max_chunk_secs: f32,
    pad_secs: f32,
    min_chunk_max_prob: f32,
}

/// Silero-gated fixed-window splitter. Runs VAD to decide which global,
/// non-overlapping fixed windows contain speech, then transcribes the full
/// retained windows. Useful as a robust fallback when VAD probabilities are
/// unreliable.
pub struct SileroFixedWindowSplitter {
    vad: VadInference,
    threshold: f32,
    window_duration_secs: f32,
    min_speech_probs: usize,
}

#[bon]
impl SileroVadSplitter {
    /// Build from an already-loaded [`VadInference`]. Defaults pull from
    /// `SVOD_VAD_ONSET` / `SVOD_VAD_OFFSET` env vars, with fallbacks
    /// `0.50` / `0.35`.
    #[builder]
    pub fn builder(
        vad: VadInference,
        #[builder(default = env_f32("SVOD_VAD_ONSET", 0.50))] onset: f32,
        #[builder(default = env_f32("SVOD_VAD_OFFSET", 0.35))] offset: f32,
        /// Minimum length (s) of a committed speech run.
        #[builder(default = env_f32("SVOD_VAD_MIN_SPEECH_SECS", 0.25))]
        min_speech_secs: f32,
        /// Consecutive `< offset` duration (s) required to close a run.
        #[builder(default = env_f32("SVOD_VAD_MIN_SILENCE_SECS", 0.10))]
        min_silence_secs: f32,
        /// Adjacent speech runs separated by ≤ this many seconds are merged.
        #[builder(default = env_f32("SVOD_VAD_MERGE_GAP_SECS", 0.40))]
        merge_gap_secs: f32,
        /// Soft floor for chunk length — used as the left edge of the legal
        /// min-cut window when splitting an over-long run.
        #[builder(default = env_f32("SVOD_VAD_MIN_CHUNK_SECS", 0.5))]
        min_chunk_secs: f32,
        /// Hard ceiling for chunk length. Clamped at split time to the
        /// encoder's prepared budget minus a 2 s safety margin. Empirically
        /// 20 s gives the best RN-T results — longer chunks (22 s, 28 s)
        /// degrade quality on heterogeneous content because the decoder
        /// "skips" earlier tokens when it encounters garbled/transition
        /// audio mid-chunk. Upstream Python (`gigaam/vad_utils.py`) uses
        /// max=22, but it accumulates short speech segments and is less
        /// affected by this failure mode.
        #[builder(default = env_f32("SVOD_VAD_MAX_CHUNK_SECS", 20.0))]
        max_chunk_secs: f32,
        /// Per-side decode-window pad in seconds. 300 ms covers the Conformer
        /// convolutional receptive field at the subsampled rate.
        #[builder(default = env_f32("SVOD_VAD_PAD_SECS", 0.30))]
        pad_secs: f32,
        /// Speech gate: drop chunks whose peak Silero prob is below this.
        #[builder(default = env_f32("SVOD_VAD_GATE", 0.3))]
        min_chunk_max_prob: f32,
    ) -> Self {
        Self {
            vad,
            onset,
            offset,
            min_speech_secs,
            min_silence_secs,
            merge_gap_secs,
            min_chunk_secs,
            max_chunk_secs,
            pad_secs,
            min_chunk_max_prob,
        }
    }

    /// Convenience: download the default Silero model from HF Hub, wrap it in
    /// a [`VadInference`], and apply env-var-driven defaults.
    pub fn from_hub() -> Result<Self, SileroVadSplitterError> {
        let model = SileroVad::from_hub().context(LoadSnafu)?;
        let vad = VadInference::new(model).context(InferenceSnafu)?;
        Ok(Self::builder().vad(vad).build())
    }

    fn effective_max_chunk_secs(&self, bounds: &EncoderBounds) -> f32 {
        let pad_samples = (self.pad_secs * bounds.sample_rate as f32).round() as usize;
        let align = bounds.align_to_samples().max(1);
        let overhead = 2 * pad_samples + 2 * align.saturating_sub(1);
        let encoder_capacity = bounds.max_samples().saturating_sub(overhead) as f32 / bounds.sample_rate as f32;
        let safe_capacity = (encoder_capacity - ENCODER_SAFETY_MARGIN_SECS).max(0.0);
        self.max_chunk_secs.min(safe_capacity).max(0.0)
    }
}

#[bon]
impl SileroFixedWindowSplitter {
    /// Build from an already-loaded [`VadInference`]. Defaults to 20-second
    /// non-overlapping windows and keeps any window with at least one
    /// above-threshold Silero probability.
    #[builder]
    pub fn builder(
        vad: VadInference,
        #[builder(default = env_f32("MOROK_VAD_THRESHOLD", 0.5))] threshold: f32,
        #[builder(default = 20.0)] window_duration_secs: f32,
        #[builder(default = 1)] min_speech_probs: usize,
    ) -> Self {
        assert!(window_duration_secs.is_finite() && window_duration_secs > 0.0, "window duration must be positive");
        Self { vad, threshold, window_duration_secs, min_speech_probs: min_speech_probs.max(1) }
    }

    pub fn from_hub() -> Result<Self, SileroVadSplitterError> {
        let model = SileroVad::from_hub().context(LoadSnafu)?;
        let vad = VadInference::new(model).context(InferenceSnafu)?;
        Ok(Self::builder().vad(vad).build())
    }

    pub fn from_hub_with_window_duration_secs(window_duration_secs: f32) -> Result<Self, SileroVadSplitterError> {
        let model = SileroVad::from_hub().context(LoadSnafu)?;
        let vad = VadInference::new(model).context(InferenceSnafu)?;
        Ok(Self::builder().vad(vad).window_duration_secs(window_duration_secs).build())
    }

    fn window_samples(&self, bounds: &EncoderBounds) -> usize {
        let duration_samples = (self.window_duration_secs * bounds.sample_rate as f32).floor() as usize;
        duration_samples.max(1).min(bounds.max_samples().max(1))
    }
}

impl Splitter for SileroFixedWindowSplitter {
    type Error = SileroVadSplitterError;

    fn split(&mut self, waveform: &[f32], bounds: &EncoderBounds) -> Result<Vec<AudioChunk>, Self::Error> {
        let probs = self.vad.probs(waveform).context(ProbsSnafu)?;
        Ok(fixed_windows_from_probs(
            &probs,
            waveform.len(),
            NUM_SAMPLES,
            self.threshold,
            self.min_speech_probs,
            self.window_samples(bounds),
        ))
    }

    fn max_chunk_samples(&self, bounds: &EncoderBounds) -> usize {
        self.window_samples(bounds)
    }
}

pub(crate) fn fixed_windows_from_probs(
    probs: &[f32],
    waveform_len: usize,
    samples_per_prob: usize,
    threshold: f32,
    min_speech_probs: usize,
    window_samples: usize,
) -> Vec<AudioChunk> {
    if probs.is_empty() || waveform_len == 0 || samples_per_prob == 0 || window_samples == 0 {
        return Vec::new();
    }

    let min_speech_probs = min_speech_probs.max(1);
    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < waveform_len {
        let end = start.saturating_add(window_samples).min(waveform_len);
        let prob_start = start / samples_per_prob;
        let prob_end = end.div_ceil(samples_per_prob).min(probs.len());
        let speech_probs = probs.get(prob_start..prob_end).unwrap_or(&[]).iter().filter(|&&p| p >= threshold).count();
        if speech_probs >= min_speech_probs {
            chunks.push(AudioChunk::new(start, end));
        }
        start = end;
    }
    chunks
}

impl Splitter for SileroVadSplitter {
    type Error = SileroVadSplitterError;

    fn split(&mut self, waveform: &[f32], bounds: &EncoderBounds) -> Result<Vec<AudioChunk>, Self::Error> {
        let probs = self.vad.probs(waveform).context(ProbsSnafu)?;
        let opts = self.chunker_opts(bounds);
        let mut chunks = svod_arch::vad::chunks_from_probs(&probs, &opts).context(ChunkSnafu)?;
        // align_to rounds chunk ends up to a stride multiple, which can push
        // the trailing chunk past `waveform.len()`. The Splitter contract
        // forbids that, so clamp the tail before returning.
        trim_chunks_to_waveform(&mut chunks, waveform.len());
        Ok(chunks)
    }

    fn max_chunk_samples(&self, bounds: &EncoderBounds) -> usize {
        let opts = self.chunker_opts(bounds);
        svod_arch::vad::max_chunk_sample_bound(opts.max_chunk_probs, NUM_SAMPLES, opts.pad_samples, opts.align_to)
    }
}

impl SileroVadSplitter {
    fn chunker_opts(&self, bounds: &EncoderBounds) -> svod_arch::vad::ChunkerOpts {
        let sr = bounds.sample_rate as f32;
        let probs_per_sec = sr / NUM_SAMPLES as f32;
        let max_chunk_secs = self.effective_max_chunk_secs(bounds);
        let max_chunk_probs = (max_chunk_secs * probs_per_sec).floor() as usize;
        let min_chunk_probs = (self.min_chunk_secs * probs_per_sec).ceil() as usize;
        svod_arch::vad::ChunkerOpts {
            sample_rate: bounds.sample_rate,
            samples_per_prob: NUM_SAMPLES,
            onset: self.onset,
            offset: self.offset,
            min_speech_probs: (self.min_speech_secs * probs_per_sec).ceil() as usize,
            min_silence_probs: (self.min_silence_secs * probs_per_sec).ceil() as usize,
            merge_gap_probs: (self.merge_gap_secs * probs_per_sec).ceil() as usize,
            min_chunk_probs: min_chunk_probs.min(max_chunk_probs.max(1)),
            max_chunk_probs: max_chunk_probs.max(1),
            pad_samples: (self.pad_secs * sr).round() as usize,
            align_to: bounds.align_to_samples().max(1),
            min_chunk_max_prob: self.min_chunk_max_prob,
        }
    }
}

fn env_f32(key: &str, fallback: f32) -> f32 {
    std::env::var(key).ok().and_then(|s| s.parse().ok()).unwrap_or(fallback)
}

#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum SileroVadSplitterError {
    #[snafu(display("loading Silero VAD model: {source}"))]
    Load {
        #[snafu(source(from(crate::silero_vad::Error, Box::new)))]
        source: Box<crate::silero_vad::Error>,
    },
    #[snafu(display("building Silero VAD JIT: {source}"))]
    Inference {
        #[snafu(source(from(crate::jit::JitError, Box::new)))]
        source: Box<crate::jit::JitError>,
    },
    #[snafu(display("running Silero VAD: {source}"))]
    Probs {
        #[snafu(source(from(crate::jit::JitError, Box::new)))]
        source: Box<crate::jit::JitError>,
    },
    #[snafu(display("chunker: {source}"))]
    Chunk { source: svod_arch::vad::Error },
}
