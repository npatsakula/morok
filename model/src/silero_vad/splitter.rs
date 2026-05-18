//! Silero-VAD-driven [`Splitter`](crate::audio::Splitter) implementation.
//!
//! Wraps [`VadInference`] + [`svod_arch::vad::chunks_from_probs`] so the
//! pre-refactor `Transcriber` chunking flow is reachable through the generic
//! splitter trait. Knobs forward to [`ChunkerOpts`](svod_arch::vad::ChunkerOpts)
//! except for `sample_rate`, `samples_per_prob`, and `align_to` — those come
//! from [`EncoderBounds`](crate::audio::EncoderBounds) at split time.

use bon::bon;
use snafu::{ResultExt, Snafu};

use crate::audio::{AudioChunk, EncoderBounds, Splitter, trim_chunks_to_waveform};
use crate::silero_vad::{NUM_SAMPLES, SileroVad, VadInference};

/// VAD-driven splitter. Construction: [`from_hub`](Self::from_hub) loads the
/// default model from HF and pulls overrides from `SVOD_VAD_THRESHOLD`. For
/// custom knobs use [`builder`](Self::builder) and supply a pre-loaded
/// [`VadInference`].
pub struct SileroVadSplitter {
    vad: VadInference,
    threshold: f32,
    min_duration: f32,
    max_duration: f32,
    strict_limit_duration: f32,
    min_speech_probs: usize,
    min_silence_probs: usize,
    merge_gap_probs: usize,
    trough_search_probs: Option<usize>,
    pad_samples: usize,
}

/// Silero-gated fixed-window splitter. Runs VAD to decide which global,
/// non-overlapping fixed windows contain speech, then transcribes the full
/// retained windows. This keeps weak speech inside a retained 20s window even
/// if Silero does not mark every frame as speech.
pub struct SileroFixedWindowSplitter {
    vad: VadInference,
    threshold: f32,
    window_duration_secs: f32,
    min_speech_probs: usize,
}

#[derive(Clone, Copy, Debug)]
struct DurationBudget {
    min: f32,
    max: f32,
    strict_limit: f32,
}

#[bon]
impl SileroVadSplitter {
    /// Build from an already-loaded [`VadInference`]. All knob defaults match
    /// [`ChunkerOpts::default`](svod_arch::vad::ChunkerOpts) except
    /// `threshold`, which consults `SVOD_VAD_THRESHOLD`.
    #[builder]
    pub fn builder(
        vad: VadInference,
        #[builder(default = std::env::var("SVOD_VAD_THRESHOLD").ok().and_then(|s| s.parse().ok()).unwrap_or(0.5))]
        threshold: f32,
        #[builder(default = 15.0)] min_duration: f32,
        #[builder(default = 22.0)] max_duration: f32,
        #[builder(default = 30.0)] strict_limit_duration: f32,
        #[builder(default = 8)] min_speech_probs: usize,
        #[builder(default = 4)] min_silence_probs: usize,
        #[builder(default = 8)] merge_gap_probs: usize,
        trough_search_probs: Option<usize>,
        /// Pad budget (samples) per chunk side. Default `1600` (= 100 ms at
        /// 16 kHz). The actual pad applied is capped at half the silence
        /// gap to the neighbouring chunk — so chunks never overlap into
        /// each other's speech (no transcript duplication), but at seams
        /// with enough surrounding silence the encoder sees up to this
        /// many extra samples of context on each side.
        #[builder(default = 1600)]
        pad_samples: usize,
    ) -> Self {
        Self {
            vad,
            threshold,
            min_duration,
            max_duration,
            strict_limit_duration,
            min_speech_probs,
            min_silence_probs,
            merge_gap_probs,
            trough_search_probs,
            pad_samples,
        }
    }

    /// Convenience: download the default Silero model from HF Hub, wrap it in
    /// a [`VadInference`], and apply env-var-driven knob defaults.
    pub fn from_hub() -> Result<Self, SileroVadSplitterError> {
        let model = SileroVad::from_hub().context(LoadSnafu)?;
        let vad = VadInference::new(model).context(InferenceSnafu)?;
        Ok(Self::builder().vad(vad).build())
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
        #[builder(default = std::env::var("MOROK_VAD_THRESHOLD").ok().and_then(|s| s.parse().ok()).unwrap_or(0.5))]
        threshold: f32,
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
        let budget = self.duration_budget(bounds);
        let chunker_opts = svod_arch::vad::ChunkerOpts {
            sample_rate: bounds.sample_rate,
            samples_per_prob: NUM_SAMPLES,
            threshold: self.threshold,
            min_duration: budget.min,
            max_duration: budget.max,
            cluster_target_duration: Some(budget.max * 0.5),
            strict_limit_duration: budget.strict_limit,
            min_speech_probs: self.min_speech_probs,
            min_silence_probs: self.min_silence_probs,
            merge_gap_probs: self.merge_gap_probs,
            trough_search_probs: self.trough_search_probs,
            trough_threshold: Some(self.threshold * 0.5),
            pad_samples: self.pad_samples,
            align_to: bounds.align_to_samples().max(1),
        };
        let mut chunks = svod_arch::vad::chunks_from_probs(&probs, &chunker_opts).context(ChunkSnafu)?;
        // `align_to` rounds chunk ends up to a stride multiple, which can push
        // the trailing chunk past `waveform.len()`. The `Splitter` contract
        // forbids that, so clamp the tail before returning.
        trim_chunks_to_waveform(&mut chunks, waveform.len());
        Ok(chunks)
    }

    /// Upper bound on chunk length the chunker can emit under this
    /// splitter's config. Translating to samples lets `Transcriber::new`
    /// size JIT buffers to this chunker's actual emission rather than the
    /// encoder's full capacity. Shared bound math lives in
    /// [`svod_arch::vad::strict_chunk_sample_bound`].
    fn max_chunk_samples(&self, bounds: &EncoderBounds) -> usize {
        let probs_per_sec = bounds.sample_rate as f32 / NUM_SAMPLES as f32;
        let strict_limit_probs = (self.duration_budget(bounds).strict_limit * probs_per_sec).ceil() as usize;
        svod_arch::vad::strict_chunk_sample_bound(
            strict_limit_probs,
            NUM_SAMPLES,
            self.pad_samples,
            bounds.align_to_samples(),
        )
    }
}

impl SileroVadSplitter {
    fn duration_budget(&self, bounds: &EncoderBounds) -> DurationBudget {
        let align = bounds.align_to_samples().max(1);
        let overhead = 2 * self.pad_samples + 2 * align.saturating_sub(1);
        let safe_core_probs = bounds.max_samples().saturating_sub(overhead) / NUM_SAMPLES;
        let configured_probs =
            (self.strict_limit_duration * bounds.sample_rate as f32 / NUM_SAMPLES as f32).ceil() as usize;
        let strict_limit_probs = configured_probs.min(safe_core_probs).max(1);
        let strict_limit = strict_limit_probs as f32 * NUM_SAMPLES as f32 / bounds.sample_rate as f32;
        let max = self.max_duration.min(strict_limit);
        DurationBudget { min: self.min_duration.min(strict_limit), max, strict_limit }
    }
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
