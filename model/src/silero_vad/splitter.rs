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
