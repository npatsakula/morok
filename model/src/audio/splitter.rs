//! Chunking abstraction for long-form audio inference.
//!
//! `Splitter` is the pluggable boundary between a raw waveform and the
//! encoder-bounded chunks an ASR inference loop consumes. The trait + the
//! built-in [`FixedLengthSplitter`] live here (audio-level preprocessing,
//! same module family as the mel front-end); VAD-driven implementations
//! ship next to their VAD model — see
//! [`SileroVadSplitter`](crate::silero_vad::SileroVadSplitter).
//!
//! [`Transcriber`](crate::gigaam::Transcriber) is generic over the splitter;
//! `S::Error` flows through `TranscribeError<E>` the same way
//! `RnntDecodeError<JitError>` carries the step-backend error.
//!
//! Users with pre-segmented audio (pyannote, manual cuts) can skip the
//! splitter entirely via
//! [`Transcriber::transcribe_chunks`](crate::gigaam::Transcriber::transcribe_chunks).

pub use svod_arch::vad::AudioChunk;

/// Encoder-derived bounds passed to [`Splitter::split`].
///
/// Carries the model-config primitives (not derived seconds counts) so
/// splitters can reason in whichever unit fits them. Helpers
/// [`max_samples`](Self::max_samples), [`align_to_samples`](Self::align_to_samples),
/// and [`encoder_capacity_secs`](Self::encoder_capacity_secs) cover the common
/// derivations.
///
/// The `2 * subsampling_factor` headroom that
/// [`encoder_capacity_secs`](Self::encoder_capacity_secs) subtracts mirrors the
/// JIT prepare loop's `subs_output_length` margin — a chunk filling
/// `max_samples()` is guaranteed to fit through the subsampling stack without
/// padding overflow.
#[derive(Clone, Copy, Debug)]
pub struct EncoderBounds {
    pub sample_rate: u32,
    pub hop_length: usize,
    pub subsampling_factor: usize,
    pub max_mel_frames: usize,
}

impl EncoderBounds {
    /// Sample-domain stride alignment: `hop_length * subsampling_factor`.
    /// Splitters that produce frame-aligned chunks should snap boundaries to
    /// this multiple.
    pub fn align_to_samples(&self) -> usize {
        self.hop_length * self.subsampling_factor
    }

    /// Maximum chunk length (in samples) the encoder can ingest. Equals
    /// `(max_mel_frames - 2 * subsampling_factor) * hop_length` — the headroom
    /// subtraction matches the JIT prepare path.
    pub fn max_samples(&self) -> usize {
        self.max_mel_frames.saturating_sub(2 * self.subsampling_factor) * self.hop_length
    }

    /// Convenience for splitters that reason in wall-clock seconds.
    pub fn encoder_capacity_secs(&self) -> f32 {
        self.max_samples() as f32 / self.sample_rate as f32
    }
}

/// Chunking strategy: turn a waveform into encoder-bounded `AudioChunk`s.
///
/// `split` is called once per
/// [`Transcriber::transcribe`](crate::gigaam::Transcriber::transcribe) call.
/// Implementations may keep mutable state across calls (the Silero VAD JIT
/// carries LSTM state internally), hence `&mut self`. Chunks must satisfy
/// `end_sample <= waveform.len()` and `end_sample - start_sample <=
/// bounds.max_samples()`; alignment is a soft preference (floor-division on
/// `hop_length` tolerates misaligned boundaries).
///
/// Splitters that align chunk ends to encoder strides
/// ([`align_to_samples`](EncoderBounds::align_to_samples)) can round the
/// trailing chunk past `waveform.len()`. The contract still requires
/// in-range chunks — use [`trim_chunks_to_waveform`] to clean up the tail
/// before returning.
pub trait Splitter {
    type Error: std::error::Error + Send + Sync + 'static;

    fn split(&mut self, waveform: &[f32], bounds: &EncoderBounds) -> Result<Vec<AudioChunk>, Self::Error>;

    /// Upper bound (in samples) on the longest chunk this splitter could
    /// emit. Consumed by the transcriber to size JIT buffers — a tighter
    /// advertised bound trades peak memory for tighter chunk handling.
    /// Default: full encoder capacity.
    fn max_chunk_samples(&self, bounds: &EncoderBounds) -> usize {
        bounds.max_samples()
    }
}

/// Trim `chunks` so every entry stays within `0..waveform_len`: drop chunks
/// starting at or past `waveform_len`, then clamp the trailing chunk's
/// `end_sample` down to `waveform_len`. Intended for [`Splitter`]
/// implementations whose stride alignment can push the last chunk past the
/// waveform end (see the trait docs).
///
/// Assumes `chunks` is in increasing `start_sample` order — the
/// `Splitter::split` contract — so a single pass from the back is enough.
pub fn trim_chunks_to_waveform(chunks: &mut Vec<AudioChunk>, waveform_len: usize) {
    while let Some(mut last) = chunks.pop() {
        if last.start_sample >= waveform_len {
            continue;
        }
        last.end_sample = last.end_sample.min(waveform_len);
        last.decode_start_sample = last.decode_start_sample.min(waveform_len);
        last.decode_end_sample = last.decode_end_sample.min(waveform_len);
        chunks.push(last);
        break;
    }
}

/// No-VAD splitter: walks the waveform in fixed-size strides. By default the
/// stride is `bounds.max_samples()`; use
/// [`with_max_duration_secs`](Self::with_max_duration_secs) for a wall-clock
/// cap such as 20 seconds.
///
/// Zero model load. Suitable when the caller already segmented the input,
/// for short utterances that fit a single chunk, or for tests. Boundary
/// context degrades transcription quality at chunk seams — for production
/// long-form ASR prefer
/// [`SileroVadSplitter`](crate::silero_vad::SileroVadSplitter).
///
/// `align_to_samples` dividing `max_samples` is not guaranteed; the final
/// aligned chunk is the floor of `len_remaining / align_to_samples` times
/// `align_to_samples` (so its mel length stays an integer multiple of
/// `subsampling_factor`). The very last chunk keeps its unaligned tail —
/// the JIT pads it.
#[derive(Clone, Debug, Default)]
pub struct FixedLengthSplitter {
    max_duration_secs: Option<f32>,
}

impl FixedLengthSplitter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_max_duration_secs(max_duration_secs: f32) -> Self {
        assert!(max_duration_secs.is_finite() && max_duration_secs > 0.0, "max duration must be positive");
        Self { max_duration_secs: Some(max_duration_secs) }
    }

    pub fn max_duration_secs(&self) -> Option<f32> {
        self.max_duration_secs
    }

    fn target_samples(&self, bounds: &EncoderBounds) -> usize {
        let encoder_cap = bounds.max_samples();
        let Some(max_duration_secs) = self.max_duration_secs else {
            return encoder_cap;
        };
        let duration_cap = (max_duration_secs * bounds.sample_rate as f32).floor() as usize;
        duration_cap.max(1).min(encoder_cap.max(1))
    }
}

impl Splitter for FixedLengthSplitter {
    type Error = std::convert::Infallible;

    fn split(&mut self, waveform: &[f32], bounds: &EncoderBounds) -> Result<Vec<AudioChunk>, Self::Error> {
        if waveform.is_empty() {
            return Ok(Vec::new());
        }
        let align = bounds.align_to_samples().max(1);
        // Saturating target so misconfigured bounds (zero, overflow) or tiny
        // duration caps never stall the loop.
        let max = if self.max_duration_secs.is_some() {
            self.target_samples(bounds).max(1)
        } else {
            self.target_samples(bounds).max(align)
        };

        let mut chunks = Vec::new();
        let mut start = 0usize;
        while start < waveform.len() {
            let nominal_end = start.saturating_add(max).min(waveform.len());
            let aligned_end = if nominal_end == waveform.len() {
                nominal_end
            } else {
                let span = nominal_end - start;
                let aligned_span = (span / align) * align;
                if aligned_span == 0 { nominal_end } else { start + aligned_span }
            };
            chunks.push(AudioChunk::new(start, aligned_end));
            start = aligned_end;
        }
        Ok(chunks)
    }

    fn max_chunk_samples(&self, bounds: &EncoderBounds) -> usize {
        self.target_samples(bounds)
    }
}
