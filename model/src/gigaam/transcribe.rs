//! High-level `Transcriber` over a unified [`GigaAm`].
//!
//! Replaces the per-example pipeline (mel extract → Silero VAD → chunker →
//! sized JIT prepare → batched pack → encoder execute → per-head decode) with
//! one stateful wrapper. Construction builds the front-ends but does NOT
//! prepare the encoder JIT — bounds depend on the audio. The first
//! [`Transcriber::transcribe`] call sizes and prepares; subsequent calls reuse
//! the cached `(b, t_mel)` plan if the new audio's chunks fit underneath.
//!
//! Hides the CTC vs RN-T asymmetry behind one return type:
//! [`TranscribeResult`] with optional per-word timestamps for both heads.
//! SentencePiece `▁ → space` post-processing, the encoder-output transpose
//! that RN-T needs (`[d_model, T_sub] → [T_sub, d_model]`), and the CTC
//! `frames_to_words` grouping all live inside [`HeadDecoder`].

use bon::bon;
use snafu::{ResultExt, Snafu};
use svod_arch::ctc::CtcDecoder;
use svod_arch::rnnt::{RnntDecoder, RnntOpts};
use svod_tensor::PrepareConfig;

pub use svod_arch::rnnt::Word;

use crate::audio::{AudioChunk, EncoderBounds, MelConfig, MelSpectrogram, Splitter};
use crate::gigaam::SubsamplingMode;
use crate::gigaam::ctc::CtcHeadJit;
use crate::gigaam::jit::GigaAmEncoderJit;
use crate::gigaam::model::{GigaAm, Head};
use crate::gigaam::rnnt::RnntStepBackend;
use crate::jit::InputSpec;

/// User-facing knobs for [`Transcriber::transcribe`].
///
/// Construct with [`TranscribeOpts::builder`] (per-field overrides) or
/// [`TranscribeOpts::from_env`] (read `SVOD_*` env vars with sensible
/// fallbacks). The two agree — `from_env()` is just `builder().build()` —
/// so `builder().word_timestamps(true).build()` still consults env for the
/// rest of the fields.
///
/// Field defaults consult these env vars:
///
/// | Field             | Env var                | Fallback |
/// |-------------------|------------------------|----------|
/// | `word_timestamps` | `SVOD_TIMESTAMPS=1`   | `false`  |
/// | `beam_decode`     | `SVOD_BEAM_DECODE=1`  | `false`  |
/// | `max_scores_mib`  | `SVOD_MAX_SCORES_MIB` | `256`    |
///
/// VAD-specific knobs (`threshold`, `min_duration`, …) live on
/// [`SileroVadSplitter`](super::SileroVadSplitter), not here.
#[derive(Clone, Debug)]
pub struct TranscribeOpts {
    /// Emit per-word `Word { text, start, end }` entries on
    /// [`ChunkResult::words`]. Both heads support this.
    pub word_timestamps: bool,
    /// Promote the model's config-default CTC decoder to a beam decoder
    /// (no-op for RN-T).
    pub beam_decode: bool,
    /// Per-allocation budget for the SDPA scores buffer. Caps `max_batch`
    /// so two simultaneously live `[B, H, T_sub², dtype]` scores tensors
    /// stay under `2 × max_scores_mib` MiB.
    pub max_scores_mib: usize,
}

impl Default for TranscribeOpts {
    fn default() -> Self {
        Self::builder().build()
    }
}

#[bon]
impl TranscribeOpts {
    /// Build via the [`bon`] builder. Each field default consults its
    /// `SVOD_*` env var (see the struct docs for the full table) before
    /// falling back to a literal — so `builder().build()` produces the same
    /// values as [`from_env`](Self::from_env), and partial overrides
    /// (`.word_timestamps(true).build()`) still env-read the rest.
    #[builder]
    pub fn builder(
        #[builder(default = std::env::var("SVOD_TIMESTAMPS").as_deref() == Ok("1"))] word_timestamps: bool,
        #[builder(default = std::env::var("SVOD_BEAM_DECODE").as_deref() == Ok("1"))] beam_decode: bool,
        #[builder(default = std::env::var("SVOD_MAX_SCORES_MIB").ok().and_then(|s| s.parse().ok()).unwrap_or(256))]
        max_scores_mib: usize,
    ) -> Self {
        Self { word_timestamps, beam_decode, max_scores_mib }
    }

    /// Build from `SVOD_*` env vars with the same fallbacks as the
    /// builder. Equivalent to `Self::builder().build()`.
    pub fn from_env() -> Self {
        Self::builder().build()
    }
}

/// Aggregated transcription output. `text` is the chunk texts joined by a
/// single space (empty chunks dropped); [`words`](Self::words) flattens word
/// timestamps across chunks (shifted by each chunk's `start_sec`).
#[derive(Clone, Debug)]
pub struct TranscribeResult {
    pub text: String,
    pub chunks: Vec<ChunkResult>,
}

impl TranscribeResult {
    /// Iterate per-word timestamps across all chunks, shifted into the
    /// original audio's timeline. Empty if `opts.word_timestamps` was false.
    pub fn words(&self) -> impl Iterator<Item = Word> + '_ {
        self.chunks.iter().flat_map(|c| {
            let offset = c.start_sec;
            c.words.iter().flatten().map(move |w| Word {
                text: w.text.clone(),
                start: w.start + offset,
                end: w.end + offset,
            })
        })
    }
}

/// One VAD-bound speech region's transcript. `start_sec`/`end_sec` reference
/// the original audio. `words` is `Some` iff [`TranscribeOpts::word_timestamps`]
/// was set; each entry's `start`/`end` is **chunk-relative** (add `start_sec`
/// to get audio-absolute, or use [`TranscribeResult::words`]).
#[derive(Clone, Debug)]
pub struct ChunkResult {
    pub start_sec: f32,
    pub end_sec: f32,
    pub text: String,
    pub words: Option<Vec<Word>>,
}

#[derive(Clone, Copy, Debug)]
struct ChunkMeta {
    decode_start: usize,
    decode_end: usize,
    mel_len: usize,
    start_sec: f32,
    end_sec: f32,
    decode_start_sec: f32,
}

impl ChunkMeta {
    fn from_chunk(chunk: &AudioChunk, mel: &MelSpectrogram, sample_rate_hz: usize) -> Option<Self> {
        let mel_len = mel.num_frames(chunk.decode_len());
        if mel_len == 0 {
            return None;
        }

        let sample_rate = sample_rate_hz as f32;
        Some(Self {
            decode_start: chunk.decode_start_sample,
            decode_end: chunk.decode_end_sample,
            mel_len,
            start_sec: chunk.start_sample as f32 / sample_rate,
            end_sec: chunk.end_sample as f32 / sample_rate,
            decode_start_sec: chunk.decode_start_sample as f32 / sample_rate,
        })
    }

    fn decode_duration_sec(&self, sample_rate_hz: usize) -> f32 {
        (self.decode_end - self.decode_start) as f32 / sample_rate_hz as f32
    }

    fn result(self, text: String, words: Option<Vec<Word>>) -> ChunkResult {
        ChunkResult { start_sec: self.start_sec, end_sec: self.end_sec, text, words }
    }
}

/// Per-head decoder + JIT state. CTC needs a bounds-tied head JIT (Conv1d
/// projection); RN-T's predictor/joint JITs ride with [`RnntStepBackend`].
/// One instance per `Transcriber`, so the variant-size disparity is
/// irrelevant — boxing would just add an allocation.
#[allow(clippy::large_enum_variant)]
pub(crate) enum HeadDecoder {
    Ctc { jit: CtcHeadJit, decoder: CtcDecoder },
    Rnnt { backend: RnntStepBackend, decoder: RnntDecoder, sentencepiece: bool },
}

/// CTC equivalent of [`RnntDecoder::frames_to_words`].
///
/// Walks the decoded `text` in lockstep with `frames` (CTC's
/// `decode_with_timestamps` returns one frame index per emitted *token*; for
/// GigaAM's char-level vocab one token == one Unicode scalar, so `text.chars()`
/// is the right zip target). Splits on ASCII space — no SentencePiece on the
/// CTC side. Returns chunk-relative `[start, end)` in seconds.
pub(crate) fn ctc_frames_to_words(text: &str, frames: &[usize], frame_shift: f32) -> Vec<Word> {
    let mut words: Vec<Word> = Vec::new();
    let mut current = String::new();
    let mut first_frame = 0usize;
    let mut last_frame = 0usize;

    let commit = |words: &mut Vec<Word>, current: &mut String, first: usize, last: usize| {
        if !current.is_empty() {
            words.push(Word {
                text: std::mem::take(current),
                start: first as f32 * frame_shift,
                end: (last + 1) as f32 * frame_shift,
            });
        }
    };

    for (ch, &frame) in text.chars().zip(frames.iter()) {
        if ch == ' ' {
            commit(&mut words, &mut current, first_frame, last_frame);
            continue;
        }
        if current.is_empty() {
            first_frame = frame;
        }
        current.push(ch);
        last_frame = frame;
    }
    commit(&mut words, &mut current, first_frame, last_frame);
    words
}

pub(crate) fn crop_words_to_core(
    words: Vec<Word>,
    decode_start_sec: f32,
    core_start_sec: f32,
    core_end_sec: f32,
) -> Vec<Word> {
    let core_duration = core_end_sec - core_start_sec;
    words
        .into_iter()
        .filter_map(|mut w| {
            let abs_start = w.start + decode_start_sec;
            let abs_end = w.end + decode_start_sec;
            let mid = 0.5 * (abs_start + abs_end);
            if !(core_start_sec..core_end_sec).contains(&mid) {
                return None;
            }
            w.start = (abs_start - core_start_sec).clamp(0.0, core_duration);
            w.end = (abs_end - core_start_sec).clamp(w.start, core_duration);
            Some(w)
        })
        .collect()
}

pub(crate) fn words_to_text(words: &[Word]) -> String {
    words.iter().map(|w| w.text.as_str()).filter(|s| !s.is_empty()).collect::<Vec<_>>().join(" ")
}

/// Transpose `[d_model, t_exec_sub]` row-major → `[actual_sub, d_model]`.
/// `actual_sub <= t_exec_sub` (the JIT pads frames beyond `actual_sub`); only
/// the first `actual_sub` frames are read.
fn transpose_dt_to_td(src: &[f32], d_model: usize, t_exec_sub: usize, actual_sub: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; actual_sub * d_model];
    for t in 0..actual_sub {
        for d in 0..d_model {
            out[t * d_model + d] = src[d * t_exec_sub + t];
        }
    }
    out
}

fn rnnt_decode_err<E: std::error::Error + 'static>(
    e: svod_arch::rnnt::RnntDecodeError<crate::jit::JitError>,
) -> TranscribeError<E> {
    TranscribeError::RnntDecode { source: Box::new(e) }
}

// ─── Errors ───────────────────────────────────────────────────────────────

/// Generic over the splitter error type so per-impl errors stay
/// pattern-matchable rather than being type-erased into `Box<dyn Error>`.
/// Mirrors the `svod_arch::rnnt::RnntDecodeError<JitError>` shape.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub(crate)))]
pub enum TranscribeError<E: std::error::Error + 'static> {
    #[snafu(display("splitter: {source}"))]
    Splitter { source: E },
    #[snafu(display("{source}"))]
    Jit {
        #[snafu(source(from(crate::jit::JitError, Box::new)))]
        source: Box<crate::jit::JitError>,
    },
    #[snafu(display("{source}"))]
    CtcDecode { source: svod_arch::ctc::DecodeError },
    #[snafu(display("{source}"))]
    RnntDecode { source: Box<svod_arch::rnnt::RnntDecodeError<crate::jit::JitError>> },
    #[snafu(display("{source}"))]
    Model {
        #[snafu(source(from(crate::gigaam::error::Error, Box::new)))]
        source: Box<crate::gigaam::error::Error>,
    },
    #[snafu(display("{source}"))]
    Tensor {
        #[snafu(source(from(svod_tensor::error::Error, Box::new)))]
        source: Box<svod_tensor::error::Error>,
    },
    #[snafu(display("{source}"))]
    Device {
        #[snafu(source(from(svod_device::error::Error, Box::new)))]
        source: Box<svod_device::error::Error>,
    },
    #[snafu(display("WAV is {wav_sr} Hz, model expects {model_sr} Hz (resample first)"))]
    SampleRateMismatch { wav_sr: u32, model_sr: u32 },
    #[snafu(display("chunk {idx} length {samples} samples exceeds encoder capacity {max_samples} samples"))]
    ChunkExceedsCapacity { idx: usize, samples: usize, max_samples: usize },
    #[snafu(display("chunk {idx} end {end_sample} exceeds waveform length {waveform_len}"))]
    ChunkOutOfRange { idx: usize, end_sample: usize, waveform_len: usize },
}

// ─── Transcriber ──────────────────────────────────────────────────────────

/// High-level transcription wrapper, generic over the chunking strategy.
/// JITs are prepared eagerly at construction; the splitter advertises its
/// max chunk length so JIT buffers can be sized tighter than the encoder's
/// hard ceiling. Use [`transcribe_chunks`](Self::transcribe_chunks) to
/// bypass the splitter for pre-segmented audio.
pub struct Transcriber<S: Splitter> {
    model: GigaAm,
    opts: TranscribeOpts,
    splitter: S,
    mel: MelSpectrogram,
    head_decoder: HeadDecoder,
    encoder_jit: GigaAmEncoderJit,
    max_batch: usize,
    max_t_mel: usize,
}

impl<S: Splitter> Transcriber<S> {
    /// Build the transcriber and prepare every JIT eagerly — subsequent
    /// `transcribe` calls just execute. `model` is cloned into each JIT
    /// (cheap: weights are shared via `Tensor` handle Arcs).
    pub fn new(model: GigaAm, splitter: S, opts: TranscribeOpts) -> Result<Self, TranscribeError<S::Error>> {
        let mel = MelSpectrogram::new(&MelConfig {
            sample_rate: model.config.sample_rate,
            n_fft: model.config.n_fft,
            hop_length: model.config.hop_length,
            win_length: model.config.win_length,
            n_mels: model.config.n_mels,
            center: model.config.mel_center,
        });

        let subsampling_factor = model.config.subsampling_factor;
        let hop_length = model.config.hop_length;
        let model_bounds = EncoderBounds {
            sample_rate: model.config.sample_rate as u32,
            hop_length,
            subsampling_factor,
            max_mel_frames: model.config.max_mel_frames,
        };
        // Splitter advertises its emission ceiling; clamp to encoder
        // capacity, then round up to the next power of two so the JIT
        // codegen sees a clean factorisation.
        let chunk_samples_cap = splitter.max_chunk_samples(&model_bounds).min(model_bounds.max_samples());
        let chunk_mel = (chunk_samples_cap / hop_length).saturating_add(2 * subsampling_factor);
        let max_t_mel = chunk_mel.max(1).next_power_of_two().min(model.config.max_mel_frames).max(subsampling_factor);

        // SDPA scores `[B, H, T_sub², dtype]` are live twice during attention;
        // budget `max_batch` so they stay under `2 * max_scores_mib`.
        let t_sub_max = (max_t_mel / subsampling_factor).max(1);
        let scores_dtype_bytes = model.encoder.input_dtype().bytes();
        let bytes_per_batch = model.config.n_heads * t_sub_max * t_sub_max * scores_dtype_bytes;
        let target_scores_bytes = opts.max_scores_mib * 1024 * 1024;
        let max_batch_by_memory = (target_scores_bytes / bytes_per_batch.max(1)).max(1);
        let max_batch = max_batch_by_memory.min(model.config.max_batch_size);

        let prepare_config = PrepareConfig::from_env();
        let mut encoder_jit = GigaAmEncoderJit::new(model.clone()).with_b_bound(max_batch).with_t_bound(max_t_mel);
        encoder_jit
            .prepare_with_config(
                InputSpec::f32(&[max_batch, model.config.n_mels, max_t_mel]),
                InputSpec::i32(&[max_batch]),
                &prepare_config,
            )
            .context(JitSnafu)?;

        let head_decoder = match &model.head {
            Head::Ctc(_) => {
                let decoder = if opts.beam_decode {
                    match &model.config.decoder {
                        CtcDecoder::Greedy(g) => CtcDecoder::Beam(Box::new(svod_arch::ctc::BeamDecoder::new(
                            g.vocabulary().to_vec(),
                            svod_arch::ctc::BeamOpts::default(),
                        ))),
                        other => other.clone(),
                    }
                } else {
                    model.config.decoder.clone()
                };
                let subs_kernel_size = match model.config.subsampling_mode {
                    SubsamplingMode::Conv1d => model.config.subs_kernel_size,
                    SubsamplingMode::Conv2d => 3,
                };
                let max_t_sub = subs_output_length(subs_kernel_size, max_t_mel);
                let mut jit = CtcHeadJit::new(model.clone()).with_b_bound(max_batch).with_t_sub_bound(max_t_sub);
                jit.prepare_with_config(InputSpec::f32(&[max_batch, model.config.d_model, max_t_sub]), &prepare_config)
                    .context(JitSnafu)?;
                HeadDecoder::Ctc { jit, decoder }
            }
            Head::Rnnt { runtime, .. } => {
                let backend = RnntStepBackend::from_model(model.clone()).context(JitSnafu)?;
                let decoder = RnntDecoder::new(
                    runtime.vocabulary.clone(),
                    RnntOpts { max_symbols_per_step: runtime.max_symbols_per_step },
                );
                HeadDecoder::Rnnt { backend, decoder, sentencepiece: runtime.sentencepiece }
            }
        };

        Ok(Self { model, opts, splitter, mel, head_decoder, encoder_jit, max_batch, max_t_mel })
    }

    /// Encoder bounds at the model's full capacity. Passed to splitters
    /// at split time so they can clamp chunks to the encoder's ceiling.
    pub fn encoder_bounds(&self, sample_rate: u32) -> Result<EncoderBounds, TranscribeError<S::Error>> {
        self.bounds_with(sample_rate, self.model.config.max_mel_frames)
    }

    /// Encoder bounds tightened to this transcriber's prepared JIT capacity.
    fn prepared_bounds(&self, sample_rate: u32) -> Result<EncoderBounds, TranscribeError<S::Error>> {
        self.bounds_with(sample_rate, self.max_t_mel)
    }

    fn bounds_with(&self, sample_rate: u32, max_mel_frames: usize) -> Result<EncoderBounds, TranscribeError<S::Error>> {
        if sample_rate as usize != self.model.config.sample_rate {
            return Err(TranscribeError::SampleRateMismatch {
                wav_sr: sample_rate,
                model_sr: self.model.config.sample_rate as u32,
            });
        }
        Ok(EncoderBounds {
            sample_rate,
            hop_length: self.model.config.hop_length,
            subsampling_factor: self.model.config.subsampling_factor,
            max_mel_frames,
        })
    }

    /// Transcribe a waveform end-to-end: bounds → splitter → mel → batched
    /// encoder → per-chunk head decode. `waveform` is fp32 PCM in `[-1, 1]`;
    /// the model expects `model.config.sample_rate` (returns
    /// [`TranscribeError::SampleRateMismatch`] otherwise).
    pub fn transcribe(
        &mut self,
        waveform: &[f32],
        sample_rate: u32,
    ) -> Result<TranscribeResult, TranscribeError<S::Error>> {
        let bounds = self.encoder_bounds(sample_rate)?;
        let chunks = self.splitter.split(waveform, &bounds).context(SplitterSnafu)?;
        self.transcribe_chunks(waveform, sample_rate, &chunks)
    }

    /// Escape hatch: caller-supplied chunks. Validates each chunk against
    /// encoder capacity (`ChunkExceedsCapacity`) and the waveform's bounds
    /// (`ChunkOutOfRange`) rather than silently truncating. Misaligned
    /// boundaries are accepted — the mel/JIT pipeline pads the trailing
    /// fractional frame.
    pub fn transcribe_chunks(
        &mut self,
        waveform: &[f32],
        sample_rate: u32,
        chunks: &[AudioChunk],
    ) -> Result<TranscribeResult, TranscribeError<S::Error>> {
        // Validate against the prepared JIT capacity, not the model's
        // worst case — oversized chunks must error here, not inside the JIT.
        let max_samples = self.prepared_bounds(sample_rate)?.max_samples();
        for (idx, chunk) in chunks.iter().enumerate() {
            if chunk.end_sample > waveform.len() {
                return Err(TranscribeError::ChunkOutOfRange {
                    idx,
                    end_sample: chunk.end_sample,
                    waveform_len: waveform.len(),
                });
            }
            if chunk.decode_end_sample > waveform.len() {
                return Err(TranscribeError::ChunkOutOfRange {
                    idx,
                    end_sample: chunk.decode_end_sample,
                    waveform_len: waveform.len(),
                });
            }
            let samples = chunk.decode_len();
            if samples > max_samples {
                return Err(TranscribeError::ChunkExceedsCapacity { idx, samples, max_samples });
            }
        }

        let n_mels = self.mel.n_mels();
        if chunks.is_empty() {
            return Ok(TranscribeResult { text: String::new(), chunks: Vec::new() });
        }

        let sample_rate_hz = self.model.config.sample_rate;
        let d_model = self.model.config.d_model;
        let subs_kernel_size = match self.model.config.subsampling_mode {
            SubsamplingMode::Conv1d => self.model.config.subs_kernel_size,
            SubsamplingMode::Conv2d => 3,
        };
        let max_t_mel = self.max_t_mel;
        let max_t_sub = subs_output_length(subs_kernel_size, max_t_mel);
        let max_batch = self.max_batch;
        let want_words = self.opts.word_timestamps;

        let chunks_meta: Vec<ChunkMeta> =
            chunks.iter().filter_map(|chunk| ChunkMeta::from_chunk(chunk, &self.mel, sample_rate_hz)).collect();
        if chunks_meta.is_empty() {
            return Ok(TranscribeResult { text: String::new(), chunks: Vec::new() });
        }

        let num_chunks = chunks_meta.len();
        let mut chunk_results: Vec<ChunkResult> = Vec::with_capacity(num_chunks);
        for chunk_batch_start in (0..num_chunks).step_by(max_batch) {
            let b = (num_chunks - chunk_batch_start).min(max_batch);
            let mut chunk_lengths = vec![0usize; b];

            let batch_mels: Vec<Vec<f32>> = (0..b)
                .map(|bi| {
                    let meta = chunks_meta[chunk_batch_start + bi];
                    let mut chunk_mel = ndarray::Array3::<f32>::zeros((1, n_mels, meta.mel_len));
                    {
                        let mut view = chunk_mel.view_mut().into_dyn();
                        self.mel.forward_into(&waveform[meta.decode_start..meta.decode_end], &mut view);
                    }
                    chunk_mel.as_slice().expect("contiguous chunk mel").to_vec()
                })
                .collect();

            // Pack mel into encoder JIT input buffer.
            {
                let buf = self.encoder_jit.mel_mut().context(JitSnafu)?;
                let mut view = buf.as_array_mut::<f32>().context(DeviceSnafu)?;
                let slice = view.as_slice_mut().expect("contiguous mel buffer");
                slice.fill(0.0);
                for (bi, chunk_len) in chunk_lengths.iter_mut().enumerate() {
                    let meta = chunks_meta[chunk_batch_start + bi];
                    *chunk_len = meta.mel_len;
                    let chunk_mel = &batch_mels[bi];
                    for mel_bin in 0..n_mels {
                        let src = mel_bin * meta.mel_len;
                        let dst = ((bi * n_mels) + mel_bin) * max_t_mel;
                        slice[dst..dst + meta.mel_len].copy_from_slice(&chunk_mel[src..src + meta.mel_len]);
                    }
                }
            }
            // Pack lengths into encoder JIT.
            {
                let buf = self.encoder_jit.lengths_mut().context(JitSnafu)?;
                let mut view = buf.as_array_mut::<i32>().context(DeviceSnafu)?;
                let slice = view.as_slice_mut().expect("contiguous lengths buffer");
                slice.fill(0);
                for (i, len) in chunk_lengths.iter().enumerate() {
                    slice[i] = *len as i32;
                }
            }

            let t_exec = chunk_lengths.iter().copied().max().unwrap_or(1).max(1);
            let t_exec_sub = subs_output_length(subs_kernel_size, t_exec);
            self.encoder_jit.execute_with_vars(&[("b", b as i64), ("t", t_exec as i64)]).context(JitSnafu)?;

            // CTC chains the encoder output into the head JIT once per batch
            // then decodes per item; RN-T decodes the encoder output
            // directly per item (its JITs ride with the backend).
            match &mut self.head_decoder {
                HeadDecoder::Ctc { jit, decoder } => {
                    // Chain encoder output [b, d_model, t_exec_sub] into the
                    // head input slab [max_batch, d_model, max_t_sub].
                    {
                        let n = b * d_model * t_exec_sub;
                        let src_flat =
                            self.encoder_jit.output().context(JitSnafu)?.as_array::<f32>().context(DeviceSnafu)?;
                        let src_3d = src_flat
                            .slice(ndarray::s![0..n])
                            .into_shape_with_order((b, d_model, t_exec_sub))
                            .expect("encoder output reshape");
                        let dst_flat =
                            jit.encoded_mut().context(JitSnafu)?.as_array_mut::<f32>().context(DeviceSnafu)?;
                        let mut dst_3d = dst_flat
                            .into_shape_with_order((max_batch, d_model, max_t_sub))
                            .expect("head input reshape");
                        dst_3d.slice_mut(ndarray::s![0..b, 0..d_model, 0..t_exec_sub]).assign(&src_3d);
                    }
                    jit.execute_with_vars(&[("b", b as i64), ("t_sub", t_exec_sub as i64)]).context(JitSnafu)?;

                    let total_vocab = decoder.total_vocab();
                    let item_stride = t_exec_sub * total_vocab;
                    let logits_buf = jit.output().context(JitSnafu)?;
                    let logits = logits_buf.as_array::<f32>().context(DeviceSnafu)?;
                    let flat = logits.as_slice().expect("contiguous head logits");
                    for (bi, mel_len) in chunk_lengths.iter().enumerate() {
                        let actual_sub = subs_output_length(subs_kernel_size, *mel_len);
                        let meta = chunks_meta[chunk_batch_start + bi];
                        let frame_shift = meta.decode_duration_sec(sample_rate_hz) / (actual_sub.max(1) as f32);

                        let item_slice = &flat[bi * item_stride..bi * item_stride + item_stride];

                        let (text, frames) = decoder
                            .decode_with_timestamps(item_slice, t_exec_sub, actual_sub)
                            .context(CtcDecodeSnafu)?;
                        let cropped_words = crop_words_to_core(
                            ctc_frames_to_words(&text, &frames, frame_shift),
                            meta.decode_start_sec,
                            meta.start_sec,
                            meta.end_sec,
                        );
                        let text = words_to_text(&cropped_words);
                        let words = want_words.then_some(cropped_words);
                        chunk_results.push(meta.result(text, words));
                    }
                }
                HeadDecoder::Rnnt { backend, decoder, sentencepiece } => {
                    let item_stride = d_model * t_exec_sub;
                    let enc_buf = self.encoder_jit.output().context(JitSnafu)?;
                    let enc = enc_buf.as_array::<f32>().context(DeviceSnafu)?;
                    let flat = enc.as_slice().expect("contiguous encoder output");
                    for (bi, mel_len) in chunk_lengths.iter().enumerate() {
                        let actual_sub = subs_output_length(subs_kernel_size, *mel_len);
                        let meta = chunks_meta[chunk_batch_start + bi];
                        let frame_shift = meta.decode_duration_sec(sample_rate_hz) / (actual_sub.max(1) as f32);

                        let item_slice = &flat[bi * item_stride..bi * item_stride + item_stride];
                        // Encoder output is [d_model, t_exec_sub] row-major;
                        // the arch decoder wants frame-major [actual_sub, d_model].
                        let frames = transpose_dt_to_td(item_slice, d_model, t_exec_sub, actual_sub);

                        let backend: &mut RnntStepBackend = backend;
                        let (_raw, emissions) = decoder
                            .decode_with_timestamps(&frames, actual_sub, actual_sub, d_model, backend)
                            .map_err(rnnt_decode_err)?;
                        let cropped_words = crop_words_to_core(
                            decoder.frames_to_words(&emissions, frame_shift),
                            meta.decode_start_sec,
                            meta.start_sec,
                            meta.end_sec,
                        );
                        let mut text = words_to_text(&cropped_words);
                        if *sentencepiece {
                            text = text.replace('\u{2581}', " ").trim().to_string();
                        }
                        let words = want_words.then_some(cropped_words);
                        chunk_results.push(meta.result(text, words));
                    }
                }
            }
        }

        let text =
            chunk_results.iter().map(|c| c.text.as_str()).filter(|s| !s.is_empty()).collect::<Vec<_>>().join(" ");
        Ok(TranscribeResult { text, chunks: chunk_results })
    }
}

/// Compute the encoder's sub-sampled output frame count from the input
/// mel-frame count. Mirrors the two-stage 2× stride conv stack used by
/// GigaAM's subsampling (kernel `subs_kernel_size`, stride 2, applied twice).
fn subs_output_length(kernel_size: usize, mel_frames: usize) -> usize {
    let pad = (kernel_size - 1) / 2;
    let mut len = mel_frames;
    for _ in 0..2 {
        len = (len + 2 * pad - kernel_size) / 2 + 1;
    }
    len
}
