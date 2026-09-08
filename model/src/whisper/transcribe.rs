//! Prepared Whisper recognition and aligned-transcription stages.
//!
//! Each decode window runs through a concrete-capacity encoder, one reusable
//! cross-K/V projection, token prefill, and fixed-slot cached decoding. The
//! independent aligner replays finalized tokens through a teacher-forced graph
//! and computes word timings on the host.

use std::time::{Duration, Instant};

use snafu::Snafu;
use svod_arch::pipelines::audio::{Transcriber, Transcript};
use svod_runtime::{RunProfile, StageProfile};
use svod_tensor::PrepareConfig;

use crate::jit::InputSpec;

use super::aligner::{AlignmentProfile, WhisperAligner, WhisperAlignmentInput};
use super::config::{N_AUDIO_CTX, N_FRAMES, N_TEXT_CTX, SAMPLE_RATE};
use super::decode::{
    DecodeOptions, DecodeScheduleStats, attempt_strategies, detect_language_profile, prefill_decode_seed,
    run_fixed_slot_decode, strategy_width,
};
use super::jit::{WhisperCrossKvJit, WhisperDecoderJit, WhisperDecoderStepJit, WhisperEncoderJit, WhisperPrefillJit};
use super::mel::WhisperMel;
use super::model::Whisper;
use super::plan::WhisperPlan;
use super::profile::{CopyProfile, GraphProfile, begin_host_copy};
use super::tokenizer::WhisperTokenizer;

pub use svod_arch::rnnt::Word;

fn timed_d2d<T>(
    enabled: bool,
    fence: &svod_device::Buffer,
    work: impl FnOnce() -> Result<T, TranscribeError>,
) -> Result<(T, Duration), TranscribeError> {
    if !enabled {
        return work().map(|value| (value, Duration::ZERO));
    }
    // Exclude prior graph work, then wait for the complete async transfer group.
    // This is synchronized host wall, not a hardware SDMA timestamp.
    fence.synchronize()?;
    let started = Instant::now();
    let value = work()?;
    fence.synchronize()?;
    Ok((value, started.elapsed()))
}

#[derive(Debug, Snafu)]
#[snafu(visibility(pub(crate)))]
pub enum TranscribeError {
    #[snafu(display("{source}"), context(false))]
    Jit {
        #[snafu(source(from(crate::jit::JitError, Box::new)))]
        source: Box<crate::jit::JitError>,
    },
    #[snafu(display("{source}"))]
    Model {
        #[snafu(source(from(super::error::Error, Box::new)))]
        source: Box<super::error::Error>,
    },
    #[snafu(display("{source}"), context(false))]
    Tensor {
        #[snafu(source(from(svod_tensor::error::Error, Box::new)))]
        source: Box<svod_tensor::error::Error>,
    },
    #[snafu(display("{source}"), context(false))]
    Device {
        #[snafu(source(from(svod_device::error::Error, Box::new)))]
        source: Box<svod_device::error::Error>,
    },
}

/// Prepared timestamp-enabled recognizer. It owns only recognition graphs;
/// word alignment is a separate [`WhisperAligner`] stage.
pub struct WhisperRecognizer {
    mel: WhisperMel,
    encoder_jit: WhisperEncoderJit,
    decoder_jit: WhisperDecoderJit,
    cross_kv_jit: WhisperCrossKvJit,
    prefill_jit: WhisperPrefillJit,
    /// Concrete fixed-capacity step graph. Requests keep stable row ownership;
    /// inactive rows execute with ignored outputs.
    batched_step_jit: WhisperDecoderStepJit,
    tokenizer: WhisperTokenizer,
    options: DecodeOptions,
    n_mels: usize,
    n_audio_state: usize,
    n_vocab: usize,
    n_text_ctx: usize,
    max_batch: usize,
    /// Max concurrent decode lanes in the batched step JIT.
    max_lanes: usize,
    plan: WhisperPlan,
    pos_embedding: Vec<f32>,
}

struct RecognizedWindow {
    result: super::decode::DecodeResult,
    cross_k: svod_device::Buffer,
    cross_v: svod_device::Buffer,
    audio_samples: usize,
}

impl WhisperRecognizer {
    pub fn new(
        model: Whisper,
        tokenizer: WhisperTokenizer,
        options: DecodeOptions,
        max_chunk_samples: usize,
    ) -> Result<Self, TranscribeError> {
        let plan = WhisperPlan::for_recognizer(&model.dims);
        Self::new_with_plan(model, tokenizer, options, max_chunk_samples, plan)
    }

    pub fn new_with_plan(
        model: Whisper,
        tokenizer: WhisperTokenizer,
        options: DecodeOptions,
        max_chunk_samples: usize,
        plan: WhisperPlan,
    ) -> Result<Self, TranscribeError> {
        plan.validate().map_err(|message| TranscribeError::Model {
            source: Box::new(super::error::Error::Decode { msg: message.to_string() }),
        })?;
        options.validate().map_err(|message| TranscribeError::Model {
            source: Box::new(super::error::Error::Decode { msg: message.to_string() }),
        })?;
        if attempt_strategies(&options).into_iter().any(|strategy| strategy_width(strategy) > plan.decoder_slots) {
            return Err(TranscribeError::Model {
                source: Box::new(super::error::Error::Decode {
                    msg: "configured decode beam width exceeds decoder_slots".to_string(),
                }),
            });
        }
        let n_mels = model.dims.n_mels;
        let n_audio_state = model.dims.n_audio_state;
        let n_vocab = model.dims.n_vocab;
        let n_text_ctx = model.dims.n_text_ctx;
        let n_text_head = model.dims.n_text_head;
        if max_chunk_samples > super::config::N_SAMPLES {
            return Err(TranscribeError::Model {
                source: Box::new(super::error::Error::Decode {
                    msg: format!(
                        "Whisper decode windows are limited to {} samples, got {max_chunk_samples}",
                        super::config::N_SAMPLES
                    ),
                }),
            });
        }
        let mel = WhisperMel::new(n_mels);

        let max_batch = plan.encoder_batch;

        let prepare_config = PrepareConfig::from_env();

        // Encoder JIT: [max_batch, n_mels, N_FRAMES], device-local output
        let mut encoder_jit = WhisperEncoderJit::new(model.clone());
        let mel_spec = InputSpec::f32(&[max_batch, n_mels, N_FRAMES]);
        let mut enc_config = prepare_config.clone();
        enc_config.device_local_outputs = true;
        encoder_jit.prepare_with_config(mel_spec, &enc_config)?;

        let n_text_state = model.dims.n_text_state;
        let n_text_layer = model.dims.n_text_layer;
        let d_head = n_text_state / n_text_head;
        let cross_cache_spec = InputSpec::f32(&[1, N_AUDIO_CTX, n_text_layer * n_text_head, d_head]).device_local();

        // Cache-consuming decoder used for language detection.
        let mut decoder_jit = WhisperDecoderJit::new(model.clone());
        let tokens_spec = InputSpec::i32(&[1, 1]);
        decoder_jit.prepare_with_config(
            cross_cache_spec.clone(),
            cross_cache_spec.clone(),
            tokens_spec,
            &prepare_config,
        )?;

        // Cross-attention K/V projection is token-independent. Compile it once
        // and execute it once per encoder window, before any fallback attempts.
        let mut cross_kv_jit = WhisperCrossKvJit::new(model.clone());
        let mut cross_config = prepare_config.clone();
        cross_config.device_local_outputs = true;
        cross_kv_jit.prepare_with_config(InputSpec::f32(&[1, N_AUDIO_CTX, model.dims.n_text_state]), &cross_config)?;

        // Timestamp-enabled prefill has a structural, model-specific prefix:
        // multilingual [SOT, language, task], English-only [SOT].
        // Compiled once at construction, reused every window.
        let init_len = if model.is_multilingual() { 3 } else { 1 };
        let mut prefill_jit = WhisperPrefillJit::new(model.clone());
        prefill_jit.prepare_with_config(
            InputSpec::i32(&[1, init_len]),
            cross_cache_spec.clone(),
            cross_cache_spec,
            &prepare_config,
        )?;

        let n_text_head_local = n_text_head;

        // Fixed concrete batch keeps tensor-core dimensions static and avoids
        // cache movement when lanes finish.
        let max_lanes = plan.decoder_slots;
        let mut batched_step_jit = WhisperDecoderStepJit::new(model.clone());
        batched_step_jit.prepare_with_config(
            InputSpec::i32(&[max_lanes, 1]),
            InputSpec::f32(&[max_lanes, 1, n_text_state]),
            InputSpec::f32(&[max_lanes, N_TEXT_CTX, n_text_layer * n_text_head_local, d_head]).device_local(),
            InputSpec::f32(&[max_lanes, N_TEXT_CTX, n_text_layer * n_text_head_local, d_head]).device_local(),
            InputSpec::f32(&[max_lanes, N_AUDIO_CTX, n_text_layer * n_text_head_local, d_head]).device_local(),
            InputSpec::f32(&[max_lanes, N_AUDIO_CTX, n_text_layer * n_text_head_local, d_head]).device_local(),
            InputSpec::i32(&[max_lanes]),
            &prepare_config,
        )?;

        // Read positional embedding eagerly (static weight, reused every window).
        // Cast to fp32 — the host decode math (pos_embedding slicing in decode.rs)
        // operates on Vec<f32>, while the weight loads at dims.dtype (fp16).
        let pe = model.decoder.positional_embedding.cast(svod_dtype::DType::Float32);
        pe.realize().map_err(|e| TranscribeError::Tensor { source: Box::new(e) })?;
        let pos_embedding = pe
            .as_ndarray::<f32>()
            .map_err(|e| TranscribeError::Tensor { source: Box::new(e) })?
            .as_slice()
            .expect("pos emb")
            .to_vec();

        Ok(Self {
            mel,
            encoder_jit,
            decoder_jit,
            cross_kv_jit,
            prefill_jit,
            batched_step_jit,
            tokenizer,
            options,
            n_mels,
            n_audio_state,
            n_vocab,
            n_text_ctx,
            max_batch,
            max_lanes,
            plan,
            pos_embedding,
        })
    }

    /// Override the decode language (`None` ⇒ auto-detect) for subsequent
    /// [`Transcriber::transcribe_windows`] calls. Lets a reusable transcriber
    /// serve requests with differing languages without rebuilding the JITs.
    pub fn set_language(&mut self, language: Option<String>) {
        self.options.language = language;
    }

    /// Max concurrent decode lanes the batched step JIT was compiled for.
    /// The scheduler treats this as the GPU concurrency bound.
    pub fn max_lanes(&self) -> usize {
        self.max_lanes
    }

    pub fn plan(&self) -> &WhisperPlan {
        &self.plan
    }

    /// Compute mel spectrogram for a window, padded/trimmed to N_FRAMES.
    fn compute_mel(&self, window: &[f32]) -> Vec<f32> {
        let mel = self.mel.compute(window);
        let total = self.n_mels * N_FRAMES;

        let mut padded = vec![0.0f32; total];
        let copy_len = mel.len().min(total);
        padded[..copy_len].copy_from_slice(&mel[..copy_len]);
        padded
    }
}

impl Transcriber for WhisperRecognizer {
    type Error = TranscribeError;

    fn sample_rate(&self) -> u32 {
        SAMPLE_RATE as u32
    }

    fn transcribe_windows(
        &mut self,
        windows: &[&[f32]],
        profile: bool,
    ) -> Result<(Vec<Transcript>, Option<RunProfile>), Self::Error> {
        let mut transcripts = Vec::with_capacity(windows.len());
        let mut profile_result = profile.then(RunProfile::default);
        // Recognized windows own device-resident cross K/V snapshots. Consume
        // them one encoder batch at a time instead of retaining one pair for
        // every window in a long recording.
        for batch in bounded_window_batches(windows, self.max_batch) {
            let (recognized, mut batch_profile, copies) = self.recognize_windows(batch, profile)?;
            transcripts.extend(recognized.into_iter().map(|recognized| {
                let segments = super::decode::split_into_segments(
                    &recognized.result.tokens,
                    &self.tokenizer,
                    recognized.audio_samples as f32 / SAMPLE_RATE as f32,
                );
                Transcript {
                    text: recognized.result.text,
                    words: Vec::new(),
                    segments,
                    language: recognized.result.language,
                }
            }));
            if let Some(batch_profile) = &mut batch_profile {
                for stage in copies.stages() {
                    batch_profile.push(stage);
                }
            }
            if let (Some(profile_result), Some(batch_profile)) = (&mut profile_result, batch_profile) {
                profile_result.merge(batch_profile);
            }
        }
        Ok((transcripts, profile_result))
    }
}

/// Timestamp-enabled recognizer composed with the independent, fixed-shape
/// word aligner. Every call returns DTW-aligned words; there is no feature flag
/// that changes the prepared recognition graph.
pub struct WhisperAlignedTranscriber {
    recognizer: WhisperRecognizer,
    aligner: WhisperAligner,
}

impl WhisperAlignedTranscriber {
    pub fn new(
        model: Whisper,
        tokenizer: WhisperTokenizer,
        options: DecodeOptions,
        size: super::config::WhisperSize,
        max_chunk_samples: usize,
    ) -> Result<Self, TranscribeError> {
        let plan = WhisperPlan::for_model(&model.dims, size);
        Self::new_with_plan(model, tokenizer, options, size, max_chunk_samples, plan)
    }

    pub fn new_with_plan(
        model: Whisper,
        tokenizer: WhisperTokenizer,
        options: DecodeOptions,
        size: super::config::WhisperSize,
        max_chunk_samples: usize,
        plan: WhisperPlan,
    ) -> Result<Self, TranscribeError> {
        plan.validate().map_err(|message| TranscribeError::Model {
            source: Box::new(super::error::Error::Decode { msg: message.to_string() }),
        })?;
        let aligner = WhisperAligner::new(model.clone(), size, plan.alignment_batch)
            .map_err(|error| TranscribeError::Model { source: Box::new(error) })?;
        let recognizer = WhisperRecognizer::new_with_plan(model, tokenizer, options, max_chunk_samples, plan)?;
        Ok(Self { recognizer, aligner })
    }

    pub fn set_language(&mut self, language: Option<String>) {
        self.recognizer.set_language(language);
    }

    fn align_recognized(
        &mut self,
        recognized: Vec<RecognizedWindow>,
        profile: bool,
        copies: &mut CopyProfile,
    ) -> Result<(Vec<Transcript>, AlignmentProfile), TranscribeError> {
        let task = self.recognizer.options.task;
        let tokenizer = &self.recognizer.tokenizer;
        let mut transcripts = Vec::with_capacity(recognized.len());
        let mut alignment_profile = AlignmentProfile::default();
        for chunk in recognized.chunks(self.recognizer.plan.alignment_batch) {
            let inputs: Vec<_> = chunk
                .iter()
                .map(|recognized| WhisperAlignmentInput {
                    cross_k: &recognized.cross_k,
                    cross_v: &recognized.cross_v,
                    decoded_tokens: &recognized.result.tokens,
                    token_probs: &recognized.result.token_probs,
                    language: recognized.result.language.as_deref(),
                    task,
                    audio_samples: recognized.audio_samples,
                })
                .collect();
            let (words, batch_profile) = self
                .aligner
                .align_batch_profiled(&inputs, tokenizer, profile.then_some(&mut *copies))
                .map_err(|error| TranscribeError::Model { source: Box::new(error) })?;
            alignment_profile.merge(batch_profile);
            for (recognized, words) in chunk.iter().zip(words) {
                let segments = super::decode::split_into_segments(
                    &recognized.result.tokens,
                    tokenizer,
                    recognized.audio_samples as f32 / SAMPLE_RATE as f32,
                );
                transcripts.push(Transcript {
                    text: recognized.result.text.clone(),
                    words,
                    segments,
                    language: recognized.result.language.clone(),
                });
            }
        }
        Ok((transcripts, alignment_profile))
    }
}

impl Transcriber for WhisperAlignedTranscriber {
    type Error = TranscribeError;

    fn sample_rate(&self) -> u32 {
        SAMPLE_RATE as u32
    }

    fn transcribe_windows(
        &mut self,
        windows: &[&[f32]],
        profile: bool,
    ) -> Result<(Vec<Transcript>, Option<RunProfile>), Self::Error> {
        let mut transcripts = Vec::with_capacity(windows.len());
        let mut profile_result = profile.then(RunProfile::default);
        for batch in bounded_window_batches(windows, self.recognizer.max_batch) {
            let (recognized, mut batch_profile, mut copies) = self.recognizer.recognize_windows(batch, profile)?;
            let (batch_transcripts, alignment) = self.align_recognized(recognized, profile, &mut copies)?;
            transcripts.extend(batch_transcripts);
            if let Some(batch_profile) = &mut batch_profile {
                batch_profile.push(alignment.graph.stage("alignment_graph"));
                batch_profile.push(StageProfile::host("alignment_cpu_dtw", alignment.cpu_dtw_wall));
                for stage in copies.stages() {
                    batch_profile.push(stage);
                }
            }
            if let (Some(profile_result), Some(batch_profile)) = (&mut profile_result, batch_profile) {
                profile_result.merge(batch_profile);
            }
        }
        Ok((transcripts, profile_result))
    }
}

pub(crate) fn bounded_window_batches<T>(windows: &[T], batch_size: usize) -> impl Iterator<Item = &[T]> {
    windows.chunks(batch_size.max(1))
}

impl WhisperRecognizer {
    fn recognize_windows(
        &mut self,
        windows: &[&[f32]],
        profile: bool,
    ) -> Result<(Vec<RecognizedWindow>, Option<RunProfile>, CopyProfile), TranscribeError> {
        if windows.is_empty() {
            return Ok((Vec::new(), profile.then(RunProfile::default), CopyProfile::default()));
        }

        let n_mels = self.n_mels;
        let d = self.n_audio_state;
        let mel_stride = n_mels * N_FRAMES;
        let item_stride = N_AUDIO_CTX * d;
        let max_batch = self.max_batch;
        let n_vocab = self.n_vocab;
        let n_text_ctx = self.n_text_ctx;

        let mut recognized = Vec::with_capacity(windows.len());
        let mut prof = profile.then(RunProfile::default);
        let mut encoder_profile = GraphProfile::default();
        let mut cross_kv_profile = GraphProfile::default();
        let mut language_profile = GraphProfile::default();
        let mut prefill_profile = GraphProfile::default();
        let mut decoder_step_profile = GraphProfile::default();
        let mut decode_stats = DecodeScheduleStats::default();
        let mut copies = CopyProfile::default();
        let (mut t_mel, mut t_decoder_scheduler) = (Duration::ZERO, Duration::ZERO);

        for batch_start in (0..windows.len()).step_by(max_batch) {
            let b = (windows.len() - batch_start).min(max_batch);

            // ── Mel: compute + pack into [b, n_mels, N_FRAMES] ──────────────
            let t = Instant::now();
            let batch_mels: Vec<Vec<f32>> = (0..b).map(|bi| self.compute_mel(windows[batch_start + bi])).collect();
            {
                let mel_buf = self.encoder_jit.mel_mut()?;
                let mut packed = vec![0f32; max_batch * mel_stride];
                for bi in 0..b {
                    packed[bi * mel_stride..(bi + 1) * mel_stride].copy_from_slice(&batch_mels[bi][..mel_stride]);
                }
                let copy_started = begin_host_copy(profile, mel_buf)
                    .map_err(|source| TranscribeError::Model { source: Box::new(source) })?;
                let dst = mel_buf.as_host_bytes_mut()?;
                let src_bytes: &[u8] = bytemuck::cast_slice(&packed);
                dst[..src_bytes.len()].copy_from_slice(src_bytes);
                if let Some(started) = copy_started {
                    copies.h2d("mel_input", 1, src_bytes.len(), started.elapsed());
                }
            }
            t_mel += t.elapsed();

            // ── Encode: one dispatch for b windows ───────────────────────────
            if profile {
                let graph_started = Instant::now();
                let kernels = self.encoder_jit.execute_profiled_static()?;
                self.encoder_jit.output()?.synchronize()?;
                encoder_profile.record(graph_started.elapsed(), kernels);
            } else {
                self.encoder_jit.execute()?;
            }

            let out_buf = self.encoder_jit.output()?;

            let mut seeds = Vec::with_capacity(b);
            let mut decode_options = Vec::with_capacity(b);

            // Cross projection and token prefill remain per-window. Their
            // immutable device-local seeds are retained until the shared
            // scheduler accepts a primary or fallback attempt.
            for bi in 0..b {
                let base = bi * item_stride;

                // Project encoder features once for all fallback prefills.
                let (_, projection_wall) = timed_d2d(profile, out_buf, || {
                    let buf = self.cross_kv_jit.audio_features_mut()?;
                    buf.copy_region_from(
                        0,
                        out_buf,
                        base * std::mem::size_of::<f32>(),
                        item_stride * std::mem::size_of::<f32>(),
                    )?;
                    Ok(())
                })?;
                if profile {
                    copies.d2d("projection_input", 1, item_stride * std::mem::size_of::<f32>(), projection_wall);
                }
                if profile {
                    let graph_started = Instant::now();
                    let kernels = self.cross_kv_jit.execute_profiled_static()?;
                    self.cross_kv_jit.cross_k()?.synchronize()?;
                    cross_kv_profile.record(graph_started.elapsed(), kernels);
                } else {
                    self.cross_kv_jit.execute()?;
                }
                let cross_k_fence = self.cross_kv_jit.cross_k()?.clone();
                let mut fanout_bytes = 0usize;
                let (_, fanout_wall) = timed_d2d(profile, &cross_k_fence, || {
                    let src = self.cross_kv_jit.cross_k()?;
                    fanout_bytes = fanout_bytes.saturating_add(src.size().saturating_mul(2));
                    self.prefill_jit.prepared_cross_k_mut()?.copy_region_from(0, src, 0, src.size())?;
                    self.decoder_jit.prepared_cross_k_mut()?.copy_region_from(0, src, 0, src.size())?;
                    let src = self.cross_kv_jit.cross_v()?;
                    fanout_bytes = fanout_bytes.saturating_add(src.size().saturating_mul(2));
                    self.prefill_jit.prepared_cross_v_mut()?.copy_region_from(0, src, 0, src.size())?;
                    self.decoder_jit.prepared_cross_v_mut()?.copy_region_from(0, src, 0, src.size())?;
                    Ok(())
                })?;
                if profile {
                    copies.d2d("cross_fanout", 4, fanout_bytes, fanout_wall);
                }

                let mut options = self.options.clone();
                if !self.tokenizer.multilingual {
                    options.language = Some("en".to_string());
                } else if options.language.is_none() {
                    let detection = detect_language_profile(
                        &mut self.decoder_jit,
                        n_vocab,
                        &self.tokenizer,
                        profile.then_some(&mut copies),
                        profile.then_some(&mut language_profile),
                    )
                    .map_err(|error| TranscribeError::Model { source: Box::new(error) })?;
                    options.language = Some(detection.language);
                }

                let seed = prefill_decode_seed(
                    &mut self.prefill_jit,
                    &self.tokenizer,
                    &options,
                    n_text_ctx,
                    n_vocab,
                    &self.pos_embedding,
                    self.n_audio_state,
                    profile.then_some(&mut copies),
                    profile.then_some(&mut prefill_profile),
                )
                .map_err(|error| TranscribeError::Model { source: Box::new(error) })?;
                seeds.push(seed);
                decode_options.push(options);
            }

            let t = Instant::now();
            let (results, batch_stats, batch_graph_profile) = run_fixed_slot_decode(
                &seeds,
                &decode_options,
                &mut self.batched_step_jit,
                self.max_lanes,
                &self.tokenizer,
                n_text_ctx,
                n_vocab,
                profile,
            )
            .map_err(|error| TranscribeError::Model { source: Box::new(error) })?;
            decode_stats.merge(batch_stats);
            decoder_step_profile.merge(batch_graph_profile);
            t_decoder_scheduler += t.elapsed();

            for (bi, ((mut result, options), seed)) in results.into_iter().zip(&decode_options).zip(seeds).enumerate() {
                if result.should_skip(options) {
                    result.clear_speech();
                }
                let (cross_k, cross_v) = seed.into_cross_kv();
                recognized.push(RecognizedWindow {
                    result,
                    cross_k,
                    cross_v,
                    audio_samples: windows[batch_start + bi].len(),
                });
            }
        }

        if let Some(p) = &mut prof {
            let encoder_executions = encoder_profile.executions;
            p.push(StageProfile::host("mel", t_mel));
            p.push(encoder_profile.stage("encoder"));
            p.push(cross_kv_profile.stage("cross_kv_projection"));
            if language_profile.executions != 0 {
                p.push(language_profile.stage("language_detection"));
            }
            p.push(prefill_profile.stage("prefill"));
            let mut decoder_step = decoder_step_profile.stage("decoder_step");
            decoder_step.meta.insert("dispatches".into(), decode_stats.dispatches.to_string());
            decoder_step.meta.insert("dispatch_semantics".into(), "logical decoder graph executions".into());
            decoder_step.meta.insert("active_row_steps".into(), decode_stats.active_row_steps.to_string());
            decoder_step.meta.insert("reserved_row_steps".into(), decode_stats.reserved_row_steps.to_string());
            decoder_step.meta.insert("capacity_row_steps".into(), decode_stats.capacity_row_steps.to_string());
            decoder_step.meta.insert("cache_clone_ops".into(), decode_stats.cache_clone_ops.to_string());
            decoder_step.meta.insert("cache_clone_bytes".into(), decode_stats.cache_clone_bytes.to_string());
            decoder_step.meta.insert("attempts".into(), decode_stats.attempts.to_string());
            decoder_step.meta.insert("fallback_attempts".into(), decode_stats.fallback_attempts.to_string());
            let utilization = if decode_stats.capacity_row_steps == 0 {
                0.0
            } else {
                decode_stats.active_row_steps as f64 / decode_stats.capacity_row_steps as f64
            };
            decoder_step.meta.insert("row_utilization".into(), format!("{utilization:.4}"));
            p.push(decoder_step);
            let mut scheduler = StageProfile::host("decoder_scheduler_total", t_decoder_scheduler);
            scheduler.meta.insert(
                "timing_semantics".into(),
                "non-additive end-to-end control-loop wall including host work, decoder_step waits, and synchronized cache/control copies".into(),
            );
            scheduler.meta.insert("batches".into(), encoder_executions.to_string());
            p.push(scheduler);
        }

        copies.merge(decode_stats.copies.clone());
        Ok((recognized, prof, copies))
    }
}
