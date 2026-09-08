//! Fixed-shape teacher-forced decoder alignment and host-side DTW.

use std::time::{Duration, Instant};

use crate::jit::InputSpec;

use super::config::{HOP_LENGTH, N_AUDIO_CTX, N_FRAMES, N_TEXT_CTX, TOKENS_PER_SECOND, WhisperSize};
use super::decode::WhisperTask;
use super::dtw::{find_alignment_path_selected, path_to_word_timings};
use super::error::Result;

use super::jit::{WhisperAlignmentJit, WhisperAlignmentModel};
use super::model::Whisper;
use super::profile::{CopyProfile, GraphProfile, begin_host_copy, timed_d2d};
use super::tokenizer::WhisperTokenizer;
use super::transcribe::Word;

/// Prepared alignment stage. Its graph is fully static and is replayed once
/// for each finalized recognition result.
pub struct WhisperAligner {
    jit: WhisperAlignmentJit,
    n_heads: usize,
    batch_size: usize,
    cache_stride: usize,
}

/// Inputs for one lane of a prepared alignment batch.
pub struct WhisperAlignmentInput<'a> {
    /// Device-resident packed cross-attention K cache for one recognition window.
    pub cross_k: &'a svod_device::Buffer,
    /// Device-resident packed cross-attention V cache for one recognition window.
    pub cross_v: &'a svod_device::Buffer,
    /// Recognition tokens, including timestamp tokens when emitted.
    pub decoded_tokens: &'a [u32],
    /// Decoder probability corresponding to each decoded token.
    pub token_probs: &'a [f32],
    /// Resolved language code used to reconstruct the decoder prompt.
    pub language: Option<&'a str>,
    /// Decoder task used to reconstruct the prompt.
    pub task: WhisperTask,
    /// Unpadded source-audio length in samples.
    pub audio_samples: usize,
}

#[derive(Debug, Default)]
pub(crate) struct AlignmentProfile {
    pub(crate) graph: GraphProfile,
    pub(crate) cpu_dtw_wall: Duration,
}

impl AlignmentProfile {
    pub(crate) fn merge(&mut self, other: Self) {
        self.graph.merge(other.graph);
        self.cpu_dtw_wall = self.cpu_dtw_wall.saturating_add(other.cpu_dtw_wall);
    }
}

impl WhisperAligner {
    pub fn new(model: Whisper, size: WhisperSize, batch_size: usize) -> Result<Self> {
        if batch_size == 0 {
            return Err(super::error::Error::Decode { msg: "alignment batch must be non-zero".to_string() });
        }
        let heads = size.alignment_heads().to_vec();
        let n_state = model.dims.n_text_state;
        let n_layer_heads = model.dims.n_text_layer * model.dims.n_text_head;
        let d_head = n_state / model.dims.n_text_head;
        let alignment_model = WhisperAlignmentModel::new(model, heads.clone());
        let mut jit = WhisperAlignmentJit::new(alignment_model);
        let cache_spec = InputSpec::f32(&[batch_size, N_AUDIO_CTX, n_layer_heads, d_head]).device_local();
        jit.prepare(cache_spec.clone(), cache_spec, InputSpec::i32(&[batch_size, N_TEXT_CTX]))?;
        Ok(Self { jit, n_heads: heads.len(), batch_size, cache_stride: N_AUDIO_CTX * n_layer_heads * d_head })
    }

    /// Align up to the concrete batch capacity prepared at construction.
    pub fn align_batch(
        &mut self,
        inputs: &[WhisperAlignmentInput<'_>],
        tokenizer: &WhisperTokenizer,
    ) -> Result<Vec<Vec<Word>>> {
        self.align_batch_profiled(inputs, tokenizer, None).map(|(words, _)| words)
    }

    pub(crate) fn align_batch_profiled(
        &mut self,
        inputs: &[WhisperAlignmentInput<'_>],
        tokenizer: &WhisperTokenizer,
        mut copies: Option<&mut CopyProfile>,
    ) -> Result<(Vec<Vec<Word>>, AlignmentProfile)> {
        if inputs.len() > self.batch_size {
            return Err(super::error::Error::Decode {
                msg: format!("alignment input {} exceeds prepared batch {}", inputs.len(), self.batch_size),
            });
        }
        if inputs.is_empty() {
            return Ok((Vec::new(), AlignmentProfile::default()));
        }

        let cache_bytes = self.cache_stride * std::mem::size_of::<f32>();
        let (_, packing_wall) = timed_d2d(copies.is_some(), inputs[0].cross_k, || {
            {
                let packed_k = self.jit.cross_k_mut()?;
                for (lane, input) in inputs.iter().enumerate() {
                    if input.cross_k.dtype() != svod_dtype::DType::Float32
                        || input.cross_k.size() != cache_bytes
                        || !std::ptr::eq(packed_k.allocator(), input.cross_k.allocator())
                    {
                        return Err(super::error::Error::Decode {
                            msg: "alignment cross K has invalid dtype, size, or allocator".to_string(),
                        });
                    }
                    packed_k.copy_region_from(lane * cache_bytes, input.cross_k, 0, cache_bytes)?;
                }
            }
            {
                let packed_v = self.jit.cross_v_mut()?;
                for (lane, input) in inputs.iter().enumerate() {
                    if input.cross_v.dtype() != svod_dtype::DType::Float32
                        || input.cross_v.size() != cache_bytes
                        || !std::ptr::eq(packed_v.allocator(), input.cross_v.allocator())
                        || !std::ptr::eq(input.cross_k.allocator(), input.cross_v.allocator())
                    {
                        return Err(super::error::Error::Decode {
                            msg: "alignment cross V has invalid dtype, size, or allocator".to_string(),
                        });
                    }
                    packed_v.copy_region_from(lane * cache_bytes, input.cross_v, 0, cache_bytes)?;
                }
            }
            Ok(())
        })?;
        if let Some(copies) = copies.as_deref_mut() {
            copies.d2d("alignment_packing", inputs.len() * 2, inputs.len() * cache_bytes * 2, packing_wall);
        }

        let mut packed_tokens = vec![tokenizer.eot() as i32; self.batch_size * N_TEXT_CTX];
        let mut metadata = Vec::with_capacity(inputs.len());
        for (lane, input) in inputs.iter().enumerate() {
            let mut text_tokens: Vec<u32> =
                input.decoded_tokens.iter().copied().filter(|&token| token < tokenizer.eot()).collect();
            let mut token_probs = input.token_probs[..input.token_probs.len().min(text_tokens.len())].to_vec();
            let mut tokens = vec![tokenizer.sot()];
            if tokenizer.multilingual {
                let language = input.language.unwrap_or("en");
                tokens.push(tokenizer.language_token_for(language).unwrap_or_else(|| tokenizer.sot()));
                tokens.push(match input.task {
                    WhisperTask::Transcribe => tokenizer.transcribe(),
                    WhisperTask::Translate => tokenizer.translate(),
                });
            }
            let sot_len = tokens.len();
            text_tokens.truncate(N_TEXT_CTX - sot_len - 2);
            token_probs.truncate(text_tokens.len());
            tokens.push(tokenizer.no_timestamps());
            tokens.extend_from_slice(&text_tokens);
            tokens.push(tokenizer.eot());
            let valid_text = tokens.len();
            for (index, token) in tokens.into_iter().enumerate() {
                packed_tokens[lane * N_TEXT_CTX + index] = token as i32;
            }
            metadata.push((text_tokens, token_probs, valid_text, sot_len));
        }

        let token_buffer = self.jit.tokens_mut()?;
        let token_started = begin_host_copy(copies.is_some(), token_buffer)?;
        token_buffer.as_host_bytes_mut()?.copy_from_slice(bytemuck::cast_slice(&packed_tokens));
        if let (Some(copies), Some(started)) = (copies.as_deref_mut(), token_started) {
            copies.h2d("alignment_tokens", 1, packed_tokens.len() * std::mem::size_of::<i32>(), started.elapsed());
        }
        let profiling = copies.is_some();
        let (graph_wall, kernels) = if profiling {
            let graph_started = Instant::now();
            let kernels = self.jit.execute_profiled_static()?;
            self.jit.output()?.synchronize()?;
            (graph_started.elapsed(), kernels)
        } else {
            self.jit.execute()?;
            (Duration::ZERO, Vec::new())
        };
        let output = self.jit.output()?;
        let output_started = begin_host_copy(copies.is_some(), output)?;
        let output_bytes = output.as_host_bytes()?;
        let qk_stride = self.n_heads * N_TEXT_CTX * N_AUDIO_CTX;
        let active_qk_bytes = inputs.len() * qk_stride * std::mem::size_of::<f32>();
        let profiled_qk = output_started.map(|_| bytemuck::cast_slice(&output_bytes[..active_qk_bytes]).to_vec());
        let qk: &[f32] = profiled_qk.as_deref().unwrap_or_else(|| bytemuck::cast_slice(output_bytes));
        if let (Some(copies), Some(started)) = (copies, output_started) {
            copies.d2h("alignment_qk", 1, active_qk_bytes, started.elapsed());
        }

        let cpu_started = Instant::now();
        let words = inputs
            .iter()
            .zip(metadata)
            .enumerate()
            .map(|(lane, (input, (text_tokens, token_probs, valid_text, sot_len)))| {
                let lane_qk = &qk[lane * qk_stride..(lane + 1) * qk_stride];
                let valid_audio = (input.audio_samples / HOP_LENGTH).min(N_FRAMES) / 2;
                let (text_indices, time_indices) = find_alignment_path_selected(
                    lane_qk,
                    self.n_heads,
                    N_TEXT_CTX,
                    N_AUDIO_CTX,
                    valid_text,
                    valid_audio,
                    7,
                    sot_len,
                );
                words_from_path(&text_indices, &time_indices, &text_tokens, &token_probs, input.language, tokenizer)
            })
            .collect();
        let mut graph = GraphProfile::default();
        if profiling {
            graph.record(graph_wall, kernels);
        }
        Ok((words, AlignmentProfile { graph, cpu_dtw_wall: cpu_started.elapsed() }))
    }
}

pub(crate) fn words_from_path(
    text_indices: &[usize],
    time_indices: &[usize],
    text_tokens: &[u32],
    token_probs: &[f32],
    language: Option<&str>,
    tokenizer: &WhisperTokenizer,
) -> Vec<Word> {
    let (word_strings, word_token_lists) = tokenizer.split_to_word_tokens_for_language(text_tokens, language);
    let mut word_boundaries = vec![0usize];
    for tokens in &word_token_lists {
        word_boundaries.push(word_boundaries.last().copied().unwrap() + tokens.len());
    }
    let mut timings = path_to_word_timings(
        text_indices,
        time_indices,
        &word_boundaries,
        &word_strings,
        &word_token_lists,
        token_probs,
        TOKENS_PER_SECOND,
    );
    refine_word_timings(&mut timings);
    timings
        .into_iter()
        .filter(|word| !word.word.trim().is_empty())
        .map(|word| Word { text: word.word, start: word.start, end: word.end })
        .collect()
}

fn refine_word_timings(words: &mut [super::dtw::WordTiming]) {
    let mut durations: Vec<f32> =
        words.iter().map(|word| word.end - word.start).filter(|&duration| duration > 0.0).collect();
    durations.sort_by(|a, b| a.total_cmp(b));
    let median = match durations.len() {
        0 => 0.0,
        len if len % 2 == 0 => (durations[len / 2 - 1] + durations[len / 2]) / 2.0,
        len => durations[len / 2],
    }
    .min(0.7);
    let max_duration = median * 2.0;
    if max_duration > 0.0 {
        const SENTENCE_END: &str = ".。!！?？";
        for index in 1..words.len() {
            if words[index].end - words[index].start > max_duration {
                if SENTENCE_END.contains(words[index].word.as_str()) {
                    words[index].end = words[index].start + max_duration;
                } else if SENTENCE_END.contains(words[index - 1].word.as_str()) {
                    words[index].start = words[index].end - max_duration;
                }
            }
        }
    }

    const PREPEND: &str = "\"'“¿([{-";
    const APPEND: &str = "\"'.。,，!！?？:：”)]}、";
    let mut following = words.len().saturating_sub(1);
    for previous in (0..words.len().saturating_sub(1)).rev() {
        if words[previous].word.starts_with(' ') && PREPEND.contains(words[previous].word.trim()) {
            let prefix = std::mem::take(&mut words[previous].word);
            words[following].word.insert_str(0, &prefix);
            let mut tokens = std::mem::take(&mut words[previous].tokens);
            tokens.append(&mut words[following].tokens);
            words[following].tokens = tokens;
        } else {
            following = previous;
        }
    }

    let mut previous = 0;
    for following in 1..words.len() {
        if !words[previous].word.ends_with(' ') && APPEND.contains(words[following].word.as_str()) {
            let suffix = std::mem::take(&mut words[following].word);
            words[previous].word.push_str(&suffix);
            let tokens = std::mem::take(&mut words[following].tokens);
            words[previous].tokens.extend(tokens);
        } else {
            previous = following;
        }
    }
}
