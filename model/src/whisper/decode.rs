//! Scheduled Whisper decoding, temperature fallback, and language detection.

use super::error::{DeviceSnafu, Error, JitSnafu, Result};
use super::jit::{WhisperDecoderJit, WhisperDecoderStepJit, WhisperPrefillJit};
use super::profile::{CopyProfile, GraphProfile, begin_host_copy, timed_d2d};
use super::tokenizer::WhisperTokenizer;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use snafu::ResultExt;
use std::cmp::Ordering;
use std::collections::VecDeque;
use svod_arch::pipelines::audio::Segment;
use svod_device::{Buffer, BufferSpec};
use svod_dtype::DType;

// ─── Language detection ─────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct LanguageDetection {
    pub language: String,
    pub language_token: u32,
    pub probabilities: Vec<(String, f32)>,
}

pub fn detect_language(
    decoder_jit: &mut WhisperDecoderJit,
    _n_text_ctx: usize,
    n_vocab: usize,
    tokenizer: &WhisperTokenizer,
) -> Result<LanguageDetection> {
    detect_language_profile(decoder_jit, n_vocab, tokenizer, None, None)
}

pub(crate) fn detect_language_profile(
    decoder_jit: &mut WhisperDecoderJit,
    n_vocab: usize,
    tokenizer: &WhisperTokenizer,
    mut copies: Option<&mut CopyProfile>,
    graph_profile: Option<&mut GraphProfile>,
) -> Result<LanguageDetection> {
    let sot = tokenizer.sot() as i32;
    let started = begin_host_copy(copies.is_some(), decoder_jit.tokens_mut().context(JitSnafu)?)?;
    write_uncached(decoder_jit, &[sot])?;
    if let (Some(copies), Some(started)) = (copies.as_deref_mut(), started) {
        copies.h2d("language_tokens", 1, std::mem::size_of::<i32>(), started.elapsed());
    }
    if let Some(graph_profile) = graph_profile {
        let graph_started = std::time::Instant::now();
        let kernels = decoder_jit.execute_profiled_static().context(JitSnafu)?;
        decoder_jit.output().context(JitSnafu)?.synchronize().context(DeviceSnafu)?;
        graph_profile.record(graph_started.elapsed(), kernels);
    } else {
        decoder_jit.execute().context(JitSnafu)?;
    }
    let started = begin_host_copy(copies.is_some(), decoder_jit.output().context(JitSnafu)?)?;
    let sot_logits = read_uncached(decoder_jit, n_vocab)?;
    if let (Some(copies), Some(started)) = (copies, started) {
        copies.d2h("language_logits", 1, n_vocab * std::mem::size_of::<f32>(), started.elapsed());
    }

    let lang_tokens = tokenizer.all_language_tokens();
    let lang_codes = tokenizer.all_language_codes();
    let mut masked = vec![f32::NEG_INFINITY; n_vocab];
    for &tok in &lang_tokens {
        masked[tok as usize] = sot_logits[tok as usize];
    }
    let best_tok = argmax(&masked) as u32;
    let max_val = masked.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let sum: f32 = masked.iter().map(|&l| (l - max_val).exp()).sum();
    let logsum = sum.ln() + max_val;

    let mut probabilities: Vec<(String, f32)> = lang_tokens
        .iter()
        .zip(&lang_codes)
        .map(|(&tok, code)| ((masked[tok as usize] - logsum).exp(), code.clone()))
        .map(|(p, c)| (c, p))
        .collect();
    probabilities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let language = tokenizer.code_for_token(best_tok).unwrap_or_else(|| "en".into());
    Ok(LanguageDetection { language, language_token: best_tok, probabilities })
}

// ─── Decode options & result ────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WhisperTask {
    Transcribe,
    Translate,
}

impl std::str::FromStr for WhisperTask {
    type Err = &'static str;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value {
            "transcribe" => Ok(Self::Transcribe),
            "translate" => Ok(Self::Translate),
            _ => Err("expected `transcribe` or `translate`"),
        }
    }
}

/// Search algorithm for the first decode attempt.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DecodeStrategy {
    /// Deterministic token-by-token argmax.
    Greedy,
    /// Beam search with a concrete number of decoder rows.
    Beam { size: usize },
    /// Multinomial sampling at a positive temperature.
    Sample { temperature: f32 },
}

impl DecodeStrategy {
    fn temperature(self) -> f32 {
        match self {
            Self::Greedy | Self::Beam { .. } => 0.0,
            Self::Sample { temperature } => temperature,
        }
    }
}

/// Quality-gated sampling attempts after the primary decode is rejected.
#[derive(Clone, Debug, PartialEq)]
pub struct FallbackPolicy {
    /// Positive sampling temperatures tried in order.
    pub sampling_temperatures: Vec<f32>,
    /// Retry when text compression exceeds this threshold.
    pub compression_ratio_threshold: Option<f32>,
    /// Retry below this average log-probability.
    pub logprob_threshold: Option<f32>,
}

impl Default for FallbackPolicy {
    fn default() -> Self {
        Self {
            sampling_temperatures: vec![0.2, 0.4, 0.6, 0.8, 1.0],
            compression_ratio_threshold: Some(2.4),
            logprob_threshold: Some(-1.0),
        }
    }
}

#[derive(Clone, Debug)]
pub struct DecodeOptions {
    /// Whether to transcribe source speech or translate it to English.
    pub task: WhisperTask,
    /// Source language code, or `None` for automatic detection.
    pub language: Option<String>,
    /// Search algorithm for the first decode attempt.
    pub strategy: DecodeStrategy,
    /// Optional quality-gated sampling retries.
    pub fallback: Option<FallbackPolicy>,
    /// Base seed for reproducible per-request sampling streams.
    pub sampling_seed: Option<u64>,
    /// Maximum generated token count; defaults to half the text context.
    pub sample_len: Option<usize>,
    /// Suppress blank/space as the first generated token.
    pub suppress_blank: bool,
    /// Token IDs to suppress; `-1` expands to Whisper's non-speech set.
    pub suppress_tokens: Option<Vec<i32>>,
    /// Latest timestamp permitted at the beginning of a window, in seconds.
    pub max_initial_timestamp: Option<f32>,
    /// Skip likely silence when no-speech probability exceeds this threshold.
    pub no_speech_threshold: Option<f32>,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self {
            task: WhisperTask::Transcribe,
            language: None,
            strategy: DecodeStrategy::Beam { size: 5 },
            fallback: Some(FallbackPolicy::default()),
            sampling_seed: None,
            sample_len: None,
            suppress_blank: true,
            suppress_tokens: Some(vec![-1]),
            max_initial_timestamp: Some(1.0),
            no_speech_threshold: Some(0.6),
        }
    }
}

impl DecodeOptions {
    /// Validate strategy geometry and sampling parameters before graph preparation.
    pub fn validate(&self) -> std::result::Result<(), &'static str> {
        match self.strategy {
            DecodeStrategy::Beam { size: 0 } => return Err("beam size must be non-zero"),
            DecodeStrategy::Sample { temperature } if !valid_temperature(temperature) => {
                return Err("sampling temperature must be finite and positive");
            }
            _ => {}
        }
        if let Some(fallback) = &self.fallback {
            if fallback.sampling_temperatures.is_empty() {
                return Err("fallback sampling temperatures must be non-empty");
            }
            if fallback.sampling_temperatures.iter().any(|&temperature| !valid_temperature(temperature)) {
                return Err("fallback sampling temperatures must be finite and positive");
            }
            if fallback.compression_ratio_threshold.is_some_and(|threshold| !threshold.is_finite() || threshold <= 0.0)
            {
                return Err("compression ratio threshold must be finite and positive");
            }
            if fallback.logprob_threshold.is_some_and(|threshold| !threshold.is_finite()) {
                return Err("log-probability threshold must be finite");
            }
        }
        if self.no_speech_threshold.is_some_and(|threshold| !threshold.is_finite() || !(0.0..=1.0).contains(&threshold))
        {
            return Err("no-speech threshold must be between zero and one");
        }
        Ok(())
    }
}

fn valid_temperature(temperature: f32) -> bool {
    temperature.is_finite() && temperature > 0.0
}

#[cfg(test)]
pub(crate) fn remaining_sample_steps(sample_len: usize) -> usize {
    sample_len.saturating_sub(1)
}

#[derive(Clone, Debug)]
pub struct DecodeResult {
    pub tokens: Vec<u32>,
    pub token_probs: Vec<f32>,
    pub text: String,
    pub avg_logprob: f32,
    pub no_speech_prob: f32,
    /// Sampling temperature of the accepted attempt; zero for greedy or beam.
    pub temperature: f32,
    pub compression_ratio: f32,
    pub language: Option<String>,
}

impl DecodeResult {
    pub fn should_skip(&self, options: &DecodeOptions) -> bool {
        let Some(no_speech_threshold) = options.no_speech_threshold else {
            return false;
        };
        if self.no_speech_prob <= no_speech_threshold {
            return false;
        }
        options
            .fallback
            .as_ref()
            .and_then(|fallback| fallback.logprob_threshold)
            .is_none_or(|threshold| self.avg_logprob <= threshold)
    }

    pub fn clear_speech(&mut self) {
        self.tokens.clear();
        self.token_probs.clear();
        self.text.clear();
    }
}

pub(crate) fn check_fallback(result: &DecodeResult, fallback: &FallbackPolicy, options: &DecodeOptions) -> bool {
    let repetitive = fallback.compression_ratio_threshold.is_some_and(|threshold| result.compression_ratio > threshold);
    let low_confidence = fallback.logprob_threshold.is_some_and(|threshold| result.avg_logprob < threshold);
    let silence =
        options.no_speech_threshold.is_some_and(|threshold| result.no_speech_prob > threshold) && low_confidence;
    (repetitive || low_confidence) && !silence
}

// ─── Fixed-slot mixed-strategy scheduler ────────────────────────────────────

/// Immutable output of token prefill. Fallback attempts reuse this seed rather
/// than rerunning prefill. Cache snapshots remain device-local and are copied
/// into a row only when that row changes request ownership.
pub(crate) struct DecodeSeed {
    pub(crate) metadata: PrefillMetadata,
    pub(crate) self_k_cache: Buffer,
    pub(crate) self_v_cache: Buffer,
    pub(crate) cross_k: Buffer,
    pub(crate) cross_v: Buffer,
    pub(crate) per_pos_bytes: usize,
    pub(crate) self_cache_bytes: usize,
    pub(crate) cross_cache_bytes: usize,
    pub(crate) self_positions: usize,
    pub(crate) cross_positions: usize,
}

impl DecodeSeed {
    /// Discard the attempt-local self cache and retain the immutable cross cache.
    pub(crate) fn into_cross_kv(self) -> (Buffer, Buffer) {
        (self.cross_k, self.cross_v)
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prefill_decode_seed(
    prefill_jit: &mut WhisperPrefillJit,
    tokenizer: &WhisperTokenizer,
    options: &DecodeOptions,
    n_text_ctx: usize,
    n_vocab: usize,
    pos_embedding: &[f32],
    n_state: usize,
    mut copies: Option<&mut CopyProfile>,
    graph_profile: Option<&mut GraphProfile>,
) -> Result<DecodeSeed> {
    let metadata = execute_prefill(
        prefill_jit,
        tokenizer,
        options,
        n_vocab,
        pos_embedding,
        n_state,
        copies.as_deref_mut(),
        graph_profile,
    )?;
    let self_k_src = prefill_jit.self_k().context(JitSnafu)?.clone();
    let self_v_src = prefill_jit.self_v().context(JitSnafu)?.clone();
    let cross_k_src = prefill_jit.prepared_cross_k_mut().context(JitSnafu)?.clone();
    let cross_v_src = prefill_jit.prepared_cross_v_mut().context(JitSnafu)?.clone();
    let bytes = self_k_src
        .size()
        .saturating_add(self_v_src.size())
        .saturating_add(cross_k_src.size())
        .saturating_add(cross_v_src.size());
    let ((self_k, self_v, cross_k, cross_v), wall) = timed_d2d(copies.is_some(), &self_k_src, || {
        Ok((
            clone_device_cache(&self_k_src)?,
            clone_device_cache(&self_v_src)?,
            clone_device_cache(&cross_k_src)?,
            clone_device_cache(&cross_v_src)?,
        ))
    })?;
    if let Some(copies) = copies {
        copies.d2d("seed_snapshots", 4, bytes, wall);
    }
    let seed = build_decode_seed(metadata, self_k, self_v, cross_k, cross_v)?;
    if seed.self_positions > n_text_ctx {
        return Err(decode_err("prefill self cache exceeds text context"));
    }
    Ok(seed)
}

pub(crate) fn clone_device_cache(src: &Buffer) -> Result<Buffer> {
    if src.dtype() != DType::Float32 || !src.size().is_multiple_of(std::mem::size_of::<f32>()) {
        return Err(decode_err("prefill cache must contain aligned float32 data"));
    }
    let mut clone = Buffer::allocate(
        src.allocator_arc(),
        DType::Float32,
        vec![src.size() / std::mem::size_of::<f32>()],
        BufferSpec { cpu_access: false, ..BufferSpec::default() },
    )
    .context(DeviceSnafu)?;
    clone.copy_from(src).context(DeviceSnafu)?;
    Ok(clone)
}

pub(crate) fn build_decode_seed(
    metadata: PrefillMetadata,
    self_k: Buffer,
    self_v: Buffer,
    cross_k: Buffer,
    cross_v: Buffer,
) -> Result<DecodeSeed> {
    if metadata.init_len == 0 || self_k.size() == 0 || self_k.size() != self_v.size() {
        return Err(decode_err("invalid prefill self-cache geometry"));
    }
    if cross_k.size() == 0 || cross_k.size() != cross_v.size() {
        return Err(decode_err("invalid prefill cross-cache geometry"));
    }
    for cache in [&self_v, &cross_k, &cross_v] {
        if !std::ptr::eq(self_k.allocator(), cache.allocator()) {
            return Err(decode_err("prefill caches use different allocators"));
        }
    }
    let per_pos_bytes = self_k
        .size()
        .checked_div(metadata.init_len)
        .filter(|&bytes| bytes != 0 && bytes.checked_mul(metadata.init_len) == Some(self_k.size()))
        .ok_or_else(|| decode_err("self cache is not position-aligned"))?;
    if per_pos_bytes % std::mem::size_of::<f32>() != 0 {
        return Err(decode_err("self cache position is not float32-aligned"));
    }
    let cross_positions = cross_k
        .size()
        .checked_div(per_pos_bytes)
        .filter(|&positions| positions != 0 && positions.checked_mul(per_pos_bytes) == Some(cross_k.size()))
        .ok_or_else(|| decode_err("cross cache is not position-aligned"))?;
    let self_cache_bytes = self_k.size();
    let cross_cache_bytes = cross_k.size();

    Ok(DecodeSeed {
        self_k_cache: self_k,
        self_v_cache: self_v,
        cross_k,
        cross_v,
        per_pos_bytes,
        self_cache_bytes,
        cross_cache_bytes,
        self_positions: metadata.init_len,
        cross_positions,
        metadata,
    })
}

pub(crate) fn strategy_width(strategy: DecodeStrategy) -> usize {
    match strategy {
        DecodeStrategy::Beam { size } => size,
        DecodeStrategy::Greedy | DecodeStrategy::Sample { .. } => 1,
    }
}

pub(crate) fn attempt_strategies(options: &DecodeOptions) -> Vec<DecodeStrategy> {
    let mut strategies = vec![options.strategy];
    if let Some(fallback) = &options.fallback {
        strategies
            .extend(fallback.sampling_temperatures.iter().map(|&temperature| DecodeStrategy::Sample { temperature }));
    }
    strategies
}

pub(crate) fn collect_ordered<T>(results: Vec<Option<T>>) -> std::result::Result<Vec<T>, &'static str> {
    results.into_iter().map(|result| result.ok_or("missing scheduled result")).collect()
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct DecodeScheduleStats {
    pub(crate) dispatches: usize,
    pub(crate) active_row_steps: usize,
    pub(crate) reserved_row_steps: usize,
    pub(crate) capacity_row_steps: usize,
    pub(crate) cache_clone_ops: usize,
    pub(crate) cache_clone_bytes: usize,
    pub(crate) attempts: usize,
    pub(crate) fallback_attempts: usize,
    pub(crate) copies: CopyProfile,
}

impl DecodeScheduleStats {
    pub(crate) fn merge(&mut self, other: Self) {
        self.dispatches += other.dispatches;
        self.active_row_steps += other.active_row_steps;
        self.reserved_row_steps += other.reserved_row_steps;
        self.capacity_row_steps += other.capacity_row_steps;
        self.cache_clone_ops += other.cache_clone_ops;
        self.cache_clone_bytes += other.cache_clone_bytes;
        self.attempts += other.attempts;
        self.fallback_attempts += other.fallback_attempts;
        self.copies.merge(other.copies);
    }
}

pub(crate) fn scheduler_seed_copy_accounting(rows: usize, self_bytes: usize, cross_bytes: usize) -> (usize, usize) {
    (rows.saturating_mul(4), rows.saturating_mul(self_bytes.saturating_add(cross_bytes)).saturating_mul(2))
}

pub(crate) fn cache_append_copy_accounting(rows: usize, per_pos_bytes: usize) -> (usize, usize) {
    (rows.saturating_mul(2), rows.saturating_mul(per_pos_bytes).saturating_mul(2))
}

pub(crate) fn beam_clone_copy_accounting(copies: usize, positions: usize, per_pos_bytes: usize) -> (usize, usize) {
    (copies.saturating_mul(2), copies.saturating_mul(positions).saturating_mul(per_pos_bytes).saturating_mul(2))
}

/// Small independently-testable allocator enforcing whole-attempt admission.
#[derive(Debug)]
pub(crate) struct SlotAllocator {
    owners: Vec<Option<usize>>,
}

impl SlotAllocator {
    pub(crate) fn new(capacity: usize) -> Self {
        Self { owners: vec![None; capacity] }
    }

    pub(crate) fn reserve(
        &mut self,
        owner: usize,
        width: usize,
    ) -> std::result::Result<Option<Vec<usize>>, &'static str> {
        if width == 0 {
            return Err("attempt width must be non-zero");
        }
        if width > self.owners.len() {
            return Err("decode attempt width exceeds decoder slots");
        }
        if self.owners.iter().filter(|slot| slot.is_none()).count() < width {
            return Ok(None);
        }
        let rows: Vec<_> = self
            .owners
            .iter()
            .enumerate()
            .filter_map(|(row, current)| current.is_none().then_some(row))
            .take(width)
            .collect();
        for &row in &rows {
            self.owners[row] = Some(owner);
        }
        Ok(Some(rows))
    }

    pub(crate) fn release(&mut self, owner: usize) {
        for slot in &mut self.owners {
            if *slot == Some(owner) {
                *slot = None;
            }
        }
    }

    fn reserved(&self) -> usize {
        self.owners.iter().filter(|owner| owner.is_some()).count()
    }

    #[cfg(test)]
    pub(crate) fn owners(&self) -> &[Option<usize>] {
        &self.owners
    }
}

struct SingleAttempt {
    next_token: u32,
    tokens: Vec<u32>,
    token_probs: Vec<f32>,
    sum_logprob: f32,
}

struct BeamAttempt {
    active: Vec<BeamHypothesis>,
    rows: Vec<usize>,
    finished: Vec<BeamHypothesis>,
    next_logical_id: usize,
}

enum AttemptKind {
    Single(SingleAttempt),
    Beam(BeamAttempt),
}

struct ScheduledAttempt {
    strategy_index: usize,
    strategy: DecodeStrategy,
    reserved_rows: Vec<usize>,
    pos: usize,
    generated_tokens: usize,
    kind: AttemptKind,
}

impl ScheduledAttempt {
    fn is_done(&self, sample_len: usize, eot: u32) -> bool {
        match &self.kind {
            AttemptKind::Single(single) => {
                sample_len == 0 || single.next_token == eot || single.tokens.len() >= sample_len
            }
            AttemptKind::Beam(beam) => {
                sample_len == 0
                    || self.generated_tokens >= sample_len
                    || beam.active.is_empty()
                    || beam.finished.len() >= self.reserved_rows.len()
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn start_attempt(
    strategy_index: usize,
    strategy: DecodeStrategy,
    rows: Vec<usize>,
    seed: &DecodeSeed,
    tokenizer: &WhisperTokenizer,
    options: &DecodeOptions,
    n_vocab: usize,
    rng: &mut StdRng,
) -> Result<ScheduledAttempt> {
    let metadata = &seed.metadata;
    let last = metadata
        .prefill_logits
        .get((metadata.init_len - 1) * n_vocab..metadata.init_len * n_vocab)
        .ok_or_else(|| decode_err("prefill logits are truncated"))?;
    let mut filtered = last.to_vec();
    apply_logit_filters(
        &mut filtered,
        tokenizer,
        options,
        &metadata.initial_tokens,
        metadata.sample_begin,
        0,
        &metadata.suppress_tokens,
    );
    let kind = match strategy {
        DecodeStrategy::Greedy | DecodeStrategy::Sample { .. } => {
            if options.sample_len == Some(0) {
                return Ok(ScheduledAttempt {
                    strategy_index,
                    strategy,
                    reserved_rows: rows,
                    pos: metadata.init_len,
                    generated_tokens: 0,
                    kind: AttemptKind::Single(SingleAttempt {
                        next_token: tokenizer.eot(),
                        tokens: Vec::new(),
                        token_probs: Vec::new(),
                        sum_logprob: 0.0,
                    }),
                });
            }
            let next_token = pick_token_with_rng(&filtered, strategy.temperature(), rng);
            let sum_logprob = log_softmax(&filtered, next_token as usize);
            let (tokens, token_probs) = if next_token == tokenizer.eot() {
                (Vec::new(), Vec::new())
            } else {
                (vec![next_token], vec![sum_logprob.exp()])
            };
            AttemptKind::Single(SingleAttempt { next_token, tokens, token_probs, sum_logprob })
        }
        DecodeStrategy::Beam { size } => {
            if options.sample_len == Some(0) {
                let active = vec![BeamHypothesis {
                    logical_id: 0,
                    tokens: metadata.initial_tokens.clone(),
                    token_probs: Vec::new(),
                    sum_logprob: 0.0,
                }];
                return Ok(ScheduledAttempt {
                    strategy_index,
                    strategy,
                    reserved_rows: rows.clone(),
                    pos: metadata.init_len,
                    generated_tokens: 0,
                    kind: AttemptKind::Beam(BeamAttempt {
                        active,
                        rows: vec![rows[0]],
                        finished: Vec::new(),
                        next_logical_id: 1,
                    }),
                });
            }
            let mut ranked: Vec<_> = log_softmax_vec(&filtered).into_iter().enumerate().collect();
            ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            let mut active = Vec::with_capacity(size);
            let mut finished = Vec::new();
            let mut next_logical_id = 0;
            for (token, logprob) in ranked {
                if active.len() >= size {
                    break;
                }
                let mut tokens = metadata.initial_tokens.clone();
                tokens.push(token as u32);
                let hypothesis = BeamHypothesis {
                    logical_id: next_logical_id,
                    tokens,
                    token_probs: vec![logprob.exp()],
                    sum_logprob: logprob,
                };
                next_logical_id += 1;
                if token as u32 == tokenizer.eot() {
                    finished.push(hypothesis);
                } else {
                    active.push(hypothesis);
                }
            }
            let active_rows = rows[..active.len()].to_vec();
            AttemptKind::Beam(BeamAttempt { active, rows: active_rows, finished, next_logical_id })
        }
    };
    Ok(ScheduledAttempt {
        strategy_index,
        strategy,
        reserved_rows: rows,
        pos: metadata.init_len,
        generated_tokens: 1,
        kind,
    })
}

fn seed_attempt_rows(
    jit: &mut WhisperDecoderStepJit,
    rows: &[usize],
    seed: &DecodeSeed,
    n_text_ctx: usize,
) -> Result<()> {
    if seed.self_positions > n_text_ctx
        || seed.self_positions.checked_mul(seed.per_pos_bytes) != Some(seed.self_cache_bytes)
        || seed.cross_positions.checked_mul(seed.per_pos_bytes) != Some(seed.cross_cache_bytes)
    {
        return Err(decode_err("decode seed cache geometry mismatch"));
    }
    let self_stride =
        n_text_ctx.checked_mul(seed.per_pos_bytes).ok_or_else(|| decode_err("self cache stride overflow"))?;
    let cross_stride = seed.cross_cache_bytes;
    for &row in rows {
        copy_device_cache_row(jit.self_k_cache_mut().context(JitSnafu)?, row, self_stride, &seed.self_k_cache)?;
        copy_device_cache_row(jit.self_v_cache_mut().context(JitSnafu)?, row, self_stride, &seed.self_v_cache)?;
        copy_device_cache_row(jit.cross_k_mut().context(JitSnafu)?, row, cross_stride, &seed.cross_k)?;
        copy_device_cache_row(jit.cross_v_mut().context(JitSnafu)?, row, cross_stride, &seed.cross_v)?;
    }
    Ok(())
}

fn append_row_cache(
    jit: &mut WhisperDecoderStepJit,
    row: usize,
    pos: usize,
    per_pos_bytes: usize,
    row_stride_bytes: usize,
) -> Result<()> {
    let dst = row
        .checked_mul(row_stride_bytes)
        .and_then(|base| pos.checked_mul(per_pos_bytes).and_then(|offset| base.checked_add(offset)))
        .ok_or_else(|| decode_err("self cache append offset overflow"))?;
    let src = row.checked_mul(per_pos_bytes).ok_or_else(|| decode_err("step cache output offset overflow"))?;
    jit.copy_output_to_self_k_cache(1, dst, src, per_pos_bytes).context(JitSnafu)?;
    jit.copy_output_to_self_v_cache(2, dst, src, per_pos_bytes).context(JitSnafu)
}

fn clone_cache_prefix(
    jit: &mut WhisperDecoderStepJit,
    copies: &[CacheCopy],
    positions: usize,
    per_pos_bytes: usize,
    row_stride_bytes: usize,
) -> Result<()> {
    let len = positions.checked_mul(per_pos_bytes).ok_or_else(|| decode_err("cache prefix length overflow"))?;
    for copy in copies {
        let src =
            copy.src_row.checked_mul(row_stride_bytes).ok_or_else(|| decode_err("cache source offset overflow"))?;
        let dst = copy
            .dst_row
            .checked_mul(row_stride_bytes)
            .ok_or_else(|| decode_err("cache destination offset overflow"))?;
        jit.self_k_cache_mut().context(JitSnafu)?.copy_within(dst, src, len).context(DeviceSnafu)?;
        jit.self_v_cache_mut().context(JitSnafu)?.copy_within(dst, src, len).context(DeviceSnafu)?;
    }
    Ok(())
}

fn finish_attempt(
    attempt: ScheduledAttempt,
    seed: &DecodeSeed,
    tokenizer: &WhisperTokenizer,
    options: &DecodeOptions,
) -> Result<DecodeResult> {
    match attempt.kind {
        AttemptKind::Single(single) => finish_decode(
            &single.tokens,
            &single.token_probs,
            tokenizer,
            if options.sample_len == Some(0) { 0.0 } else { single.sum_logprob },
            seed.metadata.no_speech_prob,
            options,
        ),
        AttemptKind::Beam(beam) => {
            let size = attempt.reserved_rows.len();
            let best =
                finalize_beam_hypotheses(beam.active, beam.finished, size, tokenizer.eot(), seed.metadata.sample_begin)
                    .ok_or_else(|| decode_err("beam produced nothing"))?;
            let tokens: Vec<_> = best.tokens[seed.metadata.sample_begin..]
                .iter()
                .copied()
                .take_while(|&token| token != tokenizer.eot())
                .collect();
            let token_probs = best.token_probs.into_iter().take(tokens.len()).collect::<Vec<_>>();
            finish_decode(&tokens, &token_probs, tokenizer, best.sum_logprob, seed.metadata.no_speech_prob, options)
        }
    }
}

/// Decode all requests through one concrete `[decoder_slots, ...]` step graph.
/// Attempts reserve their full width atomically and retain every reserved row,
/// including inactive beam rows, until quality acceptance or fallback requeue.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_fixed_slot_decode(
    seeds: &[DecodeSeed],
    request_options: &[DecodeOptions],
    step_jit: &mut WhisperDecoderStepJit,
    capacity: usize,
    tokenizer: &WhisperTokenizer,
    n_text_ctx: usize,
    n_vocab: usize,
    profile: bool,
) -> Result<(Vec<DecodeResult>, DecodeScheduleStats, GraphProfile)> {
    if seeds.len() != request_options.len() {
        return Err(decode_err("decode seed/options count mismatch"));
    }
    for options in request_options {
        options.validate().map_err(decode_err)?;
        for strategy in attempt_strategies(options) {
            if strategy_width(strategy) > capacity {
                return Err(decode_err("decode attempt width exceeds decoder slots"));
            }
        }
    }

    let strategies: Vec<_> = request_options.iter().map(attempt_strategies).collect();
    let mut queue: VecDeque<_> = (0..seeds.len()).map(|request| (request, 0usize)).collect();
    let mut allocator = SlotAllocator::new(capacity);
    let mut attempts: Vec<Option<ScheduledAttempt>> = (0..seeds.len()).map(|_| None).collect();
    let mut results: Vec<Option<DecodeResult>> = (0..seeds.len()).map(|_| None).collect();
    let mut stats = DecodeScheduleStats::default();
    let mut graph_profile = GraphProfile::default();
    let mut rngs: Vec<_> =
        request_options.iter().enumerate().map(|(request, options)| sampling_rng(options, request)).collect();

    while results.iter().any(Option::is_none) {
        while let Some(&(request, strategy_index)) = queue.front() {
            let strategy = strategies[request][strategy_index];
            let Some(rows) = allocator.reserve(request, strategy_width(strategy)).map_err(decode_err)? else {
                break;
            };
            queue.pop_front();
            let mut options = request_options[request].clone();
            options.strategy = strategy;
            let attempt = start_attempt(
                strategy_index,
                strategy,
                rows,
                &seeds[request],
                tokenizer,
                &options,
                n_vocab,
                &mut rngs[request],
            )?;
            let (ops, bytes) = scheduler_seed_copy_accounting(
                attempt.reserved_rows.len(),
                seeds[request].self_cache_bytes,
                seeds[request].cross_cache_bytes,
            );
            let (_, wall) = timed_d2d(profile, &seeds[request].self_k_cache, || {
                seed_attempt_rows(step_jit, &attempt.reserved_rows, &seeds[request], n_text_ctx)
            })?;
            if profile {
                stats.copies.d2d("scheduler_seeding", ops, bytes, wall);
            }
            attempts[request] = Some(attempt);
            stats.attempts += 1;
            stats.fallback_attempts += usize::from(strategy_index > 0);
        }

        let active_requests: Vec<_> =
            attempts.iter().enumerate().filter_map(|(request, attempt)| attempt.as_ref().map(|_| request)).collect();
        if active_requests.is_empty() {
            return Err(decode_err("fixed-slot scheduler made no progress"));
        }

        // Attempts that finish from prefill (EOT or zero budget) need no graph dispatch.
        let mut dispatch = false;
        let control_started = begin_host_copy(profile, step_jit.token_mut().context(JitSnafu)?)?;
        let mut control_ops = 0usize;
        let mut control_bytes = 0usize;
        for &request in &active_requests {
            let attempt = attempts[request].as_ref().expect("active attempt");
            let sample_len = request_options[request].sample_len.unwrap_or(n_text_ctx / 2);
            if attempt.is_done(sample_len, tokenizer.eot()) {
                continue;
            }
            let seed = &seeds[request].metadata;
            match &attempt.kind {
                AttemptKind::Single(single) => {
                    let row = attempt.reserved_rows[0];
                    write_token_row(step_jit, row, single.next_token)?;
                    write_pos_emb_row(
                        step_jit,
                        row,
                        seed.pos_embedding
                            .get(attempt.pos * seed.n_state..(attempt.pos + 1) * seed.n_state)
                            .ok_or_else(|| decode_err("position embedding is out of bounds"))?,
                    )?;
                    write_self_key_len_row(step_jit, row, attempt.pos)?;
                    control_ops = control_ops.saturating_add(3);
                    control_bytes = control_bytes.saturating_add(
                        std::mem::size_of::<i32>()
                            + seed.n_state * std::mem::size_of::<f32>()
                            + std::mem::size_of::<i32>(),
                    );
                }
                AttemptKind::Beam(beam) => {
                    for (hypothesis, &row) in beam.active.iter().zip(&beam.rows) {
                        write_token_row(
                            step_jit,
                            row,
                            *hypothesis.tokens.last().ok_or_else(|| decode_err("empty beam"))?,
                        )?;
                        write_pos_emb_row(
                            step_jit,
                            row,
                            seed.pos_embedding
                                .get(attempt.pos * seed.n_state..(attempt.pos + 1) * seed.n_state)
                                .ok_or_else(|| decode_err("position embedding is out of bounds"))?,
                        )?;
                        write_self_key_len_row(step_jit, row, attempt.pos)?;
                        control_ops = control_ops.saturating_add(3);
                        control_bytes = control_bytes.saturating_add(
                            std::mem::size_of::<i32>()
                                + seed.n_state * std::mem::size_of::<f32>()
                                + std::mem::size_of::<i32>(),
                        );
                    }
                }
            }
            dispatch = true;
        }

        if dispatch {
            if let Some(started) = control_started {
                stats.copies.h2d("decoder_controls", control_ops, control_bytes, started.elapsed());
            }
            stats.dispatches += 1;
            stats.capacity_row_steps += capacity;
            stats.reserved_row_steps += allocator.reserved();
            stats.active_row_steps += active_requests
                .iter()
                .map(|&request| match &attempts[request].as_ref().expect("active attempt").kind {
                    AttemptKind::Single(_) => 1,
                    AttemptKind::Beam(beam) => beam.active.len(),
                })
                .sum::<usize>();
            if profile {
                let graph_started = std::time::Instant::now();
                let kernels = step_jit.execute_profiled_static().context(JitSnafu)?;
                step_jit.logits().context(JitSnafu)?.synchronize().context(DeviceSnafu)?;
                graph_profile.record(graph_started.elapsed(), kernels);
            } else {
                step_jit.execute().context(JitSnafu)?;
            }
            for &request in &active_requests {
                let attempt = attempts[request].as_mut().expect("active attempt");
                let sample_len = request_options[request].sample_len.unwrap_or(n_text_ctx / 2);
                if attempt.is_done(sample_len, tokenizer.eot()) {
                    continue;
                }
                let decode_seed = &seeds[request];
                let seed = &decode_seed.metadata;
                let per_pos_bytes = decode_seed.per_pos_bytes;
                let row_stride_bytes = n_text_ctx
                    .checked_mul(per_pos_bytes)
                    .ok_or_else(|| decode_err("self cache row stride overflow"))?;
                let mut options = request_options[request].clone();
                options.strategy = attempt.strategy;
                match &mut attempt.kind {
                    AttemptKind::Single(single) => {
                        let row = attempt.reserved_rows[0];
                        let fence = step_jit.new_self_k().context(JitSnafu)?.clone();
                        let (_, wall) = timed_d2d(profile, &fence, || {
                            append_row_cache(step_jit, row, attempt.pos, per_pos_bytes, row_stride_bytes)
                        })?;
                        if profile {
                            let (ops, bytes) = cache_append_copy_accounting(1, per_pos_bytes);
                            stats.copies.d2d("cache_append", ops, bytes, wall);
                        }
                        let started = begin_host_copy(profile, step_jit.logits().context(JitSnafu)?)?;
                        let mut logits = read_logits_row(step_jit, row, n_vocab)?;
                        if let Some(started) = started {
                            stats.copies.d2h(
                                "decoder_logits",
                                1,
                                n_vocab * std::mem::size_of::<f32>(),
                                started.elapsed(),
                            );
                        }
                        let all_tokens: Vec<_> =
                            seed.initial_tokens.iter().copied().chain(single.tokens.iter().copied()).collect();
                        apply_logit_filters(
                            &mut logits,
                            tokenizer,
                            &options,
                            &all_tokens,
                            seed.sample_begin,
                            attempt.pos + 1 - seed.init_len,
                            &seed.suppress_tokens,
                        );
                        single.next_token =
                            pick_token_with_rng(&logits, attempt.strategy.temperature(), &mut rngs[request]);
                        let logprob = log_softmax(&logits, single.next_token as usize);
                        single.sum_logprob += logprob;
                        if single.next_token != tokenizer.eot() {
                            single.tokens.push(single.next_token);
                            single.token_probs.push(logprob.exp());
                        }
                    }
                    AttemptKind::Beam(beam) => {
                        let append_rows = beam.rows.clone();
                        let fence = step_jit.new_self_k().context(JitSnafu)?.clone();
                        let (_, wall) = timed_d2d(profile, &fence, || {
                            for &row in &append_rows {
                                append_row_cache(step_jit, row, attempt.pos, per_pos_bytes, row_stride_bytes)?;
                            }
                            Ok(())
                        })?;
                        if profile {
                            let (ops, bytes) = cache_append_copy_accounting(append_rows.len(), per_pos_bytes);
                            stats.copies.d2d("cache_append", ops, bytes, wall);
                        }
                        let size = attempt.reserved_rows.len();
                        let mut candidates = Vec::new();
                        for (parent_index, (hypothesis, &row)) in beam.active.iter().zip(&beam.rows).enumerate() {
                            let started = begin_host_copy(profile, step_jit.logits().context(JitSnafu)?)?;
                            let mut logits = read_logits_row(step_jit, row, n_vocab)?;
                            if let Some(started) = started {
                                stats.copies.d2h(
                                    "decoder_logits",
                                    1,
                                    n_vocab * std::mem::size_of::<f32>(),
                                    started.elapsed(),
                                );
                            }
                            apply_logit_filters(
                                &mut logits,
                                tokenizer,
                                &options,
                                &hypothesis.tokens,
                                seed.sample_begin,
                                attempt.pos + 1 - seed.init_len,
                                &seed.suppress_tokens,
                            );
                            let mut ranked: Vec<_> = log_softmax_vec(&logits).into_iter().enumerate().collect();
                            ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                            for (token, logprob) in ranked.into_iter().take(size + 1) {
                                candidates.push(BeamCandidate {
                                    parent_index,
                                    parent_logical_id: hypothesis.logical_id,
                                    parent_row: row,
                                    token_id: token as u32,
                                    token_logprob: logprob,
                                    sum_logprob: hypothesis.sum_logprob + logprob,
                                });
                            }
                        }
                        let (active, newly_finished, survivors) = select_beam_candidates(
                            &beam.active,
                            candidates,
                            size,
                            tokenizer.eot(),
                            size - beam.finished.len(),
                            &mut beam.next_logical_id,
                        );
                        beam.finished.extend(newly_finished);
                        let assignment = plan_beam_rows(&attempt.reserved_rows, &survivors).map_err(decode_err)?;
                        let fence = step_jit.self_k_cache_mut().context(JitSnafu)?.clone();
                        let (_, wall) = timed_d2d(profile, &fence, || {
                            clone_cache_prefix(
                                step_jit,
                                &assignment.copies,
                                attempt.pos + 1,
                                per_pos_bytes,
                                row_stride_bytes,
                            )
                        })?;
                        stats.cache_clone_ops += assignment.copies.len();
                        let (clone_ops, clone_bytes) =
                            beam_clone_copy_accounting(assignment.copies.len(), attempt.pos + 1, per_pos_bytes);
                        stats.cache_clone_bytes = stats.cache_clone_bytes.saturating_add(clone_bytes);
                        if profile {
                            stats.copies.d2d("beam_clone", clone_ops, clone_bytes, wall);
                        }
                        beam.active = active;
                        beam.rows = assignment.rows;
                    }
                }
                attempt.pos += 1;
                attempt.generated_tokens += 1;
            }
        }

        for request in active_requests {
            let sample_len = request_options[request].sample_len.unwrap_or(n_text_ctx / 2);
            let done = attempts[request]
                .as_ref()
                .is_some_and(|attempt| attempt.is_done(sample_len, tokenizer.eot()) || attempt.pos >= n_text_ctx);
            if !done {
                continue;
            }
            let attempt = attempts[request].take().expect("completed attempt");
            let strategy_index = attempt.strategy_index;
            let mut options = request_options[request].clone();
            options.strategy = attempt.strategy;
            options.fallback = None;
            let result = finish_attempt(attempt, &seeds[request], tokenizer, &options)?;
            allocator.release(request);
            let retry = strategies[request].get(strategy_index + 1).is_some()
                && request_options[request]
                    .fallback
                    .as_ref()
                    .is_some_and(|fallback| check_fallback(&result, fallback, &request_options[request]));
            if retry {
                queue.push_back((request, strategy_index + 1));
            } else {
                results[request] = Some(result);
            }
        }
    }

    Ok((collect_ordered(results).map_err(decode_err)?, stats, graph_profile))
}

// ─── Batched JIT buffer row helpers ─────────────────────────────────────────
//
// The batched step JIT owns max_lanes-sized buffers; each lane writes/reads
// its row. These wrap the per-row slicing so the main loop stays readable.

fn write_token_row(jit: &mut WhisperDecoderStepJit, row: usize, token: u32) -> Result<()> {
    let buf = jit.token_mut().context(JitSnafu)?;
    let dst = buf.as_host_bytes_mut().context(DeviceSnafu)?;
    // token is [max_lanes, 1] i32; row stride = 4 bytes
    let off = row.checked_mul(std::mem::size_of::<i32>()).ok_or_else(|| decode_err("token row offset overflow"))?;
    let tok = [token as i32];
    let bytes: &[u8] = bytemuck::cast_slice(&tok);
    let target = dst.get_mut(off..off + bytes.len()).ok_or_else(|| decode_err("token row is out of bounds"))?;
    target.copy_from_slice(bytes);
    Ok(())
}

fn write_pos_emb_row(jit: &mut WhisperDecoderStepJit, row: usize, emb: &[f32]) -> Result<()> {
    let buf = jit.pos_emb_mut().context(JitSnafu)?;
    let dst = buf.as_host_bytes_mut().context(DeviceSnafu)?;
    // pos_emb is [max_lanes, 1, n_state] f32
    let row_bytes =
        emb.len().checked_mul(std::mem::size_of::<f32>()).ok_or_else(|| decode_err("position row stride overflow"))?;
    let off = row.checked_mul(row_bytes).ok_or_else(|| decode_err("position row offset overflow"))?;
    let bytes: &[u8] = bytemuck::cast_slice(emb);
    let target = dst.get_mut(off..off + bytes.len()).ok_or_else(|| decode_err("position row is out of bounds"))?;
    target.copy_from_slice(bytes);
    Ok(())
}

fn write_self_key_len_row(jit: &mut WhisperDecoderStepJit, row: usize, pos: usize) -> Result<()> {
    let buf = jit.self_key_lens_mut().context(JitSnafu)?;
    let dst = buf.as_host_bytes_mut().context(DeviceSnafu)?;
    let off =
        row.checked_mul(std::mem::size_of::<i32>()).ok_or_else(|| decode_err("self key length row offset overflow"))?;
    let len = i32::try_from(pos).map_err(|_| decode_err("decoder position exceeds i32"))?;
    let bytes: &[u8] = bytemuck::bytes_of(&len);
    let target =
        dst.get_mut(off..off + bytes.len()).ok_or_else(|| decode_err("self key length row is out of bounds"))?;
    target.copy_from_slice(bytes);
    Ok(())
}

/// Seed one physical cache row from an immutable device-local snapshot.
pub(crate) fn copy_device_cache_row(
    buf: &mut Buffer,
    row: usize,
    row_stride_bytes: usize,
    data: &Buffer,
) -> Result<()> {
    if buf.dtype() != DType::Float32 || data.dtype() != DType::Float32 {
        return Err(decode_err("cache seed buffers must be float32"));
    }
    if !std::ptr::eq(buf.allocator(), data.allocator()) {
        return Err(decode_err("cache seed and decoder row use different allocators"));
    }
    let off = row.checked_mul(row_stride_bytes).ok_or_else(|| decode_err("cache row offset overflow"))?;
    let end = off.checked_add(data.size()).ok_or_else(|| decode_err("cache seed end overflow"))?;
    if end > buf.size() || data.size() > row_stride_bytes {
        return Err(decode_err("cache seed row is out of bounds"));
    }
    buf.copy_region_from(off, data, 0, data.size()).context(DeviceSnafu)
}

/// Read one lane's logits row `[n_vocab]` from the batched JIT output.
fn read_logits_row(jit: &mut WhisperDecoderStepJit, row: usize, n_vocab: usize) -> Result<Vec<f32>> {
    let buf = jit.logits().context(JitSnafu)?;
    let src = buf.as_host_bytes().context(DeviceSnafu)?;
    let row_bytes = n_vocab.checked_mul(std::mem::size_of::<f32>()).ok_or_else(|| decode_err("logits row overflow"))?;
    let off = row.checked_mul(row_bytes).ok_or_else(|| decode_err("logits offset overflow"))?;
    let end = off.checked_add(row_bytes).ok_or_else(|| decode_err("logits end overflow"))?;
    let logits = src.get(off..end).ok_or_else(|| decode_err("logits row is out of bounds"))?;
    Ok(bytemuck::cast_slice(logits).to_vec())
}

// ─── Cached beam search ─────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct BeamHypothesis {
    /// Stable search identity. Decoder rows are deliberately not part of it.
    pub(crate) logical_id: usize,
    pub(crate) tokens: Vec<u32>,
    pub(crate) token_probs: Vec<f32>,
    pub(crate) sum_logprob: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct BeamCandidate {
    pub(crate) parent_index: usize,
    pub(crate) parent_logical_id: usize,
    pub(crate) parent_row: usize,
    pub(crate) token_id: u32,
    pub(crate) token_logprob: f32,
    pub(crate) sum_logprob: f32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct BeamSurvivor {
    pub(crate) logical_id: usize,
    pub(crate) parent_row: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CacheCopy {
    pub(crate) src_row: usize,
    pub(crate) dst_row: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RowAssignment {
    /// Destination row for each survivor, in survivor (logical rank) order.
    pub(crate) rows: Vec<usize>,
    pub(crate) copies: Vec<CacheCopy>,
}

/// Assign fixed decoder rows without scratch storage or copy cycles.
///
/// The first selected child of each parent retains that parent's row. Further
/// children use only reserved rows whose old hypotheses are no longer live.
pub(crate) fn plan_beam_rows(
    reserved_rows: &[usize],
    survivors: &[BeamSurvivor],
) -> std::result::Result<RowAssignment, &'static str> {
    let mut unique_reserved = reserved_rows.to_vec();
    unique_reserved.sort_unstable();
    unique_reserved.dedup();
    if unique_reserved.len() != reserved_rows.len() {
        return Err("reserved beam rows must be unique");
    }
    if survivors.len() > reserved_rows.len() {
        return Err("more survivors than reserved beam rows");
    }
    if survivors.iter().any(|survivor| !unique_reserved.contains(&survivor.parent_row)) {
        return Err("survivor parent row is not reserved");
    }

    let mut live_parent_rows = Vec::new();
    for survivor in survivors {
        if !live_parent_rows.contains(&survivor.parent_row) {
            live_parent_rows.push(survivor.parent_row);
        }
    }
    let mut dead_rows = reserved_rows.iter().copied().filter(|row| !live_parent_rows.contains(row));
    let mut retained = Vec::new();
    let mut rows = Vec::with_capacity(survivors.len());
    let mut copies = Vec::new();
    for survivor in survivors {
        if !retained.contains(&survivor.parent_row) {
            retained.push(survivor.parent_row);
            rows.push(survivor.parent_row);
        } else {
            let dst_row = dead_rows.next().ok_or("insufficient inactive rows for duplicate beam children")?;
            rows.push(dst_row);
            copies.push(CacheCopy { src_row: survivor.parent_row, dst_row });
        }
    }
    Ok(RowAssignment { rows, copies })
}

fn candidate_order(a: &BeamCandidate, b: &BeamCandidate) -> Ordering {
    b.sum_logprob
        .total_cmp(&a.sum_logprob)
        .then_with(|| a.parent_logical_id.cmp(&b.parent_logical_id))
        .then_with(|| a.token_id.cmp(&b.token_id))
        .then_with(|| a.parent_index.cmp(&b.parent_index))
}

/// Deterministically rank candidates and split completed from active children.
pub(crate) fn select_beam_candidates(
    parents: &[BeamHypothesis],
    mut candidates: Vec<BeamCandidate>,
    beam_size: usize,
    eot: u32,
    finished_capacity: usize,
    next_logical_id: &mut usize,
) -> (Vec<BeamHypothesis>, Vec<BeamHypothesis>, Vec<BeamSurvivor>) {
    candidates.sort_by(candidate_order);
    let mut active = Vec::with_capacity(beam_size);
    let mut finished = Vec::new();
    let mut survivors = Vec::with_capacity(beam_size);
    for candidate in candidates {
        if active.len() >= beam_size {
            break;
        }
        let Some(parent) = parents.get(candidate.parent_index) else {
            continue;
        };
        let logical_id = *next_logical_id;
        *next_logical_id += 1;
        let mut child = parent.clone();
        child.logical_id = logical_id;
        child.tokens.push(candidate.token_id);
        child.token_probs.push(candidate.token_logprob.exp());
        child.sum_logprob = candidate.sum_logprob;
        if candidate.token_id == eot {
            if finished.len() < finished_capacity {
                finished.push(child);
            }
        } else {
            active.push(child);
            survivors.push(BeamSurvivor { logical_id, parent_row: candidate.parent_row });
        }
    }
    (active, finished, survivors)
}

/// Backfill unfinished hypotheses with EOT and choose the normalized best.
pub(crate) fn finalize_beam_hypotheses(
    active: Vec<BeamHypothesis>,
    mut finished: Vec<BeamHypothesis>,
    beam_size: usize,
    eot: u32,
    sample_begin: usize,
) -> Option<BeamHypothesis> {
    for mut hypothesis in active {
        if finished.len() >= beam_size {
            break;
        }
        if hypothesis.tokens.last().is_none_or(|&token| token != eot) {
            hypothesis.tokens.push(eot);
        }
        finished.push(hypothesis);
    }
    finished.sort_by(|a, b| {
        let a_score = a.sum_logprob / a.tokens.len().saturating_sub(sample_begin + 1).max(1) as f32;
        let b_score = b.sum_logprob / b.tokens.len().saturating_sub(sample_begin + 1).max(1) as f32;
        b_score.total_cmp(&a_score).then_with(|| a.logical_id.cmp(&b.logical_id))
    });
    finished.into_iter().next()
}

pub(crate) struct PrefillMetadata {
    pub(crate) initial_tokens: Vec<u32>,
    pub(crate) sample_begin: usize,
    pub(crate) init_len: usize,
    pub(crate) suppress_tokens: Vec<i32>,
    pub(crate) prefill_logits: Vec<f32>,
    pub(crate) no_speech_prob: f32,
    pub(crate) pos_embedding: Vec<f32>,
    pub(crate) n_state: usize,
}

#[allow(clippy::too_many_arguments)]
fn execute_prefill(
    prefill_jit: &mut WhisperPrefillJit,
    tokenizer: &WhisperTokenizer,
    options: &DecodeOptions,
    n_vocab: usize,
    pos_embedding: &[f32],
    n_state: usize,
    mut copies: Option<&mut CopyProfile>,
    graph_profile: Option<&mut GraphProfile>,
) -> Result<PrefillMetadata> {
    // Build initial tokens
    let mut initial_tokens = vec![tokenizer.sot()];
    if tokenizer.multilingual {
        let lang = options.language.as_ref().ok_or_else(|| decode_err("language required"))?;
        let lang_tok = tokenizer.language_token_for(lang).unwrap_or_else(|| tokenizer.sot());
        let task_tok = match options.task {
            WhisperTask::Transcribe => tokenizer.transcribe(),
            WhisperTask::Translate => tokenizer.translate(),
        };
        initial_tokens.extend([lang_tok, task_tok]);
    }
    let sample_begin = initial_tokens.len();
    let init_len = initial_tokens.len();
    let suppress_tokens = get_suppress_tokens(tokenizer, options);

    // Write tokens to prefill JIT buffer
    {
        let token_data: Vec<i32> = initial_tokens.iter().map(|&t| t as i32).collect();
        let buf = prefill_jit.tokens_mut().context(JitSnafu)?;
        let started = begin_host_copy(copies.is_some(), buf)?;
        let data = bytemuck::cast_slice(&token_data);
        write_buf(buf, data)?;
        if let (Some(copies), Some(started)) = (copies.as_deref_mut(), started) {
            copies.h2d("prefill_tokens", 1, data.len(), started.elapsed());
        }
    }

    // Execute prefill JIT (plan manages all buffers, no realize)
    if let Some(graph_profile) = graph_profile {
        let graph_started = std::time::Instant::now();
        let kernels = prefill_jit.execute_profiled_static().context(JitSnafu)?;
        prefill_jit.logits().context(JitSnafu)?.synchronize().context(DeviceSnafu)?;
        graph_profile.record(graph_started.elapsed(), kernels);
    } else {
        prefill_jit.execute().context(JitSnafu)?;
    }

    // Read logits from output 0
    let started = begin_host_copy(copies.is_some(), prefill_jit.logits().context(JitSnafu)?)?;
    let prefill_logits = {
        let buf = prefill_jit.logits().context(JitSnafu)?;
        read_buf(buf, buf.size() / std::mem::size_of::<f32>())?
    };
    if let (Some(copies), Some(started)) = (copies, started) {
        copies.d2h("prefill_logits", 1, prefill_logits.len() * std::mem::size_of::<f32>(), started.elapsed());
    }
    let no_speech_prob = tokenizer
        .no_speech()
        .map(|ns| softmax_prob(&prefill_logits[..n_vocab.min(prefill_logits.len())], ns as usize))
        .unwrap_or(f32::NAN);

    Ok(PrefillMetadata {
        initial_tokens,
        sample_begin,
        init_len,
        suppress_tokens,
        prefill_logits,
        no_speech_prob,
        pos_embedding: pos_embedding.to_vec(),
        n_state,
    })
}

// ─── Result helpers ─────────────────────────────────────────────────────────

/// Split a decoded token stream into timestamp-bounded segments.
///
/// The decoder emits paired timestamp tokens (`<|t0|> text <|t1|> text <|t2|>...`)
/// during timestamp-enabled recognition. This function finds
/// consecutive timestamp-token pairs — the boundary between segments — and
/// returns one [`Segment`] per slice, with window-relative start/end times
/// decoded from the timestamp token values.
///
/// Ported from the OpenAI reference (`transcribe.py:339-367`). When no
/// consecutive timestamp pairs are found, returns a single segment spanning
/// the whole token stream.
pub fn split_into_segments(
    tokens: &[u32],
    tokenizer: &WhisperTokenizer,
    window_duration: f32,
) -> Vec<svod_arch::pipelines::audio::Segment> {
    let ts_begin = tokenizer.timestamp_begin();
    let is_ts = |t: u32| t >= ts_begin;

    // Find indices where two adjacent tokens are both timestamps — these are
    // segment boundaries (the closing ts of one segment + the opening ts of the
    // next, shared).
    let mut boundaries: Vec<usize> = Vec::new();
    for i in 1..tokens.len() {
        if is_ts(tokens[i - 1]) && is_ts(tokens[i]) {
            boundaries.push(i);
        }
    }

    let mut segments = Vec::new();
    let terminal_timestamp = tokens.last().is_some_and(|&token| is_ts(token))
        && tokens.get(tokens.len().saturating_sub(2)).is_none_or(|&token| !is_ts(token));

    if boundaries.is_empty() {
        // Whisper treats this as one window-relative segment. If any timestamp
        // was emitted, its last value limits the segment duration.
        let start = 0.0;
        let end = tokens
            .iter()
            .rev()
            .find(|&&token| is_ts(token))
            .filter(|&&token| token != ts_begin)
            .map(|&token| token_to_seconds(token, ts_begin))
            .unwrap_or(window_duration)
            .clamp(0.0, window_duration.max(0.0));
        let text = tokenizer.decode(tokens);
        let text = text.trim();
        if !text.is_empty() && end > start {
            segments.push(Segment { text: text.to_string(), start, end });
        }
        return segments;
    }

    let mut last_slice = 0;
    for &boundary in &boundaries {
        if boundary > last_slice {
            segments.push(segment_from_tokens(&tokens[last_slice..boundary], tokenizer, ts_begin, window_duration));
        }
        last_slice = boundary;
    }

    // An unfinished tail is excluded; it will be decoded again from the last
    // completed timestamp boundary by long-form host orchestration.
    if terminal_timestamp && tokens.len() > last_slice {
        segments.push(segment_from_tokens(&tokens[last_slice..], tokenizer, ts_begin, window_duration));
    }

    // Filter empty segments (can happen when consecutive timestamps have no text between them).
    segments.retain(|s| !s.text.is_empty() && s.end > s.start);
    segments
}

/// Decode one timestamp-bounded slice into a [`Segment`].
fn segment_from_tokens(slice: &[u32], tokenizer: &WhisperTokenizer, ts_begin: u32, window_duration: f32) -> Segment {
    let extent = window_duration.max(0.0);
    let start = slice
        .first()
        .filter(|&&t| t >= ts_begin)
        .map(|&t| token_to_seconds(t, ts_begin))
        .unwrap_or(0.0)
        .clamp(0.0, extent);
    let end = slice
        .last()
        .filter(|&&t| t >= ts_begin)
        .map(|&t| token_to_seconds(t, ts_begin))
        .unwrap_or(start)
        .clamp(start, extent);
    let text = tokenizer.decode(slice).trim().to_string();
    Segment { text, start, end }
}

/// Convert a timestamp token id to seconds: `(id - timestamp_begin) / TOKENS_PER_SECOND`.
fn token_to_seconds(token: u32, ts_begin: u32) -> f32 {
    (token - ts_begin) as f32 / super::config::TOKENS_PER_SECOND
}

fn finish_decode(
    tokens: &[u32],
    token_probs: &[f32],
    tokenizer: &WhisperTokenizer,
    sum_logprob: f32,
    no_speech_prob: f32,
    options: &DecodeOptions,
) -> Result<DecodeResult> {
    let text = tokenizer.decode(tokens);
    let avg_logprob = sum_logprob / (tokens.len() + 1) as f32;
    let compression_ratio = compression_ratio_text(&text);
    Ok(DecodeResult {
        tokens: tokens.to_vec(),
        token_probs: token_probs.to_vec(),
        text,
        avg_logprob,
        no_speech_prob,
        temperature: options.strategy.temperature(),
        compression_ratio,
        language: options.language.clone(),
    })
}

fn pick_token_with_rng(logits: &[f32], temperature: f32, rng: &mut impl RngExt) -> u32 {
    if temperature > 0.0 { sample_from_logits(logits, temperature, rng) } else { argmax(logits) as u32 }
}

pub(crate) fn derived_sampling_seed(base: u64, request: usize) -> u64 {
    if request == 0 {
        return base;
    }
    let mut value = base ^ (request as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn sampling_rng(options: &DecodeOptions, request: usize) -> StdRng {
    let seed = options.sampling_seed.map(|base| derived_sampling_seed(base, request)).unwrap_or_else(rand::random);
    StdRng::seed_from_u64(seed)
}

fn decode_err(msg: &str) -> Error {
    Error::Decode { msg: msg.into() }
}

// ─── JIT buffer helpers ─────────────────────────────────────────────────────

fn write_uncached(jit: &mut WhisperDecoderJit, tokens: &[i32]) -> Result<()> {
    let buf = jit.tokens_mut().context(JitSnafu)?;
    write_buf(buf, bytemuck::cast_slice(tokens))
}

fn read_uncached(jit: &WhisperDecoderJit, n_vocab: usize) -> Result<Vec<f32>> {
    let buf = jit.output().context(JitSnafu)?;
    read_buf(buf, n_vocab)
}

/// Write data directly into the buffer's host-visible mapping.
/// `as_host_bytes_mut` syncs pending GPU work before returning the slice.
/// Subsequent `execute()` sees our writes (unified memory / BAR).
fn write_buf(buf: &svod_device::Buffer, data: &[u8]) -> Result<()> {
    let dst = buf.as_host_bytes_mut().context(DeviceSnafu)?;
    let n = data.len().min(dst.len());
    dst[..n].copy_from_slice(&data[..n]);
    Ok(())
}

/// Read data directly from the buffer's host-visible mapping.
/// `as_host_bytes` syncs pending GPU work before returning the slice.
fn read_buf(buf: &svod_device::Buffer, n: usize) -> Result<Vec<f32>> {
    let src = buf.as_host_bytes().context(DeviceSnafu)?;
    let n = n.min(src.len() / std::mem::size_of::<f32>());
    Ok(bytemuck::cast_slice(&src[..n * std::mem::size_of::<f32>()]).to_vec())
}

// ─── Logit filter helpers ───────────────────────────────────────────────────

fn get_suppress_tokens(tokenizer: &WhisperTokenizer, options: &DecodeOptions) -> Vec<i32> {
    let mut tokens: Vec<i32> = options.suppress_tokens.clone().unwrap_or_default();
    if tokens.contains(&-1) {
        tokens.retain(|&t| t >= 0);
        for &t in &tokenizer.non_speech_tokens() {
            tokens.push(t as i32);
        }
    }
    tokens.extend([
        tokenizer.transcribe() as i32,
        tokenizer.translate() as i32,
        tokenizer.sot() as i32,
        tokenizer.sot_prev() as i32,
        tokenizer.sot_lm() as i32,
    ]);
    if let Some(ns) = tokenizer.no_speech() {
        tokens.push(ns as i32);
    }
    tokens.sort();
    tokens.dedup();
    tokens
}

fn apply_logit_filters(
    logits: &mut [f32],
    tokenizer: &WhisperTokenizer,
    options: &DecodeOptions,
    tokens: &[u32],
    sample_begin: usize,
    step: usize,
    suppress_tokens: &[i32],
) {
    let eot = tokenizer.eot() as usize;
    if options.suppress_blank && step == 0 {
        for &t in &tokenizer.encode(" ") {
            if (t as usize) < logits.len() {
                logits[t as usize] = f32::NEG_INFINITY;
            }
        }
        if eot < logits.len() {
            logits[eot] = f32::NEG_INFINITY;
        }
    }
    for &t in suppress_tokens {
        if t >= 0 && (t as usize) < logits.len() {
            logits[t as usize] = f32::NEG_INFINITY;
        }
    }
    let specials =
        [tokenizer.transcribe(), tokenizer.translate(), tokenizer.sot(), tokenizer.sot_prev(), tokenizer.sot_lm()];
    for &t in &specials {
        if (t as usize) < logits.len() {
            logits[t as usize] = f32::NEG_INFINITY;
        }
    }
    if let Some(ns) = tokenizer.no_speech()
        && (ns as usize) < logits.len()
    {
        logits[ns as usize] = f32::NEG_INFINITY;
    }
    apply_timestamp_rules(logits, tokenizer, tokens, sample_begin, options);
}

fn apply_timestamp_rules(
    logits: &mut [f32],
    tokenizer: &WhisperTokenizer,
    tokens: &[u32],
    sample_begin: usize,
    options: &DecodeOptions,
) {
    let ts_begin = tokenizer.timestamp_begin() as usize;
    let eot = tokenizer.eot() as usize;
    let no_ts = tokenizer.no_timestamps() as usize;
    if no_ts < logits.len() {
        logits[no_ts] = f32::NEG_INFINITY;
    }

    let sampled = &tokens[sample_begin.min(tokens.len())..];
    let last_was_ts = sampled.last().map(|&t| (t as usize) >= ts_begin).unwrap_or(false);
    let penultimate_was_ts = sampled.len() < 2 || (sampled[sampled.len() - 2] as usize) >= ts_begin;

    if last_was_ts {
        if penultimate_was_ts {
            for t in &mut logits[ts_begin..] {
                *t = f32::NEG_INFINITY;
            }
        } else {
            for t in &mut logits[..eot] {
                *t = f32::NEG_INFINITY;
            }
        }
    }

    let ts_tokens: Vec<u32> = sampled.iter().filter(|&&t| (t as usize) >= ts_begin).copied().collect();
    if !ts_tokens.is_empty() {
        let last_ts = if last_was_ts && !penultimate_was_ts {
            ts_tokens.last().copied().unwrap_or(0) as usize
        } else {
            ts_tokens.last().copied().unwrap_or(0) as usize + 1
        };
        for (i, t) in logits[ts_begin..].iter_mut().enumerate() {
            if ts_begin + i < last_ts {
                *t = f32::NEG_INFINITY;
            }
        }
    }

    if tokens.len() == sample_begin {
        for t in &mut logits[..ts_begin] {
            *t = f32::NEG_INFINITY;
        }
        if let Some(max_init) = options.max_initial_timestamp {
            let last_allowed = ts_begin + (max_init / 0.02).round() as usize;
            if last_allowed + 1 < logits.len() {
                for t in &mut logits[last_allowed + 1..] {
                    *t = f32::NEG_INFINITY;
                }
            }
        }
    }

    let ts_logprob = logsumexp(&logits[ts_begin..]);
    let text_max = logits[..ts_begin].iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    if ts_logprob > text_max {
        for t in &mut logits[..eot] {
            *t = f32::NEG_INFINITY;
        }
    }
}

// ─── Math helpers ───────────────────────────────────────────────────────────

fn argmax(arr: &[f32]) -> usize {
    arr.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn softmax_prob(logits: &[f32], idx: usize) -> f32 {
    let max_val = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let sum: f32 = logits.iter().map(|&l| (l - max_val).exp()).sum();
    if idx < logits.len() { (logits[idx] - max_val).exp() / sum.max(1e-10) } else { 0.0 }
}

fn log_softmax(logits: &[f32], idx: usize) -> f32 {
    let max_val = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let sum: f32 = logits.iter().map(|&l| (l - max_val).exp()).sum();
    let logsum = sum.ln() + max_val;
    if idx < logits.len() { logits[idx] - logsum } else { f32::NEG_INFINITY }
}

fn log_softmax_vec(logits: &[f32]) -> Vec<f32> {
    let max_val = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let sum: f32 = logits.iter().map(|&l| (l - max_val).exp()).sum();
    let logsum = sum.ln() + max_val;
    logits.iter().map(|&l| l - logsum).collect()
}

fn logsumexp(arr: &[f32]) -> f32 {
    let max_val = arr.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    if max_val == f32::NEG_INFINITY {
        return f32::NEG_INFINITY;
    }
    (arr.iter().map(|&l| (l - max_val).exp()).sum::<f32>()).ln() + max_val
}

fn compression_ratio_text(text: &str) -> f32 {
    let raw = text.as_bytes();
    if raw.is_empty() {
        return 1.0;
    }
    use std::io::Write;
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    let _ = encoder.write_all(raw);
    let clen = encoder.finish().unwrap_or_default().len().max(1);
    raw.len() as f32 / clen as f32
}

/// Multinomial sampling from logits at temperature T. Matches the OpenAI
/// reference's `Categorical(logits=logits/T).sample()` (`decoding.py:283`),
/// which PyTorch implements as a numerically stable softmax (max-subtract
/// before exp) followed by inverse-CDF sampling.
fn sample_from_logits(logits: &[f32], temperature: f32, rng: &mut impl RngExt) -> u32 {
    // Max-subtract for numerical stability: exp(x - m) avoids overflow on
    // large positive logits. The max is a no-op for the sampling distribution
    // (it's a constant shift that cancels in normalization).
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) / temperature;
    let probs: Vec<f32> = logits.iter().map(|&l| ((l / temperature) - max_val).exp()).collect();
    let sum: f32 = probs.iter().copied().sum();
    let mut r = rng.random::<f32>() * sum;
    for (i, &p) in probs.iter().enumerate() {
        r -= p;
        if r <= 0.0 {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}
