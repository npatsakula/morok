//! Text decoder: token + positional embeddings + self/cross-attention transformer blocks.

use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::{Layer, LayerNorm, Linear, Module};

use crate::init::{Bias, fan_in_uniform, layer_norm, linear};
use crate::state::{scope_index, scoped, scoped_index};

use super::attention::MultiHeadAttention;
use super::blocks::linear_forward;
use super::config::ModelDimensions;
use super::error::{Result, tk_launch_error};

#[derive(Clone, Copy)]
struct StepAttentionConfig {
    custom_self: bool,
    custom_cross: bool,
    cross_splits: Option<usize>,
}

impl Default for StepAttentionConfig {
    fn default() -> Self {
        Self { custom_self: true, custom_cross: true, cross_splits: None }
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug)]
pub(crate) enum StepAttentionMode {
    Generic,
    CustomSelf,
    CustomCross { split: usize },
    CustomBoth { split: usize },
}

#[cfg(test)]
impl From<StepAttentionMode> for StepAttentionConfig {
    fn from(mode: StepAttentionMode) -> Self {
        match mode {
            StepAttentionMode::Generic => Self { custom_self: false, custom_cross: false, cross_splits: None },
            StepAttentionMode::CustomSelf => Self { custom_self: true, custom_cross: false, cross_splits: None },
            StepAttentionMode::CustomCross { split } => {
                Self { custom_self: false, custom_cross: true, cross_splits: Some(split) }
            }
            StepAttentionMode::CustomBoth { split } => {
                Self { custom_self: true, custom_cross: true, cross_splits: Some(split) }
            }
        }
    }
}

/// Validity mask `[B, key_count]` for a cached decoder step: the cached prefix
/// each lane actually filled, plus the key this step just appended at the end
/// of the cache. `true` = attend, the polarity SDPA's `key_padding_mask` wants.
pub(crate) fn cached_step_mask(key_lens: &Tensor, key_count: usize) -> Result<Tensor> {
    let appended = Tensor::arange(key_count as i64, None, None)?.try_eq(key_count as i64 - 1)?;
    Ok(Tensor::sequence_mask(key_lens, key_count)?.try_bitor(&appended)?)
}

/// Decoder transformer block: self-attn + cross-attn + MLP, all pre-norm.
#[derive(Clone, Module)]
pub struct DecoderBlock {
    pub attn: MultiHeadAttention,
    pub attn_ln: LayerNorm,
    pub cross_attn: MultiHeadAttention,
    pub cross_attn_ln: LayerNorm,
    #[module(key = "mlp.0")]
    pub mlp0: Linear,
    #[module(key = "mlp.2")]
    pub mlp2: Linear,
    pub mlp_ln: LayerNorm,
    pub n_state: usize,
}

impl DecoderBlock {
    pub fn empty(n_state: usize, n_head: usize) -> Self {
        Self::empty_dtype(n_state, n_head, DType::Float32)
    }

    pub fn empty_dtype(n_state: usize, n_head: usize, dtype: DType) -> Self {
        let mlp = n_state * 4;
        Self {
            attn: MultiHeadAttention::empty_dtype(n_state, n_head, dtype.clone()),
            attn_ln: layer_norm(n_state, dtype.clone()),
            cross_attn: MultiHeadAttention::empty_dtype(n_state, n_head, dtype.clone()),
            cross_attn_ln: layer_norm(n_state, dtype.clone()),
            mlp0: linear(n_state, mlp, Bias::FanIn, dtype.clone()),
            mlp2: linear(mlp, n_state, Bias::FanIn, dtype.clone()),
            mlp_ln: layer_norm(n_state, dtype),
            n_state,
        }
    }

    /// Forward with SDPA (standard path). `xa` is the encoder output.
    pub fn forward(&self, x: &Tensor, xa: &Tensor, mask: &Tensor) -> Result<Tensor> {
        // Self-attention (causal)
        let h = scoped("attn_ln", || self.attn_ln.forward(x))?;
        let attn_out = scoped("attn", || self.attn.forward(&h, None, Some(mask)))?;
        let x = x.try_add(&attn_out)?;

        // Cross-attention
        let h = scoped("cross_attn_ln", || self.cross_attn_ln.forward(&x))?;
        let cross_out = scoped("cross_attn", || self.cross_attn.forward(&h, Some(xa), None))?;
        let x = x.try_add(&cross_out)?;

        // MLP
        let h = scoped("mlp_ln", || self.mlp_ln.forward(&x))?;
        let h = self.mlp(&h)?;
        Ok(x.try_add(&h)?)
    }

    /// The two-layer MLP epilogue, shared by every decoder entry point.
    fn mlp(&self, h: &Tensor) -> Result<Tensor> {
        linear_forward(&self.mlp2, &linear_forward(&self.mlp0, h)?.gelu_exact()?)
    }
}

/// Whisper text decoder: token embedding + learned positional embedding +
/// N × DecoderBlock + LayerNorm + tied output projection.
#[derive(Clone, Module)]
pub struct TextDecoder {
    #[module(key = "token_embedding.weight")]
    pub token_embedding: Tensor, // [n_vocab, D]
    pub positional_embedding: Tensor, // [n_text_ctx, D]
    pub blocks: Vec<DecoderBlock>,
    pub ln: LayerNorm,
    pub n_state: usize,
    pub n_head: usize,
    pub n_text_ctx: usize,
    #[module(skip)]
    activation_dtype: DType,
}

impl TextDecoder {
    pub fn empty(dims: &ModelDimensions) -> Self {
        let n_state = dims.n_text_state;
        let dtype = dims.dtype.clone();
        Self {
            token_embedding: fan_in_uniform(&[dims.n_vocab, n_state], n_state, dtype.clone()),
            positional_embedding: Tensor::zeros(&[dims.n_text_ctx, n_state], dtype.clone()),
            blocks: (0..dims.n_text_layer)
                .map(|_| DecoderBlock::empty_dtype(n_state, dims.n_text_head, dtype.clone()))
                .collect(),
            ln: layer_norm(n_state, dtype),
            n_state,
            n_head: dims.n_text_head,
            n_text_ctx: dims.n_text_ctx,
            activation_dtype: dims.dtype.clone(),
        }
    }

    fn pack_kv(kvs: Vec<Tensor>) -> Result<Tensor> {
        let permuted: Vec<Tensor> =
            kvs.into_iter().map(|tensor| Ok(tensor.try_permute(&[0, 2, 1, 3])?)).collect::<Result<Vec<_>>>()?;
        Ok(Tensor::cat(&permuted.iter().collect::<Vec<_>>(), 2)?)
    }

    /// Project encoder features into the fixed packed cross-attention cache.
    /// This graph is independent of decoder tokens and runs once per window.
    pub fn project_cross_kv(&self, xa: &Tensor) -> Result<(Tensor, Tensor)> {
        let xa = xa.cast(self.activation_dtype.clone());
        let mut cross_ks = Vec::with_capacity(self.blocks.len());
        let mut cross_vs = Vec::with_capacity(self.blocks.len());
        for (index, block) in self.blocks.iter().enumerate() {
            let _origin = scope_index("blocks", index);
            // Keep each GEMM independent from the final layer/head packing.
            let k = scoped("cross_attn", || scoped("key", || linear_forward(&block.cross_attn.key, &xa)))?.contiguous();
            let v =
                scoped("cross_attn", || scoped("value", || linear_forward(&block.cross_attn.value, &xa)))?.contiguous();
            cross_ks.push(k.split_heads(self.n_head)?);
            cross_vs.push(v.split_heads(self.n_head)?);
        }
        Ok((Self::pack_kv(cross_ks)?.cast(DType::Float32), Self::pack_kv(cross_vs)?.cast(DType::Float32)))
    }

    /// Forward pass producing logits for all positions.
    /// `tokens`: `[B, L]` int tensor. `xa`: `[B, T_enc, D]` encoder output.
    /// `offset`: positional embedding offset (for KV-cached incremental decoding).
    pub fn forward(&self, tokens: &Tensor, xa: &Tensor, offset: usize) -> Result<Tensor> {
        let seq_len = tokens.dim_const(1)?;

        // Token embedding: [B, L, D]
        let tok_emb = self.token_embedding.embedding(tokens)?;

        // Positional embedding slice: [L, D]
        let pos_emb = self.positional_embedding.narrow(0, offset, seq_len)?;

        let x = tok_emb.try_add(&pos_emb)?;
        let x = x.cast(self.activation_dtype.clone());
        let xa = xa.cast(self.activation_dtype.clone());

        let mask = Tensor::causal_mask(seq_len, x.dtype())?;

        let mut x = x;
        for (index, block) in self.blocks.iter().enumerate() {
            x = scoped_index("blocks", index, || block.forward(&x, &xa, &mask))?;
        }

        // Final LayerNorm
        let x = scoped("ln", || self.ln.forward(&x))?;

        // Tied output: logits = x @ token_embedding.T  → [B, L, n_vocab]
        let output_weight = self.token_embedding.cast(x.dtype());
        let logits = x.linear().weight(&output_weight).call()?;
        Ok(logits.cast(DType::Float32))
    }

    /// Teacher-forced decoder pass over packed cross K/V, returning raw scaled
    /// QK scores for the statically selected alignment heads.
    pub fn forward_alignment(
        &self,
        tokens: &Tensor,
        cross_k: &Tensor,
        cross_v: &Tensor,
        alignment_heads: &[(usize, usize)],
    ) -> Result<Tensor> {
        let seq_len = tokens.dim_const(1)?;

        let tok_emb = self.token_embedding.embedding(tokens)?;

        let pos_emb = self.positional_embedding.narrow(0, 0usize, seq_len)?;

        let x = tok_emb.try_add(&pos_emb)?;
        let x = x.cast(self.activation_dtype.clone());
        let cross_k = cross_k.cast(self.activation_dtype.clone());
        let cross_v = cross_v.cast(self.activation_dtype.clone());

        let mask = Tensor::causal_mask(seq_len, x.dtype())?;

        let mut x = x;
        let mut selected_qk: Vec<Option<Tensor>> = (0..alignment_heads.len()).map(|_| None).collect();
        for (layer, block) in self.blocks.iter().enumerate() {
            let _origin = scope_index("blocks", layer);
            let h = scoped("attn_ln", || block.attn_ln.forward(&x))?;
            let attn_out = scoped("attn", || block.attn.forward(&h, None, Some(&mask)))?;
            x = x.try_add(&attn_out)?;

            let h = block.cross_attn_ln.forward(&x)?;
            let query = linear_forward(&block.cross_attn.query, &h)?.split_heads(self.n_head)?;
            let head_start = layer * self.n_head;
            let layer_ck = cross_k.narrow(2, head_start, self.n_head)?.try_permute(&[0, 2, 1, 3])?;
            let layer_cv = cross_v.narrow(2, head_start, self.n_head)?.try_permute(&[0, 2, 1, 3])?;
            let cross_out =
                query.scaled_dot_product_attention().key(&layer_ck).value(&layer_cv).is_causal(false).call()?;
            let cross_out = linear_forward(&block.cross_attn.out, &cross_out.merge_heads()?)?;
            x = x.try_add(&cross_out)?;

            let layer_heads: Vec<(usize, usize)> = alignment_heads
                .iter()
                .enumerate()
                .filter_map(|(selected, &(selected_layer, head))| (selected_layer == layer).then_some((selected, head)))
                .collect();
            for (selected, head) in layer_heads {
                let selected_q = query.narrow(1, head, 1usize)?;
                let selected_k = layer_ck.narrow(1, head, 1usize)?;
                let scores = selected_q.matmul(&selected_k.try_transpose(-1, -2)?)?;
                let scale = ((self.n_state / self.n_head) as f64).sqrt().recip();
                selected_qk[selected] = Some(scores.try_mul(scale)?);
            }

            let h = block.mlp_ln.forward(&x)?;
            let h = block.mlp(&h)?;
            x = x.try_add(&h)?;
        }
        let selected_qk = selected_qk
            .into_iter()
            .map(|qk| {
                qk.ok_or_else(|| super::error::Error::Tensor {
                    source: Box::new(
                        svod_tensor::error::ErrorKind::SymbolicShapeUnsupported {
                            operation: "alignment head layer out of range".into(),
                        }
                        .into(),
                    ),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Tensor::cat(&selected_qk.iter().collect::<Vec<_>>(), 1)?.cast(DType::Float32))
    }

    /// Prefill consuming fixed packed cross-attention caches.
    /// Returns `(logits[1, init_len, n_vocab], self_k, self_v)`, where each packed
    /// self K/V is [1, seq_len, n_layer*H, Dh].
    #[allow(clippy::too_many_arguments)]
    pub fn forward_prefill(
        &self,
        tokens: &Tensor,
        cross_k: &Tensor,
        cross_v: &Tensor,
        offset: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let seq_len = tokens.dim_const(1)?;

        let tok_emb = self.token_embedding.embedding(tokens)?;
        let pos_emb = self.positional_embedding.narrow(0, offset, seq_len)?;

        let x = tok_emb.try_add(&pos_emb)?;
        let x = x.cast(self.activation_dtype.clone());
        let cross_k = cross_k.cast(self.activation_dtype.clone());
        let cross_v = cross_v.cast(self.activation_dtype.clone());

        let mask = Tensor::causal_mask(seq_len, x.dtype())?;

        let mut x = x;
        let mut self_ks: Vec<Tensor> = Vec::with_capacity(self.blocks.len());
        let mut self_vs: Vec<Tensor> = Vec::with_capacity(self.blocks.len());

        for (layer, block) in self.blocks.iter().enumerate() {
            let _origin = scope_index("blocks", layer);
            let h = scoped("attn_ln", || block.attn_ln.forward(&x))?;
            let (attn_out, sk, sv) = scoped("attn", || block.attn.forward_return_kv(&h, None, Some(&mask)))?;
            x = x.try_add(&attn_out)?;

            self_ks.push(sk.split_heads(self.n_head)?);
            self_vs.push(sv.split_heads(self.n_head)?);

            let h = block.cross_attn_ln.forward(&x)?;
            let query = linear_forward(&block.cross_attn.query, &h)?.split_heads(self.n_head)?;
            let head_start = layer * self.n_head;
            let layer_ck = cross_k.narrow(2, head_start, self.n_head)?.try_permute(&[0, 2, 1, 3])?;
            let layer_cv = cross_v.narrow(2, head_start, self.n_head)?.try_permute(&[0, 2, 1, 3])?;
            let cross_out =
                query.scaled_dot_product_attention().key(&layer_ck).value(&layer_cv).is_causal(false).call()?;
            let cross_out = linear_forward(&block.cross_attn.out, &cross_out.merge_heads()?)?;
            x = x.try_add(&cross_out)?;

            let h = block.mlp_ln.forward(&x)?;
            let h = block.mlp(&h)?;
            x = x.try_add(&h)?;
        }

        let x = scoped("ln", || self.ln.forward(&x))?;
        let logits = x.linear().weight(&self.token_embedding.cast(x.dtype())).call()?.cast(DType::Float32);

        // K/V cache outputs cast to fp32 — the cache buffers are fp32 (host
        // round-trips them as Vec<f32>), while compute is dims.dtype (fp16).
        Ok((logits, Self::pack_kv(self_ks)?.cast(DType::Float32), Self::pack_kv(self_vs)?.cast(DType::Float32)))
    }

    /// Decoder logits using an already prepared cross-attention cache.
    pub fn forward_with_cross_kv(
        &self,
        tokens: &Tensor,
        cross_k: &Tensor,
        cross_v: &Tensor,
        offset: usize,
    ) -> Result<Tensor> {
        let (logits, _, _) = self.forward_prefill(tokens, cross_k, cross_v, offset)?;
        Ok(logits)
    }

    /// Single-token forward with KV cache. Used for incremental decoding.
    /// Works for any batch size B (B=1 for greedy, B=beam_size for beam search).
    ///
    /// - `token`: [B, 1] int32
    /// - `pos_emb`: [B, 1, D] positional embedding for this position
    /// - `self_k_cache`: [B, max_len, n_layer*H, Dh] self-attn K cache
    /// - `self_v_cache`: [B, max_len, n_layer*H, Dh] self-attn V cache
    /// - `cross_k`: [B, n_audio_ctx, n_layer*H, Dh] cross-attn K (fixed)
    /// - `cross_v`: [B, n_audio_ctx, n_layer*H, Dh] cross-attn V (fixed)
    /// - `self_key_lens`: [B] i32 valid cached-key counts for self-attn
    ///
    /// Returns `(logits[B, n_vocab], new_self_k[B, 1, n_layer*H, Dh], new_self_v[...])`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_step(
        &self,
        token: &Tensor,
        pos_emb: &Tensor,
        self_k_cache: &Tensor,
        self_v_cache: &Tensor,
        cross_k: &Tensor,
        cross_v: &Tensor,
        self_key_lens: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        self.forward_step_with_config(
            token,
            pos_emb,
            self_k_cache,
            self_v_cache,
            cross_k,
            cross_v,
            self_key_lens,
            StepAttentionConfig::default(),
        )
    }

    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_step_with_attention_mode(
        &self,
        token: &Tensor,
        pos_emb: &Tensor,
        self_k_cache: &Tensor,
        self_v_cache: &Tensor,
        cross_k: &Tensor,
        cross_v: &Tensor,
        self_key_lens: &Tensor,
        mode: StepAttentionMode,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        self.forward_step_with_config(
            token,
            pos_emb,
            self_k_cache,
            self_v_cache,
            cross_k,
            cross_v,
            self_key_lens,
            mode.into(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_step_with_config(
        &self,
        token: &Tensor,
        pos_emb: &Tensor,
        self_k_cache: &Tensor,
        self_v_cache: &Tensor,
        cross_k: &Tensor,
        cross_v: &Tensor,
        self_key_lens: &Tensor,
        attention: StepAttentionConfig,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let n_head = self.n_head;
        let n_layer = self.blocks.len();
        let d_head = self.n_state / n_head;

        // Infer batch from token shape
        let batch = token.dim_const(0)?;
        let self_key_count = self_k_cache.dim_const(1)? + 1;
        let cross_key_count = cross_k.dim_const(1)?;
        let cross_splits = attention
            .cross_splits
            .unwrap_or_else(|| if cross_key_count >= 1000 && cross_key_count.is_multiple_of(4) { 4 } else { 1 });

        // Embed single token + positional embedding
        let tok_emb = self.token_embedding.embedding(token)?;
        let x = tok_emb.try_add(pos_emb)?;
        let x = x.cast(self.activation_dtype.clone());

        let mut x = x;
        let mut new_ks: Vec<Tensor> = Vec::with_capacity(n_layer);
        let mut new_vs: Vec<Tensor> = Vec::with_capacity(n_layer);

        for (l, block) in self.blocks.iter().enumerate() {
            let _origin = scope_index("blocks", l);
            let lh_start = l * n_head;

            // ── Self-attn with cache ─────────────────────────────────────
            let h = scoped("attn_ln", || block.attn_ln.forward(&x))?;
            let q = linear_forward(&block.attn.query, &h)?;
            let new_k_raw = linear_forward(&block.attn.key, &h)?;
            let new_v_raw = linear_forward(&block.attn.value, &h)?;

            // Sequence-major projections are consumed directly by the custom path.
            let q_seq = q.try_reshape([batch, 1, n_head, d_head])?;
            let new_k_seq = new_k_raw.try_reshape([batch, 1, n_head, d_head])?;
            let new_v_seq = new_v_raw.try_reshape([batch, 1, n_head, d_head])?;
            let new_k_h = new_k_raw.split_heads(n_head)?;
            let new_v_h = new_v_raw.split_heads(n_head)?;

            // Slice this layer's cached K/V: [B, max_len, n_layer*H, Dh]
            // → [B, max_len, H, Dh].
            let cached_k = self_k_cache.narrow(2, lh_start, n_head)?;
            let cached_v = self_v_cache.narrow(2, lh_start, n_head)?;

            // Concatenate cached K/V with new K/V along seq dim:
            // [B, max_len, H, Dh] cat [B, 1, H, Dh] → [B, max_len+1, H, Dh]
            let full_k = Tensor::cat(&[&cached_k, &new_k_seq], 1)?;
            let full_v = Tensor::cat(&[&cached_v, &new_v_seq], 1)?;

            let direct = if attention.custom_self {
                svod_tk::single_query_attention(
                    &q_seq.cast(DType::Float32),
                    &full_k,
                    &full_v,
                    svod_tk::SqAttentionOpts { key_lens: Some(self_key_lens), include_last: true, split: 1 },
                )
                .map_err(tk_launch_error)?
            } else {
                None
            };
            let attn_out = match direct {
                Some(out) => out.try_reshape([batch, 1, self.n_state])?.cast(self.activation_dtype.clone()),
                None => {
                    let q_h = q_seq.try_permute(&[0, 2, 1, 3])?;
                    let full_k_h = full_k.cast(self.activation_dtype.clone()).try_permute(&[0, 2, 1, 3])?;
                    let full_v_h = full_v.cast(self.activation_dtype.clone()).try_permute(&[0, 2, 1, 3])?;
                    let valid = cached_step_mask(self_key_lens, self_key_count)?;
                    let out = q_h
                        .scaled_dot_product_attention()
                        .key(&full_k_h)
                        .value(&full_v_h)
                        .key_padding_mask(&valid)
                        .is_causal(false)
                        .call()?;
                    out.merge_heads()?
                }
            };
            let attn_out = linear_forward(&block.attn.out, &attn_out)?;
            x = x.try_add(&attn_out)?;

            // ── Cross-attn (fixed cache, no mask) ────────────────────────
            let h = block.cross_attn_ln.forward(&x)?;
            let cq = linear_forward(&block.cross_attn.query, &h)?;
            let cq_seq = cq.try_reshape([batch, 1, n_head, d_head])?;

            let direct = if attention.custom_cross {
                svod_tk::single_query_attention_packed(
                    &cq_seq.cast(DType::Float32),
                    cross_k,
                    cross_v,
                    lh_start,
                    svod_tk::SqAttentionOpts { split: cross_splits, ..Default::default() },
                )
                .map_err(tk_launch_error)?
            } else {
                None
            };
            let cross_out = match direct {
                Some(out) => out.try_reshape([batch, 1, self.n_state])?.cast(self.activation_dtype.clone()),
                None => {
                    let layer_ck = cross_k.narrow(2, lh_start, n_head)?.cast(self.activation_dtype.clone());
                    let layer_cv = cross_v.narrow(2, lh_start, n_head)?.cast(self.activation_dtype.clone());
                    let cq_h = cq_seq.try_permute(&[0, 2, 1, 3])?;
                    let layer_ck_h = layer_ck.try_permute(&[0, 2, 1, 3])?;
                    let layer_cv_h = layer_cv.try_permute(&[0, 2, 1, 3])?;
                    let out = cq_h
                        .scaled_dot_product_attention()
                        .key(&layer_ck_h)
                        .value(&layer_cv_h)
                        .is_causal(false)
                        .call()?;
                    out.merge_heads()?
                }
            };
            let cross_out = linear_forward(&block.cross_attn.out, &cross_out)?;
            x = x.try_add(&cross_out)?;

            // ── MLP ───────────────────────────────────────────────────────
            let h = block.mlp_ln.forward(&x)?;
            let h = block.mlp(&h)?;
            x = x.try_add(&h)?;

            // Collect new K/V for cache update: [B, H, 1, Dh]
            new_ks.push(new_k_h);
            new_vs.push(new_v_h);
        }

        // Permute each layer's K/V from [B, H, 1, Dh] to [B, 1, H, Dh],
        // then cat along dim 1 → [B, n_layer, H, Dh] → reshape [B, 1, n_layer*H, Dh].
        // Catting along dim 0 would interleave beams and layers for B > 1.
        let permuted_ks: Vec<Tensor> =
            new_ks.iter().map(|t| Ok(t.try_permute(&[0, 2, 1, 3])?)).collect::<Result<Vec<_>>>()?;
        let permuted_vs: Vec<Tensor> =
            new_vs.iter().map(|t| Ok(t.try_permute(&[0, 2, 1, 3])?)).collect::<Result<Vec<_>>>()?;

        let stacked_k = Tensor::cat(&permuted_ks.iter().collect::<Vec<_>>(), 1)?;
        let stacked_v = Tensor::cat(&permuted_vs.iter().collect::<Vec<_>>(), 1)?;

        let packed = [batch, 1, n_layer * n_head, d_head];
        let new_k_flat = stacked_k.try_reshape(packed)?;
        let new_v_flat = stacked_v.try_reshape(packed)?;
        let x = scoped("ln", || self.ln.forward(&x))?;
        let logits = x.linear().weight(&self.token_embedding.cast(x.dtype())).call()?.cast(DType::Float32);

        // logits is [B, 1, n_vocab] → reshape to [B, n_vocab]
        let n_vocab = self.token_embedding.dim_const(0)?;
        let logits = logits.try_reshape([batch, n_vocab])?;

        // K/V outputs cast to fp32 — appended into the fp32 cache buffer via SDMA.
        Ok((logits, new_k_flat.cast(DType::Float32), new_v_flat.cast(DType::Float32)))
    }
}
