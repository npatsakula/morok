//! Text decoder: token + positional embeddings + self/cross-attention transformer blocks.

use snafu::ResultExt;
use svod_dtype::DType;
use svod_ir::ConstValue;
use svod_tensor::Tensor;

use crate::init::fan_in_uniform;
use crate::state::{self, HasStateDict, StateDict, get_tensor, prefixed, scope_index, scoped, scoped_index};

use super::attention::{MultiHeadAttention, causal_mask};
use super::blocks::{LayerNormWeights, linear_with_bias};
use super::config::ModelDimensions;
use super::error::{Result, TensorSnafu, tk_launch_error};

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

pub(crate) fn cached_step_mask(key_lens: &Tensor, batch: usize, key_count: usize) -> Result<Tensor> {
    let range = Tensor::arange(key_count as i64, None, None)
        .context(TensorSnafu)?
        .try_reshape([1usize, 1, 1, key_count])
        .context(TensorSnafu)?;
    let lens = key_lens.try_reshape([batch, 1, 1, 1]).context(TensorSnafu)?;
    let beyond_prefix = range.try_ge(&lens).context(TensorSnafu)?;
    let final_key = Tensor::const_(ConstValue::Int(key_count as i64 - 1), DType::Int32);
    let not_final = range.try_ne(&final_key).context(TensorSnafu)?;
    beyond_prefix.try_bitand(&not_final).context(TensorSnafu)
}

/// Decoder transformer block: self-attn + cross-attn + MLP, all pre-norm.
#[derive(Clone)]
pub struct DecoderBlock {
    pub attn: MultiHeadAttention,
    pub attn_ln: LayerNormWeights,
    pub cross_attn: MultiHeadAttention,
    pub cross_attn_ln: LayerNormWeights,
    pub mlp0_w: Tensor,
    pub mlp0_b: Tensor,
    pub mlp1_w: Tensor,
    pub mlp1_b: Tensor,
    pub mlp_ln: LayerNormWeights,
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
            attn_ln: LayerNormWeights::empty_dtype(n_state, dtype.clone()),
            cross_attn: MultiHeadAttention::empty_dtype(n_state, n_head, dtype.clone()),
            cross_attn_ln: LayerNormWeights::empty_dtype(n_state, dtype.clone()),
            mlp0_w: fan_in_uniform(&[mlp, n_state], n_state, dtype.clone()),
            mlp0_b: fan_in_uniform(&[mlp], n_state, dtype.clone()),
            mlp1_w: fan_in_uniform(&[n_state, mlp], mlp, dtype.clone()),
            mlp1_b: fan_in_uniform(&[n_state], mlp, dtype.clone()),
            mlp_ln: LayerNormWeights::empty_dtype(n_state, dtype),
            n_state,
        }
    }

    /// Forward with SDPA (standard path). `xa` is the encoder output.
    pub fn forward(&self, x: &Tensor, xa: &Tensor, mask: &Tensor) -> Result<Tensor> {
        // Self-attention (causal)
        let h = scoped("attn_ln", || self.attn_ln.apply(x))?;
        let attn_out = scoped("attn", || self.attn.forward(&h, None, Some(mask)))?;
        let x = x.try_add(&attn_out).context(TensorSnafu)?;

        // Cross-attention
        let h = scoped("cross_attn_ln", || self.cross_attn_ln.apply(&x))?;
        let cross_out = scoped("cross_attn", || self.cross_attn.forward(&h, Some(xa), None))?;
        let x = x.try_add(&cross_out).context(TensorSnafu)?;

        // MLP
        let h = scoped("mlp_ln", || self.mlp_ln.apply(&x))?;
        let h = linear_with_bias(&h, &self.mlp0_w, &self.mlp0_b)?;
        let h = h.gelu_exact().context(TensorSnafu)?;
        let h = linear_with_bias(&h, &self.mlp1_w, &self.mlp1_b)?;
        let x = x.try_add(&h).context(TensorSnafu)?;
        Ok(x)
    }
}

impl HasStateDict for DecoderBlock {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.extend(self.attn.state_dict(&prefixed(prefix, "attn")));
        sd.extend(self.attn_ln.state_dict(&prefixed(prefix, "attn_ln")));
        sd.extend(self.cross_attn.state_dict(&prefixed(prefix, "cross_attn")));
        sd.extend(self.cross_attn_ln.state_dict(&prefixed(prefix, "cross_attn_ln")));
        sd.insert(prefixed(prefix, "mlp.0.weight"), self.mlp0_w.clone());
        sd.insert(prefixed(prefix, "mlp.0.bias"), self.mlp0_b.clone());
        sd.insert(prefixed(prefix, "mlp.2.weight"), self.mlp1_w.clone());
        sd.insert(prefixed(prefix, "mlp.2.bias"), self.mlp1_b.clone());
        sd.extend(self.mlp_ln.state_dict(&prefixed(prefix, "mlp_ln")));
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.attn.load_state_dict(sd, &prefixed(prefix, "attn"))?;
        self.attn_ln.load_state_dict(sd, &prefixed(prefix, "attn_ln"))?;
        self.cross_attn.load_state_dict(sd, &prefixed(prefix, "cross_attn"))?;
        self.cross_attn_ln.load_state_dict(sd, &prefixed(prefix, "cross_attn_ln"))?;
        self.mlp0_w = get_tensor(sd, &prefixed(prefix, "mlp.0.weight"))?;
        self.mlp0_b = get_tensor(sd, &prefixed(prefix, "mlp.0.bias"))?;
        self.mlp1_w = get_tensor(sd, &prefixed(prefix, "mlp.2.weight"))?;
        self.mlp1_b = get_tensor(sd, &prefixed(prefix, "mlp.2.bias"))?;
        self.mlp_ln.load_state_dict(sd, &prefixed(prefix, "mlp_ln"))?;
        Ok(())
    }
}

/// Whisper text decoder: token embedding + learned positional embedding +
/// N × DecoderBlock + LayerNorm + tied output projection.
#[derive(Clone)]
pub struct TextDecoder {
    pub token_embedding: Tensor,      // [n_vocab, D]
    pub positional_embedding: Tensor, // [n_text_ctx, D]
    pub blocks: Vec<DecoderBlock>,
    pub ln: LayerNormWeights,
    pub n_state: usize,
    pub n_head: usize,
    pub n_text_ctx: usize,
    activation_dtype: DType,
}

impl TextDecoder {
    pub fn empty(dims: &ModelDimensions) -> Self {
        let n_state = dims.n_text_state;
        let dtype = dims.dtype.clone();
        Self {
            token_embedding: fan_in_uniform(&[dims.n_vocab, n_state], n_state, dtype.clone()),
            positional_embedding: Tensor::zeros(&[dims.n_text_ctx, n_state], dtype.clone())
                .expect("positional embedding"),
            blocks: (0..dims.n_text_layer)
                .map(|_| DecoderBlock::empty_dtype(n_state, dims.n_text_head, dtype.clone()))
                .collect(),
            ln: LayerNormWeights::empty_dtype(n_state, dtype),
            n_state,
            n_head: dims.n_text_head,
            n_text_ctx: dims.n_text_ctx,
            activation_dtype: dims.dtype.clone(),
        }
    }

    fn pack_kv(kvs: Vec<Tensor>) -> Result<Tensor> {
        let permuted: Vec<Tensor> = kvs
            .into_iter()
            .map(|tensor| tensor.try_permute(&[0, 2, 1, 3]).context(TensorSnafu))
            .collect::<Result<Vec<_>>>()?;
        Tensor::cat(&permuted.iter().collect::<Vec<_>>(), 2).context(TensorSnafu)
    }

    /// Project encoder features into the fixed packed cross-attention cache.
    /// This graph is independent of decoder tokens and runs once per window.
    pub fn project_cross_kv(&self, xa: &Tensor) -> Result<(Tensor, Tensor)> {
        let xa = xa.cast(self.activation_dtype.clone()).context(TensorSnafu)?;
        let mut cross_ks = Vec::with_capacity(self.blocks.len());
        let mut cross_vs = Vec::with_capacity(self.blocks.len());
        for (index, block) in self.blocks.iter().enumerate() {
            let _origin = scope_index("blocks", index);
            // Keep each GEMM independent from the final layer/head packing.
            let k = scoped("cross_attn", || scoped("key", || block.cross_attn.key.forward(&xa)))?.contiguous();
            let v = scoped("cross_attn", || scoped("value", || block.cross_attn.value.forward(&xa)))?.contiguous();
            cross_ks.push(block.cross_attn.split_heads(&k)?);
            cross_vs.push(block.cross_attn.split_heads(&v)?);
        }
        Ok((
            Self::pack_kv(cross_ks)?.cast(DType::Float32).context(TensorSnafu)?,
            Self::pack_kv(cross_vs)?.cast(DType::Float32).context(TensorSnafu)?,
        ))
    }

    /// Forward pass producing logits for all positions.
    /// `tokens`: `[B, L]` int tensor. `xa`: `[B, T_enc, D]` encoder output.
    /// `offset`: positional embedding offset (for KV-cached incremental decoding).
    pub fn forward(&self, tokens: &Tensor, xa: &Tensor, offset: usize) -> Result<Tensor> {
        let seq_len =
            tokens.shape().context(TensorSnafu)?[1].as_const().ok_or_else(|| super::error::Error::Tensor {
                source: Box::new(svod_tensor::error::Error::SymbolicShapeUnsupported {
                    operation: "decoder forward seq_len".into(),
                }),
            })?;

        // Token embedding: [B, L, D]
        let tok_emb = self.token_embedding.embedding(tokens).context(TensorSnafu)?;

        // Positional embedding slice: [L, D]
        let pos_emb = self
            .positional_embedding
            .try_shrink([Some((offset as isize, (offset + seq_len) as isize)), None])
            .context(TensorSnafu)?;

        let x = tok_emb.try_add(&pos_emb).context(TensorSnafu)?;
        let x = x.cast(self.activation_dtype.clone()).context(TensorSnafu)?;
        let xa = xa.cast(self.activation_dtype.clone()).context(TensorSnafu)?;

        let mask = causal_mask(seq_len, x.uop().dtype().clone())?;

        let mut x = x;
        for (index, block) in self.blocks.iter().enumerate() {
            x = scoped_index("blocks", index, || block.forward(&x, &xa, &mask))?;
        }

        // Final LayerNorm
        let x = scoped("ln", || self.ln.apply(&x))?;

        // Tied output: logits = x @ token_embedding.T  → [B, L, n_vocab]
        let output_weight = self.token_embedding.cast(x.uop().dtype()).context(TensorSnafu)?;
        let logits = x.linear().weight(&output_weight).call().context(TensorSnafu)?;
        logits.cast(DType::Float32).context(TensorSnafu)
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
        let seq_len =
            tokens.shape().context(TensorSnafu)?[1].as_const().ok_or_else(|| super::error::Error::Tensor {
                source: Box::new(svod_tensor::error::Error::SymbolicShapeUnsupported {
                    operation: "decoder alignment seq_len".into(),
                }),
            })?;

        let tok_emb = self.token_embedding.embedding(tokens).context(TensorSnafu)?;

        let pos_emb =
            self.positional_embedding.try_shrink([Some((0isize, seq_len as isize)), None]).context(TensorSnafu)?;

        let x = tok_emb.try_add(&pos_emb).context(TensorSnafu)?;
        let x = x.cast(self.activation_dtype.clone()).context(TensorSnafu)?;
        let cross_k = cross_k.cast(self.activation_dtype.clone()).context(TensorSnafu)?;
        let cross_v = cross_v.cast(self.activation_dtype.clone()).context(TensorSnafu)?;

        let mask = causal_mask(seq_len, x.uop().dtype().clone())?;

        let mut x = x;
        let mut selected_qk: Vec<Option<Tensor>> = (0..alignment_heads.len()).map(|_| None).collect();
        for (layer, block) in self.blocks.iter().enumerate() {
            let _origin = scope_index("blocks", layer);
            let h = scoped("attn_ln", || block.attn_ln.apply(&x))?;
            let attn_out = scoped("attn", || block.attn.forward(&h, None, Some(&mask)))?;
            x = x.try_add(&attn_out).context(TensorSnafu)?;

            let h = block.cross_attn_ln.apply(&x)?;
            let query = block.cross_attn.split_heads(&block.cross_attn.query.forward(&h)?)?;
            let head_start = layer * self.n_head;
            let head_end = head_start + self.n_head;
            let layer_ck = cross_k
                .try_shrink([None, None, Some((head_start as isize, head_end as isize)), None])
                .context(TensorSnafu)?
                .try_permute(&[0, 2, 1, 3])
                .context(TensorSnafu)?;
            let layer_cv = cross_v
                .try_shrink([None, None, Some((head_start as isize, head_end as isize)), None])
                .context(TensorSnafu)?
                .try_permute(&[0, 2, 1, 3])
                .context(TensorSnafu)?;
            let cross_out = query
                .scaled_dot_product_attention()
                .key(&layer_ck)
                .value(&layer_cv)
                .is_causal(false)
                .call()
                .context(TensorSnafu)?;
            let cross_out = block.cross_attn.merge_heads(&cross_out)?;
            let cross_out = block.cross_attn.out.forward(&cross_out)?;
            x = x.try_add(&cross_out).context(TensorSnafu)?;

            let layer_heads: Vec<(usize, usize)> = alignment_heads
                .iter()
                .enumerate()
                .filter_map(|(selected, &(selected_layer, head))| (selected_layer == layer).then_some((selected, head)))
                .collect();
            for (selected, head) in layer_heads {
                let selected_q = query
                    .try_shrink([None, Some((head as isize, head as isize + 1)), None, None])
                    .context(TensorSnafu)?;
                let selected_k = layer_ck
                    .try_shrink([None, Some((head as isize, head as isize + 1)), None, None])
                    .context(TensorSnafu)?;
                let kt = selected_k.try_transpose(-1, -2).context(TensorSnafu)?;
                let scores = selected_q.matmul(&kt).context(TensorSnafu)?;
                let scale = Tensor::const_(
                    ConstValue::Float(1.0 / ((self.n_state / self.n_head) as f64).sqrt()),
                    scores.uop().dtype().clone(),
                );
                selected_qk[selected] = Some(scores.try_mul(&scale).context(TensorSnafu)?);
            }

            let h = block.mlp_ln.apply(&x)?;
            let h = linear_with_bias(&h, &block.mlp0_w, &block.mlp0_b)?;
            let h = h.gelu_exact().context(TensorSnafu)?;
            let h = linear_with_bias(&h, &block.mlp1_w, &block.mlp1_b)?;
            x = x.try_add(&h).context(TensorSnafu)?;
        }
        let selected_qk = selected_qk
            .into_iter()
            .map(|qk| {
                qk.ok_or_else(|| super::error::Error::Tensor {
                    source: Box::new(svod_tensor::error::Error::SymbolicShapeUnsupported {
                        operation: "alignment head layer out of range".into(),
                    }),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Tensor::cat(&selected_qk.iter().collect::<Vec<_>>(), 1)
            .context(TensorSnafu)?
            .cast(DType::Float32)
            .context(TensorSnafu)
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
        let seq_len =
            tokens.shape().context(TensorSnafu)?[1].as_const().ok_or_else(|| super::error::Error::Tensor {
                source: Box::new(svod_tensor::error::Error::SymbolicShapeUnsupported {
                    operation: "decoder prefill seq_len".into(),
                }),
            })?;

        let tok_emb = self.token_embedding.embedding(tokens).context(TensorSnafu)?;
        let pos_emb = self
            .positional_embedding
            .try_shrink([Some((offset as isize, (offset + seq_len) as isize)), None])
            .context(TensorSnafu)?;

        let x = tok_emb.try_add(&pos_emb).context(TensorSnafu)?;
        let x = x.cast(self.activation_dtype.clone()).context(TensorSnafu)?;
        let cross_k = cross_k.cast(self.activation_dtype.clone()).context(TensorSnafu)?;
        let cross_v = cross_v.cast(self.activation_dtype.clone()).context(TensorSnafu)?;

        let mask = causal_mask(seq_len, x.uop().dtype().clone())?;

        let mut x = x;
        let mut self_ks: Vec<Tensor> = Vec::with_capacity(self.blocks.len());
        let mut self_vs: Vec<Tensor> = Vec::with_capacity(self.blocks.len());

        for (layer, block) in self.blocks.iter().enumerate() {
            let _origin = scope_index("blocks", layer);
            let h = scoped("attn_ln", || block.attn_ln.apply(&x))?;
            let (attn_out, sk, sv) = scoped("attn", || block.attn.forward_return_kv(&h, None, Some(&mask)))?;
            x = x.try_add(&attn_out).context(TensorSnafu)?;

            self_ks.push(block.attn.split_heads(&sk)?);
            self_vs.push(block.attn.split_heads(&sv)?);

            let h = block.cross_attn_ln.apply(&x)?;
            let query = block.cross_attn.split_heads(&block.cross_attn.query.forward(&h)?)?;
            let head_start = layer * self.n_head;
            let head_end = head_start + self.n_head;
            let layer_ck = cross_k
                .try_shrink([None, None, Some((head_start as isize, head_end as isize)), None])
                .context(TensorSnafu)?
                .try_permute(&[0, 2, 1, 3])
                .context(TensorSnafu)?;
            let layer_cv = cross_v
                .try_shrink([None, None, Some((head_start as isize, head_end as isize)), None])
                .context(TensorSnafu)?
                .try_permute(&[0, 2, 1, 3])
                .context(TensorSnafu)?;
            let cross_out = query
                .scaled_dot_product_attention()
                .key(&layer_ck)
                .value(&layer_cv)
                .is_causal(false)
                .call()
                .context(TensorSnafu)?;
            let cross_out = block.cross_attn.merge_heads(&cross_out)?;
            let cross_out = block.cross_attn.out.forward(&cross_out)?;
            x = x.try_add(&cross_out).context(TensorSnafu)?;

            let h = block.mlp_ln.apply(&x)?;
            let h = linear_with_bias(&h, &block.mlp0_w, &block.mlp0_b)?;
            let h = h.gelu_exact().context(TensorSnafu)?;
            let h = linear_with_bias(&h, &block.mlp1_w, &block.mlp1_b)?;
            x = x.try_add(&h).context(TensorSnafu)?;
        }

        let x = scoped("ln", || self.ln.apply(&x))?;
        let logits = x
            .linear()
            .weight(&self.token_embedding.cast(x.uop().dtype()).context(TensorSnafu)?)
            .call()
            .context(TensorSnafu)?
            .cast(DType::Float32)
            .context(TensorSnafu)?;

        // K/V cache outputs cast to fp32 — the cache buffers are fp32 (host
        // round-trips them as Vec<f32>), while compute is dims.dtype (fp16).
        Ok((
            logits,
            Self::pack_kv(self_ks)?.cast(DType::Float32).context(TensorSnafu)?,
            Self::pack_kv(self_vs)?.cast(DType::Float32).context(TensorSnafu)?,
        ))
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
        let token_shape = token.shape().context(TensorSnafu)?;
        let batch = token_shape[0].as_const().ok_or_else(|| super::error::Error::Tensor {
            source: Box::new(svod_tensor::error::Error::SymbolicShapeUnsupported {
                operation: "forward_step batch".into(),
            }),
        })?;
        let self_key_count =
            self_k_cache.shape().context(TensorSnafu)?[1].as_const().ok_or_else(|| super::error::Error::Tensor {
                source: Box::new(svod_tensor::error::Error::SymbolicShapeUnsupported {
                    operation: "forward_step cache length".into(),
                }),
            })? + 1;
        let cross_key_count =
            cross_k.shape().context(TensorSnafu)?[1].as_const().ok_or_else(|| super::error::Error::Tensor {
                source: Box::new(svod_tensor::error::Error::SymbolicShapeUnsupported {
                    operation: "forward_step cross cache length".into(),
                }),
            })?;
        let cross_splits = attention
            .cross_splits
            .unwrap_or_else(|| if cross_key_count >= 1000 && cross_key_count.is_multiple_of(4) { 4 } else { 1 });

        // Embed single token + positional embedding
        let tok_emb = self.token_embedding.embedding(token).context(TensorSnafu)?;
        let x = tok_emb.try_add(pos_emb).context(TensorSnafu)?;
        let x = x.cast(self.activation_dtype.clone()).context(TensorSnafu)?;

        let mut x = x;
        let mut new_ks: Vec<Tensor> = Vec::with_capacity(n_layer);
        let mut new_vs: Vec<Tensor> = Vec::with_capacity(n_layer);

        for (l, block) in self.blocks.iter().enumerate() {
            let _origin = scope_index("blocks", l);
            let lh_start = l * n_head;
            let lh_end = (l + 1) * n_head;

            // ── Self-attn with cache ─────────────────────────────────────
            let h = scoped("attn_ln", || block.attn_ln.apply(&x))?;
            let q = block.attn.query.forward(&h)?;
            let new_k_raw = block.attn.key.forward(&h)?;
            let new_v_raw = block.attn.value.forward(&h)?;

            // Sequence-major projections are consumed directly by the custom path.
            let q_seq = q.try_reshape([batch, 1, n_head, d_head]).context(TensorSnafu)?;
            let new_k_seq = new_k_raw.try_reshape([batch, 1, n_head, d_head]).context(TensorSnafu)?;
            let new_v_seq = new_v_raw.try_reshape([batch, 1, n_head, d_head]).context(TensorSnafu)?;
            let new_k_h = block.attn.split_heads(&new_k_raw)?;
            let new_v_h = block.attn.split_heads(&new_v_raw)?;

            // Slice this layer's cached K/V: [B, max_len, n_layer*H, Dh]
            // → [B, max_len, H, Dh].
            let cached_k = self_k_cache
                .try_shrink([None, None, Some((lh_start as isize, lh_end as isize)), None])
                .context(TensorSnafu)?;
            let cached_v = self_v_cache
                .try_shrink([None, None, Some((lh_start as isize, lh_end as isize)), None])
                .context(TensorSnafu)?;

            // Concatenate cached K/V with new K/V along seq dim:
            // [B, max_len, H, Dh] cat [B, 1, H, Dh] → [B, max_len+1, H, Dh]
            let full_k = Tensor::cat(&[&cached_k, &new_k_seq], 1).context(TensorSnafu)?;
            let full_v = Tensor::cat(&[&cached_v, &new_v_seq], 1).context(TensorSnafu)?;

            let direct = if attention.custom_self {
                svod_tk::single_query_attention(
                    &q_seq.cast(DType::Float32).context(TensorSnafu)?,
                    &full_k,
                    &full_v,
                    svod_tk::SqAttentionOpts { key_lens: Some(self_key_lens), include_last: true, split: 1 },
                )
                .map_err(tk_launch_error)
                .context(TensorSnafu)?
            } else {
                None
            };
            let attn_out = match direct {
                Some(out) => out
                    .try_reshape([batch, 1, self.n_state])
                    .context(TensorSnafu)?
                    .cast(self.activation_dtype.clone())
                    .context(TensorSnafu)?,
                None => {
                    let q_h = q_seq.try_permute(&[0, 2, 1, 3]).context(TensorSnafu)?;
                    let full_k_h = full_k
                        .cast(self.activation_dtype.clone())
                        .context(TensorSnafu)?
                        .try_permute(&[0, 2, 1, 3])
                        .context(TensorSnafu)?;
                    let full_v_h = full_v
                        .cast(self.activation_dtype.clone())
                        .context(TensorSnafu)?
                        .try_permute(&[0, 2, 1, 3])
                        .context(TensorSnafu)?;
                    let mask = cached_step_mask(self_key_lens, batch, self_key_count)?;
                    let out = q_h
                        .scaled_dot_product_attention()
                        .key(&full_k_h)
                        .value(&full_v_h)
                        .attn_mask(&mask)
                        .is_causal(false)
                        .call()
                        .context(TensorSnafu)?;
                    block.attn.merge_heads(&out)?
                }
            };
            let attn_out = block.attn.out.forward(&attn_out)?;
            x = x.try_add(&attn_out).context(TensorSnafu)?;

            // ── Cross-attn (fixed cache, no mask) ────────────────────────
            let h = block.cross_attn_ln.apply(&x)?;
            let cq = block.cross_attn.query.forward(&h)?;
            let cq_seq = cq.try_reshape([batch, 1, n_head, d_head]).context(TensorSnafu)?;

            let direct = if attention.custom_cross {
                svod_tk::single_query_attention_packed(
                    &cq_seq.cast(DType::Float32).context(TensorSnafu)?,
                    cross_k,
                    cross_v,
                    lh_start,
                    svod_tk::SqAttentionOpts { split: cross_splits, ..Default::default() },
                )
                .map_err(tk_launch_error)
                .context(TensorSnafu)?
            } else {
                None
            };
            let cross_out = match direct {
                Some(out) => out
                    .try_reshape([batch, 1, self.n_state])
                    .context(TensorSnafu)?
                    .cast(self.activation_dtype.clone())
                    .context(TensorSnafu)?,
                None => {
                    let layer_ck = cross_k
                        .try_shrink([None, None, Some((lh_start as isize, lh_end as isize)), None])
                        .context(TensorSnafu)?
                        .cast(self.activation_dtype.clone())
                        .context(TensorSnafu)?;
                    let layer_cv = cross_v
                        .try_shrink([None, None, Some((lh_start as isize, lh_end as isize)), None])
                        .context(TensorSnafu)?
                        .cast(self.activation_dtype.clone())
                        .context(TensorSnafu)?;
                    let cq_h = cq_seq.try_permute(&[0, 2, 1, 3]).context(TensorSnafu)?;
                    let layer_ck_h = layer_ck.try_permute(&[0, 2, 1, 3]).context(TensorSnafu)?;
                    let layer_cv_h = layer_cv.try_permute(&[0, 2, 1, 3]).context(TensorSnafu)?;
                    let out = cq_h
                        .scaled_dot_product_attention()
                        .key(&layer_ck_h)
                        .value(&layer_cv_h)
                        .is_causal(false)
                        .call()
                        .context(TensorSnafu)?;
                    block.cross_attn.merge_heads(&out)?
                }
            };
            let cross_out = block.cross_attn.out.forward(&cross_out)?;
            x = x.try_add(&cross_out).context(TensorSnafu)?;

            // ── MLP ───────────────────────────────────────────────────────
            let h = block.mlp_ln.apply(&x)?;
            let h = linear_with_bias(&h, &block.mlp0_w, &block.mlp0_b)?;
            let h = h.gelu_exact().context(TensorSnafu)?;
            let h = linear_with_bias(&h, &block.mlp1_w, &block.mlp1_b)?;
            x = x.try_add(&h).context(TensorSnafu)?;

            // Collect new K/V for cache update: [B, H, 1, Dh]
            new_ks.push(new_k_h);
            new_vs.push(new_v_h);
        }

        // Permute each layer's K/V from [B, H, 1, Dh] to [B, 1, H, Dh],
        // then cat along dim 1 → [B, n_layer, H, Dh] → reshape [B, 1, n_layer*H, Dh].
        // Catting along dim 0 would interleave beams and layers for B > 1.
        let permuted_ks: Vec<Tensor> =
            new_ks.iter().map(|t| t.try_permute(&[0, 2, 1, 3]).context(TensorSnafu)).collect::<Result<Vec<_>>>()?;
        let permuted_vs: Vec<Tensor> =
            new_vs.iter().map(|t| t.try_permute(&[0, 2, 1, 3]).context(TensorSnafu)).collect::<Result<Vec<_>>>()?;

        let stacked_k = Tensor::cat(&permuted_ks.iter().collect::<Vec<_>>(), 1).context(TensorSnafu)?;
        let stacked_v = Tensor::cat(&permuted_vs.iter().collect::<Vec<_>>(), 1).context(TensorSnafu)?;

        let new_k_flat = stacked_k
            .try_reshape(&[
                svod_ir::SInt::Const(batch),
                svod_ir::SInt::Const(1usize),
                svod_ir::SInt::Const(n_layer * n_head),
                svod_ir::SInt::Const(d_head),
            ])
            .context(TensorSnafu)?;
        let new_v_flat = stacked_v
            .try_reshape(&[
                svod_ir::SInt::Const(batch),
                svod_ir::SInt::Const(1usize),
                svod_ir::SInt::Const(n_layer * n_head),
                svod_ir::SInt::Const(d_head),
            ])
            .context(TensorSnafu)?;
        let x = scoped("ln", || self.ln.apply(&x))?;
        let logits = x
            .linear()
            .weight(&self.token_embedding.cast(x.uop().dtype()).context(TensorSnafu)?)
            .call()
            .context(TensorSnafu)?
            .cast(DType::Float32)
            .context(TensorSnafu)?;

        // logits is [B, 1, n_vocab] → reshape to [B, n_vocab]
        let n_vocab = self.token_embedding.shape().context(TensorSnafu)?[0].as_const().ok_or_else(|| {
            super::error::Error::Tensor {
                source: Box::new(svod_tensor::error::Error::SymbolicShapeUnsupported {
                    operation: "forward_step n_vocab".into(),
                }),
            }
        })?;
        let logits =
            logits.try_reshape(&[svod_ir::SInt::Const(batch), svod_ir::SInt::Const(n_vocab)]).context(TensorSnafu)?;

        // K/V outputs cast to fp32 — appended into the fp32 cache buffer via SDMA.
        Ok((
            logits,
            new_k_flat.cast(DType::Float32).context(TensorSnafu)?,
            new_v_flat.cast(DType::Float32).context(TensorSnafu)?,
        ))
    }
}

impl HasStateDict for TextDecoder {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.insert(prefixed(prefix, "token_embedding.weight"), self.token_embedding.clone());
        sd.insert(prefixed(prefix, "positional_embedding"), self.positional_embedding.clone());
        for (i, block) in self.blocks.iter().enumerate() {
            sd.extend(block.state_dict(&prefixed(prefix, &format!("blocks.{i}"))));
        }
        sd.extend(self.ln.state_dict(&prefixed(prefix, "ln")));
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.token_embedding = get_tensor(sd, &prefixed(prefix, "token_embedding.weight"))?;
        self.positional_embedding = get_tensor(sd, &prefixed(prefix, "positional_embedding"))?;
        for (i, block) in self.blocks.iter_mut().enumerate() {
            block.load_state_dict(sd, &prefixed(prefix, &format!("blocks.{i}")))?;
        }
        self.ln.load_state_dict(sd, &prefixed(prefix, "ln"))?;
        Ok(())
    }
}
