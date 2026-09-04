//! WavLM self-attention with gated bucketed relative position bias.
//!
//! Direct port of `WavLMSelfAttention` from
//! `submodules/DiariZen/diarizen/models/module/wav2vec2/components.py:549-725`.
//! Notable details:
//!
//! - **Per-layer head pruning**: Q/K/V/O linears are *physically resized* to
//!   `num_kept * head_dim` (where `num_kept = remaining_heads.len()`); the
//!   gating tensors and the shared `rel_attn_embed` table stay full-width and
//!   are indexed at runtime.
//! - **Bucketed relative positions** (`num_buckets=320`, `max_distance=800`):
//!   sign-aware with logarithmic spacing past `num_buckets/2`. Implemented
//!   eagerly in Rust as a small `Tensor` of `i64` bucket ids (cheap; L is
//!   fixed per chunk).
//! - **Gating**: the gate is computed from the *raw input* `x` (NOT the
//!   projected query) reshaped to per-head form. `gru_rel_pos_linear` maps
//!   `head_dim → 8`; reshape to `(..., 2, 4)`, sum over the trailing axis,
//!   sigmoid, chunk in two; combine with `gru_rel_pos_const` to produce a
//!   `(B, total_num_heads, T, 1)` scalar that scales the position bias.
//! - **Position bias** flows between layers: layer 0 owns the bucketed table
//!   `rel_attn_embed` and produces a `(1, total_num_heads, L, L)` bias that
//!   the encoder passes down to every layer's attention forward. Each layer
//!   gates and head-selects independently.

use snafu::{OptionExt, ResultExt};
use svod_dtype::DType;
use svod_tensor::reduce::AxisSpec;
use svod_tensor::{Tensor, s};

use crate::init::{fan_in_uniform, ones, zeros};
use crate::state::{self, HasStateDict, StateDict, get_tensor, prefixed};

use super::config::WavLmConfig;
use super::error::{Result, SymbolicShapeSnafu, TensorSnafu};

// ---------------------------------------------------------------------------
// Bucket math (matches `_relative_positions_bucket(bidirectional=True)`)
// ---------------------------------------------------------------------------

/// Compute the (L_q, L_k) bucket-id table used to index `rel_attn_embed`.
/// `bidirectional=True` matches WavLM's convention.
pub fn compute_bucket_indices(
    query_length: usize,
    key_length: usize,
    num_buckets: usize,
    max_distance: usize,
) -> Vec<i64> {
    let half_buckets = num_buckets / 2;
    let max_exact = half_buckets / 2;
    let log_ratio = (max_distance as f64 / max_exact as f64).ln();

    let mut buckets = vec![0i64; query_length * key_length];
    for q in 0..query_length {
        for k in 0..key_length {
            let rel = k as i64 - q as i64;
            let sign_offset = if rel > 0 { half_buckets as i64 } else { 0 };
            let abs_rel = rel.unsigned_abs() as usize;

            let bucket = if abs_rel < max_exact {
                abs_rel
            } else {
                let log_pos = (abs_rel as f64 / max_exact as f64).ln() / log_ratio;
                let scaled = (log_pos * (half_buckets - max_exact) as f64) as usize;
                (max_exact + scaled).min(half_buckets - 1)
            };
            buckets[q * key_length + k] = sign_offset + bucket as i64;
        }
    }
    buckets
}

/// Materialize the bucket-index table as a `(L_q, L_k)` int64 `Tensor`.
pub fn bucket_index_tensor(
    query_length: usize,
    key_length: usize,
    num_buckets: usize,
    max_distance: usize,
) -> Result<Tensor> {
    let buckets = compute_bucket_indices(query_length, key_length, num_buckets, max_distance);
    let flat = Tensor::from_slice(&buckets);
    flat.try_reshape([query_length as isize, key_length as isize]).context(TensorSnafu)
}

/// Compute the un-gated position bias `(1, total_num_heads, L, L)` from the
/// shared `rel_attn_embed` table. The leading `1` broadcasts against the
/// batch dim downstream.
pub fn compute_position_bias(
    rel_attn_embed: &Tensor,
    query_len: usize,
    key_len: usize,
    num_buckets: usize,
    max_distance: usize,
) -> Result<Tensor> {
    let idx = bucket_index_tensor(query_len, key_len, num_buckets, max_distance)?;
    // Embedding lookup: (L, L) indices into (num_buckets, total_num_heads) →
    // (L, L, total_num_heads).
    let bias = rel_attn_embed.embedding(&idx).context(TensorSnafu)?;
    // → (total_num_heads, L, L)
    let bias = bias.try_permute(&[2, 0, 1]).context(TensorSnafu)?;
    // → (1, total_num_heads, L, L) for broadcast against batch dim.
    let bias = bias.try_unsqueeze(0).context(TensorSnafu)?;
    Ok(bias.contiguous())
}

// ---------------------------------------------------------------------------
// GatedRelPosAttention
// ---------------------------------------------------------------------------

/// Multi-head attention with gated bucketed relative-position bias. Forward
/// expects an `x` of shape `(B, L, embed_dim)` and a `position_bias` of
/// shape `(_, total_num_heads, L, L)` (broadcastable over batch).
///
/// Output: `(B, L, embed_dim)`.
#[derive(Clone)]
pub struct GatedRelPosAttention {
    pub embed_dim: usize,
    pub total_num_heads: usize,
    pub head_dim: usize,
    /// Surviving head indices (length = `num_kept`). Must be non-empty —
    /// callers should encode the empty-list case as `Option<Self>::None`.
    pub remaining_heads: Vec<usize>,

    // Q / K / V / O projections — physically sized for `num_kept * head_dim`.
    pub q_weight: Tensor,
    pub q_bias: Tensor,
    pub k_weight: Tensor,
    pub k_bias: Tensor,
    pub v_weight: Tensor,
    pub v_bias: Tensor,
    pub out_weight: Tensor,
    pub out_bias: Tensor,

    // Gating params — per-layer, full-width (NOT physically pruned).
    pub gru_rel_pos_linear_weight: Tensor, // (8, head_dim)
    pub gru_rel_pos_linear_bias: Tensor,   // (8,)
    pub gru_rel_pos_const: Tensor,         // (1, total_num_heads, 1, 1)
}

impl GatedRelPosAttention {
    pub fn empty(config: &WavLmConfig, layer_index: usize) -> Self {
        let embed_dim = config.encoder_embed_dim;
        let total_num_heads = config.encoder_total_num_heads[layer_index];
        let head_dim = config.encoder_head_dim;
        let remaining_heads = config.encoder_remaining_heads[layer_index].clone();
        let num_kept = remaining_heads.len();
        assert!(num_kept > 0, "GatedRelPosAttention::empty requires non-empty remaining_heads");

        let proj_out = num_kept * head_dim;
        let q_weight = fan_in_uniform(&[proj_out, embed_dim], embed_dim, DType::Float32);
        let q_bias = zeros(&[proj_out], DType::Float32);
        let k_weight = fan_in_uniform(&[proj_out, embed_dim], embed_dim, DType::Float32);
        let k_bias = zeros(&[proj_out], DType::Float32);
        let v_weight = fan_in_uniform(&[proj_out, embed_dim], embed_dim, DType::Float32);
        let v_bias = zeros(&[proj_out], DType::Float32);
        let out_weight = fan_in_uniform(&[embed_dim, proj_out], proj_out, DType::Float32);
        let out_bias = zeros(&[embed_dim], DType::Float32);

        let gru_rel_pos_linear_weight = fan_in_uniform(&[8, head_dim], head_dim, DType::Float32);
        let gru_rel_pos_linear_bias = zeros(&[8], DType::Float32);
        let gru_rel_pos_const = ones(&[1, total_num_heads, 1, 1], DType::Float32);

        Self {
            embed_dim,
            total_num_heads,
            head_dim,
            remaining_heads,
            q_weight,
            q_bias,
            k_weight,
            k_bias,
            v_weight,
            v_bias,
            out_weight,
            out_bias,
            gru_rel_pos_linear_weight,
            gru_rel_pos_linear_bias,
            gru_rel_pos_const,
        }
    }

    pub fn num_kept(&self) -> usize {
        self.remaining_heads.len()
    }

    /// Forward. `x`: `(B, L, embed_dim)`. `position_bias`: shape broadcastable
    /// to `(B, total_num_heads, L, L)` (e.g. `(1, total_num_heads, L, L)`).
    ///
    /// Python ref: `WavLMSelfAttention.forward` (components.py:668-725)
    /// + `SelfAttention.forward` (components.py:429-486)
    pub fn forward(&self, x: &Tensor, position_bias: &Tensor) -> Result<Tensor> {
        let x_shape = x.shape().context(TensorSnafu)?;
        let bsz = x_shape[0].clone();
        let seq_len: usize = x_shape[1].as_const().context(SymbolicShapeSnafu { what: "attention" })?;
        let b = bsz.clone();
        let l = seq_len;
        let nk = self.num_kept();
        let h = self.total_num_heads;
        let hd = self.head_dim;

        // =================================================================
        // Gate computation (WavLMSelfAttention.forward, lines 702-710)
        // =================================================================

        // Py:703  query_layer = query.view(bsz, seq_len, self.total_num_heads, -1)
        // Py:704  query_layer = query_layer.permute(0, 2, 1, 3)
        let query_layer = x
            .view(&[b.clone(), l.into(), h.into(), hd.into()])
            .context(TensorSnafu)?
            .try_permute(&[0, 2, 1, 3])
            .context(TensorSnafu)?; // (B, h, L, hd)

        // Py:706-708  gate_a, gate_b = torch.sigmoid(
        //   self.gru_rel_pos_linear(query_layer).view(bsz, h, seq_len, 2, 4).sum(-1)
        // ).chunk(2, dim=-1)
        let gate_raw = query_layer
            .linear()
            .weight(&self.gru_rel_pos_linear_weight)
            .bias(&self.gru_rel_pos_linear_bias)
            .call()
            .context(TensorSnafu)? // (B, h, L, 8)
            .view(&[b.clone(), h.into(), l.into(), 2usize.into(), 4usize.into()])
            .context(TensorSnafu)?
            .sum_with()
            .axes(AxisSpec::Single(-1))
            .keepdim(false)
            .call()
            .context(TensorSnafu)? // (B, h, L, 2)
            .sigmoid()
            .context(TensorSnafu)?;
        let mut gate_chunks = gate_raw.chunk(2, -1).context(TensorSnafu)?;
        let gate_b = gate_chunks.pop().expect("chunk(2) yields 2 parts");
        let gate_a = gate_chunks.pop().expect("chunk(2) yields 2 parts");

        // Py:709  gate_a_1 = gate_a * (gate_b * self.gru_rel_pos_const - 1.0) + 2.0
        let one = Tensor::const_(1.0, gate_b.uop().dtype());
        let two = Tensor::const_(2.0, gate_b.uop().dtype());
        let gate_a_1 = gate_a
            .try_mul(&gate_b.try_mul(&self.gru_rel_pos_const).context(TensorSnafu)?.try_sub(&one).context(TensorSnafu)?)
            .context(TensorSnafu)?
            .try_add(&two)
            .context(TensorSnafu)?; // (B, h, L, 1)

        // =================================================================
        // Gated position bias (lines 710-713)
        // =================================================================

        // Py:710  attn_mask_rel_pos = gate_a_1.view(bsz * h, -1, 1) * position_bias
        let attn_mask_rel_pos = gate_a_1.try_mul(position_bias).context(TensorSnafu)?; // (B, h, L, L)

        // Py:712  attn_mask_rel_pos = attn_mask_rel_pos.view((-1, seq_len, seq_len))
        // Py:713  attn_mask_rel_pos = attn_mask_rel_pos.reshape(bsz, h, seq_len, seq_len)[:, self.remaining_heads, :, :]
        let attn_mask = attn_mask_rel_pos.getitem(s![.., self.remaining_heads.clone(), .., ..]).context(TensorSnafu)?; // (B, nk, L, L)

        // =================================================================
        // Q / K / V projections (SelfAttention.forward, lines 455-458)
        // =================================================================

        // Py:455  shape = (batch_size, length, self.num_heads, self.head_dim)
        // Py:456  q = self.q_proj(x).view(*shape).transpose(2, 1)   # (B, nH, L, Hd)
        // Py:457  k = self.k_proj(x).view(*shape).permute(0, 2, 3, 1)  # (B, nH, Hd, L)
        // Py:458  v = self.v_proj(x).view(*shape).transpose(2, 1)   # (B, nH, L, Hd)
        let q = x
            .linear()
            .weight(&self.q_weight)
            .bias(&self.q_bias)
            .call()
            .context(TensorSnafu)?
            .view(&[b.clone(), l.into(), nk.into(), hd.into()])
            .context(TensorSnafu)?
            .try_transpose(2, 1)
            .context(TensorSnafu)?; // (B, nk, L, hd)
        let k = x
            .linear()
            .weight(&self.k_weight)
            .bias(&self.k_bias)
            .call()
            .context(TensorSnafu)?
            .view(&[b.clone(), l.into(), nk.into(), hd.into()])
            .context(TensorSnafu)?
            .try_permute(&[0, 2, 3, 1])
            .context(TensorSnafu)?; // (B, nk, hd, L)
        let v = x
            .linear()
            .weight(&self.v_weight)
            .bias(&self.v_bias)
            .call()
            .context(TensorSnafu)?
            .view(&[b.clone(), l.into(), nk.into(), hd.into()])
            .context(TensorSnafu)?
            .try_transpose(2, 1)
            .context(TensorSnafu)?; // (B, nk, L, hd)

        // =================================================================
        // Scaled dot-product attention (lines 461-472)
        // =================================================================

        // Py:461  weights = (self.scaling * q) @ k  # B, nH, L, L
        let scaling = (hd as f32).powf(-0.5);
        let scaling_t = Tensor::const_(scaling, q.uop().dtype());
        let weights = q.try_mul(&scaling_t).context(TensorSnafu)?.matmul(&k).context(TensorSnafu)?; // (B, nk, L, L)

        // Py:463  weights += attention_mask
        let weights = weights.try_add(&attn_mask).context(TensorSnafu)?;

        // Py:467  weights = weights - weights.max(dim=-1, keepdim=True)[0]
        // Py:469  weights = torch.nn.functional.softmax(weights, dim=-1)
        // `softmax` subtracts the row max itself, and the reference's explicit
        // subtraction changes nothing bit-for-bit: the row max of `x - m` is
        // exactly 0, so it would only add a second (all-zero) reduce.
        let weights = weights.softmax(-1).context(TensorSnafu)?;

        // Py:472  output = weights @ v  # B, nH, L, Hd
        let output = weights.matmul(&v).context(TensorSnafu)?; // (B, nk, L, hd)

        // =================================================================
        // Output projection (lines 478-480)
        // =================================================================

        // Py:478  output = output.transpose(2, 1).reshape(batch_size, length, nH * Hd)
        let output = output
            .try_transpose(2, 1)
            .context(TensorSnafu)?
            .view(&[b, l.into(), (nk * hd).into()])
            .context(TensorSnafu)?;

        // Py:480  output = self.out_proj(output)
        output.linear().weight(&self.out_weight).bias(&self.out_bias).call().context(TensorSnafu)
    }
}

impl HasStateDict for GatedRelPosAttention {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.insert(prefixed(prefix, "q_proj.weight"), self.q_weight.clone());
        sd.insert(prefixed(prefix, "q_proj.bias"), self.q_bias.clone());
        sd.insert(prefixed(prefix, "k_proj.weight"), self.k_weight.clone());
        sd.insert(prefixed(prefix, "k_proj.bias"), self.k_bias.clone());
        sd.insert(prefixed(prefix, "v_proj.weight"), self.v_weight.clone());
        sd.insert(prefixed(prefix, "v_proj.bias"), self.v_bias.clone());
        sd.insert(prefixed(prefix, "out_proj.weight"), self.out_weight.clone());
        sd.insert(prefixed(prefix, "out_proj.bias"), self.out_bias.clone());
        sd.insert(prefixed(prefix, "gru_rel_pos_linear.weight"), self.gru_rel_pos_linear_weight.clone());
        sd.insert(prefixed(prefix, "gru_rel_pos_linear.bias"), self.gru_rel_pos_linear_bias.clone());
        sd.insert(prefixed(prefix, "gru_rel_pos_const"), self.gru_rel_pos_const.clone());
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.q_weight = get_tensor(sd, &prefixed(prefix, "q_proj.weight"))?;
        self.q_bias = get_tensor(sd, &prefixed(prefix, "q_proj.bias"))?;
        self.k_weight = get_tensor(sd, &prefixed(prefix, "k_proj.weight"))?;
        self.k_bias = get_tensor(sd, &prefixed(prefix, "k_proj.bias"))?;
        self.v_weight = get_tensor(sd, &prefixed(prefix, "v_proj.weight"))?;
        self.v_bias = get_tensor(sd, &prefixed(prefix, "v_proj.bias"))?;
        self.out_weight = get_tensor(sd, &prefixed(prefix, "out_proj.weight"))?;
        self.out_bias = get_tensor(sd, &prefixed(prefix, "out_proj.bias"))?;
        self.gru_rel_pos_linear_weight = get_tensor(sd, &prefixed(prefix, "gru_rel_pos_linear.weight"))?;
        self.gru_rel_pos_linear_bias = get_tensor(sd, &prefixed(prefix, "gru_rel_pos_linear.bias"))?;
        self.gru_rel_pos_const = get_tensor(sd, &prefixed(prefix, "gru_rel_pos_const"))?;
        Ok(())
    }
}
