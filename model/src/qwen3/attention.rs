//! Qwen3 attention with Grouped Query Attention (GQA), per-head Q/K RMSNorm,
//! and RoPE.
//!
//! Key differences from ModernBERT attention:
//! - **Separate Q/K/V/O projections** (not fused QKV)
//! - **GQA**: K/V have `num_key_value_heads < num_attention_heads`; KV heads
//!   are expanded via [`repeat_kv`] before SDPA
//! - **Per-head Q/K RMSNorm** (Qwen3 signature feature): `q_norm`/`k_norm`
//!   normalize over `head_dim` only, applied AFTER view-to-heads, BEFORE RoPE
//! - **Causal attention**: `is_causal = true` in SDPA
//! - **No bias** on any projection (`attention_bias = false`)

use svod_ir::SInt;
use svod_tensor::Tensor;

use crate::init::fan_in_uniform;
use crate::state::{self, HasStateDict, StateDict, get_tensor, prefixed};

use super::error::Result;

use super::rms_norm::RmsNormWeights;
use super::rotary::RotaryTable;

/// Expand KV heads to match Q head count for GQA.
///
/// `(B, n_kv, L, hd) → (B, n_kv * n_rep, L, hd)` via unsqueeze-insert,
/// broadcast-expand, reshape-merge. The `n_rep` factor is a compile-time
/// constant, so the expand target is concrete on the head axis. Symbolic
/// batch dim `b` passes through.
fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let b = x.dim(0)?;
    let l = x.dim(2)?;
    let n_kv = x.dim_const(1)?;
    let hd = x.dim_const(3)?;
    let total = n_kv * n_rep;
    Ok(x.try_reshape([b.clone(), SInt::from(n_kv as isize), SInt::from(1isize), l.clone(), SInt::from(hd as isize)])?
        .try_expand([
            b.clone(),
            SInt::from(n_kv as isize),
            SInt::from(n_rep as isize),
            l.clone(),
            SInt::from(hd as isize),
        ])?
        .try_reshape([b, SInt::from(total as isize), l, SInt::from(hd as isize)])?)
}

#[derive(Clone)]
pub struct Qwen3Attention {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub eps: f64,
    pub q_proj_weight: Tensor,
    pub k_proj_weight: Tensor,
    pub v_proj_weight: Tensor,
    pub o_proj_weight: Tensor,
    pub q_norm: RmsNormWeights,
    pub k_norm: RmsNormWeights,
}

impl Qwen3Attention {
    pub fn empty(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        eps: f64,
        dtype: svod_dtype::DType,
    ) -> Self {
        let q_proj_weight = fan_in_uniform(&[num_heads * head_dim, hidden_size], hidden_size, dtype.clone());
        let k_proj_weight = fan_in_uniform(&[num_kv_heads * head_dim, hidden_size], hidden_size, dtype.clone());
        let v_proj_weight = fan_in_uniform(&[num_kv_heads * head_dim, hidden_size], hidden_size, dtype.clone());
        let o_proj_weight = fan_in_uniform(&[hidden_size, num_heads * head_dim], num_heads * head_dim, dtype.clone());
        let q_norm = RmsNormWeights::empty(head_dim, eps, dtype.clone());
        let k_norm = RmsNormWeights::empty(head_dim, eps, dtype);
        Self {
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            eps,
            q_proj_weight,
            k_proj_weight,
            v_proj_weight,
            o_proj_weight,
            q_norm,
            k_norm,
        }
    }

    /// Forward. `x`: `(B, L, D)` → `(B, L, D)`.
    /// `rotary`: the shared cos/sin table. `padding_mask`: optional bool
    /// `(B, 1, 1, L)` where `true` = masked out (padding) position.
    pub fn forward(&self, x: &Tensor, rotary: &RotaryTable, padding_mask: Option<&Tensor>) -> Result<Tensor> {
        let b = x.dim(0)?;
        let l = x.dim_const(1)?;
        let h = self.num_heads as isize;
        let kv_h = self.num_kv_heads as isize;
        let hd = self.head_dim as isize;
        let bsint: SInt = b;

        // Separate projections.
        let q = x.linear().weight(&self.q_proj_weight).call()?;
        let k = x.linear().weight(&self.k_proj_weight).call()?;
        let v = x.linear().weight(&self.v_proj_weight).call()?;

        // (B, L, H*kv_hd) → (B, L, n, hd) → (B, n, L, hd)
        let to_heads = |t: Tensor, n: isize| -> Result<Tensor> {
            Ok(t.view([bsint.clone(), l.into(), n.into(), hd.into()])?.try_permute(&[0, 2, 1, 3])?)
        };

        let q = to_heads(q, h)?;
        let k = to_heads(k, kv_h)?;
        let v = to_heads(v, kv_h)?;

        // Per-head Q/K RMSNorm (BEFORE RoPE) — Qwen3 signature feature.
        let q = self.q_norm.apply(&q)?;
        let k = self.k_norm.apply(&k)?;

        // RoPE applied to Q and K.
        let q = rotary.apply(&q)?;
        let k = rotary.apply(&k)?;

        // GQA: expand K/V from num_kv_heads to num_heads.
        let n_rep = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(&k, n_rep)?;
        let v = repeat_kv(&v, n_rep)?;

        // Causal SDPA + optional padding mask.
        let attn =
            q.scaled_dot_product_attention().key(&k).value(&v).is_causal(true).maybe_attn_mask(padding_mask).call()?;

        // (B, H, L, hd) → (B, L, H*hd) → output projection.
        let attn = attn.try_permute(&[0, 2, 1, 3])?.view([bsint, l.into(), (self.num_heads * self.head_dim).into()])?;
        Ok(attn.linear().weight(&self.o_proj_weight).call()?)
    }
}

impl HasStateDict for Qwen3Attention {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.insert(prefixed(prefix, "q_proj.weight"), self.q_proj_weight.clone());
        sd.insert(prefixed(prefix, "k_proj.weight"), self.k_proj_weight.clone());
        sd.insert(prefixed(prefix, "v_proj.weight"), self.v_proj_weight.clone());
        sd.insert(prefixed(prefix, "o_proj.weight"), self.o_proj_weight.clone());
        sd.extend(self.q_norm.state_dict(&prefixed(prefix, "q_norm")));
        sd.extend(self.k_norm.state_dict(&prefixed(prefix, "k_norm")));
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.q_proj_weight = get_tensor(sd, &prefixed(prefix, "q_proj.weight"))?;
        self.k_proj_weight = get_tensor(sd, &prefixed(prefix, "k_proj.weight"))?;
        self.v_proj_weight = get_tensor(sd, &prefixed(prefix, "v_proj.weight"))?;
        self.o_proj_weight = get_tensor(sd, &prefixed(prefix, "o_proj.weight"))?;
        self.q_norm.load_state_dict(sd, &prefixed(prefix, "q_norm"))?;
        self.k_norm.load_state_dict(sd, &prefixed(prefix, "k_norm"))?;
        Ok(())
    }
}
