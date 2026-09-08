//! Qwen3 attention with Grouped Query Attention (GQA), per-head Q/K RMSNorm,
//! and RoPE.
//!
//! Key differences from ModernBERT attention:
//! - **Separate Q/K/V/O projections** (not fused QKV)
//! - **GQA**: K/V carry `num_key_value_heads < num_attention_heads`; SDPA's
//!   `enable_gqa` repeats each KV head `num_heads / num_kv_heads` times
//! - **Per-head Q/K RMSNorm** (Qwen3 signature feature): `q_norm`/`k_norm`
//!   normalize over `head_dim` only, applied AFTER view-to-heads, BEFORE RoPE
//! - **Causal attention**: `is_causal = true` in SDPA
//! - **No bias** on any projection (`attention_bias = false`)

use svod_tensor::Tensor;
use svod_tensor::nn::{Layer, Module, RmsNorm};

use crate::init::fan_in_uniform;

use super::error::Result;

#[derive(Clone, Module)]
pub struct Qwen3Attention {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub eps: f64,
    #[module(key = "q_proj.weight")]
    pub q_proj_weight: Tensor,
    #[module(key = "k_proj.weight")]
    pub k_proj_weight: Tensor,
    #[module(key = "v_proj.weight")]
    pub v_proj_weight: Tensor,
    #[module(key = "o_proj.weight")]
    pub o_proj_weight: Tensor,
    pub q_norm: RmsNorm,
    pub k_norm: RmsNorm,
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
        let q_norm = RmsNorm::with_dims(head_dim, eps, dtype.clone());
        let k_norm = RmsNorm::with_dims(head_dim, eps, dtype);
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
    /// `rope`: the shared `(cos, sin)` table. `padding_mask`: optional bool
    /// `(B, L)` where `true` = real token, `false` = padding.
    pub fn forward(&self, x: &Tensor, rope: &(Tensor, Tensor), padding_mask: Option<&Tensor>) -> Result<Tensor> {
        let (cos, sin) = rope;
        let project =
            |w: &Tensor, heads: usize| -> Result<Tensor> { Ok(x.linear().weight(w).call()?.split_heads(heads)?) };

        // Per-head Q/K RMSNorm (BEFORE RoPE) — Qwen3 signature feature.
        let q = self.q_norm.forward(&project(&self.q_proj_weight, self.num_heads)?)?;
        let k = self.k_norm.forward(&project(&self.k_proj_weight, self.num_kv_heads)?)?;
        let v = project(&self.v_proj_weight, self.num_kv_heads)?;

        let q = q.apply_rotary_emb(cos, sin, false)?;
        let k = k.apply_rotary_emb(cos, sin, false)?;

        // Causal SDPA over `num_kv_heads` K/V heads, expanded to the Q head
        // count by `enable_gqa`, plus the optional padding mask.
        let attn = q
            .scaled_dot_product_attention()
            .key(&k)
            .value(&v)
            .is_causal(true)
            .enable_gqa(true)
            .maybe_key_padding_mask(padding_mask)
            .call()?;

        Ok(attn.merge_heads()?.linear().weight(&self.o_proj_weight).call()?)
    }
}
