//! Rotary position embedding (RoPE) for ModernBERT.
//!
//! GPT-NeoX-style (non-interleaved) rotation, matching the `flash_attn`
//! convention used upstream. `inv_freq`, `cos`, and `sin` are precomputed in
//! **f32** (bf16 has only 7 mantissa bits, insufficient for the
//! `position × theta` products) and cast to the compute dtype for application.
//!
//! Two rotary bases coexist in one model: `global_rope_theta` for global
//! layers, `local_rope_theta` for local (sliding-window) layers. The encoder
//! owns one [`RotaryTable`] per layer so each picks the right base.

use svod_dtype::DType;
use svod_tensor::Tensor;

use super::error::Result;

/// Precomputed `(cos, sin)` table for one rotary base, shaped
/// `(1, 1, seq_len, head_dim/2)` so it broadcasts against q/k of shape
/// `(B, H, seq_len, head_dim)`.
#[derive(Clone)]
pub struct RotaryTable {
    pub cos: Tensor,
    pub sin: Tensor,
}

impl RotaryTable {
    /// Build the `(cos, sin)` table for a given base and concrete sequence
    /// length. Computed in f32 then cast to `dtype`.
    pub fn new(theta: f64, seq_len: usize, head_dim: usize, dtype: DType) -> Result<Self> {
        let half = head_dim / 2;
        // inv_freq[i] = theta ** (-2i / head_dim)  for i in [0, half).
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| {
                let exponent = -2.0 * i as f64 / head_dim as f64;
                theta.powf(exponent) as f32
            })
            .collect();

        // freqs[s, i] = s * inv_freq[i]  → outer product (seq_len, half).
        let mut freqs = vec![0.0f32; seq_len * half];
        for s in 0..seq_len {
            for i in 0..half {
                freqs[s * half + i] = s as f32 * inv_freq[i];
            }
        }

        let cos_f32 = Tensor::from_slice(freqs.clone()).try_reshape([seq_len as isize, half as isize])?;
        let sin_f32 = Tensor::from_slice(freqs).try_reshape([seq_len as isize, half as isize])?;
        let cos_f32 = cos_f32.cos()?;
        let sin_f32 = sin_f32.sin()?;

        // (L, half) → (1, 1, L, half) to broadcast over (B, H, L, head_dim).
        let cos = cos_f32.try_unsqueeze(0)?.try_unsqueeze(0)?;
        let sin = sin_f32.try_unsqueeze(0)?.try_unsqueeze(0)?;

        Ok(Self { cos: cos.cast(dtype.clone()), sin: sin.cast(dtype) })
    }

    /// Apply RoPE to a q/k tensor of shape `(B, H, L, head_dim)`.
    pub fn apply(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.apply_rotary_emb(&self.cos, &self.sin, false)?)
    }
}
