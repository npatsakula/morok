//! Rotary position embedding (RoPE) for Qwen3.
//!
//! GPT-NeoX-style (non-interleaved) rotation, matching transformers'
//! `rotate_half`. `inv_freq`, `cos`, and `sin` are precomputed in **f32**
//! and cast to the compute dtype for application.
//!
//! Single rotary base (`rope_theta = 1_000_000`) shared across all layers.

use svod_dtype::DType;
use svod_tensor::Tensor;

use super::error::Result;

/// Precomputed `(cos, sin)` table, shaped `(1, 1, seq_len, head_dim/2)` so it
/// broadcasts against q/k of shape `(B, H, seq_len, head_dim)`.
#[derive(Clone)]
pub struct RotaryTable {
    pub cos: Tensor,
    pub sin: Tensor,
}

impl RotaryTable {
    pub fn new(theta: f64, seq_len: usize, head_dim: usize, dtype: DType) -> Result<Self> {
        let half = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| {
                let exponent = -2.0 * i as f64 / head_dim as f64;
                theta.powf(exponent) as f32
            })
            .collect();

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

        let cos = cos_f32.try_unsqueeze(0)?.try_unsqueeze(0)?;
        let sin = sin_f32.try_unsqueeze(0)?.try_unsqueeze(0)?;

        Ok(Self { cos: cos.cast(dtype.clone())?, sin: sin.cast(dtype)? })
    }

    pub fn apply(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.apply_rotary_emb(&self.cos, &self.sin, false)?)
    }
}
