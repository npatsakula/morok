use morok_tensor::Tensor;
use ndarray::Array4;

use super::GigaAmConfig;

/// Precompute RoPE cos/sin cache tensors.
///
/// Returns `(cos, sin)` each of shape `[max_encoder_frames, 1, 1, d_k/2]`
/// where `d_k = d_model / n_heads`.
///
/// GigaAM uses non-interleaved RoPE (first_half/second_half split),
/// matching `apply_rotary_emb(..., interleaved=false)`.
pub fn build_rope_cache(config: &GigaAmConfig) -> (Tensor, Tensor) {
    let d_k = config.d_model / config.n_heads;
    let half_d = d_k / 2;
    let max_len = config.max_encoder_frames;

    let inv_freq: Vec<f32> = (0..half_d).map(|i| 1.0 / 10000.0f32.powf(2.0 * i as f32 / d_k as f32)).collect();

    let mut cos_arr = Array4::<f32>::zeros((max_len, 1, 1, half_d));
    let mut sin_arr = Array4::<f32>::zeros((max_len, 1, 1, half_d));

    for pos in 0..max_len {
        for i in 0..half_d {
            let angle = pos as f32 * inv_freq[i];
            cos_arr[[pos, 0, 0, i]] = angle.cos();
            sin_arr[[pos, 0, 0, i]] = angle.sin();
        }
    }

    (Tensor::from_ndarray(&cos_arr), Tensor::from_ndarray(&sin_arr))
}
