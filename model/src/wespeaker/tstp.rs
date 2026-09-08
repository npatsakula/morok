//! Temporal-Statistical Pooling (TSTP) head from WeSpeaker.
//!
//! Takes a 4D backbone feature map `[B, C, H, T]` plus per-frame attention
//! weights `[B, T_w]`, interpolates the weights to `T` with nearest mode (as
//! pyannote's `StatsPool` does internally), and returns the weighted mean
//! and (unbiased, Bessel-corrected) standard deviation concatenated along the
//! feature axis: `[B, 2 * C * H]`. The exact arithmetic — including the
//! `v1 - v2/v1 + 1e-8` denominator and `1e-8` epsilon on `v1` — matches
//! `pyannote.audio.models.blocks.pooling.StatsPool._pool`.

use svod_ir::SInt;
use svod_tensor::Tensor;

use super::error::Result;

/// Numerical epsilon — same value pyannote's `_pool` uses
/// (`weights.sum + 1e-8`, denominator `+ 1e-8`).
const EPS: f64 = 1e-8;

/// Weighted statistics pooling. `features` is `[B, C, H, T]`,
/// `weights` is `[B, T_w]`. `T` must be concrete (the backbone bakes the
/// time stride sequence at `prepare()` time). Returns `[B, 2 * C * H]`
/// = `concat(mean, std)` along the feature axis.
pub fn tstp_forward(features: &Tensor, weights: &Tensor) -> Result<Tensor> {
    let shape = features.shape()?;
    if shape.len() != 4 {
        return Err(svod_tensor::error::Error::from(svod_tensor::error::ErrorKind::IrConstruction {
            details: format!("TSTP expects 4D features, got {}D", shape.len()),
        })
        .into());
    }
    let t_back = features.dim_const(3)?;

    // weights: [B, T_w] → [B, T_back] via a constant one-hot nearest matrix,
    // then unsqueeze to [B, 1, 1, T_back] for 4D broadcasting against features.
    // The matrix is precomputed so weights stays a simple matmul; we can't use
    // tensor::resize() here because it requires every shape dim to be concrete
    // and our batch dim is symbolic.
    let t_w = weights.dim_const(1)?;
    let mat = nearest_interp_matrix(t_w, t_back);
    let w = weights.linear().weight(&mat).call()?;
    let w = w.try_unsqueeze(1)?;
    let w = w.try_unsqueeze(2)?;

    let dtype = features.dtype();
    let eps = Tensor::const_(EPS, dtype.clone());

    // v1 = weights.sum(dim=3, keepdim=True) + eps              [B, 1, 1, 1]
    let v1_raw = w.sum_with().axes(3isize).keepdim(true).call()?;
    let v1 = v1_raw.try_add(&eps)?;

    // mean = (features * w).sum(dim=3, keepdim=True) / v1      [B, C, H, 1]
    let xw = features.try_mul(&w)?;
    let xw_sum = xw.sum_with().axes(3isize).keepdim(true).call()?;
    let mean = xw_sum.try_div(&v1)?;

    // dx2 = (features - mean)^2
    let centered = features.try_sub(&mean)?;
    let dx2 = centered.square();

    // v2 = (w^2).sum(dim=3, keepdim=True)                      [B, 1, 1, 1]
    let w_sq = w.square();
    let v2 = w_sq.sum_with().axes(3isize).keepdim(true).call()?;

    // denom = v1 - v2/v1 + eps                                 [B, 1, 1, 1]
    let denom = v1.try_sub(&v2.try_div(&v1)?)?;
    let denom = denom.try_add(&eps)?;

    // var = (dx2 * w).sum(dim=3, keepdim=True) / denom         [B, C, H, 1]
    let var_num = dx2.try_mul(&w)?;
    let var_num = var_num.sum_with().axes(3isize).keepdim(true).call()?;
    let var = var_num.try_div(&denom)?;
    let std = var.try_sqrt()?;

    // Squeeze the trailing time dim and flatten (C, H) → C*H
    // mean / std are [B, C, H, 1] → flatten dims 1..4 (i.e., 1,2,3 → C*H*1 = C*H).
    let b = features.dim(0)?;
    let c = features.dim_const(1)?;
    let h = features.dim_const(2)?;
    let stats_dim = SInt::Const(c * h);

    let mean_flat = mean.try_reshape([b.clone(), stats_dim.clone()])?;
    let std_flat = std.try_reshape([b, stats_dim])?;

    Ok(Tensor::cat(&[&mean_flat, &std_flat], 1)?)
}

/// One-hot interpolation matrix `[t_out, t_in]` such that `dst = src @ M.T`
/// performs PyTorch's `F.interpolate(..., mode="nearest")` along the trailing
/// axis (asymmetric coordinate transform, floor rounding).
///
/// For each output position `o ∈ [0, t_out)` the source position is
/// `floor(o * t_in / t_out)`. Integer math here matches the float-arithmetic
/// floor exactly because `o * t_in` is a non-negative integer.
fn nearest_interp_matrix(t_in: usize, t_out: usize) -> Tensor {
    let mut m = vec![0.0f32; t_out * t_in];
    for o in 0..t_out {
        let src = (o * t_in) / t_out;
        m[o * t_in + src] = 1.0;
    }
    Tensor::from_slice(&m).try_reshape([t_out as isize, t_in as isize]).expect("nearest interp matrix reshape")
}
