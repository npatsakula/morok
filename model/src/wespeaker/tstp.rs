//! Temporal-Statistical Pooling (TSTP) head from WeSpeaker.
//!
//! Takes a 4D backbone feature map `[B, C, H, T]` plus per-frame attention
//! weights `[B, T_w]`, resamples the weights to `T` with nearest mode (as
//! pyannote's `StatsPool` does internally), and returns the weighted mean
//! and (unbiased, Bessel-corrected) standard deviation concatenated along the
//! feature axis: `[B, 2 * C * H]`. The exact arithmetic — including the
//! `v1 - v2/v1 + 1e-8` denominator and `1e-8` epsilon on `v1` — matches
//! `pyannote.audio.models.blocks.pooling.StatsPool._pool`.

use svod_ir::SInt;
use svod_tensor::Tensor;
use svod_tensor::error::{Error, ErrorKind};
use svod_tensor::nn::{CoordinateTransformMode, NearestMode, ResizeMode};

use super::error::Result;

/// Numerical epsilon — same value pyannote's `_pool` uses
/// (`weights.sum + 1e-8`, denominator `+ 1e-8`).
const EPS: f64 = 1e-8;

/// Weighted statistics pooling. `features` is `[B, C, H, T]`,
/// `weights` is `[B, T_w]`. `T` must be concrete (the backbone bakes the
/// time stride sequence at `prepare()` time). Returns `[B, 2 * C * H]`
/// = `concat(mean, std)` along the feature axis.
pub fn tstp_forward(features: &Tensor, weights: &Tensor) -> Result<Tensor> {
    let ndim = features.ndim()?;
    if ndim != 4 {
        return Err(Error::from(ErrorKind::NdimExact { op: "tstp", expected: 4, actual: ndim }).into());
    }
    let t_back = features.dim_const(3)?;

    // `F.interpolate(weights, size=T, mode="nearest")`: an asymmetric
    // coordinate transform with floor rounding. Only the time axis resizes, so
    // the symbolic batch dim passes through; the result is unsqueezed to
    // `[B, 1, 1, T]` to broadcast against the feature map.
    let w = weights
        .resize()
        .axes(&[1])
        .sizes(&[t_back])
        .mode(ResizeMode::Nearest)
        .nearest_mode(NearestMode::Floor)
        .coordinate_transformation_mode(CoordinateTransformMode::Asymmetric)
        .call()?
        .try_unsqueeze(1)?
        .try_unsqueeze(2)?;

    let sum_t = |t: &Tensor| t.sum_with().axes(3isize).keepdim(true).call();

    // v1 = weights.sum(-1) + eps                               [B, 1, 1, 1]
    let v1 = sum_t(&w)?.try_add(EPS)?;
    // mean = (features * w).sum(-1) / v1                       [B, C, H, 1]
    let mean = sum_t(&features.try_mul(&w)?)?.try_div(&v1)?;
    // denom = v1 - (w^2).sum(-1) / v1 + eps                    [B, 1, 1, 1]
    let denom = v1.try_sub(&sum_t(&w.square())?.try_div(&v1)?)?.try_add(EPS)?;
    // std = sqrt(((features - mean)^2 * w).sum(-1) / denom)    [B, C, H, 1]
    let dx2 = features.try_sub(&mean)?.square();
    let std = sum_t(&dx2.try_mul(&w)?)?.try_div(&denom)?.try_sqrt()?;

    // Drop the trailing time dim and flatten (C, H, 1) → C*H.
    let stats = [features.dim(0)?, SInt::Const(features.dim_const(1)? * features.dim_const(2)?)];
    let mean = mean.try_reshape(stats.clone())?;
    let std = std.try_reshape(stats)?;
    Ok(Tensor::cat(&[&mean, &std], 1)?)
}
