//! Normalization: layernorm, rms_norm, group_norm.

use morok_dtype::DType;
use morok_ir::{ConstValue, UOp};
use snafu::ResultExt;

use crate::Tensor;
use crate::error::UOpSnafu;
use crate::reduce::AxisSpec;

type Result<T> = crate::Result<T>;

impl Tensor {
    /// Layer normalization over axes [axis..ndim). Casts to f32 internally.
    pub fn layernorm(&self, axis: isize, eps: f64) -> Result<Tensor> {
        let (normed, _, _) = self.layernorm_with_stats(axis, eps)?;
        Ok(normed)
    }

    /// Layer normalization returning `(normalized, mean, inv_std_dev)`.
    /// Computes in f32 for numerical stability (matches ONNX stash_type=1 / Tinygrad).
    /// Mean and inv_std_dev remain in f32.
    pub fn layernorm_with_stats(&self, axis: isize, eps: f64) -> Result<(Tensor, Tensor, Tensor)> {
        let ndim = self.ndim()?;
        let norm_axis = Tensor::normalize_axis(axis, ndim)?;
        let axes: Vec<isize> = (norm_axis..ndim).map(|a| a as isize).collect();
        let axes_spec = AxisSpec::Multiple(axes);

        let original_dtype = self.uop().dtype();
        let x32 = if original_dtype != DType::Float32 { self.cast(DType::Float32)? } else { self.clone() };

        let mean = x32.mean_with().axes(axes_spec.clone()).keepdim(true).call()?;
        let centered = x32.try_sub(&mean)?;
        let variance = centered.square()?.mean_with().axes(axes_spec).keepdim(true).call()?;
        let eps_t = Tensor::new(UOp::const_(DType::Float32, ConstValue::Float(eps)));
        let inv_std = variance.try_add(&eps_t)?.try_rsqrt()?;
        let normalized = centered.try_mul(&inv_std)?;

        let normalized = if original_dtype != DType::Float32 { normalized.cast(original_dtype)? } else { normalized };
        Ok((normalized, mean, inv_std))
    }

    /// RMS normalization over axes [axis..ndim). Like layernorm without mean subtraction.
    /// Computes in f32 for numerical stability, multiplies original input by the normalization factor.
    pub fn rms_norm(&self, axis: isize, eps: f64) -> Result<Tensor> {
        let ndim = self.ndim()?;
        let norm_axis = Tensor::normalize_axis(axis, ndim)?;
        let axes: Vec<isize> = (norm_axis..ndim).map(|a| a as isize).collect();
        let axes_spec = AxisSpec::Multiple(axes);

        let original_dtype = self.uop().dtype();
        let x32 = if original_dtype != DType::Float32 { self.cast(DType::Float32)? } else { self.clone() };

        let norm = x32
            .square()?
            .mean_with()
            .axes(axes_spec)
            .keepdim(true)
            .call()?
            .try_add(&Tensor::new(UOp::const_(DType::Float32, ConstValue::Float(eps))))?
            .try_rsqrt()?;

        self.try_mul(&norm)
    }

    /// Group normalization: reshape → layernorm → scale + bias.
    /// Matches Tinygrad's ONNX `GroupNormalization` pattern.
    /// Casts to f32 internally for numerical stability.
    pub fn group_norm(&self, scale: &Tensor, bias: &Tensor, num_groups: usize, eps: f64) -> Result<Tensor> {
        let x_shape = self.shape()?;
        let ndim = x_shape.len();
        let batch = x_shape[0].as_const().unwrap();

        // Reshape to (batch, num_groups, -1), cast to f32 before layernorm
        let reshaped = self.try_reshape(&[batch as isize, num_groups as isize, -1])?;
        let reshaped = if reshaped.uop().dtype() != DType::Float32 { reshaped.cast(DType::Float32)? } else { reshaped };
        let normed = reshaped.layernorm(-1, eps)?;
        // Cast back and reshape to original
        let normed = if self.uop().dtype() != DType::Float32 { normed.cast(self.uop().dtype())? } else { normed };
        let orig_shape = morok_ir::shape::to_vec_isize(&x_shape).context(UOpSnafu)?;
        let normed = normed.try_reshape(&orig_shape)?;

        // Scale and bias: reshape to (1, C, 1, 1, ...)
        let mut sb_shape: Vec<isize> = vec![1, -1];
        sb_shape.extend(std::iter::repeat_n(1isize, ndim - 2));
        let scale = scale.try_reshape(&sb_shape)?;
        let bias = bias.try_reshape(&sb_shape)?;
        normed.try_mul(&scale)?.try_add(&bias)
    }
}
