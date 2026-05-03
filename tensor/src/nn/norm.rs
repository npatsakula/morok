//! Normalization: layernorm, rms_norm, group_norm.

use bon::bon;
use morok_dtype::DType;
use morok_ir::{ConstValue, UOp};
use snafu::ResultExt;

use crate::Tensor;
use crate::error::{NdimMinimumSnafu, ParamRangeSnafu, UOpSnafu};
use crate::reduce::AxisSpec;

type Result<T> = crate::Result<T>;

#[bon]
impl Tensor {
    /// Layer normalization over axes `[axis..ndim)`. Casts to f32 internally
    /// for numerical stability.
    ///
    /// Normalizes the input so that the slice along the specified trailing axes
    /// has zero mean and unit variance, then returns the result cast back to
    /// the original dtype.
    ///
    /// # Examples
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// # use ndarray::array;
    /// let x = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    /// let mut y = x.layernorm(-1, 1e-5).unwrap();
    /// y.realize().unwrap();
    /// let vals = y.as_vec::<f32>().unwrap();
    /// // Each row is independently normalized to mean~0, std~1
    /// assert!((vals[0] + vals[1] + vals[2]).abs() < 1e-5);
    /// ```
    pub fn layernorm(&self, axis: isize, eps: f64) -> Result<Tensor> {
        let (normed, _, _) = self.layernorm_with_stats(axis, eps)?;
        Ok(normed)
    }

    /// Layer normalization returning `(normalized, mean, inv_std_dev)`.
    ///
    /// Computes in f32 for numerical stability (matches ONNX `stash_type=1`).
    /// The `mean` and `inv_std_dev` tensors remain in f32 regardless of input dtype.
    ///
    /// # Examples
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// # use ndarray::array;
    /// let x = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0]]);
    /// let (_normed, mut mean, _inv_std) = x.layernorm_with_stats(-1, 1e-5).unwrap();
    /// mean.realize().unwrap();
    /// let mean_val = mean.as_vec::<f32>().unwrap();
    /// assert!((mean_val[0] - 2.0).abs() < 1e-5);
    /// ```
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

    /// RMS normalization over axes `[axis..ndim)`.
    ///
    /// Like layernorm but without mean subtraction: divides each element by the
    /// root-mean-square of its slice. Computes the normalization factor in f32,
    /// then multiplies the original (unconverted) input.
    ///
    /// # Examples
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// # use ndarray::array;
    /// let x = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0]]);
    /// let mut y = x.rms_norm(-1, 1e-5).unwrap();
    /// y.realize().unwrap();
    /// let vals = y.as_vec::<f32>().unwrap();
    /// // RMS of [1,2,3] = sqrt((1+4+9)/3) ≈ 2.16
    /// // Output ≈ [0.46, 0.93, 1.39]
    /// assert!((vals[0] - 1.0 / (14.0f32 / 3.0).sqrt()).abs() < 1e-4);
    /// ```
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

    /// Lp normalization along an axis.
    ///
    /// Divides each element by the Lp norm of its slice along `axis`,
    /// so that every such slice has unit Lp norm. Only `p=1` (L1) and
    /// `p=2` (L2) are implemented; any `p != 1` defaults to L2.
    ///
    /// # Examples
    ///
    /// L2 normalization (default `p=2`):
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// # use ndarray::array;
    /// let x = Tensor::from_ndarray(&array![[3.0f32, 4.0]]);
    /// let mut y = x.lp_normalize(-1, 2).unwrap();
    /// y.realize().unwrap();
    /// let vals = y.as_vec::<f32>().unwrap();
    /// // L2 norm of [3,4] = 5, so output ≈ [0.6, 0.8]
    /// assert!((vals[0] - 0.6).abs() < 1e-5);
    /// assert!((vals[1] - 0.8).abs() < 1e-5);
    /// ```
    ///
    /// L1 normalization (`p=1`):
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// # use ndarray::array;
    /// let x = Tensor::from_ndarray(&array![[3.0f32, 4.0]]);
    /// let mut y = x.lp_normalize(-1, 1).unwrap();
    /// y.realize().unwrap();
    /// let vals = y.as_vec::<f32>().unwrap();
    /// // L1 norm of [3,4] = 7, so output ≈ [3/7, 4/7]
    /// assert!((vals[0] - 3.0 / 7.0).abs() < 1e-5);
    /// ```
    pub fn lp_normalize(&self, axis: isize, p: i64) -> Result<Tensor> {
        let norm = match p {
            1 => self.try_abs()?.sum_with().axes(AxisSpec::Single(axis)).keepdim(true).call()?,
            _ => self.square()?.sum_with().axes(AxisSpec::Single(axis)).keepdim(true).call()?.try_sqrt()?,
        };
        let eps = self.uop().dtype().base().min_positive();
        self.try_div(&norm.try_add(&Tensor::const_(eps, self.uop().dtype()))?)
    }

    /// Mean Variance Normalization.
    ///
    /// Subtracts the mean and divides by the population standard deviation
    /// (plus `eps`) over the given axes. Implements the ONNX
    /// `MeanVarianceNormalization` operator.
    ///
    /// # Examples
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// # use ndarray::array;
    /// let x = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    /// let mut y = x.mean_variance_normalize(&[0, 1], 1e-5).unwrap();
    /// y.realize().unwrap();
    /// let vals = y.as_vec::<f32>().unwrap();
    /// // Global mean = 3.5, std ≈ 1.708
    /// assert!((vals[0] - (1.0 - 3.5) / (35.0f32 / 12.0).sqrt()).abs() < 1e-4);
    /// assert!(vals[0] < 0.0);
    /// assert!(vals[5] > 0.0);
    /// ```
    pub fn mean_variance_normalize(&self, axes: &[isize], eps: f64) -> Result<Tensor> {
        let axes_spec = AxisSpec::Multiple(axes.to_vec());
        let mean = self.mean_with().axes(axes_spec.clone()).keepdim(true).call()?;
        let centered = self.try_sub(&mean)?;
        let pop_std = centered.square()?.mean_with().axes(axes_spec).keepdim(true).call()?.try_sqrt()?;
        let eps = Tensor::const_(eps, self.uop().dtype());
        centered.try_div(&pop_std.try_add(&eps)?)
    }

    /// Group normalization: reshape into groups, layernorm each group, then
    /// apply per-channel scale and bias.
    ///
    /// Input must be at least 2-D with shape `[N, C, ...]`. Channels are split
    /// into `num_groups` groups and each group is independently normalized.
    /// Casts to f32 internally for numerical stability.
    ///
    /// # Examples
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// # use ndarray::Array4;
    /// let x = Tensor::from_ndarray(&Array4::from_elem((1, 4, 2, 2), 1.0f32));
    /// let scale = Tensor::from_slice([1.0f32; 4]);
    /// let bias = Tensor::from_slice([0.0f32; 4]);
    /// let y = x.group_norm().scale(&scale).bias(&bias).num_groups(2).call().unwrap();
    /// let shape: Vec<_> = y.shape().unwrap().iter().map(|d| d.as_const().unwrap()).collect();
    /// assert_eq!(shape, [1, 4, 2, 2]);
    /// ```
    ///
    /// Custom epsilon:
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// # use ndarray::Array4;
    /// let x = Tensor::from_ndarray(&Array4::from_elem((1, 4, 2, 2), 1.0f32));
    /// let scale = Tensor::from_slice([1.0f32; 4]);
    /// let bias = Tensor::from_slice([0.0f32; 4]);
    /// let y = x.group_norm().scale(&scale).bias(&bias).num_groups(2).eps(1e-6).call().unwrap();
    /// let shape: Vec<_> = y.shape().unwrap().iter().map(|d| d.as_const().unwrap()).collect();
    /// assert_eq!(shape, [1, 4, 2, 2]);
    /// ```
    #[builder]
    pub fn group_norm(
        &self,
        scale: &Tensor,
        bias: &Tensor,
        num_groups: usize,
        #[builder(default = 1e-5)] eps: f64,
    ) -> Result<Tensor> {
        let x_shape = self.shape()?;
        let ndim = x_shape.len();
        snafu::ensure!(ndim >= 2, NdimMinimumSnafu { op: "group_norm", min: 2_usize, actual: ndim });
        snafu::ensure!(
            num_groups > 0,
            ParamRangeSnafu { op: "group_norm", param: "num_groups", value: num_groups.to_string(), constraint: "> 0" }
        );
        let batch = x_shape[0].as_const().unwrap();

        // Reshape to (batch, num_groups, -1), cast to f32 before layernorm
        let reshaped = self.try_reshape([batch as isize, num_groups as isize, -1])?;
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
