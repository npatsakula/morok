//! Normalization: layernorm, rms_norm, group_norm.

use bon::bon;
use snafu::{OptionExt, ResultExt};
use svod_dtype::DType;

use crate::Tensor;
use crate::error::{NdimMinimumSnafu, ParamRangeSnafu, SymbolicShapeUnsupportedSnafu, UOpSnafu};
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
    /// # use svod_tensor::Tensor;
    /// # use ndarray::array;
    /// let x = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    /// let y = x.layernorm(-1, 1e-5).unwrap();
    /// y.realize().unwrap();
    /// let vals = y.as_vec::<f32>().unwrap();
    /// // Each row is independently normalized to mean~0, std~1
    /// assert!((vals[0] + vals[1] + vals[2]).abs() < 1e-5);
    /// ```
    #[track_caller]
    pub fn layernorm(&self, axis: isize, eps: f64) -> Result<Tensor> {
        origin_call!("layernorm");
        self.layernorm_with().axis(axis).eps(eps).call()
    }

    /// Layer normalization with optional affine parameters: `layernorm(x) * weight + bias`.
    ///
    /// `weight` and `bias` broadcast over the normalized trailing axes, so a
    /// `[D]` parameter matches an `[N, T, D]` input with `axis = -1`.
    ///
    /// Normalization *and* the affine step run in f32 and the result is cast
    /// back to the input dtype, so an f16/bf16 input with f32 parameters rounds
    /// exactly once, at the end.
    ///
    /// # Examples
    ///
    /// ```
    /// # use svod_tensor::Tensor;
    /// # use ndarray::array;
    /// let x = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0]]);
    /// let w = Tensor::from_slice([2.0f32; 3]);
    /// let b = Tensor::from_slice([1.0f32; 3]);
    /// let y = x.layernorm_with().weight(&w).bias(&b).call().unwrap();
    /// let vals = y.to_vec::<f32>().unwrap();
    /// // Middle element normalizes to 0, so it is left with the bias.
    /// assert!((vals[1] - 1.0).abs() < 1e-4);
    /// ```
    #[builder]
    #[track_caller]
    pub fn layernorm_with(
        &self,
        #[builder(default = -1)] axis: isize,
        #[builder(default = 1e-5)] eps: f64,
        weight: Option<&Tensor>,
        bias: Option<&Tensor>,
    ) -> Result<Tensor> {
        origin_call!("layernorm_with");
        let (normed, _, _) = self.layernorm_f32(axis, eps)?;
        normed.affine_f32(weight, bias, self.uop().dtype())
    }

    /// Layer normalization returning `(normalized, mean, inv_std_dev)`.
    ///
    /// Computes in f32 for numerical stability (matches ONNX `stash_type=1`).
    /// The `mean` and `inv_std_dev` tensors remain in f32 regardless of input dtype.
    ///
    /// # Examples
    ///
    /// ```
    /// # use svod_tensor::Tensor;
    /// # use ndarray::array;
    /// let x = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0]]);
    /// let (_normed, mut mean, _inv_std) = x.layernorm_with_stats(-1, 1e-5).unwrap();
    /// mean.realize().unwrap();
    /// let mean_val = mean.as_vec::<f32>().unwrap();
    /// assert!((mean_val[0] - 2.0).abs() < 1e-5);
    /// ```
    #[track_caller]
    pub fn layernorm_with_stats(&self, axis: isize, eps: f64) -> Result<(Tensor, Tensor, Tensor)> {
        origin_call!("layernorm_with_stats");
        let original_dtype = self.uop().dtype();
        let (normalized, mean, inv_std) = self.layernorm_f32(axis, eps)?;
        let normalized = if original_dtype != DType::Float32 { normalized.cast(original_dtype) } else { normalized };
        Ok((normalized, mean, inv_std))
    }

    /// RMS normalization over axes `[axis..ndim)`.
    ///
    /// Like layernorm but without mean subtraction: divides each element by the
    /// root-mean-square of its slice. Computes in f32, then casts the result
    /// back to the input dtype.
    ///
    /// # Examples
    ///
    /// ```
    /// # use svod_tensor::Tensor;
    /// # use ndarray::array;
    /// let x = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0]]);
    /// let y = x.rms_norm(-1, 1e-5).unwrap();
    /// y.realize().unwrap();
    /// let vals = y.as_vec::<f32>().unwrap();
    /// // RMS of [1,2,3] = sqrt((1+4+9)/3) ≈ 2.16
    /// // Output ≈ [0.46, 0.93, 1.39]
    /// assert!((vals[0] - 1.0 / (14.0f32 / 3.0).sqrt()).abs() < 1e-4);
    /// ```
    #[track_caller]
    pub fn rms_norm(&self, axis: isize, eps: f64) -> Result<Tensor> {
        origin_call!("rms_norm");
        self.rms_norm_with().axis(axis).eps(eps).call()
    }

    /// RMS normalization with an optional affine scale: `rms_norm(x) * weight`.
    ///
    /// `weight` broadcasts over the normalized trailing axes. Like
    /// [`layernorm_with`](Tensor::layernorm_with), the normalization and the
    /// scale run in f32 and the result is cast back to the input dtype, so an
    /// f16/bf16 input with an f32 `weight` rounds exactly once, at the end.
    ///
    /// # Examples
    ///
    /// ```
    /// # use svod_tensor::Tensor;
    /// # use ndarray::array;
    /// let x = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0]]);
    /// let w = Tensor::from_slice([2.0f32; 3]);
    /// let y = x.rms_norm_with().weight(&w).call().unwrap();
    /// let vals = y.to_vec::<f32>().unwrap();
    /// assert!((vals[0] - 2.0 / (14.0f32 / 3.0).sqrt()).abs() < 1e-4);
    /// ```
    #[builder]
    #[track_caller]
    pub fn rms_norm_with(
        &self,
        #[builder(default = -1)] axis: isize,
        #[builder(default = 1e-5)] eps: f64,
        weight: Option<&Tensor>,
    ) -> Result<Tensor> {
        origin_call!("rms_norm_with");
        let ndim = self.ndim()?;
        let norm_axis = Tensor::normalize_axis(axis, ndim)?;
        let axes_spec = AxisSpec::Multiple((norm_axis..ndim).map(|a| a as isize).collect());

        let x32 = self.cast_f32();
        let inv_rms = x32
            .square()
            .mean_with()
            .axes(axes_spec)
            .keepdim(true)
            .call()?
            .try_add(Tensor::const_(eps, DType::Float32))?
            .try_rsqrt()?;

        x32.try_mul(&inv_rms)?.affine_f32(weight, None, self.uop().dtype())
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
    /// # use svod_tensor::Tensor;
    /// # use ndarray::array;
    /// let x = Tensor::from_ndarray(&array![[3.0f32, 4.0]]);
    /// let y = x.lp_normalize(-1, 2).unwrap();
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
    /// # use svod_tensor::Tensor;
    /// # use ndarray::array;
    /// let x = Tensor::from_ndarray(&array![[3.0f32, 4.0]]);
    /// let y = x.lp_normalize(-1, 1).unwrap();
    /// y.realize().unwrap();
    /// let vals = y.as_vec::<f32>().unwrap();
    /// // L1 norm of [3,4] = 7, so output ≈ [3/7, 4/7]
    /// assert!((vals[0] - 3.0 / 7.0).abs() < 1e-5);
    /// ```
    #[track_caller]
    pub fn lp_normalize(&self, axis: isize, p: i64) -> Result<Tensor> {
        origin_call!("lp_normalize");
        let norm = match p {
            1 => self.abs().sum_with().axes(AxisSpec::Single(axis)).keepdim(true).call()?,
            _ => self.square().sum_with().axes(AxisSpec::Single(axis)).keepdim(true).call()?.try_sqrt()?,
        };
        let eps = self.uop().dtype().base().min_positive();
        self.try_div(&norm.try_add(Tensor::const_(eps, self.uop().dtype()))?)
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
    /// # use svod_tensor::Tensor;
    /// # use ndarray::array;
    /// let x = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    /// let y = x.mean_variance_normalize(&[0, 1], 1e-5).unwrap();
    /// y.realize().unwrap();
    /// let vals = y.as_vec::<f32>().unwrap();
    /// // Global mean = 3.5, std ≈ 1.708
    /// assert!((vals[0] - (1.0 - 3.5) / (35.0f32 / 12.0).sqrt()).abs() < 1e-4);
    /// assert!(vals[0] < 0.0);
    /// assert!(vals[5] > 0.0);
    /// ```
    #[track_caller]
    pub fn mean_variance_normalize(&self, axes: &[isize], eps: f64) -> Result<Tensor> {
        origin_call!("mean_variance_normalize");
        let axes_spec = AxisSpec::Multiple(axes.to_vec());
        // Normalize in f32 like `layernorm_with_stats`: a float16 `eps` is
        // subnormal, so a constant slice divides 0 by a flushed 0 and gives NaN.
        let original_dtype = self.uop().dtype();
        // Integer inputs keep the float32 result they always produced.
        let output_dtype = if original_dtype.is_float() { original_dtype.clone() } else { DType::Float32 };
        let x32 = if original_dtype != DType::Float32 { self.cast(DType::Float32) } else { self.clone() };

        let mean = x32.mean_with().axes(axes_spec.clone()).keepdim(true).call()?;
        let centered = x32.try_sub(&mean)?;
        let pop_std = centered.square().mean_with().axes(axes_spec).keepdim(true).call()?.try_sqrt()?;
        let eps = Tensor::const_(eps, DType::Float32);
        let normalized = centered.try_div(&pop_std.try_add(&eps)?)?;

        Ok(if output_dtype != DType::Float32 { normalized.cast(output_dtype) } else { normalized })
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
    /// # use svod_tensor::Tensor;
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
    /// # use svod_tensor::Tensor;
    /// # use ndarray::Array4;
    /// let x = Tensor::from_ndarray(&Array4::from_elem((1, 4, 2, 2), 1.0f32));
    /// let scale = Tensor::from_slice([1.0f32; 4]);
    /// let bias = Tensor::from_slice([0.0f32; 4]);
    /// let y = x.group_norm().scale(&scale).bias(&bias).num_groups(2).eps(1e-6).call().unwrap();
    /// let shape: Vec<_> = y.shape().unwrap().iter().map(|d| d.as_const().unwrap()).collect();
    /// assert_eq!(shape, [1, 4, 2, 2]);
    /// ```
    #[builder]
    #[track_caller]
    pub fn group_norm(
        &self,
        scale: &Tensor,
        bias: &Tensor,
        num_groups: usize,
        #[builder(default = 1e-5)] eps: f64,
    ) -> Result<Tensor> {
        origin_call!("group_norm");
        let x_shape = self.shape()?;
        let ndim = x_shape.len();
        snafu::ensure!(ndim >= 2, NdimMinimumSnafu { op: "group_norm", min: 2_usize, actual: ndim });
        snafu::ensure!(
            num_groups > 0,
            ParamRangeSnafu { op: "group_norm", param: "num_groups", value: num_groups.to_string(), constraint: "> 0" }
        );
        let batch = x_shape[0].as_const().context(SymbolicShapeUnsupportedSnafu { operation: "group_norm" })?;

        // Reshape to (batch, num_groups, -1), cast to f32 before layernorm
        let reshaped = self.try_reshape([batch as isize, num_groups as isize, -1])?;
        let reshaped = if reshaped.uop().dtype() != DType::Float32 { reshaped.cast(DType::Float32) } else { reshaped };
        let normed = reshaped.layernorm(-1, eps)?;
        // Cast back and reshape to original
        let normed = if self.uop().dtype() != DType::Float32 { normed.cast(self.uop().dtype()) } else { normed };
        let orig_shape = svod_ir::shape::to_vec_isize(&x_shape).context(UOpSnafu)?;
        let normed = normed.try_reshape(&orig_shape)?;

        // Scale and bias: reshape to (1, C, 1, 1, ...)
        let mut sb_shape: Vec<isize> = vec![1, -1];
        sb_shape.extend(std::iter::repeat_n(1isize, ndim - 2));
        let scale = scale.try_reshape(&sb_shape)?;
        let bias = bias.try_reshape(&sb_shape)?;
        normed.try_mul(&scale)?.try_add(&bias)
    }
}

impl Tensor {
    /// This tensor in f32, without a redundant cast when it already is.
    fn cast_f32(&self) -> Tensor {
        if self.uop().dtype() == DType::Float32 { self.clone() } else { self.cast(DType::Float32) }
    }

    /// f32 body shared by the `layernorm*` entry points: returns
    /// `(normalized, mean, inv_std_dev)`, all in f32.
    #[track_caller]
    fn layernorm_f32(&self, axis: isize, eps: f64) -> Result<(Tensor, Tensor, Tensor)> {
        let ndim = self.ndim()?;
        let norm_axis = Tensor::normalize_axis(axis, ndim)?;
        let axes_spec = AxisSpec::Multiple((norm_axis..ndim).map(|a| a as isize).collect());

        let x32 = self.cast_f32();
        let mean = x32.mean_with().axes(axes_spec.clone()).keepdim(true).call()?;
        let centered = x32.try_sub(&mean)?;
        let variance = centered.square().mean_with().axes(axes_spec).keepdim(true).call()?;
        let inv_std = variance.try_add(Tensor::const_(eps, DType::Float32))?.try_rsqrt()?;
        Ok((centered.try_mul(&inv_std)?, mean, inv_std))
    }

    /// `self * weight + bias` evaluated in f32 (parameters are widened first),
    /// then cast to `out_dtype`. `self` must already be f32.
    #[track_caller]
    fn affine_f32(self, weight: Option<&Tensor>, bias: Option<&Tensor>, out_dtype: DType) -> Result<Tensor> {
        let mut out = self;
        if let Some(w) = weight {
            out = out.try_mul(w.cast_f32())?;
        }
        if let Some(b) = bias {
            out = out.try_add(b.cast_f32())?;
        }
        Ok(if out_dtype != DType::Float32 { out.cast(out_dtype) } else { out })
    }
}
