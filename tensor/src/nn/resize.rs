//! Resize operations (ONNX Resize operator building block).

use bon::bon;
use morok_dtype::DType;
use morok_ir::ConstValue;
use snafu::ResultExt;

use super::{AspectRatioPolicy, CoordinateTransformMode, NearestMode, ResizeMode};
use crate::Tensor;
use crate::error::UOpSnafu;

type Result<T> = crate::Result<T>;

#[bon]
impl Tensor {
    /// Resize a tensor using interpolation (ONNX Resize operator).
    ///
    /// Supports nearest, linear, and cubic interpolation modes with various
    /// coordinate transformation modes. Either `scales` or `sizes` must be
    /// provided to specify the target dimensions.
    ///
    /// # Examples
    ///
    /// Nearest-mode 2x upscale via `scales`:
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// # use ndarray::Array4;
    /// let x = Tensor::from_ndarray(&Array4::from_elem((1, 1, 2, 2), 1.0f32));
    /// let y = x.resize().scales(&[1.0, 1.0, 2.0, 2.0]).call().unwrap();
    /// let shape: Vec<usize> = y.shape().unwrap().iter().map(|d| d.as_const().unwrap()).collect();
    /// assert_eq!(shape, vec![1, 1, 4, 4]);
    /// assert!(y.to_vec::<f32>().unwrap().iter().all(|&v| (v - 1.0).abs() < 1e-5));
    /// ```
    ///
    /// Resize to explicit output `sizes`:
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// # use ndarray::Array4;
    /// let x = Tensor::from_ndarray(&Array4::from_elem((1, 1, 2, 2), 1.0f32));
    /// let y = x.resize().sizes(&[1, 1, 6, 6]).call().unwrap();
    /// let shape: Vec<usize> = y.shape().unwrap().iter().map(|d| d.as_const().unwrap()).collect();
    /// assert_eq!(shape, vec![1, 1, 6, 6]);
    /// assert!(y.to_vec::<f32>().unwrap().iter().all(|&v| (v - 1.0).abs() < 1e-5));
    /// ```
    ///
    /// Linear interpolation mode:
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// # use morok_tensor::nn::ResizeMode;
    /// # use ndarray::Array4;
    /// let x = Tensor::from_ndarray(&Array4::from_elem((1, 1, 2, 2), 1.0f32));
    /// let y = x.resize()
    ///     .scales(&[1.0, 1.0, 2.0, 2.0])
    ///     .mode(ResizeMode::Linear)
    ///     .call()
    ///     .unwrap();
    /// let shape: Vec<usize> = y.shape().unwrap().iter().map(|d| d.as_const().unwrap()).collect();
    /// assert_eq!(shape, vec![1, 1, 4, 4]);
    /// assert!(y.to_vec::<f32>().unwrap().iter().all(|&v| (v - 1.0).abs() < 1e-5));
    /// ```
    // Tinygrad onnx.py:789-890
    #[builder]
    #[allow(clippy::too_many_arguments)]
    pub fn resize(
        &self,
        scales: Option<&[f64]>,
        sizes: Option<&[usize]>,
        #[builder(default)] mode: ResizeMode,
        #[builder(default)] coordinate_transformation_mode: CoordinateTransformMode,
        #[builder(default)] nearest_mode: NearestMode,
        #[builder(default = -0.75)] cubic_coeff_a: f64,
        #[builder(default = false)] exclude_outside: bool,
        #[builder(default = false)] antialias: bool,
        #[builder(default)] keep_aspect_ratio_policy: AspectRatioPolicy,
        axes: Option<&[usize]>,
        roi: Option<&[f64]>,
        #[builder(default = 0.0)] extrapolation_value: f64,
    ) -> Result<Tensor> {
        let ndim = self.ndim()?;
        let shape = self.shape()?;
        let _shape_dims = morok_ir::shape::to_vec_usize(&shape).context(UOpSnafu)?;

        let axes: Vec<usize> = axes.map(|a| a.to_vec()).unwrap_or_else(|| (0..ndim).collect());

        // Permute: put target axes last
        let non_axes: Vec<usize> = (0..ndim).filter(|d| !axes.contains(d)).collect();
        let perm: Vec<isize> = non_axes.iter().chain(axes.iter()).map(|&d| d as isize).collect();
        let inv_perm = argsort_usize(&perm.iter().map(|&p| p as usize).collect::<Vec<_>>());
        let inv_perm_i: Vec<isize> = inv_perm.iter().map(|&i| i as isize).collect();

        let mut x = if perm.iter().enumerate().all(|(i, &p)| p == i as isize) {
            self.clone()
        } else {
            self.try_permute(&perm)?
        };

        // Input spatial dimensions (last len(axes) dims of permuted x)
        let x_shape = x.shape()?;
        let x_dims = morok_ir::shape::to_vec_usize(&x_shape).context(UOpSnafu)?;
        let n_spatial = axes.len();
        let input_shape: Vec<usize> = x_dims[ndim - n_spatial..].to_vec();

        // Filter scales/sizes to spatial dims only
        let scales_trimmed: Option<Vec<f64>> = scales.map(|s| s[s.len().saturating_sub(n_spatial)..].to_vec());
        let sizes_trimmed: Option<Vec<usize>> = sizes.map(|s| s[s.len().saturating_sub(n_spatial)..].to_vec());

        // Compute output sizes and scales
        let (output_sizes, final_scales) = if let Some(mut sz) = sizes_trimmed {
            if keep_aspect_ratio_policy == AspectRatioPolicy::NotLarger
                || keep_aspect_ratio_policy == AspectRatioPolicy::NotSmaller
            {
                let scale_fn: fn(f64, f64) -> f64 =
                    if keep_aspect_ratio_policy == AspectRatioPolicy::NotLarger { f64::min } else { f64::max };
                let mut scale = f64::NAN;
                for (s, &inp) in sz.iter().zip(&input_shape) {
                    let s_val = *s as f64 / inp as f64;
                    if scale.is_nan() {
                        scale = s_val;
                    } else {
                        scale = scale_fn(scale, s_val);
                    }
                }
                sz = input_shape.iter().map(|&sh| (scale * sh as f64 + 0.5) as usize).collect();
                let sc = vec![scale; n_spatial];
                (sz, sc)
            } else {
                let sc: Vec<f64> = sz.iter().zip(&input_shape).map(|(&s, &sh)| s as f64 / sh as f64).collect();
                (sz, sc)
            }
        } else if let Some(sc) = scales_trimmed {
            let sz: Vec<usize> = sc.iter().zip(&input_shape).map(|(&s, &sh)| (s * sh as f64) as usize).collect();
            (sz, sc)
        } else {
            return Err(crate::error::Error::IrConstruction {
                details: "resize: either scales or sizes must be provided".into(),
            });
        };

        // Early exit if no resize needed
        if output_sizes.iter().zip(&input_shape).all(|(&o, &i)| o == i) {
            return if perm.iter().enumerate().any(|(i, &p)| p != i as isize) {
                x.try_permute(&inv_perm_i)
            } else {
                Ok(x)
            };
        }

        // Extract per-spatial-dim ROI (start, end) pairs
        let roi_pairs: Vec<(f64, f64)> = if let Some(roi) = roi {
            let half = roi.len() / 2;
            let starts = &roi[half - n_spatial..half];
            let ends = &roi[roi.len() - n_spatial..];
            starts.iter().zip(ends).map(|(&s, &e)| (s, e)).collect()
        } else {
            vec![(0.0, 1.0); n_spatial]
        };

        // Build coordinate transforms for each spatial dim
        let dtype = x.uop().dtype();
        let indexes: Vec<Tensor> = input_shape
            .iter()
            .zip(&output_sizes)
            .zip(&final_scales)
            .zip(&roi_pairs)
            .map(|(((&inp_sz, &out_sz), &scale), &(roi_start, roi_end))| {
                apply_coordinate_transform(
                    inp_sz,
                    out_sz,
                    scale,
                    coordinate_transformation_mode,
                    &dtype,
                    roi_start,
                    roi_end,
                )
            })
            .collect::<Result<_>>()?;

        // Clip for nearest/linear modes (skip for tf_crop_and_resize — uses extrapolation instead)
        let is_tf_crop = coordinate_transformation_mode == CoordinateTransformMode::TfCropAndResize;
        let indexes: Vec<Tensor> = if !is_tf_crop && matches!(mode, ResizeMode::Nearest | ResizeMode::Linear) {
            indexes
                .into_iter()
                .zip(&input_shape)
                .map(|(idx, &sz)| {
                    let zero = Tensor::const_(ConstValue::Float(0.0), dtype.clone());
                    let max_val = Tensor::const_(ConstValue::Float((sz - 1) as f64), dtype.clone());
                    idx.clamp().min(&zero).max(&max_val).call()
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            indexes
        };

        // For tf_crop_and_resize, build a validity mask from unclipped indexes
        let validity_mask: Option<Vec<Tensor>> = if is_tf_crop {
            Some(
                indexes
                    .iter()
                    .zip(&input_shape)
                    .map(|(idx, &sz)| {
                        let zero = Tensor::const_(ConstValue::Float(0.0), dtype.clone());
                        let max_val = Tensor::const_(ConstValue::Float((sz - 1) as f64), dtype.clone());
                        idx.try_ge(&zero)?.bitwise_and(&idx.try_le(&max_val)?)
                    })
                    .collect::<Result<Vec<_>>>()?,
            )
        } else {
            None
        };

        // For tf_crop_and_resize, clip indexes before gather (to avoid OOB)
        let indexes: Vec<Tensor> = if is_tf_crop {
            indexes
                .into_iter()
                .zip(&input_shape)
                .map(|(idx, &sz)| {
                    let zero = Tensor::const_(ConstValue::Float(0.0), dtype.clone());
                    let max_val = Tensor::const_(ConstValue::Float((sz - 1) as f64), dtype.clone());
                    idx.clamp().min(&zero).max(&max_val).call()
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            indexes
        };

        if mode == ResizeMode::Nearest {
            let int_indexes: Vec<Tensor> = indexes
                .into_iter()
                .map(|idx| {
                    let rounded = match nearest_mode {
                        NearestMode::RoundPreferFloor => idx.try_sub(&Tensor::const_(0.5f64, dtype.clone()))?.ceil()?,
                        NearestMode::RoundPreferCeil => idx.try_add(&Tensor::const_(0.5f64, dtype.clone()))?.floor()?,
                        NearestMode::Floor => idx.floor()?,
                        NearestMode::Ceil => idx.ceil()?,
                    };
                    rounded.cast(DType::Int32)
                })
                .collect::<Result<Vec<_>>>()?;

            // Sequential gather per spatial dim
            for (i, idx) in int_indexes.iter().enumerate() {
                let dim = (ndim - n_spatial + i) as isize;
                let cur_shape = x.shape()?;
                let cur_dims = morok_ir::shape::to_vec_usize(&cur_shape).context(UOpSnafu)?;
                let out_sz = output_sizes[i];

                let mut idx_shape = vec![1isize; ndim];
                idx_shape[ndim - n_spatial + i] = out_sz as isize;
                let idx_reshaped = idx.try_reshape(&idx_shape)?;

                let mut expand_shape: Vec<isize> = cur_dims.iter().map(|&d| d as isize).collect();
                expand_shape[ndim - n_spatial + i] = out_sz as isize;
                let idx_expanded = idx_reshaped.try_expand(&expand_shape)?;

                x = x.gather(dim, &idx_expanded)?;
            }
        } else if mode == ResizeMode::Linear {
            let mut expand = x_dims.clone();
            for (i, &out_sz) in output_sizes.iter().enumerate() {
                let dim_pos = ndim - n_spatial + i;
                let scale = final_scales[i];
                let input_sz = input_shape[i];
                let index = &indexes[i];

                let mut reshape = vec![1isize; ndim];
                reshape[dim_pos] = out_sz as isize;
                expand[dim_pos] = out_sz;
                let expand_i: Vec<isize> = expand.iter().map(|&d| d as isize).collect();

                if antialias && scale < 1.0 {
                    x = interpolate_antialias_linear(&x, index, dim_pos, input_sz, scale, &reshape, &expand_i, &dtype)?;
                } else {
                    let low = index.floor()?.cast(DType::Int32)?.try_reshape(&reshape)?.try_expand(&expand_i)?;
                    let high = index.ceil()?.cast(DType::Int32)?.try_reshape(&reshape)?.try_expand(&expand_i)?;
                    let perc = index.try_sub(&index.floor()?)?.try_reshape(&reshape)?.try_expand(&expand_i)?;

                    let dim_i = dim_pos as isize;
                    let gathered_low = x.gather(dim_i, &low)?;
                    let gathered_high = x.gather(dim_i, &high)?;
                    x = gathered_low.lerp(&gathered_high, &perc)?;
                }
            }
        } else if mode == ResizeMode::Cubic {
            let a = cubic_coeff_a;
            let mut expand = x_dims.clone();
            for (i, &out_sz) in output_sizes.iter().enumerate() {
                let dim_pos = ndim - n_spatial + i;
                let scale = final_scales[i];
                let input_sz = input_shape[i];
                let index = &indexes[i];

                let mut reshape = vec![1isize; ndim];
                reshape[dim_pos] = out_sz as isize;
                expand[dim_pos] = out_sz;
                let expand_i: Vec<isize> = expand.iter().map(|&d| d as isize).collect();

                if antialias && scale < 1.0 {
                    x = interpolate_antialias_cubic(
                        &x, index, dim_pos, input_sz, scale, a, &reshape, &expand_i, &dtype,
                    )?;
                } else {
                    let p = index.floor()?.cast(DType::Int32)?;
                    let ratio = index.try_sub(&index.floor()?)?;

                    let one = Tensor::const_(ConstValue::Int(1), DType::Int32);
                    let two = Tensor::const_(ConstValue::Int(2), DType::Int32);
                    let idx0 = p.try_sub(&one)?;
                    let idx1 = p.clone();
                    let idx2 = p.try_add(&one)?;
                    let idx3 = p.try_add(&two)?;

                    let r1 = ratio.try_add(&Tensor::const_(1.0f64, dtype.clone()))?;
                    let c0 = poly_n(&r1, &[a, -5.0 * a, 8.0 * a, -4.0 * a], &dtype)?;
                    let c1 = poly_n(&ratio, &[a + 2.0, -(a + 3.0), 0.0, 1.0], &dtype)?;
                    let r_neg1 = Tensor::const_(1.0f64, dtype.clone()).try_sub(&ratio)?;
                    let c2 = poly_n(&r_neg1, &[a + 2.0, -(a + 3.0), 0.0, 1.0], &dtype)?;
                    let r_neg2 = Tensor::const_(2.0f64, dtype.clone()).try_sub(&ratio)?;
                    let c3 = poly_n(&r_neg2, &[a, -5.0 * a, 8.0 * a, -4.0 * a], &dtype)?;

                    let (mut c0, mut c1, mut c2, mut c3) = (c0, c1, c2, c3);
                    if exclude_outside {
                        let max_idx = Tensor::const_(ConstValue::Int(input_sz as i64), DType::Int32);
                        let zero_i = Tensor::const_(ConstValue::Int(0), DType::Int32);
                        let zero_f = Tensor::const_(0.0f64, dtype.clone());
                        let valid0 = idx0.try_ge(&zero_i)?.try_mul(&idx0.try_lt(&max_idx)?)?;
                        let valid1 = idx1.try_ge(&zero_i)?.try_mul(&idx1.try_lt(&max_idx)?)?;
                        let valid2 = idx2.try_ge(&zero_i)?.try_mul(&idx2.try_lt(&max_idx)?)?;
                        let valid3 = idx3.try_ge(&zero_i)?.try_mul(&idx3.try_lt(&max_idx)?)?;
                        c0 = c0.where_(&valid0, &zero_f)?;
                        c1 = c1.where_(&valid1, &zero_f)?;
                        c2 = c2.where_(&valid2, &zero_f)?;
                        c3 = c3.where_(&valid3, &zero_f)?;
                        let total = c0.try_add(&c1)?.try_add(&c2)?.try_add(&c3)?;
                        let eps = Tensor::const_(1e-9f64, dtype.clone());
                        let total_safe = total.try_add(&eps)?;
                        c0 = c0.try_div(&total_safe)?;
                        c1 = c1.try_div(&total_safe)?;
                        c2 = c2.try_div(&total_safe)?;
                        c3 = c3.try_div(&total_safe)?;
                    }

                    let max_val = Tensor::const_(ConstValue::Int((input_sz - 1) as i64), DType::Int32);
                    let zero_i = Tensor::const_(ConstValue::Int(0), DType::Int32);
                    let clip = |t: &Tensor| -> Result<Tensor> {
                        t.clamp().min(&zero_i).max(&max_val).call()?.try_reshape(&reshape)?.try_expand(&expand_i)
                    };
                    let ei0 = clip(&idx0)?;
                    let ei1 = clip(&idx1)?;
                    let ei2 = clip(&idx2)?;
                    let ei3 = clip(&idx3)?;

                    let ec = |c: Tensor| -> Result<Tensor> { c.try_reshape(&reshape)?.try_expand(&expand_i) };
                    let ec0 = ec(c0)?;
                    let ec1 = ec(c1)?;
                    let ec2 = ec(c2)?;
                    let ec3 = ec(c3)?;

                    let dim_i = dim_pos as isize;
                    let v0 = x.gather(dim_i, &ei0)?.try_mul(&ec0)?;
                    let v1 = x.gather(dim_i, &ei1)?.try_mul(&ec1)?;
                    let v2 = x.gather(dim_i, &ei2)?.try_mul(&ec2)?;
                    let v3 = x.gather(dim_i, &ei3)?.try_mul(&ec3)?;
                    x = v0.try_add(&v1)?.try_add(&v2)?.try_add(&v3)?;
                }
            }
        }

        // Apply extrapolation for tf_crop_and_resize: out-of-bounds → extrapolation_value
        if let Some(masks) = validity_mask {
            let extrap = Tensor::const_(ConstValue::Float(extrapolation_value), dtype.clone());
            let x_shape = x.shape()?;
            let x_dims = morok_ir::shape::to_vec_usize(&x_shape).context(UOpSnafu)?;
            let expand_shape: Vec<isize> = x_dims.iter().map(|&d| d as isize).collect();

            // Each mask_i is 1D [out_sz_i]; reshape to [1,..,out_sz_i,..,1] and broadcast
            let mut combined: Option<Tensor> = None;
            for (i, mask) in masks.into_iter().enumerate() {
                let mut shape = vec![1isize; ndim];
                shape[ndim - n_spatial + i] = output_sizes[i] as isize;
                let broad = mask.try_reshape(&shape)?.try_expand(&expand_shape)?;
                combined = Some(match combined {
                    Some(c) => c.bitwise_and(&broad)?,
                    None => broad,
                });
            }
            if let Some(valid) = combined {
                x = x.where_(&valid, &extrap)?;
            }
        }

        // Permute back
        if perm.iter().enumerate().any(|(i, &p)| p != i as isize) { x.try_permute(&inv_perm_i) } else { Ok(x) }
    }
}

/// Coordinate transform for resize operations.
///
/// Computes in f64 to avoid precision loss from IR constant folding
/// (which uses mixed f64/f32 arithmetic), then casts to target dtype.
fn apply_coordinate_transform(
    input_sz: usize,
    output_sz: usize,
    scale: f64,
    mode: CoordinateTransformMode,
    dtype: &DType,
    roi_start: f64,
    roi_end: f64,
) -> Result<Tensor> {
    let f64_dt = DType::Float64;
    let index = Tensor::arange(0, Some(output_sz as i64), None)?.cast(f64_dt.clone())?;
    let result = match mode {
        CoordinateTransformMode::HalfPixel => {
            let half = Tensor::const_(0.5f64, f64_dt.clone());
            index.try_add(&half)?.try_div(&Tensor::const_(scale, f64_dt))?.try_sub(&half)?
        }
        CoordinateTransformMode::AlignCorners => {
            // ONNX reference uses float output_width = scale * input_sz, not integer output_sz.
            // This matters when scale * input_sz is non-integer (e.g. 0.8 * 4 = 3.2 vs int 3).
            let output_width = scale * input_sz as f64;
            if output_width == 1.0 {
                Tensor::const_(0.0f64, f64_dt)
            } else {
                let ratio = (input_sz as f64 - 1.0) / (output_width - 1.0);
                index.try_mul(&Tensor::const_(ratio, f64_dt))?
            }
        }
        CoordinateTransformMode::Asymmetric => index.try_div(&Tensor::const_(scale, f64_dt))?,
        CoordinateTransformMode::PytorchHalfPixel => {
            let output_width = scale * input_sz as f64;
            if output_width == 1.0 {
                Tensor::const_(0.0f64, f64_dt)
            } else {
                let half = Tensor::const_(0.5f64, f64_dt.clone());
                index.try_add(&half)?.try_div(&Tensor::const_(scale, f64_dt))?.try_sub(&half)?
            }
        }
        CoordinateTransformMode::HalfPixelSymmetric => {
            let output_dim_scaled = input_sz as f64 * scale;
            let offset = (input_sz as f64 / 2.0) * (1.0 - output_sz as f64 / output_dim_scaled);
            let half = Tensor::const_(0.5f64, f64_dt.clone());
            let off_t = Tensor::const_(offset, f64_dt.clone());
            off_t.try_add(&index.try_add(&half)?.try_div(&Tensor::const_(scale, f64_dt))?)?.try_sub(&half)?
        }
        CoordinateTransformMode::TfCropAndResize => {
            let len = (input_sz as f64) - 1.0;
            let output_width = scale * input_sz as f64;
            if output_width == 1.0 {
                Tensor::const_((roi_end - roi_start) * len / 2.0 + roi_start * len, f64_dt)
            } else {
                let stride = (roi_end - roi_start) * len / (output_width - 1.0);
                let offset = roi_start * len;
                index.try_mul(&Tensor::const_(stride, f64_dt.clone()))?.try_add(&Tensor::const_(offset, f64_dt))?
            }
        }
    };
    result.cast(dtype.clone())
}

/// Horner's method for polynomial evaluation.
fn poly_n(x: &Tensor, coeffs: &[f64], dtype: &DType) -> Result<Tensor> {
    coeffs.iter().try_fold(Tensor::const_(0.0f64, dtype.clone()), |acc, &c| {
        acc.try_mul(x)?.try_add(&Tensor::const_(c, dtype.clone()))
    })
}

/// Antialias cubic interpolation for one spatial dimension.
/// When downsampling (scale < 1), widens the kernel by 1/scale to prevent aliasing.
/// ONNX ref: _cubic_coeffs_antialias in op_resize.py
#[allow(clippy::too_many_arguments)]
fn interpolate_antialias_cubic(
    x: &Tensor,
    index: &Tensor,
    dim_pos: usize,
    input_sz: usize,
    scale: f64,
    a: f64,
    reshape: &[isize],
    expand_i: &[isize],
    dtype: &DType,
) -> Result<Tensor> {
    let i_start = (-2.0_f64 / scale).floor() as i32 + 1;
    let i_end = 2 - i_start;
    let n_taps = (i_end - i_start) as usize;

    let floored = index.floor()?;
    let p = floored.cast(DType::Int32)?;
    let ratio = index.try_sub(&floored)?;

    let one = Tensor::const_(1.0f64, dtype.clone());
    let two = Tensor::const_(2.0f64, dtype.clone());
    let zero_f = Tensor::const_(0.0f64, dtype.clone());

    let mut coeffs = Vec::with_capacity(n_taps);
    for tap in i_start..i_end {
        let arg = ratio
            .try_mul(&Tensor::const_(-scale, dtype.clone()))?
            .try_add(&Tensor::const_(scale * tap as f64, dtype.clone()))?;
        let abs_arg = arg.try_abs()?;
        let c_inner = poly_n(&abs_arg, &[a + 2.0, -(a + 3.0), 0.0, 1.0], dtype)?;
        let c_outer = poly_n(&abs_arg, &[a, -5.0 * a, 8.0 * a, -4.0 * a], dtype)?;
        let mask_outer = abs_arg.try_lt(&two)?;
        let c = c_outer.where_(&mask_outer, &zero_f)?;
        let mask_inner = abs_arg.try_le(&one)?;
        let c = c_inner.where_(&mask_inner, &c)?;
        coeffs.push(c);
    }

    normalize_and_gather(x, coeffs, &p, i_start, dim_pos, input_sz, reshape, expand_i, dtype)
}

/// Antialias linear interpolation for one spatial dimension.
/// ONNX ref: _linear_coeffs_antialias in op_resize.py
#[allow(clippy::too_many_arguments)]
fn interpolate_antialias_linear(
    x: &Tensor,
    index: &Tensor,
    dim_pos: usize,
    input_sz: usize,
    scale: f64,
    reshape: &[isize],
    expand_i: &[isize],
    dtype: &DType,
) -> Result<Tensor> {
    let start = (-1.0_f64 / scale).floor() as i32 + 1;
    let footprint = (2 - 2 * start) as usize;

    let floored = index.floor()?;
    let p = floored.cast(DType::Int32)?;
    let ratio = index.try_sub(&floored)?;

    let one = Tensor::const_(1.0f64, dtype.clone());
    let zero_f = Tensor::const_(0.0f64, dtype.clone());

    let mut coeffs = Vec::with_capacity(footprint);
    for j in 0..footprint {
        let tap = start + j as i32;
        let arg = ratio
            .try_mul(&Tensor::const_(-scale, dtype.clone()))?
            .try_add(&Tensor::const_(scale * tap as f64, dtype.clone()))?;
        let abs_arg = arg.try_abs()?;
        let c = one.try_sub(&abs_arg)?;
        let c = c.clamp().min(&zero_f).max(&one).call()?;
        coeffs.push(c);
    }

    normalize_and_gather(x, coeffs, &p, start, dim_pos, input_sz, reshape, expand_i, dtype)
}

/// Normalize coefficients to sum to 1, then gather and accumulate weighted values.
/// Shared by antialias cubic and linear interpolation.
#[allow(clippy::too_many_arguments)]
fn normalize_and_gather(
    x: &Tensor,
    mut coeffs: Vec<Tensor>,
    p: &Tensor,
    tap_start: i32,
    dim_pos: usize,
    input_sz: usize,
    reshape: &[isize],
    expand_i: &[isize],
    dtype: &DType,
) -> Result<Tensor> {
    let mut total = coeffs[0].clone();
    for c in &coeffs[1..] {
        total = total.try_add(c)?;
    }
    let eps = Tensor::const_(1e-9f64, dtype.clone());
    let total_safe = total.try_add(&eps)?;
    for c in &mut coeffs {
        *c = c.try_div(&total_safe)?;
    }

    let max_val = Tensor::const_(ConstValue::Int((input_sz - 1) as i64), DType::Int32);
    let zero_i = Tensor::const_(ConstValue::Int(0), DType::Int32);
    let dim_i = dim_pos as isize;

    let mut result: Option<Tensor> = None;
    for (j, c) in coeffs.into_iter().enumerate() {
        let tap = tap_start + j as i32;
        let idx = p.try_add(&Tensor::const_(ConstValue::Int(tap as i64), DType::Int32))?;
        let idx_clipped = idx.clamp().min(&zero_i).max(&max_val).call()?.try_reshape(reshape)?.try_expand(expand_i)?;
        let c_expanded = c.try_reshape(reshape)?.try_expand(expand_i)?;
        let val = x.gather(dim_i, &idx_clipped)?.try_mul(&c_expanded)?;
        result = Some(match result {
            Some(acc) => acc.try_add(&val)?,
            None => val,
        });
    }
    Ok(result.unwrap())
}

fn argsort_usize(slice: &[usize]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..slice.len()).collect();
    indices.sort_by_key(|&i| slice[i]);
    indices
}
