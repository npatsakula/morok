use morok_dtype::{DType, ScalarDType};
use morok_tensor::Tensor;
use morok_tensor::nn::{
    AspectRatioPolicy, AutoPad, CoordinateTransformMode, GridSampleMode, GridSamplePaddingMode, NearestMode, Reduction,
    ResizeMode,
};
use morok_tensor::reduce::AxisSpec;
use morok_tensor::shape_ops::MeshgridIndexing;

use crate::error::{Error, Result};
use crate::parser::onnx::NodeProto;

use super::*;

/// Smallest positive normal value for a float dtype.
fn dtype_min_positive(dtype: DType) -> f64 {
    match dtype.scalar().expect("scalar dtype") {
        ScalarDType::Float16 => 6.103515625e-05,        // 2^-14
        ScalarDType::BFloat16 => 1.175494350822288e-38, // 2^-126 (same exponent range as f32)
        ScalarDType::Float32 => f32::MIN_POSITIVE as f64,
        ScalarDType::Float64 => f64::MIN_POSITIVE,
        _ => f32::MIN_POSITIVE as f64,
    }
}

/// Parse an ONNX string attribute into a typed enum.
fn parse_enum<T: std::str::FromStr>(node: &NodeProto, attr: &str, default: &str) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    let s = get_attr_string(node, attr, default);
    s.parse::<T>().map_err(|e| Error::IrConstruction { details: format!("{attr}='{s}': {e}") })
}

pub(crate) fn op_gemm(inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Tensor> {
    let alpha = get_attr_float(node, "alpha", 1.0);
    let beta = get_attr_float(node, "beta", 1.0);
    let a = inp(inputs, 0);
    let b = inp(inputs, 1);
    let a = if get_attr_int(node, "transA", 0) == 1 { a.try_transpose(0, 1)? } else { a.clone() };
    let b = if get_attr_int(node, "transB", 0) == 1 { b.try_transpose(0, 1)? } else { b.clone() };
    let mut result = a.matmul(&b)?;
    if alpha != 1.0 {
        result = result.try_mul(&Tensor::from_slice([alpha]))?;
    }
    if let Some(c) = inputs.get(2).and_then(|o| o.as_ref()) {
        let c = if beta != 1.0 { c.try_mul(&Tensor::from_slice([beta]))? } else { c.clone() };
        result = result.try_add(&c)?;
    }
    Ok(result)
}

pub(crate) fn op_batch_norm(inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Vec<Tensor>> {
    let (x, scale, bias, running_mean, running_var) =
        (inp(inputs, 0), inp(inputs, 1), inp(inputs, 2), inp(inputs, 3), inp(inputs, 4));
    let epsilon = get_attr_float(node, "epsilon", 1e-5);
    let training_mode = get_attr_int(node, "training_mode", 0) == 1;

    if training_mode {
        let momentum = get_attr_float(node, "momentum", 0.9) as f64;
        let shape = x.shape()?;
        let ndim = shape.len();
        // Reduce over all dims except channel (dim 1)
        let reduce_axes: Vec<isize> = (0..ndim).filter(|&i| i != 1).map(|i| i as isize).collect();
        let axes = AxisSpec::Multiple(reduce_axes);

        // Compute batch stats in f32
        let x32 = if x.uop().dtype() != DType::Float32 { x.cast(DType::Float32)? } else { x.clone() };
        let batch_mean = x32.mean_with().axes(axes.clone()).keepdim(false).call()?;
        let centered = x32.try_sub(&batch_mean.try_reshape(&{
            let mut s = vec![1isize; ndim];
            s[1] = -1;
            s
        })?)?;
        // Population variance (correction=0): mean(x²) not mean(x²)*N/(N-1)
        let batch_var = centered.square()?.mean_with().axes(axes).keepdim(false).call()?;

        // EMA update: running = batch * (1 - momentum) + running * momentum
        let m = Tensor::const_(momentum, DType::Float64);
        let one_minus_m = Tensor::const_(1.0 - momentum, DType::Float64);
        let new_running_mean = batch_mean
            .cast(DType::Float64)?
            .try_mul(&one_minus_m)?
            .try_add(&running_mean.cast(DType::Float64)?.try_mul(&m)?)?
            .cast(running_mean.uop().dtype())?;
        let new_running_var = batch_var
            .cast(DType::Float64)?
            .try_mul(&one_minus_m)?
            .try_add(&running_var.cast(DType::Float64)?.try_mul(&m)?)?
            .cast(running_var.uop().dtype())?;

        // Normalize with batch stats, cast back to input dtype
        let invstd = batch_var.try_add(&Tensor::from_slice([epsilon]))?.try_rsqrt()?;
        let out = x.batchnorm().scale(scale).bias(bias).mean(&batch_mean).invstd(&invstd).call()?;
        let out = out.cast(x.uop().dtype())?;
        Ok(vec![out, new_running_mean, new_running_var])
    } else {
        let invstd = running_var.try_add(&Tensor::from_slice([epsilon]))?.try_rsqrt()?;
        let out = x.batchnorm().scale(scale).bias(bias).mean(running_mean).invstd(&invstd).call()?;
        Ok(vec![out])
    }
}

pub(crate) fn op_conv(inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Tensor> {
    let ks: Vec<usize> = get_attr_ints(node, "kernel_shape").iter().map(|&k| k as usize).collect();
    let auto_pad: AutoPad = parse_enum(node, "auto_pad", "NOTSET")?;
    let pads = get_attr_ints(node, "pads");
    let strides = get_attr_ints(node, "strides");
    let dilations = get_attr_ints(node, "dilations");
    Ok(inp(inputs, 0)
        .conv()
        .weight(inp(inputs, 1))
        .maybe_bias(inputs.get(2).and_then(|o| o.as_ref()))
        .auto_pad(auto_pad)
        .group(get_attr_int(node, "group", 1) as usize)
        .maybe_kernel_shape((!ks.is_empty()).then_some(ks.as_slice()))
        .maybe_pads(non_empty_i64(&pads))
        .maybe_strides(non_empty_i64(&strides))
        .maybe_dilations(non_empty_i64(&dilations))
        .call()?)
}

pub(crate) fn op_conv_transpose(inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Tensor> {
    let ks: Vec<usize> = get_attr_ints(node, "kernel_shape").iter().map(|&k| k as usize).collect();
    let op: Vec<usize> = get_attr_ints(node, "output_padding").iter().map(|&p| p as usize).collect();
    let os = get_attr_ints(node, "output_shape");
    let auto_pad: AutoPad = parse_enum(node, "auto_pad", "NOTSET")?;
    let pads = get_attr_ints(node, "pads");
    let strides = get_attr_ints(node, "strides");
    let dilations = get_attr_ints(node, "dilations");
    Ok(inp(inputs, 0)
        .conv_transpose()
        .weight(inp(inputs, 1))
        .maybe_bias(inputs.get(2).and_then(|o| o.as_ref()))
        .auto_pad(auto_pad)
        .group(get_attr_int(node, "group", 1) as usize)
        .maybe_kernel_shape((!ks.is_empty()).then_some(ks.as_slice()))
        .maybe_pads(non_empty_i64(&pads))
        .maybe_output_shape(non_empty_i64(&os))
        .maybe_output_padding((!op.is_empty()).then_some(op.as_slice()))
        .maybe_strides(non_empty_i64(&strides))
        .maybe_dilations(non_empty_i64(&dilations))
        .call()?)
}

pub(crate) fn op_avg_pool(inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Tensor> {
    let kernel: Vec<usize> = get_attr_ints(node, "kernel_shape").iter().map(|&k| k as usize).collect();
    let auto_pad: AutoPad = parse_enum(node, "auto_pad", "NOTSET")?;
    let pads = get_attr_ints(node, "pads");
    let strides = get_attr_ints(node, "strides");
    let dilations = get_attr_ints(node, "dilations");
    Ok(inp(inputs, 0)
        .avg_pool()
        .kernel_shape(&kernel)
        .auto_pad(auto_pad)
        .ceil_mode(get_attr_int(node, "ceil_mode", 0) == 1)
        .count_include_pad(get_attr_int(node, "count_include_pad", 0) == 1)
        .maybe_pads(non_empty_i64(&pads))
        .maybe_strides(non_empty_i64(&strides))
        .maybe_dilations(non_empty_i64(&dilations))
        .call()?)
}

pub(crate) fn op_max_pool(inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Vec<Tensor>> {
    let kernel: Vec<usize> = get_attr_ints(node, "kernel_shape").iter().map(|&k| k as usize).collect();
    let auto_pad: AutoPad = parse_enum(node, "auto_pad", "NOTSET")?;
    let pads = get_attr_ints(node, "pads");
    let strides = get_attr_ints(node, "strides");
    let dilations = get_attr_ints(node, "dilations");
    let (values, indices) = inp(inputs, 0)
        .max_pool()
        .kernel_shape(&kernel)
        .auto_pad(auto_pad)
        .ceil_mode(get_attr_int(node, "ceil_mode", 0) == 1)
        .storage_order(get_attr_int(node, "storage_order", 0) as usize)
        .maybe_pads(non_empty_i64(&pads))
        .maybe_strides(non_empty_i64(&strides))
        .maybe_dilations(non_empty_i64(&dilations))
        .call()?;
    Ok(vec![values, indices])
}

pub(crate) fn op_layer_norm(inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Vec<Tensor>> {
    let x = inp(inputs, 0);
    let scale = inp(inputs, 1);
    let bias = inputs.get(2).and_then(|o| o.as_ref());
    let axis = get_attr_int(node, "axis", -1) as isize;
    let epsilon = get_attr_float(node, "epsilon", 1e-5) as f64;
    let (mut output, mean, inv_std_dev) = x.layernorm_with_stats(axis, epsilon)?;
    output = output.try_mul(scale)?;
    if let Some(bias) = bias {
        output = output.try_add(bias)?;
    }
    Ok(vec![output, mean, inv_std_dev])
}

pub(crate) fn op_group_norm(inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Tensor> {
    let x = inp(inputs, 0);
    let scale = inp(inputs, 1);
    let bias = inp(inputs, 2);
    let num_groups = get_attr_int(node, "num_groups", 1) as usize;
    let epsilon = get_attr_float(node, "epsilon", 1e-5) as f64;
    Ok(x.group_norm(scale, bias, num_groups, epsilon)?)
}

pub(crate) fn op_instance_norm(inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Tensor> {
    let x = inp(inputs, 0);
    let scale = inp(inputs, 1);
    let bias = inp(inputs, 2);
    let epsilon = get_attr_float(node, "epsilon", 1e-5) as f64;
    let num_channels = x.shape()?[1].as_const().unwrap();
    Ok(x.group_norm(scale, bias, num_channels, epsilon)?)
}

pub(crate) fn op_resize(inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Tensor> {
    let antialias = get_attr_int(node, "antialias", 0);
    if antialias != 0 {
        return Err(Error::IrConstruction { details: "Resize: antialias != 0 is not supported".into() });
    }
    let x = inp(inputs, 0);
    let scales: Option<Vec<f64>> = inputs
        .get(2)
        .and_then(|o| o.as_ref())
        .filter(|t| t.numel().unwrap_or(0) > 0)
        .map(tensor_to_f64_vec)
        .transpose()?;
    let sizes: Option<Vec<usize>> = inputs
        .get(3)
        .and_then(|o| o.as_ref())
        .filter(|t| t.numel().unwrap_or(0) > 0)
        .map(|t| tensor_to_i64_vec(t).map(|v| v.iter().map(|&x| x as usize).collect()))
        .transpose()?;
    let mode: ResizeMode = parse_enum(node, "mode", "nearest")?;
    let coord_mode: CoordinateTransformMode = parse_enum(node, "coordinate_transformation_mode", "half_pixel")?;
    let nearest_mode: NearestMode = parse_enum(node, "nearest_mode", "round_prefer_floor")?;
    let cubic_coeff = get_attr_float(node, "cubic_coeff_a", -0.75) as f64;
    let exclude_outside = get_attr_int(node, "exclude_outside", 0) != 0;
    let policy: AspectRatioPolicy = parse_enum(node, "keep_aspect_ratio_policy", "stretch")?;
    let axes_attr = get_attr_ints(node, "axes");
    let axes: Option<Vec<usize>> = if axes_attr.is_empty() {
        None
    } else {
        let ndim = x.ndim()?;
        Some(axes_attr.iter().map(|&a| if a < 0 { (ndim as i64 + a) as usize } else { a as usize }).collect())
    };
    Ok(x.resize()
        .maybe_scales(scales.as_deref())
        .maybe_sizes(sizes.as_deref())
        .mode(mode)
        .coordinate_transformation_mode(coord_mode)
        .nearest_mode(nearest_mode)
        .cubic_coeff_a(cubic_coeff)
        .exclude_outside(exclude_outside)
        .keep_aspect_ratio_policy(policy)
        .maybe_axes(axes.as_deref())
        .call()?)
}

// =========================================================================
// DepthToSpace / SpaceToDepth
// =========================================================================

pub(crate) fn op_depth_to_space(inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Tensor> {
    let x = inp(inputs, 0);
    let bs = get_attr_int(node, "blocksize", 1) as usize;
    let mode = get_attr_string(node, "mode", "DCR");
    let shape = x.shape()?;
    let (b, c, h, w) = (
        shape[0].as_const().unwrap(),
        shape[1].as_const().unwrap(),
        shape[2].as_const().unwrap(),
        shape[3].as_const().unwrap(),
    );
    let c_out = c / (bs * bs);
    let result = if mode == "CRD" {
        // CRD: reshape [B, C', bs, bs, H, W] → permute [0,1,4,2,5,3]
        x.try_reshape(&[b as isize, c_out as isize, bs as isize, bs as isize, h as isize, w as isize])?
            .try_permute(&[0, 1, 4, 2, 5, 3])?
    } else {
        // DCR: reshape [B, bs, bs, C', H, W] → permute [0,3,4,1,5,2]
        x.try_reshape(&[b as isize, bs as isize, bs as isize, c_out as isize, h as isize, w as isize])?
            .try_permute(&[0, 3, 4, 1, 5, 2])?
    };
    Ok(result.try_reshape(&[b as isize, c_out as isize, (h * bs) as isize, (w * bs) as isize])?)
}

pub(crate) fn op_space_to_depth(inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Tensor> {
    let x = inp(inputs, 0);
    let bs = get_attr_int(node, "blocksize", 1) as usize;
    let shape = x.shape()?;
    let (b, c, h, w) = (
        shape[0].as_const().unwrap(),
        shape[1].as_const().unwrap(),
        shape[2].as_const().unwrap(),
        shape[3].as_const().unwrap(),
    );
    // reshape [B, C, H/bs, bs, W/bs, bs] → permute [0,3,5,1,2,4] → reshape [B, C*bs², H/bs, W/bs]
    Ok(x.try_reshape(&[b as isize, c as isize, (h / bs) as isize, bs as isize, (w / bs) as isize, bs as isize])?
        .try_permute(&[0, 3, 5, 1, 2, 4])?
        .try_reshape(&[b as isize, (c * bs * bs) as isize, (h / bs) as isize, (w / bs) as isize])?)
}

// =========================================================================
// LpNormalization
// =========================================================================

pub(crate) fn op_lp_norm(inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Tensor> {
    let x = inp(inputs, 0);
    let axis = get_attr_int(node, "axis", -1) as isize;
    let p = get_attr_int(node, "p", 2);
    let norm = match p {
        1 => x.try_abs()?.sum_with().axes(AxisSpec::Single(axis)).keepdim(true).call()?,
        _ => x.square()?.sum_with().axes(AxisSpec::Single(axis)).keepdim(true).call()?.try_sqrt()?,
    };
    // Avoid 0/0 → NaN: add smallest normal float so 0/ε = 0 (ONNX expects 0 when norm is 0)
    let eps = dtype_min_positive(x.uop().dtype());
    Ok(x.try_div(&norm.try_add(&Tensor::const_(eps, x.uop().dtype()))?)?)
}

// =========================================================================
// MeanVarianceNormalization
// =========================================================================

pub(crate) fn op_mean_variance_norm(inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Tensor> {
    let x = inp(inputs, 0);
    let axes_attr = get_attr_ints(node, "axes");
    let axes: Vec<isize> =
        if axes_attr.is_empty() { vec![0, 2, 3] } else { axes_attr.iter().map(|&a| a as isize).collect() };
    let axes_spec = AxisSpec::Multiple(axes);
    let mean = x.mean_with().axes(axes_spec.clone()).keepdim(true).call()?;
    let centered = x.try_sub(&mean)?;
    // Population std (correction=0): sqrt(mean((x-mean)²))
    let pop_std = centered.square()?.mean_with().axes(axes_spec).keepdim(true).call()?.try_sqrt()?;
    let eps = Tensor::const_(1e-9, x.uop().dtype());
    Ok(centered.try_div(&pop_std.try_add(&eps)?)?)
}

// =========================================================================
// LRN (Local Response Normalization)
// =========================================================================

pub(crate) fn op_lrn(inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Tensor> {
    let x = inp(inputs, 0);
    let alpha = get_attr_float(node, "alpha", 0.0001) as f64;
    let beta = get_attr_float(node, "beta", 0.75) as f64;
    let bias = get_attr_float(node, "bias", 1.0) as f64;
    let size = get_attr_int(node, "size", 1) as usize;
    let shape = x.shape()?;
    let (b, c, h, w) = (
        shape[0].as_const().unwrap(),
        shape[1].as_const().unwrap(),
        shape[2].as_const().unwrap(),
        shape[3].as_const().unwrap(),
    );

    // x² → reshape [B,1,C,H*W] → pad channel dim → avg_pool2d([size,1]) → reshape back
    let x_sq = x.square()?;
    let x_sq = x_sq.try_reshape(&[b as isize, 1, c as isize, (h * w) as isize])?;
    let pad_before = ((size - 1) / 2) as isize;
    let pad_after = (size / 2) as isize;
    // Pad the channel dim (dim 2): [(0,0), (0,0), (before,after), (0,0)]
    let x_sq = x_sq.try_pad(&[(0, 0), (0, 0), (pad_before, pad_after), (0, 0)])?;
    let pooled = x_sq.avg_pool2d().kernel_size(&[size, 1]).stride(&[1, 1]).call()?;
    let pooled = pooled.try_reshape(&[b as isize, c as isize, h as isize, w as isize])?;

    // x / (bias + alpha * pooled)^beta
    let dtype = x.uop().dtype();
    let scale = pooled
        .try_mul(&Tensor::const_(alpha, dtype.clone()))?
        .try_add(&Tensor::const_(bias, dtype.clone()))?
        .try_pow(&Tensor::const_(beta, dtype))?;
    Ok(x.try_div(&scale)?)
}

// =========================================================================
// NegativeLogLikelihoodLoss / SoftmaxCrossEntropyLoss
// =========================================================================

pub(crate) fn op_nll_loss(inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Vec<Tensor>> {
    let x = inp(inputs, 0);
    let target = inp(inputs, 1);
    let weight = inputs.get(2).and_then(|o| o.as_ref());
    let reduction: Reduction = parse_enum(node, "reduction", "mean")?;
    let ignore_index = get_attr(node, "ignore_index").map(|a| a.i);
    let loss = x.nll_loss(target, weight, ignore_index, reduction)?;
    Ok(vec![loss])
}

pub(crate) fn op_softmax_ce_loss(inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Vec<Tensor>> {
    let x = inp(inputs, 0);
    let target = inp(inputs, 1);
    let weight = inputs.get(2).and_then(|o| o.as_ref());
    let reduction: Reduction = parse_enum(node, "reduction", "mean")?;
    let ignore_index = get_attr(node, "ignore_index").map(|a| a.i);
    // log_softmax over class dim (axis 1) using keepdim reduce to avoid broadcast issues
    let max_val = x.max_with().axes(1isize).keepdim(true).call()?;
    let shifted = x.try_sub(&max_val)?;
    let exp_shifted = shifted.try_exp()?;
    let sum_exp = exp_shifted.sum_with().axes(1isize).keepdim(true).call()?;
    let log_probs = shifted.try_sub(&sum_exp.try_log()?)?;
    let loss = log_probs.nll_loss(target, weight, ignore_index, reduction)?;
    Ok(vec![loss, log_probs])
}

// =========================================================================
// AffineGrid
// =========================================================================

pub(crate) fn op_affine_grid(inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Tensor> {
    let theta = inp(inputs, 0); // [N, ndim, ndim+1]
    let size = tensor_to_i64_vec(inp(inputs, 1))?;
    let align_corners = get_attr_int(node, "align_corners", 0) == 1;

    let n = size[0] as usize;
    let ndim = size.len() - 2; // spatial dims

    // Generate per-dim grids
    let spatial_dims: Vec<usize> = size[2..].iter().map(|&s| s as usize).collect();
    let mut grids = Vec::with_capacity(ndim);
    for &dim_size in &spatial_dims {
        let g = if align_corners {
            Tensor::linspace(-1.0, 1.0, dim_size, DType::Float32)?
        } else {
            // half_pixel: (-1, 1) range adjusted
            let start = -1.0 + 1.0 / dim_size as f64;
            let end = 1.0 - 1.0 / dim_size as f64;
            Tensor::linspace(start, end, dim_size, DType::Float32)?
        };
        grids.push(g);
    }

    // Meshgrid
    let grid_refs: Vec<&Tensor> = grids.iter().collect();
    let mesh = Tensor::meshgrid(&grid_refs, MeshgridIndexing::Ij)?;

    // Stack in reverse order + ones for homogeneous coordinates
    let total_elements: usize = spatial_dims.iter().product();
    let flat_shape = [total_elements as isize];
    let mut components: Vec<Tensor> = Vec::with_capacity(ndim + 1);
    for g in mesh.iter().rev() {
        components.push(g.try_reshape(&flat_shape)?);
    }
    components.push(Tensor::full(&[total_elements], 1.0, DType::Float32)?);

    // Stack to [ndim+1, prod(spatial)] and transpose to [prod(spatial), ndim+1]
    let comp_refs: Vec<&Tensor> = components.iter().collect();
    let base_grid = Tensor::cat(&comp_refs, 0)?
        .try_reshape(&[(ndim + 1) as isize, total_elements as isize])?
        .try_transpose(0, 1)?;

    // Expand to [N, prod(spatial), ndim+1]
    let base_grid =
        base_grid.try_unsqueeze(0)?.try_expand(&[n as isize, total_elements as isize, (ndim + 1) as isize])?;

    // theta: [N, ndim, ndim+1] → transpose to [N, ndim+1, ndim]
    let theta_t = theta.try_transpose(1, 2)?;

    // base_grid @ theta^T → [N, prod(spatial), ndim]
    let output = base_grid.matmul(&theta_t)?;

    // Reshape to [N, *spatial_dims, ndim]
    let mut out_shape: Vec<isize> = vec![n as isize];
    out_shape.extend(spatial_dims.iter().map(|&d| d as isize));
    out_shape.push(ndim as isize);
    Ok(output.try_reshape(&out_shape)?)
}

// =========================================================================
// GridSample
// =========================================================================

pub(crate) fn op_grid_sample(inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Tensor> {
    let x = inp(inputs, 0);
    let grid = inp(inputs, 1);
    let mode: GridSampleMode = parse_enum(node, "mode", "linear")?;
    let padding_mode: GridSamplePaddingMode = parse_enum(node, "padding_mode", "zeros")?;
    let align_corners = get_attr_int(node, "align_corners", 0) == 1;
    Ok(x.grid_sample(grid, mode, padding_mode, align_corners)?)
}

// =========================================================================
// Dropout (version-dispatched)
// =========================================================================

pub(crate) fn op_dropout(inputs: &[Option<Tensor>], node: &NodeProto, opset_version: i64) -> Result<Vec<Tensor>> {
    let x = inp(inputs, 0);

    // Extract ratio (v7+: input[1], v6: attribute)
    let ratio = if opset_version >= 7 {
        inputs.get(1).and_then(|o| o.as_ref()).map(|t| tensor_to_f64_scalar(t).unwrap_or(0.5)).unwrap_or(0.5)
    } else {
        get_attr_float(node, "ratio", 0.5) as f64
    };

    // Determine training mode
    let training = if opset_version < 7 {
        // v6: is_test attribute (default 0 = training)
        get_attr_int(node, "is_test", 0) != 1
    } else {
        // v7+: training_mode input (index 2, default false)
        inputs.get(2).and_then(|o| o.as_ref()).map(|t| tensor_to_bool_scalar(t).unwrap_or(false)).unwrap_or(false)
    };

    let (output, mask) = x.dropout(ratio, training)?;
    Ok(vec![output, mask])
}
