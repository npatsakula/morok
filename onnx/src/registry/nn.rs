use morok_dtype::DType;
use morok_tensor::Tensor;
use morok_tensor::nn::{
    AspectRatioPolicy, AutoPad, CoordinateTransformMode, DepthToSpaceMode, GridSampleMode, GridSamplePaddingMode,
    NearestMode, Reduction, ResizeMode, flat_pads_to_pairs,
};
use morok_tensor::reduce::AxisSpec;

use crate::error::Result;

use super::*;

pub(crate) fn op_gemm(inputs: &[Option<Tensor>], attrs: &mut Attrs) -> Result<Tensor> {
    let alpha = attrs.float("alpha", 1.0);
    let beta = attrs.float("beta", 1.0);
    let trans_a = attrs.int("transA", 0) == 1;
    let trans_b = attrs.int("transB", 0) == 1;
    let c = inputs.get(2).and_then(|o| o.as_ref());
    Ok(inp(inputs, 0)
        .gemm()
        .b(inp(inputs, 1))
        .alpha(alpha)
        .beta(beta)
        .trans_a(trans_a)
        .trans_b(trans_b)
        .maybe_c(c)
        .call()?)
}

pub(crate) fn op_batch_norm(inputs: &[Option<Tensor>], attrs: &mut Attrs) -> Result<Vec<Tensor>> {
    let (x, scale, bias, running_mean, running_var) =
        (inp(inputs, 0), inp(inputs, 1), inp(inputs, 2), inp(inputs, 3), inp(inputs, 4));
    let epsilon = attrs.float("epsilon", 1e-5);
    let training_mode = attrs.int("training_mode", 0) == 1;

    if training_mode {
        let momentum = attrs.float("momentum", 0.9) as f64;
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
        let invstd = batch_var.try_add(&Tensor::const_(epsilon as f64, batch_var.uop().dtype()))?.try_rsqrt()?;
        let out = x.batchnorm().scale(scale).bias(bias).mean(&batch_mean).invstd(&invstd).call()?;
        let out = out.cast(x.uop().dtype())?;
        Ok(vec![out, new_running_mean, new_running_var])
    } else {
        let invstd = running_var.try_add(&Tensor::const_(epsilon as f64, running_var.uop().dtype()))?.try_rsqrt()?;
        let out = x.batchnorm().scale(scale).bias(bias).mean(running_mean).invstd(&invstd).call()?;
        Ok(vec![out])
    }
}

pub(crate) fn op_conv(inputs: &[Option<Tensor>], attrs: &mut Attrs) -> Result<Tensor> {
    let ks: Vec<usize> = attrs.ints("kernel_shape").iter().map(|&k| k as usize).collect();
    let auto_pad: AutoPad = parse_enum(attrs, "auto_pad", "NOTSET")?;
    let pads = attrs.ints("pads");
    let strides = attrs.ints("strides");
    let dilations = attrs.ints("dilations");
    Ok(inp(inputs, 0)
        .conv()
        .weight(inp(inputs, 1))
        .maybe_bias(inputs.get(2).and_then(|o| o.as_ref()))
        .auto_pad(auto_pad)
        .group(attrs.int("group", 1) as usize)
        .maybe_kernel_shape((!ks.is_empty()).then_some(ks.as_slice()))
        .maybe_pads(non_empty_i64(&pads))
        .maybe_strides(non_empty_i64(&strides))
        .maybe_dilations(non_empty_i64(&dilations))
        .call()?)
}

pub(crate) fn op_qlinear_conv(inputs: &[Option<Tensor>], attrs: &mut Attrs) -> Result<Tensor> {
    let ks: Vec<usize> = attrs.ints("kernel_shape").iter().map(|&k| k as usize).collect();
    let auto_pad: AutoPad = parse_enum(attrs, "auto_pad", "NOTSET")?;
    let pads = attrs.ints("pads");
    let strides = attrs.ints("strides");
    let dilations = attrs.ints("dilations");
    // QLinearConv inputs: x(0), x_scale(1), x_zp(2), w(3), w_scale(4), w_zp(5), y_scale(6), y_zp(7), B(8)?
    Ok(inp(inputs, 0)
        .qlinear_conv()
        .x_scale(inp(inputs, 1))
        .x_zero_point(inp(inputs, 2))
        .weight(inp(inputs, 3))
        .w_scale(inp(inputs, 4))
        .w_zero_point(inp(inputs, 5))
        .y_scale(inp(inputs, 6))
        .y_zero_point(inp(inputs, 7))
        .maybe_bias(inputs.get(8).and_then(|o| o.as_ref()))
        .auto_pad(auto_pad)
        .group(attrs.int("group", 1) as usize)
        .maybe_kernel_shape((!ks.is_empty()).then_some(ks.as_slice()))
        .maybe_pads(non_empty_i64(&pads))
        .maybe_strides(non_empty_i64(&strides))
        .maybe_dilations(non_empty_i64(&dilations))
        .call()?)
}

pub(crate) fn op_qlinear_matmul(inputs: &[Option<Tensor>], _attrs: &mut Attrs) -> Result<Tensor> {
    // QLinearMatMul inputs: a(0), a_scale(1), a_zp(2), b(3), b_scale(4), b_zp(5), y_scale(6), y_zp(7)
    Ok(inp(inputs, 0)
        .qlinear_matmul()
        .a_scale(inp(inputs, 1))
        .a_zero_point(inp(inputs, 2))
        .b(inp(inputs, 3))
        .b_scale(inp(inputs, 4))
        .b_zero_point(inp(inputs, 5))
        .y_scale(inp(inputs, 6))
        .y_zero_point(inp(inputs, 7))
        .call()?)
}

pub(crate) fn op_conv_integer(inputs: &[Option<Tensor>], attrs: &mut Attrs) -> Result<Tensor> {
    let ks: Vec<usize> = attrs.ints("kernel_shape").iter().map(|&k| k as usize).collect();
    let auto_pad: AutoPad = parse_enum(attrs, "auto_pad", "NOTSET")?;
    let pads = attrs.ints("pads");
    let strides = attrs.ints("strides");
    let dilations = attrs.ints("dilations");
    // ConvInteger inputs: x(0), w(1), x_zero_point(2)?, w_zero_point(3)?
    Ok(inp(inputs, 0)
        .conv_integer()
        .weight(inp(inputs, 1))
        .maybe_x_zero_point(inputs.get(2).and_then(|o| o.as_ref()))
        .maybe_w_zero_point(inputs.get(3).and_then(|o| o.as_ref()))
        .auto_pad(auto_pad)
        .group(attrs.int("group", 1) as usize)
        .maybe_kernel_shape((!ks.is_empty()).then_some(ks.as_slice()))
        .maybe_pads(non_empty_i64(&pads))
        .maybe_strides(non_empty_i64(&strides))
        .maybe_dilations(non_empty_i64(&dilations))
        .call()?)
}

pub(crate) fn op_conv_transpose(inputs: &[Option<Tensor>], attrs: &mut Attrs) -> Result<Tensor> {
    let ks: Vec<usize> = attrs.ints("kernel_shape").iter().map(|&k| k as usize).collect();
    let op: Vec<usize> = attrs.ints("output_padding").iter().map(|&p| p as usize).collect();
    let os = attrs.ints("output_shape");
    let auto_pad: AutoPad = parse_enum(attrs, "auto_pad", "NOTSET")?;
    let pads = attrs.ints("pads");
    let strides = attrs.ints("strides");
    let dilations = attrs.ints("dilations");
    Ok(inp(inputs, 0)
        .conv_transpose()
        .weight(inp(inputs, 1))
        .maybe_bias(inputs.get(2).and_then(|o| o.as_ref()))
        .auto_pad(auto_pad)
        .group(attrs.int("group", 1) as usize)
        .maybe_kernel_shape((!ks.is_empty()).then_some(ks.as_slice()))
        .maybe_pads(non_empty_i64(&pads))
        .maybe_output_shape(non_empty_i64(&os))
        .maybe_output_padding((!op.is_empty()).then_some(op.as_slice()))
        .maybe_strides(non_empty_i64(&strides))
        .maybe_dilations(non_empty_i64(&dilations))
        .call()?)
}

pub(crate) fn op_avg_pool(inputs: &[Option<Tensor>], attrs: &mut Attrs) -> Result<Tensor> {
    let kernel: Vec<usize> = attrs.ints("kernel_shape").iter().map(|&k| k as usize).collect();
    let auto_pad: AutoPad = parse_enum(attrs, "auto_pad", "NOTSET")?;
    let pads = attrs.ints("pads");
    let strides = attrs.ints("strides");
    let dilations = attrs.ints("dilations");
    Ok(inp(inputs, 0)
        .avg_pool()
        .kernel_shape(&kernel)
        .auto_pad(auto_pad)
        .ceil_mode(attrs.int("ceil_mode", 0) == 1)
        .count_include_pad(attrs.int("count_include_pad", 0) == 1)
        .maybe_pads(non_empty_i64(&pads))
        .maybe_strides(non_empty_i64(&strides))
        .maybe_dilations(non_empty_i64(&dilations))
        .call()?)
}

pub(crate) fn op_lp_pool(inputs: &[Option<Tensor>], attrs: &mut Attrs) -> Result<Tensor> {
    let kernel: Vec<usize> = attrs.ints("kernel_shape").iter().map(|&k| k as usize).collect();
    let auto_pad: AutoPad = parse_enum(attrs, "auto_pad", "NOTSET")?;
    let pads = attrs.ints("pads");
    let strides = attrs.ints("strides");
    let dilations = attrs.ints("dilations");
    Ok(inp(inputs, 0)
        .lp_pool()
        .kernel_shape(&kernel)
        .p(attrs.int("p", 2) as usize)
        .auto_pad(auto_pad)
        .ceil_mode(attrs.int("ceil_mode", 0) == 1)
        .maybe_pads(non_empty_i64(&pads))
        .maybe_strides(non_empty_i64(&strides))
        .maybe_dilations(non_empty_i64(&dilations))
        .call()?)
}

pub(crate) fn op_max_pool(inputs: &[Option<Tensor>], attrs: &mut Attrs) -> Result<Vec<Tensor>> {
    let kernel: Vec<usize> = attrs.ints("kernel_shape").iter().map(|&k| k as usize).collect();
    let auto_pad: AutoPad = parse_enum(attrs, "auto_pad", "NOTSET")?;
    let pads = attrs.ints("pads");
    let strides = attrs.ints("strides");
    let dilations = attrs.ints("dilations");
    let (values, indices) = inp(inputs, 0)
        .max_pool()
        .kernel_shape(&kernel)
        .auto_pad(auto_pad)
        .ceil_mode(attrs.int("ceil_mode", 0) == 1)
        .storage_order(attrs.int("storage_order", 0) as usize)
        .maybe_pads(non_empty_i64(&pads))
        .maybe_strides(non_empty_i64(&strides))
        .maybe_dilations(non_empty_i64(&dilations))
        .call()?;
    Ok(vec![values, indices])
}

pub(crate) fn op_col2im(inputs: &[Option<Tensor>], attrs: &mut Attrs) -> Result<Tensor> {
    let image_shape: Vec<usize> = tensor_to_i64_vec(inp(inputs, 1))?.iter().map(|&v| v as usize).collect();
    let block_shape: Vec<usize> = tensor_to_i64_vec(inp(inputs, 2))?.iter().map(|&v| v as usize).collect();
    let n_spatial = image_shape.len();
    let strides_raw = attrs.ints("strides");
    let strides: Vec<usize> =
        if strides_raw.is_empty() { vec![1; n_spatial] } else { strides_raw.iter().map(|&s| s as usize).collect() };
    let dilations_raw = attrs.ints("dilations");
    let dilations: Vec<usize> =
        if dilations_raw.is_empty() { vec![1; n_spatial] } else { dilations_raw.iter().map(|&d| d as usize).collect() };
    let pads_raw = attrs.ints("pads");
    let pads: Vec<(isize, isize)> =
        if pads_raw.is_empty() { vec![(0, 0); n_spatial] } else { flat_pads_to_pairs(&pads_raw) };
    Ok(inp(inputs, 0)
        .col2im()
        .image_shape(&image_shape)
        .block_shape(&block_shape)
        .strides(&strides)
        .pads(&pads)
        .dilations(&dilations)
        .call()?)
}

pub(crate) fn op_max_unpool(inputs: &[Option<Tensor>], attrs: &mut Attrs) -> Result<Tensor> {
    let x = inp(inputs, 0);
    let indices = inp(inputs, 1);
    let kernel: Vec<usize> = attrs.ints("kernel_shape").iter().map(|&k| k as usize).collect();
    let strides = attrs.ints("strides");
    let pads = attrs.ints("pads");
    let n_spatial = kernel.len();
    let stride_u: Vec<usize> =
        if strides.is_empty() { kernel.clone() } else { strides.iter().map(|&s| s as usize).collect() };
    let padding: Vec<(isize, isize)> =
        if pads.is_empty() { vec![(0, 0); n_spatial] } else { flat_pads_to_pairs(&pads) };
    let output_shape: Option<Vec<usize>> = inputs
        .get(2)
        .and_then(|o| o.as_ref())
        .map(|t| tensor_to_i64_vec(t).map(|v| v.iter().map(|&x| x as usize).collect()))
        .transpose()?;
    Ok(x.max_unpool2d()
        .indices(indices)
        .kernel_size(&kernel)
        .stride(&stride_u)
        .padding(&padding)
        .maybe_output_size(output_shape.as_deref())
        .call()?)
}

pub(crate) fn op_layer_norm(inputs: &[Option<Tensor>], attrs: &mut Attrs) -> Result<Vec<Tensor>> {
    let x = inp(inputs, 0);
    let scale = inp(inputs, 1);
    let bias = inputs.get(2).and_then(|o| o.as_ref());
    let axis = attrs.int("axis", -1) as isize;
    let epsilon = attrs.float("epsilon", 1e-5) as f64;
    let (mut output, mean, inv_std_dev) = x.layernorm_with_stats(axis, epsilon)?;
    output = output.try_mul(scale)?;
    if let Some(bias) = bias {
        output = output.try_add(bias)?;
    }
    Ok(vec![output, mean, inv_std_dev])
}

pub(crate) fn op_group_norm(inputs: &[Option<Tensor>], attrs: &mut Attrs) -> Result<Tensor> {
    let x = inp(inputs, 0);
    let scale = inp(inputs, 1);
    let bias = inp(inputs, 2);
    let num_groups = attrs.int("num_groups", 1) as usize;
    let epsilon = attrs.float("epsilon", 1e-5) as f64;
    Ok(x.group_norm().scale(scale).bias(bias).num_groups(num_groups).eps(epsilon).call()?)
}

pub(crate) fn op_instance_norm(inputs: &[Option<Tensor>], attrs: &mut Attrs) -> Result<Tensor> {
    let x = inp(inputs, 0);
    let scale = inp(inputs, 1);
    let bias = inp(inputs, 2);
    let epsilon = attrs.float("epsilon", 1e-5) as f64;
    let num_channels = x.shape()?[1].as_const().unwrap();
    Ok(x.group_norm().scale(scale).bias(bias).num_groups(num_channels).eps(epsilon).call()?)
}

pub(crate) fn op_resize(inputs: &[Option<Tensor>], attrs: &mut Attrs) -> Result<Tensor> {
    let antialias = attrs.int("antialias", 0) != 0;
    let x = inp(inputs, 0);
    let roi: Option<Vec<f64>> = inputs
        .get(1)
        .and_then(|o| o.as_ref())
        .filter(|t| t.numel().unwrap_or(0) > 0)
        .map(tensor_to_f64_vec)
        .transpose()?;
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
    let mode: ResizeMode = parse_enum(attrs, "mode", "nearest")?;
    let coord_mode: CoordinateTransformMode = parse_enum(attrs, "coordinate_transformation_mode", "half_pixel")?;
    let nearest_mode: NearestMode = parse_enum(attrs, "nearest_mode", "round_prefer_floor")?;
    let cubic_coeff = attrs.float("cubic_coeff_a", -0.75) as f64;
    let exclude_outside = attrs.int("exclude_outside", 0) != 0;
    let extrapolation_value = attrs.float("extrapolation_value", 0.0) as f64;
    let policy: AspectRatioPolicy = parse_enum(attrs, "keep_aspect_ratio_policy", "stretch")?;
    let axes_attr = attrs.ints("axes");
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
        .antialias(antialias)
        .extrapolation_value(extrapolation_value)
        .keep_aspect_ratio_policy(policy)
        .maybe_axes(axes.as_deref())
        .maybe_roi(roi.as_deref())
        .call()?)
}

// =========================================================================
// DepthToSpace / SpaceToDepth
// =========================================================================

pub(crate) fn op_depth_to_space(inputs: &[Option<Tensor>], attrs: &mut Attrs) -> Result<Tensor> {
    let bs = attrs.int("blocksize", 1) as usize;
    let mode: DepthToSpaceMode = parse_enum(attrs, "mode", "DCR")?;
    Ok(inp(inputs, 0).depth_to_space().blocksize(bs).mode(mode).call()?)
}

pub(crate) fn op_space_to_depth(inputs: &[Option<Tensor>], attrs: &mut Attrs) -> Result<Tensor> {
    let bs = attrs.int("blocksize", 1) as usize;
    Ok(inp(inputs, 0).space_to_depth(bs)?)
}

// =========================================================================
// LpNormalization
// =========================================================================

pub(crate) fn op_lp_norm(inputs: &[Option<Tensor>], attrs: &mut Attrs) -> Result<Tensor> {
    let x = inp(inputs, 0);
    let axis = attrs.int("axis", -1) as isize;
    let p = attrs.int("p", 2);
    Ok(x.lp_normalize(axis, p)?)
}

// =========================================================================
// MeanVarianceNormalization
// =========================================================================

pub(crate) fn op_mean_variance_norm(inputs: &[Option<Tensor>], attrs: &mut Attrs) -> Result<Tensor> {
    let x = inp(inputs, 0);
    let axes_attr = attrs.ints("axes");
    let axes: Vec<isize> =
        if axes_attr.is_empty() { vec![0, 2, 3] } else { axes_attr.iter().map(|&a| a as isize).collect() };
    Ok(x.mean_variance_normalize(&axes, 1e-9)?)
}

// =========================================================================
// LRN (Local Response Normalization)
// =========================================================================

pub(crate) fn op_lrn(inputs: &[Option<Tensor>], attrs: &mut Attrs) -> Result<Tensor> {
    let x = inp(inputs, 0);
    let alpha = attrs.float("alpha", 0.0001) as f64;
    let beta = attrs.float("beta", 0.75) as f64;
    let bias = attrs.float("bias", 1.0) as f64;
    let size = attrs.int("size", 1) as usize;
    Ok(x.lrn().size(size).alpha(alpha).beta(beta).bias(bias).call()?)
}

// =========================================================================
// NegativeLogLikelihoodLoss / SoftmaxCrossEntropyLoss
// =========================================================================

pub(crate) fn op_nll_loss(inputs: &[Option<Tensor>], attrs: &mut Attrs) -> Result<Vec<Tensor>> {
    let x = inp(inputs, 0);
    let target = inp(inputs, 1);
    let weight = inputs.get(2).and_then(|o| o.as_ref());
    let reduction: Reduction = parse_enum(attrs, "reduction", "mean")?;
    let ignore_index = attrs.get("ignore_index").map(|a| a.i);
    let loss = x
        .nll_loss()
        .target(target)
        .maybe_weight(weight)
        .maybe_ignore_index(ignore_index)
        .reduction(reduction)
        .call()?;
    Ok(vec![loss])
}

pub(crate) fn op_softmax_ce_loss(inputs: &[Option<Tensor>], attrs: &mut Attrs) -> Result<Vec<Tensor>> {
    let x = inp(inputs, 0);
    let target = inp(inputs, 1);
    let weight = inputs.get(2).and_then(|o| o.as_ref());
    let reduction: Reduction = parse_enum(attrs, "reduction", "mean")?;
    let ignore_index = attrs.get("ignore_index").map(|a| a.i);
    let log_probs = x.log_softmax(1isize)?;
    let loss = log_probs
        .nll_loss()
        .target(target)
        .maybe_weight(weight)
        .maybe_ignore_index(ignore_index)
        .reduction(reduction)
        .call()?;
    Ok(vec![loss, log_probs])
}

// =========================================================================
// AffineGrid
// =========================================================================

pub(crate) fn op_affine_grid(inputs: &[Option<Tensor>], attrs: &mut Attrs) -> Result<Tensor> {
    let theta = inp(inputs, 0);
    let size = tensor_to_i64_vec(inp(inputs, 1))?;
    let align_corners = attrs.int("align_corners", 0) == 1;
    Ok(Tensor::affine_grid().theta(theta).size(&size).align_corners(align_corners).call()?)
}

// =========================================================================
// GridSample
// =========================================================================

pub(crate) fn op_grid_sample(inputs: &[Option<Tensor>], attrs: &mut Attrs) -> Result<Tensor> {
    let x = inp(inputs, 0);
    let grid = inp(inputs, 1);
    let mode: GridSampleMode = parse_enum(attrs, "mode", "linear")?;
    let padding_mode: GridSamplePaddingMode = parse_enum(attrs, "padding_mode", "zeros")?;
    let align_corners = attrs.int("align_corners", 0) == 1;
    Ok(x.grid_sample().grid(grid).mode(mode).padding_mode(padding_mode).align_corners(align_corners).call()?)
}

// =========================================================================
// Dropout (version-dispatched)
// =========================================================================

pub(crate) fn op_dropout(inputs: &[Option<Tensor>], attrs: &mut Attrs, opset_version: i64) -> Result<Vec<Tensor>> {
    let x = inp(inputs, 0);
    let _seed = attrs.int("seed", 0); // acknowledged; not used in inference

    // Extract ratio: attribute through opset 10, input[1] from opset 12+
    let ratio = if opset_version >= 12 {
        inputs.get(1).and_then(|o| o.as_ref()).map(|t| tensor_to_f64_scalar(t).unwrap_or(0.5)).unwrap_or(0.5)
    } else {
        attrs.float("ratio", 0.5) as f64
    };

    // Determine training mode: is_test attribute through opset 6, input[2] from opset 12+
    let training = if opset_version >= 12 {
        inputs.get(2).and_then(|o| o.as_ref()).map(|t| tensor_to_bool_scalar(t).unwrap_or(false)).unwrap_or(false)
    } else if opset_version <= 6 {
        attrs.int("is_test", 0) != 1
    } else {
        // opset 7-10: no is_test, no training_mode input; always inference
        false
    };

    let (output, mask) = x.dropout().p(ratio).training(training).call()?;
    Ok(vec![output, mask])
}
