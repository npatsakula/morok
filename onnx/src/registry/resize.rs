//! ONNX `Resize` (and the deprecated `Upsample`).
//!
//! Every knob of the operator — `roi`, `scales`, `sizes`, the modes — is a host
//! constant at import time, so for `nearest` the whole coordinate transform is
//! folded on the host in `f64` and only a constant index vector per axis reaches
//! the device. That is what the ONNX reference does, and it is the only way to
//! stay exact on devices without `double` (Metal): demoted to `f32` the
//! transform evaluates `(3 + 0.5) / 3.5 - 0.5` as `0.50000006`, and
//! `round_prefer_floor` then picks the next pixel. Tinygrad, which runs the
//! transform on device in `f32`, has to exclude exactly those cases from its own
//! test suite.
//!
//! `linear`/`cubic` blend neighbours, so a sub-ulp coordinate error only
//! perturbs the weights; they keep using the device-side path.

use svod_tensor::Tensor;
use svod_tensor::nn::{AspectRatioPolicy, CoordinateTransformMode, NearestMode, ResizeMode};

use super::*;
use crate::error::{Error, Result};

pub(crate) fn op_resize(inputs: &[Option<Tensor>], attrs: &mut Attrs) -> Result<Tensor> {
    let antialias = attrs.int("antialias", 0) != 0;
    let x = inp(inputs, 0);
    let opt_input = |idx: usize| inputs.get(idx).and_then(|o| o.as_ref()).filter(|t| t.numel().unwrap_or(0) > 0);
    let roi: Option<Vec<f64>> = opt_input(1).map(tensor_to_f64_vec).transpose()?;
    let scales: Option<Vec<f64>> = opt_input(2).map(tensor_to_f64_vec).transpose()?;
    let sizes: Option<Vec<usize>> =
        opt_input(3).map(|t| tensor_to_i64_vec(t).map(|v| v.iter().map(|&s| s as usize).collect())).transpose()?;
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
        let ndim = x.ndim()? as i64;
        Some(axes_attr.iter().map(|&a| if a < 0 { (ndim + a) as usize } else { a as usize }).collect())
    };

    if mode == ResizeMode::Nearest {
        return resize_nearest(
            x,
            scales.as_deref(),
            sizes.as_deref(),
            axes.as_deref(),
            roi.as_deref(),
            coord_mode,
            nearest_mode,
            policy,
            extrapolation_value,
        );
    }

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

/// One resized axis: where it sits, how big it is, its scale and its ROI.
struct Axis {
    dim: usize,
    input_sz: usize,
    output_sz: usize,
    scale: f64,
    roi: (f64, f64),
}

/// `nearest` resize: one constant `index_select` per axis, indices folded on the host.
#[allow(clippy::too_many_arguments)]
fn resize_nearest(
    x: &Tensor,
    scales: Option<&[f64]>,
    sizes: Option<&[usize]>,
    axes: Option<&[usize]>,
    roi: Option<&[f64]>,
    coord_mode: CoordinateTransformMode,
    nearest_mode: NearestMode,
    policy: AspectRatioPolicy,
    extrapolation_value: f64,
) -> Result<Tensor> {
    let ndim = x.ndim()?;
    let dims: Vec<usize> = axes.map(|a| a.to_vec()).unwrap_or_else(|| (0..ndim).collect());
    let n = dims.len();
    // `scales`/`sizes` cover either the listed axes or the trailing dims.
    let scales = scales.map(|v| v[v.len().saturating_sub(n)..].to_vec());
    let sizes = sizes.map(|v| v[v.len().saturating_sub(n)..].to_vec());

    // A scale of exactly 1 is the identity, so those axes are dropped before any
    // extent is demanded: a symbolic batch dim then passes straight through.
    let resized: Vec<usize> = match (&sizes, &scales) {
        (Some(_), _) => (0..n).collect(),
        (None, Some(sc)) => (0..n).filter(|&i| sc[i] != 1.0).collect(),
        (None, None) => {
            return Err(Error::IrConstruction { details: "resize: either scales or sizes must be provided".into() });
        }
    };
    if resized.is_empty() {
        return Ok(x.clone());
    }

    let input_shape: Vec<usize> =
        resized.iter().map(|&i| x.dim_const(dims[i] as isize)).collect::<std::result::Result<_, _>>()?;

    // `sizes` wins over `scales` when both are present, per the ONNX spec.
    let (output_sizes, out_scales): (Vec<usize>, Vec<f64>) = match &sizes {
        Some(sz) => {
            let sz: Vec<usize> = resized.iter().map(|&i| sz[i]).collect();
            let ratios: Vec<f64> = sz.iter().zip(&input_shape).map(|(&s, &sh)| s as f64 / sh as f64).collect();
            match policy {
                AspectRatioPolicy::Stretch => (sz, ratios),
                // Both spatial extents shrink/grow by one common scale, then round.
                _ => {
                    let scale = match policy {
                        AspectRatioPolicy::NotLarger => ratios.into_iter().fold(f64::INFINITY, f64::min),
                        _ => ratios.into_iter().fold(f64::NEG_INFINITY, f64::max),
                    };
                    (input_shape.iter().map(|&sh| (scale * sh as f64 + 0.5) as usize).collect(), vec![scale; sz.len()])
                }
            }
        }
        None => {
            let sc: Vec<f64> = resized.iter().map(|&i| scales.as_ref().expect("scales present")[i]).collect();
            (sc.iter().zip(&input_shape).map(|(&s, &sh)| (s * sh as f64) as usize).collect(), sc)
        }
    };

    // ROI is laid out as [start.., end..] over the resized axes (or the full rank).
    let roi_pairs: Vec<(f64, f64)> = match roi {
        Some(roi) => {
            let half = roi.len() / 2;
            let (starts, ends) = (&roi[half - resized.len()..half], &roi[roi.len() - resized.len()..]);
            starts.iter().zip(ends).map(|(&s, &e)| (s, e)).collect()
        }
        None => vec![(0.0, 1.0); resized.len()],
    };

    let is_tf_crop = coord_mode == CoordinateTransformMode::TfCropAndResize;
    let mut out = x.clone();
    let mut masks: Vec<(usize, Vec<bool>)> = Vec::new();
    for (k, &i) in resized.iter().enumerate() {
        let axis = Axis {
            dim: dims[i],
            input_sz: input_shape[k],
            output_sz: output_sizes[k],
            scale: out_scales[k],
            roi: roi_pairs[k],
        };
        let (indices, valid) = nearest_indices(&axis, coord_mode, nearest_mode);
        if is_tf_crop && valid.iter().any(|&v| !v) {
            masks.push((axis.dim, valid));
        }
        if indices.len() == axis.input_sz && indices.iter().enumerate().all(|(o, &idx)| idx as usize == o) {
            continue;
        }
        out = out.index_select(axis.dim as isize, &Tensor::from_slice(&indices))?;
    }

    // Output positions whose source coordinate left the ROI read the
    // extrapolation value instead of the (clamped) edge pixel.
    if !masks.is_empty() {
        let out_shape = out.shape()?.to_vec();
        let extrap = Tensor::const_(extrapolation_value, out.dtype());
        let mut combined: Option<Tensor> = None;
        for (dim, valid) in masks {
            let mut mask_shape = vec![1isize; ndim];
            mask_shape[dim] = valid.len() as isize;
            let mask = Tensor::from_slice(&valid).try_reshape(&mask_shape)?.try_expand(out_shape.clone())?;
            combined = Some(match combined {
                Some(c) => c.try_bitand(&mask)?,
                None => mask,
            });
        }
        out = out.where_(&combined.expect("masks is non-empty"), &extrap)?;
    }
    Ok(out)
}

/// Source index per output position for one axis, plus whether that position's
/// coordinate fell inside the input (only meaningful for `tf_crop_and_resize`).
fn nearest_indices(
    axis: &Axis,
    coord_mode: CoordinateTransformMode,
    nearest_mode: NearestMode,
) -> (Vec<i32>, Vec<bool>) {
    let max = (axis.input_sz - 1) as f64;
    let (mut indices, mut valid) = (Vec::with_capacity(axis.output_sz), Vec::with_capacity(axis.output_sz));
    for o in 0..axis.output_sz {
        let coord = source_coordinate(o as f64, axis, coord_mode);
        valid.push((0.0..=max).contains(&coord));
        let coord = coord.clamp(0.0, max);
        let rounded = match nearest_mode {
            NearestMode::RoundPreferFloor => (coord - 0.5).ceil(),
            NearestMode::RoundPreferCeil => (coord + 0.5).floor(),
            NearestMode::Floor => coord.floor(),
            NearestMode::Ceil => coord.ceil(),
        };
        indices.push(rounded.clamp(0.0, max) as i32);
    }
    (indices, valid)
}

/// ONNX coordinate transformation: output position → input coordinate.
fn source_coordinate(o: f64, axis: &Axis, mode: CoordinateTransformMode) -> f64 {
    let Axis { input_sz, output_sz, scale, roi: (roi_start, roi_end), .. } = *axis;
    // The reference uses the fractional output width `scale * input_sz`, which
    // differs from the integer `output_sz` when the product is not whole.
    let output_width = scale * input_sz as f64;
    match mode {
        CoordinateTransformMode::HalfPixel => (o + 0.5) / scale - 0.5,
        CoordinateTransformMode::AlignCorners => {
            if output_width == 1.0 {
                0.0
            } else {
                o * ((input_sz as f64 - 1.0) / (output_width - 1.0))
            }
        }
        CoordinateTransformMode::Asymmetric => o / scale,
        CoordinateTransformMode::PytorchHalfPixel => {
            if output_width == 1.0 {
                0.0
            } else {
                (o + 0.5) / scale - 0.5
            }
        }
        CoordinateTransformMode::HalfPixelSymmetric => {
            let offset = (input_sz as f64 / 2.0) * (1.0 - output_sz as f64 / output_width);
            offset + ((o + 0.5) / scale) - 0.5
        }
        CoordinateTransformMode::TfCropAndResize => {
            let len = input_sz as f64 - 1.0;
            if output_width == 1.0 {
                (roi_end - roi_start) * len / 2.0 + roi_start * len
            } else {
                o * ((roi_end - roi_start) * len / (output_width - 1.0)) + roi_start * len
            }
        }
    }
}
