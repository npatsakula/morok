//! Padding helpers: flat-to-pair conversion, auto-pad, pool pad resolution.

use bon::bon;
use morok_ir::{ConstValue, UOp};

use super::{AutoPad, PadMode};
use crate::Tensor;

type Result<T> = crate::Result<T>;

/// Convert flat pads `[begin0, begin1, ..., end0, end1, ...]` to `[(begin0, end0), ...]`.
pub fn flat_pads_to_pairs(pads: &[i64]) -> Vec<(isize, isize)> {
    let n = pads.len() / 2;
    (0..n).map(|i| (pads[i] as isize, pads[i + n] as isize)).collect()
}

/// Split total padding per dimension into `[begin0, begin1, ..., end0, end1, ...]`
/// based on auto_pad mode (SAME_UPPER: more padding at end; SAME_LOWER: more at begin).
pub fn auto_pad_split(total_pads: &[isize], auto_pad: AutoPad) -> Vec<isize> {
    let first: Vec<isize> = if auto_pad == AutoPad::SameUpper {
        total_pads.iter().map(|&p| p / 2).collect()
    } else {
        total_pads.iter().map(|&p| p - p / 2).collect()
    };
    let mut result = first.clone();
    result.extend(total_pads.iter().zip(&first).map(|(p, f)| p - f));
    result
}

/// Resolve auto_pad + flat pads into `[(begin, end), ...]` pairs.
/// Handles VALID, NOTSET, SAME_UPPER, SAME_LOWER.
pub fn resolve_pool_pads(
    input_spatial: &[usize],
    pads: &[i64],
    kernel: &[usize],
    dilations: &[usize],
    strides: &[usize],
    auto_pad: AutoPad,
) -> Vec<(isize, isize)> {
    let n = kernel.len();
    match auto_pad {
        AutoPad::Valid => vec![(0, 0); n],
        AutoPad::NotSet => {
            if pads.is_empty() {
                vec![(0, 0); n]
            } else {
                flat_pads_to_pairs(pads)
            }
        }
        AutoPad::SameUpper | AutoPad::SameLower => {
            let total_pads: Vec<isize> = (0..n)
                .map(|i| {
                    let out_size = usize::div_ceil(input_spatial[i], strides[i]);
                    let eff_kernel = dilations[i] * (kernel[i] - 1) + 1;
                    let needed = (out_size - 1) * strides[i] + eff_kernel;
                    needed.saturating_sub(input_spatial[i]) as isize
                })
                .collect();
            let flat = auto_pad_split(&total_pads, auto_pad);
            let half = flat.len() / 2;
            (0..half).map(|i| (flat[i], flat[i + half])).collect()
        }
    }
}

#[bon]
impl Tensor {
    /// Pad with a custom fill value. Delegates to try_pad when value == 0.
    pub fn try_pad_value(&self, padding: &[(isize, isize)], value: f64) -> Result<Tensor> {
        if value == 0.0 {
            return self.try_pad(padding);
        }
        // Tinygrad approach: x.pad(0) + ones_pad.where(0, fill_value)
        // ADD-based avoids fragile nested WHERE conditions that can evaluate to -inf.
        let dtype = self.uop().dtype();
        let sdtype = dtype.scalar().expect("pad_value requires scalar dtype");
        let padded = self.try_pad(padding)?;
        let ones = Tensor::new(UOp::const_(dtype.clone(), ConstValue::one(sdtype)));
        let ones = ones.broadcast_to(&self.shape()?)?;
        let ones_padded = ones.try_pad(padding)?;
        let zero_cmp = Tensor::new(UOp::const_(dtype.clone(), ConstValue::zero(sdtype)));
        let mask = ones_padded.try_ne(&zero_cmp)?;
        let zero_val = Tensor::new(UOp::const_(dtype.clone(), ConstValue::zero(sdtype)));
        let fill_val = Tensor::new(UOp::const_(dtype, ConstValue::Float(value)));
        // mask ? zero : fill_value  →  data region gets 0, pad region gets fill_value
        let fill_term = zero_val.where_(&mask, &fill_val)?;
        padded.try_add(&fill_term)
    }

    /// Pad with mode and fill value options.
    ///
    /// # Examples
    /// ```ignore
    /// // Edge (replicate) padding
    /// tensor.pad_with(&[(1, 1)]).mode(PadMode::Replicate).call()?;
    ///
    /// // Reflect padding
    /// tensor.pad_with(&[(2, 2)]).mode(PadMode::Reflect).call()?;
    ///
    /// // Constant padding with custom value
    /// tensor.pad_with(&[(1, 1)]).value(-f64::INFINITY).call()?;
    ///
    /// // Circular (wrap) padding
    /// tensor.pad_with(&[(1, 1)]).mode(PadMode::Circular).call()?;
    /// ```
    #[builder]
    pub fn pad_with(
        &self,
        padding: &[(isize, isize)],
        #[builder(default)] mode: PadMode,
        #[builder(default)] value: f64,
    ) -> Result<Tensor> {
        match mode {
            PadMode::Constant => self.try_pad_value(padding, value),
            PadMode::Replicate => pad_replicate(self, padding),
            PadMode::Reflect => pad_reflect(self, padding),
            PadMode::Circular => pad_circular(self, padding),
        }
    }
}

/// Replicate (edge) padding: repeats boundary values.
///
/// For each padded dimension, extracts edge slices via shrink, replicates
/// via expand, then concatenates. Mirrors Tinygrad's `pad(mode="replicate")`.
fn pad_replicate(data: &Tensor, padding: &[(isize, isize)]) -> Result<Tensor> {
    let mut result = data.clone();
    for (d, &(pad_before, pad_after)) in padding.iter().enumerate() {
        if pad_before == 0 && pad_after == 0 {
            continue;
        }
        let shape = result.shape()?;
        let dim_size = shape[d].as_const().expect("replicate pad requires concrete dims") as isize;
        let mut parts: Vec<Tensor> = Vec::new();

        if pad_before > 0 {
            let mut shrink_ranges: Vec<(isize, isize)> =
                shape.iter().map(|s| (0, s.as_const().unwrap() as isize)).collect();
            shrink_ranges[d] = (0, 1);
            let edge = result.try_shrink(&shrink_ranges)?;
            let mut expand_shape: Vec<isize> =
                shape.iter().map(|s| s.as_const().unwrap() as isize).collect();
            expand_shape[d] = pad_before;
            parts.push(edge.try_expand(&expand_shape)?);
        }

        parts.push(result.clone());

        if pad_after > 0 {
            let mut shrink_ranges: Vec<(isize, isize)> =
                shape.iter().map(|s| (0, s.as_const().unwrap() as isize)).collect();
            shrink_ranges[d] = (dim_size - 1, dim_size);
            let edge = result.try_shrink(&shrink_ranges)?;
            let mut expand_shape: Vec<isize> =
                shape.iter().map(|s| s.as_const().unwrap() as isize).collect();
            expand_shape[d] = pad_after;
            parts.push(edge.try_expand(&expand_shape)?);
        }

        let refs: Vec<&Tensor> = parts.iter().collect();
        result = Tensor::cat(&refs, d as isize)?;
    }
    Ok(result)
}

/// Reflect padding: mirrors values without repeating the boundary.
///
/// For each padded dimension, extracts interior slices via shrink, flips them,
/// then concatenates. E.g. `[1,2,3]` pad(2,2) → `[3,2,1,2,3,2,1]`.
fn pad_reflect(data: &Tensor, padding: &[(isize, isize)]) -> Result<Tensor> {
    let mut result = data.clone();
    for (d, &(pad_before, pad_after)) in padding.iter().enumerate() {
        if pad_before == 0 && pad_after == 0 {
            continue;
        }
        let shape = result.shape()?;
        let dim_size = shape[d].as_const().expect("reflect pad requires concrete dims") as isize;
        let mut parts: Vec<Tensor> = Vec::new();

        if pad_before > 0 {
            let mut shrink_ranges: Vec<(isize, isize)> =
                shape.iter().map(|s| (0, s.as_const().unwrap() as isize)).collect();
            shrink_ranges[d] = (1, 1 + pad_before);
            let slice = result.try_shrink(&shrink_ranges)?;
            parts.push(slice.flip(&[d as isize])?);
        }

        parts.push(result.clone());

        if pad_after > 0 {
            let mut shrink_ranges: Vec<(isize, isize)> =
                shape.iter().map(|s| (0, s.as_const().unwrap() as isize)).collect();
            shrink_ranges[d] = (dim_size - 1 - pad_after, dim_size - 1);
            let slice = result.try_shrink(&shrink_ranges)?;
            parts.push(slice.flip(&[d as isize])?);
        }

        let refs: Vec<&Tensor> = parts.iter().collect();
        result = Tensor::cat(&refs, d as isize)?;
    }
    Ok(result)
}

/// Circular (wrap) padding: wraps values from the opposite end.
///
/// Uses repeat + shrink: tile the tensor up to 3x per padded dimension,
/// then shrink to extract the wrapped window. Mirrors Tinygrad's `pad(mode="circular")`.
fn pad_circular(data: &Tensor, padding: &[(isize, isize)]) -> Result<Tensor> {
    let shape = data.shape()?;
    let ndim = shape.len();
    let repeats: Vec<usize> = padding
        .iter()
        .map(|&(pb, pa)| 1 + usize::from(pb > 0) + usize::from(pa > 0))
        .collect();
    let repeated = data.repeat(&repeats)?;
    let rep_shape = repeated.shape()?;

    let shrink_ranges: Vec<(isize, isize)> = (0..ndim)
        .map(|d| {
            let (pb, _pa) = padding[d];
            let orig = shape[d].as_const().expect("circular pad requires concrete dims") as isize;
            let rep_dim = rep_shape[d].as_const().unwrap() as isize;
            let start = if pb == 0 { 0 } else { orig - pb };
            let end = if padding[d].1 == 0 { rep_dim } else { rep_dim - orig + padding[d].1 };
            (start, end)
        })
        .collect();
    repeated.try_shrink(&shrink_ranges)
}

/// Adjust padding for ceil_mode output sizes.
/// Per arXiv:1603.07285 section 5.1, relationship 15.
pub(super) fn apply_ceil_mode(
    padding: &[(isize, isize)],
    input_spatial: &[usize],
    kernel: &[usize],
    stride: &[usize],
    dilation: &[usize],
) -> Vec<(isize, isize)> {
    let n = kernel.len();
    let grouped: Vec<(isize, isize)> = padding.to_vec();
    let mut ceil_pads = grouped.clone();
    for i in 0..n {
        let padded = input_spatial[i] as isize + grouped[i].0 + grouped[i].1;
        let eff_k = (dilation[i] * (kernel[i] - 1) + 1) as isize;
        let s = stride[i] as isize;
        // Output with ceil: ceildiv(padded - eff_k, s) + 1
        let o_ceil = (padded - eff_k + s - 1) / s + 1;
        // Output without ceil: (padded - eff_k) / s + 1
        let o_floor = (padded - eff_k) / s + 1;
        if o_ceil > o_floor {
            let last_start = s * (o_ceil - 1);
            let extra = last_start + eff_k - padded;
            // Decrease when last window starts past real data + pad_before
            let correction = (last_start - (grouped[i].0 + input_spatial[i] as isize - 1)).max(0);
            ceil_pads[i].1 += extra - correction;
        }
    }
    ceil_pads
}
