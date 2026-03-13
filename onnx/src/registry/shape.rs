use morok_ir::SInt;
use morok_ir::shape::{align_shapes_left, broadcast_shape};
use morok_tensor::Tensor;

use crate::error::{Error, Result};

use super::*;

pub(crate) fn op_reshape(inputs: &[Option<Tensor>], attrs: &mut Attrs) -> Result<Tensor> {
    let allowzero = attrs.int("allowzero", 0);
    let data = inp(inputs, 0);

    let raw_shape: Vec<isize> = {
        let attr_shape = attrs.ints("shape");
        if !attr_shape.is_empty() {
            attr_shape.iter().map(|&d| d as isize).collect()
        } else if inputs.len() > 1 && inputs[1].is_some() {
            tensor_to_i64_vec(inp(inputs, 1))?.iter().map(|&v| v as isize).collect()
        } else {
            return Err(Error::IrConstruction { details: "Reshape requires shape attribute or input".to_string() });
        }
    };

    let shape =
        if allowzero == 0 {
            let data_shape = data.shape()?;
            raw_shape
                .iter()
                .enumerate()
                .map(|(i, &d)| {
                    if d == 0 {
                        data_shape.get(i).and_then(|s| s.as_const()).map(|v| v as isize).unwrap_or(d)
                    } else {
                        d
                    }
                })
                .collect()
        } else {
            raw_shape
        };

    Ok(data.try_reshape(&shape)?)
}

pub(crate) fn op_transpose(inputs: &[Option<Tensor>], attrs: &mut Attrs) -> Result<Tensor> {
    let perm = attrs.ints("perm");
    if perm.is_empty() {
        let ndim = inp(inputs, 0).ndim()?;
        let reversed: Vec<isize> = (0..ndim).rev().map(|i| i as isize).collect();
        Ok(inp(inputs, 0).try_permute(&reversed)?)
    } else {
        let perm: Vec<isize> = perm.iter().map(|&p| p as isize).collect();
        Ok(inp(inputs, 0).try_permute(&perm)?)
    }
}

pub(crate) fn op_squeeze(inputs: &[Option<Tensor>], attrs: &mut Attrs, opset: i64) -> Result<Tensor> {
    let axes: Vec<i64> = if opset >= 13 {
        inputs.get(1).and_then(|o| o.as_ref()).map(tensor_to_i64_vec).transpose()?.unwrap_or_default()
    } else {
        attrs.ints("axes")
    };
    if axes.is_empty() {
        return Ok(inp(inputs, 0).try_squeeze(None)?);
    }
    let mut sorted: Vec<isize> = axes.iter().map(|&a| a as isize).collect();
    sorted.sort_by(|a, b| b.cmp(a)); // descending to preserve indices
    sorted.iter().try_fold(inp(inputs, 0).clone(), |t, &ax| Ok(t.try_squeeze(Some(ax))?))
}

pub(crate) fn op_unsqueeze(inputs: &[Option<Tensor>], attrs: &mut Attrs, opset: i64) -> Result<Tensor> {
    let axes: Vec<i64> = if opset >= 13 {
        inputs
            .get(1)
            .and_then(|o| o.as_ref())
            .map(tensor_to_i64_vec)
            .transpose()?
            .ok_or_else(|| Error::IrConstruction { details: "Unsqueeze (opset>=13) requires axes input".into() })?
    } else {
        let axes = attrs.ints("axes");
        if axes.is_empty() {
            return Err(Error::IrConstruction { details: "Unsqueeze requires axes attribute".into() });
        }
        axes
    };
    let mut sorted: Vec<isize> = axes.iter().map(|&a| a as isize).collect();
    sorted.sort(); // ascending for unsqueeze
    sorted.iter().try_fold(inp(inputs, 0).clone(), |t, &ax| Ok(t.try_unsqueeze(ax)?))
}

pub(crate) fn op_flatten(inputs: &[Option<Tensor>], attrs: &mut Attrs) -> Result<Tensor> {
    let shape = inp(inputs, 0).shape()?;
    let ndim = shape.len() as i64;
    let axis_raw = attrs.int("axis", 1);
    let axis = (if axis_raw < 0 { ndim + axis_raw } else { axis_raw }) as usize;
    let pre = morok_ir::sint_prod(&shape[..axis]);
    let pre_val = pre
        .as_const()
        .map(|v| v as isize)
        .ok_or_else(|| Error::IrConstruction { details: "Flatten requires concrete pre-axis dimensions".into() })?;
    Ok(inp(inputs, 0).try_reshape(&[pre_val, -1])?)
}

pub(crate) fn op_expand(inputs: &[Option<Tensor>]) -> Result<Tensor> {
    let data = inp(inputs, 0);
    let target_i64 = tensor_to_i64_vec(inp(inputs, 1))?;
    let target: morok_ir::shape::Shape = target_i64.iter().map(|&v| SInt::from(v as usize)).collect();
    let data_shape = data.shape()?;
    let aligned = align_shapes_left(&[data_shape, target]);
    let result_shape = broadcast_shape(&aligned[0], &aligned[1])
        .map_err(|e| Error::IrConstruction { details: format!("Expand broadcast: {e}") })?;
    Ok(data.broadcast_to(&result_shape)?)
}

pub(crate) fn op_pad(inputs: &[Option<Tensor>], attrs: &mut Attrs) -> Result<Tensor> {
    use morok_tensor::nn::PadMode;
    use std::str::FromStr;

    let pads = tensor_to_i64_vec(inp(inputs, 1))?;
    let mode_str = attrs.string("mode", "constant");
    let data = inp(inputs, 0);
    let ndim = data.ndim()?;
    let num_axes = pads.len() / 2;

    let axes: Option<Vec<usize>> = inputs
        .get(3)
        .and_then(|o| o.as_ref())
        .map(tensor_to_i64_vec)
        .transpose()?
        .map(|v| v.iter().map(|&a| if a < 0 { (ndim as i64 + a) as usize } else { a as usize }).collect());

    let padding: Vec<(isize, isize)> = if let Some(axes) = axes {
        let mut full = vec![(0isize, 0isize); ndim];
        for (i, &ax) in axes.iter().enumerate() {
            full[ax] = (pads[i] as isize, pads[num_axes + i] as isize);
        }
        full
    } else {
        (0..num_axes).map(|i| (pads[i] as isize, pads[num_axes + i] as isize)).collect()
    };

    let mode = PadMode::from_str(&mode_str)
        .map_err(|_| Error::IrConstruction { details: format!("Pad mode '{mode_str}' not supported") })?;

    let pad_value: f64 = match inputs.get(2).and_then(|o| o.as_ref()) {
        Some(cv) => tensor_to_f64_scalar(cv)?,
        None => 0.0,
    };

    Ok(data.pad_with().padding(&padding).mode(mode).value(pad_value).call()?)
}

pub(crate) fn op_slice(inputs: &[Option<Tensor>]) -> Result<Tensor> {
    let data = inp(inputs, 0);
    let starts = tensor_to_i64_vec(inp(inputs, 1))?;
    let ends = tensor_to_i64_vec(inp(inputs, 2))?;
    let axes: Option<Vec<i64>> = inputs.get(3).and_then(|o| o.as_ref()).map(tensor_to_i64_vec).transpose()?;
    let steps: Option<Vec<i64>> = inputs.get(4).and_then(|o| o.as_ref()).map(tensor_to_i64_vec).transpose()?;
    Ok(data
        .slice_with()
        .starts(&starts)
        .ends(&ends)
        .maybe_axes(axes.as_deref())
        .maybe_steps(steps.as_deref())
        .call()?)
}

pub(crate) fn op_split(inputs: &[Option<Tensor>], attrs: &mut Attrs, opset_version: i64) -> Result<Vec<Tensor>> {
    let axis = attrs.int("axis", 0) as isize;
    let data = inp(inputs, 0);
    let shape = data.shape()?;
    let ndim = shape.len();
    let norm_axis = if axis < 0 { (ndim as isize + axis) as usize } else { axis as usize };
    let dim_size = shape[norm_axis].as_const().unwrap();

    // Opset ≤11: split is an attribute (INTS); opset 13+: split is an optional input tensor
    let split_sizes: Vec<usize> = if opset_version < 13 {
        let attr_split = attrs.ints("split");
        if attr_split.is_empty() {
            let n = attrs.output_count();
            (0..n).map(|i| dim_size / n + if i < dim_size % n { 1 } else { 0 }).collect()
        } else {
            attr_split.iter().map(|&v| v as usize).collect()
        }
    } else if let Some(split_tensor) = inputs.get(1).and_then(|o| o.as_ref()) {
        tensor_to_i64_vec(split_tensor)?.iter().map(|&v| v as usize).collect()
    } else {
        // Opset 18+: num_outputs attribute; opset 13: infer from output count
        let mut n = attrs.int("num_outputs", 0) as usize;
        if n == 0 {
            n = attrs.output_count();
        }
        if n == 0 {
            return Err(Error::IrConstruction {
                details: "Split requires either split input or num_outputs attribute".into(),
            });
        }
        (0..n).map(|i| dim_size / n + if i < dim_size % n { 1 } else { 0 }).collect()
    };

    Ok(data.split(&split_sizes, axis)?)
}
