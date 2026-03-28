use morok_dtype::DType;
use morok_tensor::Tensor;
use morok_tensor::indexing::ScatterReduction;

use crate::error::{Error, Result};

use super::*;

pub(crate) fn op_gather_elements(inputs: &[Option<Tensor>], attrs: &mut Attrs) -> Result<Tensor> {
    let x = inp(inputs, 0);
    let idx = inp(inputs, 1);
    let axis = attrs.int("axis", 0) as isize;
    let x_shape = x.shape()?;
    let ndim = x_shape.len();
    let norm_axis = if axis < 0 { (ndim as isize + axis) as usize } else { axis as usize };
    let dim_size = x_shape[norm_axis].as_const().unwrap() as i64;
    let normalized_idx = idx.normalize_negative_indices(dim_size)?;
    Ok(x.gather(axis, &normalized_idx)?)
}

pub(crate) fn op_trilu(inputs: &[Option<Tensor>], attrs: &mut Attrs) -> Result<Tensor> {
    let x = inp(inputs, 0);
    let k = inputs.get(1).and_then(|o| o.as_ref()).map(tensor_to_i64_vec).transpose()?.map(|v| v[0]).unwrap_or(0);
    let upper = attrs.int("upper", 1) == 1;
    Ok(if upper { x.triu(k)? } else { x.tril(k)? })
}

pub(crate) fn op_one_hot(inputs: &[Option<Tensor>], attrs: &mut Attrs) -> Result<Tensor> {
    let indices = inp(inputs, 0);
    let depth = tensor_to_i64_vec(inp(inputs, 1))?[0] as usize;
    let values = inp(inputs, 2);
    let axis = attrs.int("axis", -1) as isize;
    let norm_idx = indices.normalize_negative_indices(depth as i64)?;
    let norm_idx = norm_idx.cast(DType::Int32)?;
    let ndim = norm_idx.ndim()? + 1;
    let norm_axis = if axis < 0 { (ndim as isize + axis) as usize } else { axis as usize };
    let expanded = norm_idx.try_unsqueeze(norm_axis as isize)?;
    let mask = expanded.one_hot_along_dim(depth, norm_axis as isize)?;
    let on_val = values.try_shrink([(1, 2)])?;
    let off_val = values.try_shrink([(0, 1)])?;
    Ok(on_val.where_(&mask, &off_val)?)
}

pub(crate) fn op_cumsum(inputs: &[Option<Tensor>], attrs: &mut Attrs) -> Result<Tensor> {
    let x = inp(inputs, 0);
    let axis = tensor_to_i64_vec(inp(inputs, 1))?[0] as isize;
    let exclusive = attrs.int("exclusive", 0) == 1;
    let reverse = attrs.int("reverse", 0) == 1;
    Ok(x.cumsum_with().axis(axis).exclusive(exclusive).reverse(reverse).call()?)
}

pub(crate) fn op_cumprod(inputs: &[Option<Tensor>], attrs: &mut Attrs) -> Result<Tensor> {
    let x = inp(inputs, 0);
    let axis = tensor_to_i64_vec(inp(inputs, 1))?[0] as isize;
    let exclusive = attrs.int("exclusive", 0) == 1;
    let reverse = attrs.int("reverse", 0) == 1;
    Ok(x.cumprod_with().axis(axis).exclusive(exclusive).reverse(reverse).call()?)
}

pub(crate) fn op_scatter_elements(inputs: &[Option<Tensor>], attrs: &mut Attrs) -> Result<Tensor> {
    let axis = attrs.int("axis", 0) as isize;
    let reduction = attrs.string("reduction", "none");
    let x = inp(inputs, 0);
    let idx = inp(inputs, 1);
    let updates = inp(inputs, 2);
    let x_shape = x.shape()?;
    let ndim = x_shape.len();
    let norm_axis = if axis < 0 { (ndim as isize + axis) as usize } else { axis as usize };
    let dim_size = x_shape[norm_axis].as_const().unwrap() as i64;
    let norm_idx = idx.normalize_negative_indices(dim_size)?;
    Ok(match reduction.as_str() {
        "none" => x.scatter(axis, &norm_idx, updates)?,
        other => {
            let reduce = match other {
                "add" => ScatterReduction::Sum,
                "mul" => ScatterReduction::Prod,
                "min" => ScatterReduction::Amin,
                "max" => ScatterReduction::Amax,
                _ => {
                    return Err(Error::IrConstruction {
                        details: format!("ScatterElements: unsupported reduction '{other}'"),
                    });
                }
            };
            x.scatter_reduce(axis, &norm_idx, updates, reduce, true)?
        }
    })
}

pub(crate) fn op_scatter_nd(inputs: &[Option<Tensor>], attrs: &mut Attrs) -> Result<Tensor> {
    let x = inp(inputs, 0);
    let indices = inp(inputs, 1);
    let updates = inp(inputs, 2);
    let reduction = attrs.string("reduction", "none");
    Ok(x.scatter_nd(indices, updates, &reduction)?)
}

pub(crate) fn op_tensor_scatter(inputs: &[Option<Tensor>], attrs: &mut Attrs) -> Result<Tensor> {
    let data = inp(inputs, 0);
    let update = inp(inputs, 1);
    let write_indices = inputs.get(2).and_then(|o| o.as_ref());
    let mode = attrs.string("mode", "linear");
    let axis = attrs.int("axis", -2) as isize;
    Ok(data.tensor_scatter(update, write_indices, &mode, axis)?)
}

pub(crate) fn op_gather_nd(inputs: &[Option<Tensor>], attrs: &mut Attrs) -> Result<Tensor> {
    let batch_dims = attrs.int("batch_dims", 0) as usize;
    Ok(inp(inputs, 0).gather_nd(inp(inputs, 1), batch_dims)?)
}
