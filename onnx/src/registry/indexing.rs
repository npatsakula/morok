use morok_dtype::DType;
use morok_ir::ConstValue;
use morok_tensor::Tensor;
use morok_tensor::indexing::ScatterReduction;

use crate::error::{Error, Result};
use crate::parser::onnx::NodeProto;

use super::*;

pub(crate) fn op_gather_elements(inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Tensor> {
    let x = inp(inputs, 0);
    let idx = inp(inputs, 1);
    let axis = get_attr_int(node, "axis", 0) as isize;
    let x_shape = x.shape()?;
    let ndim = x_shape.len();
    let norm_axis = if axis < 0 { (ndim as isize + axis) as usize } else { axis as usize };
    let dim_size = x_shape[norm_axis].as_const().unwrap() as i64;
    let zero = Tensor::const_(ConstValue::Int(0), idx.uop().dtype());
    let dim_t = Tensor::const_(ConstValue::Int(dim_size), idx.uop().dtype());
    let neg_mask = idx.try_lt(&zero)?;
    let normalized_idx = idx.try_add(&dim_t)?.where_(&neg_mask, idx)?;
    Ok(x.gather(axis, &normalized_idx)?)
}

pub(crate) fn op_trilu(inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Tensor> {
    let x = inp(inputs, 0);
    let k = inputs.get(1).and_then(|o| o.as_ref()).map(tensor_to_i64_vec).transpose()?.map(|v| v[0]).unwrap_or(0);
    let upper = get_attr_int(node, "upper", 1) == 1;
    Ok(if upper { x.triu(k)? } else { x.tril(k)? })
}

pub(crate) fn op_one_hot(inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Tensor> {
    let indices = inp(inputs, 0);
    let depth = tensor_to_i64_vec(inp(inputs, 1))?[0] as usize;
    let values = inp(inputs, 2);
    let axis = get_attr_int(node, "axis", -1) as isize;
    let zero = Tensor::const_(ConstValue::Int(0), indices.uop().dtype());
    let depth_t = Tensor::const_(ConstValue::Int(depth as i64), indices.uop().dtype());
    let neg_mask = indices.try_lt(&zero)?;
    let norm_idx = indices.try_add(&depth_t)?.where_(&neg_mask, indices)?;
    let norm_idx = norm_idx.cast(DType::Int32)?;
    let ndim = norm_idx.ndim()? + 1;
    let norm_axis = if axis < 0 { (ndim as isize + axis) as usize } else { axis as usize };
    let expanded = norm_idx.try_unsqueeze(norm_axis as isize)?;
    let mask = expanded.one_hot_along_dim(depth, norm_axis as isize)?;
    let on_val = values.try_shrink(&[(1, 2)])?;
    let off_val = values.try_shrink(&[(0, 1)])?;
    Ok(on_val.where_(&mask, &off_val)?)
}

pub(crate) fn op_cumsum(inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Tensor> {
    let x = inp(inputs, 0);
    let axis_raw = tensor_to_i64_vec(inp(inputs, 1))?[0];
    let ndim = x.ndim()?;
    let axis = if axis_raw < 0 { (ndim as i64 + axis_raw) as usize } else { axis_raw as usize };
    let exclusive = get_attr_int(node, "exclusive", 0) == 1;
    let reverse = get_attr_int(node, "reverse", 0) == 1;
    let mut result = x.clone();
    if reverse {
        result = result.flip(&[axis as isize])?;
    }
    if exclusive {
        let shape = result.shape()?;
        let dim_size = shape[axis].as_const().unwrap() as isize;
        let mut pad_spec: Vec<(isize, isize)> = vec![(0, 0); ndim];
        pad_spec[axis] = (1, 0);
        result = result.try_pad(&pad_spec)?;
        let mut shrink_spec: Vec<(isize, isize)> =
            result.shape()?.iter().map(|s| (0, s.as_const().unwrap() as isize)).collect();
        shrink_spec[axis] = (0, dim_size);
        result = result.try_shrink(&shrink_spec)?;
    }
    result = result.cumsum(axis as isize)?;
    if reverse {
        result = result.flip(&[axis as isize])?;
    }
    Ok(result)
}

pub(crate) fn op_scatter_elements(inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Tensor> {
    let axis = get_attr_int(node, "axis", 0) as isize;
    let reduction = get_attr_string(node, "reduction", "none");
    let x = inp(inputs, 0);
    let idx = inp(inputs, 1);
    let updates = inp(inputs, 2);
    let x_shape = x.shape()?;
    let ndim = x_shape.len();
    let norm_axis = if axis < 0 { (ndim as isize + axis) as usize } else { axis as usize };
    let dim_size = x_shape[norm_axis].as_const().unwrap() as i64;
    let zero = Tensor::const_(ConstValue::Int(0), idx.uop().dtype());
    let dim_t = Tensor::const_(ConstValue::Int(dim_size), idx.uop().dtype());
    let neg_mask = idx.try_lt(&zero)?;
    let norm_idx = idx.try_add(&dim_t)?.where_(&neg_mask, idx)?;
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

pub(crate) fn op_scatter_nd(inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Tensor> {
    let mut x = inp(inputs, 0).clone();
    let indices = inp(inputs, 1);
    let updates = inp(inputs, 2);
    let reduction = get_attr_string(node, "reduction", "none");
    let x_shape = x.shape()?;
    let x_dims = morok_ir::shape::to_vec_usize(&x_shape)?;
    let idx_shape = indices.shape()?;
    let last_idx_dim = idx_shape[idx_shape.len() - 1].as_const().unwrap();
    let strides: Vec<i64> =
        (0..last_idx_dim).map(|k| x_dims[k + 1..last_idx_dim].iter().product::<usize>() as i64).collect();
    let x_numel: usize = x_dims.iter().product();
    let inner: usize = x_dims[last_idx_dim..].iter().product();
    let outer = x_numel / inner;
    let x_flat = x.try_reshape(&[outer as isize, inner as isize])?;
    let idx_splits: Vec<Tensor> = (0..last_idx_dim)
        .map(|k| {
            let mut ranges: Vec<(isize, isize)> =
                idx_shape.iter().map(|s| (0, s.as_const().unwrap() as isize)).collect();
            ranges[idx_shape.len() - 1] = (k as isize, k as isize + 1);
            let slice = indices.try_shrink(&ranges)?;
            slice.try_squeeze(Some(-1))
        })
        .collect::<std::result::Result<_, _>>()?;
    let mut flat_idx = Tensor::const_(ConstValue::Int(0), DType::Int64);
    for (k, idx_k) in idx_splits.iter().enumerate() {
        let stride_t = Tensor::const_(ConstValue::Int(strides[k]), DType::Int64);
        flat_idx = flat_idx.try_add(&idx_k.cast(DType::Int64)?.try_mul(&stride_t)?)?;
    }
    let upd_shape = updates.shape()?;
    let upd_outer: usize =
        upd_shape[..upd_shape.len() - (x_dims.len() - last_idx_dim)].iter().map(|s| s.as_const().unwrap()).product();
    let upd_flat = updates.try_reshape(&[upd_outer as isize, inner as isize])?;
    let flat_idx = flat_idx.try_reshape(&[upd_outer as isize, 1])?.try_expand(&[upd_outer as isize, inner as isize])?;
    let flat_idx_i32 = flat_idx.cast(DType::Int32)?;
    x = match reduction.as_str() {
        "none" => x_flat.scatter(0, &flat_idx_i32, &upd_flat)?,
        "add" => x_flat.scatter_reduce(0, &flat_idx_i32, &upd_flat, ScatterReduction::Sum, true)?,
        "mul" => x_flat.scatter_reduce(0, &flat_idx_i32, &upd_flat, ScatterReduction::Prod, true)?,
        "max" => x_flat.scatter_reduce(0, &flat_idx_i32, &upd_flat, ScatterReduction::Amax, true)?,
        "min" => x_flat.scatter_reduce(0, &flat_idx_i32, &upd_flat, ScatterReduction::Amin, true)?,
        _ => {
            return Err(Error::IrConstruction { details: format!("ScatterND: unsupported reduction '{reduction}'") });
        }
    };
    let out_shape: Vec<isize> = x_dims.iter().map(|&d| d as isize).collect();
    Ok(x.try_reshape(&out_shape)?)
}

pub(crate) fn op_gather_nd(inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Tensor> {
    let x = inp(inputs, 0);
    let indices = inp(inputs, 1);
    let batch_dims = get_attr_int(node, "batch_dims", 0) as usize;
    let x_shape = x.shape()?;
    let x_dims = morok_ir::shape::to_vec_usize(&x_shape)?;
    let idx_shape = indices.shape()?;
    let idx_dims = morok_ir::shape::to_vec_usize(&idx_shape)?;
    let last_idx_dim = *idx_dims.last().unwrap();

    if batch_dims == 0 {
        let strides: Vec<i64> =
            (0..last_idx_dim).map(|k| x_dims[k + 1..last_idx_dim].iter().product::<usize>() as i64).collect();
        let inner: usize = x_dims[last_idx_dim..].iter().product();
        let outer = x_dims[..last_idx_dim].iter().product::<usize>();

        let mut flat_idx = Tensor::const_(ConstValue::Int(0), DType::Int64);
        for (k, stride) in strides.iter().enumerate() {
            let mut ranges: Vec<(isize, isize)> = idx_dims.iter().map(|&s| (0, s as isize)).collect();
            ranges[idx_dims.len() - 1] = (k as isize, k as isize + 1);
            let idx_k = indices.try_shrink(&ranges)?.try_squeeze(Some(-1))?;
            let stride_t = Tensor::const_(ConstValue::Int(*stride), DType::Int64);
            flat_idx = flat_idx.try_add(&idx_k.cast(DType::Int64)?.try_mul(&stride_t)?)?;
        }

        let x_flat = x.try_reshape(&[outer as isize, inner as isize])?;
        let gather_outer: Vec<isize> = idx_dims[..idx_dims.len() - 1].iter().map(|&d| d as isize).collect();
        let num_gathers: usize = gather_outer.iter().map(|&d| d as usize).product();

        let flat_idx_2d = flat_idx
            .try_reshape(&[num_gathers as isize, 1])?
            .try_expand(&[num_gathers as isize, inner as isize])?
            .cast(DType::Int32)?;
        let result = x_flat.gather(0, &flat_idx_2d)?;

        let mut out_shape = gather_outer;
        for &d in &x_dims[last_idx_dim..] {
            out_shape.push(d as isize);
        }
        Ok(result.try_reshape(&out_shape)?)
    } else {
        let batch_size: usize = x_dims[..batch_dims].iter().product();
        let inner_x: Vec<usize> = x_dims[batch_dims..].to_vec();
        let inner_idx: Vec<usize> = idx_dims[batch_dims..].to_vec();

        let x_flat = x.try_reshape(
            &std::iter::once(batch_size as isize).chain(inner_x.iter().map(|&d| d as isize)).collect::<Vec<_>>(),
        )?;
        let idx_flat = indices.try_reshape(
            &std::iter::once(batch_size as isize).chain(inner_idx.iter().map(|&d| d as isize)).collect::<Vec<_>>(),
        )?;

        let last_inner = *inner_idx.last().unwrap();
        let strides: Vec<i64> =
            (0..last_inner).map(|k| inner_x[k + 1..last_inner].iter().product::<usize>() as i64).collect();

        let mut flat_idx = Tensor::const_(ConstValue::Int(0), DType::Int64);
        let idx_flat_shape = idx_flat.shape()?;
        let idx_flat_dims = morok_ir::shape::to_vec_usize(&idx_flat_shape)?;
        for (k, stride) in strides.iter().enumerate() {
            let mut ranges: Vec<(isize, isize)> = idx_flat_dims.iter().map(|&s| (0, s as isize)).collect();
            ranges[idx_flat_dims.len() - 1] = (k as isize, k as isize + 1);
            let idx_k = idx_flat.try_shrink(&ranges)?.try_squeeze(Some(-1))?;
            let stride_t = Tensor::const_(ConstValue::Int(*stride), DType::Int64);
            flat_idx = flat_idx.try_add(&idx_k.cast(DType::Int64)?.try_mul(&stride_t)?)?;
        }

        let batch_stride = inner_x[..last_inner].iter().product::<usize>();
        let batch_offset_arr =
            Tensor::arange(0, Some(batch_size as i64), None)?.try_mul(&Tensor::from_slice([batch_stride as i64]))?;
        let gather_inner = idx_flat_dims[1..idx_flat_dims.len() - 1].iter().product::<usize>();
        // Ensure flat_idx has shape [batch_size, gather_inner] before adding batch_offset,
        // otherwise broadcasting produces incorrect shapes when gather_inner == 1.
        flat_idx = flat_idx.try_reshape(&[batch_size as isize, gather_inner as isize])?;
        let batch_offset = batch_offset_arr
            .try_reshape(&[batch_size as isize, 1])?
            .try_expand(&[batch_size as isize, gather_inner as isize])?;
        flat_idx = flat_idx.try_add(&batch_offset)?;

        let remaining: usize = inner_x[last_inner..].iter().product();
        let x_2d = x_flat.try_reshape(&[(batch_size * batch_stride) as isize, remaining as isize])?;
        let fi = flat_idx
            .try_reshape(&[(batch_size * gather_inner) as isize, 1])?
            .try_expand(&[(batch_size * gather_inner) as isize, remaining as isize])?
            .cast(DType::Int32)?;
        let result = x_2d.gather(0, &fi)?;

        let mut out_shape: Vec<isize> = x_dims[..batch_dims].iter().map(|&d| d as isize).collect();
        out_shape.extend(inner_idx[..inner_idx.len() - 1].iter().map(|&d| d as isize));
        out_shape.extend(inner_x[last_inner..].iter().map(|&d| d as isize));
        Ok(result.try_reshape(&out_shape)?)
    }
}
