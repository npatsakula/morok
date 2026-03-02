use morok_dtype::DType;
use morok_tensor::Tensor;
use morok_tensor::reduce::AxisSpec;

use crate::error::{Error, Result};
use crate::parser::onnx::{AttributeProto, GraphProto, NodeProto, TensorProto};

pub fn get_attr<'a>(node: &'a NodeProto, name: &str) -> Option<&'a AttributeProto> {
    node.attribute.iter().find(|a| a.name == name)
}

pub fn get_attr_int(node: &NodeProto, name: &str, default: i64) -> i64 {
    get_attr(node, name).map(|a| a.i).unwrap_or(default)
}

pub fn get_attr_float(node: &NodeProto, name: &str, default: f32) -> f32 {
    get_attr(node, name).map(|a| a.f).unwrap_or(default)
}

pub fn get_attr_bytes<'a>(node: &'a NodeProto, name: &str) -> Option<&'a [u8]> {
    get_attr(node, name).map(|a| a.s.as_slice())
}

pub fn get_attr_string(node: &NodeProto, name: &str, default: &str) -> String {
    get_attr_bytes(node, name).map(|b| String::from_utf8_lossy(b).into_owned()).unwrap_or_else(|| default.to_string())
}

pub fn get_attr_ints(node: &NodeProto, name: &str) -> Vec<i64> {
    get_attr(node, name).map(|a| a.ints.clone()).unwrap_or_default()
}

pub fn get_attr_floats(node: &NodeProto, name: &str) -> Vec<f32> {
    get_attr(node, name).map(|a| a.floats.clone()).unwrap_or_default()
}

pub fn get_attr_tensor<'a>(node: &'a NodeProto, name: &str) -> Option<&'a TensorProto> {
    get_attr(node, name).and_then(|a| a.t.as_ref())
}

#[allow(dead_code)]
pub fn get_attr_graph<'a>(node: &'a NodeProto, name: &str) -> Option<&'a GraphProto> {
    get_attr(node, name).and_then(|a| a.g.as_ref())
}

/// Extract a scalar bool from a tensor (for If condition fallback).
pub(crate) fn tensor_to_bool_scalar(t: &Tensor) -> Result<bool> {
    let arr = t
        .cast(DType::Bool)?
        .to_ndarray::<bool>()
        .map_err(|e| Error::IrConstruction { details: format!("tensor_to_bool_scalar: {e}") })?;
    arr.iter().next().copied().ok_or_else(|| Error::IrConstruction { details: "empty bool tensor".into() })
}

pub(crate) fn inp(inputs: &[Option<Tensor>], idx: usize) -> &Tensor {
    inputs[idx].as_ref().expect("missing required ONNX input")
}

pub(crate) fn non_empty_i64(v: &[i64]) -> Option<&[i64]> {
    if v.is_empty() { None } else { Some(v) }
}

/// Extract reduce axes and keepdims, opset-aware.
/// Opset >= 13: axes from input[1] tensor. Opset <= 12: axes from node attribute.
pub(crate) fn reduce_attrs(node: &NodeProto, inputs: &[Option<Tensor>], opset: i64) -> Result<(AxisSpec, bool)> {
    let keepdims = get_attr_int(node, "keepdims", 1) == 1;
    let noop_with_empty_axes = get_attr_int(node, "noop_with_empty_axes", 0) == 1;

    let axes: Vec<i64> = if opset >= 13 {
        inputs.get(1).and_then(|o| o.as_ref()).map(tensor_to_i64_vec).transpose()?.unwrap_or_default()
    } else {
        get_attr_ints(node, "axes")
    };

    let spec = if axes.is_empty() {
        if noop_with_empty_axes {
            return Ok((AxisSpec::Multiple(vec![]), keepdims));
        }
        AxisSpec::All
    } else {
        AxisSpec::Multiple(axes.iter().map(|&a| a as isize).collect())
    };
    Ok((spec, keepdims))
}

/// Extract a scalar f64 from a tensor (e.g. constant_value for Pad).
pub(crate) fn tensor_to_f64_scalar(t: &Tensor) -> Result<f64> {
    let arr = t
        .cast(DType::Float64)?
        .to_ndarray::<f64>()
        .map_err(|e| Error::IrConstruction { details: format!("tensor_to_f64_scalar: {e}") })?;
    arr.iter().next().copied().ok_or_else(|| Error::IrConstruction { details: "empty scalar tensor".into() })
}

/// Extract concrete i64 values from a tensor (shape/indices/pads inputs).
pub(crate) fn tensor_to_i64_vec(t: &Tensor) -> Result<Vec<i64>> {
    let arr = t
        .cast(DType::Int64)?
        .to_ndarray::<i64>()
        .map_err(|e| Error::IrConstruction { details: format!("tensor_to_i64_vec: {e}") })?;
    Ok(arr.iter().copied().collect())
}

/// Extract concrete f64 values from a tensor.
pub(crate) fn tensor_to_f64_vec(t: &Tensor) -> Result<Vec<f64>> {
    let arr = t
        .cast(DType::Float64)?
        .to_ndarray::<f64>()
        .map_err(|e| Error::IrConstruction { details: format!("tensor_to_f64_vec: {e}") })?;
    Ok(arr.iter().copied().collect())
}
