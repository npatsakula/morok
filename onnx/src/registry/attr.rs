use std::collections::HashMap;

use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::reduce::AxisSpec;

use crate::error::{Error, Result};
use crate::parser::onnx::{AttributeProto, GraphProto, NodeProto, TensorProto};

/// Pop-based attribute extractor that ensures all ONNX node attributes are consumed.
///
/// Each accessor removes the attribute from the internal map. After all attributes
/// have been extracted, call [`done()`](Attrs::done) to verify none were missed.
pub struct Attrs<'a> {
    attrs: HashMap<&'a str, &'a AttributeProto>,
    op_type: &'a str,
    output_count: usize,
}

impl<'a> Attrs<'a> {
    pub fn new(node: &'a NodeProto) -> Self {
        let attrs = node.attribute.iter().map(|a| (a.name.as_str(), a)).collect();
        Self { attrs, op_type: &node.op_type, output_count: node.output.len() }
    }

    /// Number of outputs declared by the ONNX node.
    pub fn output_count(&self) -> usize {
        self.output_count
    }

    /// Pop a raw attribute by name.
    pub fn get(&mut self, name: &str) -> Option<&'a AttributeProto> {
        self.attrs.remove(name)
    }

    /// Pop an integer attribute, returning `default` if absent.
    pub fn int(&mut self, name: &str, default: i64) -> i64 {
        self.get(name).map(|a| a.i).unwrap_or(default)
    }

    /// Pop a float attribute, returning `default` if absent.
    pub fn float(&mut self, name: &str, default: f32) -> f32 {
        self.get(name).map(|a| a.f).unwrap_or(default)
    }

    /// Pop a string attribute, returning `default` if absent.
    pub fn string(&mut self, name: &str, default: &str) -> String {
        self.get(name).map(|a| String::from_utf8_lossy(&a.s).into_owned()).unwrap_or_else(|| default.to_string())
    }

    /// Pop a bytes attribute.
    #[allow(dead_code)]
    pub fn bytes(&mut self, name: &str) -> Option<&'a [u8]> {
        self.get(name).map(|a| a.s.as_slice())
    }

    /// Pop an integer array attribute.
    pub fn ints(&mut self, name: &str) -> Vec<i64> {
        self.get(name).map(|a| a.ints.clone()).unwrap_or_default()
    }

    /// Pop a float array attribute.
    pub fn floats(&mut self, name: &str) -> Vec<f32> {
        self.get(name).map(|a| a.floats.clone()).unwrap_or_default()
    }

    /// Pop a tensor attribute.
    pub fn tensor(&mut self, name: &str) -> Option<&'a TensorProto> {
        self.get(name).and_then(|a| a.t.as_ref())
    }

    /// Pop a graph attribute.
    #[allow(dead_code)]
    pub fn graph(&mut self, name: &str) -> Option<&'a GraphProto> {
        self.get(name).and_then(|a| a.g.as_ref())
    }

    /// Assert all attributes have been consumed. Returns error listing any remaining.
    pub fn done(self) -> Result<()> {
        if self.attrs.is_empty() {
            return Ok(());
        }
        let mut names: Vec<String> = self.attrs.keys().map(|s| s.to_string()).collect();
        names.sort();
        Err(Error::UnhandledAttributes { op: self.op_type.to_string(), attrs: names })
    }
}

/// Parse a string attribute into a typed enum.
pub fn parse_enum<T: std::str::FromStr>(attrs: &mut Attrs, name: &str, default: &str) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    let s = attrs.string(name, default);
    s.parse::<T>().map_err(|e| Error::IrConstruction { details: format!("{name}='{s}': {e}") })
}

pub(crate) fn inp(inputs: &[Option<Tensor>], idx: usize) -> &Tensor {
    inputs[idx].as_ref().expect("missing required ONNX input")
}

pub(crate) fn non_empty_i64(v: &[i64]) -> Option<&[i64]> {
    if v.is_empty() { None } else { Some(v) }
}

/// Extract reduce axes and keepdims, opset-aware.
/// Opset >= 13: axes from input[1] tensor. Opset <= 12: axes from node attribute.
pub(crate) fn reduce_attrs(
    attrs: &mut Attrs,
    inputs: &[Option<Tensor>],
    opset: i64,
    op_type: &str,
) -> Result<(AxisSpec, bool)> {
    let keepdims = attrs.int("keepdims", 1) == 1;
    let noop_with_empty_axes = attrs.int("noop_with_empty_axes", 0) == 1;

    // ReduceSum moved axes from attribute to input[1] at opset 13;
    // all other reduce ops moved at opset 18.
    let axes_from_input_version = if op_type == "ReduceSum" { 13 } else { 18 };

    let axes: Vec<i64> = if opset >= axes_from_input_version {
        inputs.get(1).and_then(|o| o.as_ref()).map(tensor_to_i64_vec).transpose()?.unwrap_or_default()
    } else {
        attrs.ints("axes")
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

/// Extract a scalar bool from a tensor (for If condition fallback).
pub(crate) fn tensor_to_bool_scalar(t: &Tensor) -> Result<bool> {
    let casted = t.cast(DType::Bool);
    casted.realize().map_err(|e| Error::IrConstruction { details: format!("tensor_to_bool_scalar: {e}") })?;
    let vals = casted
        .as_vec::<bool>()
        .map_err(|e| Error::IrConstruction { details: format!("tensor_to_bool_scalar: {e}") })?;
    vals.into_iter().next().ok_or_else(|| Error::IrConstruction { details: "empty bool tensor".into() })
}

/// Extract a scalar f64 from a tensor (e.g. constant_value for Pad).
pub(crate) fn tensor_to_f64_scalar(t: &Tensor) -> Result<f64> {
    host_f64_values(t, "tensor_to_f64_scalar")?
        .into_iter()
        .next()
        .ok_or_else(|| Error::IrConstruction { details: "empty scalar tensor".into() })
}

/// Read a tensor's values as f64 on the host. The device computes in the
/// tensor's own dtype family (f32 for floats, i64 for integers) and the widening
/// happens here, so no f64 kernel is required of devices without `double`.
fn host_f64_values(t: &Tensor, what: &str) -> Result<Vec<f64>> {
    let failed = |e: svod_tensor::error::Error| Error::IrConstruction { details: format!("{what}: {e}") };
    let dtype = t.dtype();
    if dtype.base() == svod_dtype::ScalarDType::Float64 {
        let realized = t.clone();
        realized.realize().map_err(failed)?;
        return realized.as_vec::<f64>().map_err(failed);
    }
    let via = if dtype.is_float() { DType::Float32 } else { DType::Int64 };
    let casted = t.cast(via.clone());
    casted.realize().map_err(failed)?;
    if via == DType::Float32 {
        Ok(casted.as_vec::<f32>().map_err(failed)?.into_iter().map(f64::from).collect())
    } else {
        Ok(casted.as_vec::<i64>().map_err(failed)?.into_iter().map(|v| v as f64).collect())
    }
}

/// Extract concrete i64 values from a tensor (shape/indices/pads inputs).
pub(crate) fn tensor_to_i64_vec(t: &Tensor) -> Result<Vec<i64>> {
    let casted = t.cast(DType::Int64);
    casted.realize().map_err(|e| Error::IrConstruction { details: format!("tensor_to_i64_vec: {e}") })?;
    casted.as_vec::<i64>().map_err(|e| Error::IrConstruction { details: format!("tensor_to_i64_vec: {e}") })
}

/// Extract concrete f64 values from a tensor.
pub(crate) fn tensor_to_f64_vec(t: &Tensor) -> Result<Vec<f64>> {
    host_f64_values(t, "tensor_to_f64_vec")
}
