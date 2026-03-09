use morok_dtype::ScalarDType;
use morok_tensor::Tensor;

use crate::error::{Error, Result};

use super::*;

pub(crate) fn op_constant(attrs: &mut Attrs) -> Result<Tensor> {
    if let Some(tensor_proto) = attrs.tensor("value") {
        return tensor_from_proto(tensor_proto);
    }
    if let Some(attr) = attrs.get("value_float") {
        return Ok(Tensor::const_(attr.f as f64, morok_dtype::DType::Scalar(ScalarDType::Float32)));
    }
    let float_values = attrs.floats("value_floats");
    if !float_values.is_empty() {
        return Ok(Tensor::from_slice(&float_values));
    }
    if let Some(attr) = attrs.get("value_int") {
        return Ok(Tensor::const_(attr.i, morok_dtype::DType::Scalar(ScalarDType::Int64)));
    }
    let int_values = attrs.ints("value_ints");
    if !int_values.is_empty() {
        return Ok(Tensor::from_slice(&int_values));
    }
    if let Some(attr) = attrs.get("value_string") {
        let _ = &attr.s; // acknowledged; string tensors not supported
        return Err(Error::IrConstruction { details: "Constant value_string not supported".to_string() });
    }
    Err(Error::IrConstruction { details: "Constant node has no supported value attribute".to_string() })
}
