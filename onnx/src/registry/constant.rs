use morok_dtype::ScalarDType;
use morok_tensor::Tensor;

use crate::error::{Error, Result};
use crate::parser::onnx::NodeProto;

use super::*;

pub(crate) fn op_constant(node: &NodeProto) -> Result<Tensor> {
    if let Some(tensor_proto) = get_attr_tensor(node, "value") {
        return tensor_from_proto(tensor_proto);
    }
    if let Some(attr) = get_attr(node, "value_float") {
        return Ok(Tensor::const_(attr.f as f64, morok_dtype::DType::Scalar(ScalarDType::Float32)));
    }
    let float_values = get_attr_floats(node, "value_floats");
    if !float_values.is_empty() {
        return Ok(Tensor::from_slice(&float_values));
    }
    if let Some(attr) = get_attr(node, "value_int") {
        return Ok(Tensor::const_(attr.i, morok_dtype::DType::Scalar(ScalarDType::Int64)));
    }
    let int_values = get_attr_ints(node, "value_ints");
    if !int_values.is_empty() {
        return Ok(Tensor::from_slice(&int_values));
    }
    Err(Error::IrConstruction { details: "Constant node has no supported value attribute".to_string() })
}
