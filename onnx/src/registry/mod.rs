//! ONNX operator registry - maps ONNX ops to Morok Tensor operations.

use morok_dtype::{DType, ScalarDType};
use morok_tensor::Tensor;
use morok_tensor::reduce::AxisSpec;

use crate::error::{Error, Result, UnsupportedDTypeSnafu, UnsupportedOpSnafu};
use crate::parser::onnx::{AttributeProto, NodeProto, TensorProto, tensor_proto};

/// Convert ONNX DataType to Morok DType.
pub fn convert_onnx_dtype(onnx_dtype: i32) -> Result<DType> {
    use tensor_proto::DataType;

    let scalar = match DataType::try_from(onnx_dtype) {
        Ok(DataType::Float) => ScalarDType::Float32,
        Ok(DataType::Uint8) => ScalarDType::UInt8,
        Ok(DataType::Int8) => ScalarDType::Int8,
        Ok(DataType::Uint16) => ScalarDType::UInt16,
        Ok(DataType::Int16) => ScalarDType::Int16,
        Ok(DataType::Int32) => ScalarDType::Int32,
        Ok(DataType::Int64) => ScalarDType::Int64,
        Ok(DataType::Bool) => ScalarDType::Bool,
        Ok(DataType::Float16) => ScalarDType::Float16,
        Ok(DataType::Double) => ScalarDType::Float64,
        Ok(DataType::Uint32) => ScalarDType::UInt32,
        Ok(DataType::Uint64) => ScalarDType::UInt64,
        Ok(DataType::Bfloat16) => ScalarDType::BFloat16,
        Ok(DataType::Float8e4m3fn) | Ok(DataType::Float8e4m3fnuz) => ScalarDType::FP8E4M3,
        Ok(DataType::Float8e5m2) | Ok(DataType::Float8e5m2fnuz) => ScalarDType::FP8E5M2,
        _ => return UnsupportedDTypeSnafu { dtype: onnx_dtype }.fail(),
    };

    Ok(DType::Scalar(scalar))
}

/// Create a Tensor from ONNX TensorProto (weights/initializers).
pub fn tensor_from_proto(tensor: &TensorProto) -> Result<Tensor> {
    let dtype = convert_onnx_dtype(tensor.data_type)?;
    let dims: Vec<usize> = tensor.dims.iter().map(|&d| d as usize).collect();
    let raw_data = extract_tensor_data(tensor)?;

    create_tensor_from_raw(&raw_data, &dims, dtype)
}

/// Create Tensor from raw bytes, shape, and dtype.
fn create_tensor_from_raw(data: &[u8], dims: &[usize], dtype: DType) -> Result<Tensor> {
    let num_elements: usize = dims.iter().product();

    // Create tensor based on dtype
    let tensor = match dtype.base() {
        ScalarDType::Float32 => {
            let values: Vec<f32> = bytemuck::cast_slice(data).to_vec();
            if values.len() != num_elements {
                return Err(Error::ShapeMismatch {
                    context: "tensor_from_proto".to_string(),
                    expected: num_elements.to_string(),
                    actual: values.len().to_string(),
                });
            }
            let t = Tensor::from_slice(&values);
            // Convert usize to isize for reshape
            let shape: Vec<isize> = dims.iter().map(|&d| d as isize).collect();
            t.try_reshape(&shape)
        }
        ScalarDType::Float64 => {
            let values: Vec<f64> = bytemuck::cast_slice(data).to_vec();
            let t = Tensor::from_slice(&values);
            let shape: Vec<isize> = dims.iter().map(|&d| d as isize).collect();
            t.try_reshape(&shape)
        }
        ScalarDType::Int32 => {
            let values: Vec<i32> = bytemuck::cast_slice(data).to_vec();
            let t = Tensor::from_slice(&values);
            let shape: Vec<isize> = dims.iter().map(|&d| d as isize).collect();
            t.try_reshape(&shape)
        }
        ScalarDType::Int64 => {
            let values: Vec<i64> = bytemuck::cast_slice(data).to_vec();
            let t = Tensor::from_slice(&values);
            let shape: Vec<isize> = dims.iter().map(|&d| d as isize).collect();
            t.try_reshape(&shape)
        }
        ScalarDType::Bool => {
            // ONNX stores bools as int32
            let int_values: Vec<i32> = bytemuck::cast_slice(data).to_vec();
            let bool_values: Vec<bool> = int_values.iter().map(|&v| v != 0).collect();
            let t = Tensor::from_slice(&bool_values);
            let shape: Vec<isize> = dims.iter().map(|&d| d as isize).collect();
            t.try_reshape(&shape)
        }
        _ => {
            return Err(Error::IrConstruction {
                details: format!("Unsupported dtype for tensor creation: {:?}", dtype),
            });
        }
    };

    tensor.map_err(Error::from)
}

/// Extract raw data bytes from TensorProto.
pub fn extract_tensor_data(tensor: &TensorProto) -> Result<Vec<u8>> {
    if !tensor.raw_data.is_empty() {
        return Ok(tensor.raw_data.clone());
    }

    let dtype = convert_onnx_dtype(tensor.data_type)?;

    // Match on dtype to select the right field
    let data = match dtype.base() {
        ScalarDType::Float32 => tensor.float_data.iter().flat_map(|&v| v.to_le_bytes()).collect(),
        ScalarDType::Int32 | ScalarDType::UInt32 => tensor.int32_data.iter().flat_map(|&v| v.to_le_bytes()).collect(),
        ScalarDType::Int64 | ScalarDType::UInt64 => tensor.int64_data.iter().flat_map(|&v| v.to_le_bytes()).collect(),
        ScalarDType::UInt8 | ScalarDType::Int8 | ScalarDType::Bool => {
            tensor.int32_data.iter().map(|&v| v as u8).collect()
        }
        ScalarDType::Float16 | ScalarDType::BFloat16 => {
            tensor.int32_data.iter().flat_map(|&v| (v as u16).to_le_bytes()).collect()
        }
        _ => tensor.int32_data.iter().flat_map(|&v| v.to_le_bytes()).collect(),
    };

    Ok(data)
}

/// Get attribute from NodeProto by name.
pub fn get_attr<'a>(node: &'a NodeProto, name: &str) -> Option<&'a AttributeProto> {
    node.attribute.iter().find(|a| a.name == name)
}

/// Get int attribute value.
pub fn get_attr_int(node: &NodeProto, name: &str, default: i64) -> i64 {
    get_attr(node, name).map(|a| a.i).unwrap_or(default)
}

/// Get float attribute value.
pub fn get_attr_float(node: &NodeProto, name: &str, default: f32) -> f32 {
    get_attr(node, name).map(|a| a.f).unwrap_or(default)
}

/// Get string attribute value as bytes.
pub fn get_attr_bytes<'a>(node: &'a NodeProto, name: &str) -> Option<&'a [u8]> {
    get_attr(node, name).map(|a| a.s.as_slice())
}

/// Get string attribute value as String.
pub fn get_attr_string(node: &NodeProto, name: &str, default: &str) -> String {
    get_attr_bytes(node, name).map(|b| String::from_utf8_lossy(b).into_owned()).unwrap_or_else(|| default.to_string())
}

/// Get ints attribute value.
pub fn get_attr_ints(node: &NodeProto, name: &str) -> Vec<i64> {
    get_attr(node, name).map(|a| a.ints.clone()).unwrap_or_default()
}

/// Get floats attribute value.
pub fn get_attr_floats(node: &NodeProto, name: &str) -> Vec<f32> {
    get_attr(node, name).map(|a| a.floats.clone()).unwrap_or_default()
}

/// Get tensor attribute value (reference).
pub fn get_attr_tensor<'a>(node: &'a NodeProto, name: &str) -> Option<&'a TensorProto> {
    get_attr(node, name).and_then(|a| a.t.as_ref())
}

/// ONNX opset version (domain, version).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OpSetId {
    pub domain: String,
    pub version: i64,
}

/// Operator registry for dispatching ONNX ops to Morok Tensor operations.
pub struct OpRegistry {
    /// Supported opset versions (for future opset version dispatch)
    #[allow(dead_code)]
    opsets: Vec<OpSetId>,
}

/// Extract required input tensor at `idx`. Panics if input is missing (None).
fn inp(inputs: &[Option<Tensor>], idx: usize) -> &Tensor {
    inputs[idx].as_ref().expect("missing required ONNX input")
}

/// Parse reduction attributes (axes + keepdims) from ONNX node.
fn reduce_attrs(node: &NodeProto) -> (AxisSpec, bool) {
    let axes = get_attr_ints(node, "axes");
    let keepdims = get_attr_int(node, "keepdims", 1) == 1;
    let spec =
        if axes.is_empty() { AxisSpec::All } else { AxisSpec::Multiple(axes.iter().map(|&a| a as isize).collect()) };
    (spec, keepdims)
}

impl OpRegistry {
    pub fn new() -> Self {
        Self { opsets: Vec::new() }
    }

    /// Dispatch an ONNX operator (convenience for callers with non-optional inputs).
    pub fn dispatch(&self, op_type: &str, domain: &str, inputs: &[Tensor], node: &NodeProto) -> Result<Tensor> {
        let inputs: Vec<Option<Tensor>> = inputs.iter().cloned().map(Some).collect();
        let outputs = self.dispatch_multi(op_type, domain, &inputs, node)?;
        outputs
            .into_iter()
            .next()
            .ok_or_else(|| Error::IrConstruction { details: format!("Operator {} produced no outputs", op_type) })
    }

    /// Dispatch an ONNX operator, returning a vector of output tensors.
    /// Inputs use `Option<Tensor>` to correctly handle optional ONNX inputs
    /// (empty input names become `None`, preserving positional indices).
    pub fn dispatch_multi(
        &self,
        op_type: &str,
        domain: &str,
        inputs: &[Option<Tensor>],
        node: &NodeProto,
    ) -> Result<Vec<Tensor>> {
        let r = match op_type {
            // === Arithmetic ===
            "Add" => inp(inputs, 0).try_add(inp(inputs, 1))?,
            "Sub" => inp(inputs, 0).try_sub(inp(inputs, 1))?,
            "Mul" => inp(inputs, 0).try_mul(inp(inputs, 1))?,
            "Div" => inp(inputs, 0).try_div(inp(inputs, 1))?,
            "Neg" => inp(inputs, 0).try_neg()?,
            "Abs" => inp(inputs, 0).try_abs()?,
            "Pow" => inp(inputs, 0).try_pow(inp(inputs, 1))?,

            // === Math ===
            "Sqrt" => inp(inputs, 0).try_sqrt()?,
            "Exp" => inp(inputs, 0).try_exp()?,
            "Log" => inp(inputs, 0).try_log()?,
            "Ceil" => inp(inputs, 0).ceil()?,
            "Floor" => inp(inputs, 0).floor()?,
            "Round" => inp(inputs, 0).round()?,
            "Sign" => inp(inputs, 0).sign()?,
            "Reciprocal" => inp(inputs, 0).reciprocal()?,
            "Erf" => inp(inputs, 0).erf()?,
            "Sin" => inp(inputs, 0).sin()?,
            "Cos" => inp(inputs, 0).cos()?,
            "Tan" => inp(inputs, 0).tan()?,

            // === Activation ===
            "Relu" => inp(inputs, 0).relu()?,
            "Sigmoid" => inp(inputs, 0).sigmoid()?,
            "Tanh" => inp(inputs, 0).tanh()?,
            "Softmax" => {
                let axis = get_attr_int(node, "axis", -1) as isize;
                inp(inputs, 0).softmax(axis)?
            }
            "LogSoftmax" => {
                let axis = get_attr_int(node, "axis", -1) as isize;
                inp(inputs, 0).log_softmax(axis)?
            }

            // === Comparison ===
            "Equal" => inp(inputs, 0).try_eq(inp(inputs, 1))?,
            "Less" => inp(inputs, 0).try_lt(inp(inputs, 1))?,
            "LessOrEqual" => inp(inputs, 0).try_le(inp(inputs, 1))?,
            "Greater" => inp(inputs, 0).try_gt(inp(inputs, 1))?,
            "GreaterOrEqual" => inp(inputs, 0).try_ge(inp(inputs, 1))?,
            "Not" => inp(inputs, 0).logical_not()?,

            // === Conditional ===
            "Where" => inp(inputs, 1).where_(inp(inputs, 0), inp(inputs, 2))?,
            "Max" => inp(inputs, 0).maximum(inp(inputs, 1))?,
            "Min" => inp(inputs, 0).minimum(inp(inputs, 1))?,
            "Clip" => {
                let min = inputs.get(1).and_then(|o| o.as_ref());
                let max = inputs.get(2).and_then(|o| o.as_ref());
                inp(inputs, 0).clamp().maybe_min(min).maybe_max(max).call()?
            }

            // === Type ===
            "Cast" => {
                let to = get_attr_int(node, "to", 1);
                let dtype = convert_onnx_dtype(to as i32)?;
                inp(inputs, 0).cast(dtype)?
            }

            // === Shape ===
            "Reshape" => self.op_reshape(inputs, node)?,
            "Transpose" => self.op_transpose(inputs, node)?,
            "Squeeze" => self.op_squeeze(inputs, node)?,
            "Unsqueeze" => self.op_unsqueeze(inputs, node)?,
            "Flatten" => self.op_flatten(inputs, node)?,
            "Concat" => {
                let axis = get_attr_int(node, "axis", 0) as isize;
                let tensors: Vec<&Tensor> = inputs.iter().filter_map(|o| o.as_ref()).collect();
                Tensor::cat(&tensors, axis)?
            }
            "Shape" => inp(inputs, 0).shape_tensor()?,

            // === Indexing ===
            "Gather" => {
                let axis = get_attr_int(node, "axis", 0) as isize;
                inp(inputs, 0).gather(axis, inp(inputs, 1))?
            }

            // === Reductions ===
            "ReduceSum" => {
                let (spec, kd) = reduce_attrs(node);
                inp(inputs, 0).sum_with().axes(spec).keepdim(kd).call()?
            }
            "ReduceMean" => {
                let (spec, kd) = reduce_attrs(node);
                inp(inputs, 0).mean_with().axes(spec).keepdim(kd).call()?
            }
            "ReduceMax" => {
                let (spec, kd) = reduce_attrs(node);
                inp(inputs, 0).max_with().axes(spec).keepdim(kd).call()?
            }
            "ReduceMin" => {
                let (spec, kd) = reduce_attrs(node);
                inp(inputs, 0).min_with().axes(spec).keepdim(kd).call()?
            }
            "ReduceProd" => {
                let (spec, kd) = reduce_attrs(node);
                inp(inputs, 0).prod_with().axes(spec).keepdim(kd).call()?
            }
            "ArgMax" => {
                let axis = get_attr_int(node, "axis", 0) as isize;
                let keepdims = get_attr_int(node, "keepdims", 1) == 1;
                inp(inputs, 0).argmax_with().axis(Some(axis)).keepdim(keepdims).call()?
            }
            "ArgMin" => {
                let axis = get_attr_int(node, "axis", 0) as isize;
                let keepdims = get_attr_int(node, "keepdims", 1) == 1;
                inp(inputs, 0).argmin_with().axis(Some(axis)).keepdim(keepdims).call()?
            }

            // === NN ===
            "MatMul" => inp(inputs, 0).matmul(inp(inputs, 1))?,
            "Gemm" => self.op_gemm(inputs, node)?,
            "BatchNormalization" => self.op_batch_norm(inputs, node)?,

            // === Identity / Constant ===
            "Identity" => inp(inputs, 0).clone(),
            "Constant" => return self.op_constant(node).map(|t| vec![t]),

            _ => return UnsupportedOpSnafu { op: op_type.to_string(), domain: domain.to_string() }.fail(),
        };

        Ok(vec![r])
    }

    fn op_reshape(&self, inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Tensor> {
        let shape = get_attr_ints(node, "shape");
        if !shape.is_empty() {
            let shape: Vec<isize> = shape.iter().map(|&d| d as isize).collect();
            Ok(inp(inputs, 0).try_reshape(&shape)?)
        } else if inputs.len() > 1 && inputs[1].is_some() {
            Err(Error::IrConstruction { details: "Reshape with shape tensor not yet implemented".to_string() })
        } else {
            Err(Error::IrConstruction { details: "Reshape requires shape attribute or input".to_string() })
        }
    }

    fn op_transpose(&self, inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Tensor> {
        let perm = get_attr_ints(node, "perm");
        if perm.is_empty() {
            Ok(inp(inputs, 0).try_transpose(0, 1)?)
        } else {
            let perm: Vec<isize> = perm.iter().map(|&p| p as isize).collect();
            Ok(inp(inputs, 0).try_permute(&perm)?)
        }
    }

    fn op_squeeze(&self, inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Tensor> {
        let axes = get_attr_ints(node, "axes");
        if axes.is_empty() {
            Ok(inp(inputs, 0).try_squeeze(None)?)
        } else if axes.len() == 1 {
            Ok(inp(inputs, 0).try_squeeze(Some(axes[0] as isize))?)
        } else {
            let mut result = inp(inputs, 0).clone();
            let mut sorted_axes: Vec<isize> = axes.iter().map(|&a| a as isize).collect();
            sorted_axes.sort_by(|a, b| b.cmp(a));
            for axis in sorted_axes {
                result = result.try_squeeze(Some(axis))?;
            }
            Ok(result)
        }
    }

    fn op_unsqueeze(&self, inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Tensor> {
        let axes = get_attr_ints(node, "axes");
        if axes.is_empty() {
            return Err(Error::IrConstruction { details: "Unsqueeze requires axes attribute".to_string() });
        }
        let mut result = inp(inputs, 0).clone();
        let mut sorted_axes: Vec<isize> = axes.iter().map(|&a| a as isize).collect();
        sorted_axes.sort();
        for axis in sorted_axes {
            result = result.try_unsqueeze(axis)?;
        }
        Ok(result)
    }

    fn op_flatten(&self, inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Tensor> {
        let axis = get_attr_int(node, "axis", 1) as usize;
        let uop = inp(inputs, 0).uop();
        let shape = uop
            .shape()
            .map_err(|e| Error::IrConstruction { details: format!("Flatten shape error: {e}") })?
            .ok_or_else(|| Error::IrConstruction { details: "Flatten: no shape".to_string() })?;
        let dims: Vec<usize> = shape
            .iter()
            .map(|s| {
                s.as_const()
                    .ok_or_else(|| Error::IrConstruction { details: "Flatten requires concrete shape".to_string() })
            })
            .collect::<Result<_>>()?;

        if axis == 0 {
            let total: isize = dims.iter().product::<usize>() as isize;
            Ok(inp(inputs, 0).try_reshape(&[1, total])?)
        } else {
            let pre: isize = dims[..axis].iter().product::<usize>() as isize;
            let post: isize = dims[axis..].iter().product::<usize>() as isize;
            Ok(inp(inputs, 0).try_reshape(&[pre, post])?)
        }
    }

    fn op_gemm(&self, inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Tensor> {
        let alpha = get_attr_float(node, "alpha", 1.0);
        let beta = get_attr_float(node, "beta", 1.0);
        let trans_a = get_attr_int(node, "transA", 0) == 1;
        let trans_b = get_attr_int(node, "transB", 0) == 1;

        let mut a = inp(inputs, 0).clone();
        let mut b = inp(inputs, 1).clone();

        if trans_a {
            a = a.try_transpose(0, 1)?;
        }
        if trans_b {
            b = b.try_transpose(0, 1)?;
        }

        let mut result = a.matmul(&b)?;

        if alpha != 1.0 {
            result = result.try_mul(&Tensor::from_slice([alpha]))?;
        }

        if let Some(c) = inputs.get(2).and_then(|o| o.as_ref()) {
            let mut c = c.clone();
            if beta != 1.0 {
                c = c.try_mul(&Tensor::from_slice([beta]))?;
            }
            result = result.try_add(&c)?;
        }

        Ok(result)
    }

    fn op_batch_norm(&self, inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Tensor> {
        let (x, scale, bias, mean, var) =
            (inp(inputs, 0), inp(inputs, 1), inp(inputs, 2), inp(inputs, 3), inp(inputs, 4));
        let epsilon = get_attr_float(node, "epsilon", 1e-5);

        let var_plus_eps = var.try_add(&Tensor::from_slice([epsilon]))?;
        let invstd = var_plus_eps.try_rsqrt()?;

        Ok(x.batchnorm().scale(scale).bias(bias).mean(mean).invstd(&invstd).call()?)
    }

    fn op_constant(&self, node: &NodeProto) -> Result<Tensor> {
        if let Some(tensor_proto) = get_attr_tensor(node, "value") {
            return tensor_from_proto(tensor_proto);
        }
        if let Some(attr) = get_attr(node, "value_float") {
            return Ok(Tensor::const_(attr.f as f64, DType::Scalar(ScalarDType::Float32)));
        }
        let float_values = get_attr_floats(node, "value_floats");
        if !float_values.is_empty() {
            return Ok(Tensor::from_slice(&float_values));
        }
        if let Some(attr) = get_attr(node, "value_int") {
            return Ok(Tensor::const_(attr.i, DType::Scalar(ScalarDType::Int64)));
        }
        let int_values = get_attr_ints(node, "value_ints");
        if !int_values.is_empty() {
            return Ok(Tensor::from_slice(&int_values));
        }
        Err(Error::IrConstruction { details: "Constant node has no supported value attribute".to_string() })
    }
}

impl Default for OpRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_onnx_dtype_all_supported() {
        use tensor_proto::DataType;

        assert_eq!(convert_onnx_dtype(DataType::Float as i32).unwrap().base(), ScalarDType::Float32);
        assert_eq!(convert_onnx_dtype(DataType::Uint8 as i32).unwrap().base(), ScalarDType::UInt8);
        assert_eq!(convert_onnx_dtype(DataType::Int8 as i32).unwrap().base(), ScalarDType::Int8);
        assert_eq!(convert_onnx_dtype(DataType::Int32 as i32).unwrap().base(), ScalarDType::Int32);
        assert_eq!(convert_onnx_dtype(DataType::Int64 as i32).unwrap().base(), ScalarDType::Int64);
        assert_eq!(convert_onnx_dtype(DataType::Bool as i32).unwrap().base(), ScalarDType::Bool);
        assert_eq!(convert_onnx_dtype(DataType::Double as i32).unwrap().base(), ScalarDType::Float64);
    }

    #[test]
    fn test_convert_onnx_dtype_unsupported() {
        assert!(convert_onnx_dtype(8).is_err()); // String
        assert!(convert_onnx_dtype(999).is_err()); // Unknown
    }

    #[test]
    fn test_extract_tensor_data_raw() {
        let mut tensor = TensorProto::default();
        tensor.raw_data = vec![1, 2, 3, 4];
        let data = extract_tensor_data(&tensor).unwrap();
        assert_eq!(data, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_get_attr_int() {
        let mut node = NodeProto::default();
        let mut attr = AttributeProto::default();
        attr.name = "axis".to_string();
        attr.i = 42;
        node.attribute.push(attr);

        assert_eq!(get_attr_int(&node, "axis", 0), 42);
        assert_eq!(get_attr_int(&node, "missing", -1), -1);
    }

    #[test]
    fn test_get_attr_ints() {
        let mut node = NodeProto::default();
        let mut attr = AttributeProto::default();
        attr.name = "perm".to_string();
        attr.ints = vec![1, 2, 0];
        node.attribute.push(attr);

        assert_eq!(get_attr_ints(&node, "perm"), vec![1, 2, 0]);
        assert!(get_attr_ints(&node, "missing").is_empty());
    }

    #[test]
    fn test_tensor_from_proto_f32() {
        let mut tensor = TensorProto::default();
        tensor.data_type = 1; // FLOAT
        tensor.dims = vec![2, 3];
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        tensor.raw_data = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let result = tensor_from_proto(&tensor).unwrap();
        // Successfully created tensor
        assert!(result.buffer().is_some());
    }

    #[test]
    fn test_registry_add() {
        let registry = OpRegistry::new();
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
        let b = Tensor::from_slice(&[4.0f32, 5.0, 6.0]);
        let node = NodeProto::default();

        let result = registry.dispatch("Add", "", &[a, b], &node);
        let result = result.unwrap().realize().unwrap();
        assert!(result.buffer().is_some());
    }

    #[test]
    fn test_registry_matmul() {
        let registry = OpRegistry::new();
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0]).try_reshape(&[2, 2]).unwrap();
        let b = Tensor::from_slice(&[5.0f32, 6.0, 7.0, 8.0]).try_reshape(&[2, 2]).unwrap();
        let node = NodeProto::default();

        let result = registry.dispatch("MatMul", "", &[a, b], &node);
        let result = result.unwrap().realize().unwrap();
        assert!(result.buffer().is_some());
    }

    #[test]
    fn test_registry_relu() {
        let registry = OpRegistry::new();
        let x = Tensor::from_slice(&[-2.0f32, -1.0, 0.0, 1.0, 2.0]);
        let node = NodeProto::default();

        let result = registry.dispatch("Relu", "", &[x], &node);
        let result = result.unwrap().realize().unwrap();
        assert!(result.buffer().is_some());
    }

    #[test]
    fn test_registry_transpose() {
        let registry = OpRegistry::new();
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).try_reshape(&[2, 3]).unwrap();
        let mut node = NodeProto::default();

        let mut attr = AttributeProto::default();
        attr.name = "perm".to_string();
        attr.ints = vec![1, 0];
        node.attribute.push(attr);

        let result = registry.dispatch("Transpose", "", &[x], &node);
        let result = result.unwrap().realize().unwrap();
        assert!(result.buffer().is_some());
    }

    #[test]
    fn test_registry_sigmoid() {
        let registry = OpRegistry::new();
        let x = Tensor::from_slice(&[-1.0f32, 0.0, 1.0]);
        let node = NodeProto::default();

        let result = registry.dispatch("Sigmoid", "", &[x], &node);
        let result = result.unwrap().realize().unwrap();
        assert!(result.buffer().is_some());
    }

    #[test]
    fn test_registry_reduce_sum() {
        let registry = OpRegistry::new();
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0]).try_reshape(&[2, 2]).unwrap();
        let node = NodeProto::default();

        let result = registry.dispatch("ReduceSum", "", &[x], &node);
        let result = result.unwrap().realize().unwrap();
        assert!(result.buffer().is_some());
    }

    // === Constant Operator Tests ===

    #[test]
    fn test_constant_value_float() {
        let registry = OpRegistry::new();
        let mut node = NodeProto::default();
        node.op_type = "Constant".to_string();

        let mut attr = AttributeProto::default();
        attr.name = "value_float".to_string();
        attr.f = 3.14;
        node.attribute.push(attr);

        let result = registry.dispatch("Constant", "", &[], &node).unwrap();
        // Scalar constants need .contiguous() to force materialization (Tinygrad approach)
        let realized = result.contiguous().realize().unwrap();
        assert!(realized.buffer().is_some());
    }

    #[test]
    fn test_constant_value_floats() {
        let registry = OpRegistry::new();
        let mut node = NodeProto::default();
        node.op_type = "Constant".to_string();

        let mut attr = AttributeProto::default();
        attr.name = "value_floats".to_string();
        attr.floats = vec![1.0, 2.0, 3.0];
        node.attribute.push(attr);

        let result = registry.dispatch("Constant", "", &[], &node).unwrap();
        // from_slice creates input buffer, so realize() works directly
        let realized = result.realize().unwrap();
        assert!(realized.buffer().is_some());
    }

    #[test]
    fn test_constant_value_int() {
        let registry = OpRegistry::new();
        let mut node = NodeProto::default();
        node.op_type = "Constant".to_string();

        let mut attr = AttributeProto::default();
        attr.name = "value_int".to_string();
        attr.i = 42;
        node.attribute.push(attr);

        let result = registry.dispatch("Constant", "", &[], &node).unwrap();
        // Scalar constants need .contiguous() to force materialization (Tinygrad approach)
        let realized = result.contiguous().realize().unwrap();
        assert!(realized.buffer().is_some());
    }

    #[test]
    fn test_constant_value_ints() {
        let registry = OpRegistry::new();
        let mut node = NodeProto::default();
        node.op_type = "Constant".to_string();

        let mut attr = AttributeProto::default();
        attr.name = "value_ints".to_string();
        attr.ints = vec![0, 1, 2, 3];
        node.attribute.push(attr);

        let result = registry.dispatch("Constant", "", &[], &node).unwrap();
        // from_slice creates input buffer, so realize() works directly
        let realized = result.realize().unwrap();
        assert!(realized.buffer().is_some());
    }

    #[test]
    fn test_constant_value_tensor() {
        let registry = OpRegistry::new();
        let mut node = NodeProto::default();
        node.op_type = "Constant".to_string();

        // Create a TensorProto for the value attribute
        let mut tensor_proto = TensorProto::default();
        tensor_proto.data_type = 1; // FLOAT
        tensor_proto.dims = vec![2, 2];
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        tensor_proto.raw_data = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let mut attr = AttributeProto::default();
        attr.name = "value".to_string();
        attr.t = Some(tensor_proto);
        node.attribute.push(attr);

        let result = registry.dispatch("Constant", "", &[], &node).unwrap();
        let realized = result.realize().unwrap();
        assert!(realized.buffer().is_some());
    }

    // === New Operator Tests ===

    #[test]
    fn test_registry_abs() {
        let registry = OpRegistry::new();
        let x = Tensor::from_slice(&[-2.0f32, -1.0, 0.0, 1.0, 2.0]);
        let node = NodeProto::default();

        let result = registry.dispatch("Abs", "", &[x], &node).unwrap().realize().unwrap();
        assert!(result.buffer().is_some());
    }

    #[test]
    fn test_registry_gather() {
        let registry = OpRegistry::new();
        let data = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0]);
        let indices = Tensor::from_slice(&[0i64, 2, 4]);
        let node = NodeProto::default(); // axis defaults to 0

        let result = registry.dispatch("Gather", "", &[data, indices], &node).unwrap().realize().unwrap();
        assert!(result.buffer().is_some());
    }

    #[test]
    fn test_registry_gather_axis1() {
        let registry = OpRegistry::new();
        let data = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).try_reshape(&[2, 3]).unwrap();
        let indices = Tensor::from_slice(&[0i64, 2, 1, 0]).try_reshape(&[2, 2]).unwrap();

        let mut node = NodeProto::default();
        let mut attr = AttributeProto::default();
        attr.name = "axis".to_string();
        attr.i = 1;
        node.attribute.push(attr);

        let result = registry.dispatch("Gather", "", &[data, indices], &node).unwrap().realize().unwrap();
        assert!(result.buffer().is_some());
    }

    #[test]
    fn test_registry_equal() {
        let registry = OpRegistry::new();
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
        let b = Tensor::from_slice(&[1.0f32, 0.0, 3.0]);
        let node = NodeProto::default();

        let result = registry.dispatch("Equal", "", &[a, b], &node).unwrap().realize().unwrap();
        assert!(result.buffer().is_some());
    }

    #[test]
    fn test_registry_where() {
        let registry = OpRegistry::new();
        let condition = Tensor::from_slice(&[true, false, true]);
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
        let y = Tensor::from_slice(&[10.0f32, 20.0, 30.0]);
        let node = NodeProto::default();

        // ONNX Where: inputs are (condition, X, Y)
        let result = registry.dispatch("Where", "", &[condition, x, y], &node).unwrap().realize().unwrap();
        assert!(result.buffer().is_some());
    }

    #[test]
    fn test_registry_reduce_max() {
        let registry = OpRegistry::new();
        let x = Tensor::from_slice(&[1.0f32, 4.0, 2.0, 3.0]).try_reshape(&[2, 2]).unwrap();
        let node = NodeProto::default();

        let result = registry.dispatch("ReduceMax", "", &[x], &node).unwrap().realize().unwrap();
        assert!(result.buffer().is_some());
    }

    #[test]
    fn test_registry_reduce_with_keepdims() {
        let registry = OpRegistry::new();
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0]).try_reshape(&[2, 2]).unwrap();

        let mut node = NodeProto::default();
        let mut axes_attr = AttributeProto::default();
        axes_attr.name = "axes".to_string();
        axes_attr.ints = vec![1];
        node.attribute.push(axes_attr);
        let mut kd_attr = AttributeProto::default();
        kd_attr.name = "keepdims".to_string();
        kd_attr.i = 1;
        node.attribute.push(kd_attr);

        let result = registry.dispatch("ReduceSum", "", &[x], &node).unwrap().realize().unwrap();
        assert!(result.buffer().is_some());
    }

    #[test]
    fn test_registry_math_ops() {
        let registry = OpRegistry::new();
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
        let node = NodeProto::default();

        for op in ["Exp", "Log", "Ceil", "Floor", "Round", "Sign", "Reciprocal", "Sin", "Cos", "Tan"] {
            let result = registry.dispatch(op, "", &[x.clone()], &node);
            assert!(result.is_ok(), "Operator {op} failed: {:?}", result.err());
        }
    }

    #[test]
    fn test_registry_comparison_ops() {
        let registry = OpRegistry::new();
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
        let b = Tensor::from_slice(&[2.0f32, 2.0, 1.0]);
        let node = NodeProto::default();

        for op in ["Less", "LessOrEqual", "Greater", "GreaterOrEqual"] {
            let result = registry.dispatch(op, "", &[a.clone(), b.clone()], &node);
            assert!(result.is_ok(), "Operator {op} failed: {:?}", result.err());
        }
    }

    #[test]
    fn test_registry_flatten() {
        let registry = OpRegistry::new();
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).try_reshape(&[2, 3]).unwrap();
        let node = NodeProto::default(); // axis defaults to 1

        let result = registry.dispatch("Flatten", "", &[x], &node).unwrap();
        // Flatten [2, 3] with axis=1 should give [2, 3]
        let realized = result.realize().unwrap();
        assert!(realized.buffer().is_some());
    }

    #[test]
    fn test_registry_log_softmax() {
        let registry = OpRegistry::new();
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0]).try_reshape(&[2, 2]).unwrap();
        let node = NodeProto::default(); // axis defaults to -1

        // Graph construction succeeds (realization may hit backend limitations)
        let result = registry.dispatch("LogSoftmax", "", &[x], &node);
        assert!(result.is_ok());
    }
}
