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

impl OpRegistry {
    pub fn new() -> Self {
        Self { opsets: Vec::new() }
    }

    /// Dispatch an ONNX operator to its Morok Tensor implementation.
    /// Returns a single tensor (for backward compatibility with single-output ops).
    pub fn dispatch(&self, op_type: &str, domain: &str, inputs: &[Tensor], node: &NodeProto) -> Result<Tensor> {
        let outputs = self.dispatch_multi(op_type, domain, inputs, node)?;
        outputs
            .into_iter()
            .next()
            .ok_or_else(|| Error::IrConstruction { details: format!("Operator {} produced no outputs", op_type) })
    }

    /// Dispatch an ONNX operator, returning a vector of output tensors.
    /// This supports multi-output operators like Split, TopK, etc.
    pub fn dispatch_multi(
        &self,
        op_type: &str,
        domain: &str,
        inputs: &[Tensor],
        node: &NodeProto,
    ) -> Result<Vec<Tensor>> {
        // Empty domain is the default ONNX domain
        // Non-empty domain could be "ai.onnx" (same as default) or custom ops

        match op_type {
            // Arithmetic ops (single output)
            "Add" => self.op_add(inputs).map(|t| vec![t]),
            "Sub" => self.op_sub(inputs).map(|t| vec![t]),
            "Mul" => self.op_mul(inputs).map(|t| vec![t]),
            "Div" => self.op_div(inputs).map(|t| vec![t]),
            "Neg" => self.op_neg(inputs).map(|t| vec![t]),
            "Pow" => self.op_pow(inputs).map(|t| vec![t]),

            // Math ops (single output)
            "Sqrt" => self.op_sqrt(inputs).map(|t| vec![t]),

            // Activation ops (single Output)
            "Relu" => self.op_relu(inputs).map(|t| vec![t]),
            "Sigmoid" => self.op_sigmoid(inputs).map(|t| vec![t]),
            "Tanh" => self.op_tanh(inputs).map(|t| vec![t]),
            "Softmax" => self.op_softmax(inputs, node).map(|t| vec![t]),

            // Type ops (single output)
            "Cast" => self.op_cast(inputs, node).map(|t| vec![t]),

            // Shape ops (single output)
            "Reshape" => self.op_reshape(inputs, node).map(|t| vec![t]),
            "Transpose" => self.op_transpose(inputs, node).map(|t| vec![t]),
            "Squeeze" => self.op_squeeze(inputs, node).map(|t| vec![t]),
            "Unsqueeze" => self.op_unsqueeze(inputs, node).map(|t| vec![t]),
            "Flatten" => self.op_flatten(inputs, node).map(|t| vec![t]),
            "Concat" => self.op_concat(inputs, node).map(|t| vec![t]),

            // Reduction ops (single output)
            "ReduceMean" => self.op_reduce_mean(inputs, node).map(|t| vec![t]),
            "ReduceSum" => self.op_reduce_sum(inputs, node).map(|t| vec![t]),

            // NN ops (single output)
            "MatMul" => self.op_matmul(inputs).map(|t| vec![t]),
            "Gemm" => self.op_gemm(inputs, node).map(|t| vec![t]),

            // Conditional ops (single output)
            "Clip" => self.op_clip(inputs, node).map(|t| vec![t]),

            // Identity (single output)
            "Identity" => Ok(vec![inputs[0].clone()]),

            // Constant (no inputs, creates tensor from attributes)
            "Constant" => self.op_constant(node).map(|t| vec![t]),

            _ => UnsupportedOpSnafu { op: op_type.to_string(), domain: domain.to_string() }.fail(),
        }
    }

    // === Arithmetic Operations ===

    fn op_add(&self, inputs: &[Tensor]) -> Result<Tensor> {
        inputs[0].try_add(&inputs[1]).map_err(|e| e.into())
    }

    fn op_sub(&self, inputs: &[Tensor]) -> Result<Tensor> {
        inputs[0].try_sub(&inputs[1]).map_err(|e| e.into())
    }

    fn op_mul(&self, inputs: &[Tensor]) -> Result<Tensor> {
        inputs[0].try_mul(&inputs[1]).map_err(|e| e.into())
    }

    fn op_div(&self, inputs: &[Tensor]) -> Result<Tensor> {
        inputs[0].try_div(&inputs[1]).map_err(|e| e.into())
    }

    fn op_neg(&self, inputs: &[Tensor]) -> Result<Tensor> {
        inputs[0].try_neg().map_err(|e| e.into())
    }

    // === Activation Operations ===

    fn op_relu(&self, inputs: &[Tensor]) -> Result<Tensor> {
        inputs[0].relu().map_err(|e| e.into())
    }

    fn op_sigmoid(&self, inputs: &[Tensor]) -> Result<Tensor> {
        inputs[0].sigmoid().map_err(|e| e.into())
    }

    fn op_tanh(&self, inputs: &[Tensor]) -> Result<Tensor> {
        inputs[0].tanh().map_err(|e| e.into())
    }

    // === Shape Operations ===

    fn op_reshape(&self, inputs: &[Tensor], node: &NodeProto) -> Result<Tensor> {
        // ONNX Reshape: input, shape (as tensor)
        // For now, get shape from attribute or second input
        let shape = get_attr_ints(node, "shape");
        if !shape.is_empty() {
            let shape: Vec<isize> = shape.iter().map(|&d| d as isize).collect();
            inputs[0].try_reshape(&shape).map_err(|e| e.into())
        } else if inputs.len() > 1 {
            // Shape provided as second input - this requires evaluating the shape tensor
            Err(Error::IrConstruction { details: "Reshape with shape tensor not yet implemented".to_string() })
        } else {
            Err(Error::IrConstruction { details: "Reshape requires shape attribute or input".to_string() })
        }
    }

    fn op_transpose(&self, inputs: &[Tensor], node: &NodeProto) -> Result<Tensor> {
        let perm = get_attr_ints(node, "perm");
        if perm.is_empty() {
            // Default: reverse all dimensions - use 2D transpose as fallback
            inputs[0].try_transpose(0, 1).map_err(|e| e.into())
        } else {
            let perm: Vec<isize> = perm.iter().map(|&p| p as isize).collect();
            inputs[0].try_permute(&perm).map_err(|e| e.into())
        }
    }

    fn op_squeeze(&self, inputs: &[Tensor], node: &NodeProto) -> Result<Tensor> {
        // Squeeze removes dimensions of size 1
        // axes attribute specifies which dims to squeeze (optional)
        let axes = get_attr_ints(node, "axes");

        if axes.is_empty() {
            // No axes specified - squeeze all size-1 dimensions
            inputs[0].try_squeeze(None).map_err(|e| e.into())
        } else if axes.len() == 1 {
            // Single axis
            inputs[0].try_squeeze(Some(axes[0] as isize)).map_err(|e| e.into())
        } else {
            // Multiple axes - squeeze them one by one (in reverse order to maintain indices)
            let mut result = inputs[0].clone();
            let mut sorted_axes: Vec<isize> = axes.iter().map(|&a| a as isize).collect();
            sorted_axes.sort_by(|a, b| b.cmp(a)); // Sort descending
            for axis in sorted_axes {
                result = result.try_squeeze(Some(axis)).map_err(Error::from)?;
            }
            Ok(result)
        }
    }

    fn op_unsqueeze(&self, inputs: &[Tensor], node: &NodeProto) -> Result<Tensor> {
        // Unsqueeze adds dimensions of size 1 at specified positions
        let axes = get_attr_ints(node, "axes");

        if axes.is_empty() {
            return Err(Error::IrConstruction { details: "Unsqueeze requires axes attribute".to_string() });
        }

        let mut result = inputs[0].clone();
        // Sort axes ascending to insert dimensions in correct order
        let mut sorted_axes: Vec<isize> = axes.iter().map(|&a| a as isize).collect();
        sorted_axes.sort();
        for axis in sorted_axes {
            result = result.try_unsqueeze(axis).map_err(Error::from)?;
        }
        Ok(result)
    }

    fn op_flatten(&self, inputs: &[Tensor], node: &NodeProto) -> Result<Tensor> {
        let axis = get_attr_int(node, "axis", 1) as isize;
        // Flatten: reshape to [prod(dims[..axis]), prod(dims[axis..])]
        // For 2D flatten with axis=1, reshape to [batch, -1]
        let shape: Vec<isize> = if axis <= 0 { vec![-1] } else { vec![-1, -1] };
        inputs[0].try_reshape(&shape).map_err(|e| e.into())
    }

    fn op_concat(&self, _inputs: &[Tensor], _node: &NodeProto) -> Result<Tensor> {
        Err(Error::IrConstruction { details: "Concat not yet implemented".to_string() })
    }

    // === Reduction Operations ===

    fn op_reduce_mean(&self, inputs: &[Tensor], node: &NodeProto) -> Result<Tensor> {
        let axes = get_attr_ints(node, "axes");
        let axis_spec: AxisSpec = if axes.is_empty() {
            AxisSpec::All
        } else {
            AxisSpec::Multiple(axes.iter().map(|&a| a as isize).collect())
        };
        inputs[0].mean(axis_spec).map_err(|e| e.into())
    }

    fn op_reduce_sum(&self, inputs: &[Tensor], node: &NodeProto) -> Result<Tensor> {
        let axes = get_attr_ints(node, "axes");
        let axis_spec: AxisSpec = if axes.is_empty() {
            AxisSpec::All
        } else {
            AxisSpec::Multiple(axes.iter().map(|&a| a as isize).collect())
        };
        inputs[0].sum(axis_spec).map_err(|e| e.into())
    }

    // === Neural Network Operations ===

    fn op_matmul(&self, inputs: &[Tensor]) -> Result<Tensor> {
        inputs[0].matmul(&inputs[1]).map_err(|e| e.into())
    }

    fn op_gemm(&self, inputs: &[Tensor], node: &NodeProto) -> Result<Tensor> {
        // Gemm: Y = alpha * A' @ B' + beta * C
        // where A' = transA ? A.T : A, B' = transB ? B.T : B
        let alpha = get_attr_float(node, "alpha", 1.0);
        let beta = get_attr_float(node, "beta", 1.0);
        let trans_a = get_attr_int(node, "transA", 0) == 1;
        let trans_b = get_attr_int(node, "transB", 0) == 1;

        let mut a = inputs[0].clone();
        let mut b = inputs[1].clone();

        if trans_a {
            a = a.try_transpose(0, 1)?;
        }
        if trans_b {
            b = b.try_transpose(0, 1)?;
        }

        let mut result = a.matmul(&b)?;

        // Apply alpha scaling
        if alpha != 1.0 {
            let alpha_tensor = Tensor::from_slice([alpha]);
            result = result.try_mul(&alpha_tensor)?;
        }

        // Add bias if present
        if inputs.len() > 2 {
            let mut c = inputs[2].clone();
            // Apply beta scaling
            if beta != 1.0 {
                let beta_tensor = Tensor::from_slice([beta]);
                c = c.try_mul(&beta_tensor)?;
            }
            result = result.try_add(&c)?;
        }

        Ok(result)
    }

    // === Math Operations ===

    fn op_pow(&self, inputs: &[Tensor]) -> Result<Tensor> {
        inputs[0].try_pow(&inputs[1]).map_err(|e| e.into())
    }

    fn op_sqrt(&self, inputs: &[Tensor]) -> Result<Tensor> {
        inputs[0].try_sqrt().map_err(|e| e.into())
    }

    // === Type Operations ===

    fn op_cast(&self, inputs: &[Tensor], node: &NodeProto) -> Result<Tensor> {
        let to = get_attr_int(node, "to", 1); // Default to FLOAT
        let dtype = convert_onnx_dtype(to as i32)?;
        inputs[0].cast(dtype).map_err(|e| e.into())
    }

    // === Activation Operations (additional) ===

    fn op_softmax(&self, inputs: &[Tensor], node: &NodeProto) -> Result<Tensor> {
        let axis = get_attr_int(node, "axis", -1) as isize;
        inputs[0].softmax(axis).map_err(|e| e.into())
    }

    // === Conditional Operations ===

    fn op_clip(&self, inputs: &[Tensor], _node: &NodeProto) -> Result<Tensor> {
        // Clip: clip(input, min, max)
        // min and max can be from inputs (indices 1 and 2)
        let min = inputs.get(1);
        let max = inputs.get(2);

        // Use the builder pattern with maybe_min/maybe_max which accept Option
        inputs[0].clamp().maybe_min(min).maybe_max(max).call().map_err(|e| e.into())
    }

    // === Constant Operations ===

    /// Creates a constant tensor from ONNX Constant node attributes.
    ///
    /// The ONNX Constant operator supports multiple attribute formats:
    /// - `value`: A TensorProto containing the full tensor data
    /// - `value_float`: A single float scalar (f32)
    /// - `value_floats`: A 1D tensor of floats (Vec<f32>)
    /// - `value_int`: A single int64 scalar (i64)
    /// - `value_ints`: A 1D tensor of int64s (Vec<i64>)
    ///
    /// For the `value` attribute, we reuse `tensor_from_proto` which handles
    /// raw data and typed data fields properly.
    ///
    /// For scalar attributes, we use `Tensor::const_` which embeds constants
    /// directly in the IR. Following Tinygrad's approach, pure constants don't
    /// allocate buffers until `.contiguous().realize()` is called.
    ///
    /// For array attributes, we use `Tensor::from_slice` which creates input
    /// buffers with proper tensor shapes (not vector dtypes). This matches
    /// Tinygrad's `_frompy` approach for list/tuple inputs.
    ///
    /// See: <https://onnx.ai/onnx/operators/onnx__Constant.html>
    fn op_constant(&self, node: &NodeProto) -> Result<Tensor> {
        // Check for value attribute (full TensorProto)
        if let Some(tensor_proto) = get_attr_tensor(node, "value") {
            return tensor_from_proto(tensor_proto);
        }

        // Check for value_float (scalar float)
        if let Some(attr) = get_attr(node, "value_float") {
            let value = attr.f as f64; // ConstValue::Float is f64
            return Ok(Tensor::const_(value, DType::Scalar(ScalarDType::Float32)));
        }

        // Check for value_floats (1D float array) - use from_slice for proper tensor shape
        let float_values = get_attr_floats(node, "value_floats");
        if !float_values.is_empty() {
            return Ok(Tensor::from_slice(&float_values));
        }

        // Check for value_int (scalar int)
        if let Some(attr) = get_attr(node, "value_int") {
            let value = attr.i;
            return Ok(Tensor::const_(value, DType::Scalar(ScalarDType::Int64)));
        }

        // Check for value_ints (1D int array) - use from_slice for proper tensor shape
        let int_values = get_attr_ints(node, "value_ints");
        if !int_values.is_empty() {
            return Ok(Tensor::from_slice(&int_values));
        }

        // No supported constant attribute found
        Err(Error::IrConstruction {
            details: "Constant node has no supported value attribute (value, value_float, value_floats, value_int, value_ints)".to_string(),
        })
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
}
