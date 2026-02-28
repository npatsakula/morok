//! ONNX operator registry - maps ONNX ops to Morok Tensor operations.

use morok_dtype::{DType, ScalarDType};
use morok_ir::ConstValue;
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
    let shape: Vec<isize> = dims.iter().map(|&d| d as isize).collect();
    macro_rules! typed {
        ($ty:ty) => {{
            let values: Vec<$ty> = bytemuck::cast_slice(data).to_vec();
            Tensor::from_slice(&values).try_reshape(&shape)
        }};
    }
    let tensor = match dtype.base() {
        ScalarDType::Float32 => typed!(f32),
        ScalarDType::Float64 => typed!(f64),
        ScalarDType::Int32 => typed!(i32),
        ScalarDType::Int64 => typed!(i64),
        ScalarDType::Bool => {
            // ONNX raw_data stores bools as single bytes; int32_data stores as i32.
            // Detect format by checking if data is aligned to i32 boundaries.
            let values: Vec<bool> = if data.len() == dims.iter().product::<usize>() {
                // 1-byte-per-element (raw_data or extract_tensor_data from int32)
                data.iter().map(|&v| v != 0).collect()
            } else {
                // i32-per-element fallback
                bytemuck::cast_slice::<_, i32>(data).iter().map(|&v| v != 0).collect()
            };
            Tensor::from_slice(&values).try_reshape(&shape)
        }
        _ => {
            return Err(Error::IrConstruction { details: format!("Unsupported dtype for tensor creation: {dtype:?}") });
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

/// ONNX opset version (domain, version).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OpSetId {
    pub domain: String,
    pub version: i64,
}

/// Resolve opset version for a domain from the model's opset imports.
/// Default domain "" is equivalent to "ai.onnx". Returns 1 if not found.
pub fn onnx_opset_version(opsets: &[OpSetId], domain: &str) -> i64 {
    let normalized = if domain == "ai.onnx" { "" } else { domain };
    opsets
        .iter()
        .find(|op| {
            let op_domain = if op.domain == "ai.onnx" { "" } else { op.domain.as_str() };
            op_domain == normalized
        })
        .map(|op| op.version)
        .unwrap_or(1)
}

/// Operator registry for dispatching ONNX ops to Morok Tensor operations.
pub struct OpRegistry;

fn inp(inputs: &[Option<Tensor>], idx: usize) -> &Tensor {
    inputs[idx].as_ref().expect("missing required ONNX input")
}

/// Extract reduce axes and keepdims, opset-aware.
/// Opset >= 13: axes from input[1] tensor. Opset <= 12: axes from node attribute.
fn reduce_attrs(node: &NodeProto, inputs: &[Option<Tensor>], opset: i64) -> Result<(AxisSpec, bool)> {
    let keepdims = get_attr_int(node, "keepdims", 1) == 1;
    let noop_with_empty_axes = get_attr_int(node, "noop_with_empty_axes", 0) == 1;

    let axes: Vec<i64> = if opset >= 13 {
        // Opset 13+: axes from input[1] tensor
        inputs.get(1).and_then(|o| o.as_ref()).map(tensor_to_i64_vec).transpose()?.unwrap_or_default()
    } else {
        // Opset <= 12: axes from attribute
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
fn tensor_to_f64_scalar(t: &Tensor) -> Result<f64> {
    let arr = t
        .cast(DType::Float64)?
        .to_ndarray::<f64>()
        .map_err(|e| Error::IrConstruction { details: format!("tensor_to_f64_scalar: {e}") })?;
    arr.iter().next().copied().ok_or_else(|| Error::IrConstruction { details: "empty scalar tensor".into() })
}

/// Extract concrete i64 values from a tensor (shape/indices/pads inputs).
fn tensor_to_i64_vec(t: &Tensor) -> Result<Vec<i64>> {
    let arr = t
        .cast(DType::Int64)?
        .to_ndarray::<i64>()
        .map_err(|e| Error::IrConstruction { details: format!("tensor_to_i64_vec: {e}") })?;
    Ok(arr.iter().copied().collect())
}

impl OpRegistry {
    pub fn new() -> Self {
        Self
    }

    /// Dispatch an ONNX operator (convenience for callers with non-optional inputs).
    /// Uses a default opset version (latest). For opset-aware dispatch, use `dispatch_multi`.
    pub fn dispatch(&self, op_type: &str, domain: &str, inputs: &[Tensor], node: &NodeProto) -> Result<Tensor> {
        let inputs: Vec<Option<Tensor>> = inputs.iter().cloned().map(Some).collect();
        let outputs = self.dispatch_multi(op_type, domain, &inputs, node, i64::MAX)?;
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
        opset_version: i64,
    ) -> Result<Vec<Tensor>> {
        let r = match op_type {
            // === Arithmetic ===
            "Add" => inp(inputs, 0).try_add(inp(inputs, 1))?,
            "Sub" => inp(inputs, 0).try_sub(inp(inputs, 1))?,
            "Mul" => inp(inputs, 0).try_mul(inp(inputs, 1))?,
            "Div" => {
                let x = inp(inputs, 0);
                let y = inp(inputs, 1);
                let result = x.try_div(y)?;
                if x.uop().dtype().is_int() { result.trunc()? } else { result }
            }
            "Neg" => inp(inputs, 0).try_neg()?,
            "Abs" => inp(inputs, 0).try_abs()?,
            "Pow" => inp(inputs, 0).try_pow(inp(inputs, 1))?,
            "Mod" => {
                let fmod = get_attr_int(node, "fmod", 0);
                if fmod == 1 {
                    inp(inputs, 0).try_mod(inp(inputs, 1))?
                } else {
                    // fmod=0: x - floor(x/y) * y
                    let x = inp(inputs, 0);
                    let y = inp(inputs, 1);
                    let div = x.try_div(y)?;
                    let floored = div.floor()?;
                    let product = floored.try_mul(y)?;
                    x.try_sub(&product)?
                }
            }
            "Sum" => {
                let valid: Vec<&Tensor> = inputs.iter().filter_map(Option::as_ref).collect();
                let first = valid
                    .first()
                    .ok_or_else(|| Error::IrConstruction { details: "Sum requires at least one input".into() })?;
                let mut acc = (*first).clone();
                for t in &valid[1..] {
                    acc = acc.try_add(t)?;
                }
                acc
            }
            "Mean" => {
                let valid: Vec<&Tensor> = inputs.iter().filter_map(Option::as_ref).collect();
                let count = valid.len();
                let first = valid
                    .first()
                    .ok_or_else(|| Error::IrConstruction { details: "Mean requires at least one input".into() })?;
                let mut acc = (*first).clone();
                for t in &valid[1..] {
                    acc = acc.try_add(t)?;
                }
                acc.try_div(&Tensor::from_slice([count as f32]))?
            }

            // === Bitwise ===
            "BitShift" => {
                let dir = get_attr_string(node, "direction", "");
                if dir == "LEFT" {
                    inp(inputs, 0).lshift(inp(inputs, 1))?
                } else {
                    inp(inputs, 0).rshift(inp(inputs, 1))?
                }
            }
            "BitwiseAnd" => inp(inputs, 0).bitwise_and(inp(inputs, 1))?,
            "BitwiseOr" => inp(inputs, 0).bitwise_or(inp(inputs, 1))?,
            "BitwiseXor" => inp(inputs, 0).bitwise_xor(inp(inputs, 1))?,
            "BitwiseNot" => inp(inputs, 0).bitwise_not()?,

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
                let default_axis = if opset_version < 13 { 1 } else { -1 };
                let axis = get_attr_int(node, "axis", default_axis) as isize;
                inp(inputs, 0).softmax(axis)?
            }
            "LogSoftmax" => {
                let default_axis = if opset_version < 13 { 1 } else { -1 };
                let axis = get_attr_int(node, "axis", default_axis) as isize;
                inp(inputs, 0).log_softmax(axis)?
            }
            "Gelu" => inp(inputs, 0).gelu()?,
            "HardSigmoid" => {
                let alpha = get_attr_float(node, "alpha", 0.2) as f64;
                let beta = get_attr_float(node, "beta", 0.5) as f64;
                inp(inputs, 0).hard_sigmoid(alpha, beta)?
            }
            "LeakyRelu" => {
                let alpha = get_attr_float(node, "alpha", 0.01) as f64;
                inp(inputs, 0).leaky_relu(alpha)?
            }
            "PRelu" => inp(inputs, 0).prelu(inp(inputs, 1))?,
            "ThresholdedRelu" => {
                let alpha = get_attr_float(node, "alpha", 1.0) as f64;
                inp(inputs, 0).thresholded_relu(alpha)?
            }
            "Elu" => {
                let alpha = get_attr_float(node, "alpha", 1.0) as f64;
                inp(inputs, 0).elu(alpha)?
            }
            "Selu" => {
                let alpha = get_attr_float(node, "alpha", 1.6732632) as f64;
                let gamma = get_attr_float(node, "gamma", 1.050_701) as f64;
                inp(inputs, 0).selu(alpha, gamma)?
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
            "Max" => {
                let valid: Vec<&Tensor> = inputs.iter().filter_map(Option::as_ref).collect();
                let first = valid
                    .first()
                    .ok_or_else(|| Error::IrConstruction { details: "Max requires at least one input".into() })?;
                let mut acc = (*first).clone();
                for t in &valid[1..] {
                    acc = acc.maximum(t)?;
                }
                acc
            }
            "Min" => {
                let valid: Vec<&Tensor> = inputs.iter().filter_map(Option::as_ref).collect();
                let first = valid
                    .first()
                    .ok_or_else(|| Error::IrConstruction { details: "Min requires at least one input".into() })?;
                let mut acc = (*first).clone();
                for t in &valid[1..] {
                    acc = acc.minimum(t)?;
                }
                acc
            }
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
            "Squeeze" => self.op_squeeze(inputs, node, opset_version)?,
            "Unsqueeze" => self.op_unsqueeze(inputs, node, opset_version)?,
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
                let data = inp(inputs, 0);
                let idx = inp(inputs, 1);
                // ONNX spec: normalize negative indices (idx < 0 → idx + dim_size)
                let data_shape = data.shape()?;
                let ndim = data_shape.len();
                let norm_axis = if axis < 0 { (ndim as isize + axis) as usize } else { axis as usize };
                let dim_size = data_shape[norm_axis].as_const().ok_or_else(|| Error::IrConstruction {
                    details: format!("Gather requires concrete dimension on axis {norm_axis}"),
                })? as i64;
                let zero = Tensor::const_(ConstValue::Int(0), idx.uop().dtype());
                let dim_t = Tensor::const_(ConstValue::Int(dim_size), idx.uop().dtype());
                let neg_mask = idx.try_lt(&zero)?;
                let normalized_idx = idx.try_add(&dim_t)?.where_(&neg_mask, idx)?;
                data.gather(axis, &normalized_idx)?
            }

            // === Reductions ===
            "ReduceSum" => {
                let (spec, kd) = reduce_attrs(node, inputs, opset_version)?;
                inp(inputs, 0).sum_with().axes(spec).keepdim(kd).call()?
            }
            "ReduceMean" => {
                let (spec, kd) = reduce_attrs(node, inputs, opset_version)?;
                inp(inputs, 0).mean_with().axes(spec).keepdim(kd).call()?
            }
            "ReduceMax" => {
                let (spec, kd) = reduce_attrs(node, inputs, opset_version)?;
                inp(inputs, 0).max_with().axes(spec).keepdim(kd).call()?
            }
            "ReduceMin" => {
                let (spec, kd) = reduce_attrs(node, inputs, opset_version)?;
                inp(inputs, 0).min_with().axes(spec).keepdim(kd).call()?
            }
            "ReduceProd" => {
                let (spec, kd) = reduce_attrs(node, inputs, opset_version)?;
                inp(inputs, 0).prod_with().axes(spec).keepdim(kd).call()?
            }
            "ReduceSumSquare" => {
                let (spec, kd) = reduce_attrs(node, inputs, opset_version)?;
                inp(inputs, 0).square()?.sum_with().axes(spec).keepdim(kd).call()?
            }
            "ReduceL1" => {
                let (spec, kd) = reduce_attrs(node, inputs, opset_version)?;
                inp(inputs, 0).try_abs()?.sum_with().axes(spec).keepdim(kd).call()?
            }
            "ReduceL2" => {
                let (spec, kd) = reduce_attrs(node, inputs, opset_version)?;
                let x = inp(inputs, 0);
                let orig_dtype = x.uop().dtype();
                let needs_upcast =
                    matches!(orig_dtype.scalar(), Some(ScalarDType::Float16) | Some(ScalarDType::BFloat16));
                let x = if needs_upcast { x.cast(DType::Float32)? } else { x.clone() };
                let result = x.square()?.sum_with().axes(spec).keepdim(kd).call()?.try_sqrt()?;
                if needs_upcast { result.cast(orig_dtype)? } else { result }
            }
            "ReduceLogSum" => {
                let (spec, kd) = reduce_attrs(node, inputs, opset_version)?;
                inp(inputs, 0).sum_with().axes(spec).keepdim(kd).call()?.try_log()?
            }
            "ReduceLogSumExp" => {
                let (spec, kd) = reduce_attrs(node, inputs, opset_version)?;
                inp(inputs, 0).try_exp()?.sum_with().axes(spec).keepdim(kd).call()?.try_log()?
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

            // === Type ===
            "CastLike" => inp(inputs, 0).cast(inp(inputs, 1).uop().dtype())?,

            // === Shape (Phase 2) ===
            "Expand" => self.op_expand(inputs)?,
            "Pad" => self.op_pad(inputs, node)?,
            "Slice" => self.op_slice(inputs)?,
            "Split" => return self.op_split(inputs, node),
            "Tile" => {
                let repeats: Vec<usize> = tensor_to_i64_vec(inp(inputs, 1))?.iter().map(|&v| v as usize).collect();
                inp(inputs, 0).repeat(&repeats)?
            }
            "Range" => {
                let start_t = inp(inputs, 0);
                let out_dtype = start_t.uop().dtype();
                if out_dtype.is_float() {
                    let start = tensor_to_f64_scalar(start_t)?;
                    let limit = tensor_to_f64_scalar(inp(inputs, 1))?;
                    let delta = tensor_to_f64_scalar(inp(inputs, 2))?;
                    Tensor::arange_f64(start, limit, delta, out_dtype)?
                } else {
                    let start = tensor_to_i64_vec(start_t)?[0];
                    let limit = tensor_to_i64_vec(inp(inputs, 1))?[0];
                    let delta = tensor_to_i64_vec(inp(inputs, 2))?[0];
                    Tensor::arange_with_dtype(start, Some(limit), Some(delta), out_dtype)?
                }
            }
            "ConstantOfShape" => {
                let shape_i64 = tensor_to_i64_vec(inp(inputs, 0))?;
                let value = get_attr_tensor(node, "value")
                    .map(tensor_from_proto)
                    .transpose()?
                    .unwrap_or_else(|| Tensor::from_slice([0.0f32]));
                if shape_i64.contains(&0) {
                    Tensor::empty(value.uop().dtype())
                } else {
                    let shape: Vec<isize> = shape_i64.iter().map(|&v| v as isize).collect();
                    value.try_reshape(&[1])?.try_expand(&shape)?
                }
            }
            "Size" => Tensor::from_slice([inp(inputs, 0).numel()? as i64]),
            "Dropout" => {
                // Inference mode: return input unchanged + all-true mask matching shape
                let x = inp(inputs, 0).clone();
                let shape: Vec<usize> = x.shape()?.iter().map(|d| d.as_const().unwrap_or(1)).collect();
                let mask = Tensor::full(&shape, true, DType::Scalar(ScalarDType::Bool))?;
                return Ok(vec![x, mask]);
            }

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
            let shape: Vec<isize> = tensor_to_i64_vec(inp(inputs, 1))?.iter().map(|&v| v as isize).collect();
            Ok(inp(inputs, 0).try_reshape(&shape)?)
        } else {
            Err(Error::IrConstruction { details: "Reshape requires shape attribute or input".to_string() })
        }
    }

    fn op_transpose(&self, inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Tensor> {
        let perm = get_attr_ints(node, "perm");
        if perm.is_empty() {
            // ONNX spec: default is reverse of all dimensions
            let ndim = inp(inputs, 0).ndim()?;
            let reversed: Vec<isize> = (0..ndim).rev().map(|i| i as isize).collect();
            Ok(inp(inputs, 0).try_permute(&reversed)?)
        } else {
            let perm: Vec<isize> = perm.iter().map(|&p| p as isize).collect();
            Ok(inp(inputs, 0).try_permute(&perm)?)
        }
    }

    fn op_squeeze(&self, inputs: &[Option<Tensor>], node: &NodeProto, opset: i64) -> Result<Tensor> {
        let axes: Vec<i64> = if opset >= 13 {
            // Opset 13+: axes from input[1] tensor
            inputs.get(1).and_then(|o| o.as_ref()).map(tensor_to_i64_vec).transpose()?.unwrap_or_default()
        } else {
            get_attr_ints(node, "axes")
        };
        if axes.is_empty() {
            return Ok(inp(inputs, 0).try_squeeze(None)?);
        }
        let mut sorted: Vec<isize> = axes.iter().map(|&a| a as isize).collect();
        sorted.sort_by(|a, b| b.cmp(a)); // descending to preserve indices
        sorted.iter().try_fold(inp(inputs, 0).clone(), |t, &ax| Ok(t.try_squeeze(Some(ax))?))
    }

    fn op_unsqueeze(&self, inputs: &[Option<Tensor>], node: &NodeProto, opset: i64) -> Result<Tensor> {
        let axes: Vec<i64> =
            if opset >= 13 {
                // Opset 13+: axes from input[1] tensor
                inputs.get(1).and_then(|o| o.as_ref()).map(tensor_to_i64_vec).transpose()?.ok_or_else(|| {
                    Error::IrConstruction { details: "Unsqueeze (opset>=13) requires axes input".into() }
                })?
            } else {
                let axes = get_attr_ints(node, "axes");
                if axes.is_empty() {
                    return Err(Error::IrConstruction { details: "Unsqueeze requires axes attribute".into() });
                }
                axes
            };
        let mut sorted: Vec<isize> = axes.iter().map(|&a| a as isize).collect();
        sorted.sort(); // ascending for unsqueeze
        sorted.iter().try_fold(inp(inputs, 0).clone(), |t, &ax| Ok(t.try_unsqueeze(ax)?))
    }

    fn op_flatten(&self, inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Tensor> {
        let axis = get_attr_int(node, "axis", 1) as usize;
        let shape = inp(inputs, 0).shape()?;
        let pre: isize = shape[..axis]
            .iter()
            .try_fold(1isize, |acc, d| d.as_const().map(|v| acc * v as isize))
            .ok_or_else(|| Error::IrConstruction { details: "Flatten requires concrete shape".into() })?;
        Ok(inp(inputs, 0).try_reshape(&[pre, -1])?)
    }

    fn op_gemm(&self, inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Tensor> {
        let alpha = get_attr_float(node, "alpha", 1.0);
        let beta = get_attr_float(node, "beta", 1.0);
        let a = inp(inputs, 0);
        let b = inp(inputs, 1);
        let a = if get_attr_int(node, "transA", 0) == 1 { a.try_transpose(0, 1)? } else { a.clone() };
        let b = if get_attr_int(node, "transB", 0) == 1 { b.try_transpose(0, 1)? } else { b.clone() };
        let mut result = a.matmul(&b)?;
        if alpha != 1.0 {
            result = result.try_mul(&Tensor::from_slice([alpha]))?;
        }
        if let Some(c) = inputs.get(2).and_then(|o| o.as_ref()) {
            let c = if beta != 1.0 { c.try_mul(&Tensor::from_slice([beta]))? } else { c.clone() };
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

    fn op_expand(&self, inputs: &[Option<Tensor>]) -> Result<Tensor> {
        let data = inp(inputs, 0);
        let target_shape: Vec<isize> = tensor_to_i64_vec(inp(inputs, 1))?.iter().map(|&v| v as isize).collect();
        let data_ndim = data.ndim()?;

        // ONNX Expand uses numpy broadcasting: may add leading dimensions
        let mut result = data.clone();
        for _ in data_ndim..target_shape.len() {
            result = result.try_unsqueeze(0)?;
        }

        // -1 or same-size means keep, otherwise broadcast
        let cur_shape = result.shape()?;
        let expand_spec: Vec<isize> = target_shape
            .iter()
            .zip(cur_shape.iter())
            .map(|(&tgt, cur)| {
                let cur_val = cur.as_const().unwrap_or(1) as isize;
                if tgt == -1 || tgt == cur_val { -1 } else { tgt }
            })
            .collect();
        Ok(result.try_expand(&expand_spec)?)
    }

    fn op_pad(&self, inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Tensor> {
        let pads = tensor_to_i64_vec(inp(inputs, 1))?;
        let mode = get_attr_string(node, "mode", "constant");
        if mode != "constant" {
            return Err(Error::IrConstruction {
                details: format!("Pad mode '{}' not supported, only 'constant'", mode),
            });
        }
        let pad_value: f64 = match inputs.get(2).and_then(|o| o.as_ref()) {
            Some(cv) => tensor_to_f64_scalar(cv)?,
            None => 0.0,
        };
        // ONNX format: [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
        let ndim = pads.len() / 2;
        let padding: Vec<(isize, isize)> = (0..ndim).map(|i| (pads[i] as isize, pads[ndim + i] as isize)).collect();
        Ok(inp(inputs, 0).try_pad_value(&padding, pad_value)?)
    }

    fn op_slice(&self, inputs: &[Option<Tensor>]) -> Result<Tensor> {
        let data = inp(inputs, 0);
        let starts = tensor_to_i64_vec(inp(inputs, 1))?;
        let ends = tensor_to_i64_vec(inp(inputs, 2))?;
        let shape = data.shape()?;
        let ndim = shape.len();

        let axes: Vec<usize> = inputs
            .get(3)
            .and_then(|o| o.as_ref())
            .map(tensor_to_i64_vec)
            .transpose()?
            .map(|v| v.iter().map(|&a| if a < 0 { (ndim as i64 + a) as usize } else { a as usize }).collect())
            .unwrap_or_else(|| (0..starts.len()).collect());

        let steps: Vec<i64> = inputs
            .get(4)
            .and_then(|o| o.as_ref())
            .map(tensor_to_i64_vec)
            .transpose()?
            .unwrap_or_else(|| vec![1; starts.len()]);

        for &s in &steps {
            if s != 1 && s != -1 {
                return Err(Error::IrConstruction {
                    details: format!("Slice with step={s} not supported (only 1 and -1)"),
                });
            }
        }

        let mut ranges: Vec<(isize, isize)> =
            (0..ndim).map(|d| (0isize, shape[d].as_const().unwrap() as isize)).collect();
        let mut flip_axes: Vec<isize> = Vec::new();

        for (i, &axis) in axes.iter().enumerate() {
            let d = shape[axis].as_const().unwrap() as i64;
            let mut s = starts[i].clamp(-d, d);
            let mut e = ends[i].clamp(-d, d);
            if s < 0 {
                s += d;
            }
            if e < 0 {
                e += d;
            }

            if steps[i] == -1 {
                flip_axes.push(axis as isize);
                ranges[axis] = ((d - e) as isize, (d - s) as isize);
            } else {
                ranges[axis] = (s as isize, e as isize);
            }
        }

        let mut result = data.clone();
        if !flip_axes.is_empty() {
            result = result.flip(&flip_axes)?;
        }
        Ok(result.try_shrink(&ranges)?)
    }

    fn op_split(&self, inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Vec<Tensor>> {
        let axis = get_attr_int(node, "axis", 0) as isize;
        let data = inp(inputs, 0);
        let shape = data.shape()?;
        let ndim = shape.len();
        let norm_axis = if axis < 0 { (ndim as isize + axis) as usize } else { axis as usize };
        let dim_size = shape[norm_axis].as_const().unwrap();

        let split_sizes: Vec<usize> = if let Some(split_tensor) = inputs.get(1).and_then(|o| o.as_ref()) {
            tensor_to_i64_vec(split_tensor)?.iter().map(|&v| v as usize).collect()
        } else {
            let n = get_attr_int(node, "num_outputs", 0) as usize;
            if n == 0 {
                return Err(Error::IrConstruction {
                    details: "Split requires either split input or num_outputs attribute".into(),
                });
            }
            (0..n).map(|i| dim_size / n + if i < dim_size % n { 1 } else { 0 }).collect()
        };

        Ok(data.split(&split_sizes, axis)?)
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
    #[tracing_test::traced_test]
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

    // === Batch 1 bug fix tests ===

    #[test]
    fn test_max_variadic_3_inputs() {
        let registry = OpRegistry::new();
        let a = Tensor::from_slice(&[1.0f32, 5.0]);
        let b = Tensor::from_slice(&[3.0f32, 2.0]);
        let c = Tensor::from_slice(&[2.0f32, 4.0]);
        let inputs = vec![Some(a), Some(b), Some(c)];
        let node = NodeProto::default();

        let result = registry.dispatch_multi("Max", "", &inputs, &node, i64::MAX).unwrap();
        let arr = result[0].to_ndarray::<f32>().unwrap();
        let vals: Vec<f32> = arr.iter().copied().collect();
        assert_eq!(vals, vec![3.0, 5.0]);
    }

    #[test]
    fn test_max_single_input() {
        let registry = OpRegistry::new();
        let a = Tensor::from_slice(&[7.0f32, 3.0]);
        let inputs = vec![Some(a)];
        let node = NodeProto::default();

        let result = registry.dispatch_multi("Max", "", &inputs, &node, i64::MAX).unwrap();
        let arr = result[0].to_ndarray::<f32>().unwrap();
        let vals: Vec<f32> = arr.iter().copied().collect();
        assert_eq!(vals, vec![7.0, 3.0]);
    }

    #[test]
    fn test_min_variadic_3_inputs() {
        let registry = OpRegistry::new();
        let a = Tensor::from_slice(&[3.0f32, 1.0]);
        let b = Tensor::from_slice(&[1.0f32, 5.0]);
        let c = Tensor::from_slice(&[2.0f32, 3.0]);
        let inputs = vec![Some(a), Some(b), Some(c)];
        let node = NodeProto::default();

        let result = registry.dispatch_multi("Min", "", &inputs, &node, i64::MAX).unwrap();
        let arr = result[0].to_ndarray::<f32>().unwrap();
        let vals: Vec<f32> = arr.iter().copied().collect();
        assert_eq!(vals, vec![1.0, 1.0]);
    }

    #[test]
    fn test_split_remainder_distribution() {
        let registry = OpRegistry::new();
        let data = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        let inputs = vec![Some(data)];

        let mut node = NodeProto::default();
        let mut attr = AttributeProto::default();
        attr.name = "num_outputs".to_string();
        attr.i = 3;
        node.attribute.push(attr);

        let result = registry.dispatch_multi("Split", "", &inputs, &node, i64::MAX).unwrap();
        assert_eq!(result.len(), 3);
        // 7 / 3 = 2 rem 1, so first chunk gets 3, rest get 2
        // Verify via shape sizes
        let shapes: Vec<usize> = result.iter().map(|t| t.shape().unwrap()[0].as_const().unwrap()).collect();
        assert_eq!(shapes, vec![3, 2, 2]);
    }

    #[test]
    fn test_range_float() {
        let registry = OpRegistry::new();
        let start = Tensor::from_slice(&[0.0f32]);
        let limit = Tensor::from_slice(&[5.5f32]);
        let delta = Tensor::from_slice(&[1.5f32]);
        let node = NodeProto::default();

        let result = registry.dispatch("Range", "", &[start, limit, delta], &node).unwrap();
        let arr = result.to_ndarray::<f32>().unwrap();
        let vals: Vec<f32> = arr.iter().copied().collect();
        assert_eq!(vals, vec![0.0, 1.5, 3.0, 4.5]);
    }

    #[test]
    fn test_range_integer_regression() {
        let registry = OpRegistry::new();
        let start = Tensor::from_slice(&[0i32]);
        let limit = Tensor::from_slice(&[5i32]);
        let delta = Tensor::from_slice(&[1i32]);
        let node = NodeProto::default();

        let result = registry.dispatch("Range", "", &[start, limit, delta], &node).unwrap();
        let arr = result.to_ndarray::<i32>().unwrap();
        let vals: Vec<i32> = arr.iter().copied().collect();
        assert_eq!(vals, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_gather_negative_indices() {
        let registry = OpRegistry::new();
        let data = Tensor::from_slice(&[10.0f32, 20.0, 30.0, 40.0, 50.0]);
        let indices = Tensor::from_slice(&[0i64, -1, 2, -2]);
        let node = NodeProto::default();

        let result = registry.dispatch("Gather", "", &[data, indices], &node).unwrap();
        let arr = result.to_ndarray::<f32>().unwrap();
        let vals: Vec<f32> = arr.iter().copied().collect();
        assert_eq!(vals, vec![10.0, 50.0, 30.0, 40.0]);
    }

    #[test]
    fn test_dropout_mask_shape() {
        let registry = OpRegistry::new();
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).try_reshape(&[2, 3]).unwrap();
        let inputs = vec![Some(x)];
        let node = NodeProto::default();

        let result = registry.dispatch_multi("Dropout", "", &inputs, &node, i64::MAX).unwrap();
        assert_eq!(result.len(), 2);
        // Mask shape must match data shape [2, 3]
        let mask_shape = result[1].shape().unwrap();
        assert_eq!(mask_shape.len(), 2);
        assert_eq!(mask_shape[0].as_const().unwrap(), 2);
        assert_eq!(mask_shape[1].as_const().unwrap(), 3);
        // All mask values should be true
        let arr = result[1].to_ndarray::<bool>().unwrap();
        assert!(arr.iter().all(|&v| v));
    }

    #[test]
    fn test_constant_of_shape_empty() {
        let registry = OpRegistry::new();
        let shape = Tensor::from_slice(&[0i64]);
        let node = NodeProto::default();

        let result = registry.dispatch("ConstantOfShape", "", &[shape], &node).unwrap();
        let arr = result.to_ndarray::<f32>().unwrap();
        assert_eq!(arr.len(), 0);
        assert_eq!(arr.shape(), &[0]);
    }

    #[test]
    fn test_reduce_log_sum_exp() {
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

        // Use opset 12 so axes come from attributes
        let inputs = vec![Some(x)];
        let result = registry.dispatch_multi("ReduceLogSumExp", "", &inputs, &node, 12).unwrap();
        let arr = result[0].to_ndarray::<f32>().unwrap();
        let vals: Vec<f32> = arr.iter().copied().collect();
        // log(exp(1)+exp(2)) ≈ 2.3133, log(exp(3)+exp(4)) ≈ 4.3133
        assert!((vals[0] - 2.3133).abs() < 0.01, "got {}", vals[0]);
        assert!((vals[1] - 4.3133).abs() < 0.01, "got {}", vals[1]);
    }
}
