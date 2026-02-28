//! ONNX operator registry - maps ONNX ops to Morok Tensor operations.

use std::path::Path;

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
    tensor_from_proto_ext(tensor, None)
}

/// Create a Tensor from ONNX TensorProto, with optional model directory for external data.
pub fn tensor_from_proto_ext(tensor: &TensorProto, model_dir: Option<&Path>) -> Result<Tensor> {
    let dtype = convert_onnx_dtype(tensor.data_type)?;
    let dims: Vec<usize> = tensor.dims.iter().map(|&d| d as usize).collect();

    let raw_data = if tensor.data_location == 1 {
        let dir = model_dir
            .ok_or_else(|| Error::IrConstruction { details: "External data requires model directory path".into() })?;
        load_external_data(tensor, dir)?
    } else {
        extract_tensor_data(tensor)?
    };

    create_tensor_from_raw(&raw_data, &dims, dtype)
}

/// Load external tensor data from file.
///
/// ONNX spec: when `data_location == 1`, `external_data` contains key-value pairs
/// with "location" (file path), "offset", and "length".
fn load_external_data(tensor: &TensorProto, model_dir: &Path) -> Result<Vec<u8>> {
    let mut location = None;
    let mut offset: u64 = 0;
    let mut length: Option<u64> = None;

    for kv in &tensor.external_data {
        match kv.key.as_str() {
            "location" => location = Some(kv.value.clone()),
            "offset" => offset = kv.value.parse().unwrap_or(0),
            "length" => length = kv.value.parse().ok(),
            _ => {}
        }
    }

    let location =
        location.ok_or_else(|| Error::IrConstruction { details: "External data missing 'location' key".into() })?;

    let path = model_dir.join(&location);
    let mut file = std::fs::File::open(&path).map_err(|e| Error::IrConstruction {
        details: format!("External data file not found: {}: {e}", path.display()),
    })?;

    use std::io::{Read, Seek, SeekFrom};
    if offset > 0 {
        file.seek(SeekFrom::Start(offset))
            .map_err(|e| Error::IrConstruction { details: format!("External data seek failed: {e}") })?;
    }

    let data = match length {
        Some(len) => {
            let mut buf = vec![0u8; len as usize];
            file.read_exact(&mut buf)
                .map_err(|e| Error::IrConstruction { details: format!("External data read failed: {e}") })?;
            buf
        }
        None => {
            let mut buf = Vec::new();
            file.read_to_end(&mut buf)
                .map_err(|e| Error::IrConstruction { details: format!("External data read failed: {e}") })?;
            buf
        }
    };

    Ok(data)
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
        ScalarDType::Int8 => typed!(i8),
        ScalarDType::Int16 => typed!(i16),
        ScalarDType::Int32 => typed!(i32),
        ScalarDType::Int64 => typed!(i64),
        ScalarDType::UInt8 => typed!(u8),
        ScalarDType::UInt16 => typed!(u16),
        ScalarDType::UInt32 => typed!(u32),
        ScalarDType::UInt64 => typed!(u64),
        ScalarDType::Bool => {
            // ONNX raw_data stores bools as single bytes; int32_data stores as i32.
            let values: Vec<bool> = if data.len() == dims.iter().product::<usize>() {
                data.iter().map(|&v| v != 0).collect()
            } else {
                bytemuck::cast_slice::<_, i32>(data).iter().map(|&v| v != 0).collect()
            };
            Tensor::from_slice(&values).try_reshape(&shape)
        }
        // Float16, BFloat16, FP8: no native Rust type — use raw bytes
        ScalarDType::Float16 | ScalarDType::BFloat16 | ScalarDType::FP8E4M3 | ScalarDType::FP8E5M2 => {
            Tensor::from_raw_bytes(data, dims, dtype)
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

    // ONNX proto field mapping (per onnx.proto spec):
    //   float_data (field 4)  → Float32
    //   int32_data (field 5)  → Int32, Int8, UInt8, Int16, UInt16, Bool, Float16, BFloat16, FP8
    //   int64_data (field 7)  → Int64
    //   double_data (field 10) → Float64
    //   uint64_data (field 11) → UInt64, UInt32
    let data = match dtype.base() {
        ScalarDType::Float32 => tensor.float_data.iter().flat_map(|&v| v.to_le_bytes()).collect(),
        ScalarDType::Float64 => tensor.double_data.iter().flat_map(|&v| v.to_le_bytes()).collect(),
        ScalarDType::Int32 => tensor.int32_data.iter().flat_map(|&v| v.to_le_bytes()).collect(),
        ScalarDType::Int64 => tensor.int64_data.iter().flat_map(|&v| v.to_le_bytes()).collect(),
        ScalarDType::UInt32 => tensor.uint64_data.iter().flat_map(|&v| (v as u32).to_le_bytes()).collect(),
        ScalarDType::UInt64 => tensor.uint64_data.iter().flat_map(|&v| v.to_le_bytes()).collect(),
        ScalarDType::UInt8 | ScalarDType::Int8 | ScalarDType::Bool => {
            tensor.int32_data.iter().map(|&v| v as u8).collect()
        }
        ScalarDType::Int16 | ScalarDType::UInt16 | ScalarDType::Float16 | ScalarDType::BFloat16 => {
            tensor.int32_data.iter().flat_map(|&v| (v as u16).to_le_bytes()).collect()
        }
        ScalarDType::FP8E4M3 | ScalarDType::FP8E5M2 => tensor.int32_data.iter().map(|&v| v as u8).collect(),
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
            "Gelu" => {
                let approximate = get_attr_string(node, "approximate", "none");
                if approximate == "tanh" {
                    inp(inputs, 0).gelu()?
                } else {
                    // Exact: 0.5 * x * (1 + erf(x / sqrt(2)))
                    let x = inp(inputs, 0);
                    let dtype = x.uop().dtype();
                    let half = Tensor::const_(0.5f64, dtype.clone());
                    let one = Tensor::const_(1.0f64, dtype.clone());
                    let sqrt2 = Tensor::const_(std::f64::consts::SQRT_2, dtype);
                    half.try_mul(x)?.try_mul(&one.try_add(&x.try_div(&sqrt2)?.erf()?)?)?
                }
            }
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
                let dtype = convert_onnx_dtype(to as i32).unwrap_or_else(|_| {
                    tracing::warn!("ONNX dtype {to} unsupported, falling back to Float32");
                    DType::Float32
                });
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
            "Shape" => {
                let start = get_attr_int(node, "start", 0) as isize;
                let end = get_attr_int(node, "end", i64::MAX) as isize;
                let shape = inp(inputs, 0).shape()?;
                let ndim = shape.len() as isize;
                let s = if start < 0 { (ndim + start).max(0) } else { start.min(ndim) } as usize;
                let e = (if end < 0 { (ndim + end).max(0) } else { end.min(ndim) } as usize).max(s);
                let dims: Vec<i64> = shape[s..e]
                    .iter()
                    .map(|d| {
                        d.as_const()
                            .map(|v| v as i64)
                            .ok_or_else(|| Error::IrConstruction { details: "Shape requires concrete dims".into() })
                    })
                    .collect::<Result<Vec<_>>>()?;
                Tensor::from_slice(&dims)
            }

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
                let select_last = get_attr_int(node, "select_last_index", 0) == 1;
                let x = inp(inputs, 0);
                if select_last {
                    let shape = x.shape()?;
                    let na = if axis < 0 { shape.len() as isize + axis } else { axis } as usize;
                    let ds = shape[na].as_const().ok_or_else(|| Error::IrConstruction {
                        details: format!("ArgMax select_last_index needs concrete axis {na}"),
                    })? as i64;
                    Tensor::const_(ConstValue::Int(ds - 1), DType::Int64).try_sub(
                        &x.flip(&[axis])?
                            .argmax_with()
                            .axis(Some(axis))
                            .keepdim(keepdims)
                            .call()?
                            .cast(DType::Int64)?,
                    )?
                } else {
                    x.argmax_with().axis(Some(axis)).keepdim(keepdims).call()?.cast(DType::Int64)?
                }
            }
            "ArgMin" => {
                let axis = get_attr_int(node, "axis", 0) as isize;
                let keepdims = get_attr_int(node, "keepdims", 1) == 1;
                let select_last = get_attr_int(node, "select_last_index", 0) == 1;
                let x = inp(inputs, 0);
                let neg_x = x.try_neg()?;
                if select_last {
                    let shape = x.shape()?;
                    let na = if axis < 0 { shape.len() as isize + axis } else { axis } as usize;
                    let ds = shape[na].as_const().ok_or_else(|| Error::IrConstruction {
                        details: format!("ArgMin select_last_index needs concrete axis {na}"),
                    })? as i64;
                    Tensor::const_(ConstValue::Int(ds - 1), DType::Int64).try_sub(
                        &neg_x
                            .flip(&[axis])?
                            .argmax_with()
                            .axis(Some(axis))
                            .keepdim(keepdims)
                            .call()?
                            .cast(DType::Int64)?,
                    )?
                } else {
                    neg_x.argmax_with().axis(Some(axis)).keepdim(keepdims).call()?.cast(DType::Int64)?
                }
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
        let allowzero = get_attr_int(node, "allowzero", 0);
        let data = inp(inputs, 0);

        let raw_shape: Vec<isize> = {
            let attr_shape = get_attr_ints(node, "shape");
            if !attr_shape.is_empty() {
                attr_shape.iter().map(|&d| d as isize).collect()
            } else if inputs.len() > 1 && inputs[1].is_some() {
                tensor_to_i64_vec(inp(inputs, 1))?.iter().map(|&v| v as isize).collect()
            } else {
                return Err(Error::IrConstruction { details: "Reshape requires shape attribute or input".to_string() });
            }
        };

        let shape = if allowzero == 0 {
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
        let shape = inp(inputs, 0).shape()?;
        let ndim = shape.len() as i64;
        let axis_raw = get_attr_int(node, "axis", 1);
        let axis = (if axis_raw < 0 { ndim + axis_raw } else { axis_raw }) as usize;
        let pre = morok_ir::sint_prod(&shape[..axis]);
        let pre_val = pre
            .as_const()
            .map(|v| v as isize)
            .ok_or_else(|| Error::IrConstruction { details: "Flatten requires concrete pre-axis dimensions".into() })?;
        Ok(inp(inputs, 0).try_reshape(&[pre_val, -1])?)
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
        use morok_ir::SInt;
        use morok_ir::shape::{align_shapes_left, broadcast_shape};

        let data = inp(inputs, 0);
        let target_i64 = tensor_to_i64_vec(inp(inputs, 1))?;
        let target: morok_ir::shape::Shape = target_i64.iter().map(|&v| SInt::from(v as usize)).collect();
        let data_shape = data.shape()?;
        let aligned = align_shapes_left(&[data_shape, target]);
        let result_shape = broadcast_shape(&aligned[0], &aligned[1])
            .map_err(|e| Error::IrConstruction { details: format!("Expand broadcast: {e}") })?;
        Ok(data.broadcast_to(&result_shape)?)
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
        let data = inp(inputs, 0);
        let ndim = data.ndim()?;
        let num_axes = pads.len() / 2;

        // Optional axes input (input[3]): route pads to specific dimensions
        let axes: Option<Vec<usize>> = inputs
            .get(3)
            .and_then(|o| o.as_ref())
            .map(tensor_to_i64_vec)
            .transpose()?
            .map(|v| v.iter().map(|&a| if a < 0 { (ndim as i64 + a) as usize } else { a as usize }).collect());

        let padding = if let Some(axes) = axes {
            let mut full = vec![(0isize, 0isize); ndim];
            for (i, &ax) in axes.iter().enumerate() {
                full[ax] = (pads[i] as isize, pads[num_axes + i] as isize);
            }
            full
        } else {
            (0..num_axes).map(|i| (pads[i] as isize, pads[num_axes + i] as isize)).collect()
        };
        Ok(data.try_pad_value(&padding, pad_value)?)
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

        let mut ranges: Vec<(isize, isize)> =
            (0..ndim).map(|d| (0isize, shape[d].as_const().unwrap() as isize)).collect();
        let mut flip_axes: Vec<isize> = Vec::new();

        for (i, &axis) in axes.iter().enumerate() {
            let d = shape[axis].as_const().unwrap() as i64;
            let step = steps[i];
            if step == 0 {
                return Err(Error::IrConstruction { details: "Slice step cannot be 0".into() });
            }

            // Replicate Python's slice.indices(d): step-dependent clamping
            let (lower, upper) = if step > 0 { (0i64, d) } else { (-1i64, d - 1) };
            let mut s = starts[i].clamp(-d, d - 1);
            if s < 0 {
                s += d;
            }
            let s = s.clamp(lower, upper);

            let mut e = ends[i].clamp(-d - 1, d);
            if e < 0 {
                e += d;
            }
            let e = e.clamp(lower, upper);

            if step * (e - s) < 0 {
                // Empty range
                ranges[axis] = (0, 0);
            } else if step < 0 {
                flip_axes.push(axis as isize);
                ranges[axis] = ((e + 1) as isize, (s + 1) as isize);
            } else {
                ranges[axis] = (s as isize, e as isize);
            }
        }

        // Shrink first, then flip (Tinygrad order)
        let mut result = data.try_shrink(&ranges)?;
        if !flip_axes.is_empty() {
            result = result.flip(&flip_axes)?;
        }

        // Apply strides > 1 via pad→reshape→shrink→reshape pattern
        for (i, &axis) in axes.iter().enumerate() {
            let abs_step = steps[i].unsigned_abs() as usize;
            if abs_step <= 1 {
                continue;
            }
            let cur = result.shape()?;
            let size = cur[axis].as_const().unwrap();
            let padded = size.div_ceil(abs_step) * abs_step;
            if padded > size {
                let mut p = vec![(0isize, 0isize); cur.len()];
                p[axis] = (0, (padded - size) as isize);
                result = result.try_pad(&p)?;
            }
            let n = padded / abs_step;
            // Reshape: split axis into (n_groups, step)
            let cs = result.shape()?;
            let mut rs: Vec<isize> = Vec::new();
            for (d, dim) in cs.iter().enumerate() {
                if d == axis {
                    rs.push(n as isize);
                    rs.push(abs_step as isize);
                } else {
                    rs.push(dim.as_const().unwrap() as isize);
                }
            }
            result = result.try_reshape(&rs)?;
            // Shrink step dim to (0, 1) — take first element of each group
            let ss = result.shape()?;
            let sr: Vec<(isize, isize)> = ss
                .iter()
                .enumerate()
                .map(|(d, dim)| if d == axis + 1 { (0, 1) } else { (0, dim.as_const().unwrap() as isize) })
                .collect();
            result = result.try_shrink(&sr)?;
            // Collapse step dim back
            let fs: Vec<isize> = result
                .shape()?
                .iter()
                .enumerate()
                .filter(|&(d, _)| d != axis + 1)
                .map(|(_, dim)| dim.as_const().unwrap() as isize)
                .collect();
            result = result.try_reshape(&fs)?;
        }

        // After flip or stride operations, the view chain may not be contiguous
        if !flip_axes.is_empty() || steps.iter().any(|&s| s.unsigned_abs() > 1) {
            result = result.contiguous();
        }

        Ok(result)
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

    // === Batch 2: Semantic fixes tests ===

    #[test]
    fn test_shape_start_end() {
        let registry = OpRegistry::new();
        let x = Tensor::from_slice(&[
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0,
            20.0, 21.0, 22.0, 23.0, 24.0,
        ])
        .try_reshape(&[2, 3, 4])
        .unwrap();
        let mut node = NodeProto::default();
        let mut attr_s = AttributeProto::default();
        attr_s.name = "start".to_string();
        attr_s.i = 1;
        node.attribute.push(attr_s);
        let mut attr_e = AttributeProto::default();
        attr_e.name = "end".to_string();
        attr_e.i = 3;
        node.attribute.push(attr_e);

        let result = registry.dispatch("Shape", "", &[x], &node).unwrap();
        let arr = result.to_ndarray::<i64>().unwrap();
        let vals: Vec<i64> = arr.iter().copied().collect();
        assert_eq!(vals, vec![3, 4]);
    }

    #[test]
    fn test_shape_negative_start() {
        let registry = OpRegistry::new();
        let x = Tensor::from_slice(&[1.0f32; 24]).try_reshape(&[2, 3, 4]).unwrap();
        let mut node = NodeProto::default();
        let mut attr = AttributeProto::default();
        attr.name = "start".to_string();
        attr.i = -1;
        node.attribute.push(attr);

        let result = registry.dispatch("Shape", "", &[x], &node).unwrap();
        let arr = result.to_ndarray::<i64>().unwrap();
        let vals: Vec<i64> = arr.iter().copied().collect();
        assert_eq!(vals, vec![4]);
    }

    #[test]
    fn test_gelu_exact() {
        let registry = OpRegistry::new();
        let x = Tensor::from_slice(&[0.0f32, 1.0, -1.0]);
        let mut node = NodeProto::default();
        let mut attr = AttributeProto::default();
        attr.name = "approximate".to_string();
        attr.s = b"none".to_vec();
        node.attribute.push(attr);

        let result = registry.dispatch("Gelu", "", &[x], &node).unwrap();
        let arr = result.to_ndarray::<f32>().unwrap();
        let vals: Vec<f32> = arr.iter().copied().collect();
        assert!((vals[0] - 0.0).abs() < 1e-4, "gelu(0) = {}", vals[0]);
        assert!((vals[1] - 0.8413).abs() < 1e-3, "gelu(1) = {}", vals[1]);
        assert!((vals[2] - (-0.1587)).abs() < 1e-3, "gelu(-1) = {}", vals[2]);
    }

    #[test]
    fn test_gelu_tanh_regression() {
        let registry = OpRegistry::new();
        let x = Tensor::from_slice(&[0.0f32, 1.0]);
        let mut node = NodeProto::default();
        let mut attr = AttributeProto::default();
        attr.name = "approximate".to_string();
        attr.s = b"tanh".to_vec();
        node.attribute.push(attr);

        let result = registry.dispatch("Gelu", "", &[x], &node).unwrap();
        let arr = result.to_ndarray::<f32>().unwrap();
        let vals: Vec<f32> = arr.iter().copied().collect();
        assert!((vals[0] - 0.0).abs() < 1e-4, "gelu_tanh(0) = {}", vals[0]);
        assert!((vals[1] - 0.8412).abs() < 1e-3, "gelu_tanh(1) = {}", vals[1]);
    }

    #[test]
    fn test_reshape_allowzero_copy() {
        let registry = OpRegistry::new();
        let data = Tensor::from_slice(&[1.0f32; 24]).try_reshape(&[2, 3, 4]).unwrap();
        let shape_tensor = Tensor::from_slice(&[0i64, 3, -1]);
        let inputs = vec![Some(data), Some(shape_tensor)];
        let node = NodeProto::default(); // allowzero defaults to 0

        let result = registry.dispatch_multi("Reshape", "", &inputs, &node, i64::MAX).unwrap();
        let s = result[0].shape().unwrap();
        assert_eq!(s.iter().map(|d| d.as_const().unwrap()).collect::<Vec<_>>(), vec![2, 3, 4]);
    }

    #[test]
    fn test_argmax_select_last() {
        let registry = OpRegistry::new();
        // [1, 3, 2, 3] — max=3 appears at indices 1 and 3
        let x = Tensor::from_slice(&[1.0f32, 3.0, 2.0, 3.0]);
        let mut node = NodeProto::default();
        let mut attr_sel = AttributeProto::default();
        attr_sel.name = "select_last_index".to_string();
        attr_sel.i = 1;
        node.attribute.push(attr_sel);

        let result = registry.dispatch("ArgMax", "", &[x], &node).unwrap();
        let arr = result.to_ndarray::<i64>().unwrap();
        let vals: Vec<i64> = arr.iter().copied().collect();
        assert_eq!(vals, vec![3], "ArgMax select_last should return 3");
    }

    #[test]
    fn test_argmin_select_last() {
        let registry = OpRegistry::new();
        // [3, 1, 2, 1] — min=1 appears at indices 1 and 3
        let x = Tensor::from_slice(&[3.0f32, 1.0, 2.0, 1.0]);
        let mut node = NodeProto::default();
        let mut attr_sel = AttributeProto::default();
        attr_sel.name = "select_last_index".to_string();
        attr_sel.i = 1;
        node.attribute.push(attr_sel);

        let result = registry.dispatch("ArgMin", "", &[x], &node).unwrap();
        let arr = result.to_ndarray::<i64>().unwrap();
        let vals: Vec<i64> = arr.iter().copied().collect();
        assert_eq!(vals, vec![3], "ArgMin select_last should return 3");
    }

    #[test]
    fn test_argmax_cast_int64() {
        let registry = OpRegistry::new();
        let x = Tensor::from_slice(&[1.0f32, 3.0, 2.0]);
        let node = NodeProto::default();

        let result = registry.dispatch("ArgMax", "", &[x], &node).unwrap();
        assert_eq!(result.uop().dtype(), DType::Int64, "ArgMax should always return Int64");
    }

    #[test]
    fn test_expand_broadcast() {
        let registry = OpRegistry::new();
        let data = Tensor::from_slice(&[1.0f32, 2.0, 3.0]).try_reshape(&[3, 1]).unwrap();
        let target = Tensor::from_slice(&[2i64, 3, 4]);
        let inputs = vec![Some(data), Some(target)];
        let node = NodeProto::default();

        let result = registry.dispatch_multi("Expand", "", &inputs, &node, i64::MAX).unwrap();
        let s = result[0].shape().unwrap();
        assert_eq!(s.iter().map(|d| d.as_const().unwrap()).collect::<Vec<_>>(), vec![2, 3, 4]);
    }

    #[test]
    fn test_flatten_axis_variants() {
        let registry = OpRegistry::new();

        for (axis, expected_shape) in [(0, vec![1, 24]), (1, vec![2, 12]), (2, vec![6, 4])] {
            let x = Tensor::from_slice(&[1.0f32; 24]).try_reshape(&[2, 3, 4]).unwrap();
            let mut node = NodeProto::default();
            let mut attr = AttributeProto::default();
            attr.name = "axis".to_string();
            attr.i = axis;
            node.attribute.push(attr);

            let result = registry.dispatch("Flatten", "", &[x], &node).unwrap();
            let s = result.shape().unwrap();
            let dims: Vec<usize> = s.iter().map(|d| d.as_const().unwrap()).collect();
            assert_eq!(dims, expected_shape, "Flatten axis={axis}");
        }
    }

    #[test]
    fn test_pad_with_axes() {
        let registry = OpRegistry::new();
        let data = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).try_reshape(&[2, 3]).unwrap();
        let pads = Tensor::from_slice(&[1i64, 1]); // 1 before, 1 after
        let axes = Tensor::from_slice(&[1i64]); // only pad axis 1
        let inputs = vec![Some(data), Some(pads), None, Some(axes)];
        let node = NodeProto::default();

        let result = registry.dispatch_multi("Pad", "", &inputs, &node, i64::MAX).unwrap();
        let s = result[0].shape().unwrap();
        assert_eq!(s.iter().map(|d| d.as_const().unwrap()).collect::<Vec<_>>(), vec![2, 5]);
    }

    #[test]
    fn test_cast_fallback() {
        let registry = OpRegistry::new();
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
        let mut node = NodeProto::default();
        let mut attr = AttributeProto::default();
        attr.name = "to".to_string();
        attr.i = 999; // invalid dtype code
        node.attribute.push(attr);

        // Should not crash — falls back to Float32
        let result = registry.dispatch("Cast", "", &[x], &node);
        assert!(result.is_ok(), "Cast with invalid dtype should fallback, not crash");
    }

    #[test]
    fn test_slice_step_2() {
        let registry = OpRegistry::new();
        let data = Tensor::from_slice(&[0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let starts = Tensor::from_slice(&[0i64]);
        let ends = Tensor::from_slice(&[10i64]);
        let axes = Tensor::from_slice(&[0i64]);
        let steps = Tensor::from_slice(&[2i64]);
        let inputs = vec![Some(data), Some(starts), Some(ends), Some(axes), Some(steps)];
        let node = NodeProto::default();

        let result = registry.dispatch_multi("Slice", "", &inputs, &node, i64::MAX).unwrap();
        let arr = result[0].to_ndarray::<f32>().unwrap();
        let vals: Vec<f32> = arr.iter().copied().collect();
        assert_eq!(vals, vec![0.0, 2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_slice_step_3() {
        let registry = OpRegistry::new();
        let data = Tensor::from_slice(&[0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let starts = Tensor::from_slice(&[0i64]);
        let ends = Tensor::from_slice(&[10i64]);
        let axes = Tensor::from_slice(&[0i64]);
        let steps = Tensor::from_slice(&[3i64]);
        let inputs = vec![Some(data), Some(starts), Some(ends), Some(axes), Some(steps)];
        let node = NodeProto::default();

        let result = registry.dispatch_multi("Slice", "", &inputs, &node, i64::MAX).unwrap();
        let arr = result[0].to_ndarray::<f32>().unwrap();
        let vals: Vec<f32> = arr.iter().copied().collect();
        assert_eq!(vals, vec![0.0, 3.0, 6.0, 9.0]);
    }

    #[test]
    fn test_slice_neg_step_2() {
        let registry = OpRegistry::new();
        let data = Tensor::from_slice(&[0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let starts = Tensor::from_slice(&[5i64]);
        let ends = Tensor::from_slice(&[0i64]);
        let axes = Tensor::from_slice(&[0i64]);
        let steps = Tensor::from_slice(&[-2i64]);
        let inputs = vec![Some(data), Some(starts), Some(ends), Some(axes), Some(steps)];
        let node = NodeProto::default();

        let result = registry.dispatch_multi("Slice", "", &inputs, &node, i64::MAX).unwrap();
        let arr = result[0].to_ndarray::<f32>().unwrap();
        let vals: Vec<f32> = arr.iter().copied().collect();
        assert_eq!(vals, vec![5.0, 3.0, 1.0]);
    }

    #[test]
    fn test_slice_full_reverse() {
        // start=5, end=-100, step=-1 → should include element 0
        let registry = OpRegistry::new();
        let data = Tensor::from_slice(&[0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let starts = Tensor::from_slice(&[5i64]);
        let ends = Tensor::from_slice(&[-100i64]);
        let axes = Tensor::from_slice(&[0i64]);
        let steps = Tensor::from_slice(&[-1i64]);
        let inputs = vec![Some(data), Some(starts), Some(ends), Some(axes), Some(steps)];
        let node = NodeProto::default();

        let result = registry.dispatch_multi("Slice", "", &inputs, &node, i64::MAX).unwrap();
        let arr = result[0].to_ndarray::<f32>().unwrap();
        let vals: Vec<f32> = arr.iter().copied().collect();
        assert_eq!(vals, vec![5.0, 4.0, 3.0, 2.0, 1.0, 0.0]);
    }

    #[test]
    fn test_slice_large_start_neg_step() {
        // start=100, end=0, step=-1 → clamps start to d-1=5
        let registry = OpRegistry::new();
        let data = Tensor::from_slice(&[0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let starts = Tensor::from_slice(&[100i64]);
        let ends = Tensor::from_slice(&[0i64]);
        let axes = Tensor::from_slice(&[0i64]);
        let steps = Tensor::from_slice(&[-1i64]);
        let inputs = vec![Some(data), Some(starts), Some(ends), Some(axes), Some(steps)];
        let node = NodeProto::default();

        let result = registry.dispatch_multi("Slice", "", &inputs, &node, i64::MAX).unwrap();
        let arr = result[0].to_ndarray::<f32>().unwrap();
        let vals: Vec<f32> = arr.iter().copied().collect();
        assert_eq!(vals, vec![5.0, 4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_slice_step_zero_errors() {
        let registry = OpRegistry::new();
        let data = Tensor::from_slice(&[0.0f32, 1.0, 2.0]);
        let starts = Tensor::from_slice(&[0i64]);
        let ends = Tensor::from_slice(&[3i64]);
        let axes = Tensor::from_slice(&[0i64]);
        let steps = Tensor::from_slice(&[0i64]);
        let inputs = vec![Some(data), Some(starts), Some(ends), Some(axes), Some(steps)];
        let node = NodeProto::default();

        assert!(registry.dispatch_multi("Slice", "", &inputs, &node, i64::MAX).is_err());
    }

    #[test]
    fn test_flatten_negative_axis() {
        let registry = OpRegistry::new();
        let x = Tensor::from_slice(&[1.0f32; 24]).try_reshape(&[2, 3, 4]).unwrap();
        let mut node = NodeProto::default();
        let mut attr = AttributeProto::default();
        attr.name = "axis".to_string();
        attr.i = -1;
        node.attribute.push(attr);

        let result = registry.dispatch("Flatten", "", &[x], &node).unwrap();
        let s = result.shape().unwrap();
        let dims: Vec<usize> = s.iter().map(|d| d.as_const().unwrap()).collect();
        assert_eq!(dims, vec![6, 4]); // axis=-1 → axis=2 → pre=2*3=6, post=4
    }

    #[test]
    fn test_shape_start_gt_end() {
        // start=2, end=1 → should return empty, not panic
        let registry = OpRegistry::new();
        let x = Tensor::from_slice(&[1.0f32; 24]).try_reshape(&[2, 3, 4]).unwrap();
        let inputs = vec![Some(x)];
        let mut node = NodeProto::default();
        let mut attr_s = AttributeProto::default();
        attr_s.name = "start".to_string();
        attr_s.i = 2;
        node.attribute.push(attr_s);
        let mut attr_e = AttributeProto::default();
        attr_e.name = "end".to_string();
        attr_e.i = 1;
        node.attribute.push(attr_e);

        let result = registry.dispatch_multi("Shape", "", &inputs, &node, i64::MAX).unwrap();
        let arr = result[0].to_ndarray::<i64>().unwrap();
        let vals: Vec<i64> = arr.iter().copied().collect();
        assert!(vals.is_empty()); // empty shape
    }

    // === Tensor creation dtype tests ===

    #[test]
    fn test_create_tensor_int8() {
        let values: Vec<i8> = vec![-128, 0, 127];
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let t = create_tensor_from_raw(&data, &[3], DType::Int8).unwrap();
        let arr = t.to_ndarray::<i8>().unwrap();
        assert_eq!(arr.as_slice().unwrap(), &[-128i8, 0, 127]);
    }

    #[test]
    fn test_create_tensor_uint8() {
        let values: Vec<u8> = vec![0, 128, 255];
        let t = create_tensor_from_raw(&values, &[3], DType::UInt8).unwrap();
        let arr = t.to_ndarray::<u8>().unwrap();
        assert_eq!(arr.as_slice().unwrap(), &[0u8, 128, 255]);
    }

    #[test]
    fn test_create_tensor_int16() {
        let values: Vec<i16> = vec![-32768, 0, 32767];
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let t = create_tensor_from_raw(&data, &[3], DType::Int16).unwrap();
        let arr = t.to_ndarray::<i16>().unwrap();
        assert_eq!(arr.as_slice().unwrap(), &[-32768i16, 0, 32767]);
    }

    #[test]
    fn test_create_tensor_uint16() {
        let values: Vec<u16> = vec![0, 32768, 65535];
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let t = create_tensor_from_raw(&data, &[3], DType::UInt16).unwrap();
        let arr = t.to_ndarray::<u16>().unwrap();
        assert_eq!(arr.as_slice().unwrap(), &[0u16, 32768, 65535]);
    }

    #[test]
    fn test_create_tensor_uint32() {
        let values: Vec<u32> = vec![0, 1_000_000, u32::MAX];
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let t = create_tensor_from_raw(&data, &[3], DType::UInt32).unwrap();
        let arr = t.to_ndarray::<u32>().unwrap();
        assert_eq!(arr.as_slice().unwrap(), &[0u32, 1_000_000, u32::MAX]);
    }

    #[test]
    fn test_create_tensor_uint64() {
        let values: Vec<u64> = vec![0, 1_000_000_000, u64::MAX];
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let t = create_tensor_from_raw(&data, &[3], DType::UInt64).unwrap();
        let arr = t.to_ndarray::<u64>().unwrap();
        assert_eq!(arr.as_slice().unwrap(), &[0u64, 1_000_000_000, u64::MAX]);
    }

    #[test]
    fn test_create_tensor_float16() {
        // IEEE 754 half-precision: 1.0 = 0x3C00, 2.0 = 0x4000, 0.5 = 0x3800
        let f16_bits: Vec<u16> = vec![0x3C00, 0x4000, 0x3800];
        let data: Vec<u8> = f16_bits.iter().flat_map(|v| v.to_le_bytes()).collect();
        let t = create_tensor_from_raw(&data, &[3], DType::Float16).unwrap();
        // Cast to f32 to verify values
        let t_f32 = t.cast(DType::Float32).unwrap();
        let arr = t_f32.to_ndarray::<f32>().unwrap();
        let vals: Vec<f32> = arr.iter().copied().collect();
        assert!((vals[0] - 1.0).abs() < 1e-3);
        assert!((vals[1] - 2.0).abs() < 1e-3);
        assert!((vals[2] - 0.5).abs() < 1e-3);
    }

    #[test]
    fn test_create_tensor_bfloat16() {
        // BFloat16: 1.0 = 0x3F80, 2.0 = 0x4000, 0.5 = 0x3F00
        let bf16_bits: Vec<u16> = vec![0x3F80, 0x4000, 0x3F00];
        let data: Vec<u8> = bf16_bits.iter().flat_map(|v| v.to_le_bytes()).collect();
        let t = create_tensor_from_raw(&data, &[3], DType::BFloat16).unwrap();
        let t_f32 = t.cast(DType::Float32).unwrap();
        let arr = t_f32.to_ndarray::<f32>().unwrap();
        let vals: Vec<f32> = arr.iter().copied().collect();
        assert!((vals[0] - 1.0).abs() < 1e-2);
        assert!((vals[1] - 2.0).abs() < 1e-2);
        assert!((vals[2] - 0.5).abs() < 1e-2);
    }

    #[test]
    fn test_extract_float64_from_double_data() {
        let mut tensor = TensorProto::default();
        tensor.data_type = tensor_proto::DataType::Double as i32;
        tensor.dims = vec![2];
        tensor.double_data = vec![1.5, 2.5];

        let data = extract_tensor_data(&tensor).unwrap();
        let values: Vec<f64> = bytemuck::cast_slice(&data).to_vec();
        assert_eq!(values, vec![1.5, 2.5]);
    }

    #[test]
    fn test_extract_uint64_from_uint64_data() {
        // ONNX spec: UInt64 uses uint64_data (proto field 11), NOT int64_data.
        let mut tensor = TensorProto::default();
        tensor.data_type = tensor_proto::DataType::Uint64 as i32;
        tensor.dims = vec![2];
        tensor.uint64_data = vec![100, 200];

        let data = extract_tensor_data(&tensor).unwrap();
        let values: Vec<u64> = bytemuck::cast_slice(&data).to_vec();
        assert_eq!(values, vec![100u64, 200]);
    }

    #[test]
    fn test_extract_uint32_from_uint64_data() {
        // ONNX spec: UInt32 uses uint64_data (proto field 11), NOT int32_data.
        let mut tensor = TensorProto::default();
        tensor.data_type = tensor_proto::DataType::Uint32 as i32;
        tensor.dims = vec![2];
        tensor.uint64_data = vec![100, 200];

        let data = extract_tensor_data(&tensor).unwrap();
        let values: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        assert_eq!(values, vec![100u32, 200]);
    }

    #[test]
    fn test_external_data_loading() {
        use std::io::Write;

        let dir = std::env::temp_dir().join("morok_test_external_data");
        std::fs::create_dir_all(&dir).unwrap();

        // Write external data file: 3 x f32 values
        let values: Vec<f32> = vec![1.0, 2.0, 3.0];
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let data_path = dir.join("weights.bin");
        let mut f = std::fs::File::create(&data_path).unwrap();
        f.write_all(&raw).unwrap();

        // Create TensorProto with external data
        let mut tensor = TensorProto::default();
        tensor.data_type = tensor_proto::DataType::Float as i32;
        tensor.dims = vec![3];
        tensor.data_location = 1;
        tensor.external_data = vec![
            crate::parser::onnx::StringStringEntryProto { key: "location".into(), value: "weights.bin".into() },
            crate::parser::onnx::StringStringEntryProto { key: "offset".into(), value: "0".into() },
            crate::parser::onnx::StringStringEntryProto { key: "length".into(), value: raw.len().to_string() },
        ];

        let result = tensor_from_proto_ext(&tensor, Some(&dir)).unwrap();
        let arr = result.to_ndarray::<f32>().unwrap();
        assert_eq!(arr.as_slice().unwrap(), &[1.0f32, 2.0, 3.0]);

        // Cleanup
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_external_data_with_offset() {
        use std::io::Write;

        let dir = std::env::temp_dir().join("morok_test_external_offset");
        std::fs::create_dir_all(&dir).unwrap();

        // Write: 8 bytes padding + 2 x f32
        let padding = vec![0u8; 8];
        let values: Vec<f32> = vec![42.0, 99.0];
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let data_path = dir.join("weights_offset.bin");
        let mut f = std::fs::File::create(&data_path).unwrap();
        f.write_all(&padding).unwrap();
        f.write_all(&raw).unwrap();

        let mut tensor = TensorProto::default();
        tensor.data_type = tensor_proto::DataType::Float as i32;
        tensor.dims = vec![2];
        tensor.data_location = 1;
        tensor.external_data = vec![
            crate::parser::onnx::StringStringEntryProto { key: "location".into(), value: "weights_offset.bin".into() },
            crate::parser::onnx::StringStringEntryProto { key: "offset".into(), value: "8".into() },
            crate::parser::onnx::StringStringEntryProto { key: "length".into(), value: raw.len().to_string() },
        ];

        let result = tensor_from_proto_ext(&tensor, Some(&dir)).unwrap();
        let arr = result.to_ndarray::<f32>().unwrap();
        assert_eq!(arr.as_slice().unwrap(), &[42.0f32, 99.0]);

        std::fs::remove_dir_all(&dir).ok();
    }
}
