#![allow(clippy::result_large_err)]
//! ONNX operator registry - maps ONNX ops to Morok Tensor operations.

pub mod proto;
pub use proto::*;

pub(crate) mod attr;
pub(crate) use attr::*;

mod constant;
mod indexing;
mod nn;
mod shape;
mod transformer;

use morok_dtype::{DType, ScalarDType};
use morok_ir::ConstValue;
use morok_tensor::Tensor;
use morok_tensor::reduce::AxisSpec;

use crate::error::{Error, Result, UnsupportedOpSnafu};
use crate::parser::onnx::NodeProto;

/// Fold a variadic list of tensors with a binary operation.
fn fold_variadic(
    inputs: &[Option<Tensor>],
    op_name: &str,
    f: fn(&Tensor, &Tensor) -> std::result::Result<Tensor, morok_tensor::error::Error>,
) -> Result<Tensor> {
    let valid: Vec<&Tensor> = inputs.iter().filter_map(Option::as_ref).collect();
    let first = valid
        .first()
        .ok_or_else(|| Error::IrConstruction { details: format!("{op_name} requires at least one input") })?;
    let mut acc = (*first).clone();
    for t in &valid[1..] {
        acc = f(&acc, t)?;
    }
    Ok(acc)
}

/// Operator registry for dispatching ONNX ops to Morok Tensor operations.
pub struct OpRegistry;

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
        let mut attrs = Attrs::new(node);

        // Domain-specific ops (checked first)
        if let Some(result) = match (domain, op_type) {
            ("com.microsoft", "Attention") => Some(transformer::op_attention_contrib(inputs, &mut attrs)),
            ("com.microsoft", "SkipLayerNormalization") => Some(transformer::op_skip_layer_norm(inputs, &mut attrs)),
            ("com.microsoft", "EmbedLayerNormalization") => Some(transformer::op_embed_layer_norm(inputs, &mut attrs)),
            ("com.microsoft", "RotaryEmbedding") => Some(transformer::op_rotary_embedding_contrib(inputs, &mut attrs)),
            _ => None,
        } {
            let tensors = result?;
            attrs.done()?;
            return Ok(tensors);
        }

        let results = match op_type {
            // === Arithmetic ===
            "Add" => vec![inp(inputs, 0).try_add(inp(inputs, 1))?],
            "Sub" => vec![inp(inputs, 0).try_sub(inp(inputs, 1))?],
            "Mul" => vec![inp(inputs, 0).try_mul(inp(inputs, 1))?],
            "Div" => {
                let x = inp(inputs, 0);
                let y = inp(inputs, 1);
                let result = x.try_div(y)?;
                vec![if x.uop().dtype().is_int() { result.trunc()? } else { result }]
            }
            "Neg" => vec![inp(inputs, 0).try_neg()?],
            "Abs" => vec![inp(inputs, 0).try_abs()?],
            "Pow" => vec![inp(inputs, 0).try_pow(inp(inputs, 1))?],
            "Mod" => {
                let fmod = attrs.int("fmod", 0);
                let x = inp(inputs, 0);
                let y = inp(inputs, 1);
                vec![if fmod == 1 {
                    // fmod=1: C-style remainder (sign of dividend)
                    x.try_mod(y)?
                } else if x.uop().dtype().is_int() {
                    // fmod=0 integers: Python-style modulo (sign of divisor)
                    let trunc_mod = x.try_mod(y)?;
                    let zero = trunc_mod.zero()?;
                    let mod_ne_zero = trunc_mod.try_ne(&zero)?;
                    let signs_differ = trunc_mod.bitwise_xor(y)?.try_lt(&zero)?;
                    let needs_adj = mod_ne_zero.bitwise_and(&signs_differ)?;
                    trunc_mod.try_add(&y.where_(&needs_adj, &zero)?)?
                } else {
                    // fmod=0 floats: x - floor(x/y) * y
                    let div = x.try_div(y)?;
                    x.try_sub(&div.floor()?.try_mul(y)?)?
                }]
            }
            "Sum" => vec![fold_variadic(inputs, "Sum", |a, b| a.try_add(b))?],
            "Mean" => {
                let count = inputs.iter().filter(|o| o.is_some()).count();
                vec![fold_variadic(inputs, "Mean", |a, b| a.try_add(b))?.try_div(&Tensor::from_slice([count as f32]))?]
            }

            // === Bitwise ===
            "BitShift" => {
                let dir = attrs.string("direction", "");
                vec![if dir == "LEFT" {
                    inp(inputs, 0).lshift(inp(inputs, 1))?
                } else {
                    inp(inputs, 0).rshift(inp(inputs, 1))?
                }]
            }
            "BitwiseAnd" => vec![inp(inputs, 0).bitwise_and(inp(inputs, 1))?],
            "BitwiseOr" => vec![inp(inputs, 0).bitwise_or(inp(inputs, 1))?],
            "BitwiseXor" => vec![inp(inputs, 0).bitwise_xor(inp(inputs, 1))?],
            "BitwiseNot" => vec![inp(inputs, 0).bitwise_not()?],

            // === Math ===
            "Sqrt" => vec![inp(inputs, 0).try_sqrt()?],
            "Exp" => vec![inp(inputs, 0).try_exp()?],
            "Log" => vec![inp(inputs, 0).try_log()?],
            "Ceil" => vec![inp(inputs, 0).ceil()?],
            "Floor" => vec![inp(inputs, 0).floor()?],
            "Round" => vec![inp(inputs, 0).round()?],
            "Sign" => vec![inp(inputs, 0).sign()?],
            "Reciprocal" => vec![inp(inputs, 0).reciprocal()?],
            "Erf" => vec![inp(inputs, 0).erf()?],
            "Sin" => vec![inp(inputs, 0).sin()?],
            "Cos" => vec![inp(inputs, 0).cos()?],
            "Tan" => vec![inp(inputs, 0).tan()?],
            "Asin" => vec![inp(inputs, 0).asin()?],
            "Acos" => vec![inp(inputs, 0).acos()?],
            "Atan" => vec![inp(inputs, 0).atan()?],
            "Sinh" => vec![inp(inputs, 0).sinh()?],
            "Cosh" => vec![inp(inputs, 0).cosh()?],
            "Asinh" => vec![inp(inputs, 0).asinh()?],
            "Acosh" => vec![inp(inputs, 0).acosh()?],
            "Atanh" => vec![inp(inputs, 0).atanh()?],
            "Det" => vec![inp(inputs, 0).det()?],
            "IsNaN" => vec![inp(inputs, 0).isnan()?],
            "IsInf" => {
                let detect_negative = attrs.int("detect_negative", 1) == 1;
                let detect_positive = attrs.int("detect_positive", 1) == 1;
                vec![inp(inputs, 0).isinf(detect_positive, detect_negative)?]
            }

            // === Activation ===
            "Relu" => vec![inp(inputs, 0).relu()?],
            "Sigmoid" => vec![inp(inputs, 0).sigmoid()?],
            "Tanh" => vec![inp(inputs, 0).tanh()?],
            "Softmax" => {
                let default_axis = if opset_version < 13 { 1 } else { -1 };
                let axis = attrs.int("axis", default_axis) as isize;
                vec![inp(inputs, 0).softmax(axis)?]
            }
            "LogSoftmax" => {
                let default_axis = if opset_version < 13 { 1 } else { -1 };
                let axis = attrs.int("axis", default_axis) as isize;
                vec![inp(inputs, 0).log_softmax(axis)?]
            }
            "Gelu" => {
                let approximate = attrs.string("approximate", "none");
                vec![if approximate == "tanh" { inp(inputs, 0).gelu()? } else { inp(inputs, 0).gelu_exact()? }]
            }
            "HardSigmoid" => {
                let alpha = attrs.float("alpha", 0.2) as f64;
                let beta = attrs.float("beta", 0.5) as f64;
                vec![inp(inputs, 0).hard_sigmoid(alpha, beta)?]
            }
            "LeakyRelu" => {
                let alpha = attrs.float("alpha", 0.01) as f64;
                vec![inp(inputs, 0).leaky_relu(alpha)?]
            }
            "PRelu" => vec![inp(inputs, 0).prelu(inp(inputs, 1))?],
            "ThresholdedRelu" => {
                let alpha = attrs.float("alpha", 1.0) as f64;
                vec![inp(inputs, 0).thresholded_relu(alpha)?]
            }
            "Elu" => {
                let alpha = attrs.float("alpha", 1.0) as f64;
                vec![inp(inputs, 0).elu(alpha)?]
            }
            "Selu" => {
                let alpha = attrs.float("alpha", 1.6732632) as f64;
                let gamma = attrs.float("gamma", 1.050_701) as f64;
                vec![inp(inputs, 0).selu(alpha, gamma)?]
            }
            "Softplus" => {
                let beta = attrs.float("beta", 1.0) as f64;
                vec![inp(inputs, 0).softplus(beta)?]
            }
            "Mish" => vec![inp(inputs, 0).mish()?],
            "HardSwish" => vec![inp(inputs, 0).hardswish()?],
            "Softsign" => vec![inp(inputs, 0).softsign()?],
            "Celu" => {
                let alpha = attrs.float("alpha", 1.0) as f64;
                vec![inp(inputs, 0).celu(alpha)?]
            }

            // === Comparison ===
            "Equal" => vec![inp(inputs, 0).try_eq(inp(inputs, 1))?],
            "Less" => vec![inp(inputs, 0).try_lt(inp(inputs, 1))?],
            "LessOrEqual" => vec![inp(inputs, 0).try_le(inp(inputs, 1))?],
            "Greater" => vec![inp(inputs, 0).try_gt(inp(inputs, 1))?],
            "GreaterOrEqual" => vec![inp(inputs, 0).try_ge(inp(inputs, 1))?],
            "Not" => vec![inp(inputs, 0).logical_not()?],
            "And" => {
                let a = inp(inputs, 0).cast(DType::Bool)?;
                let b = inp(inputs, 1).cast(DType::Bool)?;
                vec![a.try_mul(&b)?]
            }
            "Or" => {
                let a = inp(inputs, 0).cast(DType::Bool)?;
                let b = inp(inputs, 1).cast(DType::Bool)?;
                // a | b = a + b - a*b
                let ab = a.try_mul(&b)?;
                vec![a.try_add(&b)?.try_sub(&ab)?]
            }
            "Xor" => {
                let x = inp(inputs, 0).cast(DType::Bool)?;
                let y = inp(inputs, 1).cast(DType::Bool)?;
                vec![x.bitwise_xor(&y)?]
            }

            // === Conditional ===
            "Where" => vec![inp(inputs, 1).where_(inp(inputs, 0), inp(inputs, 2))?],
            "Max" => vec![fold_variadic(inputs, "Max", |a, b| a.maximum(b))?],
            "Min" => vec![fold_variadic(inputs, "Min", |a, b| a.minimum(b))?],
            "Clip" => {
                let min = inputs.get(1).and_then(|o| o.as_ref());
                let max = inputs.get(2).and_then(|o| o.as_ref());
                vec![inp(inputs, 0).clamp().maybe_min(min).maybe_max(max).call()?]
            }

            // === Type ===
            "Cast" => {
                let to = attrs.int("to", 1);
                let _saturate = attrs.int("saturate", 1);
                let dtype = convert_onnx_dtype(to as i32).unwrap_or_else(|_| {
                    tracing::warn!("ONNX dtype {to} unsupported, falling back to Float32");
                    DType::Float32
                });
                vec![inp(inputs, 0).cast(dtype)?]
            }
            "CastLike" => {
                let _saturate = attrs.int("saturate", 1);
                vec![inp(inputs, 0).cast(inp(inputs, 1).uop().dtype())?]
            }
            "BitCast" => {
                let to = attrs.int("to", 1);
                let dtype = convert_onnx_dtype(to as i32)?;
                vec![inp(inputs, 0).bitcast(dtype)?]
            }

            // === Shape ===
            "Reshape" => vec![shape::op_reshape(inputs, &mut attrs)?],
            "Transpose" => vec![shape::op_transpose(inputs, &mut attrs)?],
            "Squeeze" => vec![shape::op_squeeze(inputs, &mut attrs, opset_version)?],
            "Unsqueeze" => vec![shape::op_unsqueeze(inputs, &mut attrs, opset_version)?],
            "Flatten" => vec![shape::op_flatten(inputs, &mut attrs)?],
            "Concat" => {
                let axis = attrs.int("axis", 0) as isize;
                let tensors: Vec<&Tensor> = inputs.iter().filter_map(|o| o.as_ref()).collect();
                vec![Tensor::cat(&tensors, axis)?]
            }
            "Shape" => {
                let start = attrs.int("start", 0) as isize;
                let end = attrs.int("end", i64::MAX) as isize;
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
                vec![Tensor::from_slice(&dims)]
            }
            "Expand" => vec![shape::op_expand(inputs)?],
            "Pad" => vec![shape::op_pad(inputs, &mut attrs)?],
            "Slice" => vec![shape::op_slice(inputs)?],
            "Split" => shape::op_split(inputs, &mut attrs, opset_version)?,
            "Tile" => {
                let repeats: Vec<usize> = tensor_to_i64_vec(inp(inputs, 1))?.iter().map(|&v| v as usize).collect();
                vec![inp(inputs, 0).repeat(&repeats)?]
            }
            "Range" => {
                let start_t = inp(inputs, 0);
                let out_dtype = start_t.uop().dtype();
                vec![if out_dtype.is_float() {
                    let start = tensor_to_f64_scalar(start_t)?;
                    let limit = tensor_to_f64_scalar(inp(inputs, 1))?;
                    let delta = tensor_to_f64_scalar(inp(inputs, 2))?;
                    Tensor::arange_f64(start, limit, delta, out_dtype)?
                } else {
                    let start = tensor_to_i64_vec(start_t)?[0];
                    let limit = tensor_to_i64_vec(inp(inputs, 1))?[0];
                    let delta = tensor_to_i64_vec(inp(inputs, 2))?[0];
                    Tensor::arange_with_dtype(start, Some(limit), Some(delta), out_dtype)?
                }]
            }
            "ConstantOfShape" => {
                let shape_i64 = tensor_to_i64_vec(inp(inputs, 0))?;
                let value = attrs
                    .tensor("value")
                    .map(tensor_from_proto)
                    .transpose()?
                    .unwrap_or_else(|| Tensor::from_slice([0.0f32]));
                vec![if shape_i64.contains(&0) {
                    Tensor::empty(value.uop().dtype())
                } else {
                    let shape: Vec<isize> = shape_i64.iter().map(|&v| v as isize).collect();
                    let ones = vec![1isize; shape.len()];
                    value.try_reshape(&ones)?.try_expand(&shape)?
                }]
            }
            "Size" => vec![Tensor::from_const(inp(inputs, 0).numel()? as i64)],
            "Dropout" => nn::op_dropout(inputs, &mut attrs, opset_version)?,

            // === Indexing ===
            "Gather" => {
                // ONNX Gather = np.take(data, indices, axis)
                let axis = attrs.int("axis", 0) as isize;
                let data = inp(inputs, 0);
                let idx = inp(inputs, 1);
                let data_shape = data.shape()?;
                let ndim = data_shape.len();
                let norm_axis = if axis < 0 { (ndim as isize + axis) as usize } else { axis as usize };
                let dim_size = data_shape[norm_axis].as_const().ok_or_else(|| Error::IrConstruction {
                    details: format!("Gather requires concrete dimension on axis {norm_axis}"),
                })? as i64;
                // Normalize negative indices
                let idx = idx.normalize_negative_indices(dim_size)?;
                // Flatten indices → index_select → reshape to insert indices shape
                let idx_shape = idx.shape()?;
                let idx_shape_concrete: Vec<usize> = idx_shape.iter().map(|d| d.as_const().unwrap()).collect();
                let flat_idx = idx.flatten()?;
                let selected = data.index_select(norm_axis as isize, &flat_idx)?;
                // Output shape: data[:axis] + idx_shape + data[axis+1:]
                let mut out_shape: Vec<isize> = Vec::new();
                for d in &data_shape[..norm_axis] {
                    out_shape.push(d.as_const().unwrap() as isize);
                }
                for &d in &idx_shape_concrete {
                    out_shape.push(d as isize);
                }
                for d in &data_shape[norm_axis + 1..] {
                    out_shape.push(d.as_const().unwrap() as isize);
                }
                vec![selected.try_reshape(&out_shape)?]
            }
            "GatherElements" => vec![indexing::op_gather_elements(inputs, &mut attrs)?],
            "GatherND" => vec![indexing::op_gather_nd(inputs, &mut attrs)?],
            "Trilu" => vec![indexing::op_trilu(inputs, &mut attrs)?],
            "OneHot" => vec![indexing::op_one_hot(inputs, &mut attrs)?],
            "CumSum" => vec![indexing::op_cumsum(inputs, &mut attrs)?],
            "CumProd" => vec![indexing::op_cumprod(inputs, &mut attrs)?],
            "Scatter" | "ScatterElements" => vec![indexing::op_scatter_elements(inputs, &mut attrs)?],
            "ScatterND" => vec![indexing::op_scatter_nd(inputs, &mut attrs)?],
            "TensorScatter" => vec![indexing::op_tensor_scatter(inputs, &mut attrs)?],
            "ReverseSequence" => {
                let batch_axis = attrs.int("batch_axis", 1) as usize;
                let time_axis = attrs.int("time_axis", 0) as usize;
                vec![inp(inputs, 0).reverse_sequence(inp(inputs, 1), time_axis, batch_axis)?]
            }

            // === Reductions ===
            "ReduceSum" => {
                let (spec, kd) = reduce_attrs(&mut attrs, inputs, opset_version, op_type)?;
                vec![inp(inputs, 0).sum_with().axes(spec).keepdim(kd).call()?]
            }
            "ReduceMean" => {
                let (spec, kd) = reduce_attrs(&mut attrs, inputs, opset_version, op_type)?;
                vec![inp(inputs, 0).mean_with().axes(spec).keepdim(kd).call()?]
            }
            "ReduceMax" => {
                let (spec, kd) = reduce_attrs(&mut attrs, inputs, opset_version, op_type)?;
                vec![inp(inputs, 0).max_with().axes(spec).keepdim(kd).call()?]
            }
            "ReduceMin" => {
                let (spec, kd) = reduce_attrs(&mut attrs, inputs, opset_version, op_type)?;
                vec![inp(inputs, 0).min_with().axes(spec).keepdim(kd).call()?]
            }
            "ReduceProd" => {
                let (spec, kd) = reduce_attrs(&mut attrs, inputs, opset_version, op_type)?;
                vec![inp(inputs, 0).prod_with().axes(spec).keepdim(kd).call()?]
            }
            "ReduceSumSquare" => {
                let (spec, kd) = reduce_attrs(&mut attrs, inputs, opset_version, op_type)?;
                vec![inp(inputs, 0).square()?.sum_with().axes(spec).keepdim(kd).call()?]
            }
            "ReduceL1" => {
                let (spec, kd) = reduce_attrs(&mut attrs, inputs, opset_version, op_type)?;
                vec![inp(inputs, 0).try_abs()?.sum_with().axes(spec).keepdim(kd).call()?]
            }
            "ReduceL2" => {
                let (spec, kd) = reduce_attrs(&mut attrs, inputs, opset_version, op_type)?;
                let x = inp(inputs, 0);
                let orig_dtype = x.uop().dtype();
                let needs_upcast =
                    matches!(orig_dtype.scalar(), Some(ScalarDType::Float16) | Some(ScalarDType::BFloat16));
                let x = if needs_upcast { x.cast(DType::Float32)? } else { x.clone() };
                let result = x.square()?.sum_with().axes(spec).keepdim(kd).call()?.try_sqrt()?;
                vec![if needs_upcast { result.cast(orig_dtype)? } else { result }]
            }
            "ReduceLogSum" => {
                let (spec, kd) = reduce_attrs(&mut attrs, inputs, opset_version, op_type)?;
                vec![inp(inputs, 0).sum_with().axes(spec).keepdim(kd).call()?.try_log()?]
            }
            "ReduceLogSumExp" => {
                let (spec, kd) = reduce_attrs(&mut attrs, inputs, opset_version, op_type)?;
                vec![inp(inputs, 0).try_exp()?.sum_with().axes(spec).keepdim(kd).call()?.try_log()?]
            }
            "ArgMax" => {
                let axis = attrs.int("axis", 0) as isize;
                let keepdims = attrs.int("keepdims", 1) == 1;
                let select_last = attrs.int("select_last_index", 0) == 1;
                let x = inp(inputs, 0);
                vec![if select_last {
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
                }]
            }
            "ArgMin" => {
                let axis = attrs.int("axis", 0) as isize;
                let keepdims = attrs.int("keepdims", 1) == 1;
                let select_last = attrs.int("select_last_index", 0) == 1;
                let x = inp(inputs, 0);
                let neg_x = x.try_neg()?;
                vec![if select_last {
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
                }]
            }

            // === NN ===
            "MatMul" => vec![inp(inputs, 0).matmul(inp(inputs, 1))?],
            "MatMulInteger" => {
                let a = inp(inputs, 0).cast(DType::Int32)?;
                let b = inp(inputs, 1).cast(DType::Int32)?;
                let a_zp = inputs.get(2).and_then(|o| o.as_ref()).map(|t| t.cast(DType::Int32)).transpose()?;
                let b_zp = inputs.get(3).and_then(|o| o.as_ref()).map(|t| t.cast(DType::Int32)).transpose()?;
                let a = if let Some(zp) = a_zp { a.try_sub(&zp)? } else { a };
                let b = if let Some(zp) = b_zp { b.try_sub(&zp)? } else { b };
                vec![a.matmul(&b)?]
            }
            "Gemm" => vec![nn::op_gemm(inputs, &mut attrs)?],
            "BatchNormalization" => nn::op_batch_norm(inputs, &mut attrs)?,
            "RNN" => {
                let hidden_size = attrs.int("hidden_size", 0) as usize;
                let layout = attrs.int("layout", 0) as usize;
                let _direction = attrs.string("direction", "forward");
                let _activations = attrs.floats("activation_alpha");
                let _activations_beta = attrs.floats("activation_beta");
                let _activations_list = attrs.ints("activations"); // consume but ignore
                let _clip = attrs.float("clip", 0.0);
                let out = inp(inputs, 0)
                    .rnn()
                    .w(inp(inputs, 1))
                    .r(inp(inputs, 2))
                    .hidden_size(hidden_size)
                    .maybe_bias(inputs.get(3).and_then(|o| o.as_ref()))
                    .maybe_initial_h(inputs.get(5).and_then(|o| o.as_ref()))
                    .layout(layout)
                    .call()?;
                vec![out.y, out.y_h]
            }
            "GRU" => {
                let hidden_size = attrs.int("hidden_size", 0) as usize;
                let layout = attrs.int("layout", 0) as usize;
                let linear_before_reset = attrs.int("linear_before_reset", 0) as usize;
                let _direction = attrs.string("direction", "forward");
                let _activations = attrs.floats("activation_alpha");
                let _activations_beta = attrs.floats("activation_beta");
                let _activations_list = attrs.ints("activations");
                let _clip = attrs.float("clip", 0.0);
                let out = inp(inputs, 0)
                    .gru()
                    .w(inp(inputs, 1))
                    .r_weights(inp(inputs, 2))
                    .hidden_size(hidden_size)
                    .maybe_bias(inputs.get(3).and_then(|o| o.as_ref()))
                    .maybe_initial_h(inputs.get(5).and_then(|o| o.as_ref()))
                    .linear_before_reset(linear_before_reset)
                    .layout(layout)
                    .call()?;
                vec![out.y, out.y_h]
            }
            "LSTM" => {
                let hidden_size = attrs.int("hidden_size", 0) as usize;
                let layout = attrs.int("layout", 0) as usize;
                let _direction = attrs.string("direction", "forward");
                let _activations = attrs.floats("activation_alpha");
                let _activations_beta = attrs.floats("activation_beta");
                let _activations_list = attrs.ints("activations");
                let _clip = attrs.float("clip", 0.0);
                let _input_forget = attrs.int("input_forget", 0);
                let out = inp(inputs, 0)
                    .lstm()
                    .w(inp(inputs, 1))
                    .r(inp(inputs, 2))
                    .hidden_size(hidden_size)
                    .maybe_bias(inputs.get(3).and_then(|o| o.as_ref()))
                    .maybe_initial_h(inputs.get(5).and_then(|o| o.as_ref()))
                    .maybe_initial_c(inputs.get(6).and_then(|o| o.as_ref()))
                    .maybe_peepholes(inputs.get(7).and_then(|o| o.as_ref()))
                    .layout(layout)
                    .call()?;
                vec![out.y, out.y_h, out.y_c]
            }
            "Conv" => vec![nn::op_conv(inputs, &mut attrs)?],
            "ConvTranspose" => vec![nn::op_conv_transpose(inputs, &mut attrs)?],
            "QLinearConv" => vec![nn::op_qlinear_conv(inputs, &mut attrs)?],
            "QLinearMatMul" => vec![nn::op_qlinear_matmul(inputs, &mut attrs)?],
            "ConvInteger" => vec![nn::op_conv_integer(inputs, &mut attrs)?],
            "AveragePool" => vec![nn::op_avg_pool(inputs, &mut attrs)?],
            "LpPool" => vec![nn::op_lp_pool(inputs, &mut attrs)?],
            "MaxPool" => nn::op_max_pool(inputs, &mut attrs)?,
            "MaxUnpool" => vec![nn::op_max_unpool(inputs, &mut attrs)?],
            "Col2Im" => vec![nn::op_col2im(inputs, &mut attrs)?],
            "GlobalAveragePool" => {
                let x = inp(inputs, 0);
                let axes: Vec<isize> = (2..x.ndim()? as isize).collect();
                vec![x.mean_with().axes(AxisSpec::Multiple(axes)).keepdim(true).call()?]
            }
            "GlobalLpPool" => {
                let x = inp(inputs, 0);
                let p = attrs.int("p", 2) as usize;
                let shape = x.shape()?;
                let kernel: Vec<usize> = shape[2..].iter().map(|s| s.as_const().unwrap()).collect();
                vec![x.lp_pool().kernel_shape(&kernel).p(p).call()?]
            }
            "GlobalMaxPool" => {
                let x = inp(inputs, 0);
                let axes: Vec<isize> = (2..x.ndim()? as isize).collect();
                vec![x.max_with().axes(AxisSpec::Multiple(axes)).keepdim(true).call()?]
            }
            "LayerNormalization" => nn::op_layer_norm(inputs, &mut attrs)?,
            "GroupNormalization" => vec![nn::op_group_norm(inputs, &mut attrs)?],
            "InstanceNormalization" => vec![nn::op_instance_norm(inputs, &mut attrs)?],
            "Resize" => vec![nn::op_resize(inputs, &mut attrs)?],
            "DepthToSpace" => vec![nn::op_depth_to_space(inputs, &mut attrs)?],
            "SpaceToDepth" => vec![nn::op_space_to_depth(inputs, &mut attrs)?],
            "LpNormalization" => vec![nn::op_lp_norm(inputs, &mut attrs)?],
            "MeanVarianceNormalization" => vec![nn::op_mean_variance_norm(inputs, &mut attrs)?],
            "LRN" => vec![nn::op_lrn(inputs, &mut attrs)?],
            "AffineGrid" => vec![nn::op_affine_grid(inputs, &mut attrs)?],
            "GridSample" => vec![nn::op_grid_sample(inputs, &mut attrs)?],
            "NegativeLogLikelihoodLoss" => nn::op_nll_loss(inputs, &mut attrs)?,
            "SoftmaxCrossEntropyLoss" => nn::op_softmax_ce_loss(inputs, &mut attrs)?,
            "RMSNormalization" => vec![transformer::op_rms_norm(inputs, &mut attrs)?],
            "Attention" => transformer::op_attention_onnx(inputs, &mut attrs)?,
            "RotaryEmbedding" => transformer::op_rotary_embedding(inputs, &mut attrs)?,

            // === NonZero ===
            "NonZero" => vec![inp(inputs, 0).nonzero()?.try_transpose(0, 1)?.cast(DType::Int64)?],

            // === Einsum ===
            "Einsum" => {
                let equation = attrs.string("equation", "");
                let ops: Vec<&Tensor> = inputs.iter().filter_map(|o| o.as_ref()).collect();
                vec![Tensor::einsum(&equation, &ops)?]
            }

            // === TopK ===
            "TopK" => {
                let k = tensor_to_i64_vec(inp(inputs, 1))?[0] as usize;
                let axis = attrs.int("axis", -1) as isize;
                let largest = attrs.int("largest", 1) == 1;
                let _sorted = attrs.int("sorted", 1); // consume; we always sort
                let (values, indices) = inp(inputs, 0).topk(k, axis, largest)?;
                vec![values, indices.cast(DType::Int64)?]
            }

            // === Simple Ops ===
            "Hardmax" => {
                let axis = attrs.int("axis", -1) as isize;
                vec![inp(inputs, 0).hardmax(axis)?]
            }
            "Binarizer" => {
                let threshold = attrs.float("threshold", 0.0) as f64;
                let x = inp(inputs, 0);
                vec![x.try_gt(&Tensor::const_(threshold, x.uop().dtype()))?.cast(DType::Float32)?]
            }
            "Swish" => {
                let alpha = attrs.float("alpha", 1.0) as f64;
                let x = inp(inputs, 0);
                let alpha_t = Tensor::const_(alpha, x.uop().dtype());
                vec![x.try_mul(&x.try_mul(&alpha_t)?.sigmoid()?)?]
            }
            "BiasGelu" => {
                let xb = inp(inputs, 0).try_add(inp(inputs, 1))?;
                let approximate = attrs.string("approximate", "none");
                vec![if approximate == "tanh" { xb.gelu()? } else { xb.gelu_exact()? }]
            }
            "FastGelu" => {
                let x = inp(inputs, 0);
                let xb =
                    if let Some(bias) = inputs.get(1).and_then(|o| o.as_ref()) { x.try_add(bias)? } else { x.clone() };
                vec![xb.gelu()?]
            }
            "Shrink" => {
                let bias = attrs.float("bias", 0.0) as f64;
                let lambd = attrs.float("lambd", 0.5) as f64;
                vec![inp(inputs, 0).shrink(bias, lambd)?]
            }
            "EyeLike" => {
                let x = inp(inputs, 0);
                let x_shape = x.shape()?;
                let dtype =
                    if let Some(dt) = attrs.get("dtype") { convert_onnx_dtype(dt.i as i32)? } else { x.uop().dtype() };
                let k = attrs.int("k", 0);
                let h = x_shape[0]
                    .as_const()
                    .ok_or_else(|| Error::IrConstruction { details: "EyeLike requires concrete shape".into() })?;
                let w = x_shape[1]
                    .as_const()
                    .ok_or_else(|| Error::IrConstruction { details: "EyeLike requires concrete shape".into() })?;
                let eye_size = h.min(w);
                let mut eye = Tensor::eye(eye_size, eye_size, dtype)?;
                if h != w || k != 0 {
                    let k = k as isize;
                    let pad_top = if k < 0 { -k } else { 0 };
                    let pad_left = if k > 0 { k } else { 0 };
                    let pad_bottom = h as isize - eye_size as isize - pad_top;
                    let pad_right = w as isize - eye_size as isize - pad_left;
                    eye = eye.try_pad(&[(pad_top, pad_bottom), (pad_left, pad_right)])?;
                }
                vec![eye]
            }
            "OptionalHasElement" => {
                let has = inputs.first().and_then(|o| o.as_ref()).is_some_and(|t| t.numel().unwrap_or(0) > 0);
                vec![Tensor::from_const(has)]
            }
            "OptionalGetElement" => match inputs.first().and_then(|o| o.as_ref()) {
                Some(t) => vec![t.clone()],
                None => vec![Tensor::empty(DType::Float32)],
            },
            "CenterCropPad" => {
                let t = inp(inputs, 0);
                let target_shape = tensor_to_i64_vec(inp(inputs, 1))?;
                let target: Vec<usize> = target_shape.iter().map(|&v| v as usize).collect();
                let axes: Option<Vec<usize>> = attrs.get("axes").map(|a| {
                    a.ints
                        .iter()
                        .map(|&v| if v < 0 { (t.ndim().unwrap() as i64 + v) as usize } else { v as usize })
                        .collect()
                });
                vec![t.center_crop_pad(&target, axes.as_deref())?]
            }
            "Compress" => {
                let cond_vals = tensor_to_i64_vec(inp(inputs, 1))?;
                let condition: Vec<bool> = cond_vals.iter().map(|&v| v != 0).collect();
                let axis = attrs.get("axis").map(|a| a.i as isize);
                vec![inp(inputs, 0).compress(&condition, axis)?]
            }
            "Upsample" => {
                // Upsample (deprecated) has inputs [X, scales]; Resize has [X, roi, scales, sizes].
                // Remap by inserting None for roi so op_resize reads scales from index 2.
                let remapped = vec![inputs.first().cloned().flatten(), None, inputs.get(1).cloned().flatten()];
                vec![nn::op_resize(&remapped, &mut attrs)?]
            }

            // === Identity / Constant ===
            "Identity" => vec![inp(inputs, 0).clone()],
            "Constant" => vec![constant::op_constant(&mut attrs)?],

            _ => return UnsupportedOpSnafu { op: op_type.to_string(), domain: domain.to_string() }.fail(),
        };

        attrs.done()?;
        Ok(results)
    }
}

impl Default for OpRegistry {
    fn default() -> Self {
        Self::new()
    }
}
