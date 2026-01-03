//! CPU-specific operation code generation.
//!
//! This module generates LLVM IR for individual UOp operations.
//! Operations are organized by category for clarity.

use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::intrinsics::Intrinsic;
use inkwell::module::Module;
use inkwell::values::{BasicValueEnum, CallSiteValue, FunctionValue, ValueKind};
use inkwell::{FloatPredicate, IntPredicate};
use snafu::{OptionExt, ResultExt};
use std::sync::Arc;
use tracing::{debug, trace};

use morok_dtype::DType;
use morok_ir::{AxisType, ReduceOp, prelude::*};

use crate::llvm::common::{self, loop_gen};
use crate::llvm::error::*;
use crate::llvm::helpers::ValueMap;

// ============================================================================
// Main Entry Point
// ============================================================================

/// Generate LLVM IR for a UOp node.
pub fn codegen_uop<'ctx>(
    uop: &Arc<UOp>,
    context: &'ctx Context,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
    values: &mut ValueMap<'ctx>,
) -> Result<Option<BasicValueEnum<'ctx>>> {
    // Range(Reduce) uses axis_id as the canonical identifier, not UOp ID.
    // This is because graph_rewrite may create new Range nodes with different UOp IDs
    // but the same axis_id. See: ir/src/uop/hash_consing.rs for why this happens.
    // codegen_reduce creates the loop structure and inserts by axis_id.
    if let Op::Range { axis_type: AxisType::Reduce, axis_id, .. } = uop.op() {
        if let Some(val) = values.get_range_by_axis(axis_id.value()) {
            return Ok(Some(val));
        }
        // Not yet generated - codegen_reduce will create it when processing REDUCE op
        return Ok(None);
    }

    // Check if already generated (for non-Range ops)
    if values.contains(uop.id) {
        trace!(uop_id = uop.id, op = ?std::mem::discriminant(uop.op()), "codegen_uop: cache hit");
        return Ok(values.get(uop.id));
    }

    let category = classify_op(uop.op());
    trace!(
        uop_id = uop.id,
        op = ?uop.op(),
        dtype = ?uop.dtype(),
        category = ?std::mem::discriminant(&category),
        "codegen_uop: generating"
    );

    let result = match category {
        OpCategory::Constant => codegen_constant(uop, context)?,
        OpCategory::Arithmetic => codegen_arithmetic(uop, context, module, builder, values)?,
        OpCategory::Memory => codegen_memory(uop, context, module, builder, values)?,
        OpCategory::Loop => codegen_loop(uop, context, module, builder, values)?,
        OpCategory::Meta => None,
        OpCategory::Unsupported(what) => return UnsupportedSnafu { what }.fail(),
    };

    // Store result for future lookups
    if let Some(val) = result {
        trace!(uop_id = uop.id, result = ?val, "codegen_uop: stored result");
        values.insert(uop.id, val);
    } else {
        values.mark_processed(uop.id);
    }

    Ok(result)
}

// ============================================================================
// Operation Classification
// ============================================================================

enum OpCategory {
    Constant,
    Arithmetic,
    Memory,
    Loop,
    Meta,
    Unsupported(&'static str),
}

fn classify_op(op: &Op) -> OpCategory {
    match op {
        Op::Const(_) | Op::VConst { .. } => OpCategory::Constant,

        Op::Unary(..) | Op::Binary(..) | Op::Ternary(..) | Op::Cast { .. } => OpCategory::Arithmetic,

        // Contract/Unroll are metadata wrappers from pre_expand pass - pass through to source
        Op::Contract { .. } | Op::Unroll { .. } => OpCategory::Arithmetic,

        // Vector element operations
        Op::Gep { .. } | Op::Cat { .. } | Op::PtrCat { .. } => OpCategory::Arithmetic,

        Op::Load { .. } | Op::Store { .. } | Op::Index { .. } | Op::PointerIndex { .. } => OpCategory::Memory,

        Op::Range { .. } | Op::End { .. } | Op::Reduce { .. } | Op::Bind { .. } => OpCategory::Loop,

        // Buffer/DefineGlobal/DefineLocal should already be in ValueMap from renderer
        Op::Buffer { .. } | Op::DefineGlobal(_) | Op::DefineLocal(_) => OpCategory::Meta,

        // DefineVar: For inlined outer loops, value comes from Bind (loop counter)
        // Should be in ValueMap after Bind is processed. If not there, it's an error.
        Op::DefineVar { .. } => OpCategory::Meta,

        Op::Sink { .. } | Op::Barrier { .. } | Op::Noop | Op::Unique(_) | Op::Device(_) | Op::Kernel { .. } => {
            OpCategory::Meta
        }

        // Vectorize creates vectors from elements - process in Loop handler
        Op::Vectorize { .. } => OpCategory::Loop,

        Op::Reshape { .. }
        | Op::Permute { .. }
        | Op::Expand { .. }
        | Op::Pad { .. }
        | Op::Shrink { .. }
        | Op::Flip { .. } => OpCategory::Unsupported("Movement ops must be eliminated by rangeify"),

        Op::ReduceAxis { .. } => OpCategory::Unsupported("ReduceAxis must be converted to Reduce by rangeify"),

        _ => OpCategory::Unsupported("Unknown UOp operation"),
    }
}

// ============================================================================
// Constant Operations
// ============================================================================

fn codegen_constant<'ctx>(uop: &Arc<UOp>, context: &'ctx Context) -> Result<Option<BasicValueEnum<'ctx>>> {
    match uop.op() {
        Op::Const(val_hash) => {
            let dtype = uop.dtype();
            let llvm_type = common::dtype_to_basic_type(&dtype, context)?;

            let value = match val_hash.0 {
                ConstValue::Int(i) => llvm_type.into_int_type().const_int(i as u64, true).into(),
                ConstValue::UInt(u) => llvm_type.into_int_type().const_int(u, false).into(),
                ConstValue::Float(f) => llvm_type.into_float_type().const_float(f).into(),
                ConstValue::Bool(b) => context.bool_type().const_int(b as u64, false).into(),
            };
            Ok(Some(value))
        }
        Op::VConst { values } => {
            use inkwell::types::VectorType;

            if values.is_empty() {
                return Ok(None);
            }

            // Get scalar type from UOp's dtype (which should be a Vector type)
            let dtype = uop.dtype();
            let scalar_dtype = dtype.base();
            let scalar_llvm = common::scalar_to_basic_type(scalar_dtype, context)?;

            // Create vector constant based on scalar type from dtype
            match scalar_llvm {
                inkwell::types::BasicTypeEnum::IntType(it) => {
                    let llvm_values: Vec<_> = values
                        .iter()
                        .map(|v| match v {
                            ConstValue::Int(i) => it.const_int(*i as u64, scalar_dtype.is_signed()),
                            ConstValue::UInt(u) => it.const_int(*u, false),
                            ConstValue::Bool(b) => it.const_int(*b as u64, false),
                            ConstValue::Float(f) => it.const_int(*f as u64, false),
                        })
                        .collect();
                    Ok(Some(VectorType::const_vector(&llvm_values).into()))
                }
                inkwell::types::BasicTypeEnum::FloatType(ft) => {
                    let llvm_values: Vec<_> = values
                        .iter()
                        .map(|v| match v {
                            ConstValue::Float(f) => ft.const_float(*f),
                            ConstValue::Int(i) => ft.const_float(*i as f64),
                            ConstValue::UInt(u) => ft.const_float(*u as f64),
                            ConstValue::Bool(b) => ft.const_float(if *b { 1.0 } else { 0.0 }),
                        })
                        .collect();
                    Ok(Some(VectorType::const_vector(&llvm_values).into()))
                }
                _ => {
                    // Fallback for unsupported types
                    Ok(None)
                }
            }
        }
        _ => Ok(None),
    }
}

// ============================================================================
// Arithmetic Operations
// ============================================================================

fn codegen_arithmetic<'ctx>(
    uop: &Arc<UOp>,
    context: &'ctx Context,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
    values: &mut ValueMap<'ctx>,
) -> Result<Option<BasicValueEnum<'ctx>>> {
    match uop.op() {
        Op::Unary(op, src) => {
            let src_val = require_value(src, context, module, builder, values)?;
            let src_val = auto_load_pointer(src_val, &src.dtype(), context, builder)?;
            Ok(Some(codegen_unary(*op, src_val, &uop.dtype(), context, module, builder)?))
        }
        Op::Binary(op, lhs, rhs) => {
            let lhs_val_raw = require_value(lhs, context, module, builder, values)?;
            let rhs_val_raw = require_value(rhs, context, module, builder, values)?;
            let lhs_val = auto_load_pointer(lhs_val_raw, &lhs.dtype(), context, builder)?;
            let rhs_val = auto_load_pointer(rhs_val_raw, &rhs.dtype(), context, builder)?;

            // Harmonize operands: broadcast scalar to match vector width if needed
            let (lhs_val, rhs_val) = common::harmonize_operands(builder, lhs_val, rhs_val, context)?;

            debug!(
                uop_id = uop.id,
                op = ?op,
                lhs_id = lhs.id,
                rhs_id = rhs.id,
                lhs_dtype = ?lhs.dtype(),
                rhs_dtype = ?rhs.dtype(),
                lhs_val_raw = ?lhs_val_raw,
                rhs_val_raw = ?rhs_val_raw,
                lhs_val_loaded = ?lhs_val,
                rhs_val_loaded = ?rhs_val,
                "BINARY: lhs op rhs"
            );
            // Pass operand dtype for comparisons (like Tinygrad's lop[x.src[0].dtype])
            // For Ptr types, extract the base type since auto_load_pointer already loaded the value
            let operand_dtype = DType::Scalar(lhs.dtype().base());
            Ok(Some(codegen_binary(*op, lhs_val, rhs_val, &operand_dtype, &uop.dtype(), module, builder)?))
        }
        Op::Ternary(op, a, b, c) => {
            let a_val = require_value(a, context, module, builder, values)?;
            let b_val = require_value(b, context, module, builder, values)?;
            let c_val = require_value(c, context, module, builder, values)?;
            let a_val = auto_load_pointer(a_val, &a.dtype(), context, builder)?;
            let b_val = auto_load_pointer(b_val, &b.dtype(), context, builder)?;
            let c_val = auto_load_pointer(c_val, &c.dtype(), context, builder)?;

            // Harmonize all three operands to the maximum vector width
            let max_count = [a_val, b_val, c_val].iter().map(|v| common::get_vector_count(*v)).max().unwrap_or(1);

            let a_val = if max_count > 1 && common::get_vector_count(a_val) == 1 {
                common::broadcast_to_vector(builder, a_val, max_count, context)?
            } else {
                a_val
            };
            let b_val = if max_count > 1 && common::get_vector_count(b_val) == 1 {
                common::broadcast_to_vector(builder, b_val, max_count, context)?
            } else {
                b_val
            };
            let c_val = if max_count > 1 && common::get_vector_count(c_val) == 1 {
                common::broadcast_to_vector(builder, c_val, max_count, context)?
            } else {
                c_val
            };

            Ok(Some(codegen_ternary(*op, a_val, b_val, c_val, &uop.dtype(), module, builder)?))
        }
        Op::Cast { src, dtype } => {
            let src_val = require_value(src, context, module, builder, values)?;
            let src_val = auto_load_pointer(src_val, &src.dtype(), context, builder)?;
            // Get actual source dtype after auto-load (dereference Ptr)
            let actual_src_dtype = match src.dtype() {
                DType::Ptr { base, .. } => base.as_ref().clone(),
                other => other,
            };
            Ok(Some(codegen_cast(src_val, &actual_src_dtype, dtype, context, builder)?))
        }
        // Contract/Unroll are metadata wrappers from pre_expand pass
        // They document unrolled axes but don't affect code generation - pass through to source
        Op::Contract { src, .. } | Op::Unroll { src, .. } => {
            require_value(src, context, module, builder, values).map(Some)
        }
        // GEP extracts element(s) from a vector
        Op::Gep { vector, indices } => {
            let vec_val = require_value(vector, context, module, builder, values)?;
            let vec_val = auto_load_pointer(vec_val, &vector.dtype(), context, builder)?;

            if !vec_val.is_vector_value() {
                // Not a vector - just pass through
                return Ok(Some(vec_val));
            }

            if indices.len() == 1 {
                // Single index: extractelement
                let idx = context.i32_type().const_int(indices[0] as u64, false);
                let extracted = builder
                    .build_extract_element(vec_val.into_vector_value(), idx, "gep")
                    .context(VectorExtractSnafu)?;
                Ok(Some(extracted))
            } else {
                // Multiple indices: build new vector with selected elements
                let result_count = indices.len() as u32;
                let vec = vec_val.into_vector_value();
                let elem_type = vec.get_type().get_element_type();
                let result_type = match elem_type {
                    inkwell::types::BasicTypeEnum::IntType(it) => it.vec_type(result_count),
                    inkwell::types::BasicTypeEnum::FloatType(ft) => ft.vec_type(result_count),
                    _ => return Ok(Some(vec_val)),
                };

                let mut result: BasicValueEnum = result_type.get_poison().into();
                for (i, &src_idx) in indices.iter().enumerate() {
                    let extract_idx = context.i32_type().const_int(src_idx as u64, false);
                    let extracted =
                        builder.build_extract_element(vec, extract_idx, "gep_elem").context(VectorExtractSnafu)?;
                    let insert_idx = context.i32_type().const_int(i as u64, false);
                    result = builder
                        .build_insert_element(result.into_vector_value(), extracted, insert_idx, "gep_build")
                        .context(VectorInsertSnafu)?
                        .into();
                }
                Ok(Some(result))
            }
        }
        // Cat concatenates vectors into a larger vector
        Op::Cat { sources } | Op::PtrCat { sources } => {
            // Flatten all source elements into a single vector
            let mut all_elements: Vec<BasicValueEnum> = Vec::new();
            for src in sources.iter() {
                let val = require_value(src, context, module, builder, values)?;
                let val = auto_load_pointer(val, &src.dtype(), context, builder)?;

                if val.is_vector_value() {
                    // Extract all elements from vector
                    let vec = val.into_vector_value();
                    let count = vec.get_type().get_size();
                    for i in 0..count {
                        let idx = context.i32_type().const_int(i as u64, false);
                        let elem =
                            builder.build_extract_element(vec, idx, "cat_extract").context(VectorExtractSnafu)?;
                        all_elements.push(elem);
                    }
                } else {
                    all_elements.push(val);
                }
            }

            if all_elements.is_empty() {
                return Ok(None);
            }

            if all_elements.len() == 1 {
                return Ok(Some(all_elements[0]));
            }

            // Build result vector
            let count = all_elements.len() as u32;
            let vec_type = if all_elements[0].is_float_value() {
                all_elements[0].into_float_value().get_type().vec_type(count)
            } else if all_elements[0].is_int_value() {
                all_elements[0].into_int_value().get_type().vec_type(count)
            } else {
                return Ok(Some(all_elements[0]));
            };

            let mut result: BasicValueEnum = vec_type.get_poison().into();
            for (i, elem) in all_elements.into_iter().enumerate() {
                let idx = context.i32_type().const_int(i as u64, false);
                result = builder
                    .build_insert_element(result.into_vector_value(), elem, idx, "cat_build")
                    .context(VectorInsertSnafu)?
                    .into();
            }
            Ok(Some(result))
        }
        _ => Ok(None),
    }
}

fn codegen_unary<'ctx>(
    op: UnaryOp,
    src: BasicValueEnum<'ctx>,
    result_dtype: &DType,
    context: &'ctx Context,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
) -> Result<BasicValueEnum<'ctx>> {
    let is_float = result_dtype.is_float();

    // Try simple intrinsic lookup first
    if let Some((intrinsic_base, name)) = common::unary_float_intrinsic(op) {
        let suffix = get_type_suffix(result_dtype)?;
        return call_intrinsic(&format!("{}.{}", intrinsic_base, suffix), &[src], name, module, builder);
    }

    match op {
        UnaryOp::Neg => common::build_neg(builder, src, is_float),
        UnaryOp::Square => common::build_mul(builder, src, src, is_float),
        UnaryOp::Abs => codegen_abs(src, result_dtype, context, module, builder),
        UnaryOp::Rsqrt => codegen_rsqrt(src, result_dtype, module, builder),
        UnaryOp::Tan => codegen_tan(src, result_dtype, module, builder),
        UnaryOp::Reciprocal => codegen_reciprocal(src, builder),
        UnaryOp::Trunc => codegen_trunc(src, result_dtype, is_float, module, builder),
        UnaryOp::Sign => codegen_sign(src, is_float, builder),
        UnaryOp::Erf => codegen_erf(src, module, builder),
        _ => UnsupportedSnafu { what: "Unknown unary operation" }.fail(),
    }
}

fn codegen_binary<'ctx>(
    op: BinaryOp,
    lhs: BasicValueEnum<'ctx>,
    rhs: BasicValueEnum<'ctx>,
    operand_dtype: &DType,
    result_dtype: &DType,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
) -> Result<BasicValueEnum<'ctx>> {
    // For comparisons, use operand dtype (like Tinygrad's lop[x.src[0].dtype])
    // For arithmetic ops, use result dtype
    if common::is_comparison(op) {
        let is_float = operand_dtype.is_float();
        let is_signed = operand_dtype.is_signed();
        return common::build_cmp(builder, op, lhs, rhs, is_float, is_signed);
    }

    let is_float = result_dtype.is_float();
    let is_signed = result_dtype.is_signed();

    match op {
        BinaryOp::Add => common::build_add(builder, lhs, rhs, is_float),
        BinaryOp::Sub => common::build_sub(builder, lhs, rhs, is_float),
        BinaryOp::Mul => common::build_mul(builder, lhs, rhs, is_float),
        BinaryOp::Fdiv => {
            let result: BasicValueEnum<'ctx> = if lhs.is_vector_value() {
                builder
                    .build_float_div(lhs.into_vector_value(), rhs.into_vector_value(), "fdiv")
                    .context(ArithmeticSnafu)?
                    .into()
            } else {
                builder
                    .build_float_div(lhs.into_float_value(), rhs.into_float_value(), "fdiv")
                    .context(ArithmeticSnafu)?
                    .into()
            };
            common::fast_math::apply_fast_math_flags(result);
            Ok(result)
        }
        BinaryOp::Idiv => common::build_int_div(builder, lhs, rhs, is_signed),
        BinaryOp::Mod => common::build_rem(builder, lhs, rhs, is_float, is_signed),
        BinaryOp::Max => {
            let bits = result_dtype.bytes() * 8;
            let intrinsic = common::max_intrinsic(is_float, is_signed, bits);
            call_intrinsic(&intrinsic, &[lhs, rhs], "max", module, builder)
        }
        BinaryOp::Pow => {
            let suffix = get_type_suffix(result_dtype)?;
            call_intrinsic(&format!("llvm.pow.{}", suffix), &[lhs, rhs], "pow", module, builder)
        }
        BinaryOp::And => {
            Ok(builder.build_and(lhs.into_int_value(), rhs.into_int_value(), "and").context(ArithmeticSnafu)?.into())
        }
        BinaryOp::Or => {
            Ok(builder.build_or(lhs.into_int_value(), rhs.into_int_value(), "or").context(ArithmeticSnafu)?.into())
        }
        BinaryOp::Xor => {
            Ok(builder.build_xor(lhs.into_int_value(), rhs.into_int_value(), "xor").context(ArithmeticSnafu)?.into())
        }
        BinaryOp::Shl => Ok(builder
            .build_left_shift(lhs.into_int_value(), rhs.into_int_value(), "shl")
            .context(ArithmeticSnafu)?
            .into()),
        BinaryOp::Shr => Ok(builder
            .build_right_shift(lhs.into_int_value(), rhs.into_int_value(), is_signed, "shr")
            .context(ArithmeticSnafu)?
            .into()),
        _ => UnsupportedSnafu { what: "Unknown binary operation" }.fail(),
    }
}

fn codegen_ternary<'ctx>(
    op: TernaryOp,
    a: BasicValueEnum<'ctx>,
    b: BasicValueEnum<'ctx>,
    c: BasicValueEnum<'ctx>,
    result_dtype: &DType,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
) -> Result<BasicValueEnum<'ctx>> {
    match op {
        TernaryOp::Where => Ok(builder.build_select(a.into_int_value(), b, c, "where").context(BuildSelectSnafu)?),
        TernaryOp::MulAcc => {
            let suffix = get_type_suffix(result_dtype)?;
            call_intrinsic(&format!("llvm.fma.{}", suffix), &[a, b, c], "fma", module, builder)
        }
    }
}

fn codegen_cast<'ctx>(
    src: BasicValueEnum<'ctx>,
    src_dtype: &DType,
    dst_dtype: &DType,
    context: &'ctx Context,
    builder: &Builder<'ctx>,
) -> Result<BasicValueEnum<'ctx>> {
    // Extract scalar dtype from vector types for type checks
    let scalar_src_dtype = match src_dtype {
        DType::Vector { scalar, .. } => DType::Scalar(*scalar),
        _ => src_dtype.clone(),
    };
    let scalar_dst_dtype = DType::Scalar(dst_dtype.base());

    let src_is_float = scalar_src_dtype.is_float();
    let dst_is_float = scalar_dst_dtype.is_float();
    let src_is_signed = scalar_src_dtype.is_signed();
    let dst_is_signed = scalar_dst_dtype.is_signed();

    // Handle vector types
    if src.is_vector_value() {
        let vec = src.into_vector_value();
        let vec_len = vec.get_type().get_size();

        // Get scalar destination type
        let scalar_dst_type = common::dtype_to_basic_type(&scalar_dst_dtype, context)?;

        // Build vector type for destination
        let dst_vec_type = match scalar_dst_type {
            inkwell::types::BasicTypeEnum::IntType(it) => it.vec_type(vec_len),
            inkwell::types::BasicTypeEnum::FloatType(ft) => ft.vec_type(vec_len),
            _ => return UnsupportedSnafu { what: "Unsupported cast destination type for vector" }.fail(),
        };

        // Element-wise cast
        let mut result: BasicValueEnum = dst_vec_type.get_poison().into();
        for i in 0..vec_len {
            let idx = context.i32_type().const_int(i as u64, false);
            let elem = builder.build_extract_element(vec, idx, "cast_extract").context(VectorExtractSnafu)?;

            // Cast the scalar element
            let casted = codegen_cast_scalar(
                elem,
                src_is_float,
                dst_is_float,
                src_is_signed,
                dst_is_signed,
                &scalar_src_dtype,
                &scalar_dst_dtype,
                context,
                builder,
            )?;

            result = builder
                .build_insert_element(result.into_vector_value(), casted, idx, "cast_insert")
                .context(VectorInsertSnafu)?
                .into();
        }

        return Ok(result);
    }

    // Scalar path
    codegen_cast_scalar(
        src,
        src_is_float,
        dst_is_float,
        src_is_signed,
        dst_is_signed,
        src_dtype,
        dst_dtype,
        context,
        builder,
    )
}

#[allow(clippy::too_many_arguments)]
fn codegen_cast_scalar<'ctx>(
    src: BasicValueEnum<'ctx>,
    src_is_float: bool,
    dst_is_float: bool,
    src_is_signed: bool,
    dst_is_signed: bool,
    src_dtype: &DType,
    dst_dtype: &DType,
    context: &'ctx Context,
    builder: &Builder<'ctx>,
) -> Result<BasicValueEnum<'ctx>> {
    let dst_type = common::dtype_to_basic_type(dst_dtype, context)?;

    match (src_is_float, dst_is_float) {
        (true, true) => {
            let src_val = src.into_float_value();
            let dst_type = dst_type.into_float_type();
            if src_dtype.bytes() > dst_dtype.bytes() {
                Ok(builder.build_float_trunc(src_val, dst_type, "fptrunc").context(CastSnafu)?.into())
            } else {
                Ok(builder.build_float_ext(src_val, dst_type, "fpext").context(CastSnafu)?.into())
            }
        }
        (true, false) => {
            let src_val = src.into_float_value();
            let dst_type = dst_type.into_int_type();
            if dst_is_signed {
                Ok(builder.build_float_to_signed_int(src_val, dst_type, "fptosi").context(CastSnafu)?.into())
            } else {
                Ok(builder.build_float_to_unsigned_int(src_val, dst_type, "fptoui").context(CastSnafu)?.into())
            }
        }
        (false, true) => {
            let src_val = src.into_int_value();
            let dst_type = dst_type.into_float_type();
            if src_is_signed {
                Ok(builder.build_signed_int_to_float(src_val, dst_type, "sitofp").context(CastSnafu)?.into())
            } else {
                Ok(builder.build_unsigned_int_to_float(src_val, dst_type, "uitofp").context(CastSnafu)?.into())
            }
        }
        (false, false) => {
            let src_val = src.into_int_value();
            let dst_type = dst_type.into_int_type();
            if src_dtype.bytes() > dst_dtype.bytes() {
                Ok(builder.build_int_truncate(src_val, dst_type, "trunc").context(CastSnafu)?.into())
            } else if src_is_signed {
                Ok(builder.build_int_s_extend(src_val, dst_type, "sext").context(CastSnafu)?.into())
            } else {
                Ok(builder.build_int_z_extend(src_val, dst_type, "zext").context(CastSnafu)?.into())
            }
        }
    }
}

// ============================================================================
// Memory Operations
// ============================================================================

fn codegen_memory<'ctx>(
    uop: &Arc<UOp>,
    context: &'ctx Context,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
    values: &mut ValueMap<'ctx>,
) -> Result<Option<BasicValueEnum<'ctx>>> {
    match uop.op() {
        Op::Load { buffer, index } => {
            let buffer_ptr = require_value(buffer, context, module, builder, values)?;
            let index_val = require_value(index, context, module, builder, values)?;
            Ok(Some(codegen_load(buffer_ptr, index_val, &uop.dtype(), context, builder)?))
        }
        Op::Store { buffer, index, value, ranges: _ } => {
            // ranges are handled in the expand pass - should be empty at codegen time
            let buffer_ptr = require_value(buffer, context, module, builder, values)?;
            let index_val = require_value(index, context, module, builder, values)?;
            let value_val = require_value(value, context, module, builder, values)?;
            // Auto-load if value is a pointer (from INDEX) - matches cranelift backend
            let value_val = auto_load_pointer(value_val, &value.dtype(), context, builder)?;
            codegen_store(buffer_ptr, index_val, value_val, builder)?;
            Ok(None)
        }
        Op::Index { buffer, indices, gate: None } => {
            trace!(index.id = uop.id, buffer.id = buffer.id, num_indices = indices.len(), "INDEX operation");

            let buffer_ptr = require_value(buffer, context, module, builder, values)?;
            if indices.len() == 1 {
                let index_val = require_value(&indices[0], context, module, builder, values)?;
                debug!(
                    uop_id = uop.id,
                    buffer_id = buffer.id,
                    index_id = indices[0].id,
                    index_op = ?indices[0].op(),
                    buffer_ptr = ?buffer_ptr,
                    index_val = ?index_val,
                    result_dtype = ?uop.dtype(),
                    "INDEX: buffer[index]"
                );

                // Handle vector indices (from UPCAST optimization)
                // Return the vector indices directly - LOAD will do gather, STORE will do scatter.
                // This separates addressing (INDEX) from data movement (LOAD/STORE).
                if index_val.is_vector_value() {
                    debug!(uop_id = uop.id, "INDEX: returning vector indices for gather/scatter");
                    return Ok(Some(index_val));
                }

                // Scalar index - regular GEP
                let element_type = match uop.dtype() {
                    DType::Ptr { base, .. } => common::dtype_to_basic_type(&base, context)?,
                    other => common::dtype_to_basic_type(&other, context)?,
                };
                let ptr = unsafe {
                    builder
                        .build_gep(element_type, buffer_ptr.into_pointer_value(), &[index_val.into_int_value()], "idx")
                        .context(BuildGepSnafu)?
                };
                debug!(uop_id = uop.id, result_ptr = ?ptr, "index: computed pointer");
                Ok(Some(ptr.into()))
            } else {
                // Multi-index: linearize at codegen time
                // Extract dimensions from Range.end or DefineVar.max_val
                let dims: Vec<i64> = indices
                    .iter()
                    .map(|idx_uop| {
                        if let Op::Range { end, .. } = idx_uop.op()
                            && let Op::Const(cv) = end.op()
                            && let ConstValue::Int(size) = cv.0
                        {
                            return size;
                        }
                        if let Op::DefineVar { max_val, .. } = idx_uop.op() {
                            return *max_val + 1;
                        }
                        1 // fallback for unknown dimensions
                    })
                    .collect();

                // Compute row-major strides
                let mut strides = vec![1i64; dims.len()];
                for i in (0..dims.len().saturating_sub(1)).rev() {
                    strides[i] = strides[i + 1] * dims[i + 1];
                }

                debug!(
                    uop_id = uop.id,
                    buffer_id = buffer.id,
                    num_indices = indices.len(),
                    dims = ?dims,
                    strides = ?strides,
                    "INDEX: multi-index linearization at codegen"
                );

                // Generate index values
                let index_vals: Vec<BasicValueEnum> = indices
                    .iter()
                    .map(|idx| require_value(idx, context, module, builder, values))
                    .collect::<Result<Vec<_>>>()?;

                // Check if any index is vectorized
                let is_vectorized = index_vals.iter().any(|v| v.is_vector_value());

                if is_vectorized {
                    // Vectorized multi-index - compute linearized vector indices
                    // LOAD will do gather, STORE will do scatter
                    let vec_len = index_vals
                        .iter()
                        .find(|v| v.is_vector_value())
                        .map(|v| v.into_vector_value().get_type().get_size())
                        .unwrap_or(1);

                    // Build result vector of linearized indices
                    let result_vec_type = context.i64_type().vec_type(vec_len);
                    let mut result: BasicValueEnum = result_vec_type.get_poison().into();

                    for lane in 0..vec_len {
                        let lane_const = context.i32_type().const_int(lane as u64, false);

                        // Compute linear index for this lane
                        let mut linear = context.i64_type().const_int(0, false);
                        for (idx_val, &stride) in index_vals.iter().zip(strides.iter()) {
                            let scalar_idx = if idx_val.is_vector_value() {
                                builder
                                    .build_extract_element(idx_val.into_vector_value(), lane_const, "multi_idx")
                                    .context(VectorExtractSnafu)?
                                    .into_int_value()
                            } else {
                                idx_val.into_int_value()
                            };
                            let stride_val = context.i64_type().const_int(stride as u64, false);
                            let term =
                                builder.build_int_mul(scalar_idx, stride_val, "stride_mul").context(ArithmeticSnafu)?;
                            linear = builder.build_int_add(linear, term, "linear_add").context(ArithmeticSnafu)?;
                        }

                        // Insert linearized index into result vector
                        result = builder
                            .build_insert_element(result.into_vector_value(), linear, lane_const, "linear_insert")
                            .context(VectorInsertSnafu)?
                            .into();
                    }

                    debug!(uop_id = uop.id, "INDEX: returning linearized vector indices for gather/scatter");
                    return Ok(Some(result));
                }

                // Scalar multi-index: linearize to single offset
                let mut linear = context.i64_type().const_int(0, false);
                for (idx_val, &stride) in index_vals.iter().zip(strides.iter()) {
                    let stride_val = context.i64_type().const_int(stride as u64, false);
                    let term = builder
                        .build_int_mul(idx_val.into_int_value(), stride_val, "stride_mul")
                        .context(ArithmeticSnafu)?;
                    linear = builder.build_int_add(linear, term, "linear_add").context(ArithmeticSnafu)?;
                }

                // GEP with linear index
                let element_type = match uop.dtype() {
                    DType::Ptr { base, .. } => common::dtype_to_basic_type(&base, context)?,
                    other => common::dtype_to_basic_type(&other, context)?,
                };
                let ptr = unsafe {
                    builder
                        .build_gep(element_type, buffer_ptr.into_pointer_value(), &[linear], "idx_linear")
                        .context(BuildGepSnafu)?
                };
                debug!(uop_id = uop.id, result_ptr = ?ptr, "INDEX: multi-index linearized");
                Ok(Some(ptr.into()))
            }
        }
        Op::Index { gate: Some(_), .. } => UnsupportedSnafu { what: "Gated INDEX" }.fail(),
        Op::PointerIndex { ptr, offset } => {
            let ptr_val = require_value(ptr, context, module, builder, values)?;
            let offset_val = require_value(offset, context, module, builder, values)?;
            let result = unsafe {
                builder
                    .build_gep(
                        context.i8_type(),
                        ptr_val.into_pointer_value(),
                        &[offset_val.into_int_value()],
                        "ptr_idx",
                    )
                    .context(BuildGepSnafu)?
            };
            Ok(Some(result.into()))
        }
        _ => Ok(None),
    }
}

fn codegen_load<'ctx>(
    buffer_ptr: BasicValueEnum<'ctx>,
    index: BasicValueEnum<'ctx>,
    result_dtype: &DType,
    context: &'ctx Context,
    builder: &Builder<'ctx>,
) -> Result<BasicValueEnum<'ctx>> {
    // Handle vector indices (from UPCAST optimization) - perform gather
    if index.is_vector_value() {
        let vec_indices = index.into_vector_value();
        let vec_len = vec_indices.get_type().get_size();
        let ptr_val = buffer_ptr.into_pointer_value();

        // Get scalar element type from result dtype (may be Vector or Scalar)
        let scalar_dtype = match result_dtype {
            DType::Vector { scalar, .. } => DType::Scalar(*scalar),
            _ => result_dtype.clone(),
        };
        let element_type = common::dtype_to_basic_type(&scalar_dtype, context)?;

        // Build result vector type
        let result_vec_type = match element_type {
            inkwell::types::BasicTypeEnum::FloatType(ft) => ft.vec_type(vec_len),
            inkwell::types::BasicTypeEnum::IntType(it) => it.vec_type(vec_len),
            _ => return UnsupportedSnafu { what: "Unsupported element type for vector gather" }.fail(),
        };

        let mut result: BasicValueEnum = result_vec_type.get_poison().into();

        for i in 0..vec_len {
            let i_const = context.i32_type().const_int(i as u64, false);

            // Extract scalar index
            let scalar_idx = builder
                .build_extract_element(vec_indices, i_const, "gather_idx")
                .context(VectorExtractSnafu)?
                .into_int_value();

            // GEP to get pointer
            let elem_ptr = unsafe {
                builder.build_gep(element_type, ptr_val, &[scalar_idx], "gather_gep").context(BuildGepSnafu)?
            };

            // Load element
            let elem_val = builder.build_load(element_type, elem_ptr, "gather_load").context(BuildLoadSnafu)?;

            // Insert into result vector
            result = builder
                .build_insert_element(result.into_vector_value(), elem_val, i_const, "gather_insert")
                .context(VectorInsertSnafu)?
                .into();
        }

        return Ok(result);
    }

    // Scalar path
    let element_ptr = if index.is_pointer_value() {
        index.into_pointer_value()
    } else {
        let ptr_val = buffer_ptr.into_pointer_value();
        let index_val = index.into_int_value();
        let element_type = common::dtype_to_basic_type(result_dtype, context)?;
        unsafe { builder.build_gep(element_type, ptr_val, &[index_val], "gep").context(BuildGepSnafu)? }
    };

    let load_type = common::dtype_to_basic_type(result_dtype, context)?;
    builder.build_load(load_type, element_ptr, "load").context(BuildLoadSnafu)
}

fn codegen_store<'ctx>(
    buffer_ptr: BasicValueEnum<'ctx>,
    index: BasicValueEnum<'ctx>,
    value: BasicValueEnum<'ctx>,
    builder: &Builder<'ctx>,
) -> Result<()> {
    // Handle vector indices (from UPCAST optimization) - perform scatter
    if index.is_vector_value() {
        let vec_indices = index.into_vector_value();
        let vec_len = vec_indices.get_type().get_size();
        let ptr_val = buffer_ptr.into_pointer_value();
        let block = builder.get_insert_block().context(NoInsertBlockSnafu)?;
        let context = block.get_context();

        // Get element type from value's type
        let element_type = if value.is_vector_value() {
            value.into_vector_value().get_type().get_element_type()
        } else {
            value.get_type()
        };

        // If value is also a vector, scatter each element to its index
        // If value is scalar, broadcast it to all indices (less common case)
        let is_value_vec = value.is_vector_value();

        for i in 0..vec_len {
            let i_const = context.i32_type().const_int(i as u64, false);

            // Extract scalar index
            let scalar_idx = builder
                .build_extract_element(vec_indices, i_const, "scatter_idx")
                .context(VectorExtractSnafu)?
                .into_int_value();

            // Get scalar value (extract from vector or use scalar directly)
            let scalar_val = if is_value_vec {
                builder
                    .build_extract_element(value.into_vector_value(), i_const, "scatter_val")
                    .context(VectorExtractSnafu)?
            } else {
                value
            };

            // GEP to get pointer
            let elem_ptr = unsafe {
                builder.build_gep(element_type, ptr_val, &[scalar_idx], "scatter_gep").context(BuildGepSnafu)?
            };

            // Store element
            builder.build_store(elem_ptr, scalar_val).context(BuildStoreSnafu)?;
        }

        return Ok(());
    }

    // Scalar path: existing logic
    let element_ptr = if index.is_pointer_value() {
        index.into_pointer_value()
    } else {
        let ptr_val = buffer_ptr.into_pointer_value();
        let index_val = index.into_int_value();
        let block = builder.get_insert_block().context(NoInsertBlockSnafu)?;
        unsafe {
            builder.build_gep(block.get_context().i8_type(), ptr_val, &[index_val], "gep").context(BuildGepSnafu)?
        }
    };

    builder.build_store(element_ptr, value).context(BuildStoreSnafu)?;
    Ok(())
}

// ============================================================================
// Loop Operations
// ============================================================================

fn codegen_loop<'ctx>(
    uop: &Arc<UOp>,
    context: &'ctx Context,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
    values: &mut ValueMap<'ctx>,
) -> Result<Option<BasicValueEnum<'ctx>>> {
    match uop.op() {
        Op::Range { end, axis_id, axis_type } => {
            codegen_range(uop.id, end, *axis_id, *axis_type, context, module, builder, values)
        }
        Op::End { computation, ranges } => {
            codegen_end(computation, ranges, context, module, builder, values)?;
            Ok(None)
        }
        Op::Reduce { src, ranges, reduce_op } => {
            debug!(reduce.id = uop.id, "codegen reduce");
            codegen_reduce(uop.id, src, ranges, *reduce_op, &uop.dtype(), context, module, builder, values)
        }
        Op::Bind { var, value } => {
            // BIND is now only used for non-loop contexts (e.g., REDUCE ranges wrapped during optimization)
            // Loop generation moved to codegen_range for OUTER/GLOBAL/LOOP ranges (Tinygrad approach)
            trace!(var.id = var.id, value.id = value.id, "processing bind");

            // If value is already computed, map var to it
            if let Some(val) = values.get(value.id) {
                values.insert(var.id, val);
                return Ok(Some(val));
            }

            // Otherwise try to get var value directly
            if let Some(var_value) = values.get(var.id) {
                return Ok(Some(var_value));
            }

            // Neither available - this is a programming error
            NotInValueMapSnafu { what: "BIND variable or value", id: var.id }.fail()
        }
        Op::Sink { sources } => {
            for src in sources {
                codegen_uop(src, context, module, builder, values)?;
            }
            Ok(None)
        }
        Op::Barrier { src, .. } => codegen_uop(src, context, module, builder, values),
        Op::Noop | Op::Unique(_) | Op::Device(_) => Ok(None),
        Op::Vectorize { elements } => {
            if elements.is_empty() {
                return Ok(None);
            }

            // Compile all elements
            let mut elem_vals: Vec<BasicValueEnum> = Vec::with_capacity(elements.len());
            for elem in elements.iter() {
                let val = require_value(elem, context, module, builder, values)?;
                let val = auto_load_pointer(val, &elem.dtype(), context, builder)?;
                elem_vals.push(val);
            }

            let count = elem_vals.len() as u32;

            // Single element: return as-is (scalar)
            if count == 1 {
                return Ok(Some(elem_vals[0]));
            }

            // Check if all elements are the same UOp (broadcast case)
            let all_same = elements.iter().skip(1).all(|e| e.id == elements[0].id);
            if all_same {
                return Ok(Some(common::broadcast_to_vector(builder, elem_vals[0], count, context)?));
            }

            // Build vector via insertelement chain
            let vec_type = if elem_vals[0].is_float_value() {
                elem_vals[0].into_float_value().get_type().vec_type(count)
            } else if elem_vals[0].is_int_value() {
                elem_vals[0].into_int_value().get_type().vec_type(count)
            } else {
                return Ok(Some(elem_vals[0]));
            };

            let mut vec: BasicValueEnum = vec_type.get_poison().into();
            for (i, val) in elem_vals.into_iter().enumerate() {
                let idx = context.i32_type().const_int(i as u64, false);
                vec = builder
                    .build_insert_element(vec.into_vector_value(), val, idx, &format!("vec_{}", i))
                    .context(VectorInsertSnafu)?
                    .into();
            }

            Ok(Some(vec))
        }
        _ => Ok(None),
    }
}

#[allow(clippy::too_many_arguments)]
fn codegen_range<'ctx>(
    uop_id: u64,
    end: &Arc<UOp>,
    _axis_id: morok_ir::AxisId,
    axis_type: AxisType,
    context: &'ctx Context,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
    values: &mut ValueMap<'ctx>,
) -> Result<Option<BasicValueEnum<'ctx>>> {
    // If already processed, return cached value (idempotent)
    // This must come FIRST to handle Reduce ranges that were set up by REDUCE codegen
    // and appear in INDEX indices (via shift_to substitution).
    if let Some(val) = values.get(uop_id) {
        return Ok(Some(val));
    }

    // Reduce ranges are handled entirely by REDUCE codegen
    // DON'T return the end value here - that would be cached and used incorrectly
    // REDUCE will set up the loop counter and store it for its source subgraph
    if axis_type == AxisType::Reduce {
        return Ok(None);
    }

    // THREAD ranges: Don't create loop, use thread_id parameter
    // The thread_id was set by mod.rs during kernel function setup
    if axis_type == AxisType::Thread {
        // Thread range value comes from thread_id parameter (set in mod.rs)
        // Already registered in values during kernel setup by axis_id
        if let Some(val) = values.get_range_by_axis(_axis_id.value()) {
            return Ok(Some(val));
        }
        // Thread range not yet set - this is an error in kernel setup
        return UnsupportedSnafu { what: "Thread range not found in ValueMap - kernel setup error" }.fail();
    }

    // OUTER/GLOBAL/LOOP ranges: create for-loops directly (Tinygrad approach)
    // These loops are closed by mod.rs cleanup at the end of kernel generation.
    if matches!(axis_type, AxisType::Outer | AxisType::Global | AxisType::Loop) {
        trace!(uop_id, axis_type = ?axis_type, "creating loop for range");

        // Check for size-1 loop - no loop needed, just use 0
        if is_const_one(end) {
            let zero = context.i64_type().const_int(0, false);
            values.insert(uop_id, zero.into());
            return Ok(Some(zero.into()));
        }

        // Generate end value BEFORE branching
        let end_val = require_value(end, context, module, builder, values)?;
        let end_int = end_val.into_int_value();

        // Get function for creating blocks
        let current_block = builder.get_insert_block().context(NoInsertBlockSnafu)?;
        let function = current_block.get_parent().context(NoParentFunctionSnafu)?;

        // Build loop using unified loop generation
        let (loop_ctx, counter_val) = loop_gen::build_loop(context, builder, function, end_int, uop_id)?;

        // Store loop context (closed by mod.rs cleanup)
        values.insert_loop(uop_id, loop_ctx);
        values.insert(uop_id, counter_val.into());

        return Ok(Some(counter_val.into()));
    }

    // Other axis types (handled by pre_expand): check for size-1, otherwise error
    if is_const_one(end) {
        let zero = context.i64_type().const_int(0, false);
        return Ok(Some(zero.into()));
    }

    // UPCAST/UNROLL should have been converted to UNROLL ops by pre_expand
    UnsupportedSnafu { what: "RANGE with non-loop axis_type should be handled by pre_expand" }.fail()
}

/// Check if a UOp is Const(1)
fn is_const_one(uop: &Arc<UOp>) -> bool {
    if let Op::Const(val_hash) = uop.op() {
        matches!(val_hash.0, morok_ir::ConstValue::Int(1) | morok_ir::ConstValue::UInt(1))
    } else {
        false
    }
}

fn codegen_end<'ctx>(
    computation: &Arc<UOp>,
    ranges: &[Arc<UOp>],
    context: &'ctx Context,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
    values: &mut ValueMap<'ctx>,
) -> Result<()> {
    // Generate the computation (loop body code)
    codegen_uop(computation, context, module, builder, values)?;

    // Close each range's loop (innermost to outermost)
    // Note: We DON'T skip OUTER ranges here - let END close all loops it receives.
    // The mod.rs cleanup will handle any remaining unclosed loops.
    for range_uop in ranges.iter().rev() {
        // Get and REMOVE loop context (prevents double-closing)
        // Reduce ranges don't have loop contexts (handled by REDUCE codegen)
        // take_loop returns None if already closed or never created
        if let Some(loop_ctx) = values.take_loop(range_uop.id) {
            loop_gen::close_loop(builder, &loop_ctx)?;
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn codegen_reduce<'ctx>(
    reduce_id: u64,
    src: &Arc<UOp>,
    ranges: &[Arc<UOp>],
    reduce_op: ReduceOp,
    result_dtype: &DType,
    context: &'ctx Context,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
    values: &mut ValueMap<'ctx>,
) -> Result<Option<BasicValueEnum<'ctx>>> {
    debug!(reduce.id = reduce_id, num_ranges = ranges.len(), reduce_op = ?reduce_op, "codegen_reduce");
    // If no ranges, just return the source
    if ranges.is_empty() {
        trace!("no ranges, returning source directly");
        return codegen_uop(src, context, module, builder, values);
    }

    // Detect vectorized accumulator pattern (UPCAST on reduce axis)
    // When result_dtype is vector, we use vector accumulators and horizontal reduce at the end
    let is_vector_accumulator = matches!(result_dtype, DType::Vector { .. });
    let vec_len = result_dtype.vcount();
    let scalar_dtype = DType::Scalar(result_dtype.base());

    // For vector accumulators, we need the scalar identity to splat
    let identity = codegen_reduce_identity(reduce_op, &scalar_dtype, context)?;

    let current_block = builder.get_insert_block().context(NoInsertBlockSnafu)?;
    let function = current_block.get_parent().context(NoParentFunctionSnafu)?;

    // Filter to Reduce AND Loop ranges (OUTER is handled by BIND)
    // Loop-type ranges in REDUCE's ranges must be handled here to avoid orphaned loops.
    // When REDUCE source evaluation encounters these ranges, they create loops that must be closed.
    let reduce_ranges: Vec<_> = ranges
        .iter()
        .filter(|r| {
            if let Op::Range { axis_type, .. } = r.op() {
                matches!(axis_type, AxisType::Reduce | AxisType::Loop)
            } else {
                false
            }
        })
        .collect();

    if reduce_ranges.is_empty() {
        return codegen_uop(src, context, module, builder, values);
    }

    // Use alloca-based approach for accumulator (like Tinygrad's DEFINE_REG)
    // This handles nested source loops correctly - PHI-based doesn't work when source creates loops
    // because SSA values defined inside loops aren't available at the loop exit block.
    let acc_type = common::dtype_to_basic_type(result_dtype, context)?;
    let acc_alloca = builder.build_alloca(acc_type, "reduce_acc").context(BuildAllocaSnafu)?;

    // Initialize accumulator with identity (splat to vector if needed)
    let init_val =
        if is_vector_accumulator { splat_scalar_to_vector(identity, vec_len, context, builder)? } else { identity };
    builder.build_store(acc_alloca, init_val).context(BuildStoreSnafu)?;

    // Build nested counting loops for reduce ranges
    let mut loop_ctxs = Vec::new();
    for (i, range_uop) in reduce_ranges.iter().enumerate() {
        let Op::Range { end, axis_id, .. } = range_uop.op() else {
            return UnsupportedSnafu { what: "REDUCE range must be RANGE op" }.fail();
        };

        // Check for size-1 loop - no loop needed, just use 0
        if is_const_one(end) {
            let zero = context.i64_type().const_int(0, false);
            // Insert by axis_id (the canonical key for Range nodes)
            values.insert_range(axis_id.value(), zero.into());
            // Don't push to loop_ctxs - no loop to close
            continue;
        }

        let end_val = require_value(end, context, module, builder, values)?;
        let end_int = end_val.into_int_value();

        let loop_id = reduce_id + i as u64;
        let (loop_ctx, counter_val) = loop_gen::build_loop(context, builder, function, end_int, loop_id)?;

        // Insert by axis_id (the canonical key for Range nodes)
        values.insert_range(axis_id.value(), counter_val.into());
        loop_ctxs.push(loop_ctx);
    }

    // Track loops before source evaluation - source may create additional nested loops
    let loops_before: std::collections::HashSet<u64> = values.remaining_loop_ids().into_iter().collect();

    // IMPORTANT: Snapshot cached IDs before source evaluation.
    // Values generated inside REDUCE loops are block-local and must not be reused
    // by subsequent REDUCEs (they're in different basic blocks that don't dominate each other).
    let cached_before = values.cached_ids();

    trace!(loops_before = ?loops_before, "about to evaluate reduce source");

    // Check for FMA pattern: REDUCE(Add, Mul(a, b), ...) where dtype is float
    // FMA generates: acc = fma(a, b, acc) instead of acc = acc + (a * b)
    let fma_operands = if reduce_op == ReduceOp::Add && result_dtype.base().is_float() {
        let result = try_extract_fma_operands(src);
        debug!(
            src_op = ?std::mem::discriminant(src.op()),
            src_dtype = ?src.dtype(),
            fma_detected = result.is_some(),
            "FMA detection for REDUCE source"
        );
        result
    } else {
        None
    };

    if let Some((mul_a, mul_b)) = fma_operands {
        // FMA path: compile Mul operands separately, use llvm.fma intrinsic
        trace!("FMA path: detected Mul source for Add reduce");

        // Compile Mul operands
        let a_val = require_value(mul_a, context, module, builder, values)?;
        let a_val = auto_load_pointer(a_val, &mul_a.dtype(), context, builder)?;

        let b_val = require_value(mul_b, context, module, builder, values)?;
        let b_val = auto_load_pointer(b_val, &mul_b.dtype(), context, builder)?;

        // Handle vector splat if needed (same logic as standard path)
        let (a_val, b_val) = if is_vector_accumulator {
            let a_val = if !a_val.is_vector_value() {
                splat_scalar_to_vector(a_val, vec_len, context, builder)?
            } else {
                a_val
            };
            let b_val = if !b_val.is_vector_value() {
                splat_scalar_to_vector(b_val, vec_len, context, builder)?
            } else {
                b_val
            };
            (a_val, b_val)
        } else {
            (a_val, b_val)
        };

        // Load accumulator
        let acc_val = builder.build_load(acc_type, acc_alloca, "acc_load").context(BuildLoadSnafu)?;

        // FMA: acc = a * b + acc
        let suffix = get_type_suffix(result_dtype)?;
        let new_acc =
            call_intrinsic(&format!("llvm.fma.{}", suffix), &[a_val, b_val, acc_val], "fma", module, builder)?;
        builder.build_store(acc_alloca, new_acc).context(BuildStoreSnafu)?;

        trace!("FMA intrinsic generated for reduce accumulation");
    } else {
        // Standard path: evaluate source, then accumulate
        let src_val = require_value(src, context, module, builder, values)?;

        trace!(
            loops_after = ?values.remaining_loop_ids(),
            src.dtype = ?src.dtype(),
            src_val = ?src_val,
            "REDUCE source evaluated"
        );
        let src_val = auto_load_pointer(src_val, &src.dtype(), context, builder)?;
        trace!(src_val_loaded = ?src_val, "reduce after auto_load");

        // Load accumulator AFTER source evaluation (inside any source loops)
        // This ensures we get the current value, not a stale one from before the source loops
        let acc_val = builder.build_load(acc_type, acc_alloca, "acc_load").context(BuildLoadSnafu)?;

        // Do accumulation INSIDE the source loops (before closing them)
        // For vector accumulators: if source is scalar, splat it to match accumulator width
        let acc_src_val = if is_vector_accumulator && !src_val.is_vector_value() {
            splat_scalar_to_vector(src_val, vec_len, context, builder)?
        } else {
            src_val
        };
        let new_acc = codegen_reduce_op(reduce_op, acc_val, acc_src_val, result_dtype, module, builder)?;
        builder.build_store(acc_alloca, new_acc).context(BuildStoreSnafu)?;
    }

    // Close any loops created during source evaluation (innermost first)
    let loops_after = values.remaining_loop_ids();
    for loop_id in loops_after.into_iter().rev() {
        if !loops_before.contains(&loop_id)
            && let Some(loop_ctx) = values.take_loop(loop_id)
        {
            loop_gen::close_loop(builder, &loop_ctx)?;
        }
    }

    // Close all reduce loops we created (innermost first)
    for loop_ctx in loop_ctxs.into_iter().rev() {
        loop_gen::close_loop(builder, &loop_ctx)?;
    }

    // Clear values that were generated inside REDUCE loops.
    // These are block-local and must not be reused by subsequent REDUCEs.
    let cached_after = values.cached_ids();
    for id in cached_after.difference(&cached_before) {
        values.remove(*id);
    }

    // Load final accumulator value
    let acc_final = builder.build_load(acc_type, acc_alloca, "reduce_acc_final").context(BuildLoadSnafu)?;

    // For vector accumulators: perform horizontal reduction to get scalar result
    let final_val = if is_vector_accumulator {
        trace!(vec_len, "performing horizontal reduction on vector accumulator");
        codegen_horizontal_reduce(acc_final, reduce_op, &scalar_dtype, context, module, builder)?
    } else {
        acc_final
    };

    Ok(Some(final_val))
}

// ============================================================================
// Helper Functions
// ============================================================================

fn require_value<'ctx>(
    uop: &Arc<UOp>,
    context: &'ctx Context,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
    values: &mut ValueMap<'ctx>,
) -> Result<BasicValueEnum<'ctx>> {
    codegen_uop(uop, context, module, builder, values)?
        .context(NoValueSnafu { op: format!("{:?}", uop.op()), id: uop.id })
}

fn auto_load_pointer<'ctx>(
    value: BasicValueEnum<'ctx>,
    dtype: &DType,
    context: &'ctx Context,
    builder: &Builder<'ctx>,
) -> Result<BasicValueEnum<'ctx>> {
    if value.is_pointer_value() {
        let element_type = match dtype {
            DType::Ptr { base, .. } => common::dtype_to_basic_type(base, context)?,
            _ => {
                trace!(dtype = ?dtype, value = ?value, "auto_load_pointer: not a Ptr dtype, returning as-is");
                return Ok(value);
            }
        };
        let loaded =
            builder.build_load(element_type, value.into_pointer_value(), "autoload").context(BuildLoadSnafu)?;
        debug!(
            dtype = ?dtype,
            ptr = ?value,
            loaded = ?loaded,
            "auto_load_pointer: loaded value from pointer"
        );
        Ok(loaded)
    } else {
        trace!(dtype = ?dtype, value = ?value, "auto_load_pointer: not a pointer, returning as-is");
        Ok(value)
    }
}

fn extract_value<'ctx>(call_site: CallSiteValue<'ctx>) -> Result<BasicValueEnum<'ctx>> {
    match call_site.try_as_basic_value() {
        ValueKind::Basic(v) => Ok(v),
        ValueKind::Instruction(_) => ValueExtractionFailedSnafu { expected: "BasicValue" }.fail(),
    }
}

fn get_intrinsic<'ctx>(
    name: &str,
    ret_type: BasicValueEnum<'ctx>,
    module: &Module<'ctx>,
) -> Result<FunctionValue<'ctx>> {
    let intrinsic = Intrinsic::find(name).context(IntrinsicNotFoundSnafu { name: name.to_string() })?;
    intrinsic
        .get_declaration(module, &[ret_type.get_type()])
        .context(IntrinsicDeclarationSnafu { name: name.to_string() })
}

fn call_intrinsic<'ctx>(
    intrinsic_name: &str,
    args: &[BasicValueEnum<'ctx>],
    name: &str,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
) -> Result<BasicValueEnum<'ctx>> {
    let intrinsic_fn = get_intrinsic(intrinsic_name, args[0], module)?;
    let call_site = builder
        .build_call(intrinsic_fn, &args.iter().map(|v| (*v).into()).collect::<Vec<_>>(), name)
        .context(BuildCallSnafu { intrinsic: intrinsic_name.to_string() })?;
    extract_value(call_site)
}

fn get_type_suffix(dtype: &DType) -> Result<String> {
    match dtype.scalar() {
        Some(morok_dtype::ScalarDType::Float16) => Ok("f16".to_string()),
        Some(morok_dtype::ScalarDType::BFloat16) => Ok("f16".to_string()),
        Some(morok_dtype::ScalarDType::Float32) => Ok("f32".to_string()),
        Some(morok_dtype::ScalarDType::Float64) => Ok("f64".to_string()),
        _ => UnsupportedIntrinsicTypeSnafu { dtype: format!("{:?}", dtype) }.fail(),
    }
}

// ============================================================================
// Unary Operation Helpers
// ============================================================================

fn codegen_abs<'ctx>(
    src: BasicValueEnum<'ctx>,
    result_dtype: &DType,
    _context: &'ctx Context,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
) -> Result<BasicValueEnum<'ctx>> {
    if result_dtype.is_float() {
        let suffix = get_type_suffix(result_dtype)?;
        call_intrinsic(&format!("llvm.fabs.{}", suffix), &[src], "abs", module, builder)
    } else {
        let int_val = src.into_int_value();
        let zero = int_val.get_type().const_zero();
        let is_neg = builder.build_int_compare(IntPredicate::SLT, int_val, zero, "is_neg").context(ComparisonSnafu)?;
        let neg_val = builder.build_int_neg(int_val, "neg_val").context(ArithmeticSnafu)?;
        Ok(builder.build_select(is_neg, neg_val, int_val, "abs").context(BuildSelectSnafu)?)
    }
}

fn codegen_rsqrt<'ctx>(
    src: BasicValueEnum<'ctx>,
    result_dtype: &DType,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
) -> Result<BasicValueEnum<'ctx>> {
    let suffix = get_type_suffix(result_dtype)?;
    let sqrt_val = call_intrinsic(&format!("llvm.sqrt.{}", suffix), &[src], "sqrt", module, builder)?;
    let one = src.into_float_value().get_type().const_float(1.0);
    let result: BasicValueEnum<'ctx> =
        builder.build_float_div(one, sqrt_val.into_float_value(), "rsqrt").context(ArithmeticSnafu)?.into();
    common::fast_math::apply_fast_math_flags(result);
    Ok(result)
}

fn codegen_tan<'ctx>(
    src: BasicValueEnum<'ctx>,
    result_dtype: &DType,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
) -> Result<BasicValueEnum<'ctx>> {
    let suffix = get_type_suffix(result_dtype)?;
    let sin_val = call_intrinsic(&format!("llvm.sin.{}", suffix), &[src], "sin", module, builder)?;
    let cos_val = call_intrinsic(&format!("llvm.cos.{}", suffix), &[src], "cos", module, builder)?;
    let result: BasicValueEnum<'ctx> = builder
        .build_float_div(sin_val.into_float_value(), cos_val.into_float_value(), "tan")
        .context(ArithmeticSnafu)?
        .into();
    common::fast_math::apply_fast_math_flags(result);
    Ok(result)
}

fn codegen_reciprocal<'ctx>(src: BasicValueEnum<'ctx>, builder: &Builder<'ctx>) -> Result<BasicValueEnum<'ctx>> {
    let one = src.into_float_value().get_type().const_float(1.0);
    let result: BasicValueEnum<'ctx> =
        builder.build_float_div(one, src.into_float_value(), "recip").context(ArithmeticSnafu)?.into();
    common::fast_math::apply_fast_math_flags(result);
    Ok(result)
}

fn codegen_trunc<'ctx>(
    src: BasicValueEnum<'ctx>,
    result_dtype: &DType,
    is_float: bool,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
) -> Result<BasicValueEnum<'ctx>> {
    if is_float {
        let suffix = get_type_suffix(result_dtype)?;
        call_intrinsic(&format!("llvm.trunc.{}", suffix), &[src], "trunc", module, builder)
    } else {
        Ok(src)
    }
}

fn codegen_sign<'ctx>(
    src: BasicValueEnum<'ctx>,
    is_float: bool,
    builder: &Builder<'ctx>,
) -> Result<BasicValueEnum<'ctx>> {
    if is_float {
        let float_val = src.into_float_value();
        let zero = float_val.get_type().const_zero();
        let one = float_val.get_type().const_float(1.0);

        let is_pos =
            builder.build_float_compare(FloatPredicate::OGT, float_val, zero, "is_pos").context(ComparisonSnafu)?;
        let is_neg =
            builder.build_float_compare(FloatPredicate::OLT, float_val, zero, "is_neg").context(ComparisonSnafu)?;

        let pos_val = builder.build_select(is_pos, one, zero, "pos_val").context(BuildSelectSnafu)?;
        let neg_val = builder.build_select(is_neg, one, zero, "neg_val").context(BuildSelectSnafu)?;

        Ok(builder
            .build_float_sub(pos_val.into_float_value(), neg_val.into_float_value(), "sign")
            .context(ArithmeticSnafu)?
            .into())
    } else {
        let int_val = src.into_int_value();
        let zero = int_val.get_type().const_zero();
        let one = int_val.get_type().const_int(1, false);
        let neg_one = int_val.get_type().const_int((-1_i64) as u64, true);

        let is_pos = builder.build_int_compare(IntPredicate::SGT, int_val, zero, "is_pos").context(ComparisonSnafu)?;
        let is_neg = builder.build_int_compare(IntPredicate::SLT, int_val, zero, "is_neg").context(ComparisonSnafu)?;

        let pos_val = builder.build_select(is_pos, one, zero, "pos_val").context(BuildSelectSnafu)?;
        let neg_val = builder.build_select(is_neg, neg_one, zero, "neg_val").context(BuildSelectSnafu)?;

        Ok(builder
            .build_int_add(pos_val.into_int_value(), neg_val.into_int_value(), "sign")
            .context(ArithmeticSnafu)?
            .into())
    }
}

fn codegen_erf<'ctx>(
    src: BasicValueEnum<'ctx>,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
) -> Result<BasicValueEnum<'ctx>> {
    let float_type = src.into_float_value().get_type();
    let fn_type = float_type.fn_type(&[float_type.into()], false);
    let fn_val = module.add_function("erf", fn_type, None);
    let call_site =
        builder.build_call(fn_val, &[src.into()], "erf").context(BuildCallSnafu { intrinsic: "erf".to_string() })?;
    extract_value(call_site)
}

// ============================================================================
// Reduce Helpers
// ============================================================================

fn codegen_reduce_identity<'ctx>(
    reduce_op: ReduceOp,
    dtype: &DType,
    context: &'ctx Context,
) -> Result<BasicValueEnum<'ctx>> {
    let llvm_type = common::dtype_to_basic_type(dtype, context)?;
    let is_float = dtype.is_float();
    let is_signed = dtype.is_signed();

    match reduce_op {
        ReduceOp::Add => {
            if is_float {
                Ok(llvm_type.into_float_type().const_float(0.0).into())
            } else {
                Ok(llvm_type.into_int_type().const_int(0, false).into())
            }
        }
        ReduceOp::Mul => {
            if is_float {
                Ok(llvm_type.into_float_type().const_float(1.0).into())
            } else {
                Ok(llvm_type.into_int_type().const_int(1, false).into())
            }
        }
        ReduceOp::Max => {
            if is_float {
                Ok(llvm_type.into_float_type().const_float(f64::NEG_INFINITY).into())
            } else if is_signed {
                let bits = dtype.bytes() * 8;
                let min_val = match bits {
                    8 => i8::MIN as u64,
                    16 => i16::MIN as u64,
                    32 => i32::MIN as u64,
                    64 => i64::MIN as u64,
                    _ => 0,
                };
                Ok(llvm_type.into_int_type().const_int(min_val, true).into())
            } else {
                Ok(llvm_type.into_int_type().const_int(0, false).into())
            }
        }
        ReduceOp::Min => {
            if is_float {
                Ok(llvm_type.into_float_type().const_float(f64::INFINITY).into())
            } else if is_signed {
                let bits = dtype.bytes() * 8;
                let max_val = match bits {
                    8 => i8::MAX as u64,
                    16 => i16::MAX as u64,
                    32 => i32::MAX as u64,
                    64 => i64::MAX as u64,
                    _ => u64::MAX,
                };
                Ok(llvm_type.into_int_type().const_int(max_val, true).into())
            } else {
                let bits = dtype.bytes() * 8;
                let max_val = match bits {
                    8 => u8::MAX as u64,
                    16 => u16::MAX as u64,
                    32 => u32::MAX as u64,
                    64 => u64::MAX,
                    _ => u64::MAX,
                };
                Ok(llvm_type.into_int_type().const_int(max_val, false).into())
            }
        }
    }
}

/// Try to extract FMA operands from a REDUCE source.
///
/// For matmul: REDUCE(Add, CONTRACT?(Mul(a, b), ..), ranges)
///
/// Returns Some((a, b)) if FMA candidate, None otherwise.
///
/// NOTE: Does NOT unwrap GEP - after scalar devectorization (pm_scalar_accumulators),
/// the Mul is vectorized but accumulator is scalar, so FMA can't be used.
/// FMA only works when source Mul and accumulator have matching types.
fn try_extract_fma_operands(src: &Arc<UOp>) -> Option<(&Arc<UOp>, &Arc<UOp>)> {
    // Don't match GEP - that means scalar accumulator with vector source (incompatible)
    if matches!(src.op(), Op::Gep { .. }) {
        return None;
    }

    // Unwrap CONTRACT/UNROLL wrappers (they pass through in codegen)
    let inner = match src.op() {
        Op::Contract { src, .. } | Op::Unroll { src, .. } => src,
        _ => src,
    };

    // Check for Mul pattern
    match inner.op() {
        Op::Binary(BinaryOp::Mul, a, b) => Some((a, b)),
        _ => None,
    }
}

fn codegen_reduce_op<'ctx>(
    reduce_op: ReduceOp,
    acc: BasicValueEnum<'ctx>,
    src: BasicValueEnum<'ctx>,
    dtype: &DType,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
) -> Result<BasicValueEnum<'ctx>> {
    // Use base() to get scalar type for vectors (e.g., Vector<Float32> -> Float32)
    let base = dtype.base();
    let is_float = base.is_float();
    let is_signed = base.is_signed();

    match reduce_op {
        ReduceOp::Add => common::build_add(builder, acc, src, is_float),
        ReduceOp::Mul => common::build_mul(builder, acc, src, is_float),
        ReduceOp::Max => {
            if is_float {
                let suffix = get_type_suffix(dtype)?;
                call_intrinsic(&format!("llvm.maxnum.{}", suffix), &[acc, src], "reduce_max", module, builder)
            } else if is_signed {
                call_intrinsic(&format!("llvm.smax.i{}", dtype.bytes() * 8), &[acc, src], "reduce_max", module, builder)
            } else {
                call_intrinsic(&format!("llvm.umax.i{}", dtype.bytes() * 8), &[acc, src], "reduce_max", module, builder)
            }
        }
        ReduceOp::Min => {
            if is_float {
                let suffix = get_type_suffix(dtype)?;
                call_intrinsic(&format!("llvm.minnum.{}", suffix), &[acc, src], "reduce_min", module, builder)
            } else if is_signed {
                call_intrinsic(&format!("llvm.smin.i{}", dtype.bytes() * 8), &[acc, src], "reduce_min", module, builder)
            } else {
                call_intrinsic(&format!("llvm.umin.i{}", dtype.bytes() * 8), &[acc, src], "reduce_min", module, builder)
            }
        }
    }
}

// ============================================================================
// Horizontal Reduction (Post-Loop Vector to Scalar)
// ============================================================================

/// Perform horizontal reduction on a vector accumulator to produce a scalar.
///
/// Used when UPCAST is applied to a reduce axis, creating vectorized accumulators.
/// After the reduce loop ends, this function chains all vector lanes together
/// using tree reduction to produce the final scalar result.
///
/// Example for 4-element float vector with Add:
/// ```text
/// [a, b, c, d] -> (a+b) + (c+d) -> result
/// ```
fn codegen_horizontal_reduce<'ctx>(
    vec_val: BasicValueEnum<'ctx>,
    reduce_op: ReduceOp,
    scalar_dtype: &DType,
    context: &'ctx Context,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
) -> Result<BasicValueEnum<'ctx>> {
    let vec = vec_val.into_vector_value();
    let vec_len = vec.get_type().get_size() as usize;

    // Base case: single element - just extract it
    if vec_len == 1 {
        let idx = context.i32_type().const_int(0, false);
        return builder.build_extract_element(vec, idx, "horiz_single").context(VectorExtractSnafu);
    }

    // Extract all elements
    let mut elements: Vec<BasicValueEnum<'ctx>> = Vec::with_capacity(vec_len);
    for i in 0..vec_len {
        let idx = context.i32_type().const_int(i as u64, false);
        let elem = builder.build_extract_element(vec, idx, &format!("horiz_e{}", i)).context(VectorExtractSnafu)?;
        elements.push(elem);
    }

    // Tree reduction: pairwise combine until one element remains
    let mut level = elements;
    while level.len() > 1 {
        let mut next_level = Vec::with_capacity(level.len().div_ceil(2));
        for chunk in level.chunks(2) {
            if chunk.len() == 2 {
                let combined = codegen_reduce_op(reduce_op, chunk[0], chunk[1], scalar_dtype, module, builder)?;
                next_level.push(combined);
            } else {
                // Odd element - carry forward
                next_level.push(chunk[0]);
            }
        }
        level = next_level;
    }

    Ok(level[0])
}

/// Create a vector with all lanes set to the same scalar value (splat/broadcast).
fn splat_scalar_to_vector<'ctx>(
    scalar: BasicValueEnum<'ctx>,
    vec_len: usize,
    context: &'ctx Context,
    builder: &Builder<'ctx>,
) -> Result<BasicValueEnum<'ctx>> {
    // Get appropriate vector type based on scalar type
    let vec_type = if scalar.is_float_value() {
        scalar.into_float_value().get_type().vec_type(vec_len as u32)
    } else {
        scalar.into_int_value().get_type().vec_type(vec_len as u32)
    };

    // Build the vector by inserting the scalar into each lane
    let mut vec_val: BasicValueEnum = vec_type.get_poison().into();
    for i in 0..vec_len {
        let idx = context.i32_type().const_int(i as u64, false);
        vec_val = builder
            .build_insert_element(vec_val.into_vector_value(), scalar, idx, &format!("splat_{}", i))
            .context(VectorInsertSnafu)?
            .into();
    }

    Ok(vec_val)
}
