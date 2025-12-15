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
    // Check if already generated
    if values.contains(uop.id) {
        trace!(uop_id = uop.id, op = ?std::mem::discriminant(uop.op()), "codegen_uop: cache hit");
        return Ok(values.get(uop.id));
    }

    // REDUCE ranges must ONLY be handled by codegen_reduce which creates their loop structure.
    // Don't process or cache them here - codegen_reduce will insert them into values when ready.
    // This prevents LLVM dominator errors where range counters are used before defined.
    if let Op::Range { axis_type: AxisType::Reduce, .. } = uop.op() {
        trace!(uop_id = uop.id, "codegen_uop: skipping REDUCE range - handled by codegen_reduce");
        return Ok(None);
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
        Op::Const(_) => OpCategory::Constant,

        Op::Unary(..) | Op::Binary(..) | Op::Ternary(..) | Op::Cast { .. } => OpCategory::Arithmetic,

        Op::Load { .. } | Op::Store { .. } | Op::Index { .. } | Op::PointerIndex { .. } => OpCategory::Memory,

        Op::Range { .. } | Op::End { .. } | Op::Reduce { .. } | Op::Bind { .. } => OpCategory::Loop,

        // Buffer/DefineGlobal/DefineLocal should already be in ValueMap from renderer
        Op::Buffer { .. } | Op::DefineGlobal(_) | Op::DefineLocal(_) => OpCategory::Meta,

        // DefineVar: For inlined outer loops, value comes from Bind (loop counter)
        // Should be in ValueMap after Bind is processed. If not there, it's an error.
        Op::DefineVar { .. } => OpCategory::Meta,

        Op::Sink { .. }
        | Op::Barrier { .. }
        | Op::Noop
        | Op::Unique(_)
        | Op::Device(_)
        | Op::Vectorize { .. }
        | Op::Kernel { .. } => OpCategory::Meta,

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
    let Op::Const(val_hash) = uop.op() else {
        return Ok(None);
    };

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
        BinaryOp::Fdiv => Ok(builder
            .build_float_div(lhs.into_float_value(), rhs.into_float_value(), "fdiv")
            .context(ArithmeticSnafu)?
            .into()),
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
    let dst_type = common::dtype_to_basic_type(dst_dtype, context)?;
    let src_is_float = src_dtype.is_float();
    let dst_is_float = dst_dtype.is_float();
    let src_is_signed = src_dtype.is_signed();
    let dst_is_signed = dst_dtype.is_signed();

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
        Op::Store { buffer, index, value } => {
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
                UnsupportedSnafu { what: "Multi-index INDEX" }.fail()
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
            // For OUTER ranges: Create the loop HERE (single responsibility)
            // This matches Tinygrad where loop creation happens at RANGE/BIND processing
            if let Op::Range { end, axis_type: AxisType::Outer, .. } = value.op() {
                trace!(var.id = var.id, value.id = value.id, "processing bind with outer range");
                // Check if already created (idempotent)
                if let Some(val) = values.get(value.id) {
                    trace!(value.id = value.id, "bind reusing existing counter");
                    values.insert(var.id, val);
                    return Ok(Some(val));
                }
                trace!(range.id = value.id, "creating new loop for outer range");

                // Check for size-1 loop - no loop needed, just use 0
                if is_const_one(end) {
                    let zero = context.i64_type().const_int(0, false);
                    values.insert(var.id, zero.into());
                    values.insert(value.id, zero.into());
                    // Don't insert loop context - mod.rs cleanup will skip it
                    return Ok(Some(zero.into()));
                }

                // Generate end value
                let end_val = require_value(end, context, module, builder, values)?;
                let end_int = end_val.into_int_value();

                // Get function for creating blocks
                let current_block = builder.get_insert_block().context(NoInsertBlockSnafu)?;
                let function = current_block.get_parent().context(NoParentFunctionSnafu)?;

                // Build loop - use Range's uop_id for consistent lookup
                let (loop_ctx, counter_val) = loop_gen::build_loop(context, builder, function, end_int, value.id)?;

                // Store loop context under Range.id (for mod.rs cleanup to close)
                values.insert_loop(value.id, loop_ctx);

                // Map both var.id and value.id to counter
                values.insert(var.id, counter_val.into());
                values.insert(value.id, counter_val.into());

                return Ok(Some(counter_val.into()));
            }

            // Non-OUTER: just return variable value
            let var_value = values.get(var.id).context(NotInValueMapSnafu { what: "BIND variable", id: var.id })?;
            Ok(Some(var_value))
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
            if elements.len() == 1 {
                codegen_uop(&elements[0], context, module, builder, values)
            } else {
                let mut last_val = None;
                for elem in elements {
                    last_val = codegen_uop(elem, context, module, builder, values)?;
                }
                Ok(last_val)
            }
        }
        _ => Ok(None),
    }
}

#[allow(clippy::too_many_arguments)]
fn codegen_range<'ctx>(
    uop_id: u64,
    end: &Arc<UOp>,
    axis_id: morok_ir::AxisId,
    axis_type: AxisType,
    context: &'ctx Context,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
    values: &mut ValueMap<'ctx>,
) -> Result<Option<BasicValueEnum<'ctx>>> {
    // Outer ranges: loop was created by BIND - return cached counter value
    if axis_type == AxisType::Outer {
        if let Some(val) = values.get(uop_id) {
            return Ok(Some(val));
        }
        // Should have been created by BIND - error if not found
        return NotInValueMapSnafu { what: "OUTER range (expected from BIND)", id: uop_id }.fail();
    }

    // Reduce ranges are handled entirely by REDUCE codegen
    // DON'T return the end value here - that would be cached and used incorrectly
    // REDUCE will set up the loop counter and store it for its source subgraph
    if axis_type == AxisType::Reduce {
        return Ok(None);
    }

    // Loop ranges: if already processed by REDUCE, return cached value
    // Otherwise create loop normally (will be closed by END)
    if axis_type == AxisType::Loop
        && let Some(val) = values.get(uop_id)
    {
        // Already processed by REDUCE - return cached counter value
        return Ok(Some(val));
    }
    // Fall through to create loop - will be closed by END

    // Check for size-1 loop - no loop needed, just use 0
    if is_const_one(end) {
        let zero = context.i64_type().const_int(0, false);
        // Don't insert loop context - END will just skip it (take_loop returns None)
        return Ok(Some(zero.into()));
    }

    // Generate end value BEFORE branching
    let end_val = require_value(end, context, module, builder, values)?;
    let end_int = end_val.into_int_value();

    // Get function for creating blocks
    let current_block = builder.get_insert_block().context(NoInsertBlockSnafu)?;
    let function = current_block.get_parent().context(NoParentFunctionSnafu)?;

    // Build loop using unified loop generation
    let (loop_ctx, counter_val) = loop_gen::build_loop(context, builder, function, end_int, axis_id.value() as u64)?;

    // Store loop context for END
    values.insert_loop(uop_id, loop_ctx);

    Ok(Some(counter_val.into()))
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

    let identity = codegen_reduce_identity(reduce_op, result_dtype, context)?;

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

    // Initialize accumulator with identity
    builder.build_store(acc_alloca, identity).context(BuildStoreSnafu)?;

    // Build nested counting loops for reduce ranges
    let mut loop_ctxs = Vec::new();
    for (i, range_uop) in reduce_ranges.iter().enumerate() {
        let Op::Range { end, .. } = range_uop.op() else {
            return UnsupportedSnafu { what: "REDUCE range must be RANGE op" }.fail();
        };

        // Check for size-1 loop - no loop needed, just use 0
        if is_const_one(end) {
            let zero = context.i64_type().const_int(0, false);
            values.insert(range_uop.id, zero.into());
            // Don't push to loop_ctxs - no loop to close
            continue;
        }

        let end_val = require_value(end, context, module, builder, values)?;
        let end_int = end_val.into_int_value();

        let loop_id = reduce_id + i as u64;
        let (loop_ctx, counter_val) = loop_gen::build_loop(context, builder, function, end_int, loop_id)?;

        values.insert(range_uop.id, counter_val.into());
        loop_ctxs.push(loop_ctx);
    }

    // Track loops before source evaluation - source may create additional nested loops
    let loops_before: std::collections::HashSet<u64> = values.remaining_loop_ids().into_iter().collect();

    // IMPORTANT: Snapshot cached IDs before source evaluation.
    // Values generated inside REDUCE loops are block-local and must not be reused
    // by subsequent REDUCEs (they're in different basic blocks that don't dominate each other).
    let cached_before = values.cached_ids();

    trace!(loops_before = ?loops_before, "about to evaluate reduce source");

    // Evaluate source first - this may create nested loops
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
    let new_acc = codegen_reduce_op(reduce_op, acc_val, src_val, result_dtype, module, builder)?;
    builder.build_store(acc_alloca, new_acc).context(BuildStoreSnafu)?;

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

    // Load final result
    let final_val = builder.build_load(acc_type, acc_alloca, "reduce_result").context(BuildLoadSnafu)?;
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
    Ok(builder.build_float_div(one, sqrt_val.into_float_value(), "rsqrt").context(ArithmeticSnafu)?.into())
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
    Ok(builder
        .build_float_div(sin_val.into_float_value(), cos_val.into_float_value(), "tan")
        .context(ArithmeticSnafu)?
        .into())
}

fn codegen_reciprocal<'ctx>(src: BasicValueEnum<'ctx>, builder: &Builder<'ctx>) -> Result<BasicValueEnum<'ctx>> {
    let one = src.into_float_value().get_type().const_float(1.0);
    Ok(builder.build_float_div(one, src.into_float_value(), "recip").context(ArithmeticSnafu)?.into())
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

fn codegen_reduce_op<'ctx>(
    reduce_op: ReduceOp,
    acc: BasicValueEnum<'ctx>,
    src: BasicValueEnum<'ctx>,
    dtype: &DType,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
) -> Result<BasicValueEnum<'ctx>> {
    let is_float = dtype.is_float();
    let is_signed = dtype.is_signed();

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
