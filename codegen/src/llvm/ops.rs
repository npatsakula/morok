//! Operation-specific code generation using inkwell.

use inkwell::{
    FloatPredicate, IntPredicate,
    builder::Builder,
    context::Context,
    intrinsics::Intrinsic,
    module::Module,
    values::{BasicValueEnum, CallSiteValue, FunctionValue, ValueKind},
};
use snafu::{OptionExt, ResultExt};
use std::sync::Arc;

use morok_dtype::DType;
use morok_ir::{AxisType, prelude::*};

use super::builders;
use super::error::*;
use super::helpers::{LoopContext, ValueMap};
use super::intrinsics;
use super::types::dtype_to_basic_type;

/// Generate a UOp and require it to produce a value.
///
/// Returns `NoValue` error if the UOp doesn't produce a value (e.g., STORE, SINK).
/// Use this when the IR contract requires the operand to have a value.
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

/// Generate LLVM IR for a UOp node.
pub fn codegen_uop<'ctx>(
    uop: &Arc<UOp>,
    context: &'ctx Context,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
    values: &mut ValueMap<'ctx>,
) -> Result<Option<BasicValueEnum<'ctx>>> {
    // Check if already generated (either has value or was processed with no value)
    if values.contains(uop.id) {
        return Ok(values.get(uop.id));
    }

    let result = match uop.op() {
        Op::Const(val_hash) => Some(codegen_const(val_hash.0, &uop.dtype(), context)?),
        Op::Unary(op, src) => {
            let src_val = require_value(src, context, module, builder, values)?;
            let src_val = auto_load_pointer(src_val, &src.dtype(), context, builder)?;
            Some(codegen_unary(*op, src_val, &uop.dtype(), context, module, builder)?)
        }
        Op::Binary(op, lhs, rhs) => {
            let lhs_val = require_value(lhs, context, module, builder, values)?;
            let rhs_val = require_value(rhs, context, module, builder, values)?;
            // Auto-load pointer values: if operands are pointers (from INDEX), load their values
            let lhs_val = auto_load_pointer(lhs_val, &lhs.dtype(), context, builder)?;
            let rhs_val = auto_load_pointer(rhs_val, &rhs.dtype(), context, builder)?;
            Some(codegen_binary(*op, lhs_val, rhs_val, &uop.dtype(), context, module, builder)?)
        }
        Op::Ternary(op, a, b, c) => {
            let a_val = require_value(a, context, module, builder, values)?;
            let b_val = require_value(b, context, module, builder, values)?;
            let c_val = require_value(c, context, module, builder, values)?;
            // Auto-load pointer values for ternary operations
            let a_val = auto_load_pointer(a_val, &a.dtype(), context, builder)?;
            let b_val = auto_load_pointer(b_val, &b.dtype(), context, builder)?;
            let c_val = auto_load_pointer(c_val, &c.dtype(), context, builder)?;
            Some(codegen_ternary(*op, a_val, b_val, c_val, &uop.dtype(), context, module, builder)?)
        }
        Op::Cast { src, dtype } => {
            let src_val = require_value(src, context, module, builder, values)?;
            let src_val = auto_load_pointer(src_val, &src.dtype(), context, builder)?;
            Some(codegen_cast(src_val, &src.dtype(), dtype, context, builder)?)
        }
        Op::Load { buffer, index } => {
            let buffer_ptr = require_value(buffer, context, module, builder, values)?;
            let index_val = require_value(index, context, module, builder, values)?;
            Some(codegen_load(buffer_ptr, index_val, &uop.dtype(), context, builder)?)
        }
        Op::Store { buffer, index, value } => {
            let buffer_ptr = require_value(buffer, context, module, builder, values)?;
            let index_val = require_value(index, context, module, builder, values)?;
            let value_val = require_value(value, context, module, builder, values)?;
            codegen_store(buffer_ptr, index_val, value_val, builder)?;
            None // Store doesn't produce a value
        }
        Op::Buffer { .. } => {
            // BUFFER operations should be handled by the renderer and added to ValueMap before codegen starts.
            return NotInValueMapSnafu { what: "Buffer", id: uop.id }.fail();
        }
        Op::DefineGlobal(_) | Op::DefineLocal(_) => {
            // DEFINE_GLOBAL/LOCAL should already be in ValueMap as function parameters.
            return NotInValueMapSnafu { what: "DefineGlobal/Local", id: uop.id }.fail();
        }
        Op::DefineVar { .. } => {
            // DEFINE_VAR should already be in ValueMap as a function parameter.
            return NotInValueMapSnafu { what: "DefineVar", id: uop.id }.fail();
        }
        Op::Bind { var, .. } => {
            // BIND evaluates to its variable value (the variable parameter).
            let var_value = values.get(var.id).context(NotInValueMapSnafu { what: "BIND variable", id: var.id })?;
            Some(var_value)
        }
        Op::Sink { sources } => {
            // SINK is a graph termination marker - evaluate all sources for side effects
            // (like STORE), but doesn't produce a value itself.
            for src in sources {
                codegen_uop(src, context, module, builder, values)?;
            }
            None // SINK doesn't produce a value
        }
        Op::Index { buffer, indices, gate: None } => {
            // INDEX computes pointer arithmetic: buffer + sum(index * stride)
            // For now, handle single-index case (common for 1D buffers)
            let buffer_ptr = require_value(buffer, context, module, builder, values)?;

            if indices.len() == 1 {
                let index_val = require_value(&indices[0], context, module, builder, values)?;

                // GEP needs the element type, not the pointer type
                // INDEX dtype is Ptr<element_type>, so extract the base type
                let element_type = match uop.dtype() {
                    DType::Ptr { base, .. } => dtype_to_basic_type(&base, context)?,
                    other => dtype_to_basic_type(&other, context)?,
                };
                let ptr = unsafe {
                    builder
                        .build_gep(element_type, buffer_ptr.into_pointer_value(), &[index_val.into_int_value()], "idx")
                        .context(BuildGepSnafu)?
                };
                Some(ptr.into())
            } else {
                // Multi-index case - compute linear offset
                // offset = sum(indices[i] * strides[i])
                // For now, return error - can be implemented when needed
                return UnsupportedSnafu { what: "Multi-index INDEX" }.fail();
            }
        }
        Op::Index { gate: Some(_), .. } => {
            // Gated INDEX - conditional access
            return UnsupportedSnafu { what: "Gated INDEX" }.fail();
        }
        Op::PointerIndex { ptr, offset } => {
            // POINTER_INDEX: ptr + offset (pointer arithmetic)
            let ptr_val = require_value(ptr, context, module, builder, values)?;
            let offset_val = require_value(offset, context, module, builder, values)?;

            // GEP with byte offset
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
            Some(result.into())
        }
        Op::End { computation, ranges } => {
            // Generate the computation (loop body code)
            codegen_uop(computation, context, module, builder, values)?;

            // Close each range's loop (process innermost to outermost)
            // Skip Outer ranges - they don't generate loops
            for range_uop in ranges.iter().rev() {
                // Check if this is an Outer range (skip loop closing)
                if let Op::Range { axis_type: AxisType::Outer, .. } = range_uop.op() {
                    continue;
                }

                // Get the loop context for this range
                let loop_ctx =
                    values.get_loop(range_uop.id).context(LoopContextNotFoundSnafu { id: range_uop.id })?.clone();

                // Branch from current position (end of body) to footer
                builder.build_unconditional_branch(loop_ctx.footer_block).context(BuildBranchSnafu)?;

                // Footer block: complete PHI and branch back to latch
                builder.position_at_end(loop_ctx.footer_block);

                // Add incoming edge to PHI: incremented value from footer
                loop_ctx.phi.add_incoming(&[(&loop_ctx.incremented, loop_ctx.footer_block)]);

                // Branch back to latch
                builder.build_unconditional_branch(loop_ctx.latch_block).context(BuildBranchSnafu)?;

                // Position at exit block for code after loop
                builder.position_at_end(loop_ctx.exit_block);
            }

            None // END doesn't produce a value
        }
        Op::Range { end, axis_id, axis_type } => {
            // Outer ranges are kernel-level scheduling ranges that don't become
            // loops inside the kernel. They're handled by the scheduler externally.
            if *axis_type == AxisType::Outer {
                return Ok(Some(require_value(end, context, module, builder, values)?));
            }

            // Reduce ranges are handled internally by REDUCE codegen.
            // Skip them here - REDUCE will set up its own loops and store the counter.
            if *axis_type == AxisType::Reduce {
                return Ok(Some(require_value(end, context, module, builder, values)?));
            }

            // Get the function for creating basic blocks
            let current_block = builder.get_insert_block().context(NoInsertBlockSnafu)?;
            let function = current_block.get_parent().context(NoParentFunctionSnafu)?;

            // Generate unique block names using axis_id
            let id = axis_id.value();
            let entry_name = format!("loop_entry_{}", id);
            let latch_name = format!("loop_latch_{}", id);
            let body_name = format!("loop_body_{}", id);
            let footer_name = format!("loop_footer_{}", id);
            let exit_name = format!("loop_exit_{}", id);

            // Create all basic blocks
            let entry_block = context.append_basic_block(function, &entry_name);
            let latch_block = context.append_basic_block(function, &latch_name);
            let body_block = context.append_basic_block(function, &body_name);
            let footer_block = context.append_basic_block(function, &footer_name);
            let exit_block = context.append_basic_block(function, &exit_name);

            // Generate end value BEFORE branching (in current block)
            let end_val = require_value(end, context, module, builder, values)?;

            // Branch from current position to entry
            builder.build_unconditional_branch(entry_block).context(BuildBranchSnafu)?;

            // Entry block: just branch to latch
            builder.position_at_end(entry_block);
            builder.build_unconditional_branch(latch_block).context(BuildBranchSnafu)?;

            // Latch block: PHI, increment, condition, conditional branch
            builder.position_at_end(latch_block);

            // DType::Index maps to i64
            let counter_type = context.i64_type();

            // Create PHI node for loop counter
            let phi = builder.build_phi(counter_type, &format!("i{}", id)).context(BuildPhiSnafu)?;

            // Add incoming edge: 0 from entry block
            let zero = counter_type.const_int(0, false);
            phi.add_incoming(&[(&zero, entry_block)]);

            let counter_val = phi.as_basic_value().into_int_value();

            // Increment: counter + 1
            let one = counter_type.const_int(1, false);
            let incremented =
                builder.build_int_add(counter_val, one, &format!("i{}_next", id)).context(ArithmeticSnafu)?;

            // Cast end_val to i64 if needed (might be i32)
            let end_i64 = if end_val.is_int_value() {
                let end_int = end_val.into_int_value();
                if end_int.get_type() != counter_type {
                    builder.build_int_z_extend(end_int, counter_type, "end_ext").context(CastSnafu)?
                } else {
                    end_int
                }
            } else {
                return RangeEndNotIntegerSnafu { actual: format!("{:?}", end_val.get_type()) }.fail();
            };

            // Condition: counter < end (unsigned comparison)
            let cmp = builder
                .build_int_compare(IntPredicate::ULT, counter_val, end_i64, &format!("i{}_cmp", id))
                .context(ComparisonSnafu)?;

            // Conditional branch: if counter < end, go to body, else exit
            builder.build_conditional_branch(cmp, body_block, exit_block).context(BuildBranchSnafu)?;

            // Position builder at body block for loop body code
            builder.position_at_end(body_block);

            // Store loop context for END to complete
            values.insert_loop(uop.id, LoopContext { latch_block, footer_block, exit_block, phi, incremented });

            // RANGE produces the loop counter value
            Some(counter_val.into())
        }
        Op::Barrier { src, deps: _ } => {
            // BARRIER ensures dependencies complete before src
            // For CPU execution, dependencies are implicit - just evaluate src
            codegen_uop(src, context, module, builder, values)?
        }
        Op::Reduce { src, ranges, reduce_op } => {
            // REDUCE accumulates values across reduction ranges.
            // Uses a simple loop structure with accumulator PHI.
            //
            // For a single range, the structure is:
            //   current_block:
            //     branch to header
            //   header:
            //     acc_phi = phi [identity, current_block], [new_acc, body]
            //     counter_phi = phi [0, current_block], [inc, body]
            //     cmp = counter < end
            //     br cmp, body, exit
            //   body:
            //     src_val = compute(src)
            //     new_acc = reduce_op(acc_phi, src_val)
            //     inc = counter + 1
            //     br header
            //   exit:
            //     use acc_phi

            // If no ranges, just return the source value (no reduction needed)
            if ranges.is_empty() {
                return codegen_uop(src, context, module, builder, values);
            }

            let result_dtype = uop.dtype();
            let llvm_type = dtype_to_basic_type(&result_dtype, context)?;

            // Get function for creating basic blocks
            let current_block = builder.get_insert_block().context(NoInsertBlockSnafu)?;
            let function = current_block.get_parent().context(NoParentFunctionSnafu)?;

            // Generate identity value for the reduce operation
            let identity = codegen_reduce_identity(*reduce_op, &result_dtype, context)?;

            // For simplicity, handle single range first (most common case)
            // TODO: Support nested ranges if needed
            let reduce_id = uop.id;

            // Filter to only Loop/Reduce ranges (skip Outer)
            let loop_ranges: Vec<_> =
                ranges
                    .iter()
                    .filter(|r| {
                        if let Op::Range { axis_type, .. } = r.op() { *axis_type != AxisType::Outer } else { false }
                    })
                    .collect();

            // Handle Outer ranges - just evaluate them
            for range_uop in ranges.iter() {
                if let Op::Range { end, axis_type: AxisType::Outer, .. } = range_uop.op() {
                    let end_val = require_value(end, context, module, builder, values)?;
                    values.insert(range_uop.id, end_val);
                }
            }

            // If no loop ranges after filtering, just return source
            if loop_ranges.is_empty() {
                return codegen_uop(src, context, module, builder, values);
            }

            // Create basic blocks
            let header_block = context.append_basic_block(function, &format!("reduce_header_{}", reduce_id));
            let body_block = context.append_basic_block(function, &format!("reduce_body_{}", reduce_id));
            let exit_block = context.append_basic_block(function, &format!("reduce_exit_{}", reduce_id));

            // Branch from current block to header
            builder.build_unconditional_branch(header_block).context(BuildBranchSnafu)?;

            // Build header block
            builder.position_at_end(header_block);

            // Create accumulator PHI
            let acc_phi = builder.build_phi(llvm_type, "acc").context(BuildPhiSnafu)?;
            acc_phi.add_incoming(&[(&identity, current_block)]);

            // For now, support single range reduction
            // Get the first (innermost) range
            let range_uop = loop_ranges[0];
            let Op::Range { end, axis_id, .. } = range_uop.op() else {
                return UnsupportedSnafu { what: "REDUCE range must be RANGE op" }.fail();
            };

            // Get end value
            let end_val = require_value(end, context, module, builder, values)?;

            // Create counter PHI
            let counter_type = context.i64_type();
            let counter_phi =
                builder.build_phi(counter_type, &format!("reduce_i_{}", axis_id.value())).context(BuildPhiSnafu)?;
            let zero = counter_type.const_int(0, false);
            counter_phi.add_incoming(&[(&zero, current_block)]);

            let counter_val = counter_phi.as_basic_value().into_int_value();

            // Condition: counter < end
            let cmp = builder
                .build_int_compare(IntPredicate::ULT, counter_val, end_val.into_int_value(), "cmp")
                .context(ComparisonSnafu)?;

            // Conditional branch: if counter < end, go to body, else exit
            builder.build_conditional_branch(cmp, body_block, exit_block).context(BuildBranchSnafu)?;

            // Build body block
            builder.position_at_end(body_block);

            // Store counter value for this range (so src can reference it)
            values.insert(range_uop.id, counter_val.into());

            // Generate the source computation
            let src_val = require_value(src, context, module, builder, values)?;

            // Auto-load if source is pointer
            let src_val = auto_load_pointer(src_val, &src.dtype(), context, builder)?;

            // Get current accumulator value (from PHI)
            let acc_val = acc_phi.as_basic_value();

            // Apply reduce operation: new_acc = reduce_op(acc, src)
            let new_acc = codegen_reduce_op(*reduce_op, acc_val, src_val, &result_dtype, module, builder)?;

            // Increment counter
            let one = counter_type.const_int(1, false);
            let incremented = builder.build_int_add(counter_val, one, "inc").context(ArithmeticSnafu)?;

            // Branch back to header
            builder.build_unconditional_branch(header_block).context(BuildBranchSnafu)?;

            // Complete PHI nodes with incoming from body
            acc_phi.add_incoming(&[(&new_acc, body_block)]);
            counter_phi.add_incoming(&[(&incremented, body_block)]);

            // Position at exit block
            builder.position_at_end(exit_block);

            // Return the final accumulated value
            Some(acc_phi.as_basic_value())
        }
        Op::Noop => {
            // NOOP does nothing
            None
        }
        Op::Unique(_) | Op::Device(_) => {
            // Metadata operations that don't produce code
            // UNIQUE is for buffer identity tracking
            // DEVICE specifies target device (irrelevant at codegen time)
            None
        }
        Op::Vectorize { elements } => {
            // VECTORIZE creates a vector from scalar elements
            // For single-element, just return the element's value
            if elements.len() == 1 {
                codegen_uop(&elements[0], context, module, builder, values)?
            } else {
                // Multi-element vectorization - create LLVM vector
                // For now, just evaluate all elements (actual vectorization handled later)
                let mut last_val = None;
                for elem in elements {
                    last_val = codegen_uop(elem, context, module, builder, values)?;
                }
                last_val
            }
        }
        Op::Reshape { .. }
        | Op::Permute { .. }
        | Op::Expand { .. }
        | Op::Pad { .. }
        | Op::Shrink { .. }
        | Op::Flip { .. } => {
            // Movement ops MUST be eliminated by rangeify before reaching codegen.
            // If we get here, it means:
            // 1. transform_single_source didn't wrap the movement op with INDEX
            // 2. movement_op_patterns didn't push the movement through INDEX
            // 3. The pipeline has a bug
            return UnsupportedSnafu { what: "Movement ops must be eliminated by rangeify" }.fail();
        }
        _ => {
            return UnsupportedSnafu { what: "Unknown UOp operation" }.fail();
        }
    };

    // Store result for future lookups
    if let Some(val) = result {
        values.insert(uop.id, val);
    } else {
        // Mark as processed even with no value (prevents re-processing of END, SINK, etc.)
        values.mark_processed(uop.id);
    }

    Ok(result)
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Extract BasicValueEnum from CallSiteValue.
fn extract_value<'ctx>(call_site: CallSiteValue<'ctx>) -> Result<BasicValueEnum<'ctx>> {
    match call_site.try_as_basic_value() {
        ValueKind::Basic(v) => Ok(v),
        ValueKind::Instruction(_) => ValueExtractionFailedSnafu { expected: "BasicValue" }.fail(),
    }
}

/// Get LLVM intrinsic function.
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

/// Call an LLVM intrinsic and extract the result.
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

/// Get LLVM type suffix for intrinsics (e.g., "f32" for Float32).
fn get_type_suffix(dtype: &morok_dtype::DType) -> Result<String> {
    match dtype.scalar() {
        Some(morok_dtype::ScalarDType::Float16) => Ok("f16".to_string()),
        Some(morok_dtype::ScalarDType::BFloat16) => Ok("f16".to_string()),
        Some(morok_dtype::ScalarDType::Float32) => Ok("f32".to_string()),
        Some(morok_dtype::ScalarDType::Float64) => Ok("f64".to_string()),
        _ => UnsupportedIntrinsicTypeSnafu { dtype: format!("{:?}", dtype) }.fail(),
    }
}

/// Generate identity value for a reduce operation.
///
/// Identity values:
/// - Add: 0 (0 + x = x)
/// - Mul: 1 (1 * x = x)
/// - Max: -INF for float, MIN for signed int, 0 for unsigned int
/// - Min: +INF for float, MAX for signed int, MAX for unsigned int
fn codegen_reduce_identity<'ctx>(
    reduce_op: ReduceOp,
    dtype: &DType,
    context: &'ctx Context,
) -> Result<BasicValueEnum<'ctx>> {
    let llvm_type = dtype_to_basic_type(dtype, context)?;
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
                // Signed MIN (e.g., i32::MIN)
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
                // Unsigned: 0 is identity for max (any value >= 0)
                Ok(llvm_type.into_int_type().const_int(0, false).into())
            }
        }
        ReduceOp::Min => {
            if is_float {
                Ok(llvm_type.into_float_type().const_float(f64::INFINITY).into())
            } else if is_signed {
                // Signed MAX (e.g., i32::MAX)
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
                // Unsigned MAX
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

/// Apply reduce operation to accumulator and source value.
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
        ReduceOp::Add => builders::build_add(builder, acc, src, is_float),
        ReduceOp::Mul => builders::build_mul(builder, acc, src, is_float),
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
// Codegen Functions
// ============================================================================

/// Generate LLVM constant value.
fn codegen_const<'ctx>(
    val: ConstValue,
    dtype: &morok_dtype::DType,
    context: &'ctx Context,
) -> Result<BasicValueEnum<'ctx>> {
    let llvm_type = dtype_to_basic_type(dtype, context)?;

    let value = match val {
        ConstValue::Int(i) => llvm_type.into_int_type().const_int(i as u64, true).into(),
        ConstValue::UInt(u) => llvm_type.into_int_type().const_int(u, false).into(),
        ConstValue::Float(f) => llvm_type.into_float_type().const_float(f).into(),
        ConstValue::Bool(b) => context.bool_type().const_int(b as u64, false).into(),
    };

    Ok(value)
}

/// Generate code for unary operations.
fn codegen_unary<'ctx>(
    op: UnaryOp,
    src: BasicValueEnum<'ctx>,
    result_dtype: &morok_dtype::DType,
    _context: &'ctx Context,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
) -> Result<BasicValueEnum<'ctx>> {
    let is_float = result_dtype.is_float();

    // Try simple intrinsic lookup first (Sqrt, Exp, Log, Sin, Cos, Floor, Ceil, Round, etc.)
    if let Some((intrinsic_base, name)) = intrinsics::unary_float_intrinsic(op) {
        let suffix = get_type_suffix(result_dtype)?;
        return call_intrinsic(&format!("{}.{}", intrinsic_base, suffix), &[src], name, module, builder);
    }

    match op {
        UnaryOp::Neg => builders::build_neg(builder, src, is_float),
        UnaryOp::Square => builders::build_mul(builder, src, src, is_float),

        UnaryOp::Abs => {
            if is_float {
                let suffix = get_type_suffix(result_dtype)?;
                call_intrinsic(&format!("llvm.fabs.{}", suffix), &[src], "abs", module, builder)
            } else {
                // Integer abs: x < 0 ? -x : x
                let int_val = src.into_int_value();
                let zero = int_val.get_type().const_zero();
                let is_neg =
                    builder.build_int_compare(IntPredicate::SLT, int_val, zero, "is_neg").context(ComparisonSnafu)?;
                let neg_val = builder.build_int_neg(int_val, "neg_val").context(ArithmeticSnafu)?;
                Ok(builder.build_select(is_neg, neg_val, int_val, "abs").context(BuildSelectSnafu)?)
            }
        }

        UnaryOp::Rsqrt => {
            // rsqrt = 1 / sqrt(x)
            let suffix = get_type_suffix(result_dtype)?;
            let sqrt_val = call_intrinsic(&format!("llvm.sqrt.{}", suffix), &[src], "sqrt", module, builder)?;
            let one = src.into_float_value().get_type().const_float(1.0);
            Ok(builder.build_float_div(one, sqrt_val.into_float_value(), "rsqrt").context(ArithmeticSnafu)?.into())
        }

        UnaryOp::Tan => {
            // tan(x) = sin(x) / cos(x)
            let suffix = get_type_suffix(result_dtype)?;
            let sin_val = call_intrinsic(&format!("llvm.sin.{}", suffix), &[src], "sin", module, builder)?;
            let cos_val = call_intrinsic(&format!("llvm.cos.{}", suffix), &[src], "cos", module, builder)?;
            Ok(builder
                .build_float_div(sin_val.into_float_value(), cos_val.into_float_value(), "tan")
                .context(ArithmeticSnafu)?
                .into())
        }

        UnaryOp::Reciprocal => {
            let one = src.into_float_value().get_type().const_float(1.0);
            Ok(builder.build_float_div(one, src.into_float_value(), "recip").context(ArithmeticSnafu)?.into())
        }

        UnaryOp::Trunc => {
            if is_float {
                let suffix = get_type_suffix(result_dtype)?;
                call_intrinsic(&format!("llvm.trunc.{}", suffix), &[src], "trunc", module, builder)
            } else {
                Ok(src) // No-op for integers
            }
        }

        UnaryOp::Sign => codegen_sign(src, is_float, builder),

        UnaryOp::Erf => {
            // Call external erf() from libm
            let float_type = src.into_float_value().get_type();
            let fn_type = float_type.fn_type(&[float_type.into()], false);
            let fn_val = module.add_function("erf", fn_type, None);
            let call_site = builder
                .build_call(fn_val, &[src.into()], "erf")
                .context(BuildCallSnafu { intrinsic: "erf".to_string() })?;
            extract_value(call_site)
        }

        _ => UnsupportedSnafu { what: "Unknown unary operation" }.fail(),
    }
}

/// Generate code for sign operation: returns -1, 0, or 1.
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

/// Generate code for binary operations.
fn codegen_binary<'ctx>(
    op: BinaryOp,
    lhs: BasicValueEnum<'ctx>,
    rhs: BasicValueEnum<'ctx>,
    result_dtype: &morok_dtype::DType,
    _context: &'ctx Context,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
) -> Result<BasicValueEnum<'ctx>> {
    let is_float = result_dtype.is_float();
    let is_signed = result_dtype.is_signed();

    // Handle comparisons uniformly via table-driven dispatch
    if builders::is_comparison(op) {
        return builders::build_cmp(builder, op, lhs, rhs, is_float, is_signed);
    }

    match op {
        // Arithmetic with type dispatch
        BinaryOp::Add => builders::build_add(builder, lhs, rhs, is_float),
        BinaryOp::Sub => builders::build_sub(builder, lhs, rhs, is_float),
        BinaryOp::Mul => builders::build_mul(builder, lhs, rhs, is_float),
        BinaryOp::Fdiv => Ok(builder
            .build_float_div(lhs.into_float_value(), rhs.into_float_value(), "fdiv")
            .context(ArithmeticSnafu)?
            .into()),
        BinaryOp::Idiv => builders::build_int_div(builder, lhs, rhs, is_signed),
        BinaryOp::Mod => builders::build_rem(builder, lhs, rhs, is_float, is_signed),

        // Intrinsics
        BinaryOp::Max => {
            let bits = result_dtype.bytes() * 8;
            let intrinsic = intrinsics::max_intrinsic(is_float, is_signed, bits);
            call_intrinsic(&intrinsic, &[lhs, rhs], "max", module, builder)
        }
        BinaryOp::Pow => {
            let suffix = get_type_suffix(result_dtype)?;
            call_intrinsic(&format!("llvm.pow.{}", suffix), &[lhs, rhs], "pow", module, builder)
        }

        // Bitwise operations (int only)
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

/// Generate code for ternary operations.
#[allow(clippy::too_many_arguments)]
fn codegen_ternary<'ctx>(
    op: TernaryOp,
    a: BasicValueEnum<'ctx>,
    b: BasicValueEnum<'ctx>,
    c: BasicValueEnum<'ctx>,
    result_dtype: &morok_dtype::DType,
    _context: &'ctx Context,
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

/// Generate code for cast operations.
fn codegen_cast<'ctx>(
    src: BasicValueEnum<'ctx>,
    src_dtype: &morok_dtype::DType,
    dst_dtype: &morok_dtype::DType,
    context: &'ctx Context,
    builder: &Builder<'ctx>,
) -> Result<BasicValueEnum<'ctx>> {
    let dst_type = dtype_to_basic_type(dst_dtype, context)?;

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

/// Generate a load instruction.
///
/// The `index` parameter can be either:
/// - A pointer (from INDEX operation) - load directly from it
/// - An integer offset - compute GEP then load
fn codegen_load<'ctx>(
    buffer_ptr: BasicValueEnum<'ctx>,
    index: BasicValueEnum<'ctx>,
    result_dtype: &morok_dtype::DType,
    context: &'ctx Context,
    builder: &Builder<'ctx>,
) -> Result<BasicValueEnum<'ctx>> {
    // Check if index is already a pointer (from INDEX operation)
    // or an integer offset that needs GEP computation
    let element_ptr = if index.is_pointer_value() {
        // INDEX already computed the pointer - use it directly
        index.into_pointer_value()
    } else {
        // Integer offset - compute GEP
        let ptr_val = buffer_ptr.into_pointer_value();
        let index_val = index.into_int_value();
        let element_type = dtype_to_basic_type(result_dtype, context)?;
        unsafe { builder.build_gep(element_type, ptr_val, &[index_val], "gep").context(BuildGepSnafu)? }
    };

    // Load from the pointer
    let load_type = dtype_to_basic_type(result_dtype, context)?;
    builder.build_load(load_type, element_ptr, "load").context(BuildLoadSnafu)
}

/// Generate a store instruction.
///
/// The `index` parameter can be either:
/// - A pointer (from INDEX operation) - store directly to it
/// - An integer offset - compute GEP then store
fn codegen_store<'ctx>(
    buffer_ptr: BasicValueEnum<'ctx>,
    index: BasicValueEnum<'ctx>,
    value: BasicValueEnum<'ctx>,
    builder: &Builder<'ctx>,
) -> Result<()> {
    // Check if index is already a pointer (from INDEX operation)
    // or an integer offset that needs GEP computation
    let element_ptr = if index.is_pointer_value() {
        // INDEX already computed the pointer - use it directly
        index.into_pointer_value()
    } else {
        // Integer offset - compute GEP
        let ptr_val = buffer_ptr.into_pointer_value();
        let index_val = index.into_int_value();
        let block = builder.get_insert_block().context(NoInsertBlockSnafu)?;
        unsafe {
            builder
                .build_gep(
                    // Use i8 type for opaque pointer arithmetic
                    block.get_context().i8_type(),
                    ptr_val,
                    &[index_val],
                    "gep",
                )
                .context(BuildGepSnafu)?
        }
    };

    // Store the value
    builder.build_store(element_ptr, value).context(BuildStoreSnafu)?;
    Ok(())
}

/// Auto-load a value if it's a pointer.
///
/// This handles the case where INDEX produces a pointer but we need a scalar value
/// for arithmetic operations. If the value is a pointer, we load from it.
fn auto_load_pointer<'ctx>(
    value: BasicValueEnum<'ctx>,
    dtype: &DType,
    context: &'ctx Context,
    builder: &Builder<'ctx>,
) -> Result<BasicValueEnum<'ctx>> {
    if value.is_pointer_value() {
        // Get the element type from the Ptr dtype
        let element_type = match dtype {
            DType::Ptr { base, .. } => dtype_to_basic_type(base, context)?,
            _ => return Ok(value), // Not a Ptr dtype, shouldn't happen
        };

        // Load from the pointer
        Ok(builder.build_load(element_type, value.into_pointer_value(), "autoload").context(BuildLoadSnafu)?)
    } else {
        Ok(value)
    }
}
