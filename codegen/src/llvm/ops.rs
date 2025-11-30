//! Operation-specific code generation using inkwell.

use crate::{Result, UnsupportedOpSnafu};
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::values::{BasicValueEnum, CallSiteValue, FunctionValue};
use inkwell::{FloatPredicate, IntPredicate};
use morok_dtype::DType;
use morok_ir::{AxisType, BinaryOp, ConstValue, Op, ReduceOp, TernaryOp, UOp, UnaryOp};
use snafu::ensure;
use std::rc::Rc;

use super::helpers::{LoopContext, ValueMap};
use super::types::dtype_to_basic_type;

/// Generate LLVM IR for a UOp node.
pub fn codegen_uop<'ctx>(
    uop: &Rc<UOp>,
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
            let src_val = codegen_uop(src, context, module, builder, values)?
                .ok_or_else(|| crate::Error::Missing { what: "source value for unary op".to_string() })?;
            // Auto-load if source is pointer and we need a scalar value
            let src_val = auto_load_pointer(src_val, &src.dtype(), context, builder)?;
            Some(codegen_unary(*op, src_val, &uop.dtype(), context, module, builder)?)
        }
        Op::Binary(op, lhs, rhs) => {
            let lhs_val = codegen_uop(lhs, context, module, builder, values)?
                .ok_or_else(|| crate::Error::Missing { what: "lhs value for binary op".to_string() })?;
            let rhs_val = codegen_uop(rhs, context, module, builder, values)?
                .ok_or_else(|| crate::Error::Missing { what: "rhs value for binary op".to_string() })?;

            // Auto-load pointer values: if operands are pointers (from INDEX), load their values
            // This handles the case where INDEX produces pointers but we need scalar values for arithmetic
            let lhs_val = auto_load_pointer(lhs_val, &lhs.dtype(), context, builder)?;
            let rhs_val = auto_load_pointer(rhs_val, &rhs.dtype(), context, builder)?;

            Some(codegen_binary(*op, lhs_val, rhs_val, &uop.dtype(), context, module, builder)?)
        }
        Op::Ternary(op, a, b, c) => {
            let a_val = codegen_uop(a, context, module, builder, values)?
                .ok_or_else(|| crate::Error::Missing { what: "first value for ternary op".to_string() })?;
            let b_val = codegen_uop(b, context, module, builder, values)?
                .ok_or_else(|| crate::Error::Missing { what: "second value for ternary op".to_string() })?;
            let c_val = codegen_uop(c, context, module, builder, values)?
                .ok_or_else(|| crate::Error::Missing { what: "third value for ternary op".to_string() })?;
            // Auto-load pointer values for ternary operations
            let a_val = auto_load_pointer(a_val, &a.dtype(), context, builder)?;
            let b_val = auto_load_pointer(b_val, &b.dtype(), context, builder)?;
            let c_val = auto_load_pointer(c_val, &c.dtype(), context, builder)?;
            Some(codegen_ternary(*op, a_val, b_val, c_val, &uop.dtype(), context, module, builder)?)
        }
        Op::Cast { src, dtype } => {
            let src_val = codegen_uop(src, context, module, builder, values)?
                .ok_or_else(|| crate::Error::Missing { what: "source value for cast".to_string() })?;
            // Auto-load pointer value for cast
            let src_val = auto_load_pointer(src_val, &src.dtype(), context, builder)?;
            Some(codegen_cast(src_val, &src.dtype(), dtype, context, builder)?)
        }
        Op::Load { buffer, index } => {
            let buffer_ptr = codegen_uop(buffer, context, module, builder, values)?
                .ok_or_else(|| crate::Error::Missing { what: "buffer pointer for load".to_string() })?;
            let index_val = codegen_uop(index, context, module, builder, values)?
                .ok_or_else(|| crate::Error::Missing { what: "index value for load".to_string() })?;
            Some(codegen_load(buffer_ptr, index_val, &uop.dtype(), context, builder)?)
        }
        Op::Store { buffer, index, value } => {
            let buffer_ptr = codegen_uop(buffer, context, module, builder, values)?
                .ok_or_else(|| crate::Error::Missing { what: "buffer pointer for store".to_string() })?;
            let index_val = codegen_uop(index, context, module, builder, values)?
                .ok_or_else(|| crate::Error::Missing { what: "index value for store".to_string() })?;
            let value_val = codegen_uop(value, context, module, builder, values)?
                .ok_or_else(|| crate::Error::Missing { what: "value for store".to_string() })?;
            codegen_store(buffer_ptr, index_val, value_val, builder)?;
            None // Store doesn't produce a value
        }
        Op::Buffer { .. } => {
            // BUFFER operations should be handled by the renderer and added to ValueMap
            // before codegen starts. If we reach here, it means the buffer wasn't registered.
            return Err(crate::Error::Missing { what: format!("Buffer UOp {} should be in ValueMap", uop.id) });
        }
        Op::DefineGlobal(_) | Op::DefineLocal(_) => {
            // DEFINE_GLOBAL/LOCAL should already be in ValueMap as function parameters.
            // If we reach here, the kernel arguments weren't properly registered.
            return Err(crate::Error::Missing {
                what: format!("DefineGlobal/Local UOp {} should be in ValueMap as function parameter", uop.id),
            });
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
            let buffer_ptr = codegen_uop(buffer, context, module, builder, values)?
                .ok_or_else(|| crate::Error::Missing { what: "buffer pointer for index".to_string() })?;

            if indices.len() == 1 {
                let index_val = codegen_uop(&indices[0], context, module, builder, values)?
                    .ok_or_else(|| crate::Error::Missing { what: "index value".to_string() })?;

                // GEP needs the element type, not the pointer type
                // INDEX dtype is Ptr<element_type>, so extract the base type
                let element_type = match uop.dtype() {
                    DType::Ptr { base, .. } => dtype_to_basic_type(&base, context),
                    other => dtype_to_basic_type(&other, context),
                };
                let ptr = unsafe {
                    builder
                        .build_gep(element_type, buffer_ptr.into_pointer_value(), &[index_val.into_int_value()], "idx")
                        .map_err(|e| crate::Error::LlvmError { reason: format!("build_gep for index: {}", e) })?
                };
                Some(ptr.into())
            } else {
                // Multi-index case - compute linear offset
                // offset = sum(indices[i] * strides[i])
                // For now, return error - can be implemented when needed
                return Err(crate::Error::UnsupportedOp { op: "Multi-index INDEX".to_string() });
            }
        }
        Op::Index { gate: Some(_), .. } => {
            // Gated INDEX - conditional access
            return Err(crate::Error::UnsupportedOp { op: "Gated INDEX".to_string() });
        }
        Op::PointerIndex { ptr, offset } => {
            // POINTER_INDEX: ptr + offset (pointer arithmetic)
            let ptr_val = codegen_uop(ptr, context, module, builder, values)?
                .ok_or_else(|| crate::Error::Missing { what: "ptr for pointer_index".to_string() })?;
            let offset_val = codegen_uop(offset, context, module, builder, values)?
                .ok_or_else(|| crate::Error::Missing { what: "offset for pointer_index".to_string() })?;

            // GEP with byte offset
            let result = unsafe {
                builder
                    .build_gep(
                        context.i8_type(),
                        ptr_val.into_pointer_value(),
                        &[offset_val.into_int_value()],
                        "ptr_idx",
                    )
                    .map_err(|e| crate::Error::LlvmError { reason: format!("build_gep for pointer_index: {}", e) })?
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
                let loop_ctx = values.get_loop(range_uop.id).ok_or_else(|| crate::Error::Missing {
                    what: format!("loop context for range id {}", range_uop.id),
                })?;
                // Clone to avoid borrow issues
                let loop_ctx = loop_ctx.clone();

                // Branch from current position (end of body) to footer
                builder
                    .build_unconditional_branch(loop_ctx.footer_block)
                    .map_err(|e| crate::Error::LlvmError { reason: format!("branch to footer: {}", e) })?;

                // Footer block: complete PHI and branch back to latch
                builder.position_at_end(loop_ctx.footer_block);

                // Add incoming edge to PHI: incremented value from footer
                loop_ctx.phi.add_incoming(&[(&loop_ctx.incremented, loop_ctx.footer_block)]);

                // Branch back to latch
                builder
                    .build_unconditional_branch(loop_ctx.latch_block)
                    .map_err(|e| crate::Error::LlvmError { reason: format!("branch to latch: {}", e) })?;

                // Position at exit block for code after loop
                builder.position_at_end(loop_ctx.exit_block);
            }

            None // END doesn't produce a value
        }
        Op::Range { end, axis_id, axis_type } => {
            // Outer ranges are kernel-level scheduling ranges that don't become
            // loops inside the kernel. They're handled by the scheduler externally.
            // For these, just return the end value (like scalar execution).
            if *axis_type == AxisType::Outer {
                let end_val = codegen_uop(end, context, module, builder, values)?
                    .ok_or_else(|| crate::Error::Missing { what: "end value for outer range".to_string() })?;
                return Ok(Some(end_val));
            }

            // Reduce ranges are handled internally by REDUCE codegen.
            // Skip them here - REDUCE will set up its own loops and store the counter.
            if *axis_type == AxisType::Reduce {
                // Just evaluate the end value and return it
                // REDUCE will handle the actual loop structure
                let end_val = codegen_uop(end, context, module, builder, values)?
                    .ok_or_else(|| crate::Error::Missing { what: "end value for reduce range".to_string() })?;
                return Ok(Some(end_val));
            }

            // Get the function for creating basic blocks
            let current_block = builder
                .get_insert_block()
                .ok_or_else(|| crate::Error::LlvmError { reason: "No insert block for RANGE".to_string() })?;
            let function = current_block
                .get_parent()
                .ok_or_else(|| crate::Error::LlvmError { reason: "No parent function for RANGE".to_string() })?;

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
            let end_val = codegen_uop(end, context, module, builder, values)?
                .ok_or_else(|| crate::Error::Missing { what: "end value for range".to_string() })?;

            // Branch from current position to entry
            builder
                .build_unconditional_branch(entry_block)
                .map_err(|e| crate::Error::LlvmError { reason: format!("branch to entry: {}", e) })?;

            // Entry block: just branch to latch
            builder.position_at_end(entry_block);
            builder
                .build_unconditional_branch(latch_block)
                .map_err(|e| crate::Error::LlvmError { reason: format!("branch to latch: {}", e) })?;

            // Latch block: PHI, increment, condition, conditional branch
            builder.position_at_end(latch_block);

            // DType::Index maps to i64
            let counter_type = context.i64_type();

            // Create PHI node for loop counter
            let phi = builder
                .build_phi(counter_type, &format!("i{}", id))
                .map_err(|e| crate::Error::LlvmError { reason: format!("build_phi: {}", e) })?;

            // Add incoming edge: 0 from entry block
            let zero = counter_type.const_int(0, false);
            phi.add_incoming(&[(&zero, entry_block)]);

            let counter_val = phi.as_basic_value().into_int_value();

            // Increment: counter + 1
            let one = counter_type.const_int(1, false);
            let incremented = builder
                .build_int_add(counter_val, one, &format!("i{}_next", id))
                .map_err(|e| crate::Error::LlvmError { reason: format!("build_int_add: {}", e) })?;

            // Cast end_val to i64 if needed (might be i32)
            let end_i64 = if end_val.is_int_value() {
                let end_int = end_val.into_int_value();
                if end_int.get_type() != counter_type {
                    builder
                        .build_int_z_extend(end_int, counter_type, "end_ext")
                        .map_err(|e| crate::Error::LlvmError { reason: format!("build_int_z_extend: {}", e) })?
                } else {
                    end_int
                }
            } else {
                return Err(crate::Error::TypeError {
                    reason: "RANGE end value must be integer".to_string(),
                });
            };

            // Condition: counter < end (unsigned comparison)
            let cmp = builder
                .build_int_compare(IntPredicate::ULT, counter_val, end_i64, &format!("i{}_cmp", id))
                .map_err(|e| crate::Error::LlvmError { reason: format!("build_int_compare: {}", e) })?;

            // Conditional branch: if counter < end, go to body, else exit
            builder
                .build_conditional_branch(cmp, body_block, exit_block)
                .map_err(|e| crate::Error::LlvmError { reason: format!("build_conditional_branch: {}", e) })?;

            // Position builder at body block for loop body code
            builder.position_at_end(body_block);

            // Store loop context for END to complete
            values.insert_loop(
                uop.id,
                LoopContext { latch_block, footer_block, exit_block, phi, incremented },
            );

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
            let llvm_type = dtype_to_basic_type(&result_dtype, context);

            // Get function for creating basic blocks
            let current_block = builder
                .get_insert_block()
                .ok_or_else(|| crate::Error::LlvmError { reason: "No insert block for REDUCE".to_string() })?;
            let function = current_block
                .get_parent()
                .ok_or_else(|| crate::Error::LlvmError { reason: "No parent function for REDUCE".to_string() })?;

            // Generate identity value for the reduce operation
            let identity = codegen_reduce_identity(*reduce_op, &result_dtype, context)?;

            // For simplicity, handle single range first (most common case)
            // TODO: Support nested ranges if needed
            let reduce_id = uop.id;

            // Filter to only Loop/Reduce ranges (skip Outer)
            let loop_ranges: Vec<_> = ranges
                .iter()
                .filter(|r| {
                    if let Op::Range { axis_type, .. } = r.op() {
                        *axis_type != AxisType::Outer
                    } else {
                        false
                    }
                })
                .collect();

            // Handle Outer ranges - just evaluate them
            for range_uop in ranges.iter() {
                if let Op::Range { end, axis_type: AxisType::Outer, .. } = range_uop.op() {
                    let end_val = codegen_uop(end, context, module, builder, values)?
                        .ok_or_else(|| crate::Error::Missing { what: "end value for outer range in reduce".to_string() })?;
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
            builder
                .build_unconditional_branch(header_block)
                .map_err(|e| crate::Error::LlvmError { reason: format!("build_unconditional_branch to reduce header: {}", e) })?;

            // Build header block
            builder.position_at_end(header_block);

            // Create accumulator PHI
            let acc_phi = builder
                .build_phi(llvm_type, "acc")
                .map_err(|e| crate::Error::LlvmError { reason: format!("build_phi for accumulator: {}", e) })?;
            acc_phi.add_incoming(&[(&identity, current_block)]);

            // For now, support single range reduction
            // Get the first (innermost) range
            let range_uop = loop_ranges[0];
            let Op::Range { end, axis_id, .. } = range_uop.op() else {
                return Err(crate::Error::UnsupportedOp {
                    op: format!("REDUCE range must be RANGE op, got {:?}", range_uop.op()),
                });
            };

            // Get end value
            let end_val = codegen_uop(end, context, module, builder, values)?
                .ok_or_else(|| crate::Error::Missing { what: "end value for reduce range".to_string() })?;

            // Create counter PHI
            let counter_type = context.i64_type();
            let counter_phi = builder
                .build_phi(counter_type, &format!("reduce_i_{}", axis_id.value()))
                .map_err(|e| crate::Error::LlvmError { reason: format!("build_phi for reduce counter: {}", e) })?;
            let zero = counter_type.const_int(0, false);
            counter_phi.add_incoming(&[(&zero, current_block)]);

            let counter_val = counter_phi.as_basic_value().into_int_value();

            // Condition: counter < end
            let cmp = builder
                .build_int_compare(IntPredicate::ULT, counter_val, end_val.into_int_value(), "cmp")
                .map_err(|e| crate::Error::LlvmError { reason: format!("build_int_compare for reduce: {}", e) })?;

            // Conditional branch: if counter < end, go to body, else exit
            builder
                .build_conditional_branch(cmp, body_block, exit_block)
                .map_err(|e| crate::Error::LlvmError { reason: format!("build_conditional_branch for reduce: {}", e) })?;

            // Build body block
            builder.position_at_end(body_block);

            // Store counter value for this range (so src can reference it)
            values.insert(range_uop.id, counter_val.into());

            // Generate the source computation
            let src_val = codegen_uop(src, context, module, builder, values)?
                .ok_or_else(|| crate::Error::Missing { what: "source value for reduce".to_string() })?;

            // Auto-load if source is pointer
            let src_val = auto_load_pointer(src_val, &src.dtype(), context, builder)?;

            // Get current accumulator value (from PHI)
            let acc_val = acc_phi.as_basic_value();

            // Apply reduce operation: new_acc = reduce_op(acc, src)
            let new_acc = codegen_reduce_op(*reduce_op, acc_val, src_val, &result_dtype, module, builder)?;

            // Increment counter
            let one = counter_type.const_int(1, false);
            let incremented = builder
                .build_int_add(counter_val, one, "inc")
                .map_err(|e| crate::Error::LlvmError { reason: format!("build_int_add for reduce increment: {}", e) })?;

            // Branch back to header
            builder
                .build_unconditional_branch(header_block)
                .map_err(|e| crate::Error::LlvmError { reason: format!("build_unconditional_branch to reduce header: {}", e) })?;

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
            return Err(crate::Error::UnsupportedOp {
                op: format!(
                    "Movement op {:?} should be eliminated by rangeify pipeline. \
                     Check transform.rs (is_movement_chain_on_buffer) and \
                     split_kernel.rs (movement_op_patterns integration).",
                    uop.op()
                ),
            });
        }
        _ => {
            ensure!(false, UnsupportedOpSnafu { op: format!("{:?}", uop.op()) });
            unreachable!()
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
fn extract_value<'ctx>(call_site: CallSiteValue<'ctx>) -> BasicValueEnum<'ctx> {
    match call_site.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        inkwell::values::ValueKind::Instruction(_) => {
            panic!("Expected basic value from intrinsic call")
        }
    }
}

/// Get LLVM intrinsic function.
fn get_intrinsic<'ctx>(
    name: &str,
    ret_type: BasicValueEnum<'ctx>,
    module: &Module<'ctx>,
) -> Result<FunctionValue<'ctx>> {
    use inkwell::intrinsics::Intrinsic;

    let intrinsic = Intrinsic::find(name)
        .ok_or_else(|| crate::Error::LlvmError { reason: format!("Intrinsic {} not found", name) })?;

    intrinsic
        .get_declaration(module, &[ret_type.get_type()])
        .ok_or_else(|| crate::Error::LlvmError { reason: format!("Failed to get declaration for {}", name) })
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
        .map_err(|e| crate::Error::LlvmError { reason: format!("build_call {}: {}", intrinsic_name, e) })?;
    Ok(extract_value(call_site))
}

/// Get LLVM type suffix for intrinsics (e.g., "f32" for Float32).
fn get_type_suffix(dtype: &morok_dtype::DType) -> Result<String> {
    match dtype.scalar() {
        Some(morok_dtype::ScalarDType::Float16) => Ok("f16".to_string()),
        Some(morok_dtype::ScalarDType::BFloat16) => Ok("f16".to_string()),
        Some(morok_dtype::ScalarDType::Float32) => Ok("f32".to_string()),
        Some(morok_dtype::ScalarDType::Float64) => Ok("f64".to_string()),
        _ => Err(crate::Error::TypeError { reason: format!("Type {:?} not supported for intrinsics", dtype) }),
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
    let llvm_type = dtype_to_basic_type(dtype, context);
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
        ReduceOp::Add => {
            if is_float {
                Ok(builder
                    .build_float_add(acc.into_float_value(), src.into_float_value(), "reduce_add")
                    .map_err(|e| crate::Error::LlvmError { reason: format!("build_float_add for reduce: {}", e) })?
                    .into())
            } else {
                Ok(builder
                    .build_int_add(acc.into_int_value(), src.into_int_value(), "reduce_add")
                    .map_err(|e| crate::Error::LlvmError { reason: format!("build_int_add for reduce: {}", e) })?
                    .into())
            }
        }
        ReduceOp::Mul => {
            if is_float {
                Ok(builder
                    .build_float_mul(acc.into_float_value(), src.into_float_value(), "reduce_mul")
                    .map_err(|e| crate::Error::LlvmError { reason: format!("build_float_mul for reduce: {}", e) })?
                    .into())
            } else {
                Ok(builder
                    .build_int_mul(acc.into_int_value(), src.into_int_value(), "reduce_mul")
                    .map_err(|e| crate::Error::LlvmError { reason: format!("build_int_mul for reduce: {}", e) })?
                    .into())
            }
        }
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
    let llvm_type = dtype_to_basic_type(dtype, context);

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

    match op {
        UnaryOp::Neg => {
            if is_float {
                Ok(builder
                    .build_float_neg(src.into_float_value(), "neg")
                    .map_err(|e| crate::Error::LlvmError { reason: format!("build_float_neg: {}", e) })?
                    .into())
            } else {
                Ok(builder
                    .build_int_neg(src.into_int_value(), "neg")
                    .map_err(|e| crate::Error::LlvmError { reason: format!("build_int_neg: {}", e) })?
                    .into())
            }
        }
        UnaryOp::Abs => {
            if is_float {
                let suffix = get_type_suffix(result_dtype)?;
                call_intrinsic(&format!("llvm.fabs.{}", suffix), &[src], "abs", module, builder)
            } else {
                // For integers: x < 0 ? -x : x
                let int_val = src.into_int_value();
                let zero = int_val.get_type().const_zero();
                let is_neg = builder
                    .build_int_compare(IntPredicate::SLT, int_val, zero, "is_neg")
                    .map_err(|e| crate::Error::LlvmError { reason: format!("build_int_compare: {}", e) })?;
                let neg_val = builder
                    .build_int_neg(int_val, "neg_val")
                    .map_err(|e| crate::Error::LlvmError { reason: format!("build_int_neg: {}", e) })?;
                Ok(builder
                    .build_select(is_neg, neg_val, int_val, "abs")
                    .map_err(|e| crate::Error::LlvmError { reason: format!("build_select: {}", e) })?)
            }
        }
        UnaryOp::Sqrt => {
            let suffix = get_type_suffix(result_dtype)?;
            call_intrinsic(&format!("llvm.sqrt.{}", suffix), &[src], "sqrt", module, builder)
        }
        UnaryOp::Rsqrt => {
            // rsqrt = 1 / sqrt(x)
            let suffix = get_type_suffix(result_dtype)?;
            let sqrt_val = call_intrinsic(&format!("llvm.sqrt.{}", suffix), &[src], "sqrt", module, builder)?;
            let one = src.into_float_value().get_type().const_float(1.0);
            Ok(builder
                .build_float_div(one, sqrt_val.into_float_value(), "rsqrt")
                .map_err(|e| crate::Error::LlvmError { reason: format!("build_float_div: {}", e) })?
                .into())
        }
        UnaryOp::Exp => {
            let suffix = get_type_suffix(result_dtype)?;
            call_intrinsic(&format!("llvm.exp.{}", suffix), &[src], "exp", module, builder)
        }
        UnaryOp::Exp2 => {
            let suffix = get_type_suffix(result_dtype)?;
            call_intrinsic(&format!("llvm.exp2.{}", suffix), &[src], "exp2", module, builder)
        }
        UnaryOp::Log => {
            let suffix = get_type_suffix(result_dtype)?;
            call_intrinsic(&format!("llvm.log.{}", suffix), &[src], "log", module, builder)
        }
        UnaryOp::Log2 => {
            let suffix = get_type_suffix(result_dtype)?;
            call_intrinsic(&format!("llvm.log2.{}", suffix), &[src], "log2", module, builder)
        }
        UnaryOp::Sin => {
            let suffix = get_type_suffix(result_dtype)?;
            call_intrinsic(&format!("llvm.sin.{}", suffix), &[src], "sin", module, builder)
        }
        UnaryOp::Cos => {
            let suffix = get_type_suffix(result_dtype)?;
            call_intrinsic(&format!("llvm.cos.{}", suffix), &[src], "cos", module, builder)
        }
        _ => {
            ensure!(false, UnsupportedOpSnafu { op: format!("{:?}", op) });
            unreachable!()
        }
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

    match op {
        BinaryOp::Add => {
            if is_float {
                Ok(builder
                    .build_float_add(lhs.into_float_value(), rhs.into_float_value(), "add")
                    .map_err(|e| crate::Error::LlvmError { reason: format!("build_float_add: {}", e) })?
                    .into())
            } else {
                Ok(builder
                    .build_int_add(lhs.into_int_value(), rhs.into_int_value(), "add")
                    .map_err(|e| crate::Error::LlvmError { reason: format!("build_int_add: {}", e) })?
                    .into())
            }
        }
        BinaryOp::Mul => {
            if is_float {
                Ok(builder
                    .build_float_mul(lhs.into_float_value(), rhs.into_float_value(), "mul")
                    .map_err(|e| crate::Error::LlvmError { reason: format!("build_float_mul: {}", e) })?
                    .into())
            } else {
                Ok(builder
                    .build_int_mul(lhs.into_int_value(), rhs.into_int_value(), "mul")
                    .map_err(|e| crate::Error::LlvmError { reason: format!("build_int_mul: {}", e) })?
                    .into())
            }
        }
        BinaryOp::Sub => {
            if is_float {
                Ok(builder
                    .build_float_sub(lhs.into_float_value(), rhs.into_float_value(), "sub")
                    .map_err(|e| crate::Error::LlvmError { reason: format!("build_float_sub: {}", e) })?
                    .into())
            } else {
                Ok(builder
                    .build_int_sub(lhs.into_int_value(), rhs.into_int_value(), "sub")
                    .map_err(|e| crate::Error::LlvmError { reason: format!("build_int_sub: {}", e) })?
                    .into())
            }
        }
        BinaryOp::Fdiv => Ok(builder
            .build_float_div(lhs.into_float_value(), rhs.into_float_value(), "fdiv")
            .map_err(|e| crate::Error::LlvmError { reason: format!("build_float_div: {}", e) })?
            .into()),
        BinaryOp::Idiv => {
            if is_signed {
                Ok(builder
                    .build_int_signed_div(lhs.into_int_value(), rhs.into_int_value(), "idiv")
                    .map_err(|e| crate::Error::LlvmError { reason: format!("build_int_signed_div: {}", e) })?
                    .into())
            } else {
                Ok(builder
                    .build_int_unsigned_div(lhs.into_int_value(), rhs.into_int_value(), "idiv")
                    .map_err(|e| crate::Error::LlvmError { reason: format!("build_int_unsigned_div: {}", e) })?
                    .into())
            }
        }
        BinaryOp::Mod => {
            if is_float {
                Ok(builder
                    .build_float_rem(lhs.into_float_value(), rhs.into_float_value(), "mod")
                    .map_err(|e| crate::Error::LlvmError { reason: format!("build_float_rem: {}", e) })?
                    .into())
            } else if is_signed {
                Ok(builder
                    .build_int_signed_rem(lhs.into_int_value(), rhs.into_int_value(), "mod")
                    .map_err(|e| crate::Error::LlvmError { reason: format!("build_int_signed_rem: {}", e) })?
                    .into())
            } else {
                Ok(builder
                    .build_int_unsigned_rem(lhs.into_int_value(), rhs.into_int_value(), "mod")
                    .map_err(|e| crate::Error::LlvmError { reason: format!("build_int_unsigned_rem: {}", e) })?
                    .into())
            }
        }
        BinaryOp::Max => {
            if is_float {
                let suffix = get_type_suffix(result_dtype)?;
                call_intrinsic(&format!("llvm.maxnum.{}", suffix), &[lhs, rhs], "max", module, builder)
            } else if is_signed {
                call_intrinsic(&format!("llvm.smax.i{}", result_dtype.bytes() * 8), &[lhs, rhs], "max", module, builder)
            } else {
                call_intrinsic(&format!("llvm.umax.i{}", result_dtype.bytes() * 8), &[lhs, rhs], "max", module, builder)
            }
        }
        BinaryOp::Pow => {
            let suffix = get_type_suffix(result_dtype)?;
            call_intrinsic(&format!("llvm.pow.{}", suffix), &[lhs, rhs], "pow", module, builder)
        }
        BinaryOp::And => Ok(builder
            .build_and(lhs.into_int_value(), rhs.into_int_value(), "and")
            .map_err(|e| crate::Error::LlvmError { reason: format!("build_and: {}", e) })?
            .into()),
        BinaryOp::Or => Ok(builder
            .build_or(lhs.into_int_value(), rhs.into_int_value(), "or")
            .map_err(|e| crate::Error::LlvmError { reason: format!("build_or: {}", e) })?
            .into()),
        BinaryOp::Xor => Ok(builder
            .build_xor(lhs.into_int_value(), rhs.into_int_value(), "xor")
            .map_err(|e| crate::Error::LlvmError { reason: format!("build_xor: {}", e) })?
            .into()),
        BinaryOp::Shl => Ok(builder
            .build_left_shift(lhs.into_int_value(), rhs.into_int_value(), "shl")
            .map_err(|e| crate::Error::LlvmError { reason: format!("build_left_shift: {}", e) })?
            .into()),
        BinaryOp::Shr => Ok(builder
            .build_right_shift(lhs.into_int_value(), rhs.into_int_value(), is_signed, "shr")
            .map_err(|e| crate::Error::LlvmError { reason: format!("build_right_shift: {}", e) })?
            .into()),
        BinaryOp::Lt => {
            if is_float {
                Ok(builder
                    .build_float_compare(FloatPredicate::OLT, lhs.into_float_value(), rhs.into_float_value(), "lt")
                    .map_err(|e| crate::Error::LlvmError { reason: format!("build_float_compare: {}", e) })?
                    .into())
            } else {
                let pred = if is_signed { IntPredicate::SLT } else { IntPredicate::ULT };
                Ok(builder
                    .build_int_compare(pred, lhs.into_int_value(), rhs.into_int_value(), "lt")
                    .map_err(|e| crate::Error::LlvmError { reason: format!("build_int_compare: {}", e) })?
                    .into())
            }
        }
        BinaryOp::Eq => {
            if is_float {
                Ok(builder
                    .build_float_compare(FloatPredicate::OEQ, lhs.into_float_value(), rhs.into_float_value(), "eq")
                    .map_err(|e| crate::Error::LlvmError { reason: format!("build_float_compare: {}", e) })?
                    .into())
            } else {
                Ok(builder
                    .build_int_compare(IntPredicate::EQ, lhs.into_int_value(), rhs.into_int_value(), "eq")
                    .map_err(|e| crate::Error::LlvmError { reason: format!("build_int_compare: {}", e) })?
                    .into())
            }
        }
        _ => {
            ensure!(false, UnsupportedOpSnafu { op: format!("{:?}", op) });
            unreachable!()
        }
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
        TernaryOp::Where => Ok(builder
            .build_select(a.into_int_value(), b, c, "where")
            .map_err(|e| crate::Error::LlvmError { reason: format!("build_select: {}", e) })?),
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
    let dst_type = dtype_to_basic_type(dst_dtype, context);

    let src_is_float = src_dtype.is_float();
    let dst_is_float = dst_dtype.is_float();
    let src_is_signed = src_dtype.is_signed();
    let dst_is_signed = dst_dtype.is_signed();

    match (src_is_float, dst_is_float) {
        (true, true) => {
            let src_val = src.into_float_value();
            let dst_type = dst_type.into_float_type();
            if src_dtype.bytes() > dst_dtype.bytes() {
                Ok(builder
                    .build_float_trunc(src_val, dst_type, "fptrunc")
                    .map_err(|e| crate::Error::LlvmError { reason: format!("build_float_trunc: {}", e) })?
                    .into())
            } else {
                Ok(builder
                    .build_float_ext(src_val, dst_type, "fpext")
                    .map_err(|e| crate::Error::LlvmError { reason: format!("build_float_ext: {}", e) })?
                    .into())
            }
        }
        (true, false) => {
            let src_val = src.into_float_value();
            let dst_type = dst_type.into_int_type();
            if dst_is_signed {
                Ok(builder
                    .build_float_to_signed_int(src_val, dst_type, "fptosi")
                    .map_err(|e| crate::Error::LlvmError { reason: format!("build_float_to_signed_int: {}", e) })?
                    .into())
            } else {
                Ok(builder
                    .build_float_to_unsigned_int(src_val, dst_type, "fptoui")
                    .map_err(|e| crate::Error::LlvmError { reason: format!("build_float_to_unsigned_int: {}", e) })?
                    .into())
            }
        }
        (false, true) => {
            let src_val = src.into_int_value();
            let dst_type = dst_type.into_float_type();
            if src_is_signed {
                Ok(builder
                    .build_signed_int_to_float(src_val, dst_type, "sitofp")
                    .map_err(|e| crate::Error::LlvmError { reason: format!("build_signed_int_to_float: {}", e) })?
                    .into())
            } else {
                Ok(builder
                    .build_unsigned_int_to_float(src_val, dst_type, "uitofp")
                    .map_err(|e| crate::Error::LlvmError { reason: format!("build_unsigned_int_to_float: {}", e) })?
                    .into())
            }
        }
        (false, false) => {
            let src_val = src.into_int_value();
            let dst_type = dst_type.into_int_type();
            if src_dtype.bytes() > dst_dtype.bytes() {
                Ok(builder
                    .build_int_truncate(src_val, dst_type, "trunc")
                    .map_err(|e| crate::Error::LlvmError { reason: format!("build_int_truncate: {}", e) })?
                    .into())
            } else if src_is_signed {
                Ok(builder
                    .build_int_s_extend(src_val, dst_type, "sext")
                    .map_err(|e| crate::Error::LlvmError { reason: format!("build_int_s_extend: {}", e) })?
                    .into())
            } else {
                Ok(builder
                    .build_int_z_extend(src_val, dst_type, "zext")
                    .map_err(|e| crate::Error::LlvmError { reason: format!("build_int_z_extend: {}", e) })?
                    .into())
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
        unsafe {
            builder
                .build_gep(dtype_to_basic_type(result_dtype, context), ptr_val, &[index_val], "gep")
                .map_err(|e| crate::Error::LlvmError { reason: format!("build_gep: {}", e) })?
        }
    };

    // Load from the pointer
    let loaded = builder
        .build_load(dtype_to_basic_type(result_dtype, context), element_ptr, "load")
        .map_err(|e| crate::Error::LlvmError { reason: format!("build_load: {}", e) })?;

    Ok(loaded)
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
        unsafe {
            builder
                .build_gep(
                    // Use i8 type for opaque pointer arithmetic
                    builder.get_insert_block().unwrap().get_context().i8_type(),
                    ptr_val,
                    &[index_val],
                    "gep",
                )
                .map_err(|e| crate::Error::LlvmError { reason: format!("build_gep: {}", e) })?
        }
    };

    // Store the value
    builder
        .build_store(element_ptr, value)
        .map_err(|e| crate::Error::LlvmError { reason: format!("build_store: {}", e) })?;

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
            DType::Ptr { base, .. } => dtype_to_basic_type(base, context),
            _ => return Ok(value), // Not a Ptr dtype, shouldn't happen
        };

        // Load from the pointer
        let loaded = builder
            .build_load(element_type, value.into_pointer_value(), "autoload")
            .map_err(|e| crate::Error::LlvmError { reason: format!("auto_load build_load: {}", e) })?;

        Ok(loaded)
    } else {
        Ok(value)
    }
}
