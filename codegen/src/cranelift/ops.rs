//! Operation-specific Cranelift IR code generation.

use std::collections::HashMap;
use std::sync::Arc;

use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::immediates::Offset32;
use cranelift_codegen::ir::instructions::BlockArg;
use cranelift_codegen::ir::{InstBuilder, MemFlags, Value, types as cl_types};
use cranelift_frontend::FunctionBuilder;
use snafu::OptionExt;

use morok_dtype::DType;
use morok_ir::{AxisType, BinaryOp, ConstValue, Op, ReduceOp, TernaryOp, UOp, UnaryOp};

use super::error::{
    InvalidReduceRangeSnafu, NotFoundSnafu, RequiresDecompositionSnafu, UnknownOpSnafu, UnsupportedSnafu,
};
use super::helpers::LoopContext;
use super::types::{dtype_to_cranelift_type, is_float, is_signed};

use crate::Result;

/// Auto-load a value if the source UOp is a pointer.
///
/// This handles the case where INDEX produces a pointer but we need a scalar value
/// for arithmetic operations. If the source dtype is a Ptr, we load from the pointer.
fn auto_load_pointer(value: Value, src_dtype: &DType, builder: &mut FunctionBuilder) -> Result<Value> {
    if let DType::Ptr { base, .. } = src_dtype {
        // Get the element type from the Ptr dtype
        let element_type = dtype_to_cranelift_type(base);

        // Load from the pointer
        let flags = MemFlags::new();
        let loaded = builder.ins().load(element_type, flags, value, Offset32::new(0));
        Ok(loaded)
    } else {
        Ok(value)
    }
}

/// Generate Cranelift IR for a UOp node.
pub(crate) fn codegen_uop(
    uop: &Arc<UOp>,
    builder: &mut FunctionBuilder,
    values: &mut HashMap<u64, Value>,
    loop_contexts: &mut HashMap<u64, LoopContext>,
) -> Result<Option<Value>> {
    // Check if already generated
    if let Some(&val) = values.get(&uop.id) {
        return Ok(Some(val));
    }

    let result = match uop.op() {
        Op::Const(val_hash) => Some(codegen_const(val_hash.0, &uop.dtype(), builder)?),

        Op::Unary(op, src) => {
            let src_val = get_value(src, builder, values, loop_contexts)?;
            // Auto-load if source is a pointer (from INDEX)
            let src_val = auto_load_pointer(src_val, &src.dtype(), builder)?;
            Some(codegen_unary(*op, src_val, &uop.dtype(), builder)?)
        }

        Op::Binary(op, lhs, rhs) => {
            let lhs_val = get_value(lhs, builder, values, loop_contexts)?;
            let rhs_val = get_value(rhs, builder, values, loop_contexts)?;
            // Auto-load if operands are pointers (from INDEX)
            let lhs_val = auto_load_pointer(lhs_val, &lhs.dtype(), builder)?;
            let rhs_val = auto_load_pointer(rhs_val, &rhs.dtype(), builder)?;
            // For comparisons, use the operand dtype, not result dtype (which is Bool)
            // After auto-load, get the actual element type
            let operand_dtype = match lhs.dtype() {
                DType::Ptr { base, .. } => base.as_ref().clone(),
                other => other,
            };
            Some(codegen_binary(*op, lhs_val, rhs_val, &uop.dtype(), &operand_dtype, builder)?)
        }

        Op::Ternary(op, a, b, c) => {
            let a_val = get_value(a, builder, values, loop_contexts)?;
            let b_val = get_value(b, builder, values, loop_contexts)?;
            let c_val = get_value(c, builder, values, loop_contexts)?;
            // Auto-load if operands are pointers (from INDEX)
            let a_val = auto_load_pointer(a_val, &a.dtype(), builder)?;
            let b_val = auto_load_pointer(b_val, &b.dtype(), builder)?;
            let c_val = auto_load_pointer(c_val, &c.dtype(), builder)?;
            Some(codegen_ternary(*op, a_val, b_val, c_val, &uop.dtype(), builder)?)
        }

        Op::Cast { src, dtype } => {
            let src_val = get_value(src, builder, values, loop_contexts)?;
            // Auto-load if source is a pointer (from INDEX)
            let src_val = auto_load_pointer(src_val, &src.dtype(), builder)?;
            // Get the actual source dtype after auto-load
            let actual_src_dtype = match src.dtype() {
                DType::Ptr { base, .. } => base.as_ref().clone(),
                other => other,
            };
            Some(codegen_cast(src_val, &actual_src_dtype, dtype, builder)?)
        }

        Op::BitCast { src, dtype } => {
            let src_val = get_value(src, builder, values, loop_contexts)?;
            Some(codegen_bitcast(src_val, &src.dtype(), dtype, builder)?)
        }

        Op::Load { buffer, index } => {
            let buffer_val = get_value(buffer, builder, values, loop_contexts)?;
            let index_val = get_value(index, builder, values, loop_contexts)?;
            Some(codegen_load(buffer_val, index_val, &uop.dtype(), builder)?)
        }

        Op::Store { buffer, index, value } => {
            let buffer_val = get_value(buffer, builder, values, loop_contexts)?;
            let index_val = get_value(index, builder, values, loop_contexts)?;
            let value_val = get_value(value, builder, values, loop_contexts)?;
            // Auto-load if value is a pointer (from INDEX)
            let value_val = auto_load_pointer(value_val, &value.dtype(), builder)?;
            // Get the actual value dtype after auto-load
            let actual_value_dtype = match value.dtype() {
                DType::Ptr { base, .. } => base.as_ref().clone(),
                other => other,
            };

            // Check if index is already an indexed pointer (from INDEX op)
            // If so, use it directly as the store address without recomputing the offset
            // This is the common pattern: STORE { buffer: DefineGlobal, index: INDEX(...) }
            if let Op::Index { .. } = index.op() {
                // Index is already a computed pointer - store directly to it
                let flags = MemFlags::new();
                builder.ins().store(flags, value_val, index_val, Offset32::new(0));
            } else if buffer.id == index.id {
                // Special case: buffer and index are the same UOp
                // This means "store at index 0" - store directly to buffer base
                let flags = MemFlags::new();
                builder.ins().store(flags, value_val, buffer_val, Offset32::new(0));
            } else {
                // Index is an integer index - compute offset and store
                codegen_store(buffer_val, index_val, value_val, &actual_value_dtype, builder)?;
            }
            None
        }

        Op::Buffer { .. } | Op::DefineGlobal(_) | Op::DefineLocal(_) | Op::DefineVar { .. } => {
            // These should already be in values map as function parameters
            return Ok(values.get(&uop.id).copied());
        }

        Op::Bind { var, .. } => {
            // BIND evaluates to its variable value
            values.get(&var.id).copied()
        }

        Op::Sink { sources } => {
            // Evaluate all sources for side effects
            for src in sources {
                codegen_uop(src, builder, values, loop_contexts)?;
            }
            None
        }

        Op::Index { buffer, indices, gate: None } => {
            let buffer_val = get_value(buffer, builder, values, loop_contexts)?;

            let linear_index = if indices.len() == 1 {
                // Single-index case: use directly
                get_value(&indices[0], builder, values, loop_contexts)?
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

                // Build linear index: sum(idx[i] * stride[i])
                let mut linear = builder.ins().iconst(cl_types::I64, 0);
                for (idx_uop, &stride) in indices.iter().zip(strides.iter()) {
                    let idx_val = get_value(idx_uop, builder, values, loop_contexts)?;
                    let stride_val = builder.ins().iconst(cl_types::I64, stride);
                    let term = builder.ins().imul(idx_val, stride_val);
                    linear = builder.ins().iadd(linear, term);
                }
                linear
            };

            // Compute byte offset
            let element_size = match uop.dtype() {
                DType::Ptr { base, .. } => base.bytes(),
                other => other.bytes(),
            };
            let size_val = builder.ins().iconst(cl_types::I64, element_size as i64);
            let byte_offset = builder.ins().imul(linear_index, size_val);
            let ptr = builder.ins().iadd(buffer_val, byte_offset);
            Some(ptr)
        }

        Op::Index { gate: Some(_), .. } => {
            return UnsupportedSnafu { what: "Gated INDEX" }.fail().map_err(Into::into);
        }

        Op::PointerIndex { ptr, offset } => {
            let ptr_val = get_value(ptr, builder, values, loop_contexts)?;
            let offset_val = get_value(offset, builder, values, loop_contexts)?;
            let result = builder.ins().iadd(ptr_val, offset_val);
            Some(result)
        }

        Op::Range { end, axis_type, .. } => {
            // Outer ranges don't generate loops - they represent kernel grid dimensions
            if *axis_type == AxisType::Outer {
                let end_val = get_value(end, builder, values, loop_contexts)?;
                return Ok(Some(end_val));
            }

            // Reduce ranges are handled by REDUCE op directly
            if *axis_type == AxisType::Reduce {
                let end_val = get_value(end, builder, values, loop_contexts)?;
                return Ok(Some(end_val));
            }

            // Unroll ranges should have been statically expanded by the optimizer.
            // At codegen time, we just return the end value (the range variable
            // will be substituted with concrete values by the unroller).
            if *axis_type == AxisType::Unroll || *axis_type == AxisType::Upcast {
                let end_val = get_value(end, builder, values, loop_contexts)?;
                return Ok(Some(end_val));
            }

            // Get end value before creating blocks
            let end_val = get_value(end, builder, values, loop_contexts)?;

            // Create loop blocks
            let header_block = builder.create_block();
            let body_block = builder.create_block();
            let exit_block = builder.create_block();

            // Add block parameter for loop variable
            builder.append_block_param(header_block, cl_types::I64);

            // Jump to header with initial value (0)
            let zero = builder.ins().iconst(cl_types::I64, 0);
            let zero_arg: BlockArg = zero.into();
            builder.ins().jump(header_block, &[zero_arg]);

            // Header block: check condition
            builder.switch_to_block(header_block);
            let loop_var_val = builder.block_params(header_block)[0];

            // Condition: loop_var < end
            let cond = builder.ins().icmp(IntCC::UnsignedLessThan, loop_var_val, end_val);
            let empty_args: &[BlockArg] = &[];
            builder.ins().brif(cond, body_block, empty_args, exit_block, empty_args);

            // Body block
            builder.switch_to_block(body_block);

            // Create variable for tracking
            let var = builder.declare_var(cl_types::I64);
            builder.def_var(var, loop_var_val);

            // Store loop context for END to complete
            loop_contexts.insert(uop.id, LoopContext { header_block, body_block, exit_block, loop_var: var });

            // Store loop variable value
            values.insert(uop.id, loop_var_val);

            // Return the loop variable value
            Some(loop_var_val)
        }

        Op::End { computation, ranges } => {
            // Generate computation (loop body)
            codegen_uop(computation, builder, values, loop_contexts)?;

            // Close each range's loop
            for range_uop in ranges.iter().rev() {
                if let Op::Range { axis_type: AxisType::Outer, .. } = range_uop.op() {
                    continue;
                }

                if let Some(loop_ctx) = loop_contexts.get(&range_uop.id) {
                    // Get current loop variable value
                    let current_val = builder.use_var(loop_ctx.loop_var);

                    // Increment
                    let one = builder.ins().iconst(cl_types::I64, 1);
                    let next_val = builder.ins().iadd(current_val, one);

                    // Jump back to header with incremented value
                    let next_arg: BlockArg = next_val.into();
                    builder.ins().jump(loop_ctx.header_block, &[next_arg]);

                    // Seal body block (only predecessor is header)
                    builder.seal_block(loop_ctx.body_block);

                    // Seal header block (predecessors: entry block + body block)
                    builder.seal_block(loop_ctx.header_block);

                    // Switch to exit block for code after loop
                    builder.switch_to_block(loop_ctx.exit_block);
                    builder.seal_block(loop_ctx.exit_block);
                }
            }

            None
        }

        Op::Barrier { src, .. } => {
            // For CPU, just evaluate src
            codegen_uop(src, builder, values, loop_contexts)?
        }

        Op::Reduce { src, ranges, reduce_op } => {
            // Simple reduce without ranges - just return source
            if ranges.is_empty() {
                return codegen_uop(src, builder, values, loop_contexts);
            }

            // Get result type
            let result_type = dtype_to_cranelift_type(&uop.dtype());

            // Get identity value for reduce op
            let identity = codegen_reduce_identity(*reduce_op, &uop.dtype(), builder)?;

            // Filter to only Loop/Reduce ranges
            let loop_ranges: Vec<_> =
                ranges
                    .iter()
                    .filter(|r| {
                        if let Op::Range { axis_type, .. } = r.op() { *axis_type != AxisType::Outer } else { false }
                    })
                    .collect();

            // Handle Outer ranges
            for range_uop in ranges.iter() {
                if let Op::Range { end, axis_type: AxisType::Outer, .. } = range_uop.op() {
                    let end_val = get_value(end, builder, values, loop_contexts)?;
                    values.insert(range_uop.id, end_val);
                }
            }

            if loop_ranges.is_empty() {
                return codegen_uop(src, builder, values, loop_contexts);
            }

            // Multi-range reduction: use flattened iteration approach
            // For ranges with ends [e0, e1, ...], total iterations = e0 * e1 * ...
            // Single counter i can be decomposed into individual range values using div/mod
            //
            // Example: ranges [2, 3] → total = 6, for counter i:
            //   r0 = i / 3, r1 = i % 3
            //
            // General formula (for ranges ordered [e0, e1, e2]):
            //   stride0 = e1 * e2, r0 = i / stride0
            //   stride1 = e2,      r1 = (i / stride1) % e1
            //   stride2 = 1,       r2 = i % e2

            // Collect end values for all loop ranges
            let mut end_vals: Vec<Value> = Vec::with_capacity(loop_ranges.len());
            for range_uop in &loop_ranges {
                let Op::Range { end, .. } = range_uop.op() else {
                    return InvalidReduceRangeSnafu { id: range_uop.id }.fail().map_err(Into::into);
                };
                let end_val = get_value(end, builder, values, loop_contexts)?;
                end_vals.push(end_val);
            }

            // Compute total iterations (product of all ends)
            let mut total_iter = builder.ins().iconst(cl_types::I64, 1);
            for &end_val in &end_vals {
                total_iter = builder.ins().imul(total_iter, end_val);
            }

            // Compute strides for decomposition (stride[i] = product of ends[i+1:])
            let mut strides: Vec<Value> = vec![builder.ins().iconst(cl_types::I64, 1); loop_ranges.len()];
            for i in (0..loop_ranges.len().saturating_sub(1)).rev() {
                strides[i] = builder.ins().imul(strides[i + 1], end_vals[i + 1]);
            }

            // Create blocks
            let header_block = builder.create_block();
            let body_block = builder.create_block();
            let exit_block = builder.create_block();

            // Add block params for counter and accumulator
            builder.append_block_param(header_block, cl_types::I64); // counter
            builder.append_block_param(header_block, result_type); // accumulator

            // Jump to header with initial values
            let zero = builder.ins().iconst(cl_types::I64, 0);
            let zero_arg: BlockArg = zero.into();
            let identity_arg: BlockArg = identity.into();
            builder.ins().jump(header_block, &[zero_arg, identity_arg]);

            // Header: condition check (use total_iter, not single end_val!)
            builder.switch_to_block(header_block);
            let counter = builder.block_params(header_block)[0];
            let acc = builder.block_params(header_block)[1];

            let cond = builder.ins().icmp(IntCC::UnsignedLessThan, counter, total_iter);
            let empty_args: &[BlockArg] = &[];
            builder.ins().brif(cond, body_block, empty_args, exit_block, empty_args);

            // Body: decompose counter into individual range values and accumulate
            builder.switch_to_block(body_block);

            // Decompose counter into individual range values using div/mod
            // For each range i: r[i] = (counter / stride[i]) % end[i]
            // Store each value with the corresponding range_uop.id
            for (i, range_uop) in loop_ranges.iter().enumerate() {
                let range_val = if i == loop_ranges.len() - 1 {
                    // Last range: just use counter % end[i]
                    builder.ins().urem(counter, end_vals[i])
                } else {
                    // Other ranges: (counter / stride[i]) % end[i]
                    let divided = builder.ins().udiv(counter, strides[i]);
                    builder.ins().urem(divided, end_vals[i])
                };
                values.insert(range_uop.id, range_val);
            }

            // IMPORTANT: Clear cached values for nodes that depend on ANY range
            // so they regenerate inside the loop body using the decomposed values.
            // We need to regenerate INDEX and any computations that use the ranges.
            //
            // Collect all range IDs
            let range_ids: Vec<u64> = loop_ranges.iter().map(|r| r.id).collect();

            // Collect all node IDs that transitively depend on any range
            let mut to_clear: Vec<u64> = Vec::new();
            for node in src.toposort() {
                // Check if this node uses any range directly or indirectly
                let uses_range = node.op().sources().iter().any(|s| range_ids.contains(&s.id));
                if uses_range || to_clear.iter().any(|&id| node.op().sources().iter().any(|s| s.id == id)) {
                    to_clear.push(node.id);
                }
            }
            for id in &to_clear {
                values.remove(id);
            }

            // Generate source - now it will regenerate using the decomposed range values
            let src_val = codegen_uop(src, builder, values, loop_contexts)?
                .context(NotFoundSnafu { what: "source value for reduce", id: src.id })?;

            // Auto-load if source is a pointer (from INDEX)
            let src_val = auto_load_pointer(src_val, &src.dtype(), builder)?;

            // Apply reduce op
            let new_acc = codegen_reduce_op(*reduce_op, acc, src_val, &uop.dtype(), builder)?;

            // Increment counter
            let one = builder.ins().iconst(cl_types::I64, 1);
            let next_counter = builder.ins().iadd(counter, one);

            // Jump back to header
            let next_counter_arg: BlockArg = next_counter.into();
            let new_acc_arg: BlockArg = new_acc.into();
            builder.ins().jump(header_block, &[next_counter_arg, new_acc_arg]);

            // Seal blocks
            builder.seal_block(body_block);
            builder.seal_block(header_block);

            // Exit block
            builder.switch_to_block(exit_block);
            builder.seal_block(exit_block);

            // Return final accumulator (from header block params at exit)
            Some(acc)
        }

        Op::Noop => None,

        Op::Unique(_) | Op::Device(_) => None,

        Op::Vectorize { elements } => {
            if elements.len() == 1 {
                codegen_uop(&elements[0], builder, values, loop_contexts)?
            } else {
                let mut last = None;
                for elem in elements {
                    last = codegen_uop(elem, builder, values, loop_contexts)?;
                }
                last
            }
        }

        Op::Reshape { src, .. } => {
            // RESHAPE is a view operation - for scalar values it's a no-op.
            // The shape change is metadata only, the value is unchanged.
            // This handles cases like axis-wise reductions where RESHAPE
            // wraps the REDUCE result to produce the correct output shape.
            codegen_uop(src, builder, values, loop_contexts)?
        }

        _ => {
            return UnknownOpSnafu { op: format!("{:?}", uop.op()) }.fail().map_err(Into::into);
        }
    };

    // Store result
    if let Some(val) = result {
        values.insert(uop.id, val);
    }

    Ok(result)
}

/// Get value for a UOp, generating code if needed.
fn get_value(
    uop: &Arc<UOp>,
    builder: &mut FunctionBuilder,
    values: &mut HashMap<u64, Value>,
    loop_contexts: &mut HashMap<u64, LoopContext>,
) -> Result<Value> {
    if let Some(&val) = values.get(&uop.id) {
        return Ok(val);
    }

    codegen_uop(uop, builder, values, loop_contexts)?
        .context(NotFoundSnafu { what: "value for UOp", id: uop.id })
        .map_err(Into::into)
}

/// Generate constant value.
fn codegen_const(val: ConstValue, dtype: &DType, builder: &mut FunctionBuilder) -> Result<Value> {
    let cl_type = dtype_to_cranelift_type(dtype);

    let value = match val {
        ConstValue::Int(i) => builder.ins().iconst(cl_type, i),
        ConstValue::UInt(u) => builder.ins().iconst(cl_type, u as i64),
        ConstValue::Float(f) => {
            if cl_type == cl_types::F32 {
                builder.ins().f32const(f as f32)
            } else {
                builder.ins().f64const(f)
            }
        }
        ConstValue::Bool(b) => builder.ins().iconst(cl_types::I8, b as i64),
    };

    Ok(value)
}

/// Generate unary operation.
fn codegen_unary(op: UnaryOp, src: Value, result_dtype: &DType, builder: &mut FunctionBuilder) -> Result<Value> {
    let is_fp = is_float(result_dtype);

    let result = match op {
        UnaryOp::Neg => {
            if is_fp {
                builder.ins().fneg(src)
            } else {
                builder.ins().ineg(src)
            }
        }
        UnaryOp::Abs => {
            if is_fp {
                builder.ins().fabs(src)
            } else {
                // Integer abs: x < 0 ? -x : x
                let zero = builder.ins().iconst(dtype_to_cranelift_type(result_dtype), 0);
                let is_neg = builder.ins().icmp(IntCC::SignedLessThan, src, zero);
                let neg = builder.ins().ineg(src);
                builder.ins().select(is_neg, neg, src)
            }
        }
        UnaryOp::Sqrt => builder.ins().sqrt(src),
        UnaryOp::Ceil => builder.ins().ceil(src),
        UnaryOp::Floor => builder.ins().floor(src),
        UnaryOp::Trunc => {
            if is_fp {
                builder.ins().trunc(src)
            } else {
                src
            }
        }
        UnaryOp::Round => builder.ins().nearest(src),
        UnaryOp::Rsqrt => {
            let sqrt_val = builder.ins().sqrt(src);
            let one = if dtype_to_cranelift_type(result_dtype) == cl_types::F32 {
                builder.ins().f32const(1.0)
            } else {
                builder.ins().f64const(1.0)
            };
            builder.ins().fdiv(one, sqrt_val)
        }
        UnaryOp::Reciprocal => {
            let one = if dtype_to_cranelift_type(result_dtype) == cl_types::F32 {
                builder.ins().f32const(1.0)
            } else {
                builder.ins().f64const(1.0)
            };
            builder.ins().fdiv(one, src)
        }
        UnaryOp::Square => {
            if is_fp {
                builder.ins().fmul(src, src)
            } else {
                builder.ins().imul(src, src)
            }
        }
        // Transcendentals - these should be decomposed at UOp level before reaching codegen
        UnaryOp::Exp => {
            return RequiresDecompositionSnafu { op: "Exp" }.fail().map_err(Into::into);
        }
        UnaryOp::Exp2 => {
            return RequiresDecompositionSnafu { op: "Exp2" }.fail().map_err(Into::into);
        }
        UnaryOp::Log => {
            return RequiresDecompositionSnafu { op: "Log" }.fail().map_err(Into::into);
        }
        UnaryOp::Log2 => {
            return RequiresDecompositionSnafu { op: "Log2" }.fail().map_err(Into::into);
        }
        UnaryOp::Sin => {
            return RequiresDecompositionSnafu { op: "Sin" }.fail().map_err(Into::into);
        }
        UnaryOp::Cos => {
            return RequiresDecompositionSnafu { op: "Cos" }.fail().map_err(Into::into);
        }
        UnaryOp::Tan => {
            return RequiresDecompositionSnafu { op: "Tan" }.fail().map_err(Into::into);
        }
        UnaryOp::Erf => {
            return RequiresDecompositionSnafu { op: "Erf" }.fail().map_err(Into::into);
        }
        _ => {
            return UnknownOpSnafu { op: format!("{:?}", op) }.fail().map_err(Into::into);
        }
    };

    Ok(result)
}

/// Generate binary operation.
fn codegen_binary(
    op: BinaryOp,
    lhs: Value,
    rhs: Value,
    _result_dtype: &DType,
    operand_dtype: &DType,
    builder: &mut FunctionBuilder,
) -> Result<Value> {
    // Use operand_dtype for determining int/float behavior (important for comparisons)
    let is_fp = is_float(operand_dtype);
    let is_sgn = is_signed(operand_dtype);

    let result = match op {
        BinaryOp::Add => {
            if is_fp {
                builder.ins().fadd(lhs, rhs)
            } else {
                builder.ins().iadd(lhs, rhs)
            }
        }
        BinaryOp::Sub => {
            if is_fp {
                builder.ins().fsub(lhs, rhs)
            } else {
                builder.ins().isub(lhs, rhs)
            }
        }
        BinaryOp::Mul => {
            if is_fp {
                builder.ins().fmul(lhs, rhs)
            } else {
                builder.ins().imul(lhs, rhs)
            }
        }
        BinaryOp::Fdiv => builder.ins().fdiv(lhs, rhs),
        BinaryOp::Idiv => {
            if is_sgn {
                builder.ins().sdiv(lhs, rhs)
            } else {
                builder.ins().udiv(lhs, rhs)
            }
        }
        BinaryOp::Mod => {
            if is_fp {
                // Cranelift doesn't have frem, compute as: a - floor(a/b) * b
                let div = builder.ins().fdiv(lhs, rhs);
                let floored = builder.ins().floor(div);
                let mul = builder.ins().fmul(floored, rhs);
                builder.ins().fsub(lhs, mul)
            } else if is_sgn {
                builder.ins().srem(lhs, rhs)
            } else {
                builder.ins().urem(lhs, rhs)
            }
        }
        BinaryOp::Max => {
            if is_fp {
                builder.ins().fmax(lhs, rhs)
            } else if is_sgn {
                builder.ins().smax(lhs, rhs)
            } else {
                builder.ins().umax(lhs, rhs)
            }
        }
        BinaryOp::And => builder.ins().band(lhs, rhs),
        BinaryOp::Or => builder.ins().bor(lhs, rhs),
        BinaryOp::Xor => builder.ins().bxor(lhs, rhs),
        BinaryOp::Shl => builder.ins().ishl(lhs, rhs),
        BinaryOp::Shr => {
            if is_sgn {
                builder.ins().sshr(lhs, rhs)
            } else {
                builder.ins().ushr(lhs, rhs)
            }
        }
        BinaryOp::Lt => {
            if is_fp {
                builder.ins().fcmp(FloatCC::LessThan, lhs, rhs)
            } else if is_sgn {
                builder.ins().icmp(IntCC::SignedLessThan, lhs, rhs)
            } else {
                builder.ins().icmp(IntCC::UnsignedLessThan, lhs, rhs)
            }
        }
        BinaryOp::Le => {
            if is_fp {
                builder.ins().fcmp(FloatCC::LessThanOrEqual, lhs, rhs)
            } else if is_sgn {
                builder.ins().icmp(IntCC::SignedLessThanOrEqual, lhs, rhs)
            } else {
                builder.ins().icmp(IntCC::UnsignedLessThanOrEqual, lhs, rhs)
            }
        }
        BinaryOp::Gt => {
            if is_fp {
                builder.ins().fcmp(FloatCC::GreaterThan, lhs, rhs)
            } else if is_sgn {
                builder.ins().icmp(IntCC::SignedGreaterThan, lhs, rhs)
            } else {
                builder.ins().icmp(IntCC::UnsignedGreaterThan, lhs, rhs)
            }
        }
        BinaryOp::Ge => {
            if is_fp {
                builder.ins().fcmp(FloatCC::GreaterThanOrEqual, lhs, rhs)
            } else if is_sgn {
                builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, lhs, rhs)
            } else {
                builder.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, lhs, rhs)
            }
        }
        BinaryOp::Eq => {
            if is_fp {
                builder.ins().fcmp(FloatCC::Equal, lhs, rhs)
            } else {
                builder.ins().icmp(IntCC::Equal, lhs, rhs)
            }
        }
        BinaryOp::Ne => {
            if is_fp {
                builder.ins().fcmp(FloatCC::NotEqual, lhs, rhs)
            } else {
                builder.ins().icmp(IntCC::NotEqual, lhs, rhs)
            }
        }
        BinaryOp::Pow => {
            // Pow should be decomposed at UOp level: pow(x,y) = exp2(y * log2(x))
            return RequiresDecompositionSnafu { op: "Pow" }.fail().map_err(Into::into);
        }
        _ => {
            return UnknownOpSnafu { op: format!("{:?}", op) }.fail().map_err(Into::into);
        }
    };

    Ok(result)
}

/// Generate ternary operation.
fn codegen_ternary(
    op: TernaryOp,
    a: Value,
    b: Value,
    c: Value,
    _result_dtype: &DType,
    builder: &mut FunctionBuilder,
) -> Result<Value> {
    let result = match op {
        TernaryOp::Where => builder.ins().select(a, b, c),
        TernaryOp::MulAcc => {
            // fma(a, b, c) = a * b + c
            builder.ins().fma(a, b, c)
        }
    };

    Ok(result)
}

/// Generate bitcast operation (reinterpret bits as different type).
fn codegen_bitcast(src: Value, src_dtype: &DType, dst_dtype: &DType, builder: &mut FunctionBuilder) -> Result<Value> {
    let dst_type = dtype_to_cranelift_type(dst_dtype);

    // Check if sizes match
    if src_dtype.bytes() != dst_dtype.bytes() {
        return UnknownOpSnafu {
            op: format!(
                "BitCast size mismatch: {:?} ({} bytes) to {:?} ({} bytes)",
                src_dtype,
                src_dtype.bytes(),
                dst_dtype,
                dst_dtype.bytes()
            ),
        }
        .fail()
        .map_err(Into::into);
    }

    // Cranelift bitcast preserves bits
    let result = builder.ins().bitcast(dst_type, MemFlags::new(), src);
    Ok(result)
}

/// Generate cast operation.
fn codegen_cast(src: Value, src_dtype: &DType, dst_dtype: &DType, builder: &mut FunctionBuilder) -> Result<Value> {
    // Handle no-op cast (same type)
    if src_dtype == dst_dtype {
        return Ok(src);
    }

    let dst_type = dtype_to_cranelift_type(dst_dtype);
    let src_is_float = is_float(src_dtype);
    let dst_is_float = is_float(dst_dtype);
    let src_is_signed = is_signed(src_dtype);

    let result = match (src_is_float, dst_is_float) {
        (true, true) => {
            // Float to float
            if src_dtype.bytes() > dst_dtype.bytes() {
                builder.ins().fdemote(dst_type, src)
            } else if src_dtype.bytes() < dst_dtype.bytes() {
                builder.ins().fpromote(dst_type, src)
            } else {
                // Same size float (e.g., float32 to float32) - no-op
                src
            }
        }
        (true, false) => {
            // Float to int
            if is_signed(dst_dtype) {
                builder.ins().fcvt_to_sint(dst_type, src)
            } else {
                builder.ins().fcvt_to_uint(dst_type, src)
            }
        }
        (false, true) => {
            // Int to float
            // Cranelift fcvt_from_sint/uint requires source and dest to have same bit width.
            // If source is wider than dest, first reduce to matching int size.
            let src_bytes = src_dtype.bytes();
            let dst_bytes = dst_dtype.bytes();

            let adjusted_src = if src_bytes > dst_bytes {
                // Reduce int to match float width (e.g., i64 → i32 for f32)
                let reduced_type = match dst_bytes {
                    4 => cranelift_codegen::ir::types::I32,
                    8 => cranelift_codegen::ir::types::I64,
                    _ => dst_type, // Fallback
                };
                builder.ins().ireduce(reduced_type, src)
            } else if src_bytes < dst_bytes {
                // Extend int to match float width (e.g., i32 → i64 for f64)
                let extended_type = match dst_bytes {
                    4 => cranelift_codegen::ir::types::I32,
                    8 => cranelift_codegen::ir::types::I64,
                    _ => dst_type, // Fallback
                };
                if src_is_signed {
                    builder.ins().sextend(extended_type, src)
                } else {
                    builder.ins().uextend(extended_type, src)
                }
            } else {
                src
            };

            if src_is_signed {
                builder.ins().fcvt_from_sint(dst_type, adjusted_src)
            } else {
                builder.ins().fcvt_from_uint(dst_type, adjusted_src)
            }
        }
        (false, false) => {
            // Int to int
            if src_dtype.bytes() > dst_dtype.bytes() {
                builder.ins().ireduce(dst_type, src)
            } else if src_dtype.bytes() < dst_dtype.bytes() {
                if src_is_signed { builder.ins().sextend(dst_type, src) } else { builder.ins().uextend(dst_type, src) }
            } else {
                // Same size int (e.g., i32 to u32 or u32 to i32) - no-op at bit level
                src
            }
        }
    };

    Ok(result)
}

/// Generate load operation.
fn codegen_load(buffer_ptr: Value, index: Value, result_dtype: &DType, builder: &mut FunctionBuilder) -> Result<Value> {
    let element_type = dtype_to_cranelift_type(result_dtype);
    let element_size = result_dtype.bytes() as i64;

    // Compute byte offset
    let size_val = builder.ins().iconst(cl_types::I64, element_size);
    let byte_offset = builder.ins().imul(index, size_val);

    // Add to base pointer
    let addr = builder.ins().iadd(buffer_ptr, byte_offset);

    // Load
    let flags = MemFlags::new();
    let result = builder.ins().load(element_type, flags, addr, Offset32::new(0));

    Ok(result)
}

/// Generate store operation.
fn codegen_store(
    buffer_ptr: Value,
    index: Value,
    value: Value,
    value_dtype: &DType,
    builder: &mut FunctionBuilder,
) -> Result<()> {
    let element_size = value_dtype.bytes() as i64;

    // Compute byte offset
    let size_val = builder.ins().iconst(cl_types::I64, element_size);
    let byte_offset = builder.ins().imul(index, size_val);

    // Add to base pointer
    let addr = builder.ins().iadd(buffer_ptr, byte_offset);

    // Store
    let flags = MemFlags::new();
    builder.ins().store(flags, value, addr, Offset32::new(0));

    Ok(())
}

/// Generate identity value for reduce operation.
fn codegen_reduce_identity(reduce_op: ReduceOp, dtype: &DType, builder: &mut FunctionBuilder) -> Result<Value> {
    let cl_type = dtype_to_cranelift_type(dtype);
    let is_fp = is_float(dtype);

    let value = match reduce_op {
        ReduceOp::Add => {
            if is_fp {
                if cl_type == cl_types::F32 { builder.ins().f32const(0.0) } else { builder.ins().f64const(0.0) }
            } else {
                builder.ins().iconst(cl_type, 0)
            }
        }
        ReduceOp::Mul => {
            if is_fp {
                if cl_type == cl_types::F32 { builder.ins().f32const(1.0) } else { builder.ins().f64const(1.0) }
            } else {
                builder.ins().iconst(cl_type, 1)
            }
        }
        ReduceOp::Max => {
            if is_fp {
                if cl_type == cl_types::F32 {
                    builder.ins().f32const(f32::NEG_INFINITY)
                } else {
                    builder.ins().f64const(f64::NEG_INFINITY)
                }
            } else if is_signed(dtype) {
                let min_val = match dtype.bytes() {
                    1 => i8::MIN as i64,
                    2 => i16::MIN as i64,
                    4 => i32::MIN as i64,
                    _ => i64::MIN,
                };
                builder.ins().iconst(cl_type, min_val)
            } else {
                builder.ins().iconst(cl_type, 0)
            }
        }
        ReduceOp::Min => {
            if is_fp {
                if cl_type == cl_types::F32 {
                    builder.ins().f32const(f32::INFINITY)
                } else {
                    builder.ins().f64const(f64::INFINITY)
                }
            } else if is_signed(dtype) {
                let max_val = match dtype.bytes() {
                    1 => i8::MAX as i64,
                    2 => i16::MAX as i64,
                    4 => i32::MAX as i64,
                    _ => i64::MAX,
                };
                builder.ins().iconst(cl_type, max_val)
            } else {
                let max_val = match dtype.bytes() {
                    1 => u8::MAX as i64,
                    2 => u16::MAX as i64,
                    4 => u32::MAX as i64,
                    _ => i64::MAX, // Can't represent u64::MAX in i64
                };
                builder.ins().iconst(cl_type, max_val)
            }
        }
    };

    Ok(value)
}

/// Apply reduce operation.
fn codegen_reduce_op(
    reduce_op: ReduceOp,
    acc: Value,
    src: Value,
    dtype: &DType,
    builder: &mut FunctionBuilder,
) -> Result<Value> {
    let is_fp = is_float(dtype);
    let is_sgn = is_signed(dtype);

    let result = match reduce_op {
        ReduceOp::Add => {
            if is_fp {
                builder.ins().fadd(acc, src)
            } else {
                builder.ins().iadd(acc, src)
            }
        }
        ReduceOp::Mul => {
            if is_fp {
                builder.ins().fmul(acc, src)
            } else {
                builder.ins().imul(acc, src)
            }
        }
        ReduceOp::Max => {
            if is_fp {
                builder.ins().fmax(acc, src)
            } else if is_sgn {
                builder.ins().smax(acc, src)
            } else {
                builder.ins().umax(acc, src)
            }
        }
        ReduceOp::Min => {
            if is_fp {
                builder.ins().fmin(acc, src)
            } else if is_sgn {
                builder.ins().smin(acc, src)
            } else {
                builder.ins().umin(acc, src)
            }
        }
    };

    Ok(result)
}
