//! Unified loop generation for LLVM code generation.
//!
//! All loop constructs (RANGE, REDUCE) share the same 5-block structure matching Tinygrad:
//!
//! ```text
//! entry_N:  br latch_N
//! latch_N:  phi i = [0, entry], [i+1, footer]
//!           phi acc = [identity, entry], [new_acc, footer]  // REDUCE only
//!           cmp = i < end
//!           br cmp, body_N, exit_N
//! body_N:   <loop body>
//! footer_N: br latch_N
//! exit_N:   <continuation>
//! ```

use inkwell::basic_block::BasicBlock;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::values::{BasicValueEnum, FunctionValue, IntValue, PhiValue};
use inkwell::IntPredicate;
use snafu::{OptionExt, ResultExt};

use crate::llvm::error::*;
use crate::llvm::helpers::LoopContext;

/// Build a standard loop with the 5-block structure.
///
/// This creates:
/// - entry_block: unconditional branch to latch
/// - latch_block: PHI node, increment, condition check, conditional branch
/// - body_block: positioned here after call (loop body goes here)
/// - footer_block: branch back to latch
/// - exit_block: continuation after loop
///
/// Returns the loop context and the loop counter value (PHI result).
pub fn build_loop<'ctx>(
    context: &'ctx Context,
    builder: &Builder<'ctx>,
    function: FunctionValue<'ctx>,
    end_val: IntValue<'ctx>,
    loop_id: u64,
) -> Result<(LoopContext<'ctx>, IntValue<'ctx>)> {
    // Create all basic blocks
    let entry_block = context.append_basic_block(function, &format!("loop_entry_{}", loop_id));
    let latch_block = context.append_basic_block(function, &format!("loop_latch_{}", loop_id));
    let body_block = context.append_basic_block(function, &format!("loop_body_{}", loop_id));
    let footer_block = context.append_basic_block(function, &format!("loop_footer_{}", loop_id));
    let exit_block = context.append_basic_block(function, &format!("loop_exit_{}", loop_id));

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
    let phi = builder.build_phi(counter_type, &format!("i{}", loop_id)).context(BuildPhiSnafu)?;

    // Add incoming edge: 0 from entry block
    let zero = counter_type.const_int(0, false);
    phi.add_incoming(&[(&zero, entry_block)]);

    let counter_val = phi.as_basic_value().into_int_value();

    // Increment: counter + 1
    let one = counter_type.const_int(1, false);
    let incremented =
        builder.build_int_add(counter_val, one, &format!("i{}_next", loop_id)).context(ArithmeticSnafu)?;

    // Cast end_val to i64 if needed
    let end_i64 = if end_val.get_type() != counter_type {
        builder.build_int_z_extend(end_val, counter_type, "end_ext").context(CastSnafu)?
    } else {
        end_val
    };

    // Condition: counter < end (unsigned comparison)
    let cmp = builder
        .build_int_compare(IntPredicate::ULT, counter_val, end_i64, &format!("i{}_cmp", loop_id))
        .context(ComparisonSnafu)?;

    // Conditional branch: if counter < end, go to body, else exit
    builder.build_conditional_branch(cmp, body_block, exit_block).context(BuildBranchSnafu)?;

    // Position builder at body block for loop body code
    builder.position_at_end(body_block);

    let loop_ctx = LoopContext { latch_block, footer_block, exit_block, phi, incremented };

    Ok((loop_ctx, counter_val))
}

/// Close a loop by completing the footer block and positioning at exit.
///
/// This should be called when END is encountered for this loop.
pub fn close_loop<'ctx>(builder: &Builder<'ctx>, loop_ctx: &LoopContext<'ctx>) -> Result<()> {
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

    Ok(())
}

/// Build a reduce loop with an accumulator PHI.
///
/// This creates the same 5-block structure as `build_loop` but with an additional
/// accumulator PHI node for reduction operations.
///
/// Returns (loop context, counter value, accumulator PHI).
pub fn build_reduce_loop<'ctx>(
    context: &'ctx Context,
    builder: &Builder<'ctx>,
    function: FunctionValue<'ctx>,
    end_val: IntValue<'ctx>,
    identity: BasicValueEnum<'ctx>,
    reduce_id: u64,
) -> Result<(BasicBlock<'ctx>, BasicBlock<'ctx>, IntValue<'ctx>, PhiValue<'ctx>, PhiValue<'ctx>)> {
    let current_block = builder.get_insert_block().context(NoInsertBlockSnafu)?;

    // Create basic blocks
    let header_block = context.append_basic_block(function, &format!("reduce_header_{}", reduce_id));
    let body_block = context.append_basic_block(function, &format!("reduce_body_{}", reduce_id));
    let exit_block = context.append_basic_block(function, &format!("reduce_exit_{}", reduce_id));

    // Branch from current block to header
    builder.build_unconditional_branch(header_block).context(BuildBranchSnafu)?;

    // Build header block
    builder.position_at_end(header_block);

    // Create accumulator PHI
    let acc_phi = builder.build_phi(identity.get_type(), "acc").context(BuildPhiSnafu)?;
    acc_phi.add_incoming(&[(&identity, current_block)]);

    // Create counter PHI
    let counter_type = context.i64_type();
    let counter_phi = builder.build_phi(counter_type, &format!("reduce_i_{}", reduce_id)).context(BuildPhiSnafu)?;
    let zero = counter_type.const_int(0, false);
    counter_phi.add_incoming(&[(&zero, current_block)]);

    let counter_val = counter_phi.as_basic_value().into_int_value();

    // Cast end_val to i64 if needed
    let end_i64 = if end_val.get_type() != counter_type {
        builder.build_int_z_extend(end_val, counter_type, "end_ext").context(CastSnafu)?
    } else {
        end_val
    };

    // Condition: counter < end
    let cmp =
        builder.build_int_compare(IntPredicate::ULT, counter_val, end_i64, "reduce_cmp").context(ComparisonSnafu)?;

    // Conditional branch: if counter < end, go to body, else exit
    builder.build_conditional_branch(cmp, body_block, exit_block).context(BuildBranchSnafu)?;

    // Position at body block
    builder.position_at_end(body_block);

    Ok((header_block, exit_block, counter_val, counter_phi, acc_phi))
}

/// Complete a reduce loop after computing the new accumulator value.
///
/// This branches back to header, completes PHI nodes, and positions at exit.
pub fn complete_reduce_loop<'ctx>(
    context: &'ctx Context,
    builder: &Builder<'ctx>,
    header_block: BasicBlock<'ctx>,
    exit_block: BasicBlock<'ctx>,
    counter_phi: PhiValue<'ctx>,
    acc_phi: PhiValue<'ctx>,
    new_acc: BasicValueEnum<'ctx>,
) -> Result<BasicValueEnum<'ctx>> {
    let body_block = builder.get_insert_block().context(NoInsertBlockSnafu)?;

    // Increment counter
    let counter_type = context.i64_type();
    let one = counter_type.const_int(1, false);
    let counter_val = counter_phi.as_basic_value().into_int_value();
    let incremented = builder.build_int_add(counter_val, one, "inc").context(ArithmeticSnafu)?;

    // Branch back to header
    builder.build_unconditional_branch(header_block).context(BuildBranchSnafu)?;

    // Complete PHI nodes with incoming from body
    acc_phi.add_incoming(&[(&new_acc, body_block)]);
    counter_phi.add_incoming(&[(&incremented, body_block)]);

    // Position at exit block
    builder.position_at_end(exit_block);

    // Return the final accumulated value
    Ok(acc_phi.as_basic_value())
}
