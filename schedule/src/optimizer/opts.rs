//! Optimization operation implementations.
//!
//! Implements: UPCAST (SIMD), LOCAL (shared memory), GROUP (two-stage reduction),
//! UNROLL (loop unrolling), SWAP (axis reordering), NOLOCALS (disable local mem).

use std::collections::HashMap;
use std::sync::Arc;

use morok_ir::{AxisType, ConstValue, Op, UOp, UOpKey};

use crate::optimizer::{Opt, OptOps, Scheduler, error::*, tc};

// ============================================================================
// DISPATCHER
// ============================================================================

/// Apply an optimization to the scheduler.
pub fn apply_opt(scheduler: &mut Scheduler, opt: &Opt, append_opt: bool) -> Result<(), OptError> {
    let real_axis = scheduler.real_axis(opt.op, opt.axis)?;
    let rng = if real_axis >= 0 { Some(scheduler.rngs()[real_axis as usize].clone()) } else { None };

    match opt.op {
        OptOps::TC => {
            let (tc_select, tc_opt, use_tensor_cores) = opt.arg.tc()?;
            tc::apply(scheduler, tc_select, tc_opt, use_tensor_cores)?;
        }
        OptOps::UPCAST => {
            apply_upcast(scheduler, rng.ok_or_else(|| MissingAxisParameterSnafu.build())?, opt.arg.int()?)?;
        }
        OptOps::LOCAL => {
            apply_local(scheduler, rng.ok_or_else(|| MissingAxisParameterSnafu.build())?, opt.arg.int()?)?;
        }
        OptOps::UNROLL => {
            apply_unroll(scheduler, opt.axis.ok_or_else(|| MissingAxisParameterSnafu.build())?, opt.arg.int()?)?;
        }
        OptOps::NOLOCALS => {
            apply_nolocals(scheduler)?;
        }
        OptOps::SWAP => {
            apply_swap(scheduler, opt.axis.ok_or_else(|| MissingAxisParameterSnafu.build())?, opt.arg.swap()?)?;
        }
        OptOps::GROUP => {
            apply_group(scheduler, rng.ok_or_else(|| MissingAxisParameterSnafu.build())?, opt.arg.int()?, false)?;
        }
        OptOps::GROUPTOP => {
            apply_group(scheduler, rng.ok_or_else(|| MissingAxisParameterSnafu.build())?, opt.arg.int()?, true)?;
        }
        OptOps::THREAD => {
            apply_thread(scheduler, rng.ok_or_else(|| MissingAxisParameterSnafu.build())?, opt.arg.int()?)?;
        }
        OptOps::PADTO => {
            apply_padto(scheduler, rng.ok_or_else(|| MissingAxisParameterSnafu.build())?, opt.arg.int()?)?;
        }
    }

    if append_opt {
        scheduler.applied_opts.push(opt.clone());
    }
    Ok(())
}

// ============================================================================
// UPCAST - Vectorization (SIMD)
// ============================================================================

/// Split dimension into smaller range + UPCAST for vector operations.
///
/// UPCAST is for output dimension vectorization (OUTER/GLOBAL/LOCAL/LOOP).
/// For reduce axis unrolling, use UNROLL instead.
fn apply_upcast(scheduler: &mut Scheduler, rng: Arc<UOp>, amount: usize) -> Result<(), OptError> {
    let axis_type = match rng.op() {
        Op::Range { axis_type, .. } => *axis_type,
        _ => return ExpectedRangeOperationSnafu.fail(),
    };

    // UPCAST is for output dimension vectorization (parallel lanes compute different outputs)
    // Allowed: OUTER (reduce kernel outputs), GLOBAL/LOCAL/LOOP (elementwise outputs)
    // REDUCE/GROUP_REDUCE should use UNROLL instead (unrolled iterations, scalar accumulators)
    if !matches!(axis_type, AxisType::Outer | AxisType::Global | AxisType::Local | AxisType::Loop) {
        return ValidationFailedSnafu { op: "UPCAST", reason: "can only upcast Outer/Global/Local/Loop axes" }.fail();
    }

    if amount > scheduler.ren.upcast_max {
        return DeviceLimitExceededSnafu { limit_type: "upcast", value: amount, max: scheduler.ren.upcast_max }.fail();
    }

    scheduler.shift_to(rng, amount, AxisType::Upcast, false, None)?;
    Ok(())
}

// ============================================================================
// LOCAL - Shared memory (GPU workgroup)
// ============================================================================

/// Split dimension into smaller range + LOCAL for GPU workgroup threads.
fn apply_local(scheduler: &mut Scheduler, rng: Arc<UOp>, amount: usize) -> Result<(), OptError> {
    if !scheduler.ren.has_local {
        return UnsupportedFeatureSnafu { feature: "local memory" }.fail();
    }
    if scheduler.dont_use_locals {
        return ValidationFailedSnafu { op: "LOCAL", reason: "NOLOCALS was applied" }.fail();
    }

    let axis_type = match rng.op() {
        Op::Range { axis_type, .. } => *axis_type,
        _ => return ExpectedRangeOperationSnafu.fail(),
    };

    if !matches!(axis_type, AxisType::Global | AxisType::Loop) {
        return ValidationFailedSnafu { op: "LOCAL", reason: "can only localize Global/Loop axes" }.fail();
    }

    scheduler.shift_to(rng, amount, AxisType::Local, false, None)?;
    Ok(())
}

// ============================================================================
// GROUP/GROUPTOP - Two-stage reduction
// ============================================================================

/// Split reduction into smaller range + GROUP_REDUCE using shared memory.
fn apply_group(scheduler: &mut Scheduler, rng: Arc<UOp>, amount: usize, top: bool) -> Result<(), OptError> {
    if !scheduler.ren.has_local {
        return UnsupportedFeatureSnafu { feature: "local memory" }.fail();
    }
    if !scheduler.ren.has_shared {
        return UnsupportedFeatureSnafu { feature: "shared memory" }.fail();
    }

    let axis_type = match rng.op() {
        Op::Range { axis_type, .. } => *axis_type,
        _ => return ExpectedRangeOperationSnafu.fail(),
    };

    if axis_type != AxisType::Reduce {
        return ValidationFailedSnafu { op: "GROUP", reason: "can only group REDUCE axes" }.fail();
    }

    // Calculate shared memory usage
    let upcast_local_sz: usize = scheduler
        .rngs()
        .iter()
        .filter_map(|r| {
            if let Op::Range { axis_type, end, .. } = r.op()
                && matches!(axis_type, AxisType::Upcast | AxisType::Warp | AxisType::Local | AxisType::GroupReduce)
                && let Op::Const(cv) = end.op()
                && let ConstValue::Int(sz) = cv.0
            {
                return Some(sz as usize);
            }
            None
        })
        .product();

    let reduce_uop = find_reduce_using_range(scheduler, &rng)?;
    let smem_sz = amount * upcast_local_sz * reduce_uop.dtype().bytes();

    if smem_sz > scheduler.ren.shared_max {
        return DeviceLimitExceededSnafu { limit_type: "shared memory", value: smem_sz, max: scheduler.ren.shared_max }
            .fail();
    }

    // Check not inside nested reduction
    let reduce_ptr = Arc::as_ptr(&reduce_uop);
    for node in reduce_uop.backward_slice() {
        if let Op::Reduce { .. } = node.op()
            && Arc::as_ptr(&node) != reduce_ptr
        {
            return ValidationFailedSnafu { op: "GROUP", reason: "cannot apply GROUP inside another reduction" }.fail();
        }
    }

    scheduler.shift_to(rng, amount, AxisType::GroupReduce, top, None)?;
    Ok(())
}

fn find_reduce_using_range(scheduler: &Scheduler, rng: &Arc<UOp>) -> Result<Arc<UOp>, OptError> {
    for reduce in scheduler.reduceops() {
        if let Op::Reduce { ranges, .. } = reduce.op()
            && ranges.iter().any(|r| Arc::ptr_eq(r, rng))
        {
            return Ok(reduce);
        }
    }
    ValidationFailedSnafu { op: "GROUP", reason: "could not find REDUCE using this range" }.fail()
}

// ============================================================================
// UNROLL - Loop unrolling
// ============================================================================

/// Split reduction into smaller range + UNROLL for compile-time expansion.
/// When `amount == 0`, the entire axis is unrolled (full unroll), matching Tinygrad's convention.
fn apply_unroll(scheduler: &mut Scheduler, axis: usize, amount: usize) -> Result<(), OptError> {
    let unrollable = scheduler.unrollable_dims();
    let real_axis =
        *unrollable.get(axis).ok_or_else(|| AxisOutOfBoundsSnafu { axis, max: unrollable.len() }.build())?;
    let rng = scheduler.rngs()[real_axis].clone();

    // Resolve amount=0 to full axis size (full unroll, matching Tinygrad's convention)
    let amount = if amount == 0 {
        if let Op::Range { end, .. } = rng.op()
            && let Op::Const(cv) = end.op()
            && let morok_ir::ConstValue::Int(sz) = cv.0
        {
            sz as usize
        } else {
            return ValidationFailedSnafu { op: "UNROLL", reason: "full unroll requires constant axis size" }.fail();
        }
    } else {
        amount
    };

    const MAX_UNROLL: usize = 32;
    if amount > MAX_UNROLL {
        return DeviceLimitExceededSnafu { limit_type: "unroll", value: amount, max: MAX_UNROLL }.fail();
    }

    scheduler.shift_to(rng, amount, AxisType::Unroll, false, None)?;
    Ok(())
}

// ============================================================================
// SWAP - Axis reordering
// ============================================================================

/// Swap axis_id values of two GLOBAL ranges for memory access optimization.
fn apply_swap(scheduler: &mut Scheduler, axis: usize, other_axis: usize) -> Result<(), OptError> {
    if axis == other_axis {
        return ValidationFailedSnafu { op: "SWAP", reason: "cannot swap axis with itself" }.fail();
    }

    let rngs = scheduler.rngs();
    if axis >= rngs.len() {
        return AxisOutOfBoundsSnafu { axis, max: rngs.len() }.fail();
    }
    if other_axis >= rngs.len() {
        return AxisOutOfBoundsSnafu { axis: other_axis, max: rngs.len() }.fail();
    }

    let (rng1, rng2) = (&rngs[axis], &rngs[other_axis]);

    let (end1, axis_id1, axis_type1) = match rng1.op() {
        Op::Range { end, axis_id, axis_type, .. } => (end.clone(), *axis_id, *axis_type),
        _ => return ExpectedRangeOperationSnafu.fail(),
    };

    let (end2, axis_id2, axis_type2) = match rng2.op() {
        Op::Range { end, axis_id, axis_type, .. } => (end.clone(), *axis_id, *axis_type),
        _ => return ExpectedRangeOperationSnafu.fail(),
    };

    if axis_type1 != AxisType::Global || axis_type2 != AxisType::Global {
        return ValidationFailedSnafu { op: "SWAP", reason: "both axes must be GLOBAL" }.fail();
    }

    let new_rng1 = UOp::range_axis(end1, axis_id2, axis_type1);
    let new_rng2 = UOp::range_axis(end2, axis_id1, axis_type2);

    #[allow(clippy::mutable_key_type)]
    let mut subst_map = HashMap::new();
    subst_map.insert(UOpKey(rng1.clone()), new_rng1);
    subst_map.insert(UOpKey(rng2.clone()), new_rng2);
    let new_ast = scheduler.ast().substitute(&subst_map);
    scheduler.set_ast(new_ast);

    Ok(())
}

// ============================================================================
// NOLOCALS - Disable local memory
// ============================================================================

/// Set flag to prevent future LOCAL/WARP/GROUP_REDUCE optimizations.
fn apply_nolocals(scheduler: &mut Scheduler) -> Result<(), OptError> {
    for rng in scheduler.rngs() {
        if let Op::Range { axis_type, .. } = rng.op()
            && matches!(axis_type, AxisType::Local | AxisType::Warp | AxisType::GroupReduce)
        {
            return ValidationFailedSnafu {
                op: "NOLOCALS",
                reason: "cannot apply after LOCAL/WARP/GROUP_REDUCE exist",
            }
            .fail();
        }
    }
    scheduler.dont_use_locals = true;
    Ok(())
}

// ============================================================================
// THREAD - CPU parallel dispatch
// ============================================================================

// ============================================================================
// PADTO - Tensor core alignment padding
// ============================================================================

/// Pad dimension to alignment for tensor core compatibility.
///
/// PADTO rounds up a loop dimension to enable tensor core alignment.
/// Based on Tinygrad's PADTO (kernel.py).
///
/// # Constraints
///
/// - Only pad constant-sized axes
/// - Cannot pad UPCAST/UNROLL/THREAD axes (already vectorized/expanded)
/// - For REDUCE axes: only with ADD reduction and no unsafe ops before reduce
/// - Don't add more than 4x work (padding 1→5 rejected)
///
/// # Algorithm
///
/// 1. Round up range size to alignment
/// 2. Create validity condition: idx < old_size
/// 3. Add validity gate to all INDEX ops using this range
fn apply_padto(scheduler: &mut Scheduler, rng: Arc<UOp>, alignment: usize) -> Result<(), OptError> {
    use morok_ir::ReduceOp;

    let (end, axis_id, axis_type) = match rng.op() {
        Op::Range { end, axis_id, axis_type, .. } => (end.clone(), *axis_id, *axis_type),
        _ => return ExpectedRangeOperationSnafu.fail(),
    };

    // Constraint 1: only pad constant-sized axes
    let old_sz = match end.op() {
        Op::Const(cv) => match cv.0 {
            ConstValue::Int(v) if v > 0 => v as usize,
            _ => return ValidationFailedSnafu { op: "PADTO", reason: "range end must be positive integer" }.fail(),
        },
        _ => return ValidationFailedSnafu { op: "PADTO", reason: "can only pad constant-sized axes" }.fail(),
    };

    // Constraint 2: cannot pad UPCAST/UNROLL/THREAD axes
    if matches!(axis_type, AxisType::Upcast | AxisType::Unroll | AxisType::Thread) {
        return ValidationFailedSnafu { op: "PADTO", reason: "cannot pad vectorized/unrolled/thread axes" }.fail();
    }

    // Calculate new padded size
    let new_sz = old_sz.div_ceil(alignment) * alignment;

    // No-op if already aligned
    if new_sz == old_sz {
        return Ok(());
    }

    // Constraint 4: don't add more than 4x work
    if old_sz * 4 < new_sz {
        return ValidationFailedSnafu { op: "PADTO", reason: "padding would add more than 4x work" }.fail();
    }

    // Constraint 3: for REDUCE axes, only with ADD and no unsafe ops
    if matches!(axis_type, AxisType::Reduce | AxisType::GroupReduce)
        && let Some(reduce_op) = scheduler.reduceop()
    {
        // Check reduce operation is ADD
        if let Op::Reduce { reduce_op: op, .. } = reduce_op.op()
            && *op != ReduceOp::Add
        {
            return ValidationFailedSnafu { op: "PADTO", reason: "can only pad ADD reductions (not MAX/MUL)" }.fail();
        }
        // Check for unsafe operations before reduce
        if has_unsafe_ops_before_reduce(&reduce_op) {
            return ValidationFailedSnafu {
                op: "PADTO",
                reason: "cannot pad with unsafe ops (EXP, LOG, DIV, comparisons) before reduce",
            }
            .fail();
        }
    }

    // Create new padded range
    let new_end = UOp::index_const(new_sz as i64);
    let new_rng = UOp::range_axis(new_end, axis_id, axis_type);

    // Create validity condition: new_rng < old_size
    let old_sz_const = UOp::index_const(old_sz as i64);
    let valid = new_rng
        .try_cmplt(&old_sz_const)
        .map_err(|_| ValidationFailedSnafu { op: "PADTO", reason: "failed to create validity condition" }.build())?;

    // Build substitution map
    #[allow(clippy::mutable_key_type)]
    let mut subst_map = HashMap::new();
    subst_map.insert(UOpKey(rng.clone()), new_rng.clone());

    // Update INDEX operations that use this range - add validity gate
    for buf_op in scheduler.bufs() {
        if buf_uses_range(&buf_op, &rng)
            && let Op::Index { buffer, indices, gate } = buf_op.op()
        {
            // Combine validity condition with existing gate if present
            let combined_gate = match gate {
                Some(existing) => valid
                    .try_and_op(existing)
                    .map_err(|_| ValidationFailedSnafu { op: "PADTO", reason: "failed to combine gates" }.build())?,
                None => valid.clone(),
            };
            // Create new INDEX with combined gate
            let new_index =
                UOp::index().buffer(buffer.clone()).indices(indices.clone()).gate(combined_gate).call().map_err(
                    |_| ValidationFailedSnafu { op: "PADTO", reason: "failed to create gated INDEX" }.build(),
                )?;
            subst_map.insert(UOpKey(buf_op.clone()), new_index);
        }
    }

    // Apply substitutions
    let new_ast = scheduler.ast().substitute(&subst_map);
    scheduler.set_ast(new_ast);

    Ok(())
}

/// Check if a buffer INDEX operation uses a specific range.
fn buf_uses_range(buf_op: &Arc<UOp>, rng: &Arc<UOp>) -> bool {
    if let Op::Index { indices, .. } = buf_op.op() {
        // Check if the range appears in the indices dependency graph
        for idx in indices {
            for node in idx.toposort() {
                if Arc::ptr_eq(&node, rng) {
                    return true;
                }
            }
        }
    }
    false
}

/// Check for unsafe operations before reduce that prevent PADTO.
///
/// Tinygrad's UnsafePad group - cannot pad reduce axes if these appear before reduction:
/// - RECIPROCAL, LOG2, EXP2, IDIV, POW (non-linear ops where padding zeros changes result)
/// - Comparisons (LT, etc.) that could mask valid data
fn has_unsafe_ops_before_reduce(reduce_op: &Arc<UOp>) -> bool {
    use morok_ir::types::{BinaryOp, UnaryOp};

    for node in reduce_op.toposort() {
        match node.op() {
            // Unsafe unary ops
            Op::Unary(UnaryOp::Reciprocal | UnaryOp::Log2 | UnaryOp::Exp2, _) => return true,
            // Unsafe binary ops
            Op::Binary(BinaryOp::Idiv | BinaryOp::Pow, _, _) => return true,
            // Comparisons before sum are unsafe (padding zeros would add false comparisons)
            Op::Binary(BinaryOp::Lt, _, _) => return true,
            _ => {}
        }
    }
    false
}

// ============================================================================
// THREAD - CPU parallel dispatch
// ============================================================================

/// Split dimension into smaller range + THREAD for CPU parallel dispatch.
///
/// THREAD works like GPU's GLOBAL but for CPU: instead of GPU thread blocks,
/// we use OS threads (via rayon). The work partition is baked into index
/// expressions at optimization time - runtime just provides thread_id.
///
/// # Safety
///
/// Buffer safety is guaranteed by shift_to() transformation:
/// - Each thread_id maps to disjoint output indices
/// - Index formula: `output[thread_id * chunk_size + local_idx]`
/// - Same buffer pointers can be safely passed to all threads
fn apply_thread(scheduler: &mut Scheduler, rng: Arc<UOp>, amount: usize) -> Result<(), OptError> {
    // Validate renderer supports threads
    if !scheduler.ren.has_threads {
        return UnsupportedFeatureSnafu { feature: "CPU threads" }.fail();
    }

    // Check if already threaded - make THREAD opt idempotent
    // This allows replaying cached opts even when prepare_scheduler pre-applies threading
    let thread_axes = scheduler.axes_of(&[AxisType::Thread]);
    if !thread_axes.is_empty() {
        tracing::debug!("THREAD opt skipped: scheduler already has Thread axis");
        return Ok(());
    }

    // Validate thread count within limits
    if let Some(global_max) = &scheduler.ren.global_max
        && let Some(&max_threads) = global_max.first()
        && amount > max_threads
    {
        return DeviceLimitExceededSnafu { limit_type: "thread count", value: amount, max: max_threads }.fail();
    }

    // Validate axis type (must be parallelizable)
    let axis_type = match rng.op() {
        Op::Range { axis_type, .. } => *axis_type,
        _ => return ExpectedRangeOperationSnafu.fail(),
    };

    // Outer, Global, Loop can be threaded
    // Note: Reduce kernels keep Outer axes (convert_outer_to_loop skips them)
    if !matches!(axis_type, AxisType::Outer | AxisType::Global | AxisType::Loop) {
        return ValidationFailedSnafu { op: "THREAD", reason: "can only thread Outer/Global/Loop axes" }.fail();
    }

    // Apply shift_to with top=true (outer-most position, like Tinygrad's core_id)
    let _ = scheduler.shift_to(rng, amount, AxisType::Thread, true, None)?;
    Ok(())
}
