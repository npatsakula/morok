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
        _ => return ValidationFailedSnafu { op: "apply_opt", reason: "operation not yet implemented" }.fail(),
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
fn apply_upcast(scheduler: &mut Scheduler, rng: Arc<UOp>, amount: usize) -> Result<(), OptError> {
    let axis_type = match rng.op() {
        Op::Range { axis_type, .. } => *axis_type,
        _ => return ExpectedRangeOperationSnafu.fail(),
    };

    if !matches!(axis_type, AxisType::Global | AxisType::Local | AxisType::Loop) {
        return ValidationFailedSnafu { op: "UPCAST", reason: "can only upcast Global/Local/Loop axes" }.fail();
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
fn apply_unroll(scheduler: &mut Scheduler, axis: usize, amount: usize) -> Result<(), OptError> {
    let unrollable = scheduler.unrollable_dims();
    let real_axis =
        *unrollable.get(axis).ok_or_else(|| AxisOutOfBoundsSnafu { axis, max: unrollable.len() }.build())?;
    let rng = scheduler.rngs()[real_axis].clone();

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
        Op::Range { end, axis_id, axis_type } => (end.clone(), *axis_id, *axis_type),
        _ => return ExpectedRangeOperationSnafu.fail(),
    };

    let (end2, axis_id2, axis_type2) = match rng2.op() {
        Op::Range { end, axis_id, axis_type } => (end.clone(), *axis_id, *axis_type),
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
