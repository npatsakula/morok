//! GROUP/GROUPTOP optimization - Two-stage reduction with shared memory.
//!
//! Splits a reduction dimension into a smaller reduction and a GROUP_REDUCE dimension.
//! Uses GPU shared memory for inter-thread communication in the reduction.

use crate::optimizer::{Scheduler, error::*};
use morok_ir::{AxisType, Op, UOp};
use std::sync::Arc;

/// Apply GROUP optimization for two-stage reduction.
///
/// Splits the given reduction range into a smaller range and a GROUP_REDUCE dimension.
/// The GROUP_REDUCE dimension represents a reduction stage using shared memory.
///
/// # Arguments
///
/// * `scheduler` - The scheduler to modify
/// * `rng` - The reduction range to split (must be REDUCE)
/// * `amount` - The group size (number of threads cooperating)
/// * `top` - If true, group dimension is outer (GROUPTOP), else inner (GROUP)
///
/// # Validation
///
/// - Backend must support shared memory (has_local && has_shared)
/// - Shared memory usage must not exceed shared_max
/// - Range must not be inside another reduction
/// - Range size must be divisible by amount
///
/// # Example
///
/// ```ignore
/// // Reduce(64) -> Reduce(8) + GroupReduce(8)
/// apply(&mut scheduler, reduce_range, 8, false)?;
/// ```
pub fn apply(scheduler: &mut Scheduler, rng: Arc<UOp>, amount: usize, top: bool) -> Result<(), OptError> {
    // 1. Validate backend support
    if !scheduler.ren.has_local {
        return UnsupportedFeatureSnafu { feature: "local memory" }.fail();
    }

    if !scheduler.ren.has_shared {
        return UnsupportedFeatureSnafu { feature: "shared memory" }.fail();
    }

    // 2. Validate axis type
    let axis_type = if let Op::Range { axis_type, .. } = rng.op() {
        *axis_type
    } else {
        return ExpectedRangeOperationSnafu.fail();
    };

    if axis_type != AxisType::Reduce {
        return ValidationFailedSnafu { op: "GROUP", reason: "can only group REDUCE axes" }.fail();
    }

    // 3. Calculate shared memory usage
    // upcast_local_sz = product of Upcast, Warp, Local, GroupReduce dimensions
    let upcast_local_sz: usize = scheduler
        .rngs()
        .iter()
        .filter(|r| {
            if let Op::Range { axis_type, .. } = r.op() {
                matches!(axis_type, AxisType::Upcast | AxisType::Warp | AxisType::Local | AxisType::GroupReduce)
            } else {
                false
            }
        })
        .filter_map(|r| {
            if let Op::Range { end, .. } = r.op()
                && let Op::Const(cv) = end.op()
                && let morok_ir::ConstValue::Int(sz) = cv.0
            {
                return Some(sz as usize);
            }
            None
        })
        .product();

    // Find the reduce operation using this range to get dtype
    let reduce_uop = find_reduce_using_range(scheduler, &rng)?;
    let dtype = reduce_uop.dtype();

    // Calculate shared memory size
    let smem_sz = amount * upcast_local_sz * dtype.bytes();

    if smem_sz > scheduler.ren.shared_max {
        return DeviceLimitExceededSnafu { limit_type: "shared memory", value: smem_sz, max: scheduler.ren.shared_max }
            .fail();
    }

    // 4. Check not inside another reduce (nested reductions)
    // Look for OTHER REDUCE operations in the backward slice
    // If found, GROUP is being applied inside a nested reduction
    let reduce_ptr = Arc::as_ptr(&reduce_uop) as *const _;
    for node in reduce_uop.backward_slice() {
        if let Op::Reduce { .. } = node.op() {
            let node_ptr = Arc::as_ptr(&node) as *const _;
            // Skip the current reduce we're working on
            if node_ptr == reduce_ptr {
                continue;
            }

            // Found a different REDUCE operation in the backward slice
            // This means we're trying to GROUP inside a nested reduction
            return ValidationFailedSnafu { op: "GROUP", reason: "cannot apply GROUP inside another reduction" }.fail();
        }
    }

    // 5. Apply transformation
    scheduler.shift_to(rng, amount, AxisType::GroupReduce, top, None)?;

    Ok(())
}

/// Find the REDUCE operation that uses the given range.
fn find_reduce_using_range(scheduler: &Scheduler, rng: &Arc<UOp>) -> Result<Arc<UOp>, OptError> {
    for reduce in scheduler.reduceops() {
        if let Op::Reduce { ranges, .. } = reduce.op() {
            for r in ranges.iter() {
                if Arc::ptr_eq(r, rng) {
                    return Ok(reduce);
                }
            }
        }
    }

    ValidationFailedSnafu { op: "GROUP", reason: "could not find REDUCE operation using this range" }.fail()
}
