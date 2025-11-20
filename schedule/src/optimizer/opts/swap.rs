//! SWAP optimization - Axis reordering.
//!
//! Reorders two GLOBAL axes to optimize memory access patterns.
//! Swapping axes can improve cache locality and coalescing.

use crate::optimizer::{Scheduler, error::*};
use morok_ir::{AxisType, Op, UOp, UOpKey};
use std::collections::HashMap;

/// Apply SWAP optimization to reorder two axes.
///
/// Swaps the axis_id values of two GLOBAL ranges, which changes their
/// relative ordering in the loop nest without changing iteration counts.
///
/// # Arguments
///
/// * `scheduler` - The scheduler to modify
/// * `axis` - First axis index (into rngs)
/// * `other_axis` - Second axis index (into rngs)
///
/// # Validation
///
/// - Both axes must be GLOBAL type
/// - Axes must be different
/// - Both axes must be valid indices
///
/// # Example
///
/// ```ignore
/// // Swap axes 0 and 1 to change from row-major to column-major
/// apply(&mut scheduler, 0, 1)?;
/// ```
pub fn apply(scheduler: &mut Scheduler, axis: usize, other_axis: usize) -> Result<(), OptError> {
    // Validate axes are different
    if axis == other_axis {
        return ValidationFailedSnafu { op: "SWAP", reason: "cannot swap axis with itself" }.fail();
    }

    let rngs = scheduler.rngs();

    // Validate indices
    if axis >= rngs.len() {
        return AxisOutOfBoundsSnafu { axis, max: rngs.len() }.fail();
    }
    if other_axis >= rngs.len() {
        return AxisOutOfBoundsSnafu { axis: other_axis, max: rngs.len() }.fail();
    }

    let rng1 = &rngs[axis];
    let rng2 = &rngs[other_axis];

    // Extract axis properties
    let (end1, axis_id1, axis_type1) = if let Op::Range { end, axis_id, axis_type } = rng1.op() {
        (end.clone(), *axis_id, *axis_type)
    } else {
        return ExpectedRangeOperationSnafu.fail();
    };

    let (end2, axis_id2, axis_type2) = if let Op::Range { end, axis_id, axis_type } = rng2.op() {
        (end.clone(), *axis_id, *axis_type)
    } else {
        return ExpectedRangeOperationSnafu.fail();
    };

    // Validate both are GLOBAL
    if axis_type1 != AxisType::Global {
        return ValidationFailedSnafu { op: "SWAP", reason: "first axis must be GLOBAL" }.fail();
    }
    if axis_type2 != AxisType::Global {
        return ValidationFailedSnafu { op: "SWAP", reason: "second axis must be GLOBAL" }.fail();
    }

    // Create new ranges with swapped axis_ids
    let new_rng1 = UOp::range_axis(end1, axis_id2, axis_type1);
    let new_rng2 = UOp::range_axis(end2, axis_id1, axis_type2);

    // Substitute both ranges
    #[allow(clippy::mutable_key_type)] // UOpKey uses stable ID for Hash/Eq (safe)
    let mut subst_map = HashMap::new();
    subst_map.insert(UOpKey(rng1.clone()), new_rng1);
    subst_map.insert(UOpKey(rng2.clone()), new_rng2);
    let new_ast = scheduler.ast().substitute(&subst_map);
    scheduler.set_ast(new_ast);

    Ok(())
}
