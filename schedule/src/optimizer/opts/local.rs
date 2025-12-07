//! LOCAL optimization - Shared memory (GPU workgroup).
//!
//! Moves a dimension into GPU shared/local memory by creating a LOCAL axis.
//! This enables data sharing between threads in a workgroup.

use crate::optimizer::{Scheduler, error::*};
use morok_ir::{AxisType, Op, UOp};
use std::sync::Arc;

/// Apply LOCAL optimization to use shared memory.
///
/// Splits the given range into a smaller range and a LOCAL dimension.
/// The LOCAL dimension represents iterations that map to GPU workgroup
/// threads with access to shared memory.
///
/// # Arguments
///
/// * `scheduler` - The scheduler to modify
/// * `rng` - The range to split (must be Global or Loop)
/// * `amount` - The local dimension size
///
/// # Validation
///
/// - Backend must support local memory (has_local)
/// - NOLOCALS must not have been applied
/// - Range must be Global or Loop axis type
/// - Range size must be divisible by amount
///
/// # Example
///
/// ```ignore
/// // Global(64) -> Global(8) + Local(8)
/// apply(&mut scheduler, global_range, 8)?;
/// ```
pub fn apply(scheduler: &mut Scheduler, rng: Arc<UOp>, amount: usize) -> Result<(), OptError> {
    // 1. Validate backend support
    if !scheduler.ren.has_local {
        return UnsupportedFeatureSnafu { feature: "local memory" }.fail();
    }

    if scheduler.dont_use_locals {
        return ValidationFailedSnafu { op: "LOCAL", reason: "NOLOCALS was applied" }.fail();
    }

    // 2. Validate axis type
    let axis_type = if let Op::Range { axis_type, .. } = rng.op() {
        *axis_type
    } else {
        return ExpectedRangeOperationSnafu.fail();
    };

    if !matches!(axis_type, AxisType::Global | AxisType::Loop) {
        return ValidationFailedSnafu { op: "LOCAL", reason: "can only localize Global/Loop axes" }.fail();
    }

    // 3. Apply transformation
    scheduler.shift_to(rng, amount, AxisType::Local, false, None)?;

    Ok(())
}
