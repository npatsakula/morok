//! UPCAST optimization - Vectorization (SIMD).
//!
//! Splits a dimension to combine multiple iterations into vector operations.
//! Used for SIMD instructions and register-level parallelism.

use crate::optimizer::{Scheduler, error::*};
use morok_ir::{AxisType, Op, UOp};
use std::rc::Rc;

/// Apply UPCAST optimization to vectorize a dimension.
///
/// Splits the given range into a smaller range and an UPCAST dimension.
/// The UPCAST dimension represents vectorized iterations that execute
/// in parallel via SIMD instructions.
///
/// # Arguments
///
/// * `scheduler` - The scheduler to modify
/// * `rng` - The range to split (must be Global, Local, or Loop)
/// * `amount` - The vectorization factor (must be <= upcast_max)
///
/// # Validation
///
/// - Range must be Global, Local, or Loop axis type
/// - Amount must not exceed device's upcast_max limit
/// - Range size must be divisible by amount
///
/// # Example
///
/// ```ignore
/// // Global(16) -> Global(4) + Upcast(4)
/// apply(&mut scheduler, global_range, 4)?;
/// ```
pub fn apply(scheduler: &mut Scheduler, rng: Rc<UOp>, amount: usize) -> Result<(), OptError> {
    // 1. Validate axis type
    let axis_type = if let Op::Range { axis_type, .. } = rng.op() {
        *axis_type
    } else {
        return ExpectedRangeOperationSnafu.fail();
    };

    if !matches!(axis_type, AxisType::Global | AxisType::Local | AxisType::Loop) {
        return ValidationFailedSnafu { op: "UPCAST", reason: "can only upcast Global/Local/Loop axes" }.fail();
    }

    // 2. Validate amount
    if amount > scheduler.ren.upcast_max {
        return DeviceLimitExceededSnafu { limit_type: "upcast", value: amount, max: scheduler.ren.upcast_max }.fail();
    }

    // 3. Apply transformation
    scheduler.shift_to(rng, amount, AxisType::Upcast, false, None)?;

    Ok(())
}
