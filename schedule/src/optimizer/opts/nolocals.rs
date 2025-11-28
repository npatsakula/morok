//! NOLOCALS optimization - Disable local memory usage.
//!
//! Prevents use of GPU shared/local memory by setting a flag that blocks
//! LOCAL, WARP, and GROUP_REDUCE optimizations.

use crate::optimizer::{Scheduler, error::*};
use morok_ir::{AxisType, Op};

/// Apply NOLOCALS optimization to disable local memory.
///
/// Sets the `dont_use_locals` flag to prevent future LOCAL, WARP, or
/// GROUP_REDUCE optimizations. Useful for debugging or when local memory
/// causes performance issues.
///
/// # Arguments
///
/// * `scheduler` - The scheduler to modify
///
/// # Validation
///
/// - No LOCAL, WARP, or GROUP_REDUCE axes must already exist
///
/// # Example
///
/// ```ignore
/// apply(&mut scheduler)?;
/// // Now LOCAL optimizations will fail
/// ```
pub fn apply(scheduler: &mut Scheduler) -> Result<(), OptError> {
    // Validate no LOCAL/WARP/GROUP_REDUCE axes exist
    for rng in scheduler.rngs() {
        if let Op::Range { axis_type, .. } = rng.op()
            && matches!(axis_type, AxisType::Local | AxisType::Warp | AxisType::GroupReduce)
        {
            return ValidationFailedSnafu {
                op: "NOLOCALS",
                reason: "cannot apply NOLOCALS after LOCAL/WARP/GROUP_REDUCE axes exist",
            }
            .fail();
        }
    }

    scheduler.dont_use_locals = true;
    Ok(())
}
