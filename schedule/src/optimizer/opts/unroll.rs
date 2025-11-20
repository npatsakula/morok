//! UNROLL optimization - Loop unrolling for reductions.
//!
//! Unrolls reduction loops to improve instruction-level parallelism
//! and reduce loop overhead.

use crate::optimizer::{Scheduler, error::*};
use morok_ir::AxisType;

/// Apply UNROLL optimization to a reduction dimension.
///
/// Splits a reduction range into a smaller range and an UNROLL dimension.
/// The UNROLL dimension represents loop iterations that are fully expanded
/// at compile time for better instruction scheduling.
///
/// # Arguments
///
/// * `scheduler` - The scheduler to modify
/// * `axis` - Logical index into unrollable_dims() (not absolute axis)
/// * `amount` - The unroll factor (typically 2-32)
///
/// # Validation
///
/// - Axis must be valid index into unrollable_dims()
/// - Amount should be reasonable (typically <= 32 for code size)
/// - Range size must be divisible by amount
///
/// # Example
///
/// ```ignore
/// // Reduce(64) -> Reduce(8) + Unroll(8)
/// apply(&mut scheduler, 0, 8)?;  // 0 = first unrollable dimension
/// ```
pub fn apply(scheduler: &mut Scheduler, axis: usize, amount: usize) -> Result<(), OptError> {
    // 1. Map logical axis to real axis
    let unrollable = scheduler.unrollable_dims();
    let real_axis =
        *unrollable.get(axis).ok_or_else(|| AxisOutOfBoundsSnafu { axis, max: unrollable.len() }.build())?;

    let rng = scheduler.rngs()[real_axis].clone();

    // 2. Validate amount (reasonable limit for code size)
    const MAX_UNROLL: usize = 32;
    if amount > MAX_UNROLL {
        return DeviceLimitExceededSnafu { limit_type: "unroll", value: amount, max: MAX_UNROLL }.fail();
    }

    // 3. Apply transformation
    scheduler.shift_to(rng, amount, AxisType::Unroll, false, None)?;

    Ok(())
}
