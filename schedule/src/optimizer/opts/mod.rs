//! Optimization operation implementations.
//!
//! Each module implements a specific kernel optimization:
//! - `tc`: Tensor cores (hardware-accelerated matrix multiplication)
//! - `upcast`: Vectorization (SIMD)
//! - `local`: Shared memory (GPU workgroup)
//! - `unroll`: Loop unrolling (reduce)
//! - `nolocals`: Disable local memory
//! - `swap`: Axis reordering
//! - `group`: Two-stage reduction with shared memory

pub mod group;
pub mod local;
pub mod nolocals;
pub mod swap;
pub mod tc;
pub mod unroll;
pub mod upcast;

use super::error::*;
use crate::optimizer::{Opt, OptOps, Scheduler};

/// Apply an optimization to the scheduler.
///
/// Maps logical axis indices to physical ranges and dispatches to the
/// appropriate optimization implementation.
///
/// # Arguments
///
/// * `scheduler` - The scheduler to modify
/// * `opt` - The optimization to apply
/// * `append_opt` - Whether to record this optimization in applied_opts
///
/// # Returns
///
/// `Ok(())` if successful, or an error if validation fails.
pub fn apply_opt(scheduler: &mut Scheduler, opt: &Opt, append_opt: bool) -> Result<(), OptError> {
    // Map logical axis to real range
    let real_axis = scheduler.real_axis(opt.op, opt.axis)?;
    let rng = if real_axis >= 0 { Some(scheduler.rngs()[real_axis as usize].clone()) } else { None };

    // Dispatch to specific implementation
    match opt.op {
        OptOps::TC => {
            let (tc_select, tc_opt, use_tensor_cores) = opt.arg.tc()?;
            tc::apply(scheduler, tc_select, tc_opt, use_tensor_cores)?;
        }
        OptOps::UPCAST => {
            let amount = opt.arg.int()?;
            upcast::apply(scheduler, rng.ok_or_else(|| MissingAxisParameterSnafu.build())?, amount)?;
        }
        OptOps::LOCAL => {
            let amount = opt.arg.int()?;
            local::apply(scheduler, rng.ok_or_else(|| MissingAxisParameterSnafu.build())?, amount)?;
        }
        OptOps::UNROLL => {
            let amount = opt.arg.int()?;
            let axis = opt.axis.ok_or_else(|| MissingAxisParameterSnafu.build())?;
            unroll::apply(scheduler, axis, amount)?;
        }
        OptOps::NOLOCALS => {
            nolocals::apply(scheduler)?;
        }
        OptOps::SWAP => {
            let axis = opt.axis.ok_or_else(|| MissingAxisParameterSnafu.build())?;
            let other_axis = opt.arg.swap()?;
            swap::apply(scheduler, axis, other_axis)?;
        }
        OptOps::GROUP => {
            let amount = opt.arg.int()?;
            group::apply(scheduler, rng.ok_or_else(|| MissingAxisParameterSnafu.build())?, amount, false)?;
        }
        OptOps::GROUPTOP => {
            let amount = opt.arg.int()?;
            group::apply(scheduler, rng.ok_or_else(|| MissingAxisParameterSnafu.build())?, amount, true)?;
        }
        _ => return ValidationFailedSnafu { op: "apply_opt", reason: "operation not yet implemented" }.fail(),
    }

    // Record optimization
    if append_opt {
        scheduler.applied_opts.push(opt.clone());
    }

    Ok(())
}
