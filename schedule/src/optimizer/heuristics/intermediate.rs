//! Intermediate optimization heuristics.
//!
//! These heuristics require more analysis but are still straightforward:
//! - Masked upcasts: Upcast small masked dimensions
//! - Grouped reduction: Two-stage reductions with shared memory
//! - Threading: CPU parallelization

use crate::optimizer::{Opt, Scheduler, apply_opt};

use super::helpers::is_masked;

/// Apply upcasts for small masked dimensions.
///
/// Masked dimensions (those appearing in WHERE conditionals) benefit from
/// full upcasting when small (≤ 7 elements). This eliminates branching
/// overhead by fully unrolling the masked loop.
///
/// # Returns
///
/// True if any masked upcasts were applied, false otherwise.
pub fn apply_masked_upcasts(scheduler: &mut Scheduler) -> bool {
    let mut applied = false;

    // Check all upcastable axes
    let upcastable = scheduler.upcastable_dims();

    for axis_idx in upcastable {
        // Check if this axis is masked
        if !is_masked(scheduler, axis_idx) {
            continue;
        }

        // Get the size of this axis
        let rngs = scheduler.rngs();
        if axis_idx >= rngs.len() {
            continue;
        }

        let rng = &rngs[axis_idx];
        if let morok_ir::Op::Range { end, .. } = rng.op()
            && let morok_ir::Op::Const(cv) = end.op()
            && let morok_ir::ConstValue::Int(size) = cv.0
        {
            // Upcast if size is small enough (≤ 7)
            if size > 1 && size <= 7 {
                // Upcast the entire axis
                if apply_opt(scheduler, &Opt::upcast(axis_idx, size as usize), true).is_ok() {
                    applied = true;
                }
            }
        }
    }

    applied
}

/// Try applying grouped reduction for large reductions.
///
/// For kernels with large reduction dimensions, split the reduction into
/// two stages using shared memory:
/// 1. Local reduction within each workgroup (GROUP)
/// 2. Global reduction across workgroups (synchronization)
///
/// This is beneficial when the reduction dimension is much larger than
/// the local workgroup size, as it reduces memory bandwidth requirements.
///
/// # Returns
///
/// True if grouped reduction was applied, false otherwise.
pub fn try_grouped_reduction(scheduler: &mut Scheduler) -> bool {
    // Only apply to GPU backends with local memory
    if !scheduler.renderer().has_local {
        return false;
    }

    // Get all reduce axes
    let reduce_axes = scheduler.axes_of(&[morok_ir::AxisType::Reduce]);
    if reduce_axes.is_empty() {
        return false;
    }

    // Find the largest reduction axis
    let rngs = scheduler.rngs();
    let mut largest_axis = None;
    let mut largest_size = 0;

    for &axis_idx in &reduce_axes {
        if axis_idx >= rngs.len() {
            continue;
        }

        let rng = &rngs[axis_idx];
        if let morok_ir::Op::Range { end, .. } = rng.op()
            && let morok_ir::Op::Const(cv) = end.op()
            && let morok_ir::ConstValue::Int(size) = cv.0
            && size > largest_size
        {
            largest_size = size;
            largest_axis = Some(axis_idx);
        }
    }

    // Only apply if reduction is large enough (> 256)
    if largest_size <= 256 {
        return false;
    }

    if let Some(axis_idx) = largest_axis {
        // Map physical axis to logical reduce axis index
        if let Some(logical_axis) = reduce_axes.iter().position(|&a| a == axis_idx) {
            // Split into groups of 256 (typical workgroup size)
            let group_size = 256.min(largest_size as usize);
            if apply_opt(scheduler, &Opt::group(logical_axis, group_size), true).is_ok() {
                return true;
            }
        }
    }

    false
}

/// Apply CPU threading for parallelization.
///
/// For CPU backends, add threading to parallelize outer global axes.
/// This distributes work across CPU cores using a thread pool.
///
/// Typically threads the first (outermost) global axis to balance
/// work distribution and minimize synchronization overhead.
///
/// # Returns
///
/// True if threading was applied, false otherwise.
pub fn apply_threading(scheduler: &mut Scheduler) -> bool {
    // Only apply to CPU backends with threading support
    if !scheduler.renderer().has_threads {
        return false;
    }

    // Get all global axes
    let global_axes = scheduler.axes_of(&[morok_ir::AxisType::Global]);
    if global_axes.is_empty() {
        return false;
    }

    // Thread the first (outermost) global axis
    let axis_idx = global_axes[0];

    // Get the size to determine thread count
    let rngs = scheduler.rngs();
    if axis_idx >= rngs.len() {
        return false;
    }

    let rng = &rngs[axis_idx];
    if let morok_ir::Op::Range { end, .. } = rng.op()
        && let morok_ir::Op::Const(cv) = end.op()
        && let morok_ir::ConstValue::Int(size) = cv.0
    {
        // Use reasonable thread count (typically 4-8 for CPU)
        // Clamp to actual size to avoid empty work
        let thread_count = (size as usize).min(8);
        if thread_count > 1 && apply_opt(scheduler, &Opt::thread(axis_idx, thread_count), true).is_ok() {
            return true;
        }
    }

    false
}
