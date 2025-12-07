//! Simple optimization heuristics.
//!
//! These are straightforward heuristics that don't require complex analysis:
//! - Image upcast: Vectorize for GPU image types
//! - Default upcast: Fallback vectorization
//! - Unroll: Unroll small reduction loops

use crate::optimizer::{Opt, Scheduler, apply_opt};

/// Apply image-specific upcasting for GPU image types.
///
/// GPU image types (image2d_t, image3d_t) benefit from float4 vectorization.
/// If the kernel output is an image type, upcast the last axis by 4.
///
/// # Returns
///
/// True if image upcast was applied, false otherwise.
///
/// # Note
///
/// Currently, morok doesn't track image types explicitly, so this heuristic
/// is a no-op placeholder. It will be implemented when image type support is added.
pub fn apply_image_upcasts(_scheduler: &mut Scheduler) -> bool {
    // TODO: Implement when image types are tracked
    // In Tinygrad: checks bufs[0].dtype.name.startswith('image')
    // If image type, upcast last axis by 4: scheduler.apply_opt(Opt::upcast(last_axis, 4))
    false
}

/// Apply default upcast for fallback vectorization.
///
/// If no upcasting has been applied yet, this applies a conservative
/// 4x vectorization to the first available axis. This ensures some
/// SIMD optimization even when more specialized heuristics don't apply.
///
/// # Returns
///
/// True if default upcast was applied, false otherwise.
pub fn apply_default_upcast(scheduler: &mut Scheduler) -> bool {
    // Only apply if nothing has been upcasted yet
    if scheduler.upcasted() {
        return false;
    }

    // Get the first upcastable axis
    let upcastable = scheduler.upcastable_dims();
    if upcastable.is_empty() {
        return false;
    }

    // Upcast first axis by 4 (conservative SIMD vectorization)
    let axis = upcastable[0];
    if apply_opt(scheduler, &Opt::upcast(axis, 4), true).is_ok() {
        return true;
    }

    false
}

/// Apply loop unrolling for small reduction axes.
///
/// Unrolls reduction loops with size ≤ 32 to eliminate loop overhead
/// and improve instruction-level parallelism. Small reductions benefit
/// from full unrolling since the code size increase is manageable.
///
/// # Returns
///
/// True if any unrolls were applied, false otherwise.
pub fn apply_unroll(scheduler: &mut Scheduler) -> bool {
    let mut applied = false;

    // Get all reduction axes that can be unrolled
    let unrollable = scheduler.unrollable_dims();

    for axis_idx in unrollable {
        // Get the size of this axis
        let rngs = scheduler.rngs();
        if axis_idx >= rngs.len() {
            continue;
        }

        let rng = &rngs[axis_idx];
        if let morok_ir::Op::Range { end, .. } = rng.op()
            && let morok_ir::Op::Const(cv) = end.op()
            && let morok_ir::ConstValue::Int(size) = cv.0
            && size > 1
            && size <= 32
            && false
        // TEMPORARY: disable unroll until codegen supports Unroll axes
        {
            // Map physical axis to logical unroll axis index
            let unroll_axes = scheduler.unrollable_dims();
            if let Some(logical_axis) = unroll_axes.iter().position(|&a| a == axis_idx) {
                // Unroll the entire axis (amount = size)
                if apply_opt(scheduler, &Opt::unroll(logical_axis, size as usize), true).is_ok() {
                    applied = true;
                }
            }
        }
    }

    applied
}
