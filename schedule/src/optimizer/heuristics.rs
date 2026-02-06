//! Hand-coded optimization heuristics for kernel optimization.
//!
//! Implements Tinygrad-style heuristics for reasonable performance without auto-tuning.
//! Applies optimizations in order: TC → Image → GroupReduce → Upcasts → Unroll → Local → Thread.

use std::sync::Arc;

use morok_ir::{AxisType, BinaryOp, Op, ReduceOp, TernaryOp};

use crate::optimizer::config::HeuristicsConfig;
use crate::optimizer::{Opt, Scheduler, apply_opt};

// ============================================================================
// CONSTANTS
// ============================================================================

/// Default vectorization factor for UPCAST when no other heuristic applies.
/// Value 4 provides good SIMD utilization on most architectures (SSE/NEON).
pub const DEFAULT_UPCAST_FACTOR: usize = 4;

// ============================================================================
// MAIN ENTRY POINT
// ============================================================================

/// Apply hand-coded optimization heuristics to a kernel.
///
/// Heuristics are applied in order:
/// 1. Tensor cores (if matmul pattern)
/// 2. Image upcasts (if image type)
/// 3. Grouped reduction (if large reduce)
/// 4. Masked upcasts (small masked dims)
/// 5. Heuristic upcasts (stride-based ranking)
/// 6. Unroll (small reduction loops)
/// 7. Default upcast (fallback)
/// 8. Local dims (GPU workgroup)
/// 9. Threading (CPU parallel)
pub fn hand_coded_optimizations(scheduler: &mut Scheduler, config: &HeuristicsConfig) {
    use tracing::debug;

    debug!("hand_coded_optimizations: starting");

    // 1. Tensor cores (skip other opts if applied)
    if try_tensor_cores(scheduler, config) {
        apply_local_dims(scheduler, config);
        debug!("hand_coded_optimizations: tensor cores applied, skipping remaining opts");
        return;
    }

    // 2. Image upcasts
    apply_image_upcasts(scheduler);

    // 3. Grouped reduction
    try_grouped_reduction(scheduler, config);

    // 4. Masked upcasts
    apply_masked_upcasts(scheduler);

    // 5. Heuristic upcasts (stride-based ranking, matches Tinygrad's "more upcasts" loop)
    apply_heuristic_upcasts(scheduler);

    // 6. Unroll (BEFORE threading, matching Tinygrad order)
    apply_unroll(scheduler);

    // 7. Default upcast
    if !scheduler.upcasted() {
        apply_default_upcast(scheduler);
    }

    // 8. Local dims
    apply_local_dims(scheduler, config);

    // 9. Threading
    debug!("hand_coded_optimizations: calling apply_threading with max_threads={}", config.thread_count);
    let threading_applied = apply_threading(scheduler, config.thread_count);
    debug!(threading_applied, "hand_coded_optimizations: apply_threading completed");
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Check if kernel has matmul pattern: REDUCE(ADD) of MUL of INDEX ops.
pub fn has_matmul_pattern(scheduler: &Scheduler) -> bool {
    let Some(reduceop) = scheduler.reduceop() else { return false };

    if let Op::Reduce { src, reduce_op, .. } = reduceop.op() {
        if *reduce_op != ReduceOp::Add {
            return false;
        }
        if let Op::Binary(BinaryOp::Mul, left, right) = src.op() {
            let left_is_index = matches!(left.op(), Op::Index { .. })
                || matches!(left.op(), Op::Cast { src, .. } if matches!(src.op(), Op::Index { .. }));
            let right_is_index = matches!(right.op(), Op::Index { .. })
                || matches!(right.op(), Op::Cast { src, .. } if matches!(src.op(), Op::Index { .. }));
            return left_is_index && right_is_index;
        }
    }
    false
}

/// Check if axis is masked (appears in WHERE conditionals).
pub fn is_masked(scheduler: &Scheduler, axis: usize) -> bool {
    let rngs = scheduler.rngs();
    if axis >= rngs.len() {
        return false;
    }
    let target_rng = &rngs[axis];

    for node in scheduler.ast().toposort() {
        if let Op::Ternary(TernaryOp::Where, cond, _, _) = node.op()
            && cond.backward_slice().iter().any(|dep| Arc::ptr_eq(dep, target_rng))
        {
            return true;
        }
    }
    false
}

/// Check if axis has broadcast pattern (stride-0 in some buffer).
pub fn has_broadcast_pattern(scheduler: &Scheduler, axis: usize) -> bool {
    let rngs = scheduler.rngs();
    if axis >= rngs.len() {
        return false;
    }
    let target_rng = &rngs[axis];

    for buf in scheduler.bufs() {
        let in_backward = buf.backward_slice().iter().any(|dep| Arc::ptr_eq(dep, target_rng));
        if !in_backward {
            continue;
        }
        if let Op::Index { indices, .. } = buf.op() {
            let in_index =
                indices.iter().any(|idx| idx.backward_slice().iter().any(|dep| Arc::ptr_eq(dep, target_rng)));
            if !in_index {
                return true;
            }
        }
    }
    false
}

/// Count strides for axis in buffer accesses. Returns (num_buffers, sum_strides).
///
/// Matches Tinygrad's stride counting (heuristic.py:119-126):
/// - num_strides: number of buffers whose index references this range
/// - sum_strides: sum of actual stride values from the index's ADD decomposition
///   (1 for unit stride, CONST value for `range * CONST`)
pub fn count_strides(scheduler: &Scheduler, axis: usize) -> (usize, usize) {
    let rngs = scheduler.rngs();
    if axis >= rngs.len() {
        return (0, 0);
    }
    let target_rng = &rngs[axis];
    let mut num_strides = 0;
    let mut sum_strides: usize = 0;

    for buf in scheduler.bufs() {
        if let Op::Index { indices, .. } = buf.op() {
            // Get the combined linearized index and unwrap WHERE if present
            // Tinygrad: idx = b.src[1].get_idx()
            let idx = indices.first().map(|i| i.get_idx()).unwrap_or_else(|| buf.clone());

            // Tinygrad: if rng in idx.backward_slice: num_strides += 1
            if idx.backward_slice().iter().any(|dep| Arc::ptr_eq(dep, target_rng)) {
                num_strides += 1;
            }

            // Tinygrad: for c in idx.split_uop(Ops.ADD):
            for term in idx.split_uop(BinaryOp::Add) {
                if Arc::ptr_eq(&term, target_rng) {
                    // c is rng → stride 1
                    sum_strides += 1;
                } else if let Op::Binary(BinaryOp::Mul, lhs, rhs) = term.op() {
                    // c.op is Ops.MUL and one side is rng and other is CONST
                    if Arc::ptr_eq(lhs, target_rng)
                        && let Op::Const(cv) = rhs.op()
                        && let morok_ir::ConstValue::Int(v) = cv.0
                    {
                        sum_strides += v as usize;
                    } else if Arc::ptr_eq(rhs, target_rng)
                        && let Op::Const(cv) = lhs.op()
                        && let morok_ir::ConstValue::Int(v) = cv.0
                    {
                        sum_strides += v as usize;
                    }
                }
            }
        }
    }
    (num_strides, sum_strides)
}

// ============================================================================
// SIMPLE HEURISTICS
// ============================================================================

/// Image-specific upcasting (placeholder - not yet implemented).
pub fn apply_image_upcasts(_scheduler: &mut Scheduler) -> bool {
    false
}

/// Default upcast fallback: 4x vectorization on first upcastable axis.
pub fn apply_default_upcast(scheduler: &mut Scheduler) -> bool {
    use morok_ir::Op;
    use tracing::debug;

    if scheduler.upcasted() {
        debug!("apply_default_upcast: skipping (already upcasted)");
        return false;
    }
    let upcastable = scheduler.upcastable_dims();
    debug!(upcastable_dims = ?upcastable, "apply_default_upcast: checking upcastable dims");
    if upcastable.is_empty() {
        debug!("apply_default_upcast: no upcastable dims");
        return false;
    }

    // Tinygrad: upcastable_dims[-1] (innermost upcastable axis)
    let axis_idx = *upcastable.last().unwrap();
    let rngs = scheduler.rngs();

    // Get axis size and check divisibility (Tinygrad: k.full_shape[axis] % upcast_amount != 0)
    if axis_idx < rngs.len()
        && let Op::Range { end, .. } = rngs[axis_idx].op()
        && let Op::Const(cv) = end.op()
        && let morok_ir::ConstValue::Int(size) = cv.0
        && size % DEFAULT_UPCAST_FACTOR as i64 != 0
    {
        debug!(axis_idx, size, factor = DEFAULT_UPCAST_FACTOR, "apply_default_upcast: skipping (size not divisible)");
        return false;
    }

    let result = apply_opt(scheduler, &Opt::upcast(axis_idx, DEFAULT_UPCAST_FACTOR), true);
    debug!(?result, axis = axis_idx, factor = DEFAULT_UPCAST_FACTOR, "apply_default_upcast: apply_opt result");
    result.is_ok()
}

/// Unroll reduction loops, matching Tinygrad's logic (heuristic.py:135-148).
///
/// Conditions: `unrollable_dims.not_empty() AND (upcast_size() <= 4 OR no UNROLL axes) AND upcast_size() < 64`
/// - Small dims (size <= 32): full unroll (amount=0)
/// - Large dims: partial unroll by 4
pub fn apply_unroll(scheduler: &mut Scheduler) -> bool {
    use tracing::debug;

    let unrollable = scheduler.unrollable_dims();
    if unrollable.is_empty() {
        return false;
    }

    let upcast_size = scheduler.upcast_size();
    let has_unroll = !scheduler.axes_of(&[AxisType::Unroll]).is_empty();

    // Tinygrad: (upcast_size() <= 4 or not axes_of(UNROLL)) and upcast_size() < 64
    if upcast_size >= 64 || (upcast_size > 4 && has_unroll) {
        debug!(upcast_size, has_unroll, "apply_unroll: skipping (upcast_size guard)");
        return false;
    }

    // Get last unrollable dim's size (Tinygrad: k.unrollable_dims[-1])
    let last_unrollable = *unrollable.last().unwrap();
    let rngs = scheduler.rngs();
    let size = if last_unrollable < rngs.len()
        && let Op::Range { end, .. } = rngs[last_unrollable].op()
        && let Op::Const(cv) = end.op()
        && let morok_ir::ConstValue::Int(sz) = cv.0
    {
        sz as usize
    } else {
        return false;
    };

    let logical_idx = unrollable.len() - 1;

    if size <= 32 {
        // Full unroll (amount=0 means full unroll).
        // UNROLL creates expanded scalar operations (not vectors like UPCAST),
        // so non-power-of-2 sizes are safe.
        debug!(last_unrollable, size, "apply_unroll: full unroll");
        if apply_opt(scheduler, &Opt::unroll(logical_idx, 0), true).is_ok() {
            // Tinygrad: if small, try unrolling a second reduce dimension too
            if size <= 3 {
                let unrollable2 = scheduler.unrollable_dims();
                if let Some(&last2) = unrollable2.last() {
                    let rngs2 = scheduler.rngs();
                    if last2 < rngs2.len()
                        && let Op::Range { end, .. } = rngs2[last2].op()
                        && let Op::Const(cv) = end.op()
                        && let morok_ir::ConstValue::Int(sz2) = cv.0
                        && sz2 <= 3
                    {
                        let _ = apply_opt(scheduler, &Opt::unroll(unrollable2.len() - 1, 0), true);
                    }
                }
            }
            return true;
        }
    }

    // Partial unroll by 4
    for splits in [4] {
        if size % splits == 0 {
            debug!(last_unrollable, size, splits, "apply_unroll: partial unroll");
            if apply_opt(scheduler, &Opt::unroll(logical_idx, splits), true).is_ok() {
                return true;
            }
        }
    }

    false
}

// ============================================================================
// INTERMEDIATE HEURISTICS
// ============================================================================

/// Upcast small masked dimensions (size <= 7).
pub fn apply_masked_upcasts(scheduler: &mut Scheduler) -> bool {
    let mut applied = false;
    let upcastable = scheduler.upcastable_dims();

    for axis_idx in upcastable {
        if !is_masked(scheduler, axis_idx) {
            continue;
        }
        let rngs = scheduler.rngs();
        if axis_idx >= rngs.len() {
            continue;
        }
        let rng = &rngs[axis_idx];
        if let Op::Range { end, .. } = rng.op()
            && let Op::Const(cv) = end.op()
            && let morok_ir::ConstValue::Int(size) = cv.0
            && size > 1
            && size <= 7
            && apply_opt(scheduler, &Opt::upcast(axis_idx, size as usize), true).is_ok()
        {
            applied = true;
        }
    }
    applied
}

/// Grouped reduction for large reduce dimensions (> threshold).
pub fn try_grouped_reduction(scheduler: &mut Scheduler, config: &HeuristicsConfig) -> bool {
    if !scheduler.renderer().has_local || config.disable_locals {
        return false;
    }
    let reduce_axes = scheduler.axes_of(&[AxisType::Reduce]);
    if reduce_axes.is_empty() {
        return false;
    }

    let rngs = scheduler.rngs();
    let mut largest_axis = None;
    let mut largest_size = 0;
    let threshold = config.grouped_threshold as i64;

    for &axis_idx in &reduce_axes {
        if axis_idx >= rngs.len() {
            continue;
        }
        let rng = &rngs[axis_idx];
        if let Op::Range { end, .. } = rng.op()
            && let Op::Const(cv) = end.op()
            && let morok_ir::ConstValue::Int(size) = cv.0
            && size > largest_size
        {
            largest_size = size;
            largest_axis = Some(axis_idx);
        }
    }

    if largest_size <= threshold {
        return false;
    }
    if let Some(axis_idx) = largest_axis
        && let Some(logical) = reduce_axes.iter().position(|&a| a == axis_idx)
    {
        let group_size = (config.grouped_threshold).min(largest_size as usize);
        if apply_opt(scheduler, &Opt::group(logical, group_size), true).is_ok() {
            return true;
        }
    }
    false
}

/// Apply matmul-specific 2D output tiling (register blocking).
///
/// For matmul C[M,N] = A[M,K] @ B[K,N], this creates a tile of output elements
/// that are computed together, amortizing memory loads across multiple outputs.
///
/// Tinygrad achieves 8×8 register blocking with 64 scalar accumulators.
/// We achieve the same by applying UPCAST to both M and N output axes:
/// - UPCAST M by up to 8 → 8 rows of output
/// - UPCAST N by up to 8 → 8 cols of output → up to 8×8 = 64 outputs
///
/// The devectorize pass (no_vectorized_alu) converts these to independent scalar
/// accumulators via MulAcc splitting, matching Tinygrad's `float acc0[64]` pattern.
///
/// Tile sizes are chosen flexibly based on divisibility: tries 8, 7, 6, 5, 4 in order.
pub fn apply_matmul_tiling(scheduler: &mut Scheduler, config: &HeuristicsConfig) -> bool {
    use tracing::debug;

    // Only apply to matmul patterns
    if !has_matmul_pattern(scheduler) {
        return false;
    }

    // Skip if output_upcast is disabled in config
    if !config.output_upcast {
        debug!("apply_matmul_tiling: skipped (output_upcast disabled)");
        return false;
    }

    // Get output axes directly - include OUTER for reduce kernels (matmul)
    // Note: upcastable_dims() excludes Outer, but matmul needs it
    let output_axes = scheduler.axes_of(&[AxisType::Outer, AxisType::Global, AxisType::Loop]);
    debug!(output_axes = ?output_axes, "apply_matmul_tiling: output axes");

    // Need at least 2 output axes for 2D tiling
    if output_axes.len() < 2 {
        debug!("apply_matmul_tiling: not enough output axes (need 2)");
        return false;
    }

    // Upcast factors in decreasing order of preference
    // Larger tiles = more register blocking = better memory amortization
    const UPCAST_FACTORS: [usize; 5] = [8, 7, 6, 5, 4];

    // Collect axes with their sizes
    let rngs = scheduler.rngs();
    let mut axes_with_sizes: Vec<(usize, usize)> = Vec::new();

    for &axis_idx in output_axes.iter().take(2) {
        if axis_idx >= rngs.len() {
            continue;
        }
        if let Op::Range { end, .. } = rngs[axis_idx].op()
            && let Op::Const(cv) = end.op()
            && let morok_ir::ConstValue::Int(size) = cv.0
            && size >= 4
        {
            axes_with_sizes.push((axis_idx, size as usize));
        }
    }

    if axes_with_sizes.len() < 2 {
        debug!(found = axes_with_sizes.len(), "apply_matmul_tiling: not enough output axes");
        return false;
    }

    // Apply UPCAST to each axis with the largest divisible factor
    let mut applied = false;
    for (axis_idx, size) in axes_with_sizes {
        // Find largest factor that divides size evenly
        if let Some(&factor) = UPCAST_FACTORS.iter().find(|&&f| size >= f && size % f == 0)
            && apply_opt(scheduler, &Opt::upcast(axis_idx, factor), true).is_ok()
        {
            debug!(axis = axis_idx, factor, size, "apply_matmul_tiling: applied UPCAST");
            applied = true;
        }
    }

    applied
}

/// Legacy function for compatibility - calls apply_matmul_tiling
pub fn apply_matmul_output_upcasting(scheduler: &mut Scheduler, config: &HeuristicsConfig) -> bool {
    apply_matmul_tiling(scheduler, config)
}

/// CPU threading for parallelizable loop axes.
///
/// Matches Tinygrad's threading heuristic (heuristic.py:179-188):
/// 1. Descending thread list: [32, 16, 12, 8, 6, 5, 4, 3, 2]
/// 2. Minimum work check: skip if `prod(full_shape) / 131072 < threads`
/// 3. Only LOOP axes (matmul output dims are Loop from rangeify)
pub fn apply_threading(scheduler: &mut Scheduler, max_threads: usize) -> bool {
    use tracing::debug;

    if !scheduler.renderer().has_threads || max_threads <= 1 {
        return false;
    }

    // Minimum work check: prod(full_shape) // (128 << 10) < threads → skip
    let total_elements: i64 = scheduler.full_shape().iter().copied().product();

    // Tinygrad's descending thread count list
    const THREAD_LIST: [usize; 9] = [32, 16, 12, 8, 6, 5, 4, 3, 2];

    for &threads in &THREAD_LIST {
        if threads > max_threads {
            continue;
        }
        // Skip if not enough work per thread (Tinygrad: prod(full_shape) // (128 << 10) < threads)
        if total_elements < (threads as i64) * 131072 {
            continue;
        }

        // Only thread LOOP axes (matching Tinygrad)
        let loop_axes = scheduler.axes_of(&[AxisType::Loop]);
        let mut thread_applied = false;
        for &axis_idx in &loop_axes {
            let rngs = scheduler.rngs();
            if axis_idx >= rngs.len() {
                continue;
            }
            if let Op::Range { end, .. } = rngs[axis_idx].op()
                && let Op::Const(cv) = end.op()
                && let morok_ir::ConstValue::Int(size) = cv.0
                && (size as usize).is_multiple_of(threads)
            {
                // Found divisible axis — try applying, then break regardless of success
                // (Tinygrad: break is inside the divisibility check)
                thread_applied = apply_opt(scheduler, &Opt::thread(axis_idx, threads), true).is_ok();
                if thread_applied {
                    debug!(axis = axis_idx, threads, "apply_threading: applied THREAD");
                }
                break;
            }
        }
        if thread_applied {
            return true;
        }
    }

    false
}

// ============================================================================
// COMPLEX HEURISTICS
// ============================================================================

/// Heuristic upcasts based on stride analysis.
///
/// Matches Tinygrad's "more upcasts" loop (heuristic.py:107-133):
/// - Only enters loop when `prod(output_shape[upcastable_dims]) >= 1024`
/// - Terminates when `upcast_size() >= 32`
/// - Uses factors `[3, 4]`
/// - Ranks by `(num_strides, sum_strides)` ascending (fewest strides = best)
/// - Excludes axes that are NOT stride-0 in any buffer (Tinygrad's broadcast check)
pub fn apply_heuristic_upcasts(scheduler: &mut Scheduler) -> bool {
    use tracing::debug;

    let mut applied = false;
    let mut upcasted_axes: Vec<usize> = Vec::new();

    loop {
        // Tinygrad: while prod(output_shape[i] for i in upcastable_dims) >= 1024 and upcast_size() < 32
        let upcastable = scheduler.upcastable_dims();
        if upcastable.is_empty() {
            break;
        }

        let output_shape_product: i64 = {
            let rngs = scheduler.rngs();
            upcastable
                .iter()
                .filter_map(|&idx| {
                    if idx < rngs.len()
                        && let Op::Range { end, .. } = rngs[idx].op()
                        && let Op::Const(cv) = end.op()
                        && let morok_ir::ConstValue::Int(sz) = cv.0
                    {
                        Some(sz)
                    } else {
                        None
                    }
                })
                .product()
        };

        if output_shape_product < 1024 || scheduler.upcast_size() >= 32 {
            debug!(
                output_shape_product,
                upcast_size = scheduler.upcast_size(),
                "apply_heuristic_upcasts: terminating (threshold)"
            );
            break;
        }

        // Build choices: (num_strides, sum_strides, axis, upcast_amount)
        // Tinygrad: for axis, upcast_amount in product(upcastable_dims, [3, 4])
        let mut choices: Vec<(usize, usize, usize, usize)> = Vec::new();

        let upcast_and_unroll_ranges = scheduler.ranges_of(&[AxisType::Upcast, AxisType::Unroll]);

        for &axis_idx in &upcastable {
            if upcasted_axes.contains(&axis_idx) {
                continue;
            }

            let rngs = scheduler.rngs();
            if axis_idx >= rngs.len() {
                continue;
            }
            let rng = &rngs[axis_idx];

            // Tinygrad stride-0 check (lines 117-118):
            // axis must be NOT in some buffer's index backward_slice AND
            // all existing UPCAST/UNROLL ranges ARE in that buffer's index backward_slice
            let has_stride0 = {
                let bufs = scheduler.bufs();
                bufs.iter().any(|buf| {
                    if let Op::Index { indices, .. } = buf.op() {
                        let idx_backward: Vec<_> =
                            indices.iter().flat_map(|idx| idx.backward_slice().into_iter()).collect();
                        let rng_not_in_idx = !idx_backward.iter().any(|dep| Arc::ptr_eq(dep, rng));
                        let all_upcast_in_idx = upcast_and_unroll_ranges
                            .iter()
                            .all(|r2| idx_backward.iter().any(|dep| Arc::ptr_eq(dep, r2)));
                        rng_not_in_idx && all_upcast_in_idx
                    } else {
                        false
                    }
                })
            };

            if !has_stride0 {
                continue;
            }

            for &upcast_amount in &[3usize, 4] {
                let size = if let Op::Range { end, .. } = rng.op()
                    && let Op::Const(cv) = end.op()
                    && let morok_ir::ConstValue::Int(sz) = cv.0
                {
                    sz
                } else {
                    continue;
                };

                if size % upcast_amount as i64 != 0 {
                    continue;
                }

                let (num_strides, sum_strides) = count_strides(scheduler, axis_idx);
                choices.push((num_strides, sum_strides, axis_idx, upcast_amount));
            }
        }

        if choices.is_empty() {
            debug!("apply_heuristic_upcasts: no valid choices, breaking");
            break;
        }

        // Sort ascending by (num_strides, sum_strides) — fewest strides wins
        choices.sort();
        let (_, _, best_axis, best_amount) = choices[0];

        debug!(best_axis, best_amount, "apply_heuristic_upcasts: applying upcast");
        if apply_opt(scheduler, &Opt::upcast(best_axis, best_amount), true).is_ok() {
            upcasted_axes.push(best_axis);
            applied = true;
        } else {
            break;
        }
    }

    applied
}

/// GPU workgroup configuration (2D for 2D+ kernels, 1D fallback).
pub fn apply_local_dims(scheduler: &mut Scheduler, config: &HeuristicsConfig) -> bool {
    if !scheduler.renderer().has_local || config.disable_locals {
        return false;
    }
    let global_axes = scheduler.axes_of(&[AxisType::Global]);
    if global_axes.is_empty() {
        return false;
    }

    let local_max = scheduler.renderer().local_max.unwrap_or(1024);
    let output_shape = scheduler.output_shape();

    // 2D layout for 2D+ kernels
    if output_shape.len() >= 2 && global_axes.len() >= 2 && local_max >= 256 {
        let axis0 = global_axes[0];
        let axis1 = global_axes[1];
        if apply_opt(scheduler, &Opt::local(axis0, 16), true).is_ok() {
            let _ = apply_opt(scheduler, &Opt::local(axis1, 16), true);
            return true;
        }
    }

    // 1D fallback
    if !global_axes.is_empty() {
        let local_size = local_max.min(256);
        if apply_opt(scheduler, &Opt::local(global_axes[0], local_size), true).is_ok() {
            return true;
        }
    }
    false
}

/// Tensor core optimization for matmul patterns.
pub fn try_tensor_cores(scheduler: &mut Scheduler, config: &HeuristicsConfig) -> bool {
    use crate::optimizer::config::TcUsage;

    if config.tc_enabled == TcUsage::Disabled {
        return false;
    }
    if scheduler.renderer().tensor_cores.is_empty() {
        return false;
    }
    if !has_matmul_pattern(scheduler) {
        return false;
    }

    let opt = Opt::tc(None, config.tc_select.as_i32(), config.tc_opt.as_usize(), config.tc_enabled.as_usize());
    apply_opt(scheduler, &opt, true).is_ok()
}
