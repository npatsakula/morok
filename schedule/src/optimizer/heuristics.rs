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

    // 3.5. Matmul output dimension upcasting (CPU-only, vectorizes along N)
    apply_matmul_output_upcasting(scheduler, config);

    // 4. Masked upcasts
    apply_masked_upcasts(scheduler);

    // 5. Heuristic upcasts
    apply_heuristic_upcasts(scheduler);

    // 6. Default upcast
    if !scheduler.upcasted() {
        apply_default_upcast(scheduler);
    }

    // 7. Local dims
    apply_local_dims(scheduler, config);

    // 8. Threading
    debug!("hand_coded_optimizations: calling apply_threading with max_threads={}", config.thread_count);
    let threading_applied = apply_threading(scheduler, config.thread_count);
    debug!(threading_applied, "hand_coded_optimizations: apply_threading completed");

    // 9. Unroll (AFTER threading so we can check output-per-thread compatibility)
    // Unrolling reduce creates vector accumulators; must ensure unroll factor <= output-per-thread
    apply_unroll(scheduler, config);
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
pub fn count_strides(scheduler: &Scheduler, axis: usize) -> (usize, usize) {
    let rngs = scheduler.rngs();
    if axis >= rngs.len() {
        return (0, 0);
    }
    let target_rng = &rngs[axis];
    let mut num_strides = 0;
    let mut sum_strides = 0;

    for buf in scheduler.bufs() {
        if let Op::Index { indices, .. } = buf.op() {
            let uses_axis =
                indices.iter().any(|idx| idx.backward_slice().iter().any(|dep| Arc::ptr_eq(dep, target_rng)));
            if uses_axis {
                num_strides += 1;
                for idx in indices.iter() {
                    if idx.backward_slice().iter().any(|dep| Arc::ptr_eq(dep, target_rng)) {
                        sum_strides += 1;
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

    let axis_idx = upcastable[0];
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

/// Unroll small reduction loops (size <= threshold).
///
/// Uses `upcast_size()` as a generic guard to prevent exponential vector width growth.
/// This matches Tinygrad's approach (heuristic.py:138) of checking total vectorization width.
///
/// When threading is active, unrolling reduce creates vectorized accumulators.
/// We must ensure the unroll factor is compatible with output-per-thread:
/// - output_per_thread = output_size / thread_count
/// - If unroll_factor > output_per_thread, the vector store would overflow thread's output slice
///
/// When register blocking (4×4 output tiling) is active:
/// - upcast_size() is already 16 (4×4)
/// - K-vectorization would create 16×K wide vectors, causing strided B access
/// - We skip K-unrolling entirely; LLVM can unroll the scalar K loop if beneficial
pub fn apply_unroll(scheduler: &mut Scheduler, config: &HeuristicsConfig) -> bool {
    use tracing::debug;

    let mut applied = false;
    let unrollable = scheduler.unrollable_dims();
    let threshold = config.unroll_threshold as i64;

    debug!(?unrollable, threshold, "apply_unroll: starting");

    // Calculate output-per-thread - determines how many elements each thread writes.
    // For full reductions (no output axes), output_per_thread = 1, preventing UNROLL.
    //
    // When unrolling reduce, each thread writes Vector<N> where N = unroll factor.
    // We must ensure N <= output_per_thread to avoid overlapping writes.
    let output_per_thread: usize = {
        let rngs = scheduler.rngs();

        // Helper to extract constant size from a Range
        let get_axis_size = |idx: usize| -> Option<usize> {
            if idx < rngs.len()
                && let Op::Range { end, .. } = rngs[idx].op()
                && let Op::Const(cv) = end.op()
                && let morok_ir::ConstValue::Int(sz) = cv.0
            {
                return Some(sz as usize);
            }
            None
        };

        // Calculate output size (product of output axes: Outer, Loop, Global)
        // For full reductions (no output axes), this is 1
        let output_axes = scheduler.axes_of(&[AxisType::Outer, AxisType::Loop, AxisType::Global]);
        let output_size: usize = output_axes.iter().filter_map(|&idx| get_axis_size(idx)).product::<usize>().max(1);

        // Account for existing UPCAST on output
        let upcast_size = scheduler.upcast_size();
        let effective_output_size = output_size * upcast_size.max(1);

        let thread_axes = scheduler.axes_of(&[AxisType::Thread]);
        if thread_axes.is_empty() {
            // No threading: use effective_output_size directly
            // For full reductions, this is 1, preventing UNROLL
            effective_output_size
        } else {
            let thread_count: usize = thread_axes.iter().filter_map(|&idx| get_axis_size(idx)).product();
            if thread_count > 0 { effective_output_size / thread_count } else { effective_output_size }
        }
    };

    debug!(output_per_thread, upcast_size = scheduler.upcast_size(), "apply_unroll: calculated output_per_thread");

    // Compute SIMD width from reduce dtype (assuming AVX 256-bit = 32 bytes)
    // For f32: 32/4 = 8, for f64: 32/8 = 4, for f16: 32/2 = 16
    let simd_width: usize = scheduler
        .reduceop()
        .map(|r| 32 / r.dtype().bytes().max(1))
        .unwrap_or(8) // Default to f32 width if no reduce
        .clamp(2, 16); // Reasonable bounds

    // Iterate innermost first (like Tinygrad) to unroll inner reduce loops first
    for axis_idx in unrollable.into_iter().rev() {
        let rngs = scheduler.rngs();
        if axis_idx >= rngs.len() {
            continue;
        }
        let rng = &rngs[axis_idx];
        let size = if let Op::Range { end, .. } = rng.op()
            && let Op::Const(cv) = end.op()
            && let morok_ir::ConstValue::Int(sz) = cv.0
        {
            sz
        } else {
            continue;
        };
        if size <= 1 {
            continue;
        }

        // Determine unroll factor:
        // - Small dims (<= threshold): full unroll
        // - Large dims (> threshold): largest SIMD-aligned divisor of size
        let factor = if size <= threshold {
            size as usize // Full unroll for small dimensions
        } else if config.k_vectorize {
            // Find largest divisor of size that fits in SIMD width
            // Try simd_width, simd_width/2, simd_width/4, ... down to 2
            let mut f = simd_width;
            while f >= 2 {
                if size % (f as i64) == 0 {
                    break;
                }
                f /= 2;
            }
            if f < 2 {
                continue; // No suitable divisor found
            }
            f
        } else {
            continue;
        };

        // Generic guard: skip if combined upcast_size would exceed 32
        // This prevents exponential vector width growth from multiple unroll sources
        if scheduler.upcast_size() * factor > 32 {
            debug!(
                axis_idx,
                factor,
                upcast_size = scheduler.upcast_size(),
                "apply_unroll: skipping (upcast would exceed 32)"
            );
            continue;
        }

        // Threading guard: skip if unroll factor exceeds output-per-thread
        // Unrolling reduce creates Vector<N> accumulators; each thread must have
        // enough output elements to store the vector without overlapping neighbors
        if factor > output_per_thread {
            debug!(axis_idx, factor, output_per_thread, "apply_unroll: skipping (factor > output_per_thread)");
            continue;
        }

        debug!(axis_idx, factor, "apply_unroll: applying unroll");
        let unroll_axes = scheduler.unrollable_dims();
        if let Some(logical) = unroll_axes.iter().position(|&a| a == axis_idx)
            && apply_opt(scheduler, &Opt::unroll(logical, factor), true).is_ok()
        {
            applied = true;
            debug!(axis_idx, factor, "apply_unroll: successfully applied");
        }
    }
    applied
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

/// CPU threading for outer parallelizable axes.
///
/// Finds the outermost threadable axis and applies THREAD optimization.
/// Threadable axes: Outer (reduce kernels), Loop, Global (elementwise kernels).
///
/// # Arguments
///
/// * `scheduler` - The scheduler to apply threading to
/// * `max_threads` - Maximum thread count (from config.thread_count)
///
/// Note: Reduce kernels keep Outer axes (convert_outer_to_loop skips them),
/// so we must check Outer in addition to Loop/Global.
pub fn apply_threading(scheduler: &mut Scheduler, max_threads: usize) -> bool {
    if !scheduler.renderer().has_threads || max_threads <= 1 {
        return false;
    }

    // Threadable axes: Outer (reduce kernels), Loop/Global (elementwise)
    // Reduce kernels keep Outer axes because convert_outer_to_loop() skips them
    let threadable_axes = scheduler.axes_of(&[AxisType::Outer, AxisType::Loop, AxisType::Global]);
    if threadable_axes.is_empty() {
        return false;
    }

    // Apply to first (outermost) threadable axis
    let axis_idx = threadable_axes[0];
    let rngs = scheduler.rngs();
    if axis_idx >= rngs.len() {
        return false;
    }

    let rng = &rngs[axis_idx];
    if let Op::Range { end, .. } = rng.op()
        && let Op::Const(cv) = end.op()
        && let morok_ir::ConstValue::Int(size) = cv.0
    {
        let size = size as usize;

        // Find largest divisor of size that is <= max_threads
        // This ensures even division (THREAD opt requires it)
        let thread_count = (2..=max_threads).rev().find(|&t| size.is_multiple_of(t)).unwrap_or(1);

        if thread_count > 1 && apply_opt(scheduler, &Opt::thread(axis_idx, thread_count), true).is_ok() {
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
/// For matmul patterns, limits to single axis UPCAST to avoid vector width
/// mismatches (A vectorized along rows, B vectorized along cols are incompatible).
pub fn apply_heuristic_upcasts(scheduler: &mut Scheduler) -> bool {
    use tracing::debug;

    let mut applied = false;
    let upcastable = scheduler.upcastable_dims();
    debug!(upcastable = ?upcastable, "apply_heuristic_upcasts: starting");
    if upcastable.is_empty() {
        debug!("apply_heuristic_upcasts: no upcastable dims");
        return false;
    }

    // Matmul pattern: limit to single axis to avoid incompatible vectorizations
    // A[i,k] vectorized along i and B[k,j] vectorized along j can't multiply element-wise
    let is_matmul = has_matmul_pattern(scheduler);
    let max_upcast_axes = if is_matmul { 1 } else { usize::MAX };

    // Rank axes by desirability
    let mut ranked_axes: Vec<(usize, i32)> = upcastable
        .iter()
        .map(|&axis_idx| {
            let mut score = 0;
            if has_broadcast_pattern(scheduler, axis_idx) {
                score += 1000;
            }
            let (num_strides, sum_strides) = count_strides(scheduler, axis_idx);
            score -= num_strides as i32 * 10;
            score -= sum_strides as i32;
            (axis_idx, score)
        })
        .collect();
    ranked_axes.sort_by(|a, b| b.1.cmp(&a.1));

    let mut upcast_product = 1;
    let mut upcast_count = 0;
    let target_upcast = 8;

    for (axis_idx, _) in ranked_axes {
        if upcast_product >= target_upcast || upcast_count >= max_upcast_axes {
            break;
        }
        let rngs = scheduler.rngs();
        if axis_idx >= rngs.len() {
            continue;
        }
        let rng = &rngs[axis_idx];
        if let Op::Range { end, .. } = rng.op()
            && let Op::Const(cv) = end.op()
            && let morok_ir::ConstValue::Int(size) = cv.0
        {
            let remaining = target_upcast / upcast_product;
            let factors = if remaining >= 4 {
                vec![4, 3, 2]
            } else if remaining >= 3 {
                vec![3, 2]
            } else if remaining >= 2 {
                vec![2]
            } else {
                continue;
            };

            for factor in factors {
                // Skip factors that don't divide evenly (Tinygrad: k.full_shape[axis] % upcast_amount != 0)
                if size % factor as i64 != 0 {
                    debug!(axis_idx, factor, size, "apply_heuristic_upcasts: skipping (not divisible)");
                    continue;
                }
                debug!(axis_idx, factor, size, "apply_heuristic_upcasts: trying upcast");
                let result = apply_opt(scheduler, &Opt::upcast(axis_idx, factor), true);
                debug!(?result, axis_idx, factor, "apply_heuristic_upcasts: apply_opt result");
                if size >= factor as i64 && result.is_ok() {
                    upcast_product *= factor;
                    upcast_count += 1;
                    applied = true;
                    break;
                }
            }
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
