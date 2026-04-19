//! Hand-coded optimization heuristics for kernel optimization.
//!
//! Implements Tinygrad-style heuristics for reasonable performance without auto-tuning.
//! Applies optimizations in order: TC → Image → GroupReduce → Upcasts → Unroll → Local → Thread.

use std::sync::Arc;

use morok_dtype::DType;
use morok_ir::{AxisId, AxisType, BinaryOp, Op, ReduceOp, TernaryOp};

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
    // Post-TC UPCAST/LOCAL are handled inside try_tensor_cores (non-AMX only)
    if try_tensor_cores(scheduler, config) {
        debug!("hand_coded_optimizations: tensor cores applied, skipping remaining opts");
        return;
    }

    // 2. Image upcasts
    apply_image_upcasts(scheduler);

    // 2.5. Matvec fast-path
    if apply_matvec_fast_path(scheduler, config) {
        debug!("hand_coded_optimizations: matvec fast-path applied, skipping remaining opts");
        return;
    }

    // 3. Grouped reduction
    try_grouped_reduction(scheduler, config);

    // Guard: no more opts if we are grouping (Tinygrad: if k.group_for_reduces: return k)
    if scheduler.group_for_reduces() > 0 {
        debug!("hand_coded_optimizations: group_for_reduces active, skipping remaining opts");
        return;
    }

    // 4. Masked upcasts
    apply_masked_upcasts(scheduler);

    // 5. Heuristic upcasts (stride-based ranking, matches Tinygrad's "more upcasts" loop)
    apply_heuristic_upcasts(scheduler);

    // 6. Unroll (BEFORE threading, matching Tinygrad order)
    apply_unroll(scheduler);

    // 7. Default upcast
    if scheduler.axes_of(&[AxisType::Upcast]).is_empty() {
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
            && cond.backward_slice_ids().contains(&target_rng.id)
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
        let in_backward = buf.backward_slice_ids().contains(&target_rng.id);
        if !in_backward {
            continue;
        }
        if let Op::Index { indices, .. } = buf.op() {
            let in_index = indices.iter().any(|idx| idx.backward_slice_ids().contains(&target_rng.id));
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
            if idx.backward_slice_ids().contains(&target_rng.id) {
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

/// Image-specific upcasting/unrolling parity with Tinygrad.
///
/// For image buffers, find a unit-stride axis whose extent is divisible by 4.
/// Prefer UPCAST on that axis when it's output-parallel; otherwise UNROLL the
/// same axis when it's a reduction axis.
pub fn apply_image_upcasts(scheduler: &mut Scheduler) -> bool {
    let mut applied = false;

    // Snapshot to avoid borrow conflicts while mutating scheduler.
    let bufs = scheduler.bufs().to_vec();
    for buf in bufs {
        let Op::Index { buffer, indices, .. } = buf.op() else {
            continue;
        };
        if !matches!(buffer.dtype(), DType::Image { .. }) {
            continue;
        }

        let Some(first_idx) = indices.first() else {
            continue;
        };
        let linear_idx = first_idx.get_idx();

        // Tinygrad parity: choose first range term in linearized index with size % 4 == 0.
        let axis = linear_idx
            .split_uop(BinaryOp::Add)
            .into_iter()
            .filter_map(|term| {
                if !matches!(term.op(), Op::Range { .. }) || term.divisible_by(4).is_none() {
                    return None;
                }
                scheduler.rngs().iter().position(|r| Arc::ptr_eq(r, &term))
            })
            .next();

        let Some(axis) = axis else {
            continue;
        };

        if scheduler.upcastable_dims().contains(&axis) {
            if apply_opt(scheduler, &Opt::upcast(axis, 4), true).is_ok() {
                applied = true;
            }
        } else {
            let unrollable = scheduler.unrollable_dims();
            if let Some(logical_axis) = unrollable.iter().position(|&i| i == axis)
                && apply_opt(scheduler, &Opt::unroll(logical_axis, 4), true).is_ok()
            {
                applied = true;
            }
        }
    }

    applied
}

/// Default upcast fallback: 4x vectorization on first upcastable axis.
pub fn apply_default_upcast(scheduler: &mut Scheduler) -> bool {
    use morok_ir::Op;
    use tracing::debug;

    if !scheduler.axes_of(&[AxisType::Upcast]).is_empty() {
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
///
/// Matches Tinygrad heuristic.py:97-105: collect all masked-upcastable axes first,
/// then apply in REVERSE order. Reverse iteration is critical — upcast of a higher-indexed
/// axis doesn't shift lower-indexed axes in the rngs list, preserving index validity.
pub fn apply_masked_upcasts(scheduler: &mut Scheduler) -> bool {
    let upcastable = scheduler.upcastable_dims();

    // Phase 1: Collect candidates (Tinygrad heuristic.py:97-104)
    let mut product: i64 = 1;
    let mut to_upcast: Vec<(usize, usize)> = Vec::new();

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
            && product * size <= 49
        {
            to_upcast.push((axis_idx, size as usize));
            product *= size;
        }
    }

    // Phase 2: Apply in reverse order (Tinygrad: to_upcast[::-1])
    let mut applied = false;
    for (axis_idx, size) in to_upcast.into_iter().rev() {
        if apply_opt(scheduler, &Opt::upcast(axis_idx, size), true).is_ok() {
            applied = true;
        }
    }
    applied
}

/// Grouped reduction for small output dimensions (Tinygrad heuristic.py:83-89).
///
/// When the product of upcastable output dimensions is small (<= 2048),
/// apply GROUPTOP on output axes to enable local reduction.
pub fn try_grouped_reduction(scheduler: &mut Scheduler, config: &HeuristicsConfig) -> bool {
    if !scheduler.renderer().has_local || config.disable_locals || !scheduler.renderer().has_shared {
        return false;
    }

    // Tinygrad: prod(k.output_shape[i] for i in k.upcastable_dims) <= 2048
    let upcastable = scheduler.upcastable_dims();
    let full_shape = scheduler.full_shape();
    let group_for_reduces: i64 = upcastable.iter().map(|&i| full_shape.get(i).copied().unwrap_or(1)).product();

    if group_for_reduces > 2048 {
        return false;
    }

    // Tinygrad: for axis, sz in itertools.product((0, 1, 2), (16,)):
    //   try: k.apply_opt(Opt(OptOps.GROUPTOP, axis, sz)); break
    for axis in 0..3 {
        if apply_opt(scheduler, &Opt::grouptop(axis, 16), true).is_ok() {
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

fn find_axis_by_axis_id(scheduler: &Scheduler, axis_id: AxisId) -> Option<usize> {
    scheduler.rngs().iter().enumerate().find_map(|(i, rng)| {
        if let Op::Range { axis_id: id, .. } = rng.op()
            && *id == axis_id
        {
            return Some(i);
        }
        None
    })
}

/// Tinygrad-style matvec fast-path.
///
/// Applies `GROUP` on the reduce axis and `LOCAL`/`UPCAST` on one global output
/// axis when the index structure matches matrix-vector style access.
pub fn apply_matvec_fast_path(scheduler: &mut Scheduler, config: &HeuristicsConfig) -> bool {
    use tracing::debug;

    let block_size = config.matvec_blocksize;
    let threads_per_row = config.threads_per_row;
    let rows_per_thread = config.rows_per_thread;

    if !scheduler.renderer().has_local
        || !scheduler.renderer().has_shared
        || !config.matvec_enabled
        || (block_size <= 1 && threads_per_row <= 1 && rows_per_thread <= 1)
    {
        return false;
    }

    if block_size == 0 || threads_per_row == 0 || rows_per_thread == 0 {
        return false;
    }

    let Some(reduceop) = scheduler.reduceop() else {
        return false;
    };
    let Op::Reduce { src, reduce_op, .. } = reduceop.op() else {
        return false;
    };
    if *reduce_op != ReduceOp::Add || scheduler.full_shape().len() < 2 {
        return false;
    }

    let Op::Binary(BinaryOp::Mul, left, right) = src.op() else {
        return false;
    };
    let (idx0_src, idx1_src) = match (left.op(), right.op()) {
        (Op::Index { indices: i0, .. }, Op::Index { indices: i1, .. }) => {
            let Some(i0) = i0.first() else {
                return false;
            };
            let Some(i1) = i1.first() else {
                return false;
            };
            (i0.get_idx(), i1.get_idx())
        }
        _ => return false,
    };

    let Some(first_reduce_rng) = scheduler.ranges_of(&[AxisType::Reduce]).first().cloned() else {
        return false;
    };

    // Tinygrad parity checks:
    // 1) idx0 must contain the first reduce range as a top-level ADD term.
    // 2) idx1 must include all ranges used by idx0.
    let idx0_has_first_reduce = idx0_src.split_uop(BinaryOp::Add).iter().any(|u| Arc::ptr_eq(u, &first_reduce_rng));
    if !idx0_has_first_reduce {
        return false;
    }

    let idx1_ranges = idx1_src.ranges();
    if !idx0_src.ranges().iter().all(|r| idx1_ranges.iter().any(|cand| Arc::ptr_eq(cand, r))) {
        return false;
    }

    if first_reduce_rng.divisible_by(threads_per_row).is_none() {
        return false;
    }

    let Some(row_tile) = block_size.checked_mul(rows_per_thread) else {
        return false;
    };
    if row_tile == 0 {
        return false;
    }

    let full_shape = scheduler.full_shape();
    for global_idx in scheduler.axes_of(&[AxisType::Global]) {
        let Some(&global_dim) = full_shape.get(global_idx) else {
            continue;
        };
        if global_dim <= 0 || (global_dim as usize) % row_tile != 0 {
            continue;
        }

        let mut trial = scheduler.clone();

        // Tinygrad behavior: GROUP is best-effort in this fast path.
        if threads_per_row > 1 {
            let _ = apply_opt(&mut trial, &Opt::group(0, threads_per_row), true);
        }

        let mut current_axis = global_idx;
        let axis_id = trial
            .rngs()
            .get(current_axis)
            .and_then(|rng| if let Op::Range { axis_id, .. } = rng.op() { Some(*axis_id) } else { None });

        if block_size > 1 {
            if apply_opt(&mut trial, &Opt::local(current_axis, block_size), true).is_err() {
                continue;
            }
            if let Some(axis_id) = axis_id {
                if let Some(updated_axis) = find_axis_by_axis_id(&trial, axis_id) {
                    current_axis = updated_axis;
                } else if rows_per_thread > 1 {
                    continue;
                }
            }
        }

        if rows_per_thread > 1 && apply_opt(&mut trial, &Opt::upcast(current_axis, rows_per_thread), true).is_err() {
            continue;
        }

        debug!(global_idx, block_size, threads_per_row, rows_per_thread, "apply_matvec_fast_path: applied");
        *scheduler = trial;
        return true;
    }

    false
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

    // Minimum work check: prod(full_shape) // (128 << 10) < threads → skip.
    // Use conservative upper-bound extents for symbolic range ends (vmax/const_factor)
    // so dynamic kernels don't underestimate work and collapse to tiny thread counts.
    let total_elements = estimate_total_elements(scheduler);

    // Tinygrad's descending thread count list
    const THREAD_LIST: [usize; 9] = [32, 16, 12, 8, 6, 5, 4, 3, 2];

    for &threads in &THREAD_LIST {
        if threads > max_threads {
            continue;
        }
        // Skip if not enough work per thread (Tinygrad: prod(full_shape) // (128 << 10) < threads)
        if total_elements / 131072 < threads as i64 {
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
            if rngs[axis_idx].divisible_by(threads).is_some() {
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

fn estimate_total_elements(scheduler: &Scheduler) -> i64 {
    let mut prod: i128 = 1;
    for rng in scheduler.rngs() {
        let extent = match rng.op() {
            Op::Range { end, .. } => {
                if let Op::Const(cv) = end.op()
                    && let morok_ir::ConstValue::Int(sz) = cv.0
                    && sz > 0
                {
                    sz
                } else if let Some(vmax) = end.vmax().try_int() {
                    vmax.max(1)
                } else {
                    let cf = end.const_factor();
                    if cf > 0 { cf } else { 1 }
                }
            }
            _ => 1,
        };
        prod = (prod.saturating_mul(extent as i128)).min(i64::MAX as i128);
    }
    prod.max(1) as i64
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
                        let rng_not_in_idx = !indices.iter().any(|idx| idx.backward_slice_ids().contains(&rng.id));
                        let all_upcast_in_idx = upcast_and_unroll_ranges
                            .iter()
                            .all(|r2| indices.iter().any(|idx| idx.backward_slice_ids().contains(&r2.id)));
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

/// Stride-ranked LOCAL workgroup configuration (Tinygrad heuristic.py:156-175).
///
/// Prioritizes expand axes (stride-0 in some buffer = broadcast) for LOCAL,
/// then higher axis indices. Tries sizes [32, 16, 8, 4, 3, 2] for axis 0
/// and [16, 8, 4, 3, 2] for others, with cumulative LOCAL size ≤ 128.
pub fn apply_local_dims(scheduler: &mut Scheduler, config: &HeuristicsConfig) -> bool {
    if !scheduler.renderer().has_local || config.disable_locals {
        return false;
    }

    // Rank axes: (has_expand_pattern, axis_index)
    // Tinygrad: prioritize expand axes (stride-0 in some buffer), then higher indices
    let eligible_axes = scheduler.axes_of(&[AxisType::Global, AxisType::Loop]);
    let full_shape = scheduler.full_shape();

    let mut local_axis_ranking: Vec<(bool, usize)> = Vec::new();
    for &axis in &eligible_axes {
        let rngs = scheduler.rngs();
        if axis >= rngs.len() {
            continue;
        }
        // Only CONST-end ranges (no symbolic dims)
        if let Op::Range { end, .. } = rngs[axis].op() {
            if !matches!(end.op(), Op::Const(..)) {
                continue;
            }
        } else {
            continue;
        }
        let is_expand = has_broadcast_pattern(scheduler, axis);
        local_axis_ranking.push((is_expand, axis));
    }

    // Sort descending by (is_expand, axis) — expand axes first, higher index first
    local_axis_ranking.sort_by(|a, b| b.cmp(a));

    // Collect LOCAL candidates with cumulative size constraint
    let mut to_local: Vec<(usize, usize)> = Vec::new();
    for &(_, axis) in &local_axis_ranking {
        let cumulative_local: usize = to_local.iter().map(|(_, sz)| *sz).product::<usize>().max(1);
        let axis_size = full_shape[axis];
        if axis_size <= 0 {
            continue;
        }

        // Tinygrad: axis 0 gets [32, 16, 8, 4, 3, 2], others get [16, 8, 4, 3, 2]
        let candidates: &[usize] = if axis == 0 { &[32, 16, 8, 4, 3, 2] } else { &[16, 8, 4, 3, 2] };

        let local_sz =
            candidates.iter().copied().find(|&x| (axis_size as usize).is_multiple_of(x) && cumulative_local * x <= 128);

        if let Some(sz) = local_sz {
            to_local.push((axis, sz));
        }
    }

    // Apply at most 3 LOCALs, sorted by axis (ascending)
    // Track deleted shapes: if local_sz == full_shape[axis], axis merges and shifts indices
    let mut to_apply: Vec<(usize, usize)> = to_local.into_iter().take(3).collect();
    to_apply.sort();

    let mut applied = false;
    let mut deleted_shape = 0usize;
    for (axis, local_sz) in to_apply {
        let adjusted_axis = axis - deleted_shape;
        let will_delete = local_sz == full_shape[axis] as usize;
        if apply_opt(scheduler, &Opt::local(adjusted_axis, local_sz), true).is_ok() {
            applied = true;
            if will_delete {
                deleted_shape += 1;
            }
        }
    }
    applied
}

/// Tensor core optimization for matmul patterns.
///
/// Matches Tinygrad's heuristic.py:28-46:
/// - Guard: skip when >1 reduce axis unless tc_opt >= 1
/// - Apply TC opts via tc::apply, capturing returned axes [N, M, K]
/// - Post-TC (non-AMX only): UPCAST M then N with [5,4,3,2], LOCAL N with [4,2]
pub fn try_tensor_cores(scheduler: &mut Scheduler, config: &HeuristicsConfig) -> bool {
    use crate::optimizer::config::TcUsage;
    use crate::optimizer::tc;

    if config.tc_enabled == TcUsage::Disabled {
        return false;
    }
    if scheduler.renderer().tensor_cores.is_empty() {
        return false;
    }
    if !has_matmul_pattern(scheduler) {
        return false;
    }

    // Guard: skip TC when >1 reduce axis and tc_opt < 1 (Bug 8)
    // Tinygrad: len(axes_of(GROUP_REDUCE, REDUCE)) == 1 or TC_OPT >= 1
    let reduce_count = scheduler.axes_of(&[AxisType::GroupReduce, AxisType::Reduce]).len();
    if reduce_count != 1 && config.tc_opt.as_usize() < 1 {
        return false;
    }

    let axis_choice_count = match tc::detect_matmul(scheduler) {
        Ok(Some(pattern)) => pattern.axis_choices.len(),
        _ => return false,
    };

    let mut rejections = Vec::new();

    for axis_choice in 0..axis_choice_count {
        // Clone the scheduler for trial - if this axis choice fails, no partial mutations.
        let mut trial = scheduler.clone();
        let tc_result = tc::apply_with_axis_choice(
            &mut trial,
            config.tc_select.as_i32(),
            config.tc_opt.as_usize(),
            config.tc_enabled.as_usize(),
            Some(axis_choice),
        );

        let axes = match tc_result {
            Ok(axes) => axes,
            Err(err) => {
                let err_msg = err.to_string();
                tracing::debug!(axis_choice, reason = %err_msg, "try_tensor_cores: axis choice rejected");
                rejections.push((axis_choice, err_msg));
                continue;
            }
        };

        // Record the TC opt with explicit axis choice.
        let opt = Opt::tc(
            Some(axis_choice),
            config.tc_select.as_i32(),
            config.tc_opt.as_usize(),
            config.tc_enabled.as_usize(),
        );
        trial.applied_opts.push(opt);

        // Post-TC optimizations (non-AMX only)
        // Tinygrad: if good_tc_opt and not AMX
        if !trial.renderer().is_amx() {
            // Track current N/M ranges (axes[0]=N, axes[1]=M, axes[2]=K)
            let mut tc_rngs = [axes[0].clone(), axes[1].clone()];

            // UPCAST M (dim=1) then N (dim=0) with factors [5,4,3,2]
            for tc_dim in [1usize, 0] {
                for &sz in &[5usize, 4, 3, 2] {
                    if tc_rngs[tc_dim].divisible_by(sz).is_some() {
                        // Find the range's position in the scheduler
                        if let Some(rng_idx) = trial.rngs().iter().position(|r| Arc::ptr_eq(r, &tc_rngs[tc_dim]))
                            && let Ok((replaced, _)) =
                                trial.shift_to(tc_rngs[tc_dim].clone(), sz, AxisType::Upcast, false, None)
                        {
                            trial.applied_opts.push(Opt::upcast(rng_idx, sz));
                            tc_rngs[tc_dim] = replaced;
                        }
                        break;
                    }
                }
            }

            // LOCAL N (dim=0) with factors [4,2]
            if trial.renderer().has_local {
                for &sz in &[4usize, 2] {
                    if tc_rngs[0].divisible_by(sz).is_some() {
                        if let Some(rng_idx) = trial.rngs().iter().position(|r| Arc::ptr_eq(r, &tc_rngs[0]))
                            && trial.shift_to(tc_rngs[0].clone(), sz, AxisType::Local, false, None).is_ok()
                        {
                            trial.applied_opts.push(Opt::local(rng_idx, sz));
                        }
                        break;
                    }
                }
            }
        }

        *scheduler = trial;
        return true;
    }

    tracing::debug!(?rejections, "try_tensor_cores: all axis choices rejected");
    false
}
