//! Hand-coded optimization heuristics for kernel optimization.
//!
//! Implements Tinygrad-style heuristics for reasonable performance without auto-tuning.
//! Applies optimizations in order: TC → Image → GroupReduce → Upcasts → Unroll → Local → Thread.

use std::sync::Arc;

use morok_ir::{AxisType, BinaryOp, Op, ReduceOp, TernaryOp};

use crate::optimizer::config::HeuristicsConfig;
use crate::optimizer::{Opt, Scheduler, apply_opt};

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
    // 1. Tensor cores (skip other opts if applied)
    if try_tensor_cores(scheduler, config) {
        apply_local_dims(scheduler, config);
        return;
    }

    // 2. Image upcasts
    apply_image_upcasts(scheduler);

    // 3. Grouped reduction
    try_grouped_reduction(scheduler, config);

    // 4. Masked upcasts
    apply_masked_upcasts(scheduler);

    // 5. Heuristic upcasts
    apply_heuristic_upcasts(scheduler);

    // 6. Unroll
    apply_unroll(scheduler, config);

    // 7. Default upcast
    if !scheduler.upcasted() {
        apply_default_upcast(scheduler);
    }

    // 8. Local dims
    apply_local_dims(scheduler, config);

    // 9. Threading
    apply_threading(scheduler);
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
    if scheduler.upcasted() {
        return false;
    }
    let upcastable = scheduler.upcastable_dims();
    if upcastable.is_empty() {
        return false;
    }
    apply_opt(scheduler, &Opt::upcast(upcastable[0], 4), true).is_ok()
}

/// Unroll small reduction loops (size <= threshold).
pub fn apply_unroll(scheduler: &mut Scheduler, config: &HeuristicsConfig) -> bool {
    let mut applied = false;
    let unrollable = scheduler.unrollable_dims();
    let threshold = config.unroll_threshold as i64;

    for axis_idx in unrollable {
        let rngs = scheduler.rngs();
        if axis_idx >= rngs.len() {
            continue;
        }
        let rng = &rngs[axis_idx];
        if let Op::Range { end, .. } = rng.op()
            && let Op::Const(cv) = end.op()
            && let morok_ir::ConstValue::Int(size) = cv.0
            && size > 1
            && size <= threshold
            && false
        // TEMPORARY: disable until codegen supports Unroll
        {
            let unroll_axes = scheduler.unrollable_dims();
            if let Some(logical) = unroll_axes.iter().position(|&a| a == axis_idx)
                && apply_opt(scheduler, &Opt::unroll(logical, size as usize), true).is_ok()
            {
                applied = true;
            }
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

/// CPU threading for outer global axes.
pub fn apply_threading(scheduler: &mut Scheduler) -> bool {
    if !scheduler.renderer().has_threads {
        return false;
    }
    let global_axes = scheduler.axes_of(&[AxisType::Global]);
    if global_axes.is_empty() {
        return false;
    }

    let axis_idx = global_axes[0];
    let rngs = scheduler.rngs();
    if axis_idx >= rngs.len() {
        return false;
    }
    let rng = &rngs[axis_idx];
    if let Op::Range { end, .. } = rng.op()
        && let Op::Const(cv) = end.op()
        && let morok_ir::ConstValue::Int(size) = cv.0
    {
        let thread_count = (size as usize).min(8);
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
pub fn apply_heuristic_upcasts(scheduler: &mut Scheduler) -> bool {
    let mut applied = false;
    let upcastable = scheduler.upcastable_dims();
    if upcastable.is_empty() {
        return false;
    }

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
    let target_upcast = 8;

    for (axis_idx, _) in ranked_axes {
        if upcast_product >= target_upcast {
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
                if size >= factor as i64 && apply_opt(scheduler, &Opt::upcast(axis_idx, factor), true).is_ok() {
                    upcast_product *= factor;
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
