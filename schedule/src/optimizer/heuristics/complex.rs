//! Complex optimization heuristics.
//!
//! These heuristics require sophisticated analysis and decision-making:
//! - Heuristic upcasts: Stride-based ranking and iterative upcast selection
//! - Local dims: GPU workgroup dimension configuration
//! - Tensor cores: Hardware-accelerated matmul detection and optimization

use crate::optimizer::{Opt, Scheduler, apply_opt};

use super::config::*;
use super::helpers::{count_strides, has_broadcast_pattern, has_matmul_pattern};

/// Apply sophisticated upcast heuristics based on stride analysis.
///
/// This is the most complex upcast strategy. It ranks all upcastable axes
/// by analyzing their memory access patterns:
///
/// 1. Prefers axes with broadcast patterns (stride-0, same value repeated)
/// 2. Ranks by stride complexity (fewer strides = simpler access)
/// 3. Iteratively tries upcast factors (2, 3, 4) for top-ranked axes
/// 4. Stops when total upcast product reaches target (typically 8)
///
/// # Returns
///
/// True if any heuristic upcasts were applied, false otherwise.
pub fn apply_heuristic_upcasts(scheduler: &mut Scheduler) -> bool {
    let mut applied = false;

    // Get all upcastable axes
    let upcastable = scheduler.upcastable_dims();
    if upcastable.is_empty() {
        return false;
    }

    // Rank axes by desirability for upcasting
    let mut ranked_axes: Vec<(usize, i32)> = upcastable
        .iter()
        .map(|&axis_idx| {
            let mut score = 0;

            // Strongly prefer broadcast axes (stride-0)
            if has_broadcast_pattern(scheduler, axis_idx) {
                score += 1000;
            }

            // Prefer axes with simpler stride patterns
            let (num_strides, sum_strides) = count_strides(scheduler, axis_idx);

            // Fewer buffers using this axis = simpler (better)
            score -= num_strides as i32 * 10;

            // Lower total stride count = better locality
            score -= sum_strides as i32;

            (axis_idx, score)
        })
        .collect();

    // Sort by score (higher is better)
    ranked_axes.sort_by(|a, b| b.1.cmp(&a.1));

    // Try to upcast top-ranked axes
    let mut upcast_product = 1;
    let target_upcast = 8; // Typical SIMD target

    for (axis_idx, _score) in ranked_axes {
        if upcast_product >= target_upcast {
            break;
        }

        // Get axis size
        let rngs = scheduler.rngs();
        if axis_idx >= rngs.len() {
            continue;
        }

        let rng = &rngs[axis_idx];
        if let morok_ir::Op::Range { end, .. } = rng.op()
            && let morok_ir::Op::Const(cv) = end.op()
            && let morok_ir::ConstValue::Int(size) = cv.0
        {
            // Try upcast factors in order of preference
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
                // Check if axis is large enough for this factor
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

/// Apply GPU workgroup (local memory) dimension configuration.
///
/// Configures local memory dimensions for optimal shared memory usage on GPUs.
/// This determines the workgroup size and layout (1D, 2D, or 3D) based on:
/// - Kernel shape
/// - Backend local_max constraint
/// - Memory access patterns
///
/// Typical configurations:
/// - 1D: [256] or [512] for simple reductions
/// - 2D: [16, 16] or [32, 8] for matmul-like patterns
/// - 3D: [8, 8, 8] for 3D convolutions
///
/// # Returns
///
/// True if local dimensions were configured, false otherwise.
pub fn apply_local_dims(scheduler: &mut Scheduler) -> bool {
    // Only apply to GPU backends with local memory
    if !scheduler.renderer().has_local {
        return false;
    }

    // Get global axes that can be localized
    let global_axes = scheduler.axes_of(&[morok_ir::AxisType::Global]);
    if global_axes.is_empty() {
        return false;
    }

    // Get local_max constraint (max threads per workgroup)
    let local_max = scheduler.renderer().local_max.unwrap_or(1024);

    // Strategy: Use 2D workgroup layout for 2D+ kernels, 1D otherwise
    let output_shape = scheduler.output_shape();

    if output_shape.len() >= 2 {
        // 2D layout: Try [16, 16] = 256 threads
        if global_axes.len() >= 2 && local_max >= 256 {
            let axis0 = global_axes[0];
            let axis1 = global_axes[1];

            // Apply local to both axes
            if apply_opt(scheduler, &Opt::local(axis0, 16), true).is_ok() {
                let _ = apply_opt(scheduler, &Opt::local(axis1, 16), true);
                return true;
            }
        }
    }

    // Fallback: 1D layout on first axis
    if !global_axes.is_empty() {
        let axis0 = global_axes[0];
        let local_size = local_max.min(256); // Conservative 1D size

        if apply_opt(scheduler, &Opt::local(axis0, local_size), true).is_ok() {
            return true;
        }
    }

    false
}

/// Try applying tensor core optimization for matmul patterns.
///
/// Detects matrix multiplication patterns and applies tensor core acceleration
/// if available on the backend. Tensor cores provide significant speedup (often
/// 8-16x) for matmul operations on modern GPUs (NVIDIA Tensor Cores, AMD Matrix Cores).
///
/// Requirements:
/// - Backend has tensor cores available (renderer.tensor_cores non-empty)
/// - Kernel matches matmul pattern: REDUCE(ADD) of MUL of INDEX operations
/// - Tensor core configuration enabled (USE_TC > 0)
/// - Matrix dimensions compatible with tensor core sizes
///
/// # Returns
///
/// True if tensor cores were applied, false otherwise.
pub fn try_tensor_cores(scheduler: &mut Scheduler) -> bool {
    // Check if tensor cores are enabled
    if USE_TC == 0 {
        return false;
    }

    // Check if backend supports tensor cores
    if scheduler.renderer().tensor_cores.is_empty() {
        return false;
    }

    // Check if kernel has matmul pattern
    if !has_matmul_pattern(scheduler) {
        return false;
    }

    // Apply tensor core optimization
    // TC_SELECT: -1 = auto-select, >= 0 = use specific TC config
    // TC_OPT: optimization level (0=strict, 1=relaxed, 2=padded)
    let opt = Opt::tc(None, TC_SELECT, TC_OPT, USE_TC);

    if apply_opt(scheduler, &opt, true).is_ok() {
        return true;
    }

    false
}
