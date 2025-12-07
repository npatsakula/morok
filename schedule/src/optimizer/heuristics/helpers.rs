//! Helper functions for pattern detection in heuristics.
//!
//! These functions analyze kernels to detect specific patterns (matmul, matvec, etc.)
//! and provide information for making optimization decisions.

use std::sync::Arc;

use morok_ir::{BinaryOp, Op, ReduceOp, TernaryOp};

use crate::optimizer::Scheduler;

/// Check if kernel has matrix multiplication pattern.
///
/// Detects the pattern: REDUCE(ADD) of MUL of two INDEX operations.
/// This is the typical pattern for matmul: C[i,j] = sum_k(A[i,k] * B[k,j])
///
/// Returns true if the pattern is found, false otherwise.
pub fn has_matmul_pattern(scheduler: &Scheduler) -> bool {
    // Check if there's a reduce operation
    let Some(reduceop) = scheduler.reduceop() else {
        return false;
    };

    // Check if it's a SUM reduction (ADD)
    if let Op::Reduce { src, reduce_op, .. } = reduceop.op() {
        if *reduce_op != ReduceOp::Add {
            return false;
        }

        // Check if the source is a MUL operation
        if let Op::Binary(BinaryOp::Mul, left, right) = src.op() {
            // Check if both operands are INDEX operations (possibly under CAST)
            let left_is_index = matches!(left.op(), Op::Index { .. })
                || matches!(left.op(), Op::Cast { src, .. } if matches!(src.op(), Op::Index { .. }));

            let right_is_index = matches!(right.op(), Op::Index { .. })
                || matches!(right.op(), Op::Cast { src, .. } if matches!(src.op(), Op::Index { .. }));

            return left_is_index && right_is_index;
        }
    }

    false
}

/// Check if an axis appears in WHERE gates (is masked).
///
/// An axis is considered masked if the corresponding range node appears
/// in the backward slice of any WHERE conditional operations.
///
/// This is used by masked upcast heuristic to identify dimensions with
/// conditional behavior that should be fully unrolled.
pub fn is_masked(scheduler: &Scheduler, axis: usize) -> bool {
    let rngs = scheduler.rngs();
    if axis >= rngs.len() {
        return false;
    }

    let target_rng = &rngs[axis];

    // Find all WHERE operations in the AST
    for node in scheduler.ast().toposort() {
        if let Op::Ternary(TernaryOp::Where, cond, _, _) = node.op() {
            // Check if target range appears in the condition's backward slice
            if cond.backward_slice().iter().any(|dep| Arc::ptr_eq(dep, target_rng)) {
                return true;
            }
        }
    }

    false
}

/// Check if an axis has broadcast pattern (stride-0 in some buffer).
///
/// An axis has broadcast pattern if it appears in the backward slice of
/// some buffer but not in the buffer's index expression. This means the
/// axis is being broadcast (same value repeated across that dimension).
///
/// Used by heuristic upcast to prefer broadcasting axes for vectorization.
pub fn has_broadcast_pattern(scheduler: &Scheduler, axis: usize) -> bool {
    let rngs = scheduler.rngs();
    if axis >= rngs.len() {
        return false;
    }

    let target_rng = &rngs[axis];

    for buf in scheduler.bufs() {
        // Check if range is in buffer's backward slice
        let in_backward_slice = buf.backward_slice().iter().any(|dep| Arc::ptr_eq(dep, target_rng));

        if !in_backward_slice {
            continue;
        }

        // Check if it's NOT in the index expression itself
        // If it's in backward slice but not in index, it's broadcast
        if let Op::Index { indices, .. } = buf.op() {
            let in_index =
                indices.iter().any(|idx| idx.backward_slice().iter().any(|dep| Arc::ptr_eq(dep, target_rng)));

            if !in_index {
                return true; // Found broadcast pattern
            }
        }
    }

    false
}

/// Count strides for an axis in buffer accesses.
///
/// Returns (num_buffers_using_axis, sum_of_strides).
///
/// Used by heuristic upcast ranking to prefer axes with simpler access patterns.
/// Lower stride count generally means better memory locality.
///
/// Note: This is a simplified version. Full stride analysis would need to:
/// - Parse index expressions to extract strides
/// - Handle symbolic strides
/// - Consider all RANGE nodes in the index
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
            // Check if this buffer uses the axis
            let uses_axis =
                indices.iter().any(|idx| idx.backward_slice().iter().any(|dep| Arc::ptr_eq(dep, target_rng)));

            if uses_axis {
                num_strides += 1;

                // Simplified stride counting: count occurrences in index
                // Full implementation would parse MUL operations to extract coefficients
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
