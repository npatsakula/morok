//! Pattern matcher for pushing movement operations through INDEX.
//!
//! This module implements the `pm_mops` pattern matcher from Tinygrad's rangeify.py.
//! The key transformation is:
//!
//! ```ignore
//! MovementOp(src).INDEX(ranges) → src.INDEX(apply_movement_op(ranges))
//! ```
//!
//! This transformation is critical for `detect_expanded_dimensions()` in split_reduceop,
//! where we need to determine which dimensions are broadcast (expanded) by analyzing
//! which RANGE operations survive after substituting the base buffer with NOOP.
//!
//! Without this transformation, movement ops remain as structural nodes and incorrectly
//! report that all ranges are present, leading to wrong expanded dimension detection.
//!
//! ## Example
//!
//! ```ignore
//! // Input: RESHAPE(buffer, [10, 1, 20]).EXPAND([10, 5, 20]).INDEX([r0, r1, r2])
//! // Pattern matches and transforms to:
//! // buffer.INDEX([r0, 0, r2])  // r1 becomes 0 due to broadcast!
//! ```
//!
//! Based on Tinygrad's pm_mops (tinygrad/schedule/rangeify.py:18-25).

use std::rc::Rc;

use morok_ir::UOp;

use crate::pattern::matcher::PatternMatcher;
use crate::rangeify::helpers::apply_movement_op;

/// Create pattern matcher for pushing movement ops through INDEX operations.
///
/// This implements Tinygrad's `pm_mops` which transforms:
/// ```ignore
/// MovementOp(src).INDEX(ranges) → src.INDEX(apply_movement_op(op, src.shape, ranges))
/// ```
///
/// The transformation is essential for correct expanded dimension detection in split_reduceop.
///
/// # Returns
///
/// A PatternMatcher with movement operation fusion patterns.
///
/// # Example
///
/// ```ignore
/// use morok_schedule::rangeify::movement_patterns::movement_op_patterns;
/// use morok_schedule::rewrite::graph_rewrite;
///
/// let pm = movement_op_patterns();
/// let indexed = buffer.reshape([10, 1, 20]).expand([10, 5, 20]).index([r0, r1, r2])?;
/// let simplified = graph_rewrite(&pm, indexed, &mut ());
/// // Result: buffer.index([r0, 0, r2]) - broadcast dimension becomes 0
/// ```
pub fn movement_op_patterns() -> PatternMatcher {
    // Pattern 1: INDEX(movement_op) → movement_op.src[0].INDEX(apply_movement_op(indices))
    //
    // Matches: INDEX(buffer, indices...)
    //   where buffer is a movement op (RESHAPE, PERMUTE, EXPAND, PAD, SHRINK, FLIP)
    //
    // Transform: buffer.src[0].INDEX(transformed_indices)
    //   where transformed_indices = apply_movement_op(buffer.op, buffer.src[0].shape, indices)
    //
    // Pattern 2: INDEX(INDEX(buffer, idx1), idx2) → INDEX(buffer, idx1) when idx1 == idx2
    //
    // This handles the case where nested INDEX operations use the same index,
    // which happens when RESHAPE is a no-op and we've already indexed once.
    crate::patterns! {
        Index { buffer: mop, indices, gate } if mop.op().is_movement() => {
            transform_movement_through_index(mop, &indices, &gate)
        }
        Index { buffer: inner_idx, indices, gate: None }
            if matches!(inner_idx.op(), morok_ir::Op::Index { .. }) => {
            flatten_nested_index(inner_idx, &indices)
        }
    }
}

/// Transform a movement op through INDEX by applying the movement to indices.
fn transform_movement_through_index(
    mop: &Rc<UOp>,
    indices: &smallvec::SmallVec<[Rc<UOp>; 4]>,
    gate: &Option<Rc<UOp>>,
) -> Option<Rc<UOp>> {
    // Get source buffer (first source of movement op)
    let src = &mop.op().sources()[0];

    // Get source shape using existing UOp::shape() method
    let src_shape = src.shape().ok()??;

    // Transform indices through movement op
    let transformed = apply_movement_op(mop.op(), src_shape, indices.as_slice());

    // Create new INDEX with transformed indices
    let result = match gate {
        Some(g) => UOp::index_gated(src.clone(), transformed, g.clone()),
        None => UOp::index(src.clone(), transformed),
    };

    result.ok()
}

/// Flatten nested INDEX operations: INDEX(INDEX(buffer, idx1), idx2) → INDEX(buffer, idx1)
///
/// When both INDEX operations use the same single index (common after no-op RESHAPE),
/// we can just use the inner INDEX directly.
fn flatten_nested_index(
    inner_idx: &Rc<UOp>,
    outer_indices: &smallvec::SmallVec<[Rc<UOp>; 4]>,
) -> Option<Rc<UOp>> {
    if let morok_ir::Op::Index { buffer: _inner_buffer, indices: inner_indices, gate: None } = inner_idx.op() {
        // If both use the same single index, just return the inner INDEX
        if inner_indices.len() == 1 && outer_indices.len() == 1 {
            if inner_indices[0].id == outer_indices[0].id {
                // Same index - use inner INDEX directly
                return Some(inner_idx.clone());
            }
        }
    }
    None
}
