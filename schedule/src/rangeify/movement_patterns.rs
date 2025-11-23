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

use std::mem::discriminant;
use std::rc::Rc;

use morok_ir::{Op, UOp};
use smallvec::SmallVec;

use crate::pattern::matcher::{PatternMatcher, RewriteFn};
use crate::pattern::upat::{OpFilter, SrcPattern, UPat};
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
/// let simplified = graph_rewrite(&pm, indexed);
/// // Result: buffer.index([r0, 0, r2]) - broadcast dimension becomes 0
/// ```
pub fn movement_op_patterns() -> PatternMatcher {
    let mut patterns: Vec<(UPat, RewriteFn)> = vec![];

    // Pattern: INDEX(movement_op) → movement_op.src[0].INDEX(apply_movement_op(indices))
    //
    // Matches: INDEX(buffer, indices...)
    //   where buffer is a movement op (RESHAPE, PERMUTE, EXPAND, PAD, SHRINK, FLIP)
    //
    // Transform: buffer.src[0].INDEX(transformed_indices)
    //   where transformed_indices = apply_movement_op(buffer.op, buffer.src[0].shape, indices)
    {
        // Create pattern that matches INDEX where buffer is bound to "mop"
        // Both the INDEX node ("idx") and its buffer ("mop") are bound
        let idx_pattern = UPat::Match {
            op: Some(vec![OpFilter::Discriminant(discriminant(&Op::Index {
                buffer: UOp::noop(),
                indices: SmallVec::new(),
                gate: None,
            }))]),
            dtype: None,
            src: Some(SrcPattern::Tuple(vec![UPat::var("mop")])),
            arg: None,
            name: Some("idx".into()),
        };

        pattern!(patterns, idx_pattern => |idx: &Rc<UOp>, mop: &Rc<UOp>| {
            // Only match if buffer is a movement operation
            if !mop.op().is_movement() {
                return None;
            }

            // Extract INDEX components
            let Op::Index { indices, gate, .. } = idx.op() else {
                return None;
            };

            // Get source buffer (first source of movement op)
            let src = &mop.op().sources()[0];

            // Get source shape using existing UOp::shape() method
            let src_shape = src.shape().ok()??;

            // Transform indices through movement op
            // Convert SmallVec to slice for apply_movement_op
            let transformed = apply_movement_op(mop.op(), src_shape, indices.as_slice());

            // Create new INDEX with transformed indices
            let result = match gate {
                Some(g) => UOp::index_gated(src.clone(), transformed, g.clone()),
                None => UOp::index(src.clone(), transformed),
            };

            result.ok()
        });
    }

    PatternMatcher::new(patterns)
}
