//! Linearization module for converting UOp DAGs to linear instruction sequences.
//!
//! This module implements priority-aware topological sorting for control flow,
//! primarily for future GPU/NPU backends that require linear instruction streams.
//!
//! # Architecture
//!
//! ```text
//! Kernel AST (Arc<UOp>)
//!     ↓
//! pm_split_ends                  → Split multi-range ENDs into nested single-range ENDs
//!     ↓
//! CFGContext::new(sink)         → Compute control flow edges
//!     ↓
//! linearize_with_cfg(sink)      → Priority-aware toposort with CFG edges
//!     ↓
//! Vec<Arc<UOp>>                  → Linear instruction sequence
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use morok_schedule::linearize::{linearize_with_cfg, linearize, CFGContext, pm_split_ends};
//!
//! // For backends that need proper control flow ordering:
//! let instructions = linearize_with_cfg(kernel_ast);
//!
//! // Or without CFG edges (simpler cases):
//! let instructions = linearize(kernel_ast);
//! ```
//!
//! # Control Flow Edges
//!
//! When sibling loops exist at the same nesting level, CFGContext computes
//! ordering edges to ensure proper linearization. These edges are passed
//! directly to the linearizer rather than modifying the UOp graph.

mod cfg_context;
#[allow(clippy::module_inception)]
mod linearize;

use std::sync::Arc;

use morok_ir::UOp;
use morok_ir::op::Op;
use morok_ir::pattern::TypedPatternMatcher;
use morok_ir::rewrite::graph_rewrite_bottom_up;
use smallvec::SmallVec;

pub use cfg_context::CFGContext;
pub use linearize::{linearize, linearize_with_edges};

/// Split multi-range ENDs into nested single-range ENDs.
///
/// Transforms `END(computation, [r1, r2, r3])` into `END(END(END(computation, r3), r2), r1)`
/// where ranges are sorted by axis_id in descending order (innermost first).
///
/// This is required for proper linearization - the linearizer expects single-range ENDs
/// to correctly order loop closures.
///
/// Based on Tinygrad's `pm_split_ends` (linearizer.py:93-100).
///
/// Note: The `ranges` field may contain non-RANGE ops (like CONST or Add expressions)
/// after optimization passes. We extract actual RANGE ops first, matching Tinygrad's
/// `.ranges` property behavior.
pub fn pm_split_ends() -> TypedPatternMatcher {
    crate::patterns! {
        // Match ALL END ops - split_end handles extraction and filtering
        End { computation, ranges } => |computation, ranges| {
            split_end(computation, ranges)
        },
    }
}

/// Split a multi-range END into nested single-range ENDs.
///
/// Extracts actual RANGE ops from the ranges field (which may contain arbitrary
/// expressions after optimization), then creates nested single-range ENDs.
fn split_end(computation: &Arc<UOp>, ranges: &SmallVec<[Arc<UOp>; 4]>) -> Option<Arc<UOp>> {
    // Step 1: Extract ONLY RANGE ops from the ranges field.
    // The ranges field may contain non-RANGE ops (CONST, Add, etc.) after optimization.
    // This matches Tinygrad's `UOp.sink(*e.src[1:]).ranges` which recursively extracts RANGEs.
    let sink = UOp::sink(ranges.iter().cloned().collect());
    let actual_ranges: Vec<Arc<UOp>> =
        sink.toposort().into_iter().filter(|node| matches!(node.op(), Op::Range { .. })).collect();

    // No actual RANGEs found - nothing to split
    if actual_ranges.is_empty() {
        return None;
    }

    // Single RANGE - create simple single-range END
    if actual_ranges.len() == 1 {
        let new_end = computation.end(SmallVec::from_elem(actual_ranges[0].clone(), 1));
        // Only return Some if this is different from the original
        if ranges.len() == 1 && ranges[0].id == actual_ranges[0].id {
            return None; // No change needed
        }
        return Some(new_end);
    }

    // Step 2: Sort RANGEs by axis_id descending (innermost first)
    // Note: Tinygrad sorts by full `x.arg` tuple (start, stop, axis_type) in reverse.
    // We sort by axis_id only, which is sufficient for most cases since axis_id
    // uniquely identifies the loop nesting level. Edge cases with duplicate axis_ids
    // would need full tuple sorting.
    let mut sorted_ranges = actual_ranges;
    sorted_ranges.sort_by(|a, b| {
        let a_id = match a.op() {
            Op::Range { axis_id, .. } => axis_id.value(),
            _ => unreachable!("filtered to RANGEs only"),
        };
        let b_id = match b.op() {
            Op::Range { axis_id, .. } => axis_id.value(),
            _ => unreachable!("filtered to RANGEs only"),
        };
        // Descending order: higher axis_id first (innermost)
        b_id.cmp(&a_id)
    });

    // Step 3: Wrap computation in nested single-range ENDs
    let mut result = computation.clone();
    for range in sorted_ranges {
        result = result.end(SmallVec::from_elem(range, 1));
    }

    Some(result)
}

/// Linearize a UOp DAG with proper control flow ordering.
///
/// This is the preferred entry point for linearization. It:
/// 1. Builds CFGContext to compute control flow edges
/// 2. Passes edges directly to the linearizer for dependency tracking
/// 3. Runs the priority-aware linearizer
///
/// # Example
///
/// ```ignore
/// use morok_schedule::linearize::linearize_with_cfg;
///
/// let instructions = linearize_with_cfg(kernel_ast);
/// ```
pub fn linearize_with_cfg(sink: Arc<UOp>) -> Vec<Arc<UOp>> {
    // Split multi-range ENDs into nested single-range ENDs.
    // Required for proper linearization ordering. (Tinygrad linearizer.py:93-100)
    let sink = graph_rewrite_bottom_up(&pm_split_ends(), sink, &mut ());

    let cfg = CFGContext::new(&sink);
    linearize_with_edges(sink, &cfg.edges)
}
