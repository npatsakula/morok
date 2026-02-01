//! Linearization module for converting UOp DAGs to linear instruction sequences.
//!
//! This module implements priority-aware topological sorting for control flow,
//! primarily for future GPU/NPU backends that require linear instruction streams.
//!
//! # Architecture (Tinygrad-aligned)
//!
//! ```text
//! Kernel AST (Arc<UOp>)
//!     ↓
//! pm_split_ends                  → Split multi-range ENDs into nested single-range ENDs
//!     ↓
//! CFGContext::new(sink)          → Compute control flow edges
//!     ↓
//! linearize_with_edges(sink)     → Priority-aware toposort with CFG edge constraints
//!     ↓
//! pm_linearize_cleanups          → Inject IF/ENDIF for gated stores (line rewrite)
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
//! ordering edges to ensure proper linearization. These edges are passed to
//! linearize_with_edges which treats them as additional dependencies.

mod cfg_context;
#[allow(clippy::module_inception)]
mod linearize;

use std::collections::HashMap;
use std::sync::Arc;

use morok_ir::UOp;
use morok_ir::op::Op;
use morok_ir::pattern::TypedPatternMatcher;
use morok_ir::rewrite::graph_rewrite_bottom_up;
use smallvec::{SmallVec, smallvec};

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
/// 1. Splits multi-range ENDs into nested single-range ENDs
/// 2. Builds CFGContext to compute control flow edges
/// 3. Applies pm_add_control_flow to add edges to RANGE ops (BOTTOM_UP)
/// 4. Runs the priority-aware linearizer
///
/// # Example
///
/// ```ignore
/// use morok_schedule::linearize::linearize_with_cfg;
///
/// let instructions = linearize_with_cfg(kernel_ast);
/// ```
pub fn linearize_with_cfg(sink: Arc<UOp>) -> Vec<Arc<UOp>> {
    // Stage 19 (partial): Split multi-range ENDs into nested single-range ENDs.
    // Required for proper linearization ordering. (Tinygrad linearizer.py:93-100)
    let sink = graph_rewrite_bottom_up(&pm_split_ends(), sink, &mut ());

    // Stage 20-21: Build CFG context and linearize with edges
    // CFGContext computes control flow edges for sibling loops.
    // linearize_with_edges treats these edges as virtual consumer relationships,
    // ensuring proper ordering without modifying RANGE sources (which would panic).
    let cfg = CFGContext::new(&sink);
    linearize_with_edges(sink, &cfg.edges)
}

/// Line rewrite infrastructure for operating on linearized instruction lists.
///
/// Unlike DAG-based graph_rewrite, this operates on the linear instruction sequence
/// and can output multiple instructions for a single input instruction.
///
/// Based on Tinygrad's `line_rewrite` (linearizer.py).
///
/// # Arguments
///
/// * `lst` - The linearized instruction list
/// * `rewrite_fn` - Function that returns (replacement, outputs) for each UOp.
///   - `replacement`: The UOp to use in subsequent source substitutions
///   - `outputs`: The UOps to emit in the output list
fn line_rewrite<F>(lst: Vec<Arc<UOp>>, rewrite_fn: F) -> Vec<Arc<UOp>>
where
    F: Fn(&Arc<UOp>, &HashMap<u64, Arc<UOp>>) -> Option<(Arc<UOp>, Vec<Arc<UOp>>)>,
{
    let mut newlst = Vec::with_capacity(lst.len() * 2);
    let mut replaced: HashMap<u64, Arc<UOp>> = HashMap::new();

    for u in lst {
        let nu = replace_sources_from_map(&u, &replaced);
        let (replacement, outputs) = match rewrite_fn(&nu, &replaced) {
            Some((repl, outs)) => (repl, outs),
            None => (nu.clone(), vec![nu]),
        };
        replaced.insert(u.id, replacement);
        newlst.extend(outputs);
    }
    newlst
}

/// Replace sources of a UOp using a substitution map.
fn replace_sources_from_map(uop: &Arc<UOp>, replaced: &HashMap<u64, Arc<UOp>>) -> Arc<UOp> {
    let sources = uop.op().sources();
    if sources.is_empty() {
        return uop.clone();
    }

    let new_sources: Vec<Arc<UOp>> =
        sources.iter().map(|src| replaced.get(&src.id).cloned().unwrap_or_else(|| src.clone())).collect();

    if sources.iter().zip(&new_sources).all(|(old, new)| old.id == new.id) {
        return uop.clone();
    }
    uop.replace().src(new_sources).call()
}

/// Pattern for converting gated STORE to IF/STORE/ENDIF.
///
/// Based on Tinygrad's `pm_linearize_cleanups` (codegen/__init__.py:107-113).
///
/// Transforms:
/// ```text
/// STORE(INDEX(buf, idx, gate), value) → IF(gate) + STORE(INDEX(buf, idx), value) + ENDIF
/// ```
fn linearize_cleanup_pattern(uop: &Arc<UOp>, _replaced: &HashMap<u64, Arc<UOp>>) -> Option<(Arc<UOp>, Vec<Arc<UOp>>)> {
    // Panic if IF/ENDIF already present
    if matches!(uop.op(), Op::If { .. } | Op::EndIf { .. }) {
        panic!("IF/ENDIF not allowed in graph before line_rewrite_cleanups");
    }

    // Match STORE with gated INDEX
    let Op::Store { index, value, ranges } = uop.op() else {
        return None;
    };
    let Op::Index { buffer, indices, gate: Some(gate) } = index.op() else {
        return None;
    };

    // Create ungated INDEX and STORE
    let ungated_index = UOp::index().buffer(buffer.clone()).indices(indices.clone()).call().expect("ungated INDEX");
    let ungated_store = ungated_index.store_with_ranges(value.clone(), ranges.clone());

    // Wrap in IF/ENDIF
    let if_op = UOp::if_(gate.clone(), smallvec![ungated_index.clone()]);
    let endif_op = UOp::endif(if_op.clone());

    Some((ungated_store.clone(), vec![if_op, ungated_store, endif_op]))
}

/// Line rewrite for injecting IF/ENDIF around gated stores.
///
/// Based on Tinygrad's `pm_linearize_cleanups` (codegen/__init__.py:107-113).
///
/// This operates on the linearized instruction list (not the DAG) to convert:
/// ```text
/// STORE(INDEX(buf, idx, gate), value) → IF(gate) + STORE(INDEX(buf, idx), value) + ENDIF
/// ```
///
/// Only needed for backends that don't support gated stores natively.
/// LLVM, CUDA, and Metal support predicated stores, so this may be a no-op for them.
///
/// # Arguments
///
/// * `lst` - The linearized instruction list
///
/// # Returns
///
/// Modified instruction list with IF/ENDIF injected around gated stores.
pub fn line_rewrite_cleanups(lst: Vec<Arc<UOp>>) -> Vec<Arc<UOp>> {
    line_rewrite(lst, linearize_cleanup_pattern)
}
