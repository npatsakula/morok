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
//! pm_add_control_flow (bpm)      → Embed CFG edges as deps on RANGE nodes
//!     ↓
//! linearize(sink)                → Priority-aware toposort
//!     ↓
//! pm_linearize_cleanups          → Inject IF/ENDIF for gated stores (line rewrite)
//!     ↓
//! Vec<Arc<UOp>>                  → Linear instruction sequence
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use svod_schedule::linearize::{linearize_with_cfg, linearize, CFGContext, pm_split_ends};
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
//! ordering edges to ensure proper linearization. These edges are embedded
//! as deps on RANGE nodes via `pm_add_control_flow` with `graph_rewrite_bottom_up`.

mod cfg_context;
#[allow(clippy::module_inception)]
mod linearize;

use std::collections::HashMap;
use std::sync::Arc;

use smallvec::{SmallVec, smallvec};
use svod_ir::op::Op;
use svod_ir::pattern::TypedPatternMatcher;
use svod_ir::rewrite::{graph_rewrite, graph_rewrite_bottom_up};
use svod_ir::{DType, UOp};

pub use cfg_context::CFGContext;
pub use linearize::linearize;
pub(crate) use linearize::{tinygrad_tuplize_cmp, tinygrad_weakint_expr};
use svod_ir::ops;

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
pub fn pm_split_ends() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        end @ End { computation, ranges } => |end, computation, ranges| {
            split_end(end, computation, ranges)
        },
    }
}

/// Split a multi-range END into nested single-range ENDs.
///
/// Tag preservation: the input END's tag (set by `reduce_to_acc` —
/// `TAG_MERGEABLE`) is restored on the outermost nested END so
/// downstream reduction merging can still find the result. Without
/// this, `UOp::end(...)` constructs a fresh END with no tag and the
/// merge pass becomes a no-op.
pub(crate) fn split_end_with_tag(
    original: &Arc<UOp>,
    computation: &Arc<UOp>,
    sources: &SmallVec<[Arc<UOp>; 4]>,
    tag: Option<smallvec::SmallVec<[usize; 2]>>,
) -> Option<Arc<UOp>> {
    let (backedges, targets): (Vec<Arc<UOp>>, Vec<Arc<UOp>>) =
        sources.iter().cloned().partition(|source| source.dtype() == DType::Void || source.dtype() == DType::Bool);
    let mut sorted_ranges = UOp::sink(targets).ranges().clone();

    // Matches Tinygrad's `sorted(..., key=lambda x: x.arg, reverse=True)` where
    // x.arg = (axis_id, axis_type, ...) -- tuple comparison gives lex ordering.
    sorted_ranges.sort_by(|a, b| {
        let (a_id, a_ty) = match a.op() {
            Op::Range(ops::Range { axis_id, axis_type, .. }) => (axis_id, axis_type.priority()),
            _ => unreachable!("filtered to RANGEs only"),
        };
        let (b_id, b_ty) = match b.op() {
            Op::Range(ops::Range { axis_id, axis_type, .. }) => (axis_id, axis_type.priority()),
            _ => unreachable!("filtered to RANGEs only"),
        };
        (b_id, b_ty).cmp(&(a_id, a_ty))
    });

    let mut result = computation.clone();
    for range in sorted_ranges {
        result = result.end(SmallVec::from_elem(range, 1));
    }
    result = result.end(backedges.into()).rtag(tag);

    (!Arc::ptr_eq(&result, original)).then_some(result)
}

fn split_end(original: &Arc<UOp>, computation: &Arc<UOp>, ranges: &SmallVec<[Arc<UOp>; 4]>) -> Option<Arc<UOp>> {
    split_end_with_tag(original, computation, ranges, original.tag().clone())
}

/// Pattern matcher for adding control flow dependencies to RANGE operations.
///
/// Matches Tinygrad's `pm_add_control_flow` (linearizer.py:89-91) which adds
/// CFG predecessors as extra sources to RANGE nodes via `x.replace(src=x.src+(y,))`.
///
/// In Svod, we add predecessors to the `deps` field of `Op::Range`, which makes
/// `InScopeRangesProperty` (via `children()`) naturally accumulate parent loop
/// ranges. This gives nested RANGE nodes a higher `run_count`, ensuring they
/// sort after operations that must appear outside them.
///
/// Used with `graph_rewrite_bottom_up` so patterns see original RANGE nodes
/// (matching `cfg.edges` keys), while the engine handles transitive rewrites
/// automatically — eliminating stale reference issues from manual substitution.
fn pm_add_control_flow() -> TypedPatternMatcher<CFGContext> {
    crate::patterns! {
        @context CFGContext;
        // Mirrors Tinygrad's: x.replace(src=x.src+(y,)) if (y:=ctx.edges.get(x)) is not None
        range @ Range { end: _, .. } => {
            let pred = ctx.get_predecessor(range)?;
            let mut srcs = range.op().sources().to_vec();
            srcs.push(pred.clone());
            Some(range.with_sources(srcs))
        },
    }
}

/// Linearize a UOp DAG with proper control flow ordering.
///
/// This is the preferred entry point for linearization. It:
/// 1. Splits multi-range ENDs into nested single-range ENDs
/// 2. Builds CFGContext to compute control flow edges
/// 3. Rewrites RANGE nodes to include CFG predecessors in their deps
/// 4. Runs the priority-aware linearizer
///
/// Matches Tinygrad's approach (linearizer.py:89-100):
/// ```python
/// sink = graph_rewrite(sink, pm_split_ends)
/// sink = graph_rewrite(sink, pm_add_control_flow, ctx=CFGContext(sink), bottom_up=True)
/// linearize(sink)
/// ```
pub fn add_control_flow(sink: Arc<UOp>) -> Arc<UOp> {
    let sink = graph_rewrite(pm_split_ends(), sink, &mut ());
    let mut cfg = CFGContext::new(&sink);
    graph_rewrite_bottom_up(&pm_add_control_flow(), sink, &mut cfg)
}

pub fn linearize_with_cfg(sink: Arc<UOp>) -> Vec<Arc<UOp>> {
    linearize(add_control_flow(sink))
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
/// STORE(index, value, gate) → IF(gate) + STORE(index, value) + ENDIF
/// ```
///
/// The address may be INDEX/SHRINK or a Cast-wrapped INDEX/SHRINK.
fn linearize_cleanup_pattern(uop: &Arc<UOp>, _replaced: &HashMap<u64, Arc<UOp>>) -> Option<(Arc<UOp>, Vec<Arc<UOp>>)> {
    if matches!(uop.op(), Op::If(..) | Op::EndIf(..)) {
        panic!("if not allowed in graph");
    }

    let Op::Store(ops::Store { index, value, gate: Some(gate) }) = uop.op() else {
        return None;
    };
    if gate.dtype() != svod_dtype::DType::Bool {
        return None;
    }

    let address_op = match index.op() {
        Op::Cast(ops::Cast { src, .. }) => src.op(),
        op => op,
    };
    if !matches!(address_op, Op::Index(..) | Op::Shrink(..)) {
        return None;
    }

    let ungated_store =
        UOp::new(Op::Store(ops::Store { index: index.clone(), value: value.clone(), gate: None }), uop.dtype());
    let if_op = UOp::if_(gate.clone(), smallvec![index.clone()]);
    let endif_op = UOp::endif(if_op.clone());

    Some((ungated_store.clone(), vec![if_op, ungated_store, endif_op]))
}

/// Line rewrite for injecting IF/ENDIF around gated stores.
///
/// Based on Tinygrad's `pm_linearize_cleanups` (codegen/__init__.py:107-113).
///
/// This operates on the linearized instruction list (not the DAG) to convert:
/// ```text
/// STORE(index, value, gate) → IF(gate) + STORE(index, value) + ENDIF
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
