//! Regression tests for END merging across reduce-nesting contexts.
//!
//! Covers `merge_sibling_ends` / `ReduceContext::merge_reduce_ends`:
//!
//! - Only ENDs tagged `TAG_MERGEABLE` participate in the merge step.
//! - Tagged ENDs sharing reduce ranges *and* enclosing context merge into
//!   one `GROUP.end`.
//! - Tagged ENDs sharing reduce ranges but living at *different nesting
//!   depths* (different `in_scope_ranges`) are kept apart — the second
//!   context-group's RANGEs are cloned with fresh axis ids so each RANGE is
//!   reachable from at most one merged END.

use std::collections::HashSet;
use std::sync::Arc;

use morok_dtype::DType;
use morok_ir::{Op, UOp};
use smallvec::smallvec;

use super::helpers::*;
use crate::devectorize::{TAG_MERGEABLE, merge_sibling_ends};

/// Count distinct (deduplicated) nodes in `uop`'s subgraph that match
/// `pred`. `count_ops` walks every path without dedup, so a node reachable
/// via two SINK sources is counted twice — for structural assertions about
/// merged graphs we want the toposort-based count instead.
fn distinct_count<F>(uop: &Arc<UOp>, pred: F) -> usize
where
    F: Fn(&Arc<UOp>) -> bool,
{
    let mut seen: HashSet<u64> = HashSet::new();
    for n in uop.toposort() {
        if pred(&n) {
            seen.insert(n.id);
        }
    }
    seen.len()
}

/// Build `INDEX(buf, [0]).store(val).end(ranges).with_tag([TAG_MERGEABLE])`.
fn build_tagged_end(buf: Arc<UOp>, val: Arc<UOp>, ranges: smallvec::SmallVec<[Arc<UOp>; 4]>) -> Arc<UOp> {
    let zero = UOp::index_const(0);
    let idx = UOp::index().buffer(buf).indices(vec![zero]).call().expect("index");
    idx.store_value(val).end(ranges).with_tag(smallvec![TAG_MERGEABLE])
}

#[test]
fn merge_same_context_produces_single_group() {
    // Two tagged ENDs share R0 with no other ranges in scope.
    // They must merge into a single `GROUP(...).end([R0])`.
    let r0 = create_range_reduce(8, 0);
    let acc_a = UOp::define_reg_typed(1, DType::Float32);
    let acc_b = UOp::define_reg_typed(1, DType::Float32);

    let end_a = build_tagged_end(acc_a, create_float_const(1.0), smallvec![r0.clone()]);
    let end_b = build_tagged_end(acc_b, create_float_const(2.0), smallvec![r0]);

    let sink = UOp::sink(vec![end_a, end_b]);
    let merged = merge_sibling_ends(&sink);

    let distinct_ends = distinct_count(&merged, |u| matches!(u.op(), Op::End { .. }));
    assert_eq!(distinct_ends, 1, "same-context tagged ENDs should merge into one GROUP.end");
    let distinct_groups = distinct_count(&merged, |u| matches!(u.op(), Op::Group { .. }));
    assert_eq!(distinct_groups, 1, "merged ENDs should be wrapped in a single GROUP");
}

#[test]
fn merge_skips_untagged_ends() {
    // Tagged + untagged END with identical reduce range. The untagged one is
    // invisible to the merge step; with only one tagged END there is no
    // merge candidate, so both survive untouched.
    let r0 = create_range_reduce(8, 0);
    let acc_tagged = UOp::define_reg_typed(1, DType::Float32);
    let acc_untagged = UOp::define_reg_typed(1, DType::Float32);

    let zero = UOp::index_const(0);
    let idx_tagged = UOp::index().buffer(acc_tagged).indices(vec![zero.clone()]).call().expect("index");
    let idx_untagged = UOp::index().buffer(acc_untagged).indices(vec![zero]).call().expect("index");

    let end_tagged =
        idx_tagged.store_value(create_float_const(1.0)).end(smallvec![r0.clone()]).with_tag(smallvec![TAG_MERGEABLE]);
    // Note: deliberately NOT tagged — must be ignored by the merge step.
    let end_untagged = idx_untagged.store_value(create_float_const(2.0)).end(smallvec![r0]);

    let sink = UOp::sink(vec![end_tagged, end_untagged]);
    let merged = merge_sibling_ends(&sink);

    let distinct_ends = distinct_count(&merged, |u| matches!(u.op(), Op::End { .. }));
    assert_eq!(distinct_ends, 2, "untagged END must be ignored by the merge step");
    let distinct_groups = distinct_count(&merged, |u| matches!(u.op(), Op::Group { .. }));
    assert_eq!(distinct_groups, 0, "no GROUP should be inserted when only one tagged END is visible");
}

#[test]
fn merge_splits_across_nesting_contexts() {
    // Two tagged ENDs share R0 but live at different nesting depths:
    //   end_outer: computation has no other RANGE in scope.
    //   end_inner: computation has R_outer in scope (via `acc.after([R_outer])`).
    //
    // Expected: NOT merged into one GROUP. The inner context-group's RANGE
    // is cloned with a fresh axis id.
    let r0 = create_range_reduce(8, 0);
    let r_outer = create_range_loop(4, 100);
    let zero = UOp::index_const(0);

    let acc_outer = UOp::define_reg_typed(1, DType::Float32);
    let acc_inner_base = UOp::define_reg_typed(1, DType::Float32);
    let acc_inner = acc_inner_base.after(smallvec![r_outer.clone()]);

    let idx_outer = UOp::index().buffer(acc_outer).indices(vec![zero.clone()]).call().expect("index");
    let idx_inner = UOp::index().buffer(acc_inner).indices(vec![zero]).call().expect("index");

    let end_outer =
        idx_outer.store_value(create_float_const(1.0)).end(smallvec![r0.clone()]).with_tag(smallvec![TAG_MERGEABLE]);
    let end_inner =
        idx_inner.store_value(create_float_const(2.0)).end(smallvec![r0.clone()]).with_tag(smallvec![TAG_MERGEABLE]);

    // Sanity-check the test setup: the two ENDs must have different
    // `in_scope_ranges`, otherwise the test would degenerate to the
    // same-context case.
    let outer_ctx: std::collections::HashSet<u64> = end_outer.in_scope_ranges().iter().map(|k| k.0.id).collect();
    let inner_ctx: std::collections::HashSet<u64> = end_inner.in_scope_ranges().iter().map(|k| k.0.id).collect();
    assert_ne!(outer_ctx, inner_ctx, "test setup: ENDs must have different in_scope_ranges");

    let sink = UOp::sink(vec![end_outer, end_inner]);
    let pre_distinct_ranges = distinct_count(&sink, |u| matches!(u.op(), Op::Range { .. }));
    let merged = merge_sibling_ends(&sink);

    let distinct_ends = distinct_count(&merged, |u| matches!(u.op(), Op::End { .. }));
    assert_eq!(distinct_ends, 2, "tagged ENDs at different nesting depths must NOT cross-merge");
    let distinct_groups = distinct_count(&merged, |u| matches!(u.op(), Op::Group { .. }));
    assert_eq!(distinct_groups, 0, "no GROUP should wrap ENDs from disjoint nesting contexts");
    let post_distinct_ranges = distinct_count(&merged, |u| matches!(u.op(), Op::Range { .. }));
    assert!(
        post_distinct_ranges > pre_distinct_ranges,
        "expected a cloned RANGE to appear in the merged graph (pre={pre_distinct_ranges}, post={post_distinct_ranges})"
    );
}
