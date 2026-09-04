//! `flatten_range`: canonicalise the RANGE *expressions* an END closes over.

use std::sync::Arc;

use smallvec::smallvec;
use svod_ir::{Op, UOp};
use test_case::test_case;

use crate::rangeify::transforms::{flatten_range_impl, flatten_ranges};
use svod_ir::ops;

fn range(end: i64, axis_id: usize) -> Arc<UOp> {
    UOp::range(UOp::index_const(end), axis_id)
}

fn nested_ends(depth: usize) -> Arc<UOp> {
    (0..depth).fold(UOp::native_const(1.0f32), |inner, i| inner.end(smallvec![range(10 * (i as i64 + 1), i)]))
}

/// Only the explicit ended-range sources are flattened — computation ENDs are
/// left nested, matching tinygrad. Returning `Some` for an unchanged END would
/// also spin the rewrite engine, so a single flat range yields `None` too.
#[test_case(nested_ends(1) ; "one end")]
#[test_case(nested_ends(2) ; "two nested ends")]
#[test_case(nested_ends(3) ; "three nested ends")]
#[test_case(UOp::native_const(1.0f32) ; "not an end at all")]
#[test_case(UOp::index_const(0).store(UOp::native_const(1.0f32)) ; "store without ranges")]
fn nothing_to_canonicalize_returns_none(root: Arc<UOp>) {
    assert!(flatten_range_impl(&root).is_none());
    assert!(Arc::ptr_eq(&flatten_ranges(&root), &root), "the graph walk is the identity too");
}

/// An END whose range source is an *expression* over ranges is rewritten to close
/// over the ranges themselves, keeping the computation untouched.
#[test]
fn a_range_expression_is_split_into_its_ranges() {
    let add = UOp::native_const(1.0f32).try_add(&UOp::native_const(2.0f32)).expect("add");
    let combined = range(10, 0).add(&range(20, 1));

    let flattened = flatten_range_impl(&add.clone().end(smallvec![combined])).expect("the expression must flatten");

    let Op::End(ops::End { computation, ranges }) = flattened.op() else {
        panic!("expected END, got {}", flattened.tree())
    };
    assert!(Arc::ptr_eq(computation, &add));
    assert_eq!(ranges.as_slice().len(), 2);
}
