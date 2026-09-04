//! Dead loop elimination: `Range(end)` folds to `Const(0)` when `vmax(end) <= 0`.
//!
//! END/REDUCE empty-range folds were removed: they conflated the trivial-Range
//! `Const(0, Index)` with dead-Range markers, breaking `Range(Unroll, end=1)` inside
//! REDUCE/END. Downstream `reduce_to_acc` handles dead and empty ranges instead.

use smallvec::smallvec;
use std::sync::Arc;
use svod_dtype::DType;
use svod_ir::types::ConstValue;
use svod_ir::{AxisId, AxisType, Op, UOp};
use test_case::test_case;

use crate::rewrite::graph_rewrite;

use super::helpers::{assert_const_value, get_matcher};
use svod_ir::ops;

fn zero_trip() -> Arc<UOp> {
    UOp::native_const(0i32)
}

fn negative_trip() -> Arc<UOp> {
    UOp::native_const(-5i32)
}

fn clamped_to_zero() -> Arc<UOp> {
    UOp::native_const(-10i32).try_max(&UOp::native_const(0i32)).unwrap()
}

/// `size` is in [0, 5], so `size - 10` has `vmax == -5`.
fn symbolically_empty() -> Arc<UOp> {
    UOp::variable("size".into(), 0, 5, DType::Int32).try_sub(&UOp::native_const(10i32)).unwrap()
}

#[test_case(zero_trip(); "zero trip count")]
#[test_case(negative_trip(); "negative trip count")]
#[test_case(clamped_to_zero(); "vmax exactly zero")]
#[test_case(symbolically_empty(); "symbolic vmax below zero")]
fn dead_range_folds_to_zero(end: Arc<UOp>) {
    let dtype = end.dtype();
    let range = UOp::new(
        Op::Range(ops::Range { end, axis_id: AxisId::Renumbered(0), axis_type: AxisType::Loop, deps: smallvec![] }),
        dtype,
    );

    assert_const_value(&graph_rewrite(get_matcher(), range, &mut ()), ConstValue::Int(0));
}

#[test]
fn end_without_ranges_returns_the_computation() {
    let store = UOp::noop();
    assert!(Arc::ptr_eq(&Arc::clone(&store).end(smallvec![]), &store));
}
