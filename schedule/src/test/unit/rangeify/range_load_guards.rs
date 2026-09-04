//! Parity tests for the cached RANGE/LOAD match guards.
//!
//! `no_range` / `no_load` used to run an uncached `any_in_subtree` DFS on every
//! pattern attempt. They now read the cached `RangesProperty` /
//! `has_index_in_sources` flags; these tests pin the truth table against the
//! original DFS implementations, kept here as oracles.

use std::sync::Arc;

use smallvec::smallvec;
use svod_device::DeviceSpec;
use svod_dtype::DType;
use svod_ir::{AxisId, AxisType, Op, ReduceOp, UOp};
use test_case::test_case;

use crate::rangeify::indexing::no_range;
use crate::rangeify::patterns::no_load;

fn no_range_oracle(u: &Arc<UOp>) -> bool {
    !u.any_in_subtree(|x| matches!(x.op(), Op::Range(..)))
}

fn no_load_oracle(u: &Arc<UOp>) -> bool {
    !u.any_in_subtree(|x| matches!(x.op(), Op::Index(..)))
}

fn range() -> Arc<UOp> {
    UOp::range_axis(UOp::index_const(10), AxisId::Renumbered(0), AxisType::Reduce)
}

fn load_at(idx: Arc<UOp>) -> Arc<UOp> {
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let index = UOp::index().buffer(buffer).indices(vec![idx]).call().expect("index");
    UOp::load().index(index).call()
}

fn g_const() -> Arc<UOp> {
    UOp::native_const(42i32)
}

fn g_const_arith() -> Arc<UOp> {
    UOp::native_const(10i32).try_add(&UOp::native_const(20i32)).expect("add")
}

fn g_range() -> Arc<UOp> {
    range()
}

fn g_range_arith() -> Arc<UOp> {
    range().cast(DType::Int32).try_add(&UOp::native_const(5i32)).expect("add")
}

/// REDUCE ends the range, so `in_scope_ranges` is empty here — but the RANGE is
/// still in the backward slice and the guard must keep reporting "has range".
fn g_reduce_over_range() -> Arc<UOp> {
    let r = range();
    r.cast(DType::Int32).reduce(smallvec![r], ReduceOp::Add)
}

fn g_load_const_index() -> Arc<UOp> {
    load_at(UOp::index_const(0))
}

fn g_load_range_index() -> Arc<UOp> {
    load_at(range())
}

fn g_where_over_load() -> Arc<UOp> {
    let cond = range().try_cmplt(&UOp::index_const(5)).expect("cmplt");
    let zero = UOp::native_const(0.0f32);
    UOp::try_where(cond, load_at(UOp::index_const(0)), zero).expect("where")
}

fn g_where_load_free() -> Arc<UOp> {
    let cond = UOp::index_const(1).try_cmplt(&UOp::index_const(5)).expect("cmplt");
    UOp::try_where(cond, UOp::native_const(1.0f32), UOp::native_const(0.0f32)).expect("where")
}

#[test_case(g_const, true, true; "bare const")]
#[test_case(g_const_arith, true, true; "const arithmetic")]
#[test_case(g_range, false, true; "bare range")]
#[test_case(g_range_arith, false, true; "range arithmetic")]
#[test_case(g_reduce_over_range, false, true; "range consumed by reduce")]
#[test_case(g_load_const_index, true, false; "load at const index")]
#[test_case(g_load_range_index, false, false; "load at range index")]
#[test_case(g_where_over_load, false, false; "load under a where")]
#[test_case(g_where_load_free, true, true; "range-free load-free where")]
fn cached_guards_match_dfs_oracle(build: fn() -> Arc<UOp>, expect_no_range: bool, expect_no_load: bool) {
    let u = build();
    assert_eq!(no_range_oracle(&u), expect_no_range, "oracle no_range");
    assert_eq!(no_load_oracle(&u), expect_no_load, "oracle no_load");
    assert_eq!(no_range(&u), no_range_oracle(&u), "cached no_range diverged from DFS oracle");
    assert_eq!(no_load(&u), no_load_oracle(&u), "cached no_load diverged from DFS oracle");
}
