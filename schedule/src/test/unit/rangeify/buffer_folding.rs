//! `buffer_folding`: noop STAGE removal and constant propagation through
//! STAGE / INDEX / COPY.

use std::sync::Arc;

use svod_dtype::{DType, DeviceSpec};
use svod_ir::{ConstValue, Op, UOp};
use test_case::test_case;

use crate::pattern::RewriteResult;
use crate::rangeify::patterns::buffer_folding;
use crate::rewrite::graph_rewrite;

fn fold(root: Arc<UOp>) -> Arc<UOp> {
    graph_rewrite(&buffer_folding(), root, &mut ())
}

fn range(end: i64, axis_id: usize) -> Arc<UOp> {
    UOp::range_const(end, axis_id)
}

/// `INDEX(STAGE(x, R), R) → x` — the buffer would be read back at exactly the
/// coordinates it was written at, so it is a noop.
#[test]
fn a_stage_read_at_its_own_ranges_folds_away() {
    let x = UOp::param(1, 10, DType::Float32, None);
    let r = range(10, 0);

    let staged = UOp::stage_global(Arc::clone(&x), vec![r.clone()]);
    let result = fold(UOp::index().buffer(staged).indices(vec![r]).call().expect("index"));

    assert!(Arc::ptr_eq(&result, &x));
}

/// With several ranges the fold still removes the STAGE, but the index has to be
/// relinearised, so the result is a view of `x` rather than `x` itself.
#[test]
fn a_multi_range_noop_stage_is_removed_but_reindexed() {
    let x = UOp::param(1, 200, DType::Float32, None);
    let ranges = vec![range(10, 0), range(20, 1)];

    let staged = UOp::stage_global(Arc::clone(&x), ranges.clone());
    let result = fold(UOp::index().buffer(staged).indices(ranges).call().expect("index"));

    assert!(!result.toposort().iter().any(|n| matches!(n.op(), Op::Stage(..))), "{}", result.tree());
    assert!(result.toposort().iter().any(|n| Arc::ptr_eq(n, &x)), "{}", result.tree());
}

#[test]
fn a_stage_read_at_other_ranges_is_kept() {
    let x = UOp::param(1, 1024, DType::Float32, None);
    let staged = UOp::stage_global(x, vec![range(10, 0)]);
    let indexed = UOp::index().buffer(staged).indices(vec![range(10, 1)]).call().expect("index");

    assert!(Arc::ptr_eq(&fold(indexed.clone()), &indexed));
}

/// The noop fold is structural — it does not care what the staged compute is.
#[test]
fn the_noop_fold_applies_to_arbitrary_compute() {
    let compute = UOp::var("x", DType::Float32, 0, 100).try_add(&UOp::var("y", DType::Float32, 0, 100)).expect("add");
    let r = range(10, 0);

    let staged = UOp::stage_global(Arc::clone(&compute), vec![r.clone()]);
    let result = fold(UOp::index().buffer(staged).indices(vec![r]).call().expect("index"));

    assert!(Arc::ptr_eq(&result, &compute));
}

fn staged(c: Arc<UOp>) -> Arc<UOp> {
    UOp::stage_global(c, vec![range(10, 0)])
}

fn indexed(c: Arc<UOp>) -> Arc<UOp> {
    UOp::index().buffer(c).indices(vec![range(10, 0), range(20, 1)]).call().expect("index")
}

fn copied(c: Arc<UOp>) -> Arc<UOp> {
    c.copy(DeviceSpec::Cuda { device_id: 0 })
}

fn staged_then_indexed(c: Arc<UOp>) -> Arc<UOp> {
    let r = range(15, 0);
    UOp::index().buffer(UOp::stage_global(c, vec![r.clone()])).indices(vec![r]).call().expect("index")
}

/// A constant has no storage to allocate, index into, or transfer: every wrapper
/// folds straight back to it.
#[test_case(super::staged, DType::Int32, ConstValue::Int(42) ; "stage of int")]
#[test_case(super::staged, DType::Bool, ConstValue::Bool(true) ; "stage of bool")]
#[test_case(super::staged, DType::Float32, ConstValue::Float(std::f64::consts::PI) ; "stage of float")]
#[test_case(super::indexed, DType::Float32, ConstValue::Float(2.5) ; "index of const")]
#[test_case(super::copied, DType::Int32, ConstValue::Int(99) ; "copy of const")]
#[test_case(super::staged_then_indexed, DType::Int32, ConstValue::Int(123) ; "index of stage of const")]
fn constants_fold_out_of_every_wrapper(wrap: fn(Arc<UOp>) -> Arc<UOp>, dtype: DType, value: ConstValue) {
    let c = UOp::const_(dtype, value);
    assert!(Arc::ptr_eq(&fold(wrap(Arc::clone(&c))), &c));
}

#[test]
fn a_copy_of_a_const_is_dropped_whatever_the_target_device() {
    let c = UOp::native_const(1.5f32);
    for device in [DeviceSpec::Cpu, DeviceSpec::Cuda { device_id: 0 }] {
        assert!(Arc::ptr_eq(&fold(c.copy(device)), &c));
    }
}

#[test]
fn buffer_folding_leaves_unrelated_nodes_alone() {
    let c = UOp::native_const(1.0f32);
    assert!(matches!(buffer_folding().rewrite(&c, &mut ()), RewriteResult::NoMatch));
    assert!(matches!(fold(c).op(), Op::Const(_)));
}
