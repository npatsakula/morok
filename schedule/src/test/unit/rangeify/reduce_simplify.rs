//! `reduce_unparented` (drop ranges the source never reads, factor constants out
//! of a MUL chain) and `reduce_collapse` (lift a range-independent body out).

use std::sync::Arc;

use svod_dtype::DType;
use svod_ir::{AxisId, AxisType, BinaryOp, ConstValue, Op, ReduceOp, UOp, pattern::RewriteResult};
use test_case::test_case;

use crate::rangeify::indexing::{no_range, range_size_as_i64};
use crate::rangeify::transforms::reduce_collapse as reduce_collapse_inner;
use svod_ir::ops;

fn reduce_unparented(reduce: &Arc<UOp>) -> Option<Arc<UOp>> {
    match crate::rangeify::patterns::pm_reduce_simplify().rewrite(reduce, &mut ()) {
        RewriteResult::Rewritten(r) => Some(r),
        _ => None,
    }
}

fn reduce_collapse(reduce: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Reduce(ops::Reduce { src, ranges, .. }) = reduce.op() else { return None };
    reduce_collapse_inner(src, ranges)
}

fn reduce_range(end: i64, axis_id: usize) -> Arc<UOp> {
    UOp::range_axis(UOp::index_const(end), AxisId::Renumbered(axis_id), AxisType::Reduce)
}

fn has_reduce(uop: &Arc<UOp>) -> bool {
    uop.toposort().iter().any(|n| matches!(n.op(), Op::Reduce(..) | Op::ReduceAxis(..)))
}

fn has_range(uop: &Arc<UOp>) -> bool {
    uop.toposort().iter().any(|n| matches!(n.op(), Op::Range(..)))
}

// ===== reduce_unparented =====

/// A range the source never reads is folded into arithmetic on the source:
/// ADD scales it, MUL raises it to the extent, MAX is idempotent. Tinygrad
/// 8c8b43de handles exactly these three; MIN is deliberately absent.
#[test_case(ReduceOp::Add, Some(BinaryOp::Mul) ; "add becomes a multiply")]
#[test_case(ReduceOp::Mul, Some(BinaryOp::Pow) ; "mul becomes a power")]
#[test_case(ReduceOp::Max, None ; "max returns the source")]
fn an_unparented_range_is_folded_into_the_source(op: ReduceOp, expected: Option<BinaryOp>) {
    let src = UOp::native_const(5i32);
    let reduce = src.clone().reduce(vec![reduce_range(10, 0)].into(), op);

    let result = reduce_unparented(&reduce).expect("an unparented range must fold");
    match expected {
        Some(binary) => assert!(matches!(result.op(), Op::Binary(op, _, _) if *op == binary), "{}", result.tree()),
        None => assert!(Arc::ptr_eq(&result, &src)),
    }
}

#[test]
fn min_is_not_an_unparented_fold() {
    let reduce = UOp::native_const(42i32).reduce(vec![reduce_range(5, 0)].into(), ReduceOp::Min);
    assert!(reduce_unparented(&reduce).is_none());
}

#[test]
fn a_range_the_source_reads_is_not_unparented() {
    let range = reduce_range(10, 0);
    let reduce = Arc::clone(&range).reduce(vec![range].into(), ReduceOp::Add);
    assert!(reduce_unparented(&reduce).is_none());
}

/// Two unparented ranges fold one at a time, nesting the scale factors.
#[test]
fn every_unparented_range_folds() {
    let ranges = vec![reduce_range(3, 0), reduce_range(4, 1)];
    let reduce = UOp::native_const(5i32).reduce(ranges.into(), ReduceOp::Add);

    let result = reduce_unparented(&reduce).expect("both ranges must fold");
    let Op::Binary(BinaryOp::Mul, inner, _) = result.op() else { panic!("expected MUL, got {}", result.tree()) };
    assert!(matches!(inner.op(), Op::Binary(BinaryOp::Mul, _, _)), "{}", result.tree());
}

/// A mix keeps the parented range inside the REDUCE and scales by the other.
#[test]
fn a_parented_range_stays_inside_the_reduce() {
    let (parented, unparented) = (reduce_range(5, 0), reduce_range(10, 1));
    let src = UOp::native_const(3i32).try_add(&parented.cast(DType::Int32)).expect("add");
    let reduce = src.reduce(vec![parented.clone(), unparented].into(), ReduceOp::Add);

    let result = reduce_unparented(&reduce).expect("the unparented range must fold");
    let Op::Binary(BinaryOp::Mul, inner, _) = result.op() else { panic!("expected MUL, got {}", result.tree()) };
    let Op::Reduce(ops::Reduce { ranges, .. }) = inner.op() else {
        panic!("expected an inner REDUCE, got {}", result.tree())
    };
    assert_eq!(ranges.as_slice().len(), 1);
    assert!(Arc::ptr_eq(&ranges[0], &parented));
}

/// Constant factors in a MUL chain lift out of ADD unconditionally, and out of
/// MAX only when non-negative (a negative factor inverts the ordering).
#[test_case(ReduceOp::Add, 3, true ; "add with a positive factor")]
#[test_case(ReduceOp::Add, -1, true ; "add with a negative factor")]
#[test_case(ReduceOp::Max, 3, true ; "max with a positive factor")]
#[test_case(ReduceOp::Max, -1, false ; "max with a negative factor")]
fn constant_factors_lift_out_of_the_reduce(op: ReduceOp, factor: i64, lifts: bool) {
    let range = reduce_range(10, 0);
    let src = range.cast(DType::Int32).mul(&UOp::native_const(factor as i32));
    let reduce = src.reduce(vec![range].into(), op);

    let lifted = reduce_unparented(&reduce).is_some_and(|result| {
        matches!(result.op(), Op::Binary(BinaryOp::Mul, _, f)
            if matches!(f.op(), Op::Const(c) if c.0 == ConstValue::Int(factor)))
    });
    assert_eq!(lifted, lifts);
}

#[test]
fn several_constant_factors_lift_together() {
    let range = reduce_range(10, 0);
    let range_int = range.cast(DType::Int32);
    let src = UOp::native_const(2i32).mul(&range_int).mul(&UOp::native_const(5i32));
    let reduce = src.reduce(vec![range].into(), ReduceOp::Add);

    let result = reduce_unparented(&reduce).expect("constants must lift");
    assert!(matches!(result.op(), Op::Binary(BinaryOp::Mul, _, _)), "{}", result.tree());
}

// ===== reduce_collapse =====

/// A body that does not read the range collapses: neither the RANGE nor the
/// REDUCE survives, and the dtype is unchanged.
#[test_case(ReduceOp::Add ; "add")]
#[test_case(ReduceOp::Mul ; "mul")]
#[test_case(ReduceOp::Max ; "max")]
#[test_case(ReduceOp::Min ; "min")]
fn a_range_independent_body_collapses(op: ReduceOp) {
    let src = UOp::native_const(2.5f64);
    let reduce = src.clone().reduce(vec![reduce_range(100, 0)].into(), op);

    let result = reduce_collapse(&reduce).expect("a range-independent body must collapse");
    assert!(!has_range(&result));
    assert!(!has_reduce(&result));
    assert_eq!(result.dtype(), src.dtype());
}

#[test]
fn independent_ranges_all_collapse_together() {
    let ranges = vec![reduce_range(10, 0), reduce_range(20, 1)];
    let reduce = UOp::native_const(5i32).reduce(ranges.into(), ReduceOp::Add);

    let result = reduce_collapse(&reduce).expect("both ranges must collapse");
    assert!(no_range(&result));
}

/// Symbolic simplification runs first, so a body that only *looks* like it reads
/// the range collapses once the algebra cancels.
#[test_case(ReduceOp::Add, 0i32, BinaryOp::Add ; "x plus zero")]
#[test_case(ReduceOp::Mul, 1i32, BinaryOp::Mul ; "x times one")]
fn algebra_runs_before_the_collapse(op: ReduceOp, identity: i32, binary: BinaryOp) {
    let x = UOp::native_const(42i32);
    let src = match binary {
        BinaryOp::Add => x.try_add(&UOp::native_const(identity)),
        _ => x.try_mul(&UOp::native_const(identity)),
    }
    .expect("identity op");
    let reduce = src.reduce(vec![reduce_range(10, 0)].into(), op);

    let result = reduce_collapse(&reduce).expect("the identity must cancel and let the reduce collapse");
    assert!(!has_range(&result));
    assert!(!has_reduce(&result));
    assert!(
        !result.toposort().iter().any(|n| matches!(n.op(), Op::Binary(op, _, _) if *op == binary)),
        "the identity operand must be gone: {}",
        result.tree()
    );
}

#[test]
fn a_body_that_reads_the_range_does_not_collapse() {
    let range = reduce_range(10, 0);
    let src = range.cast(DType::Int32).try_add(&UOp::native_const(1i32)).expect("add");

    assert!(reduce_collapse(&src.reduce(vec![range].into(), ReduceOp::Add)).is_none());
}

#[test]
fn a_reduce_without_ranges_does_not_collapse() {
    assert!(reduce_collapse(&UOp::native_const(5i32).reduce(vec![].into(), ReduceOp::Add)).is_none());
}

/// The arange fold: `sum(r in [0, 32) of (r + v < 31 ? 0 : 1))` must collapse to
/// bound arithmetic with no RANGE left. ADD is commutative and morok's canonical
/// ordering puts the substituted scalar first whenever it sorts below the RANGE,
/// so the Lt lift has to fire for either operand order — tinygrad's UPat matches
/// commutative sources in both positions (`codegen/simplify.py:101`).
#[test_case(true ; "range on the left")]
#[test_case(false ; "range on the right")]
fn reduce_collapse_lifts_a_commutative_add_in_either_order(range_first: bool) {
    let range = reduce_range(32, 0);
    let scalar = UOp::variable("in0".into(), 0, 31, range.dtype());
    let sum = if range_first { range.try_add(&scalar) } else { scalar.try_add(&range) }
        .expect("range and scalar share a dtype");
    let bound = UOp::const_(sum.dtype(), ConstValue::Int(31));
    let gate = sum.try_cmplt(&bound).expect("comparison against a same-dtype bound");
    let body = UOp::try_where(gate, UOp::native_const(0i32), UOp::native_const(1i32)).expect("both branches are Int32");
    let reduce = body.reduce(vec![range].into(), ReduceOp::Add);

    let result = reduce_collapse(&reduce).expect("the arange fold must collapse this reduce");
    assert!(!has_range(&result), "reduce_collapse left a RANGE behind: {}", result.tree());
    assert!(!has_reduce(&result), "reduce_collapse left a REDUCE behind: {}", result.tree());
}

// ===== range_size_as_i64 =====

/// Only a RANGE with a constant extent has a size; `no_range` truth table rows
/// live in `range_load_guards.rs`.
#[test]
fn only_a_constant_range_reports_a_size() {
    assert_eq!(range_size_as_i64(&UOp::range_const(100, 0)), Some(100));
    assert_eq!(range_size_as_i64(&reduce_range(42, 1)), Some(42));

    let symbolic = UOp::range_axis(UOp::define_var("N".to_string(), 0, 1000), AxisId::Renumbered(0), AxisType::Loop);
    assert_eq!(range_size_as_i64(&symbolic), None);
    assert_eq!(range_size_as_i64(&UOp::native_const(100i32)), None);
}
