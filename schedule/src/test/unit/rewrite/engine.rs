use crate::{PatternMatcher, graph_rewrite, pattern::UPat};
use morok_dtype::DType;
use morok_ir::{BinaryOp, ConstValue, Op, UOp, pattern};
use std::sync::Arc;

fn const_uop(val: i64) -> Arc<UOp> {
    UOp::const_(DType::Int32, ConstValue::Int(val))
}

fn binary_uop(op: BinaryOp, a: Arc<UOp>, b: Arc<UOp>) -> Arc<UOp> {
    let dtype = a.dtype();
    UOp::new(Op::Binary(op, a, b), dtype)
}

#[test]
fn test_simple_rewrite() {
    // Pattern: x + 0 -> x
    let mut patterns = vec![];

    pattern!(patterns,
        UPat::binary(vec![BinaryOp::Add], vec![UPat::var("x"), UPat::cvar("zero")]) => |x: &Arc<UOp>, zero: &Arc<UOp>| {
            if let Op::Const(cv) = zero.op()
                && cv.0 == ConstValue::Int(0) {
                    return Some(Arc::clone(x));
                }
            None
        }
    );

    let matcher = PatternMatcher::new(patterns);

    // Test: 5 + 0 should rewrite to 5
    let five = const_uop(5);
    let zero = const_uop(0);
    let add = binary_uop(BinaryOp::Add, five.clone(), zero);

    let result = graph_rewrite(&matcher, add, &mut ());
    assert!(Arc::ptr_eq(&result, &five));
}

#[test]
fn test_nested_rewrite() {
    // Pattern: x + 0 -> x
    let mut patterns = vec![];

    pattern!(patterns,
        UPat::binary(vec![BinaryOp::Add], vec![UPat::var("x"), UPat::cvar("zero")]) => |x: &Arc<UOp>, zero: &Arc<UOp>| {
            if let Op::Const(cv) = zero.op()
                && cv.0 == ConstValue::Int(0) {
                    return Some(Arc::clone(x));
                }
            None
        }
    );

    let matcher = PatternMatcher::new(patterns);

    // Test: (5 + 0) + 0 should rewrite to 5
    let five = const_uop(5);
    let zero = const_uop(0);
    let inner_add = binary_uop(BinaryOp::Add, five.clone(), zero.clone());
    let outer_add = binary_uop(BinaryOp::Add, inner_add, zero);

    let result = graph_rewrite(&matcher, outer_add, &mut ());
    assert!(Arc::ptr_eq(&result, &five));
}

#[test]
fn test_fixed_point_iteration() {
    // Pattern: x * 1 -> x
    let mut patterns = vec![];

    pattern!(patterns,
        UPat::binary(vec![BinaryOp::Mul], vec![UPat::var("x"), UPat::cvar("one")]) => |x: &Arc<UOp>, one: &Arc<UOp>| {
            if let Op::Const(cv) = one.op()
                && cv.0 == ConstValue::Int(1) {
                    return Some(Arc::clone(x));
                }
            None
        }
    );

    let matcher = PatternMatcher::new(patterns);

    // Test: ((5 * 1) * 1) should fully simplify to 5
    let five = const_uop(5);
    let one = const_uop(1);
    let inner_mul = binary_uop(BinaryOp::Mul, five.clone(), one.clone());
    let outer_mul = binary_uop(BinaryOp::Mul, inner_mul, one);

    let result = graph_rewrite(&matcher, outer_mul, &mut ());
    assert!(Arc::ptr_eq(&result, &five));
}

#[test]
fn test_multiple_patterns() {
    // Pattern 1: x + 0 -> x
    // Pattern 2: x * 1 -> x
    let mut patterns = vec![];

    pattern!(patterns,
        UPat::binary(vec![BinaryOp::Add], vec![UPat::var("x"), UPat::cvar("c")]) => |x: &Arc<UOp>, c: &Arc<UOp>| {
            if let Op::Const(cv) = c.op()
                && cv.0 == ConstValue::Int(0) {
                    return Some(Arc::clone(x));
                }
            None
        }
    );

    pattern!(patterns,
        UPat::binary(vec![BinaryOp::Mul], vec![UPat::var("x"), UPat::cvar("c")]) => |x: &Arc<UOp>, c: &Arc<UOp>| {
            if let Op::Const(cv) = c.op()
                && cv.0 == ConstValue::Int(1) {
                    return Some(Arc::clone(x));
                }
            None
        }
    );

    let matcher = PatternMatcher::new(patterns);

    // Test: (5 + 0) * 1 should simplify to 5
    let five = const_uop(5);
    let zero = const_uop(0);
    let one = const_uop(1);
    let add = binary_uop(BinaryOp::Add, five.clone(), zero);
    let mul = binary_uop(BinaryOp::Mul, add, one);

    let result = graph_rewrite(&matcher, mul, &mut ());
    assert!(Arc::ptr_eq(&result, &five));
}

#[test]
fn test_no_rewrite() {
    // Pattern: x + 0 -> x
    let mut patterns = vec![];

    pattern!(patterns,
        UPat::binary(vec![BinaryOp::Add], vec![UPat::var("x"), UPat::cvar("zero")]) => |x: &Arc<UOp>, zero: &Arc<UOp>| {
            if let Op::Const(cv) = zero.op()
                && cv.0 == ConstValue::Int(0) {
                    return Some(Arc::clone(x));
                }
            None
        }
    );

    let matcher = PatternMatcher::new(patterns);

    // Test: 5 + 3 should not rewrite
    let five = const_uop(5);
    let three = const_uop(3);
    let add = binary_uop(BinaryOp::Add, five, three);

    let result = graph_rewrite(&matcher, add.clone(), &mut ());
    // Should return original (no rewrite)
    assert!(Arc::ptr_eq(&result, &add));
}
