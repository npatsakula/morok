//! Tests for bounds check elimination using range analysis.

use morok_dtype::DType;
use morok_ir::types::{BinaryOp, ConstValue, TernaryOp};
use morok_ir::{Op, UOp};
use std::rc::Rc;

use crate::rewrite::graph_rewrite;
use crate::symbolic::symbolic_simple;

#[test]
fn test_lt_always_true() {
    // idx in [0, 15], size = 32
    // idx < size is always true
    let idx = UOp::var("idx", DType::Int32, 0, 15);
    let size = UOp::const_(DType::Int32, ConstValue::Int(32));
    let check = UOp::new(Op::Binary(BinaryOp::Lt, idx, size), DType::Bool);

    let matcher = symbolic_simple();
    let result = graph_rewrite(&matcher, check, &mut ());

    // Should be constant true
    match result.op() {
        Op::Const(c) => assert_eq!(c.0, ConstValue::Bool(true)),
        other => panic!("Expected Op::Const, got {:?}", other),
    }
}

#[test]
fn test_lt_always_false() {
    // idx in [100, 200], size = 50
    // idx < size is always false
    let idx = UOp::var("idx", DType::Int32, 100, 200);
    let size = UOp::const_(DType::Int32, ConstValue::Int(50));
    let check = UOp::new(Op::Binary(BinaryOp::Lt, idx, size), DType::Bool);

    let matcher = symbolic_simple();
    let result = graph_rewrite(&matcher, check, &mut ());

    // Should be constant false
    match result.op() {
        Op::Const(c) => assert_eq!(c.0, ConstValue::Bool(false)),
        other => panic!("Expected Op::Const, got {:?}", other),
    }
}

#[test]
fn test_lt_unknown() {
    // idx in [0, 100], size = 50
    // idx < size could be true or false
    let idx = UOp::var("idx", DType::Int32, 0, 100);
    let size = UOp::const_(DType::Int32, ConstValue::Int(50));
    let check = UOp::new(Op::Binary(BinaryOp::Lt, idx.clone(), size.clone()), DType::Bool);

    let matcher = symbolic_simple();
    let result = graph_rewrite(&matcher, check, &mut ());

    // Should not be constant
    match result.op() {
        Op::Binary(BinaryOp::Lt, a, b) => {
            assert!(Rc::ptr_eq(a, &idx));
            assert!(Rc::ptr_eq(b, &size));
        }
        other => panic!("Expected Op::Binary(Lt, _, _), got {:?}", other),
    }
}

#[test]
fn test_eq_same_var() {
    // x == x is always true for integers
    let x = UOp::var("x", DType::Int32, 0, 100);
    let check = UOp::new(Op::Binary(BinaryOp::Eq, x.clone(), x.clone()), DType::Bool);

    let matcher = symbolic_simple();
    let result = graph_rewrite(&matcher, check, &mut ());

    // Should be constant true
    match result.op() {
        Op::Const(c) => assert_eq!(c.0, ConstValue::Bool(true)),
        other => panic!("Expected Op::Const, got {:?}", other),
    }
}

#[test]
fn test_eq_non_overlapping_ranges() {
    // x in [0, 10], y in [20, 30]
    // x == y is always false (ranges don't overlap)
    let x = UOp::var("x", DType::Int32, 0, 10);
    let y = UOp::var("y", DType::Int32, 20, 30);
    let check = UOp::new(Op::Binary(BinaryOp::Eq, x, y), DType::Bool);

    let matcher = symbolic_simple();
    let result = graph_rewrite(&matcher, check, &mut ());

    // Should be constant false
    match result.op() {
        Op::Const(c) => assert_eq!(c.0, ConstValue::Bool(false)),
        other => panic!("Expected Op::Const, got {:?}", other),
    }
}

#[test]
fn test_eq_same_constant() {
    // Both are the same constant value
    let x = UOp::var("x", DType::Int32, 5, 5);
    let y = UOp::var("y", DType::Int32, 5, 5);
    let check = UOp::new(Op::Binary(BinaryOp::Eq, x, y), DType::Bool);

    let matcher = symbolic_simple();
    let result = graph_rewrite(&matcher, check, &mut ());

    // Should be constant true
    match result.op() {
        Op::Const(c) => assert_eq!(c.0, ConstValue::Bool(true)),
        other => panic!("Expected Op::Const, got {:?}", other),
    }
}

#[test]
fn test_ne_same_var() {
    // x != x is always false for integers
    let x = UOp::var("x", DType::Int32, 0, 100);
    let check = UOp::new(Op::Binary(BinaryOp::Ne, x.clone(), x.clone()), DType::Bool);

    let matcher = symbolic_simple();
    let result = graph_rewrite(&matcher, check, &mut ());

    // Should be constant false
    match result.op() {
        Op::Const(c) => assert_eq!(c.0, ConstValue::Bool(false)),
        other => panic!("Expected Op::Const, got {:?}", other),
    }
}

#[test]
fn test_ne_non_overlapping_ranges() {
    // x in [0, 10], y in [20, 30]
    // x != y is always true (ranges don't overlap)
    let x = UOp::var("x", DType::Int32, 0, 10);
    let y = UOp::var("y", DType::Int32, 20, 30);
    let check = UOp::new(Op::Binary(BinaryOp::Ne, x, y), DType::Bool);

    let matcher = symbolic_simple();
    let result = graph_rewrite(&matcher, check, &mut ());

    // Should be constant true
    match result.op() {
        Op::Const(c) => assert_eq!(c.0, ConstValue::Bool(true)),
        other => panic!("Expected Op::Const, got {:?}", other),
    }
}

#[test]
fn test_cascading_bounds_elimination() {
    // Test that eliminated bounds checks enable further optimizations
    let idx = UOp::var("idx", DType::Int32, 0, 10);
    let size = UOp::const_(DType::Int32, ConstValue::Int(20));

    // idx < size is always true
    let bounds_check = UOp::new(Op::Binary(BinaryOp::Lt, idx, size), DType::Bool);

    // Use bounds check in WHERE
    let safe_val = UOp::const_(DType::Int32, ConstValue::Int(42));
    let error_val = UOp::const_(DType::Int32, ConstValue::Int(-1));
    let where_op = UOp::new(Op::Ternary(TernaryOp::Where, bounds_check, safe_val.clone(), error_val), DType::Int32);

    let matcher = symbolic_simple();
    let result = graph_rewrite(&matcher, where_op, &mut ());

    // Should eliminate to safe_val
    assert!(Rc::ptr_eq(&result, &safe_val));
}
