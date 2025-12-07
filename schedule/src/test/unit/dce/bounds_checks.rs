//! Tests for bounds check elimination using range analysis.

use morok_dtype::DType;
use morok_ir::types::{BinaryOp, ConstValue};
use morok_ir::{Op, UOp};
use std::sync::Arc;

use crate::rewrite::graph_rewrite;
use crate::symbolic::symbolic_simple;

#[test]
fn test_lt_always_true() {
    // idx in [0, 15], size = 32
    // idx < size is always true
    let idx = UOp::var("idx", DType::Int32, 15);
    let size = UOp::native_const(32i32);
    let check = idx.try_cmplt(&size).unwrap();

    let matcher = symbolic_simple();
    let result = graph_rewrite(&matcher, check, &mut ());

    // Should be constant true
    match result.op() {
        Op::Const(c) => assert_eq!(c.0, ConstValue::Bool(true)),
        other => panic!("Expected Op::Const, got {:?}", other),
    }
}

#[test]
fn test_lt_unknown() {
    // idx in [0, 100], size = 50
    // idx < size could be true or false
    let idx = UOp::var("idx", DType::Int32, 100);
    let size = UOp::native_const(50i32);
    let check = idx.try_cmplt(&size).unwrap();

    let matcher = symbolic_simple();
    let result = graph_rewrite(&matcher, check, &mut ());

    // Should not be constant
    match result.op() {
        Op::Binary(BinaryOp::Lt, a, b) => {
            assert!(Arc::ptr_eq(a, &idx));
            assert!(Arc::ptr_eq(b, &size));
        }
        other => panic!("Expected Op::Binary(Lt, _, _), got {:?}", other),
    }
}

#[test]
fn test_eq_same_var() {
    // x == x is always true for integers
    let x = UOp::var("x", DType::Int32, 100);
    let check = x.try_cmpeq(&x).unwrap();

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
    let x = UOp::var("x", DType::Int32, 100);
    let check = x.try_cmpne(&x).unwrap();

    let matcher = symbolic_simple();
    let result = graph_rewrite(&matcher, check, &mut ());

    // Should be constant false
    match result.op() {
        Op::Const(c) => assert_eq!(c.0, ConstValue::Bool(false)),
        other => panic!("Expected Op::Const, got {:?}", other),
    }
}

#[test]
fn test_cascading_bounds_elimination() {
    // Test that eliminated bounds checks enable further optimizations
    let idx = UOp::var("idx", DType::Int32, 10);
    let size = UOp::native_const(20i32);

    // idx < size is always true
    let bounds_check = idx.try_cmplt(&size).unwrap();

    // Use bounds check in WHERE
    let safe_val = UOp::native_const(42i32);
    let error_val = UOp::native_const(-1i32);
    let where_op = UOp::try_where(bounds_check, safe_val.clone(), error_val).unwrap();

    let matcher = symbolic_simple();
    let result = graph_rewrite(&matcher, where_op, &mut ());

    // Should eliminate to safe_val
    assert!(Arc::ptr_eq(&result, &safe_val));
}
