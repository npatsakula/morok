//! Tests for dead branch elimination in WHERE operations.

use morok_dtype::DType;
use morok_ir::types::TernaryOp;
use morok_ir::{Op, UOp};
use std::sync::Arc;

use crate::rewrite::graph_rewrite_top_down;
use crate::symbolic::symbolic_simple;

#[test]
fn test_where_always_true() {
    // Create a WHERE with condition that's always true
    let cond = UOp::native_const(true);
    let true_branch = UOp::native_const(42i32);
    let false_branch = UOp::native_const(0i32);

    let where_op = UOp::try_where(cond, true_branch.clone(), false_branch).unwrap();

    let matcher = symbolic_simple();
    let result = graph_rewrite_top_down(&matcher, where_op, &mut ());

    // Should eliminate to true branch
    assert!(Arc::ptr_eq(&result, &true_branch));
}

#[test]
fn test_where_always_false() {
    // Create a WHERE with condition that's always false
    let cond = UOp::native_const(false);
    let true_branch = UOp::native_const(42i32);
    let false_branch = UOp::native_const(0i32);

    let where_op = UOp::try_where(cond, true_branch, false_branch.clone()).unwrap();

    let matcher = symbolic_simple();
    let result = graph_rewrite_top_down(&matcher, where_op, &mut ());

    // Should eliminate to false branch
    assert!(Arc::ptr_eq(&result, &false_branch));
}

#[test]
fn test_where_range_based_true() {
    // Create a comparison that's always true based on ranges
    let x = UOp::var("x", DType::Int32, 0, 10);
    let twenty = UOp::native_const(20i32);
    let cond = x.try_cmplt(&twenty).expect("LT should succeed"); // 0..10 < 20 is always true

    let true_branch = UOp::native_const(1i32);
    let false_branch = UOp::native_const(0i32);

    let where_op = UOp::try_where(cond, true_branch.clone(), false_branch).unwrap();

    let matcher = symbolic_simple();
    let result = graph_rewrite_top_down(&matcher, where_op, &mut ());

    // The comparison should be folded to true, then WHERE should select true branch
    assert!(Arc::ptr_eq(&result, &true_branch));
}

#[test]
fn test_where_unknown_condition() {
    // Create a WHERE with unknown condition
    let x = UOp::var("x", DType::Int32, 0, 100);
    let fifty = UOp::native_const(50i32);
    let cond = x.try_cmplt(&fifty).expect("LT should succeed"); // Could be true or false

    let true_branch = UOp::native_const(1i32);
    let false_branch = UOp::native_const(0i32);

    let where_op = UOp::try_where(cond.clone(), true_branch.clone(), false_branch.clone()).unwrap();

    let matcher = symbolic_simple();
    let result = graph_rewrite_top_down(&matcher, where_op, &mut ());

    // Should not be eliminated (condition is not constant)
    match result.op() {
        Op::Ternary(TernaryOp::Where, result_cond, result_true, result_false) => {
            assert!(Arc::ptr_eq(result_cond, &cond));
            assert!(Arc::ptr_eq(result_true, &true_branch));
            assert!(Arc::ptr_eq(result_false, &false_branch));
        }
        other => panic!("Expected Op::Ternary(Where, _, _, _), got {:?}", other),
    }
}

#[test]
fn test_nested_where_elimination() {
    // Create nested WHERE operations that can be eliminated
    let cond1 = UOp::native_const(true);
    let cond2 = UOp::native_const(false);

    let val1 = UOp::native_const(1i32);
    let val2 = UOp::native_const(2i32);
    let val3 = UOp::native_const(3i32);
    let _val4 = UOp::native_const(4i32);

    // Inner WHERE: if false then val1 else val2 → val2
    let inner = UOp::try_where(cond2, val1, val2.clone()).unwrap();

    // Outer WHERE: if true then inner else val3 → inner → val2
    let outer = UOp::try_where(cond1, inner, val3).unwrap();

    let matcher = symbolic_simple();
    let result = graph_rewrite_top_down(&matcher, outer, &mut ());

    // Should eliminate to val2
    assert!(Arc::ptr_eq(&result, &val2));
}
