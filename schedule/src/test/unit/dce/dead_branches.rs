//! Tests for dead branch elimination in WHERE operations.

use morok_dtype::DType;
use morok_ir::types::BinaryOp;
use morok_ir::types::{ConstValue, TernaryOp};
use morok_ir::{Op, UOp};
use std::rc::Rc;

use crate::rewrite::graph_rewrite;
use crate::symbolic::symbolic_simple;

#[test]
fn test_where_always_true() {
    // Create a WHERE with condition that's always true
    let cond = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let true_branch = UOp::const_(DType::Int32, ConstValue::Int(42));
    let false_branch = UOp::const_(DType::Int32, ConstValue::Int(0));

    let where_op = UOp::new(Op::Ternary(TernaryOp::Where, cond, true_branch.clone(), false_branch), DType::Int32);

    let matcher = symbolic_simple();
    let result = graph_rewrite(&matcher, where_op);

    // Should eliminate to true branch
    assert!(Rc::ptr_eq(&result, &true_branch));
}

#[test]
fn test_where_always_false() {
    // Create a WHERE with condition that's always false
    let cond = UOp::const_(DType::Bool, ConstValue::Bool(false));
    let true_branch = UOp::const_(DType::Int32, ConstValue::Int(42));
    let false_branch = UOp::const_(DType::Int32, ConstValue::Int(0));

    let where_op = UOp::new(Op::Ternary(TernaryOp::Where, cond, true_branch, false_branch.clone()), DType::Int32);

    let matcher = symbolic_simple();
    let result = graph_rewrite(&matcher, where_op);

    // Should eliminate to false branch
    assert!(Rc::ptr_eq(&result, &false_branch));
}

#[test]
fn test_where_range_based_true() {
    // Create a comparison that's always true based on ranges
    let x = UOp::var("x", DType::Int32, 0, 10);
    let twenty = UOp::const_(DType::Int32, ConstValue::Int(20));
    let cond = UOp::new(Op::Binary(BinaryOp::Lt, x, twenty), DType::Bool); // 0..10 < 20 is always true

    let true_branch = UOp::const_(DType::Int32, ConstValue::Int(1));
    let false_branch = UOp::const_(DType::Int32, ConstValue::Int(0));

    let where_op = UOp::new(Op::Ternary(TernaryOp::Where, cond, true_branch.clone(), false_branch), DType::Int32);

    let matcher = symbolic_simple();
    let result = graph_rewrite(&matcher, where_op);

    // The comparison should be folded to true, then WHERE should select true branch
    assert!(Rc::ptr_eq(&result, &true_branch));
}

#[test]
fn test_where_range_based_false() {
    // Create a comparison that's always false based on ranges
    let x = UOp::var("x", DType::Int32, 100, 200);
    let fifty = UOp::const_(DType::Int32, ConstValue::Int(50));
    let cond = UOp::new(Op::Binary(BinaryOp::Lt, x, fifty), DType::Bool); // 100..200 < 50 is always false

    let true_branch = UOp::const_(DType::Int32, ConstValue::Int(1));
    let false_branch = UOp::const_(DType::Int32, ConstValue::Int(0));

    let where_op = UOp::new(Op::Ternary(TernaryOp::Where, cond, true_branch, false_branch.clone()), DType::Int32);

    let matcher = symbolic_simple();
    let result = graph_rewrite(&matcher, where_op);

    // The comparison should be folded to false, then WHERE should select false branch
    assert!(Rc::ptr_eq(&result, &false_branch));
}

#[test]
fn test_where_unknown_condition() {
    // Create a WHERE with unknown condition
    let x = UOp::var("x", DType::Int32, 0, 100);
    let fifty = UOp::const_(DType::Int32, ConstValue::Int(50));
    let cond = UOp::new(Op::Binary(BinaryOp::Lt, x.clone(), fifty.clone()), DType::Bool); // Could be true or false

    let true_branch = UOp::const_(DType::Int32, ConstValue::Int(1));
    let false_branch = UOp::const_(DType::Int32, ConstValue::Int(0));

    let where_op =
        UOp::new(Op::Ternary(TernaryOp::Where, cond.clone(), true_branch.clone(), false_branch.clone()), DType::Int32);

    let matcher = symbolic_simple();
    let result = graph_rewrite(&matcher, where_op);

    // Should not be eliminated (condition is not constant)
    match result.op() {
        Op::Ternary(TernaryOp::Where, result_cond, result_true, result_false) => {
            assert!(Rc::ptr_eq(result_cond, &cond));
            assert!(Rc::ptr_eq(result_true, &true_branch));
            assert!(Rc::ptr_eq(result_false, &false_branch));
        }
        other => panic!("Expected Op::Ternary(Where, _, _, _), got {:?}", other),
    }
}

#[test]
fn test_nested_where_elimination() {
    // Create nested WHERE operations that can be eliminated
    let cond1 = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let cond2 = UOp::const_(DType::Bool, ConstValue::Bool(false));

    let val1 = UOp::const_(DType::Int32, ConstValue::Int(1));
    let val2 = UOp::const_(DType::Int32, ConstValue::Int(2));
    let val3 = UOp::const_(DType::Int32, ConstValue::Int(3));
    let _val4 = UOp::const_(DType::Int32, ConstValue::Int(4));

    // Inner WHERE: if false then val1 else val2 → val2
    let inner = UOp::new(Op::Ternary(TernaryOp::Where, cond2, val1, val2.clone()), DType::Int32);

    // Outer WHERE: if true then inner else val3 → inner → val2
    let outer = UOp::new(Op::Ternary(TernaryOp::Where, cond1, inner, val3), DType::Int32);

    let matcher = symbolic_simple();
    let result = graph_rewrite(&matcher, outer);

    // Should eliminate to val2
    assert!(Rc::ptr_eq(&result, &val2));
}
