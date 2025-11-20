//! Ternary operation tests.
//!
//! Tests ternary operations: Where (conditional selection) and MulAcc (fused multiply-add).

use morok_dtype::DType;

use crate::{ConstValue, UOp};

// =========================================================================
// Where Operation Tests (condition ? true_val : false_val)
// =========================================================================

#[test]
fn test_where_basic() {
    let condition = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let true_val = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let false_val = UOp::const_(DType::Float32, ConstValue::Float(0.0));

    let result = UOp::where_op(condition, true_val, false_val).unwrap();
    // Where preserves the dtype of the branches
    assert_eq!(result.dtype(), DType::Float32);
}

#[test]
fn test_where_int32() {
    let condition = UOp::const_(DType::Bool, ConstValue::Bool(false));
    let true_val = UOp::const_(DType::Int32, ConstValue::Int(100));
    let false_val = UOp::const_(DType::Int32, ConstValue::Int(200));

    let result = UOp::where_op(condition, true_val, false_val).unwrap();
    assert_eq!(result.dtype(), DType::Int32);
}

#[test]
fn test_where_with_comparison() {
    let a = UOp::const_(DType::Int32, ConstValue::Int(5));
    let b = UOp::const_(DType::Int32, ConstValue::Int(10));

    // condition: a < b
    let condition = UOp::cmplt(&a, &b).unwrap();

    let true_val = UOp::const_(DType::Int32, ConstValue::Int(1));
    let false_val = UOp::const_(DType::Int32, ConstValue::Int(0));

    let result = UOp::where_op(condition, true_val, false_val).unwrap();
    assert_eq!(result.dtype(), DType::Int32);
}

#[test]
fn test_where_same_branches() {
    let condition = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let value = UOp::const_(DType::Float32, ConstValue::Float(42.0));

    // where(cond, x, x) should be optimizable to just x
    let result = UOp::where_op(condition, value.clone(), value).unwrap();
    assert_eq!(result.dtype(), DType::Float32);
}

#[test]
fn test_where_const_true_condition() {
    let true_cond = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let true_val = UOp::const_(DType::Int32, ConstValue::Int(100));
    let false_val = UOp::const_(DType::Int32, ConstValue::Int(200));

    // where(true, x, y) should be optimizable to x
    let result = UOp::where_op(true_cond, true_val, false_val).unwrap();
    assert_eq!(result.dtype(), DType::Int32);
}

#[test]
fn test_where_const_false_condition() {
    let false_cond = UOp::const_(DType::Bool, ConstValue::Bool(false));
    let true_val = UOp::const_(DType::Int32, ConstValue::Int(100));
    let false_val = UOp::const_(DType::Int32, ConstValue::Int(200));

    // where(false, x, y) should be optimizable to y
    let result = UOp::where_op(false_cond, true_val, false_val).unwrap();
    assert_eq!(result.dtype(), DType::Int32);
}

#[test]
fn test_where_nested() {
    let cond1 = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let cond2 = UOp::const_(DType::Bool, ConstValue::Bool(false));

    let val1 = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let val2 = UOp::const_(DType::Float32, ConstValue::Float(2.0));
    let val3 = UOp::const_(DType::Float32, ConstValue::Float(3.0));

    // Nested: where(cond1, val1, where(cond2, val2, val3))
    let inner = UOp::where_op(cond2, val2, val3).unwrap();
    let result = UOp::where_op(cond1, val1, inner).unwrap();

    assert_eq!(result.dtype(), DType::Float32);
}

#[test]
fn test_where_with_different_dtypes() {
    let condition = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let int_val = UOp::const_(DType::Int32, ConstValue::Int(5));
    let float_val = UOp::const_(DType::Float32, ConstValue::Float(5.0));

    // Branches should have compatible types
    // Result takes dtype from true branch (first non-condition arg)
    let result = UOp::where_op(condition, int_val, float_val).unwrap();
    assert_eq!(result.dtype(), DType::Int32);
}

#[test]
fn test_where_bool_branches() {
    let condition = UOp::const_(DType::Bool, ConstValue::Bool(false));
    let true_branch = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let false_branch = UOp::const_(DType::Bool, ConstValue::Bool(false));

    let result = UOp::where_op(condition, true_branch, false_branch).unwrap();
    assert_eq!(result.dtype(), DType::Bool);
}

#[test]
fn test_where_with_zero() {
    let condition = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let zero = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    let one = UOp::const_(DType::Float32, ConstValue::Float(1.0));

    let result = UOp::where_op(condition, one, zero).unwrap();
    assert_eq!(result.dtype(), DType::Float32);
}

// =========================================================================
// MulAcc Operation Tests (a * b + c)
// =========================================================================

#[test]
fn test_mulacc_basic() {
    let a = UOp::const_(DType::Float32, ConstValue::Float(2.0));
    let b = UOp::const_(DType::Float32, ConstValue::Float(3.0));
    let c = UOp::const_(DType::Float32, ConstValue::Float(4.0));

    let result = UOp::mulacc_op(a, b, c).unwrap(); // 2*3 + 4 = 10
    // MulAcc preserves dtype of first operand
    assert_eq!(result.dtype(), DType::Float32);
}

#[test]
fn test_mulacc_int32() {
    let a = UOp::const_(DType::Int32, ConstValue::Int(5));
    let b = UOp::const_(DType::Int32, ConstValue::Int(6));
    let c = UOp::const_(DType::Int32, ConstValue::Int(7));

    let result = UOp::mulacc_op(a, b, c).unwrap(); // 5*6 + 7 = 37
    assert_eq!(result.dtype(), DType::Int32);
}

#[test]
fn test_mulacc_with_zero_multiplier() {
    let zero = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    let b = UOp::const_(DType::Float32, ConstValue::Float(100.0));
    let c = UOp::const_(DType::Float32, ConstValue::Float(5.0));

    let result = UOp::mulacc_op(zero, b, c).unwrap(); // 0*100 + 5 = 5
    assert_eq!(result.dtype(), DType::Float32);
}

#[test]
fn test_mulacc_with_zero_accumulator() {
    let a = UOp::const_(DType::Float32, ConstValue::Float(2.0));
    let b = UOp::const_(DType::Float32, ConstValue::Float(3.0));
    let zero = UOp::const_(DType::Float32, ConstValue::Float(0.0));

    let result = UOp::mulacc_op(a, b, zero).unwrap(); // 2*3 + 0 = 6
    assert_eq!(result.dtype(), DType::Float32);
}

#[test]
fn test_mulacc_with_one() {
    let one = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let b = UOp::const_(DType::Float32, ConstValue::Float(5.0));
    let c = UOp::const_(DType::Float32, ConstValue::Float(3.0));

    let result = UOp::mulacc_op(one, b, c).unwrap(); // 1*5 + 3 = 8
    assert_eq!(result.dtype(), DType::Float32);
}

#[test]
fn test_mulacc_negative_values() {
    let a = UOp::const_(DType::Int32, ConstValue::Int(-2));
    let b = UOp::const_(DType::Int32, ConstValue::Int(3));
    let c = UOp::const_(DType::Int32, ConstValue::Int(10));

    let result = UOp::mulacc_op(a, b, c).unwrap(); // -2*3 + 10 = 4
    assert_eq!(result.dtype(), DType::Int32);
}

#[test]
fn test_mulacc_vs_separate_ops() {
    let a = UOp::const_(DType::Float32, ConstValue::Float(2.0));
    let b = UOp::const_(DType::Float32, ConstValue::Float(3.0));
    let c = UOp::const_(DType::Float32, ConstValue::Float(4.0));

    // Fused: a*b + c
    let fused = UOp::mulacc_op(a.clone(), b.clone(), c.clone()).unwrap();

    // Separate: (a * b) + c
    let mul = a.try_mul_op(&b).unwrap();
    let separate = mul.try_add_op(&c).unwrap();

    // Both should have same dtype
    assert_eq!(fused.dtype(), separate.dtype());
    assert_eq!(fused.dtype(), DType::Float32);
}

#[test]
fn test_mulacc_chained() {
    let a = UOp::const_(DType::Float32, ConstValue::Float(2.0));
    let b = UOp::const_(DType::Float32, ConstValue::Float(3.0));
    let c = UOp::const_(DType::Float32, ConstValue::Float(4.0));
    let d = UOp::const_(DType::Float32, ConstValue::Float(5.0));

    // First mulacc: 2*3 + 4 = 10
    let result1 = UOp::mulacc_op(a.clone(), b.clone(), c).unwrap();

    // Chained mulacc: (2*3 + 4) * 5 + ...
    // This tests using mulacc result in another operation
    let result2 = result1.try_mul_op(&d).unwrap();

    assert_eq!(result2.dtype(), DType::Float32);
}
