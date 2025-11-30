//! Ternary operation tests.
//!
//! Tests ternary operations: Where (conditional selection) and MulAcc (fused multiply-add).

use morok_dtype::DType;

use crate::UOp;

// =========================================================================
// Where Operation Tests (condition ? true_val : false_val)
// =========================================================================

#[test]
fn test_where_basic() {
    // Where preserves the dtype of the branches
    assert_eq!(
        UOp::try_where(UOp::native_const(true), UOp::native_const(1.0f32), UOp::native_const(0.0f32)).unwrap().dtype(),
        DType::Float32
    );
}

#[test]
fn test_where_int32() {
    assert_eq!(
        UOp::try_where(UOp::native_const(false), UOp::native_const(100i32), UOp::native_const(200i32)).unwrap().dtype(),
        DType::Int32
    );
}

#[test]
fn test_where_with_comparison() {
    // condition: a < b
    assert_eq!(
        UOp::try_where(
            UOp::native_const(5i32).try_cmplt(&UOp::native_const(10i32)).unwrap(),
            UOp::native_const(1i32),
            UOp::native_const(0i32)
        )
        .unwrap()
        .dtype(),
        DType::Int32
    );
}

#[test]
fn test_where_same_branches() {
    let value = UOp::native_const(42.0f32);

    // where(cond, x, x) should be optimizable to just x
    let result = UOp::try_where(UOp::native_const(true), value.clone(), value).unwrap();
    assert_eq!(result.dtype(), DType::Float32);
}

#[test]
fn test_where_const_true_condition() {
    // where(true, x, y) should be optimizable to x
    assert_eq!(
        UOp::try_where(UOp::native_const(true), UOp::native_const(100i32), UOp::native_const(200i32)).unwrap().dtype(),
        DType::Int32
    );
}

#[test]
fn test_where_const_false_condition() {
    // where(false, x, y) should be optimizable to y
    assert_eq!(
        UOp::try_where(UOp::native_const(false), UOp::native_const(100i32), UOp::native_const(200i32)).unwrap().dtype(),
        DType::Int32
    );
}

#[test]
fn test_where_nested() {
    // Nested: where(cond1, val1, where(cond2, val2, val3))
    let inner = UOp::try_where(UOp::native_const(false), UOp::native_const(2.0f32), UOp::native_const(3.0f32)).unwrap();
    let result = UOp::try_where(UOp::native_const(true), UOp::native_const(1.0f32), inner).unwrap();

    assert_eq!(result.dtype(), DType::Float32);
}

#[test]
fn test_where_with_different_dtypes() {
    // Branches should have compatible types
    // Result takes dtype from true branch (first non-condition arg)
    assert_eq!(
        UOp::try_where(UOp::native_const(true), UOp::native_const(5i32), UOp::native_const(5.0f32)).unwrap().dtype(),
        DType::Int32
    );
}

#[test]
fn test_where_bool_branches() {
    assert_eq!(
        UOp::try_where(UOp::native_const(false), UOp::native_const(true), UOp::native_const(false)).unwrap().dtype(),
        DType::Bool
    );
}

#[test]
fn test_where_with_zero() {
    assert_eq!(
        UOp::try_where(UOp::native_const(true), UOp::native_const(1.0f32), UOp::native_const(0.0f32)).unwrap().dtype(),
        DType::Float32
    );
}

// =========================================================================
// MulAcc Operation Tests (a * b + c)
// =========================================================================

#[test]
fn test_mulacc_basic() {
    // MulAcc preserves dtype of first operand (2*3 + 4 = 10)
    assert_eq!(
        UOp::try_mulacc(UOp::native_const(2.0f32), UOp::native_const(3.0f32), UOp::native_const(4.0f32))
            .unwrap()
            .dtype(),
        DType::Float32
    );
}

#[test]
fn test_mulacc_int32() {
    assert_eq!(
        UOp::try_mulacc(UOp::native_const(5i32), UOp::native_const(6i32), UOp::native_const(7i32)).unwrap().dtype(),
        DType::Int32
    );
}

#[test]
fn test_mulacc_with_zero_multiplier() {
    assert_eq!(
        UOp::try_mulacc(UOp::native_const(0.0f32), UOp::native_const(100.0f32), UOp::native_const(5.0f32))
            .unwrap()
            .dtype(),
        DType::Float32
    );
}

#[test]
fn test_mulacc_with_zero_accumulator() {
    assert_eq!(
        UOp::try_mulacc(UOp::native_const(2.0f32), UOp::native_const(3.0f32), UOp::native_const(0.0f32))
            .unwrap()
            .dtype(),
        DType::Float32
    );
}

#[test]
fn test_mulacc_with_one() {
    assert_eq!(
        UOp::try_mulacc(UOp::native_const(1.0f32), UOp::native_const(5.0f32), UOp::native_const(3.0f32))
            .unwrap()
            .dtype(),
        DType::Float32
    );
}

#[test]
fn test_mulacc_negative_values() {
    assert_eq!(
        UOp::try_mulacc(UOp::native_const(-2i32), UOp::native_const(3i32), UOp::native_const(10i32)).unwrap().dtype(),
        DType::Int32
    );
}

#[test]
fn test_mulacc_vs_separate_ops() {
    let a = UOp::native_const(2.0f32);
    let b = UOp::native_const(3.0f32);
    let c = UOp::native_const(4.0f32);

    // Fused: a*b + c
    let fused = UOp::try_mulacc(a.clone(), b.clone(), c.clone()).unwrap();

    // Separate: (a * b) + c
    let mul = a.try_mul(&b).unwrap();
    let separate = mul.try_add(&c).unwrap();

    // Both should have same dtype
    assert_eq!(fused.dtype(), separate.dtype());
    assert_eq!(fused.dtype(), DType::Float32);
}

#[test]
fn test_mulacc_chained() {
    let a = UOp::native_const(2.0f32);
    let b = UOp::native_const(3.0f32);
    let c = UOp::native_const(4.0f32);
    let d = UOp::native_const(5.0f32);

    // First mulacc: 2*3 + 4 = 10
    let result1 = UOp::try_mulacc(a.clone(), b.clone(), c).unwrap();

    // Chained mulacc: (2*3 + 4) * 5 + ...
    // This tests using mulacc result in another operation
    let result2 = result1.try_mul(&d).unwrap();

    assert_eq!(result2.dtype(), DType::Float32);
}
