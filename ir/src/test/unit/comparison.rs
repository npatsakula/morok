//! Comparison operation tests.
//!
//! Tests comparison operations (Lt, Eq, Ne) which always return Bool dtype.

use std::f32::consts::{E, PI};

use morok_dtype::DType;

use crate::UOp;

// =========================================================================
// Less Than (Lt) Tests
// =========================================================================

#[test]
fn test_lt_int32() {
    // Comparison operations always return Bool
    assert_eq!(UOp::native_const(5i32).try_cmplt(&UOp::native_const(10i32)).unwrap().dtype(), DType::Bool);
}

#[test]
fn test_lt_float32() {
    assert_eq!(UOp::native_const(PI).try_cmplt(&UOp::native_const(E)).unwrap().dtype(), DType::Bool);
}

#[test]
fn test_lt_with_type_promotion() {
    // Int32 should promote to Float32 for comparison
    let result = UOp::native_const(5i32).try_cmplt(&UOp::native_const(10.0f32)).unwrap();
    assert_eq!(result.dtype(), DType::Bool);
}

#[test]
fn test_lt_negative_numbers() {
    let result = UOp::native_const(-10i32).try_cmplt(&UOp::native_const(-5i32)).unwrap(); // -10 < -5 = true
    assert_eq!(result.dtype(), DType::Bool);
}

#[test]
fn test_lt_equal_values() {
    let result = UOp::native_const(42i32).try_cmplt(&UOp::native_const(42i32)).unwrap(); // 42 < 42 = false
    assert_eq!(result.dtype(), DType::Bool);
}

#[test]
fn test_lt_zero() {
    let result1 = UOp::native_const(0.0f32).try_cmplt(&UOp::native_const(1.0f32)).unwrap(); // 0 < 1 = true
    assert_eq!(result1.dtype(), DType::Bool);

    let result2 = UOp::native_const(-1.0f32).try_cmplt(&UOp::native_const(0.0f32)).unwrap(); // -1 < 0 = true
    assert_eq!(result2.dtype(), DType::Bool);
}

// =========================================================================
// Equality (Eq) Tests
// =========================================================================

#[test]
fn test_eq_int32() {
    let result = UOp::native_const(42i32).try_cmpeq(&UOp::native_const(42i32)).unwrap();
    assert_eq!(result.dtype(), DType::Bool);
}

#[test]
fn test_eq_float32() {
    let result = UOp::native_const(PI).try_cmpeq(&UOp::native_const(PI)).unwrap();
    assert_eq!(result.dtype(), DType::Bool);
}

#[test]
fn test_eq_different_values() {
    let result = UOp::native_const(5i32).try_cmpeq(&UOp::native_const(10i32)).unwrap();
    assert_eq!(result.dtype(), DType::Bool);
}

#[test]
fn test_eq_bool() {
    let result = UOp::native_const(true).try_cmpeq(&UOp::native_const(true)).unwrap();
    assert_eq!(result.dtype(), DType::Bool);
}

#[test]
fn test_eq_with_type_promotion() {
    let result = UOp::native_const(5i32).try_cmpeq(&UOp::native_const(5.0f32)).unwrap();
    assert_eq!(result.dtype(), DType::Bool);
}

#[test]
fn test_eq_zero() {
    let result1 = UOp::native_const(0i32).try_cmpeq(&UOp::native_const(0i32)).unwrap();
    assert_eq!(result1.dtype(), DType::Bool);

    let result2 = UOp::native_const(0.0f32).try_cmpeq(&UOp::native_const(0.0f32)).unwrap();
    assert_eq!(result2.dtype(), DType::Bool);
}

// =========================================================================
// Inequality (Ne) Tests
// =========================================================================

#[test]
fn test_ne_int32() {
    let result = UOp::native_const(5i32).try_cmpne(&UOp::native_const(10i32)).unwrap();
    assert_eq!(result.dtype(), DType::Bool);
}

#[test]
fn test_ne_float32() {
    let result = UOp::native_const(PI).try_cmpne(&UOp::native_const(E)).unwrap();
    assert_eq!(result.dtype(), DType::Bool);
}

#[test]
fn test_ne_equal_values() {
    let result = UOp::native_const(42i32).try_cmpne(&UOp::native_const(42i32)).unwrap(); // 42 != 42 = false
    assert_eq!(result.dtype(), DType::Bool);
}

#[test]
fn test_ne_bool() {
    let result = UOp::native_const(true).try_cmpne(&UOp::native_const(false)).unwrap();
    assert_eq!(result.dtype(), DType::Bool);
}

#[test]
fn test_ne_with_type_promotion() {
    // Note: Int8 is kept as UOp::const_ (SKIP rule)
    let int_val = UOp::const_(DType::Int8, crate::ConstValue::Int(5));
    let result = int_val.try_cmpne(&UOp::native_const(10i32)).unwrap();
    assert_eq!(result.dtype(), DType::Bool);
}

// =========================================================================
// Mixed Type Comparisons
// =========================================================================

#[test]
fn test_comparison_signed_unsigned() {
    // Should promote to common type for comparison
    let result = UOp::native_const(-1i32).try_cmplt(&UOp::native_const(1u32)).unwrap();
    assert_eq!(result.dtype(), DType::Bool);
}

#[test]
fn test_comparison_different_int_sizes() {
    // Note: Int8 is kept as UOp::const_ (SKIP rule)
    let small = UOp::const_(DType::Int8, crate::ConstValue::Int(5));
    let result = small.try_cmplt(&UOp::native_const(10i32)).unwrap();
    assert_eq!(result.dtype(), DType::Bool);
}

#[test]
fn test_comparison_int_and_float() {
    let result = UOp::native_const(5i32).try_cmplt(&UOp::native_const(5.5f32)).unwrap();
    assert_eq!(result.dtype(), DType::Bool);
}

// =========================================================================
// Comparison Chaining (Using Result)
// =========================================================================

#[test]
fn test_comparison_chain() {
    let a = UOp::native_const(1i32);
    let b = UOp::native_const(2i32);
    let c = UOp::native_const(3i32);

    // a < b
    let cmp1 = a.try_cmplt(&b).unwrap();
    assert_eq!(cmp1.dtype(), DType::Bool);

    // b < c
    let cmp2 = b.try_cmplt(&c).unwrap();
    assert_eq!(cmp2.dtype(), DType::Bool);

    // (a < b) AND (b < c) - both comparisons return Bool
    let result = cmp1.try_and_op(&cmp2).unwrap();
    assert_eq!(result.dtype(), DType::Bool);
}

// =========================================================================
// Edge Cases
// =========================================================================

#[test]
fn test_comparison_with_zero_float() {
    // In IEEE 754, +0.0 == -0.0
    let result = UOp::native_const(0.0f32).try_cmpeq(&UOp::native_const(-0.0f32)).unwrap();
    assert_eq!(result.dtype(), DType::Bool);
}

#[test]
fn test_comparison_preserves_dtype_independence() {
    // Verify that comparison result type doesn't depend on operand types
    let int_cmp = UOp::native_const(1i32).try_cmplt(&UOp::native_const(2i32)).unwrap();
    let float_cmp = UOp::native_const(1.0f32).try_cmplt(&UOp::native_const(2.0f32)).unwrap();

    // Both should return Bool regardless of operand types
    assert_eq!(int_cmp.dtype(), DType::Bool);
    assert_eq!(float_cmp.dtype(), DType::Bool);
    assert_eq!(int_cmp.dtype(), float_cmp.dtype());
}
