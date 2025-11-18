//! Comparison operation tests.
//!
//! Tests comparison operations (Lt, Eq, Ne) which always return Bool dtype.

use morok_dtype::DType;

use crate::{ConstValue, UOp};

// =========================================================================
// Less Than (Lt) Tests
// =========================================================================

#[test]
fn test_lt_int32() {
    let a = UOp::const_(DType::Int32, ConstValue::Int(5));
    let b = UOp::const_(DType::Int32, ConstValue::Int(10));

    let result = UOp::cmplt(&a, &b).unwrap();
    // Comparison operations always return Bool
    assert_eq!(result.dtype(), DType::Bool);
}

#[test]
fn test_lt_float32() {
    let a = UOp::const_(DType::Float32, ConstValue::Float(std::f32::consts::PI as f64));
    let b = UOp::const_(DType::Float32, ConstValue::Float(std::f32::consts::E as f64));

    let result = UOp::cmplt(&a, &b).unwrap();
    assert_eq!(result.dtype(), DType::Bool);
}

#[test]
fn test_lt_with_type_promotion() {
    let int_val = UOp::const_(DType::Int32, ConstValue::Int(5));
    let float_val = UOp::const_(DType::Float32, ConstValue::Float(10.0));

    // Int32 should promote to Float32 for comparison
    let result = UOp::cmplt(&int_val, &float_val).unwrap();
    assert_eq!(result.dtype(), DType::Bool);
}

#[test]
fn test_lt_negative_numbers() {
    let a = UOp::const_(DType::Int32, ConstValue::Int(-10));
    let b = UOp::const_(DType::Int32, ConstValue::Int(-5));

    let result = UOp::cmplt(&a, &b).unwrap(); // -10 < -5 = true
    assert_eq!(result.dtype(), DType::Bool);
}

#[test]
fn test_lt_equal_values() {
    let a = UOp::const_(DType::Int32, ConstValue::Int(42));
    let b = UOp::const_(DType::Int32, ConstValue::Int(42));

    let result = UOp::cmplt(&a, &b).unwrap(); // 42 < 42 = false
    assert_eq!(result.dtype(), DType::Bool);
}

#[test]
fn test_lt_zero() {
    let zero = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    let positive = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let negative = UOp::const_(DType::Float32, ConstValue::Float(-1.0));

    let result1 = UOp::cmplt(&zero, &positive).unwrap(); // 0 < 1 = true
    assert_eq!(result1.dtype(), DType::Bool);

    let result2 = UOp::cmplt(&negative, &zero).unwrap(); // -1 < 0 = true
    assert_eq!(result2.dtype(), DType::Bool);
}

// =========================================================================
// Equality (Eq) Tests
// =========================================================================

#[test]
fn test_eq_int32() {
    let a = UOp::const_(DType::Int32, ConstValue::Int(42));
    let b = UOp::const_(DType::Int32, ConstValue::Int(42));

    let result = UOp::cmpeq(&a, &b).unwrap();
    assert_eq!(result.dtype(), DType::Bool);
}

#[test]
fn test_eq_float32() {
    let a = UOp::const_(DType::Float32, ConstValue::Float(std::f32::consts::PI as f64));
    let b = UOp::const_(DType::Float32, ConstValue::Float(std::f32::consts::PI as f64));

    let result = UOp::cmpeq(&a, &b).unwrap();
    assert_eq!(result.dtype(), DType::Bool);
}

#[test]
fn test_eq_different_values() {
    let a = UOp::const_(DType::Int32, ConstValue::Int(5));
    let b = UOp::const_(DType::Int32, ConstValue::Int(10));

    let result = UOp::cmpeq(&a, &b).unwrap();
    assert_eq!(result.dtype(), DType::Bool);
}

#[test]
fn test_eq_bool() {
    let a = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let b = UOp::const_(DType::Bool, ConstValue::Bool(true));

    let result = UOp::cmpeq(&a, &b).unwrap();
    assert_eq!(result.dtype(), DType::Bool);
}

#[test]
fn test_eq_with_type_promotion() {
    let int_val = UOp::const_(DType::Int32, ConstValue::Int(5));
    let float_val = UOp::const_(DType::Float32, ConstValue::Float(5.0));

    let result = UOp::cmpeq(&int_val, &float_val).unwrap();
    assert_eq!(result.dtype(), DType::Bool);
}

#[test]
fn test_eq_zero() {
    let zero_int = UOp::const_(DType::Int32, ConstValue::Int(0));
    let zero_float = UOp::const_(DType::Float32, ConstValue::Float(0.0));

    let result1 = UOp::cmpeq(&zero_int, &zero_int).unwrap();
    assert_eq!(result1.dtype(), DType::Bool);

    let result2 = UOp::cmpeq(&zero_float, &zero_float).unwrap();
    assert_eq!(result2.dtype(), DType::Bool);
}

// =========================================================================
// Inequality (Ne) Tests
// =========================================================================

#[test]
fn test_ne_int32() {
    let a = UOp::const_(DType::Int32, ConstValue::Int(5));
    let b = UOp::const_(DType::Int32, ConstValue::Int(10));

    let result = UOp::cmpne(&a, &b).unwrap();
    assert_eq!(result.dtype(), DType::Bool);
}

#[test]
fn test_ne_float32() {
    let a = UOp::const_(DType::Float32, ConstValue::Float(std::f32::consts::PI as f64));
    let b = UOp::const_(DType::Float32, ConstValue::Float(std::f32::consts::E as f64));

    let result = UOp::cmpne(&a, &b).unwrap();
    assert_eq!(result.dtype(), DType::Bool);
}

#[test]
fn test_ne_equal_values() {
    let a = UOp::const_(DType::Int32, ConstValue::Int(42));
    let b = UOp::const_(DType::Int32, ConstValue::Int(42));

    let result = UOp::cmpne(&a, &b).unwrap(); // 42 != 42 = false
    assert_eq!(result.dtype(), DType::Bool);
}

#[test]
fn test_ne_bool() {
    let a = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let b = UOp::const_(DType::Bool, ConstValue::Bool(false));

    let result = UOp::cmpne(&a, &b).unwrap();
    assert_eq!(result.dtype(), DType::Bool);
}

#[test]
fn test_ne_with_type_promotion() {
    let int_val = UOp::const_(DType::Int8, ConstValue::Int(5));
    let large_int = UOp::const_(DType::Int32, ConstValue::Int(10));

    let result = UOp::cmpne(&int_val, &large_int).unwrap();
    assert_eq!(result.dtype(), DType::Bool);
}

// =========================================================================
// Mixed Type Comparisons
// =========================================================================

#[test]
fn test_comparison_signed_unsigned() {
    let signed = UOp::const_(DType::Int32, ConstValue::Int(-1));
    let unsigned = UOp::const_(DType::UInt32, ConstValue::UInt(1));

    // Should promote to common type for comparison
    let result = UOp::cmplt(&signed, &unsigned).unwrap();
    assert_eq!(result.dtype(), DType::Bool);
}

#[test]
fn test_comparison_different_int_sizes() {
    let small = UOp::const_(DType::Int8, ConstValue::Int(5));
    let large = UOp::const_(DType::Int32, ConstValue::Int(10));

    let result = UOp::cmplt(&small, &large).unwrap();
    assert_eq!(result.dtype(), DType::Bool);
}

#[test]
fn test_comparison_int_and_float() {
    let int_val = UOp::const_(DType::Int32, ConstValue::Int(5));
    let float_val = UOp::const_(DType::Float32, ConstValue::Float(5.5));

    let result = UOp::cmplt(&int_val, &float_val).unwrap();
    assert_eq!(result.dtype(), DType::Bool);
}

// =========================================================================
// Comparison Chaining (Using Result)
// =========================================================================

#[test]
fn test_comparison_chain() {
    let a = UOp::const_(DType::Int32, ConstValue::Int(1));
    let b = UOp::const_(DType::Int32, ConstValue::Int(2));
    let c = UOp::const_(DType::Int32, ConstValue::Int(3));

    // a < b
    let cmp1 = UOp::cmplt(&a, &b).unwrap();
    assert_eq!(cmp1.dtype(), DType::Bool);

    // b < c
    let cmp2 = UOp::cmplt(&b, &c).unwrap();
    assert_eq!(cmp2.dtype(), DType::Bool);

    // (a < b) AND (b < c) - both comparisons return Bool
    let result = UOp::try_and_op(cmp1, cmp2).unwrap();
    assert_eq!(result.dtype(), DType::Bool);
}

// =========================================================================
// Edge Cases
// =========================================================================

#[test]
fn test_comparison_with_zero_float() {
    let zero = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    let neg_zero = UOp::const_(DType::Float32, ConstValue::Float(-0.0));

    // In IEEE 754, +0.0 == -0.0
    let result = UOp::cmpeq(&zero, &neg_zero).unwrap();
    assert_eq!(result.dtype(), DType::Bool);
}

#[test]
fn test_comparison_preserves_dtype_independence() {
    // Verify that comparison result type doesn't depend on operand types
    let int_cmp =
        UOp::cmplt(&UOp::const_(DType::Int32, ConstValue::Int(1)), &UOp::const_(DType::Int32, ConstValue::Int(2)))
            .unwrap();

    let float_cmp = UOp::cmplt(
        &UOp::const_(DType::Float32, ConstValue::Float(1.0)),
        &UOp::const_(DType::Float32, ConstValue::Float(2.0)),
    )
    .unwrap();

    // Both should return Bool regardless of operand types
    assert_eq!(int_cmp.dtype(), DType::Bool);
    assert_eq!(float_cmp.dtype(), DType::Bool);
    assert_eq!(int_cmp.dtype(), float_cmp.dtype());
}
