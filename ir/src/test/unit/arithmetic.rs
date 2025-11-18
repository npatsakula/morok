//! Arithmetic operation tests.
//!
//! Tests all arithmetic operations including basic ops, type promotion, and error handling.

use morok_dtype::DType;

use crate::{ConstValue, UOp, error::Error};

// =========================================================================
// Basic Arithmetic Operations
// =========================================================================

#[test]
fn test_add_same_type() {
    let a = UOp::const_(DType::Int32, ConstValue::Int(5));
    let b = UOp::const_(DType::Int32, ConstValue::Int(3));

    let result = UOp::try_add_op(a, b).unwrap();
    assert_eq!(result.dtype(), DType::Int32);
}

#[test]
fn test_sub_same_type() {
    let a = UOp::const_(DType::Float32, ConstValue::Float(10.0));
    let b = UOp::const_(DType::Float32, ConstValue::Float(3.0));

    let result = UOp::try_sub_op(a, b).unwrap();
    assert_eq!(result.dtype(), DType::Float32);
}

#[test]
fn test_mul_same_type() {
    let a = UOp::const_(DType::Int32, ConstValue::Int(4));
    let b = UOp::const_(DType::Int32, ConstValue::Int(5));

    let result = UOp::try_mul_op(a, b).unwrap();
    assert_eq!(result.dtype(), DType::Int32);
}

#[test]
fn test_mod_same_type() {
    let a = UOp::const_(DType::Int32, ConstValue::Int(10));
    let b = UOp::const_(DType::Int32, ConstValue::Int(3));

    let result = UOp::try_mod_op(a, b).unwrap();
    assert_eq!(result.dtype(), DType::Int32);
}

#[test]
fn test_idiv_same_type() {
    let a = UOp::const_(DType::Int32, ConstValue::Int(10));
    let b = UOp::const_(DType::Int32, ConstValue::Int(3));

    let result = UOp::try_idiv_op(a, b).unwrap();
    assert_eq!(result.dtype(), DType::Int32);
}

#[test]
fn test_fdiv_same_type() {
    let a = UOp::const_(DType::Float32, ConstValue::Float(10.0));
    let b = UOp::const_(DType::Float32, ConstValue::Float(3.0));

    let result = UOp::try_fdiv_op(a, b).unwrap();
    assert_eq!(result.dtype(), DType::Float32);
}

#[test]
fn test_max_same_type() {
    let a = UOp::const_(DType::Int32, ConstValue::Int(10));
    let b = UOp::const_(DType::Int32, ConstValue::Int(20));

    let result = UOp::try_max_op(a, b).unwrap();
    assert_eq!(result.dtype(), DType::Int32);
}

#[test]
fn test_pow_same_type() {
    let a = UOp::const_(DType::Float32, ConstValue::Float(2.0));
    let b = UOp::const_(DType::Float32, ConstValue::Float(3.0));

    let result = UOp::try_pow_op(a, b).unwrap();
    assert_eq!(result.dtype(), DType::Float32);
}

// =========================================================================
// Unary Operations
// =========================================================================

#[test]
fn test_neg_int() {
    let val = UOp::const_(DType::Int32, ConstValue::Int(5));
    let result = UOp::neg_op(val);
    assert_eq!(result.dtype(), DType::Int32);
}

#[test]
fn test_neg_float() {
    let val = UOp::const_(DType::Float32, ConstValue::Float(std::f32::consts::PI as f64));
    let result = UOp::neg_op(val);
    assert_eq!(result.dtype(), DType::Float32);
}

// =========================================================================
// Type Promotion Tests
// =========================================================================

#[test]
fn test_add_type_promotion_int_to_float() {
    let int_val = UOp::const_(DType::Int32, ConstValue::Int(5));
    let float_val = UOp::const_(DType::Float32, ConstValue::Float(std::f32::consts::PI as f64));

    let result = UOp::try_add_op(int_val, float_val).unwrap();
    // Int32 should promote to Float32
    assert_eq!(result.dtype(), DType::Float32);
}

#[test]
fn test_mul_type_promotion_smaller_to_larger() {
    let small = UOp::const_(DType::Int8, ConstValue::Int(5));
    let large = UOp::const_(DType::Int32, ConstValue::Int(10));

    let result = UOp::try_mul_op(small, large).unwrap();
    // Int8 should promote to Int32
    assert_eq!(result.dtype(), DType::Int32);
}

#[test]
fn test_sub_type_promotion_uint_to_int() {
    let uint_val = UOp::const_(DType::UInt8, ConstValue::UInt(5));
    let int_val = UOp::const_(DType::Int32, ConstValue::Int(10));

    let result = UOp::try_sub_op(uint_val, int_val).unwrap();
    // UInt8 should promote to Int32
    assert_eq!(result.dtype(), DType::Int32);
}

// =========================================================================
// Division by Zero Tests
// =========================================================================

#[test]
fn test_idiv_by_zero() {
    let numerator = UOp::const_(DType::Int32, ConstValue::Int(10));
    let zero = UOp::const_(DType::Int32, ConstValue::Int(0));

    let result = UOp::try_idiv_op(numerator, zero);
    assert!(matches!(result, Err(Error::DivisionByZero)));
}

#[test]
fn test_fdiv_by_zero() {
    let numerator = UOp::const_(DType::Float32, ConstValue::Float(10.0));
    let zero = UOp::const_(DType::Float32, ConstValue::Float(0.0));

    let result = UOp::try_fdiv_op(numerator, zero);
    assert!(matches!(result, Err(Error::DivisionByZero)));
}

#[test]
fn test_mod_by_zero() {
    let numerator = UOp::const_(DType::Int32, ConstValue::Int(10));
    let zero = UOp::const_(DType::Int32, ConstValue::Int(0));

    let result = UOp::try_mod_op(numerator, zero);
    assert!(matches!(result, Err(Error::DivisionByZero)));
}

// =========================================================================
// Void Type Error Tests
// =========================================================================

#[test]
fn test_add_void_type() {
    let void_val = UOp::const_(DType::Void, ConstValue::Int(0));
    let int_val = UOp::const_(DType::Int32, ConstValue::Int(5));

    let result = UOp::try_add_op(void_val, int_val);
    assert!(matches!(result, Err(Error::VoidTypeInOp)));
}

#[test]
fn test_mul_void_type() {
    let void_val = UOp::const_(DType::Void, ConstValue::Int(0));
    let float_val = UOp::const_(DType::Float32, ConstValue::Float(std::f32::consts::PI as f64));

    let result = UOp::try_mul_op(void_val, float_val);
    assert!(matches!(result, Err(Error::VoidTypeInOp)));
}

// =========================================================================
// Mixed Type Tests
// =========================================================================

#[test]
fn test_add_bool_and_int() {
    let bool_val = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let int_val = UOp::const_(DType::Int32, ConstValue::Int(5));

    let result = UOp::try_add_op(bool_val, int_val).unwrap();
    // Bool should promote to Int32
    assert_eq!(result.dtype(), DType::Int32);
}

#[test]
fn test_mul_different_float_types() {
    let f16 = UOp::const_(DType::Float16, ConstValue::Float(2.0));
    let f32 = UOp::const_(DType::Float32, ConstValue::Float(3.0));

    let result = UOp::try_mul_op(f16, f32).unwrap();
    // Float16 should promote to Float32
    assert_eq!(result.dtype(), DType::Float32);
}

// =========================================================================
// Operation Chaining Tests
// =========================================================================

#[test]
fn test_chained_operations() {
    let a = UOp::const_(DType::Int32, ConstValue::Int(10));
    let b = UOp::const_(DType::Int32, ConstValue::Int(5));
    let c = UOp::const_(DType::Int32, ConstValue::Int(2));

    // (a + b) * c
    let sum = UOp::try_add_op(a, b).unwrap();
    let product = UOp::try_mul_op(sum, c).unwrap();
    assert_eq!(product.dtype(), DType::Int32);
}

#[test]
fn test_chained_with_promotion() {
    let int_val = UOp::const_(DType::Int32, ConstValue::Int(10));
    let float_val = UOp::const_(DType::Float32, ConstValue::Float(2.5));

    // int + float -> Float32
    let sum = UOp::try_add_op(int_val.clone(), float_val).unwrap();
    assert_eq!(sum.dtype(), DType::Float32);

    // Float32 * Int32 -> Float32
    let product = UOp::try_mul_op(sum, int_val).unwrap();
    assert_eq!(product.dtype(), DType::Float32);
}
