//! Arithmetic operation tests.
//!
//! Tests all arithmetic operations including basic ops, type promotion, and error handling.

use std::f32::consts::PI;

use morok_dtype::DType;

use crate::{ConstValue, UOp, error::Error}; // ConstValue kept for Void, Float16, i8, u8

// =========================================================================
// Basic Arithmetic Operations
// =========================================================================

#[test]
fn test_add_same_type() {
    assert_eq!(UOp::native_const(5i32).try_add_op(&UOp::native_const(3i32)).unwrap().dtype(), DType::Int32);
}

#[test]
fn test_sub_same_type() {
    assert_eq!(UOp::native_const(10.0f32).try_sub_op(&UOp::native_const(3.0f32)).unwrap().dtype(), DType::Float32);
}

#[test]
fn test_mul_same_type() {
    assert_eq!(UOp::native_const(4i32).try_mul_op(&UOp::native_const(5i32)).unwrap().dtype(), DType::Int32);
}

#[test]
fn test_mod_same_type() {
    assert_eq!(UOp::native_const(10i32).try_mod_op(&UOp::native_const(3i32)).unwrap().dtype(), DType::Int32);
}

#[test]
fn test_idiv_same_type() {
    assert_eq!(UOp::native_const(10i32).try_idiv_op(&UOp::native_const(3i32)).unwrap().dtype(), DType::Int32);
}

#[test]
fn test_fdiv_same_type() {
    assert_eq!(UOp::native_const(10.0f32).try_fdiv_op(&UOp::native_const(3.0f32)).unwrap().dtype(), DType::Float32);
}

#[test]
fn test_max_same_type() {
    assert_eq!(UOp::native_const(10i32).try_max_op(&UOp::native_const(20i32)).unwrap().dtype(), DType::Int32);
}

#[test]
fn test_pow_same_type() {
    assert_eq!(UOp::native_const(2.0f32).try_pow_op(&UOp::native_const(3.0f32)).unwrap().dtype(), DType::Float32);
}

// =========================================================================
// Unary Operations
// =========================================================================

#[test]
fn test_neg_int() {
    let result = UOp::neg_op(UOp::native_const(5i32));
    assert_eq!(result.dtype(), DType::Int32);
}

#[test]
fn test_neg_float() {
    let result = UOp::neg_op(UOp::native_const(PI));
    assert_eq!(result.dtype(), DType::Float32);
}

// =========================================================================
// Type Promotion Tests
// =========================================================================

#[test]
fn test_add_type_promotion_int_to_float() {
    let int_val = UOp::native_const(5i32);
    let float_val = UOp::native_const(PI);

    let result = int_val.try_add_op(&float_val).unwrap();
    // Int32 should promote to Float32
    assert_eq!(result.dtype(), DType::Float32);
}

#[test]
fn test_mul_type_promotion_smaller_to_larger() {
    let small = UOp::const_(DType::Int8, ConstValue::Int(5));
    let large = UOp::native_const(10i32);

    let result = small.try_mul_op(&large).unwrap();
    // Int8 should promote to Int32
    assert_eq!(result.dtype(), DType::Int32);
}

#[test]
fn test_sub_type_promotion_uint_to_int() {
    let uint_val = UOp::const_(DType::UInt8, ConstValue::UInt(5));
    let int_val = UOp::native_const(10i32);

    let result = uint_val.try_sub_op(&int_val).unwrap();
    // UInt8 should promote to Int32
    assert_eq!(result.dtype(), DType::Int32);
}

// =========================================================================
// Division by Zero Tests
// =========================================================================

#[test]
fn test_idiv_by_zero() {
    let numerator = UOp::native_const(10i32);
    let zero = UOp::native_const(0i32);

    let result = numerator.try_idiv_op(&zero);
    assert!(matches!(result, Err(Error::DivisionByZero)));
}

#[test]
fn test_fdiv_by_zero() {
    let numerator = UOp::native_const(10.0f32);
    let zero = UOp::native_const(0.0f32);

    let result = numerator.try_fdiv_op(&zero);
    assert!(matches!(result, Err(Error::DivisionByZero)));
}

#[test]
fn test_mod_by_zero() {
    let numerator = UOp::native_const(10i32);
    let zero = UOp::native_const(0i32);

    let result = numerator.try_mod_op(&zero);
    assert!(matches!(result, Err(Error::DivisionByZero)));
}

// =========================================================================
// Void Type Error Tests
// =========================================================================

#[test]
fn test_add_void_type() {
    let void_val = UOp::const_(DType::Void, ConstValue::Int(0));
    let int_val = UOp::const_(DType::Int32, ConstValue::Int(5));

    let result = void_val.try_add_op(&int_val);
    assert!(matches!(result, Err(Error::VoidTypeInOp)));
}

#[test]
fn test_mul_void_type() {
    let void_val = UOp::const_(DType::Void, ConstValue::Int(0));
    let float_val = UOp::const_(DType::Float32, ConstValue::Float(PI as f64));

    let result = void_val.try_mul_op(&float_val);
    assert!(matches!(result, Err(Error::VoidTypeInOp)));
}

// =========================================================================
// Mixed Type Tests
// =========================================================================

#[test]
fn test_add_bool_and_int() {
    let bool_val = UOp::native_const(true);
    let int_val = UOp::native_const(5i32);

    let result = bool_val.try_add_op(&int_val).unwrap();
    // Bool should promote to Int32
    assert_eq!(result.dtype(), DType::Int32);
}

#[test]
fn test_mul_different_float_types() {
    let f16 = UOp::const_(DType::Float16, ConstValue::Float(2.0));
    let f32 = UOp::const_(DType::Float32, ConstValue::Float(3.0));

    let result = f16.try_mul_op(&f32).unwrap();
    // Float16 should promote to Float32
    assert_eq!(result.dtype(), DType::Float32);
}

// =========================================================================
// Operation Chaining Tests
// =========================================================================

#[test]
fn test_chained_operations() {
    let a = UOp::native_const(10i32);
    let b = UOp::native_const(5i32);
    let c = UOp::native_const(2i32);

    // (a + b) * c
    let sum = a.try_add_op(&b).unwrap();
    let product = sum.try_mul_op(&c).unwrap();
    assert_eq!(product.dtype(), DType::Int32);
}

#[test]
fn test_chained_with_promotion() {
    let int_val = UOp::native_const(10i32);
    let float_val = UOp::native_const(2.5f32);

    // int + float -> Float32
    let sum = int_val.try_add_op(&float_val).unwrap();
    assert_eq!(sum.dtype(), DType::Float32);

    // Float32 * Int32 -> Float32
    let product = sum.try_mul_op(&int_val).unwrap();
    assert_eq!(product.dtype(), DType::Float32);
}
