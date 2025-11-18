//! Bitwise operation tests.
//!
//! Tests all bitwise operations including and, or, xor, shifts, and dtype validation.

use morok_dtype::DType;

use crate::{ConstValue, UOp, error::Error};

// =========================================================================
// Basic Bitwise Operations with Int Types
// =========================================================================

#[test]
fn test_and_int32() {
    let a = UOp::const_(DType::Int32, ConstValue::Int(0b1010));
    let b = UOp::const_(DType::Int32, ConstValue::Int(0b1100));

    let result = UOp::try_and_op(a, b).unwrap();
    assert_eq!(result.dtype(), DType::Int32);
}

#[test]
fn test_or_int32() {
    let a = UOp::const_(DType::Int32, ConstValue::Int(0b1010));
    let b = UOp::const_(DType::Int32, ConstValue::Int(0b1100));

    let result = UOp::try_or_op(a, b).unwrap();
    assert_eq!(result.dtype(), DType::Int32);
}

#[test]
fn test_xor_int32() {
    let a = UOp::const_(DType::Int32, ConstValue::Int(0b1010));
    let b = UOp::const_(DType::Int32, ConstValue::Int(0b1100));

    let result = UOp::try_xor_op(a, b).unwrap();
    assert_eq!(result.dtype(), DType::Int32);
}

// =========================================================================
// Shift Operations
// =========================================================================

#[test]
fn test_shl_int32() {
    let value = UOp::const_(DType::Int32, ConstValue::Int(8));
    let shift = UOp::const_(DType::Int32, ConstValue::Int(2));

    let result = UOp::try_shl_op(value, shift).unwrap();
    assert_eq!(result.dtype(), DType::Int32);
}

#[test]
fn test_shr_int32() {
    let value = UOp::const_(DType::Int32, ConstValue::Int(32));
    let shift = UOp::const_(DType::Int32, ConstValue::Int(2));

    let result = UOp::try_shr_op(value, shift).unwrap();
    assert_eq!(result.dtype(), DType::Int32);
}

#[test]
fn test_shift_preserves_lhs_dtype() {
    let value = UOp::const_(DType::Int64, ConstValue::Int(100));
    let shift = UOp::const_(DType::Int32, ConstValue::Int(3));

    // Shift should preserve LHS dtype (Int64), not promote
    let result = UOp::try_shl_op(value, shift).unwrap();
    assert_eq!(result.dtype(), DType::Int64);
}

// =========================================================================
// Boolean Type Operations
// =========================================================================

#[test]
fn test_and_bool() {
    let a = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let b = UOp::const_(DType::Bool, ConstValue::Bool(false));

    let result = UOp::try_and_op(a, b).unwrap();
    assert_eq!(result.dtype(), DType::Bool);
}

#[test]
fn test_or_bool() {
    let a = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let b = UOp::const_(DType::Bool, ConstValue::Bool(false));

    let result = UOp::try_or_op(a, b).unwrap();
    assert_eq!(result.dtype(), DType::Bool);
}

#[test]
fn test_xor_bool() {
    let a = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let b = UOp::const_(DType::Bool, ConstValue::Bool(false));

    let result = UOp::try_xor_op(a, b).unwrap();
    assert_eq!(result.dtype(), DType::Bool);
}

// =========================================================================
// Unsigned Integer Operations
// =========================================================================

#[test]
fn test_and_uint32() {
    let a = UOp::const_(DType::UInt32, ConstValue::UInt(15));
    let b = UOp::const_(DType::UInt32, ConstValue::UInt(7));

    let result = UOp::try_and_op(a, b).unwrap();
    // UInt32 promotes to Int64 in type promotion
    assert_eq!(result.dtype(), DType::Int64);
}

#[test]
fn test_shl_uint64() {
    let value = UOp::const_(DType::UInt64, ConstValue::UInt(1));
    let shift = UOp::const_(DType::UInt32, ConstValue::UInt(10));

    let result = UOp::try_shl_op(value, shift).unwrap();
    assert_eq!(result.dtype(), DType::UInt64);
}

// =========================================================================
// Type Promotion in Bitwise Ops
// =========================================================================

#[test]
fn test_and_type_promotion() {
    let small = UOp::const_(DType::Int8, ConstValue::Int(15));
    let large = UOp::const_(DType::Int32, ConstValue::Int(255));

    let result = UOp::try_and_op(small, large).unwrap();
    // Int8 should promote to Int32
    assert_eq!(result.dtype(), DType::Int32);
}

#[test]
fn test_or_mixed_int_types() {
    let i16 = UOp::const_(DType::Int16, ConstValue::Int(100));
    let i32 = UOp::const_(DType::Int32, ConstValue::Int(200));

    let result = UOp::try_or_op(i16, i32).unwrap();
    // Int16 should promote to Int32
    assert_eq!(result.dtype(), DType::Int32);
}

// =========================================================================
// Invalid DType Error Tests
// =========================================================================

#[test]
fn test_and_float_error() {
    let float_val = UOp::const_(DType::Float32, ConstValue::Float(std::f32::consts::PI as f64));
    let int_val = UOp::const_(DType::Int32, ConstValue::Int(5));

    let result = UOp::try_and_op(float_val, int_val);
    assert!(matches!(result, Err(Error::InvalidDTypeForOp { operation: "try_and_op", .. })));
}

#[test]
fn test_or_float_error() {
    let float_val = UOp::const_(DType::Float32, ConstValue::Float(2.5));
    let int_val = UOp::const_(DType::Int32, ConstValue::Int(10));

    let result = UOp::try_or_op(float_val, int_val);
    assert!(matches!(result, Err(Error::InvalidDTypeForOp { operation: "try_or_op", .. })));
}

#[test]
fn test_xor_float_error() {
    let float_val = UOp::const_(DType::Float64, ConstValue::Float(1.5));
    let int_val = UOp::const_(DType::Int64, ConstValue::Int(7));

    let result = UOp::try_xor_op(float_val, int_val);
    assert!(matches!(result, Err(Error::InvalidDTypeForOp { operation: "try_xor_op", .. })));
}

#[test]
fn test_shl_float_error() {
    let float_val = UOp::const_(DType::Float32, ConstValue::Float(8.0));
    let shift = UOp::const_(DType::Int32, ConstValue::Int(2));

    let result = UOp::try_shl_op(float_val, shift);
    assert!(matches!(result, Err(Error::InvalidDTypeForOp { operation: "try_shl_op", .. })));
}

#[test]
fn test_shr_float_error() {
    let float_val = UOp::const_(DType::Float32, ConstValue::Float(16.0));
    let shift = UOp::const_(DType::Int32, ConstValue::Int(1));

    let result = UOp::try_shr_op(float_val, shift);
    assert!(matches!(result, Err(Error::InvalidDTypeForOp { operation: "try_shr_op", .. })));
}

#[test]
fn test_and_both_floats_error() {
    let f1 = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let f2 = UOp::const_(DType::Float32, ConstValue::Float(2.0));

    let result = UOp::try_and_op(f1, f2);
    assert!(matches!(result, Err(Error::InvalidDTypeForOp { operation: "try_and_op", .. })));
}

// =========================================================================
// Mixed Bool and Int Operations
// =========================================================================

#[test]
fn test_and_bool_and_int() {
    let bool_val = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let int_val = UOp::const_(DType::Int32, ConstValue::Int(5));

    let result = UOp::try_and_op(bool_val, int_val).unwrap();
    // Bool should promote to Int32
    assert_eq!(result.dtype(), DType::Int32);
}

#[test]
fn test_or_int_and_bool() {
    let int_val = UOp::const_(DType::Int8, ConstValue::Int(7));
    let bool_val = UOp::const_(DType::Bool, ConstValue::Bool(false));

    let result = UOp::try_or_op(int_val, bool_val).unwrap();
    // Bool should promote to Int8
    assert_eq!(result.dtype(), DType::Int8);
}

// =========================================================================
// Edge Cases
// =========================================================================

#[test]
fn test_and_with_zero() {
    let val = UOp::const_(DType::Int32, ConstValue::Int(0xFF));
    let zero = UOp::const_(DType::Int32, ConstValue::Int(0));

    let result = UOp::try_and_op(val, zero).unwrap();
    assert_eq!(result.dtype(), DType::Int32);
}

#[test]
fn test_or_with_max() {
    let val = UOp::const_(DType::UInt8, ConstValue::UInt(0x55));
    let max = UOp::const_(DType::UInt8, ConstValue::UInt(0xFF));

    let result = UOp::try_or_op(val, max).unwrap();
    // UInt8 promotes to Int16 in type promotion
    assert_eq!(result.dtype(), DType::Int16);
}

#[test]
fn test_xor_with_self() {
    let val = UOp::const_(DType::Int32, ConstValue::Int(42));

    let result = UOp::try_xor_op(val.clone(), val).unwrap();
    assert_eq!(result.dtype(), DType::Int32);
}

#[test]
fn test_shl_by_zero() {
    let val = UOp::const_(DType::Int32, ConstValue::Int(100));
    let zero = UOp::const_(DType::Int32, ConstValue::Int(0));

    let result = UOp::try_shl_op(val, zero).unwrap();
    assert_eq!(result.dtype(), DType::Int32);
}
