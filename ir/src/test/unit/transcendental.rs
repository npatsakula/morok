//! Transcendental function tests.
//!
//! Tests mathematical transcendental functions: Sqrt, Exp2, Log2, Sin, Reciprocal, Trunc.

use morok_dtype::DType;

use crate::{ConstValue, UOp};

// =========================================================================
// Square Root (Sqrt) Tests
// =========================================================================

#[test]
fn test_sqrt_positive() {
    let val = UOp::const_(DType::Float32, ConstValue::Float(9.0));
    let result = UOp::sqrt(&val).unwrap();
    assert_eq!(result.dtype(), DType::Float32);
}

#[test]
fn test_sqrt_zero() {
    let zero = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    let result = UOp::sqrt(&zero).unwrap();
    // sqrt(0) = 0
    assert_eq!(result.dtype(), DType::Float32);
}

#[test]
fn test_sqrt_one() {
    let one = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let result = UOp::sqrt(&one).unwrap();
    // sqrt(1) = 1
    assert_eq!(result.dtype(), DType::Float32);
}

#[test]
fn test_sqrt_preserves_dtype() {
    let val_f32 = UOp::const_(DType::Float32, ConstValue::Float(4.0));
    let result_f32 = UOp::sqrt(&val_f32).unwrap();
    assert_eq!(result_f32.dtype(), DType::Float32);

    let val_f64 = UOp::const_(DType::Float64, ConstValue::Float(4.0));
    let result_f64 = UOp::sqrt(&val_f64).unwrap();
    assert_eq!(result_f64.dtype(), DType::Float64);
}

// =========================================================================
// Exponential Base-2 (Exp2) Tests
// =========================================================================

#[test]
fn test_exp2_zero() {
    let zero = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    let result = UOp::exp2(&zero).unwrap();
    // 2^0 = 1
    assert_eq!(result.dtype(), DType::Float32);
}

#[test]
fn test_exp2_positive() {
    let val = UOp::const_(DType::Float32, ConstValue::Float(3.0));
    let result = UOp::exp2(&val).unwrap();
    // 2^3 = 8
    assert_eq!(result.dtype(), DType::Float32);
}

#[test]
fn test_exp2_negative() {
    let val = UOp::const_(DType::Float32, ConstValue::Float(-2.0));
    let result = UOp::exp2(&val).unwrap();
    // 2^-2 = 0.25
    assert_eq!(result.dtype(), DType::Float32);
}

#[test]
fn test_exp2_one() {
    let one = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let result = UOp::exp2(&one).unwrap();
    // 2^1 = 2
    assert_eq!(result.dtype(), DType::Float32);
}

// =========================================================================
// Logarithm Base-2 (Log2) Tests
// =========================================================================

#[test]
fn test_log2_one() {
    let one = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let result = UOp::log2(&one).unwrap();
    // log2(1) = 0
    assert_eq!(result.dtype(), DType::Float32);
}

#[test]
fn test_log2_power_of_two() {
    let val = UOp::const_(DType::Float32, ConstValue::Float(8.0));
    let result = UOp::log2(&val).unwrap();
    // log2(8) = 3
    assert_eq!(result.dtype(), DType::Float32);
}

#[test]
fn test_log2_two() {
    let two = UOp::const_(DType::Float32, ConstValue::Float(2.0));
    let result = UOp::log2(&two).unwrap();
    // log2(2) = 1
    assert_eq!(result.dtype(), DType::Float32);
}

#[test]
fn test_log2_fractional() {
    let val = UOp::const_(DType::Float32, ConstValue::Float(0.5));
    let result = UOp::log2(&val).unwrap();
    // log2(0.5) = -1
    assert_eq!(result.dtype(), DType::Float32);
}

// =========================================================================
// Sine (Sin) Tests
// =========================================================================

#[test]
fn test_sin_zero() {
    let zero = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    let result = UOp::sin_op(zero).unwrap();
    // sin(0) = 0
    assert_eq!(result.dtype(), DType::Float32);
}

#[test]
fn test_sin_positive() {
    let val = UOp::const_(DType::Float32, ConstValue::Float(1.57)); // ~π/2
    let result = UOp::sin_op(val).unwrap();
    // sin(π/2) ≈ 1
    assert_eq!(result.dtype(), DType::Float32);
}

#[test]
fn test_sin_negative() {
    let val = UOp::const_(DType::Float32, ConstValue::Float(-1.57)); // ~-π/2
    let result = UOp::sin_op(val).unwrap();
    // sin(-π/2) ≈ -1
    assert_eq!(result.dtype(), DType::Float32);
}

#[test]
fn test_sin_preserves_dtype() {
    let val_f32 = UOp::const_(DType::Float32, ConstValue::Float(0.5));
    let result_f32 = UOp::sin_op(val_f32).unwrap();
    assert_eq!(result_f32.dtype(), DType::Float32);

    let val_f64 = UOp::const_(DType::Float64, ConstValue::Float(0.5));
    let result_f64 = UOp::sin_op(val_f64).unwrap();
    assert_eq!(result_f64.dtype(), DType::Float64);
}

// =========================================================================
// Reciprocal (1/x) Tests
// =========================================================================

#[test]
fn test_reciprocal_one() {
    let one = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let result = UOp::reciprocal_op(one);
    // 1/1 = 1
    assert_eq!(result.dtype(), DType::Float32);
}

#[test]
fn test_reciprocal_two() {
    let two = UOp::const_(DType::Float32, ConstValue::Float(2.0));
    let result = UOp::reciprocal_op(two);
    // 1/2 = 0.5
    assert_eq!(result.dtype(), DType::Float32);
}

#[test]
fn test_reciprocal_negative() {
    let val = UOp::const_(DType::Float32, ConstValue::Float(-4.0));
    let result = UOp::reciprocal_op(val);
    // 1/(-4) = -0.25
    assert_eq!(result.dtype(), DType::Float32);
}

#[test]
fn test_reciprocal_fractional() {
    let val = UOp::const_(DType::Float32, ConstValue::Float(0.5));
    let result = UOp::reciprocal_op(val);
    // 1/0.5 = 2
    assert_eq!(result.dtype(), DType::Float32);
}

// =========================================================================
// Truncate (Trunc) Tests
// =========================================================================

#[test]
fn test_trunc_positive() {
    let val = UOp::const_(DType::Float32, ConstValue::Float(3.7));
    let result = UOp::trunc_op(val);
    // trunc(3.7) = 3.0
    assert_eq!(result.dtype(), DType::Float32);
}

#[test]
fn test_trunc_negative() {
    let val = UOp::const_(DType::Float32, ConstValue::Float(-3.7));
    let result = UOp::trunc_op(val);
    // trunc(-3.7) = -3.0 (towards zero)
    assert_eq!(result.dtype(), DType::Float32);
}

#[test]
fn test_trunc_zero() {
    let zero = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    let result = UOp::trunc_op(zero);
    // trunc(0) = 0
    assert_eq!(result.dtype(), DType::Float32);
}

#[test]
fn test_trunc_integer_value() {
    let val = UOp::const_(DType::Float32, ConstValue::Float(5.0));
    let result = UOp::trunc_op(val);
    // trunc(5.0) = 5.0
    assert_eq!(result.dtype(), DType::Float32);
}

#[test]
fn test_trunc_near_zero() {
    let pos_val = UOp::const_(DType::Float32, ConstValue::Float(0.9));
    let result_pos = UOp::trunc_op(pos_val);
    // trunc(0.9) = 0.0
    assert_eq!(result_pos.dtype(), DType::Float32);

    let neg_val = UOp::const_(DType::Float32, ConstValue::Float(-0.9));
    let result_neg = UOp::trunc_op(neg_val);
    // trunc(-0.9) = -0.0 (towards zero)
    assert_eq!(result_neg.dtype(), DType::Float32);
}

// =========================================================================
// Combined/Chained Operations
// =========================================================================

#[test]
fn test_sqrt_exp2_identity() {
    let val = UOp::const_(DType::Float32, ConstValue::Float(4.0));

    // log2(x)
    let log_val = UOp::log2(&val).unwrap();
    // exp2(log2(x)) should be x
    let result = UOp::exp2(&log_val).unwrap();

    assert_eq!(result.dtype(), DType::Float32);
}

#[test]
fn test_reciprocal_twice() {
    let val = UOp::const_(DType::Float32, ConstValue::Float(5.0));

    // 1/x
    let recip1 = UOp::reciprocal_op(val.clone());
    // 1/(1/x) = x
    let recip2 = UOp::reciprocal_op(recip1);

    assert_eq!(recip2.dtype(), DType::Float32);
}

#[test]
fn test_transcendental_composition() {
    let val = UOp::const_(DType::Float32, ConstValue::Float(2.0));

    // sqrt(exp2(x))
    let exp_val = UOp::exp2(&val).unwrap();
    let result = UOp::sqrt(&exp_val).unwrap();

    assert_eq!(result.dtype(), DType::Float32);
}
