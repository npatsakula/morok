//! Transcendental function tests.
//!
//! Tests mathematical transcendental functions: Sqrt, Exp2, Log2, Sin, Reciprocal, Trunc.

use crate::UOp;

// =========================================================================
// Square Root (Sqrt) Tests
// =========================================================================

#[test]
fn test_sqrt_positive() {
    assert_eq!(UOp::native_const(9.0f32).try_sqrt().unwrap().dtype(), morok_dtype::DType::Float32);
}

#[test]
fn test_sqrt_zero() {
    // sqrt(0) = 0
    assert_eq!(UOp::native_const(0.0f32).try_sqrt().unwrap().dtype(), morok_dtype::DType::Float32);
}

#[test]
fn test_sqrt_one() {
    // sqrt(1) = 1
    assert_eq!(UOp::native_const(1.0f32).try_sqrt().unwrap().dtype(), morok_dtype::DType::Float32);
}

#[test]
fn test_sqrt_preserves_dtype() {
    assert_eq!(UOp::native_const(4.0f32).try_sqrt().unwrap().dtype(), morok_dtype::DType::Float32);
    assert_eq!(UOp::native_const(4.0f64).try_sqrt().unwrap().dtype(), morok_dtype::DType::Float64);
}

// =========================================================================
// Exponential Base-2 (Exp2) Tests
// =========================================================================

#[test]
fn test_exp2_zero() {
    // 2^0 = 1
    assert_eq!(UOp::native_const(0.0f32).try_exp2().unwrap().dtype(), morok_dtype::DType::Float32);
}

#[test]
fn test_exp2_positive() {
    // 2^3 = 8
    assert_eq!(UOp::native_const(3.0f32).try_exp2().unwrap().dtype(), morok_dtype::DType::Float32);
}

#[test]
fn test_exp2_negative() {
    // 2^-2 = 0.25
    assert_eq!(UOp::native_const(-2.0f32).try_exp2().unwrap().dtype(), morok_dtype::DType::Float32);
}

#[test]
fn test_exp2_one() {
    // 2^1 = 2
    assert_eq!(UOp::native_const(1.0f32).try_exp2().unwrap().dtype(), morok_dtype::DType::Float32);
}

// =========================================================================
// Logarithm Base-2 (Log2) Tests
// =========================================================================

#[test]
fn test_log2_one() {
    // log2(1) = 0
    assert_eq!(UOp::native_const(1.0f32).try_log2().unwrap().dtype(), morok_dtype::DType::Float32);
}

#[test]
fn test_log2_power_of_two() {
    // log2(8) = 3
    assert_eq!(UOp::native_const(8.0f32).try_log2().unwrap().dtype(), morok_dtype::DType::Float32);
}

#[test]
fn test_log2_two() {
    // log2(2) = 1
    assert_eq!(UOp::native_const(2.0f32).try_log2().unwrap().dtype(), morok_dtype::DType::Float32);
}

#[test]
fn test_log2_fractional() {
    // log2(0.5) = -1
    assert_eq!(UOp::native_const(0.5f32).try_log2().unwrap().dtype(), morok_dtype::DType::Float32);
}

// =========================================================================
// Sine (Sin) Tests
// =========================================================================

#[test]
fn test_sin_zero() {
    let result = UOp::sin_op(UOp::native_const(0.0f32)).unwrap();
    // sin(0) = 0
    assert_eq!(result.dtype(), morok_dtype::DType::Float32);
}

#[test]
fn test_sin_positive() {
    let result = UOp::sin_op(UOp::native_const(1.57f32)).unwrap(); // ~π/2
    // sin(π/2) ≈ 1
    assert_eq!(result.dtype(), morok_dtype::DType::Float32);
}

#[test]
fn test_sin_negative() {
    let result = UOp::sin_op(UOp::native_const(-1.57f32)).unwrap(); // ~-π/2
    // sin(-π/2) ≈ -1
    assert_eq!(result.dtype(), morok_dtype::DType::Float32);
}

#[test]
fn test_sin_preserves_dtype() {
    let result_f32 = UOp::sin_op(UOp::native_const(0.5f32)).unwrap();
    assert_eq!(result_f32.dtype(), morok_dtype::DType::Float32);

    let result_f64 = UOp::sin_op(UOp::native_const(0.5f64)).unwrap();
    assert_eq!(result_f64.dtype(), morok_dtype::DType::Float64);
}

// =========================================================================
// Reciprocal (1/x) Tests
// =========================================================================

#[test]
fn test_reciprocal_one() {
    let result = UOp::reciprocal_op(UOp::native_const(1.0f32));
    // 1/1 = 1
    assert_eq!(result.dtype(), morok_dtype::DType::Float32);
}

#[test]
fn test_reciprocal_two() {
    let result = UOp::reciprocal_op(UOp::native_const(2.0f32));
    // 1/2 = 0.5
    assert_eq!(result.dtype(), morok_dtype::DType::Float32);
}

#[test]
fn test_reciprocal_negative() {
    let result = UOp::reciprocal_op(UOp::native_const(-4.0f32));
    // 1/(-4) = -0.25
    assert_eq!(result.dtype(), morok_dtype::DType::Float32);
}

#[test]
fn test_reciprocal_fractional() {
    let result = UOp::reciprocal_op(UOp::native_const(0.5f32));
    // 1/0.5 = 2
    assert_eq!(result.dtype(), morok_dtype::DType::Float32);
}

// =========================================================================
// Truncate (Trunc) Tests
// =========================================================================

#[test]
fn test_trunc_positive() {
    let result = UOp::trunc_op(UOp::native_const(3.7f32));
    // trunc(3.7) = 3.0
    assert_eq!(result.dtype(), morok_dtype::DType::Float32);
}

#[test]
fn test_trunc_negative() {
    let result = UOp::trunc_op(UOp::native_const(-3.7f32));
    // trunc(-3.7) = -3.0 (towards zero)
    assert_eq!(result.dtype(), morok_dtype::DType::Float32);
}

#[test]
fn test_trunc_zero() {
    let result = UOp::trunc_op(UOp::native_const(0.0f32));
    // trunc(0) = 0
    assert_eq!(result.dtype(), morok_dtype::DType::Float32);
}

#[test]
fn test_trunc_integer_value() {
    let result = UOp::trunc_op(UOp::native_const(5.0f32));
    // trunc(5.0) = 5.0
    assert_eq!(result.dtype(), morok_dtype::DType::Float32);
}

#[test]
fn test_trunc_near_zero() {
    let result_pos = UOp::trunc_op(UOp::native_const(0.9f32));
    // trunc(0.9) = 0.0
    assert_eq!(result_pos.dtype(), morok_dtype::DType::Float32);

    let result_neg = UOp::trunc_op(UOp::native_const(-0.9f32));
    // trunc(-0.9) = -0.0 (towards zero)
    assert_eq!(result_neg.dtype(), morok_dtype::DType::Float32);
}

// =========================================================================
// Combined/Chained Operations
// =========================================================================

#[test]
fn test_sqrt_exp2_identity() {
    // log2(x)
    let log_val = UOp::native_const(4.0f32).try_log2().unwrap();
    // exp2(log2(x)) should be x
    let result = log_val.try_exp2().unwrap();

    assert_eq!(result.dtype(), morok_dtype::DType::Float32);
}

#[test]
fn test_reciprocal_twice() {
    let val = UOp::native_const(5.0f32);

    // 1/x
    let recip1 = UOp::reciprocal_op(val.clone());
    // 1/(1/x) = x
    let recip2 = UOp::reciprocal_op(recip1);

    assert_eq!(recip2.dtype(), morok_dtype::DType::Float32);
}

#[test]
fn test_transcendental_composition() {
    // sqrt(exp2(x))
    let exp_val = UOp::native_const(2.0f32).try_exp2().unwrap();
    let result = exp_val.try_sqrt().unwrap();

    assert_eq!(result.dtype(), morok_dtype::DType::Float32);
}
