use crate::*;
use morok_dtype::DType;

// Trigonometric tests
#[test]
fn test_sin_basic() {
    let t = Tensor::from_slice([0.0f32, 1.0]);
    let result = t.sin().unwrap();
    assert_eq!(result.uop.dtype(), DType::Float32);
}

#[test]
fn test_cos_basic() {
    let t = Tensor::from_slice([0.0f32, 1.0]);
    let result = t.cos().unwrap();
    assert_eq!(result.uop.dtype(), DType::Float32);
}

#[test]
fn test_tan_basic() {
    let t = Tensor::from_slice([0.0f32, 1.0]);
    let result = t.tan().unwrap();
    assert_eq!(result.uop.dtype(), DType::Float32);
}

#[test]
fn test_trig_error_on_int() {
    let t = Tensor::from_slice([1i32, 2, 3]);
    assert!(t.sin().is_err());
    assert!(t.cos().is_err());
    assert!(t.tan().is_err());
}

// Rounding tests
#[test]
fn test_floor_basic() {
    let t = Tensor::from_slice([1.2f32, -1.2, 2.8]);
    let result = t.floor().unwrap();
    assert_eq!(result.uop.dtype(), DType::Float32);
}

#[test]
fn test_ceil_basic() {
    let t = Tensor::from_slice([1.2f32, -1.2, 2.8]);
    let result = t.ceil().unwrap();
    assert_eq!(result.uop.dtype(), DType::Float32);
}

#[test]
fn test_round_basic() {
    let t = Tensor::from_slice([1.2f32, 1.5, 2.5]);
    let result = t.round().unwrap();
    assert_eq!(result.uop.dtype(), DType::Float32);
}

#[test]
fn test_trunc_basic() {
    let t = Tensor::from_slice([1.2f32, -1.2, 2.8]);
    let result = t.trunc().unwrap();
    assert_eq!(result.uop.dtype(), DType::Float32);
}

#[test]
fn test_rounding_on_int() {
    let t = Tensor::from_slice([1i32, 2, 3]);
    // Rounding operations should work on integers (no-op)
    assert!(t.floor().is_ok());
    assert!(t.ceil().is_ok());
    assert!(t.round().is_ok());
    assert!(t.trunc().is_ok());
}

// Advanced math tests
#[test]
fn test_erf_basic() {
    let t = Tensor::from_slice([0.0f32, 1.0, -1.0]);
    let result = t.erf().unwrap();
    assert_eq!(result.uop.dtype(), DType::Float32);
}

#[test]
fn test_erf_error_on_int() {
    let t = Tensor::from_slice([1i32, 2, 3]);
    assert!(t.erf().is_err());
}

#[test]
fn test_reciprocal_basic() {
    let t = Tensor::from_slice([1.0f32, 2.0, 4.0]);
    let result = t.reciprocal().unwrap();
    assert_eq!(result.uop.dtype(), DType::Float32);
}

#[test]
fn test_square_basic() {
    let t = Tensor::from_slice([1.0f32, 2.0, -3.0]);
    let result = t.square().unwrap();
    assert_eq!(result.uop.dtype(), DType::Float32);
}

#[test]
fn test_square_int() {
    let t = Tensor::from_slice([1i32, 2, -3]);
    let result = t.square().unwrap();
    assert_eq!(result.uop.dtype(), DType::Int32);
}

#[test]
fn test_sign_basic() {
    let t = Tensor::from_slice([-5.0f32, 0.0, 3.0]);
    let result = t.sign().unwrap();
    assert_eq!(result.uop.dtype(), DType::Float32);
}

#[test]
fn test_sign_int() {
    let t = Tensor::from_slice([-5i32, 0, 3]);
    let result = t.sign().unwrap();
    assert_eq!(result.uop.dtype(), DType::Int32);
}
