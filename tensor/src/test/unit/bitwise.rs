use crate::*;
use morok_dtype::DType;

#[test]
fn test_bitwise_and_basic() {
    let a = Tensor::from_slice([0b1010i32, 0b1100]);
    let b = Tensor::from_slice([0b1100i32, 0b0011]);
    let result = a.bitwise_and(&b).unwrap();
    assert_eq!(result.uop.dtype(), DType::Int32);
}

#[test]
fn test_bitwise_or_basic() {
    let a = Tensor::from_slice([0b1010i32, 0b1100]);
    let b = Tensor::from_slice([0b1100i32, 0b0011]);
    let result = a.bitwise_or(&b).unwrap();
    assert_eq!(result.uop.dtype(), DType::Int32);
}

#[test]
fn test_bitwise_xor_basic() {
    let a = Tensor::from_slice([0b1010i32, 0b1100]);
    let b = Tensor::from_slice([0b1100i32, 0b0011]);
    let result = a.bitwise_xor(&b).unwrap();
    assert_eq!(result.uop.dtype(), DType::Int32);
}

#[test]
fn test_lshift_basic() {
    let t = Tensor::from_slice([1i32, 2, 3]);
    let shift = Tensor::from_slice([1i32, 2, 3]);
    let result = t.lshift(&shift).unwrap();
    assert_eq!(result.uop.dtype(), DType::Int32);
}

#[test]
fn test_rshift_basic() {
    let t = Tensor::from_slice([8i32, 16, 24]);
    let shift = Tensor::from_slice([1i32, 2, 3]);
    let result = t.rshift(&shift).unwrap();
    assert_eq!(result.uop.dtype(), DType::Int32);
}

#[test]
fn test_bitwise_bool() {
    let a = Tensor::from_slice([true, false, true]);
    let b = Tensor::from_slice([true, true, false]);
    let result = a.bitwise_and(&b).unwrap();
    assert_eq!(result.uop.dtype(), DType::Bool);
}

#[test]
fn test_bitwise_error_on_float() {
    let a = Tensor::from_slice([1.0f32, 2.0]);
    let b = Tensor::from_slice([3.0f32, 4.0]);
    let result = a.bitwise_and(&b);
    assert!(result.is_err());
}
