use crate::*;
use morok_dtype::DType;

#[test]
fn test_relu_basic() {
    let x = Tensor::from_slice([-2.0f32, -1.0, 0.0, 1.0, 2.0]);
    let y = x.relu();
    if let Err(e) = &y {
        eprintln!("ReLU error: {:?}", e);
    }
    assert!(y.is_ok());

    // Verify dtype preserved
    assert_eq!(y.unwrap().uop().dtype(), DType::Float32);
}

#[test]
fn test_sigmoid_basic() {
    let x = Tensor::from_slice([-2.0f32, -1.0, 0.0, 1.0, 2.0]);
    let y = x.sigmoid();
    assert!(y.is_ok());

    assert_eq!(y.unwrap().uop().dtype(), DType::Float32);
}

#[test]
fn test_tanh_basic() {
    let x = Tensor::from_slice([-2.0f32, -1.0, 0.0, 1.0, 2.0]);
    let y = x.tanh();
    assert!(y.is_ok());

    assert_eq!(y.unwrap().uop().dtype(), DType::Float32);
}

#[test]
fn test_softmax_basic() {
    let x = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);
    let y = x.softmax(-1);
    assert!(y.is_ok());

    assert_eq!(y.unwrap().uop().dtype(), DType::Float32);
}

#[test]
fn test_log_softmax_basic() {
    let x = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);
    let y = x.log_softmax(-1);
    assert!(y.is_ok());

    assert_eq!(y.unwrap().uop().dtype(), DType::Float32);
}

#[test]
fn test_logsumexp_basic() {
    let x = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);
    let y = x.logsumexp(-1);
    if let Err(e) = &y {
        eprintln!("logsumexp error: {:?}", e);
    }
    assert!(y.is_ok());

    assert_eq!(y.unwrap().uop().dtype(), DType::Float32);
}

#[test]
fn test_gelu_basic() {
    let x = Tensor::from_slice([-2.0f32, -1.0, 0.0, 1.0, 2.0]);
    let y = x.gelu();
    assert!(y.is_ok());

    assert_eq!(y.unwrap().uop().dtype(), DType::Float32);
}

#[test]
fn test_swish_basic() {
    let x = Tensor::from_slice([-2.0f32, -1.0, 0.0, 1.0, 2.0]);
    let y = x.swish();
    assert!(y.is_ok());

    assert_eq!(y.unwrap().uop().dtype(), DType::Float32);
}

#[test]
fn test_silu_alias() {
    let x = Tensor::from_slice([-2.0f32, -1.0, 0.0, 1.0, 2.0]);
    let y = x.silu();
    assert!(y.is_ok());

    assert_eq!(y.unwrap().uop().dtype(), DType::Float32);
}
