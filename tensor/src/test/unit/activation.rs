#![allow(clippy::approx_constant)]

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

// =========================================================================
// Batch Normalization Tests
// =========================================================================

#[test]
fn test_batchnorm_basic() {
    // Create input tensor [2, 3]
    let x = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).try_reshape([2, 3]).unwrap();

    // Create parameters (each has shape [3] for axis=1)
    let scale = Tensor::from_slice([1.0f32, 1.0, 1.0]);
    let bias = Tensor::from_slice([0.0f32, 0.0, 0.0]);
    let mean = Tensor::from_slice([2.5f32, 3.5, 4.5]);

    // invstd = 1/sqrt(var + eps). For var=[2.25, 2.25, 2.25] and eps=1e-5, invstd ≈ 0.666
    let invstd = Tensor::from_slice([0.666_666_7f32, 0.666_666_7, 0.666_666_7]);

    let result = x.batchnorm().scale(&scale).bias(&bias).mean(&mean).invstd(&invstd).call().unwrap();

    // Verify dtype preserved
    assert_eq!(result.uop().dtype(), DType::Float32);

    // Verify shape preserved
    let uop = result.uop();
    let shape = uop.shape().unwrap().unwrap();
    assert_eq!(shape.len(), 2);
    assert_eq!(shape[0].as_const(), Some(2));
    assert_eq!(shape[1].as_const(), Some(3));
}

#[test]
fn test_batchnorm_no_scale_bias() {
    let x = Tensor::from_slice([1.0f32, 2.0, 3.0]).try_reshape([3, 1]).unwrap();

    // Without scale/bias (None)
    let mean = Tensor::from_slice([2.0f32]);
    let invstd = Tensor::from_slice([1.0f32]);

    let result = x.batchnorm().mean(&mean).invstd(&invstd).call().unwrap();

    assert_eq!(result.uop().dtype(), DType::Float32);
}

#[test]
fn test_batchnorm_different_axis() {
    // Input shape [2, 3, 4] - normalize over axis 0
    // For axis=0, mean/scale/bias need shape [2] (size of dim 0)
    let x = Tensor::from_slice([1.0f32; 24]).try_reshape([2, 3, 4]).unwrap();

    let scale = Tensor::from_slice([1.0f32, 1.0]);
    let bias = Tensor::from_slice([0.0f32, 0.0]);
    let mean = Tensor::from_slice([0.5f32, 0.5]);
    let invstd = Tensor::from_slice([1.0f32, 1.0]);

    let result = x
        .batchnorm()
        .scale(&scale)
        .bias(&bias)
        .mean(&mean)
        .invstd(&invstd)
        .axis(reduce::AxisSpec::Single(0))
        .call()
        .unwrap();

    // Shape should be preserved
    let uop = result.uop();
    let shape = uop.shape().unwrap().unwrap();
    assert_eq!(shape.len(), 3);
}

#[test]
fn test_batchnorm_4d() {
    // Input shape [2, 3, 4, 5] - typical CNN shape
    let x = Tensor::from_slice([1.0f32; 120]).try_reshape([2, 3, 4, 5]).unwrap();

    // BatchNorm2d normalizes over channels (axis=1)
    let scale = Tensor::from_slice([1.0f32, 1.0, 1.0]);
    let bias = Tensor::from_slice([0.0f32, 0.0, 0.0]);
    let mean = Tensor::from_slice([0.5f32, 0.5, 0.5]);
    let invstd = Tensor::from_slice([1.0f32, 1.0, 1.0]);

    let result = x.batchnorm().scale(&scale).bias(&bias).mean(&mean).invstd(&invstd).call().unwrap();

    let uop = result.uop();
    let shape = uop.shape().unwrap().unwrap();
    assert_eq!(shape[0].as_const(), Some(2));
    assert_eq!(shape[1].as_const(), Some(3));
    assert_eq!(shape[2].as_const(), Some(4));
    assert_eq!(shape[3].as_const(), Some(5));
}

// =========================================================================
// Activation Value Tests
// =========================================================================

crate::codegen_tests! {
    fn test_softplus_values(config) {
        let x = Tensor::from_slice([0.0f32, 1.0, -1.0]);
        let mut r = x.softplus(1.0).unwrap();
        r.realize_with(&config).unwrap();
        crate::test::helpers::assert_close_f32(
            &r.as_vec::<f32>().unwrap(),
            &[0.6931, 1.3133, 0.3133],
            1e-3,
        );
    }

    fn test_softplus_beta(config) {
        let x = Tensor::from_slice([0.0f32, 1.0]);
        // softplus(0, beta=2) = log(1+exp(0))/2 = ln(2)/2 = 0.3466
        // softplus(1, beta=2) = log(1+exp(2))/2 = ln(8.389)/2 = 1.0635
        let mut r = x.softplus(2.0).unwrap();
        r.realize_with(&config).unwrap();
        crate::test::helpers::assert_close_f32(&r.as_vec::<f32>().unwrap(), &[0.3466, 1.0635], 1e-3);
    }

    fn test_softplus_large_input(config) {
        // softplus(100) ≈ 100.0 (should not overflow to inf)
        let x = Tensor::from_slice([100.0f32, -100.0]);
        let mut r = x.softplus(1.0).unwrap();
        r.realize_with(&config).unwrap();
        crate::test::helpers::assert_close_f32(&r.as_vec::<f32>().unwrap(), &[100.0, 0.0], 1e-3);
    }

    fn test_mish_values(config) {
        let x = Tensor::from_slice([0.0f32, 1.0, -1.0]);
        let mut r = x.mish().unwrap();
        r.realize_with(&config).unwrap();
        crate::test::helpers::assert_close_f32(&r.as_vec::<f32>().unwrap(), &[0.0, 0.8651, -0.3034], 1e-3);
    }

    fn test_relu6_values(config) {
        let x = Tensor::from_slice([-1.0f32, 0.0, 3.0, 6.0, 9.0]);
        let mut r = x.relu6().unwrap();
        r.realize_with(&config).unwrap();
        crate::test::helpers::assert_close_f32(
            &r.as_vec::<f32>().unwrap(),
            &[0.0, 0.0, 3.0, 6.0, 6.0],
            1e-4,
        );
    }

    fn test_hardswish_values(config) {
        let x = Tensor::from_slice([-4.0f32, -3.0, 0.0, 3.0, 4.0]);
        let mut r = x.hardswish().unwrap();
        r.realize_with(&config).unwrap();
        crate::test::helpers::assert_close_f32(
            &r.as_vec::<f32>().unwrap(),
            &[0.0, 0.0, 0.0, 3.0, 4.0],
            1e-3,
        );
    }

    fn test_softsign_values(config) {
        let x = Tensor::from_slice([-2.0f32, 0.0, 2.0]);
        let mut r = x.softsign().unwrap();
        r.realize_with(&config).unwrap();
        crate::test::helpers::assert_close_f32(
            &r.as_vec::<f32>().unwrap(),
            &[-0.6667, 0.0, 0.6667],
            1e-3,
        );
    }

    fn test_celu_values(config) {
        let x = Tensor::from_slice([-1.0f32, 0.0, 1.0]);
        let mut r = x.celu(1.0).unwrap();
        r.realize_with(&config).unwrap();
        crate::test::helpers::assert_close_f32(&r.as_vec::<f32>().unwrap(), &[-0.6321, 0.0, 1.0], 1e-3);
    }
}
