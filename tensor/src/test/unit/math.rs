use crate::*;
use morok_dtype::DType;

// Trigonometric tests
#[test]
fn test_sin_basic() {
    let t = Tensor::from_slice([0.0f32, 1.0]);
    let result = t.sin().unwrap();
    assert_eq!(result.uop().dtype(), DType::Float32);
}

#[test]
fn test_cos_basic() {
    let t = Tensor::from_slice([0.0f32, 1.0]);
    let result = t.cos().unwrap();
    assert_eq!(result.uop().dtype(), DType::Float32);
}

#[test]
fn test_tan_basic() {
    let t = Tensor::from_slice([0.0f32, 1.0]);
    let result = t.tan().unwrap();
    assert_eq!(result.uop().dtype(), DType::Float32);
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
    assert_eq!(result.uop().dtype(), DType::Float32);
}

#[test]
fn test_ceil_basic() {
    let t = Tensor::from_slice([1.2f32, -1.2, 2.8]);
    let result = t.ceil().unwrap();
    assert_eq!(result.uop().dtype(), DType::Float32);
}

#[test]
fn test_round_basic() {
    let t = Tensor::from_slice([1.2f32, 1.5, 2.5]);
    let result = t.round().unwrap();
    assert_eq!(result.uop().dtype(), DType::Float32);
}

#[test]
fn test_trunc_basic() {
    let t = Tensor::from_slice([1.2f32, -1.2, 2.8]);
    let result = t.trunc().unwrap();
    assert_eq!(result.uop().dtype(), DType::Float32);
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
    assert_eq!(result.uop().dtype(), DType::Float32);
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
    assert_eq!(result.uop().dtype(), DType::Float32);
}

#[test]
fn test_square_basic() {
    let t = Tensor::from_slice([1.0f32, 2.0, -3.0]);
    let result = t.square().unwrap();
    assert_eq!(result.uop().dtype(), DType::Float32);
}

#[test]
fn test_square_int() {
    let t = Tensor::from_slice([1i32, 2, -3]);
    let result = t.square().unwrap();
    assert_eq!(result.uop().dtype(), DType::Int32);
}

#[test]
fn test_sign_basic() {
    let t = Tensor::from_slice([-5.0f32, 0.0, 3.0]);
    let result = t.sign().unwrap();
    assert_eq!(result.uop().dtype(), DType::Float32);
}

#[test]
fn test_sign_int() {
    let t = Tensor::from_slice([-5i32, 0, 3]);
    let result = t.sign().unwrap();
    assert_eq!(result.uop().dtype(), DType::Int32);
}

// NaN/Inf detection tests
#[test]
fn test_isnan() {
    let t = Tensor::from_slice([1.0f32, f32::NAN, 3.0]);
    let result = t.isnan().unwrap();
    assert_eq!(result.uop().dtype(), DType::Bool);
}

#[test]
fn test_isnan_values() {
    let t = Tensor::from_slice([1.0f32, f32::NAN, 3.0]);
    let result = t.isnan().unwrap();
    let arr = result.realize().unwrap().to_ndarray::<bool>().unwrap();
    let vals: Vec<bool> = arr.iter().copied().collect();
    assert_eq!(vals, vec![false, true, false]);
}

#[test]
fn test_isinf() {
    let t = Tensor::from_slice([1.0f32, f32::INFINITY, f32::NEG_INFINITY]);
    let result = t.isinf(true, true).unwrap();
    assert_eq!(result.uop().dtype(), DType::Bool);
}

#[test]
fn test_isinf_positive_only() {
    let t = Tensor::from_slice([1.0f32, f32::INFINITY, f32::NEG_INFINITY]);
    let result = t.isinf(true, false).unwrap();
    let arr = result.realize().unwrap().to_ndarray::<bool>().unwrap();
    let vals: Vec<bool> = arr.iter().copied().collect();
    assert_eq!(vals, vec![false, true, false]);
}

// Hyperbolic function tests
#[test]
fn test_sinh_values() {
    use crate::test::helpers::*;
    let t = Tensor::from_slice([0.0f32, 1.0]);
    let arr = realize_f32(t.sinh().unwrap());
    assert_close_f32(&arr, &[0.0, 1.1752], 1e-3);
}

#[test]
fn test_cosh_values() {
    use crate::test::helpers::*;
    let t = Tensor::from_slice([0.0f32, 1.0]);
    let arr = realize_f32(t.cosh().unwrap());
    assert_close_f32(&arr, &[1.0, 1.5431], 1e-3);
}

#[test]
fn test_asinh_values() {
    use crate::test::helpers::*;
    let t = Tensor::from_slice([0.0f32, 1.0]);
    let arr = realize_f32(t.asinh().unwrap());
    assert_close_f32(&arr, &[0.0, 0.8814], 1e-3);
}

#[test]
fn test_acosh_values() {
    use crate::test::helpers::*;
    let t = Tensor::from_slice([1.0f32, 2.0]);
    let arr = realize_f32(t.acosh().unwrap());
    assert_close_f32(&arr, &[0.0, 1.3170], 1e-3);
}

#[test]
fn test_atanh_values() {
    use crate::test::helpers::*;
    let t = Tensor::from_slice([0.0f32, 0.5]);
    let arr = realize_f32(t.atanh().unwrap());
    assert_close_f32(&arr, &[0.0, 0.5493], 1e-3);
}

// Inverse trigonometric tests
#[test]
fn test_asin_values() {
    use crate::test::helpers::*;
    let t = Tensor::from_slice([0.0f32, 0.5, 1.0]);
    let arr = realize_f32(t.asin().unwrap());
    assert_close_f32(&arr, &[0.0, 0.5236, 1.5708], 1e-3);
}

#[test]
fn test_acos_values() {
    use crate::test::helpers::*;
    let t = Tensor::from_slice([0.0f32, 0.5, 1.0]);
    let arr = realize_f32(t.acos().unwrap());
    assert_close_f32(&arr, &[1.5708, 1.0472, 0.0], 1e-3);
}

#[test]
fn test_atan_values() {
    use crate::test::helpers::*;
    let t = Tensor::from_slice([0.0f32, 1.0]);
    let arr = realize_f32(t.atan().unwrap());
    assert_close_f32(&arr, &[0.0, 0.7854], 1e-3);
}

// Shrink test
#[test]
fn test_shrink_values() {
    use crate::test::helpers::*;
    let t = Tensor::from_slice([-2.0f32, -0.3, 0.0, 0.3, 2.0]);
    let arr = realize_f32(t.shrink(0.0, 0.5).unwrap());
    assert_close_f32(&arr, &[-2.0, 0.0, 0.0, 0.0, 2.0], 1e-4);
}
