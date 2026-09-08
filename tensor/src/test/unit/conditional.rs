use crate::*;
use svod_dtype::DType;

#[test]
fn test_where_basic() {
    // condition ? x : y
    let x = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);
    let y = Tensor::from_slice([10.0f32, 20.0, 30.0, 40.0]);
    let z = Tensor::from_slice([2.5f32, 2.5, 2.5, 2.5]); // Same shape for now

    // Create condition using comparison
    let condition = x.try_gt(&z).unwrap();

    // For now, just test that the method exists and compiles
    // Full integration tests will come later with proper execution
    let result = x.where_(&condition, &y);
    assert!(result.is_ok());
}

#[test]
fn test_maximum_shapes() {
    let a = Tensor::from_slice([1.0f32, 5.0, 3.0]);
    let b = Tensor::from_slice([2.0f32, 3.0, 4.0]);

    let result = a.maximum(&b);
    assert!(result.is_ok());

    // Check dtype is preserved
    assert_eq!(result.unwrap().uop().dtype(), DType::Float32);
}

#[test]
fn test_minimum_shapes() {
    let a = Tensor::from_slice([1.0f32, 5.0, 3.0]);
    let b = Tensor::from_slice([2.0f32, 3.0, 4.0]);

    let result = a.minimum(&b);
    assert!(result.is_ok());

    // Check dtype is preserved
    assert_eq!(result.unwrap().uop().dtype(), DType::Float32);
}

#[test]
fn test_clamp_both_bounds() {
    let x = Tensor::from_slice([-1.0f32, 0.0, 1.0, 2.0, 3.0]);
    // Same shape for now - broadcasting not yet implemented
    let min = Tensor::from_slice([0.0f32, 0.0, 0.0, 0.0, 0.0]);
    let max = Tensor::from_slice([2.0f32, 2.0, 2.0, 2.0, 2.0]);

    let result = x.clamp().min(&min).max(&max).call();
    assert!(result.is_ok());
}

#[test]
fn test_clamp_only_min() {
    let x = Tensor::from_slice([-1.0f32, 0.0, 1.0, 2.0, 3.0]);
    // Same shape for now - broadcasting not yet implemented
    let min = Tensor::from_slice([0.0f32, 0.0, 0.0, 0.0, 0.0]);

    let result = x.clamp().min(&min).call();
    assert!(result.is_ok());
}

#[test]
fn test_clamp_only_max() {
    let x = Tensor::from_slice([-1.0f32, 0.0, 1.0, 2.0, 3.0]);
    // Same shape for now - broadcasting not yet implemented
    let max = Tensor::from_slice([2.0f32, 2.0, 2.0, 2.0, 2.0]);

    let result = x.clamp().max(&max).call();
    assert!(result.is_ok());
}

#[test]
fn test_clip_alias() {
    let x = Tensor::from_slice([-1.0f32, 0.0, 1.0, 2.0, 3.0]);
    // Same shape for now - broadcasting not yet implemented
    let min = Tensor::from_slice([0.0f32, 0.0, 0.0, 0.0, 0.0]);
    let max = Tensor::from_slice([2.0f32, 2.0, 2.0, 2.0, 2.0]);

    let result = x.clip().min(&min).max(&max).call();
    assert!(result.is_ok());
}

#[test]
fn test_masked_fill_scalar() {
    let x = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);
    let mask = Tensor::from_slice([true, false, true, false]);
    let r = x.masked_fill(&mask, 0.0f32).unwrap();
    r.realize().unwrap();
    assert_eq!(r.as_vec::<f32>().unwrap(), vec![0.0, 2.0, 0.0, 4.0]);
}

#[test]
fn test_masked_fill_scalar_int_cast() {
    // Scalar arm must cast the const to self's dtype (Int32 here).
    let x = Tensor::from_slice([1i32, 2, 3, 4]);
    let mask = Tensor::from_slice([true, false, true, false]);
    let r = x.masked_fill(&mask, 7i32).unwrap();
    assert_eq!(r.uop().dtype(), DType::Int32);
    r.realize().unwrap();
    assert_eq!(r.as_vec::<i32>().unwrap(), vec![7, 2, 7, 4]);
}

#[test]
fn test_masked_fill_tensor() {
    let x = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);
    let mask = Tensor::from_slice([true, false, true, false]);
    let fill = Tensor::from_slice([-1.0f32, -2.0, -3.0, -4.0]);
    let r = x.masked_fill(&mask, &fill).unwrap();
    r.realize().unwrap();
    assert_eq!(r.as_vec::<f32>().unwrap(), vec![-1.0, 2.0, -3.0, 4.0]);
}

#[test]
fn test_masked_fill_broadcast_mask() {
    // Broadcastable [2,1] mask over [2,3] data → `mask ? fill : self`.
    // mask [[true],[false]] broadcasts to [[t,t,t],[f,f,f]]: row 0 filled, row 1 kept.
    let x = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).try_reshape([2, 3]).unwrap();
    let mask = Tensor::from_slice([true, false]).try_reshape([2, 1]).unwrap();
    let r = x.masked_fill(&mask, 0.0f32).unwrap();
    assert_eq!(r.shape().unwrap().iter().map(|s| s.as_const().unwrap()).collect::<Vec<_>>(), vec![2, 3]);
    r.realize().unwrap();
    assert_eq!(r.as_vec::<f32>().unwrap(), vec![0.0, 0.0, 0.0, 4.0, 5.0, 6.0]);
}
