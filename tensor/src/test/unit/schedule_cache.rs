use crate::*;

fn cfg() -> PrepareConfig {
    PrepareConfig::from(morok_schedule::OptimizerConfig::default())
}

/// Verify that same-shape computations produce identical content hashes
/// after normalization (the key property that enables kernel caching).
#[test]
fn test_same_shape_same_hash() {
    morok_schedule::testing::setup_test_tracing();
    let cfg = cfg();

    let c1 = &Tensor::from_slice([1.0f32, 2.0, 3.0]) + &Tensor::from_slice([4.0f32, 5.0, 6.0]);
    let c2 = &Tensor::from_slice([10.0f32, 20.0, 30.0]) + &Tensor::from_slice([40.0f32, 50.0, 60.0]);

    let (h1, d1) = crate::schedule_cache::cache_key_for(&c1, &cfg).unwrap();
    let (h2, d2) = crate::schedule_cache::cache_key_for(&c2, &cfg).unwrap();
    assert_eq!(h1, h2, "same-shape computations must produce same content hash");
    assert_eq!(d1, d2);
}

/// Verify that different shapes produce different hashes.
#[test]
fn test_different_shape_different_hash() {
    morok_schedule::testing::setup_test_tracing();
    let cfg = cfg();

    let c1 = &Tensor::from_slice([1.0f32, 2.0, 3.0]) + &Tensor::from_slice([4.0f32, 5.0, 6.0]);
    let c2 = &Tensor::from_slice([1.0f32, 2.0]) + &Tensor::from_slice([3.0f32, 4.0]);

    let (h1, _) = crate::schedule_cache::cache_key_for(&c1, &cfg).unwrap();
    let (h2, _) = crate::schedule_cache::cache_key_for(&c2, &cfg).unwrap();
    assert_ne!(h1, h2, "different shapes must produce different hashes");
}

/// Verify that different ops produce different hashes.
#[test]
fn test_different_op_different_hash() {
    morok_schedule::testing::setup_test_tracing();
    let cfg = cfg();

    let add = &Tensor::from_slice([1.0f32, 2.0, 3.0]) + &Tensor::from_slice([4.0f32, 5.0, 6.0]);
    let mul = &Tensor::from_slice([1.0f32, 2.0, 3.0]) * &Tensor::from_slice([4.0f32, 5.0, 6.0]);

    let (h_add, _) = crate::schedule_cache::cache_key_for(&add, &cfg).unwrap();
    let (h_mul, _) = crate::schedule_cache::cache_key_for(&mul, &cfg).unwrap();
    assert_ne!(h_add, h_mul, "different ops must produce different hashes");
}

/// Verify that realizing the same shape twice produces correct results.
/// The second call should reuse cached compiled kernels (not re-optimize/recompile).
#[test]
fn test_repeated_realize_correct() {
    morok_schedule::testing::setup_test_tracing();
    let cfg = cfg();

    let mut c1 = &Tensor::from_slice([1.0f32, 2.0, 3.0]) + &Tensor::from_slice([4.0f32, 5.0, 6.0]);
    c1.realize_with(&cfg).unwrap();
    assert_eq!(c1.as_vec::<f32>().unwrap(), vec![5.0, 7.0, 9.0]);

    let mut c2 = &Tensor::from_slice([10.0f32, 20.0, 30.0]) + &Tensor::from_slice([40.0f32, 50.0, 60.0]);
    c2.realize_with(&cfg).unwrap();
    assert_eq!(c2.as_vec::<f32>().unwrap(), vec![50.0, 70.0, 90.0]);
}

/// Verify matmul kernel caching produces correct results on repeated calls.
#[test]
fn test_repeated_matmul_correct() {
    morok_schedule::testing::setup_test_tracing();
    let cfg = cfg();

    let a1 = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]).try_reshape([2, 2]).unwrap();
    let b1 = Tensor::from_slice([5.0f32, 6.0, 7.0, 8.0]).try_reshape([2, 2]).unwrap();
    let mut c1 = a1.matmul(&b1).unwrap();
    c1.realize_with(&cfg).unwrap();
    assert_eq!(c1.as_vec::<f32>().unwrap(), vec![19.0, 22.0, 43.0, 50.0]);

    let a2 = Tensor::from_slice([10.0f32, 20.0, 30.0, 40.0]).try_reshape([2, 2]).unwrap();
    let b2 = Tensor::from_slice([50.0f32, 60.0, 70.0, 80.0]).try_reshape([2, 2]).unwrap();
    let mut c2 = a2.matmul(&b2).unwrap();
    c2.realize_with(&cfg).unwrap();
    assert_eq!(c2.as_vec::<f32>().unwrap(), vec![1900.0, 2200.0, 4300.0, 5000.0]);
}
