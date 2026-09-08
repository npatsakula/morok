//! Tests for `Tensor::rand_like` / `Tensor::randn_like` / `Tensor::randint_like`.

use svod_dtype::DType;
use svod_ir::shape::Shape;

use crate::Tensor;
use crate::rand::manual_seed;

use super::RAND_TEST_LOCK;

fn realize_f32(t: &mut Tensor, config: &crate::PrepareConfig) -> Vec<f32> {
    t.realize_with(config).expect("realize");
    t.as_vec::<f32>().expect("read")
}

fn shape_to_usize(shape: &Shape) -> Vec<usize> {
    shape.iter().map(|s| s.as_const().expect("concrete")).collect()
}

// =========================================================================
// Graph-shape tests: build a lazy rand_like / randn_like / randint_like and
// check shape/dtype/device only — no realize, so codegen is irrelevant.
// =========================================================================

#[test]
fn rand_like_inherits_shape_dtype_device() {
    let _g = RAND_TEST_LOCK.lock();
    manual_seed(1);
    let template = Tensor::empty(&[80, 44], DType::Float32);
    let r = template.rand_like().unwrap();
    assert_eq!(shape_to_usize(&r.shape().unwrap()), vec![80, 44]);
    assert_eq!(r.uop().dtype(), template.uop().dtype());
    assert_eq!(r.device(), template.device());
}

#[test]
fn randn_like_inherits_shape_dtype_device() {
    let _g = RAND_TEST_LOCK.lock();
    manual_seed(2);
    let template = Tensor::empty(&[80, 44], DType::Float32);
    let r = template.randn_like().unwrap();
    assert_eq!(shape_to_usize(&r.shape().unwrap()), vec![80, 44]);
    assert_eq!(r.uop().dtype(), template.uop().dtype());
    assert_eq!(r.device(), template.device());
}

#[test]
fn rand_like_dtype_inheritance_and_override() {
    let _g = RAND_TEST_LOCK.lock();
    manual_seed(5);
    let template_f16 = Tensor::empty(&[16], DType::Float16);

    let r_inherit = template_f16.rand_like().unwrap();
    assert_eq!(r_inherit.uop().dtype(), DType::Float16);

    let r_override = template_f16.rand_like_with_dtype(DType::Float64).unwrap();
    assert_eq!(r_override.uop().dtype(), DType::Float64);
}

#[test]
fn randn_like_dtype_inheritance_and_override() {
    let _g = RAND_TEST_LOCK.lock();
    manual_seed(6);
    let template = Tensor::empty(&[16], DType::Float32);

    let r_inherit = template.randn_like().unwrap();
    assert_eq!(r_inherit.uop().dtype(), DType::Float32);

    let r_override = template.randn_like_with_dtype(DType::Float16).unwrap();
    assert_eq!(r_override.uop().dtype(), DType::Float16);
}

#[test]
fn randint_like_inherits_shape_and_keeps_int32() {
    let _g = RAND_TEST_LOCK.lock();
    manual_seed(8);
    let template = Tensor::empty(&[16, 8], DType::Int32);
    let r = template.randint_like(-3, 7).unwrap();
    assert_eq!(shape_to_usize(&r.shape().unwrap()), vec![16, 8]);
    assert_eq!(r.uop().dtype(), DType::Int32);
}

#[test]
fn randint_like_low_must_be_less_than_high() {
    let _g = RAND_TEST_LOCK.lock();
    let template = Tensor::empty(&[4], DType::Int32);
    assert!(template.randint_like(5, 5).is_err());
    assert!(template.randint_like(10, 5).is_err());
}

// =========================================================================
// Realize tests: parameterised across codegen backends.
// =========================================================================

crate::codegen_tests! {
    /// Zero-shape inputs yield empty tensors without crashing the rand pipeline.
    fn rand_like_zero_shape_yields_empty_tensor(config) {
        let _g = RAND_TEST_LOCK.lock();
        manual_seed(3);
        let template = Tensor::empty(&[0, 5], DType::Float32);
        let r = template.rand_like().unwrap();
        assert_eq!(shape_to_usize(&r.shape().unwrap()), vec![0, 5]);
        r.realize_with(&config).expect("realize empty tensor");
        assert_eq!(r.numel().unwrap(), 0);
    }

    /// 6D shapes work end-to-end.
    fn rand_like_handles_6d_shape(config) {
        let _g = RAND_TEST_LOCK.lock();
        manual_seed(4);
        let template = Tensor::empty(&[2, 3, 4, 5, 6, 7], DType::Float32);
        let mut r = template.rand_like().unwrap();
        assert_eq!(shape_to_usize(&r.shape().unwrap()), vec![2, 3, 4, 5, 6, 7]);
        let v = realize_f32(&mut r, &config);
        assert_eq!(v.len(), 2 * 3 * 4 * 5 * 6 * 7);
        for (i, &x) in v.iter().enumerate() {
            assert!(x.is_finite() && (0.0..1.0).contains(&x), "rand_like[{i}] = {x} outside [0, 1)");
        }
    }

    /// `randint_like` with an Int64 template casts the result up to Int64.
    fn randint_like_casts_to_int64_template(config) {
        let _g = RAND_TEST_LOCK.lock();
        manual_seed(9);
        let template = Tensor::empty(&[32], DType::Int64);
        let r = template.randint_like(0, 1000).unwrap();
        assert_eq!(r.uop().dtype(), DType::Int64);
        r.realize_with(&config).expect("realize");
        let v = r.as_vec::<i64>().expect("read");
        for (i, &x) in v.iter().enumerate() {
            assert!((0..1000).contains(&x), "randint_like[{i}] = {x} outside [0, 1000)");
        }
    }

    /// `randint_like` values land in `[low, high)`.
    fn randint_like_values_in_range(config) {
        let _g = RAND_TEST_LOCK.lock();
        manual_seed(10);
        let template = Tensor::empty(&[4096], DType::Int32);
        let r = template.randint_like(-5, 8).unwrap();
        r.realize_with(&config).expect("realize");
        let v = r.as_vec::<i32>().expect("read");
        for (i, &x) in v.iter().enumerate() {
            assert!((-5..8).contains(&x), "randint_like[{i}] = {x} outside [-5, 8)");
        }
    }

    /// Determinism — rand_like with the same seed produces the same output.
    fn rand_like_is_deterministic(config) {
        let _g = RAND_TEST_LOCK.lock();
        let template = Tensor::empty(&[64], DType::Float32);

        manual_seed(7);
        let mut a = template.rand_like().unwrap();
        let va = realize_f32(&mut a, &config);

        manual_seed(7);
        let mut b = template.rand_like().unwrap();
        let vb = realize_f32(&mut b, &config);

        assert_eq!(va, vb, "rand_like should be deterministic under fixed seed");
    }
}
