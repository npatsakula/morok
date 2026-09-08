#![allow(clippy::approx_constant)]

use crate::test::helpers::*;
use crate::*;
use svod_dtype::DType;

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
    assert!(result.uop().toposort().iter().any(|node| matches!(node.op(), Op::Unary(svod_ir::UnaryOp::Erf, _))));
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
fn test_isinf() {
    let t = Tensor::from_slice([1.0f32, f32::INFINITY, f32::NEG_INFINITY]);
    let result = t.isinf(true, true).unwrap();
    assert_eq!(result.uop().dtype(), DType::Bool);
}

crate::codegen_tests! {
    /// Half builtins must round to half on every backend: the C backend computes
    /// them at f32 and used to hand the f32 result straight to the consumer.
    fn test_float16_builtins_round_to_float16(config) {
        let x = Tensor::from_slice([0.5f32, 100.0]).cast(DType::Float16).unwrap();
        let sqrt = x.try_sqrt().unwrap().cast(DType::Float32).unwrap();
        sqrt.realize_with(&config).unwrap();
        assert_eq!(sqrt.as_vec::<f32>().unwrap()[0], 0.70703125);
        let exp2 = x.try_exp2().unwrap().cast(DType::Float32).unwrap();
        exp2.realize_with(&config).unwrap();
        assert!(exp2.as_vec::<f32>().unwrap()[1].is_infinite(), "exp2(100) must overflow float16");
    }
}

crate::codegen_tests! {
    /// The lowered transcendentals must match libm. `sin(+-1e6)` exercises
    /// Payne-Hanek reduction rather than the small-angle Cody-Waite path.
    fn test_transcendental_values(config) {
        let check = |inputs: &[f32], op: fn(&Tensor) -> crate::Result<Tensor>, reference: fn(f32) -> f32| {
            let actual = op(&Tensor::from_slice(inputs)).unwrap().realize_with_and(&config).as_vec::<f32>().unwrap();
            assert_close_f32(&actual, &inputs.iter().map(|&x| reference(x)).collect::<Vec<_>>(), 2e-6);
        };
        check(&[-3.25, -0.5, 0.0, 0.75, 4.0], Tensor::try_exp2, f32::exp2);
        check(&[f32::MIN_POSITIVE, 0.125, 0.75, 1.0, 17.0], Tensor::try_log2, f32::log2);
        check(&[-1_000_000.0, -31.0, -0.5, 0.0, 31.0, 1_000_000.0], Tensor::sin, f32::sin);
        check(&[0.0, 0.25, 1.0, 2.0, 100.0], Tensor::try_sqrt, f32::sqrt);
    }

    fn test_isnan_values(config) {
        let t = Tensor::from_slice([1.0f32, f32::NAN, 3.0]);
        let vals = t.isnan().unwrap().realize_with_and(&config).as_vec::<bool>().unwrap();
        assert_eq!(vals, [false, true, false]);
    }

    fn test_isinf_positive_only(config) {
        let t = Tensor::from_slice([1.0f32, f32::INFINITY, f32::NEG_INFINITY]);
        let vals = t.isinf(true, false).unwrap().realize_with_and(&config).as_vec::<bool>().unwrap();
        assert_eq!(vals, [false, true, false]);
    }

    // Hyperbolic function tests
    fn test_sinh_values(config) {
        let t = Tensor::from_slice([0.0f32, 1.0]);
        assert_close_f32(&t.sinh().unwrap().realize_with_and(&config).as_vec::<f32>().unwrap(), &[0.0, 1.1752], 1e-3);
    }

    fn test_cosh_values(config) {
        let t = Tensor::from_slice([0.0f32, 1.0]);
        assert_close_f32(&t.cosh().unwrap().realize_with_and(&config).as_vec::<f32>().unwrap(), &[1.0, 1.5431], 1e-3);
    }

    fn test_asinh_values(config) {
        let t = Tensor::from_slice([0.0f32, 1.0]);
        assert_close_f32(&t.asinh().unwrap().realize_with_and(&config).as_vec::<f32>().unwrap(), &[0.0, 0.8814], 1e-3);
    }

    fn test_acosh_values(config) {
        let t = Tensor::from_slice([1.0f32, 2.0]);
        assert_close_f32(&t.acosh().unwrap().realize_with_and(&config).as_vec::<f32>().unwrap(), &[0.0, 1.3170], 1e-3);
    }

    fn test_atanh_values(config) {
        let t = Tensor::from_slice([0.0f32, 0.5]);
        assert_close_f32(&t.atanh().unwrap().realize_with_and(&config).as_vec::<f32>().unwrap(), &[0.0, 0.5493], 1e-3);
    }

    // Inverse trigonometric tests
    fn test_asin_values(config) {
        let t = Tensor::from_slice([0.0f32, 0.5, 1.0]);
        assert_close_f32(&t.asin().unwrap().realize_with_and(&config).as_vec::<f32>().unwrap(), &[0.0, 0.5236, 1.5708], 1e-3);
    }

    fn test_acos_values(config) {
        let t = Tensor::from_slice([0.0f32, 0.5, 1.0]);
        assert_close_f32(&t.acos().unwrap().realize_with_and(&config).as_vec::<f32>().unwrap(), &[1.5708, 1.0472, 0.0], 1e-3);
    }

    fn test_atan_values(config) {
        let t = Tensor::from_slice([0.0f32, 1.0]);
        assert_close_f32(&t.atan().unwrap().realize_with_and(&config).as_vec::<f32>().unwrap(), &[0.0, 0.7854], 1e-3);
    }

    // Shrink test
    fn test_shrink_values(config) {
        let t = Tensor::from_slice([-2.0f32, -0.3, 0.0, 0.3, 2.0]);
        assert_close_f32(&t.shrink(0.0, 0.5).unwrap().realize_with_and(&config).as_vec::<f32>().unwrap(), &[-2.0, 0.0, 0.0, 0.0, 2.0], 1e-4);
    }
}

// =========================================================================
// Decompositions (cholesky / eigh) — built on the s!/getitem/set indexing API.
// =========================================================================
crate::codegen_tests! {
    fn test_cholesky_2x2(config) {
        // A = [[4,2],[2,3]] → L = [[2,0],[1,√2]]
        let a = Tensor::from_slice([4.0f32, 2.0, 2.0, 3.0]).try_reshape([2, 2]).unwrap();
        let l = a.cholesky().unwrap();
        assert_close_f32(
            &l.realize_with_and(&config).as_vec::<f32>().unwrap(),
            &[2.0, 0.0, 1.0, std::f32::consts::SQRT_2],
            1e-4,
        );
    }

    fn test_cholesky_3x3_reconstruct(config) {
        // SPD matrix; verify L @ Lᵀ ≈ A.
        let vals = [4.0f32, 2.0, 2.0, 2.0, 5.0, 3.0, 2.0, 3.0, 6.0];
        let a = Tensor::from_slice(vals).try_reshape([3, 3]).unwrap();
        let l = a.cholesky().unwrap();
        let recon = l.matmul(&l.try_transpose(-2, -1).unwrap()).unwrap();
        assert_close_f32(&recon.realize_with_and(&config).as_vec::<f32>().unwrap(), &vals, 1e-4);
    }

    fn test_qr_3x3_reconstruct(config) {
        // Verify Q @ R ≈ A and Qᵀ Q ≈ I (Q orthonormal).
        let vals = [12.0f32, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0];
        let a = Tensor::from_slice(vals).try_reshape([3, 3]).unwrap();
        let (q, r) = a.qr().unwrap();
        let recon = q.matmul(&r).unwrap();
        assert_close_f32(&recon.realize_with_and(&config).as_vec::<f32>().unwrap(), &vals, 1e-3);
        let qtq = q.try_transpose(-2, -1).unwrap().matmul(&q).unwrap();
        assert_close_f32(
            &qtq.realize_with_and(&config).as_vec::<f32>().unwrap(),
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            1e-4,
        );
    }

    // Regression: batched (ndim>2) qr previously panicked in `w.set(s![Ellipsis,0,0],..)`.
    fn test_qr_batched_reconstruct(config) {
        let vals = [
            12.0f32, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0, // batch 0
            2.0, 0.0, 1.0, 0.0, 3.0, 0.0, 1.0, 0.0, 2.0, // batch 1
        ];
        let a = Tensor::from_slice(vals).try_reshape([2, 3, 3]).unwrap();
        let (q, r) = a.qr().unwrap();
        let recon = q.matmul(&r).unwrap();
        assert_close_f32(&recon.realize_with_and(&config).as_vec::<f32>().unwrap(), &vals, 1e-3);
    }
}
