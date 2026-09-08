//! Affine normalization (`layernorm_with`, `rms_norm_with`), `conv1d` and the
//! `batchnorm(var)` path.

use svod_dtype::DType;
use test_case::test_case;

use crate::Tensor;
use crate::error::ErrorKind;
use crate::nn::Pad1d;

/// Deterministic spread-out values; `n` elements starting at `seed`.
fn ramp(n: usize, seed: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32 + seed) * 0.7).sin() * 3.0 + 0.25 * i as f32).collect()
}

fn tensor(data: Vec<f32>, shape: [isize; 3]) -> Tensor {
    Tensor::from_slice(data).try_reshape(shape).unwrap()
}

/// Read back as f32 regardless of the tensor dtype.
fn read(t: &Tensor) -> Vec<f32> {
    t.cast(DType::Float32).to_vec::<f32>().unwrap()
}

/// `tol` is relative to `max(1, |expected|)`: the two sides may round `rsqrt`
/// and fused multiply-adds differently per backend.
fn assert_close(got: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(got.len(), expected.len());
    for (i, (g, e)) in got.iter().zip(expected).enumerate() {
        assert!((g - e).abs() <= tol * e.abs().max(1.0), "element {i}: got {g}, expected {e}");
    }
}

// =========================================================================
// layernorm_with / rms_norm_with
// =========================================================================

#[test_case(DType::Float32, 1e-6; "f32 input")]
#[test_case(DType::Float16, 1e-3; "f16 input with f32 params")]
fn layernorm_with_matches_manual_affine(dtype: DType, tol: f32) {
    let x = tensor(ramp(24, 0.0), [2, 3, 4]).cast(dtype.clone());
    let weight = Tensor::from_slice(ramp(4, 1.5));
    let bias = Tensor::from_slice(ramp(4, 2.5));

    let got = x.layernorm_with().weight(&weight).bias(&bias).call().unwrap();
    assert_eq!(got.uop().dtype(), dtype, "affine layernorm must return the input dtype");

    // The policy this replaces: normalize and scale in f32, round once at the end.
    let expected = x
        .cast(DType::Float32)
        .layernorm(-1, 1e-5)
        .unwrap()
        .try_mul(&weight)
        .unwrap()
        .try_add(&bias)
        .unwrap()
        .cast(dtype);

    assert_close(&read(&got), &read(&expected), tol);
}

#[test_case(DType::Float32, 1e-6; "f32 input")]
#[test_case(DType::Float16, 1e-3; "f16 input with f32 params")]
fn rms_norm_with_matches_manual_affine(dtype: DType, tol: f32) {
    let x = tensor(ramp(24, 3.0), [2, 3, 4]).cast(dtype.clone());
    let weight = Tensor::from_slice(ramp(4, 0.5));

    let got = x.rms_norm_with().weight(&weight).call().unwrap();
    assert_eq!(got.uop().dtype(), dtype, "affine rms_norm must return the input dtype");

    let expected = x.cast(DType::Float32).rms_norm(-1, 1e-5).unwrap().try_mul(&weight).unwrap().cast(dtype);
    assert_close(&read(&got), &read(&expected), tol);
}

#[test]
fn norm_builders_default_to_the_positional_wrappers() {
    let x = tensor(ramp(12, 1.0), [1, 3, 4]);
    assert_close(&read(&x.layernorm_with().call().unwrap()), &read(&x.layernorm(-1, 1e-5).unwrap()), 0.0);
    assert_close(&read(&x.rms_norm_with().call().unwrap()), &read(&x.rms_norm(-1, 1e-5).unwrap()), 0.0);
}

#[test]
fn layernorm_with_normalizes_over_a_leading_axis() {
    // axis=1 on [1,3,4] normalizes the whole 12-element trailing block.
    let x = tensor(ramp(12, 2.0), [1, 3, 4]);
    let got = x.layernorm_with().axis(1).call().unwrap();
    let expected = x.layernorm(1, 1e-5).unwrap();
    assert_close(&read(&got), &read(&expected), 0.0);
    let mean: f32 = read(&got).iter().sum::<f32>() / 12.0;
    assert!(mean.abs() < 1e-5, "normalized block must have zero mean, got {mean}");
}

#[test]
fn norm_eps_reaches_the_computation() {
    // A constant row has zero variance: only eps keeps 1/sqrt(var + eps) finite,
    // so a huge eps must visibly shrink the RMS-normalized magnitude.
    let x = tensor(vec![4.0; 4], [1, 1, 4]);
    let small = read(&x.rms_norm_with().eps(1e-5).call().unwrap())[0];
    let large = read(&x.rms_norm_with().eps(1e3).call().unwrap())[0];
    assert!((small - 1.0).abs() < 1e-3, "expected ~1.0, got {small}");
    assert!(large < 0.5, "large eps must damp the output, got {large}");
}

// =========================================================================
// conv1d
// =========================================================================

/// `[N, C, L]` conv1d must equal conv2d over the `[N, C, 1, L]` view.
#[test_case(1, (0, 0), 1, 1; "plain")]
#[test_case(2, (0, 0), 1, 1; "stride 2")]
#[test_case(1, (2, 2), 1, 1; "symmetric padding")]
#[test_case(1, (2, 0), 1, 1; "causal padding")]
#[test_case(1, (0, 0), 2, 1; "dilation 2")]
#[test_case(1, (1, 1), 1, 2; "grouped")]
#[test_case(2, (3, 1), 2, 2; "everything at once")]
fn conv1d_matches_conv2d_on_a_singleton_height(stride: usize, padding: (isize, isize), dilation: usize, groups: usize) {
    const N: isize = 2;
    const CIN: isize = 4;
    const COUT: isize = 6;
    const LEN: isize = 9;
    const K: isize = 3;

    let cin_per_group = CIN / groups as isize;
    let x = tensor(ramp((N * CIN * LEN) as usize, 0.0), [N, CIN, LEN]);
    let w = tensor(ramp((COUT * cin_per_group * K) as usize, 5.0), [COUT, cin_per_group, K]);
    let b = Tensor::from_slice(ramp(COUT as usize, 9.0));

    let got = x
        .conv1d()
        .weight(&w)
        .bias(&b)
        .stride(stride)
        .padding(padding)
        .dilation(dilation)
        .groups(groups)
        .call()
        .unwrap();

    let expected = x
        .try_unsqueeze(2)
        .unwrap()
        .conv2d()
        .weight(&w.try_unsqueeze(2).unwrap())
        .bias(&b)
        .stride(&[1, stride])
        .dilation(&[1, dilation])
        .padding(&[(0, 0), padding])
        .groups(groups)
        .call()
        .unwrap()
        .try_squeeze(Some(2))
        .unwrap();

    assert_eq!(got.dims().unwrap(), expected.dims().unwrap());
    assert_close(&read(&got), &read(&expected), 1e-5);
}

#[test]
fn conv1d_symmetric_padding_from_a_scalar() {
    let x = tensor(ramp(8, 0.0), [1, 1, 8]);
    let w = tensor(ramp(3, 1.0), [1, 1, 3]);
    let scalar = x.conv1d().weight(&w).padding(2).call().unwrap();
    let pair = x.conv1d().weight(&w).padding((2, 2)).call().unwrap();
    assert_eq!(Pad1d::from(2), Pad1d::from((2, 2)));
    assert_close(&read(&scalar), &read(&pair), 0.0);
}

#[test]
fn conv1d_rejects_non_3d_operands() {
    let x4 = Tensor::from_slice(ramp(8, 0.0)).try_reshape([1, 1, 2, 4]).unwrap();
    let w = tensor(ramp(3, 1.0), [1, 1, 3]);
    let err = x4.conv1d().weight(&w).call().unwrap_err();
    assert!(matches!(err.kind(), ErrorKind::NdimExact { op: "conv1d", expected: 3, actual: 4 }), "{err}");

    let x = tensor(ramp(8, 0.0), [1, 1, 8]);
    let w4 = Tensor::from_slice(ramp(3, 1.0)).try_reshape([1, 1, 1, 3]).unwrap();
    assert!(x.conv1d().weight(&w4).call().is_err());
}

// =========================================================================
// batchnorm(var, eps)
// =========================================================================

#[test_case(DType::Float32, 1e-6; "f32")]
#[test_case(DType::Float16, 5e-3; "f16 input")]
fn batchnorm_var_matches_a_precomputed_invstd(dtype: DType, tol: f32) {
    const EPS: f64 = 1e-5;
    let x = tensor(ramp(12, 0.0), [1, 3, 4]).cast(dtype);
    let mean = Tensor::from_slice(ramp(3, 4.0));
    let var_data: Vec<f32> = ramp(3, 6.0).iter().map(|v| v.abs() + 0.5).collect();
    let var = Tensor::from_slice(&var_data);
    let scale = Tensor::from_slice(ramp(3, 8.0));
    let bias = Tensor::from_slice(ramp(3, 2.0));

    let invstd_data: Vec<f32> = var_data.iter().map(|v| 1.0 / (v + EPS as f32).sqrt()).collect();
    let invstd = Tensor::from_slice(&invstd_data);

    let from_var = x.batchnorm().scale(&scale).bias(&bias).mean(&mean).var(&var).eps(EPS).call().unwrap();
    let from_invstd = x.batchnorm().scale(&scale).bias(&bias).mean(&mean).invstd(&invstd).call().unwrap();

    assert_close(&read(&from_var), &read(&from_invstd), tol);
}

#[test]
fn batchnorm_var_defaults_to_1e_minus_5() {
    let x = tensor(ramp(4, 0.0), [1, 2, 2]);
    let mean = Tensor::from_slice([0.0f32, 0.0]);
    let var = Tensor::from_slice([1.0f32, 4.0]);
    let explicit = x.batchnorm().mean(&mean).var(&var).eps(1e-5).call().unwrap();
    let implicit = x.batchnorm().mean(&mean).var(&var).call().unwrap();
    assert_close(&read(&implicit), &read(&explicit), 0.0);
}

#[test]
fn batchnorm_requires_exactly_one_of_invstd_and_var() {
    let x = tensor(ramp(4, 0.0), [1, 2, 2]);
    let mean = Tensor::from_slice([0.0f32, 0.0]);
    let var = Tensor::from_slice([1.0f32, 4.0]);
    let invstd = Tensor::from_slice([1.0f32, 0.5]);

    for err in [
        x.batchnorm().mean(&mean).call().unwrap_err(),
        x.batchnorm().mean(&mean).var(&var).invstd(&invstd).call().unwrap_err(),
    ] {
        assert!(matches!(err.kind(), ErrorKind::ExclusiveParams { op: "batchnorm", .. }), "{err}");
        assert!(err.to_string().contains("exactly one of"), "{err}");
    }
}
