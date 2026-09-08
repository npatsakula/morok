//! Tests for distribution wrappers (`uniform`, `randn`, `normal`, `randint`,
//! `scaled_uniform`, `glorot_uniform`, `kaiming_uniform`, `kaiming_normal`).
//!
//! We verify:
//! 1. **Range invariants** — values land where the distribution says they should.
//! 2. **Mean/stddev sanity** — sample mean within 4·stddev/√N of theory.
//! 3. **Determinism** — same seed → same output.
//! 4. **Validation** — invalid args produce errors.
//!
//! Sample size N=4096 keeps runtime fast while still giving ~4σ tolerance of
//! ~0.02 around mean and stddev — plenty for catching algorithmic bugs.

use svod_dtype::DType;

use crate::Tensor;
use crate::rand::manual_seed;

use super::RAND_TEST_LOCK;

const N: usize = 4096;
const TOL: f64 = 0.05; // ~4σ for N=4096

fn realize_f32(t: &mut Tensor, config: &crate::PrepareConfig) -> Vec<f32> {
    t.realize_with(config).expect("realize");
    t.as_vec::<f32>().expect("read")
}

fn realize_i32(t: &mut Tensor, config: &crate::PrepareConfig) -> Vec<i32> {
    t.realize_with(config).expect("realize");
    t.as_vec::<i32>().expect("read")
}

fn mean_stddev(v: &[f32]) -> (f64, f64) {
    let n = v.len() as f64;
    let mean = v.iter().map(|&x| x as f64).sum::<f64>() / n;
    let var = v.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n;
    (mean, var.sqrt())
}

crate::codegen_tests! {
    fn uniform_values_in_range_and_match_theory(config) {
        let _g = RAND_TEST_LOCK.lock();
        manual_seed(0xCAFE);
        let (low, high) = (-2.5_f64, 7.5);
        let mut t = Tensor::uniform(&[N], low, high).unwrap();
        let v = realize_f32(&mut t, &config);

        for &x in &v {
            assert!(x.is_finite(), "non-finite: {x}");
            assert!((x as f64) >= low && (x as f64) < high, "out of range: {x}");
        }
        let (m, s) = mean_stddev(&v);
        let expected_mean = (low + high) / 2.0;
        let expected_stddev = (high - low) / 12.0_f64.sqrt();
        assert!((m - expected_mean).abs() < TOL * (high - low), "mean={m}, expected≈{expected_mean}");
        assert!((s - expected_stddev).abs() < TOL * expected_stddev, "stddev={s}, expected≈{expected_stddev}");
    }

    fn randn_mean_zero_stddev_one(config) {
        let _g = RAND_TEST_LOCK.lock();
        manual_seed(123);
        let mut t = Tensor::randn(&[N]).unwrap();
        let v = realize_f32(&mut t, &config);
        for &x in &v {
            assert!(x.is_finite(), "randn produced non-finite: {x}");
        }
        let (m, s) = mean_stddev(&v);
        assert!(m.abs() < TOL, "randn mean={m}, expected≈0");
        assert!((s - 1.0).abs() < TOL, "randn stddev={s}, expected≈1");
    }

    fn normal_applies_scale_and_shift(config) {
        let _g = RAND_TEST_LOCK.lock();
        manual_seed(456);
        let (mean, std) = (10.0, 2.0);
        let mut t = Tensor::normal(&[N], mean, std).unwrap();
        let v = realize_f32(&mut t, &config);
        let (m, s) = mean_stddev(&v);
        assert!((m - mean).abs() < TOL * std, "normal mean={m}, expected≈{mean}");
        assert!((s - std).abs() < TOL * std, "normal stddev={s}, expected≈{std}");
    }

    fn randint_int_dtype_and_in_range(config) {
        let _g = RAND_TEST_LOCK.lock();
        manual_seed(0xBEEF);
        let (low, high) = (-5, 8);
        let mut t = Tensor::randint(&[N], low, high).unwrap();
        assert_eq!(t.uop().dtype(), DType::Int32);
        let v = realize_i32(&mut t, &config);
        for &x in &v {
            assert!(x >= low as i32 && x < high as i32, "out of range: {x}");
        }
    }

    /// Regression: for negative `low` the cast to int32 must happen on
    /// `(high - low) * rand` *before* adding `low`. The naive
    /// `(scaled + low).cast(i32)` direction truncates toward zero, producing
    /// an off-by-one (e.g. -2 instead of -3) for rand draws near 0.
    fn randint_negative_low_includes_lower_edge(config) {
        let _g = RAND_TEST_LOCK.lock();
        manual_seed(0xD00D);
        let (low, high) = (-3, 3);
        let mut t = Tensor::randint(&[N], low, high).unwrap();
        let v = realize_i32(&mut t, &config);
        for &x in &v {
            assert!(x >= low as i32 && x < high as i32, "out of range: {x}");
        }
        // Mean of uniform integers on [-3, 3) is -0.5; with N=4096 the sample
        // mean lands within ~0.1 of theory under correct truncation.
        let sample_mean = v.iter().map(|&x| x as f64).sum::<f64>() / v.len() as f64;
        assert!((sample_mean - -0.5).abs() < 0.15, "sample mean {sample_mean} indicates truncation bug");
        assert!(v.contains(&-3), "lower edge -3 never sampled (truncation bug)");
    }

    fn scaled_uniform_bounds(config) {
        let _g = RAND_TEST_LOCK.lock();
        manual_seed(11);
        let shape = &[64, 64];
        let mut t = Tensor::scaled_uniform(shape).unwrap();
        let v = realize_f32(&mut t, &config);
        let bound = (shape.iter().product::<usize>() as f64).powf(-0.5);
        for &x in &v {
            assert!((x as f64).abs() < bound, "scaled_uniform out of bound {bound}: {x}");
        }
    }

    fn glorot_uniform_bounds_and_mean_zero(config) {
        let _g = RAND_TEST_LOCK.lock();
        manual_seed(22);
        let shape = &[32, 32]; // fan_in = 32, fan_out = 32
        let bound = (6.0_f64 / 64.0).sqrt();
        let mut t = Tensor::glorot_uniform(shape).unwrap();
        let v = realize_f32(&mut t, &config);
        for &x in &v {
            assert!((x as f64).abs() < bound, "glorot_uniform out of bound {bound}: {x}");
        }
        let (m, _) = mean_stddev(&v);
        assert!(m.abs() < TOL * bound, "glorot_uniform mean={m}, expected≈0");
    }

    fn kaiming_uniform_bounds(config) {
        let _g = RAND_TEST_LOCK.lock();
        manual_seed(33);
        // For Conv2d-like shape [out_ch, in_ch, kH, kW], fan_in = in_ch*kH*kW.
        let shape = &[32, 16, 3, 3]; // fan_in = 16*3*3 = 144
        let a = 0.01_f64;
        let bound = (6.0_f64 / ((1.0 + a * a) * 144.0)).sqrt();
        let mut t = Tensor::kaiming_uniform(shape, a).unwrap();
        let v = realize_f32(&mut t, &config);
        for &x in &v {
            assert!((x as f64).abs() < bound, "kaiming_uniform out of bound {bound}: {x}");
        }
    }

    fn kaiming_normal_stddev_matches_theory(config) {
        let _g = RAND_TEST_LOCK.lock();
        manual_seed(44);
        let shape = &[64, 64]; // fan_in = 64
        let a = 0.0_f64;
        let expected_std = (2.0_f64 / ((1.0 + a * a) * 64.0)).sqrt();
        let mut t = Tensor::kaiming_normal(shape, a).unwrap();
        let v = realize_f32(&mut t, &config);
        let (m, s) = mean_stddev(&v);
        assert!(m.abs() < TOL * expected_std, "mean={m}, expected≈0");
        assert!((s - expected_std).abs() < TOL * expected_std, "stddev={s}, expected≈{expected_std}");
    }

    fn distributions_deterministic_for_same_seed(config) {
        let _g = RAND_TEST_LOCK.lock();
        let shape = &[16usize];

        let cfg = &config;
        macro_rules! check_deterministic {
            ($name:expr, $build:expr) => {{
                manual_seed(99);
                let mut a = ($build)();
                let va = realize_f32(&mut a, cfg);
                manual_seed(99);
                let mut b = ($build)();
                let vb = realize_f32(&mut b, cfg);
                assert_eq!(va, vb, "{} is non-deterministic for fixed seed", $name);
            }};
        }

        check_deterministic!("uniform", || Tensor::uniform(shape, -1.0, 2.0).unwrap());
        check_deterministic!("randn", || Tensor::randn(shape).unwrap());
        check_deterministic!("normal", || Tensor::normal(shape, 1.0, 0.5).unwrap());
        check_deterministic!("scaled_uniform", || Tensor::scaled_uniform(shape).unwrap());
        check_deterministic!("glorot_uniform", || Tensor::glorot_uniform(&[8, 8]).unwrap());
        check_deterministic!("kaiming_uniform", || Tensor::kaiming_uniform(&[8, 4], 0.01).unwrap());
        check_deterministic!("kaiming_normal", || Tensor::kaiming_normal(&[8, 4], 0.01).unwrap());
    }
}

// Validation-only tests (don't realize anything; no codegen variants needed).
#[test]
fn uniform_low_must_be_less_than_high() {
    let _g = RAND_TEST_LOCK.lock();
    assert!(Tensor::uniform(&[4], 1.0, 1.0).is_err());
    assert!(Tensor::uniform(&[4], 2.0, 1.0).is_err());
}

#[test]
fn normal_std_must_be_nonneg() {
    let _g = RAND_TEST_LOCK.lock();
    assert!(Tensor::normal(&[4], 0.0, -0.5).is_err());
}

#[test]
fn randint_low_must_be_less_than_high() {
    let _g = RAND_TEST_LOCK.lock();
    assert!(Tensor::randint(&[4], 5, 5).is_err());
    assert!(Tensor::randint(&[4], 10, 5).is_err());
}

#[test]
fn shape_validators_reject_empty_shape() {
    let _g = RAND_TEST_LOCK.lock();
    assert!(Tensor::glorot_uniform(&[]).is_err());
    assert!(Tensor::kaiming_uniform(&[], 0.0).is_err());
    assert!(Tensor::kaiming_normal(&[], 0.0).is_err());
}

/// Regression: a draw whose word count is not a multiple of the THREEFRY pair
/// width (`numel = 15` → 8 pairs, 16 words shrunk to 15) collapsed the
/// `arange.cast(u32).cast(u64)` chain of the counts before the weak index
/// math committed to an int width, and index lowering then promoted the
/// `range % 8` to a float `fmod` that failed kernel verification.
#[test_case::test_case(&[15]; "n15")]
#[test_case::test_case(&[3, 5]; "s3x5")]
#[test_case::test_case(&[5, 3]; "s5x3")]
#[test_case::test_case(&[7]; "n7")]
#[test_case::test_case(&[2, 4]; "s2x4")]
#[test_case::test_case(&[64, 64]; "s64x64")]
fn uniform_realizes_for_any_numel(shape: &[usize]) {
    let _g = RAND_TEST_LOCK.lock();
    manual_seed(7);
    let t = Tensor::uniform(shape, -1.0, 1.0).unwrap();
    t.realize().expect("realize");
    let v = t.as_vec::<f32>().expect("read");
    assert_eq!(v.len(), shape.iter().product::<usize>());
    assert!(v.iter().all(|x| (-1.0..1.0).contains(x)), "{v:?}");
}
