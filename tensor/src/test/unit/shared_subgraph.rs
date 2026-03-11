//! MRE for NoKernelsFound when a fully-lazy matrix (no buffer root) is used
//! in matmul.
//!
//! The bug triggers when the matmul LHS is constructed purely from lazy ops
//! (e.g. arange → reshape → outer product) without any realized buffer in its
//! lineage. When the LHS has a buffer root (from `from_slice`), the pipeline
//! works correctly even with unary transformations on top.
//!
//! Discovered while implementing DFT: `out = cos(angles) @ x` where `angles`
//! is built from `arange`.

use crate::Tensor;
use crate::test::helpers::*;
use morok_dtype::DType;
use ndarray::Array2;
use test_case::test_case;

// =========================================================================
// PASSING: buffer-rooted LHS
// =========================================================================

/// Baseline: realized matrix @ column vector. Always works.
#[test_case(2 ; "N=2")]
#[test_case(4 ; "N=4")]
fn test_realized_matrix_matmul(n: usize) {
    test_setup();

    let data: Vec<f32> = (0..n * n).map(|i| i as f32 * 0.1).collect();
    let matrix = Tensor::from_ndarray(&Array2::from_shape_vec((n, n), data).unwrap());

    let x = Tensor::from_ndarray(&Array2::from_shape_vec((n, 1), vec![1.0f32; n]).unwrap());

    let out = matrix.dot(&x).unwrap().try_reshape(&[n as isize]).unwrap();

    let result = out.realize().expect("realized matrix matmul");
    assert_eq!(result.to_vec::<f32>().unwrap().len(), n);
}

/// Unary on buffer-rooted matrix, then matmul. Works because the
/// buffer root is preserved through the unary op.
#[test_case(2 ; "N=2")]
#[test_case(4 ; "N=4")]
fn test_unary_on_buffer_rooted_matmul(n: usize) {
    test_setup();

    let data: Vec<f32> = (0..n * n).map(|i| i as f32 * 0.1).collect();
    let matrix = Tensor::from_ndarray(&Array2::from_shape_vec((n, n), data).unwrap()).cos().unwrap();

    let x = Tensor::from_ndarray(&Array2::from_shape_vec((n, 1), vec![1.0f32; n]).unwrap());

    let out = matrix.dot(&x).unwrap().try_reshape(&[n as isize]).unwrap();

    let result = out.realize().expect("unary on buffer-rooted matmul");
    assert_eq!(result.to_vec::<f32>().unwrap().len(), n);
}

/// Element-wise diamond (no matmul): cos(t) + sin(t). Always works.
#[test]
fn test_diamond_elementwise_no_matmul() {
    test_setup();

    let t = Tensor::from_ndarray(&Array2::from_shape_vec((2, 2), vec![1.0f32, 2.0, 3.0, 4.0]).unwrap());

    let out = t.cos().unwrap().try_add(&t.sin().unwrap()).unwrap();

    let result = out.realize().expect("diamond elementwise");

    let expected: Vec<f32> = [1.0f32, 2.0, 3.0, 4.0].iter().map(|x| x.cos() + x.sin()).collect();
    assert_close_f32(&result.to_vec::<f32>().unwrap(), &expected, 1e-5);
}

// =========================================================================
// FAILING: fully-lazy LHS (no buffer root)
// =========================================================================

use std::sync::Once;
use tracing_subscriber::EnvFilter;

static TRACING_INIT: Once = Once::new();

pub fn setup_tracing() {
    TRACING_INIT.call_once(|| {
        tracing_subscriber::fmt()
            .json()
            .with_env_filter(EnvFilter::from_default_env())
            .with_test_writer() // Ensures output is captured by `cargo test`
            .init();
    });
}

/// Simplest failing case: arange → outer product → matmul.
/// No unary, no diamond — just a lazy [N,N] matrix from arange.
#[test_case(2 ; "N=2")]
#[test_case(4 ; "N=4")]
fn test_lazy_outer_product_matmul(n: usize) {
    setup_tracing();
    test_setup();

    let indices = Tensor::arange(n as i64, None, None).unwrap().cast(DType::Float32).unwrap();
    let k = indices.try_reshape(&[n as isize, 1]).unwrap();
    let j = indices.try_reshape(&[1, n as isize]).unwrap();
    let matrix = k.try_mul(&j).unwrap();

    let x = Tensor::from_ndarray(&Array2::from_shape_vec((n, 1), vec![1.0f32; n]).unwrap());

    let out = matrix.dot(&x).unwrap().try_reshape(&[n as isize]).unwrap();

    let result = out.realize().expect("lazy outer product matmul");
    assert_eq!(result.to_vec::<f32>().unwrap().len(), n);
}

/// Lazy outer product → unary → matmul.
#[test_case(2 ; "N=2")]
#[test_case(4 ; "N=4")]
fn test_lazy_outer_product_unary_matmul(n: usize) {
    test_setup();

    let indices = Tensor::arange(n as i64, None, None).unwrap().cast(DType::Float32).unwrap();
    let k = indices.try_reshape(&[n as isize, 1]).unwrap();
    let j = indices.try_reshape(&[1, n as isize]).unwrap();
    let matrix = k.try_mul(&j).unwrap().cos().unwrap();

    let x = Tensor::from_ndarray(&Array2::from_shape_vec((n, 1), vec![1.0f32; n]).unwrap());

    let out = matrix.dot(&x).unwrap().try_reshape(&[n as isize]).unwrap();

    let result = out.realize().expect("lazy outer product unary matmul");
    assert_eq!(result.to_vec::<f32>().unwrap().len(), n);
}

/// Full DFT pattern: cos(angles) @ x + sin(angles) @ x.
/// Combines lazy matrix construction, diamond sharing, and matmul.
#[test_case(2 ; "N=2")]
#[test_case(4 ; "N=4")]
fn test_dft_pattern(n: usize) {
    test_setup();

    let indices = Tensor::arange(n as i64, None, None).unwrap().cast(DType::Float32).unwrap();
    let k = indices.try_reshape(&[n as isize, 1]).unwrap();
    let j = indices.try_reshape(&[1, n as isize]).unwrap();
    let angles = k.try_mul(&j).unwrap().try_mul(&Tensor::from_slice([-0.5f32])).unwrap();

    let cos_w = angles.cos().unwrap();
    let sin_w = angles.sin().unwrap();

    let x = Tensor::from_ndarray(&Array2::from_shape_vec((n, 1), vec![1.0f32; n]).unwrap());

    let out = cos_w.dot(&x).unwrap().try_add(&sin_w.dot(&x).unwrap()).unwrap().try_reshape(&[n as isize]).unwrap();

    let result = out.realize().expect("DFT pattern");
    assert_eq!(result.to_vec::<f32>().unwrap().len(), n);
}
