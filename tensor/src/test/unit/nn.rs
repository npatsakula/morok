//! Tests for neural network operations: pool, conv, normalization, resize.

use ndarray::{Array2, Array4, array};
use svod_dtype::DType;

use crate::Tensor;
use crate::nn::{Conv1d, LSTMCell, Layer, Reduction, ResizeMode};
use crate::test::helpers::RealizeTestExt;

fn get_shape(tensor: &Tensor) -> Vec<usize> {
    tensor.uop().shape().unwrap().unwrap().iter().map(|s| s.as_const().unwrap()).collect()
}

#[test]
fn dynamic_quantized_linear_matches_integer_reference() {
    let x = Tensor::from_slice([127.0f32, -64.0, 32.0, 0.0, 63.5, -32.0, 16.0, 0.0]).try_reshape([2, 4]).unwrap();
    let weight = Tensor::from_slice([1i8, 1, 1, 1, 2, -1, 0, 1, -1, 0, 2, -2]).try_reshape([3, 4]).unwrap();
    let weight_scale = Tensor::from_slice([0.5f32, 1.0, 2.0]);
    let bias = Tensor::from_slice([1.0f32, -2.0, 0.5]);

    let output = x.dynamic_quantized_linear().weight(&weight).weight_scale(&weight_scale).bias(&bias).call().unwrap();
    output.realize().unwrap();

    assert_eq!(output.uop().dtype(), DType::Float32);
    let expected = [48.5f32, 316.0, -125.5, 24.75, 157.0, -62.5];
    for (actual, expected) in output.as_vec::<f32>().unwrap().iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-4, "actual={actual}, expected={expected}");
    }
}

// =========================================================================
// Pool shape tests
// =========================================================================

#[test]
fn test_pool_2d_basic() {
    // (1,1,4,4) k=2 s=1 d=1 → (1,1,3,3,2,2)
    let x = Tensor::from_ndarray(&Array4::<f32>::zeros((1, 1, 4, 4)));
    let pooled = x.pool(&[2, 2], &[1, 1], &[1, 1]).unwrap();
    let shape = pooled.shape().unwrap();
    let dims: Vec<usize> = shape.iter().map(|s| s.as_const().unwrap()).collect();
    assert_eq!(dims, vec![1, 1, 3, 3, 2, 2]);
}

#[test]
fn test_pool_2d_stride() {
    // (1,1,6,6) k=3 s=2 d=1 → (1,1,2,2,3,3)
    let x = Tensor::from_ndarray(&Array4::<f32>::zeros((1, 1, 6, 6)));
    let pooled = x.pool(&[3, 3], &[2, 2], &[1, 1]).unwrap();
    let shape = pooled.shape().unwrap();
    let dims: Vec<usize> = shape.iter().map(|s| s.as_const().unwrap()).collect();
    assert_eq!(dims, vec![1, 1, 2, 2, 3, 3]);
}

#[test]
fn test_pool_2d_dilation() {
    // (1,1,7,7) k=3 s=1 d=2 → (1,1,3,3,3,3)
    let x = Tensor::from_ndarray(&Array4::<f32>::zeros((1, 1, 7, 7)));
    let pooled = x.pool(&[3, 3], &[1, 1], &[2, 2]).unwrap();
    let shape = pooled.shape().unwrap();
    let dims: Vec<usize> = shape.iter().map(|s| s.as_const().unwrap()).collect();
    assert_eq!(dims, vec![1, 1, 3, 3, 3, 3]);
}

// =========================================================================
// Ceil mode pooling shape tests
// =========================================================================

#[test]
fn test_avg_pool2d_ceil_mode_shape() {
    // (1,1,7,7) with k=2 s=3 ceil_mode=true → output should be 3x3 (ceil) vs 2x2 (floor)
    let x = Tensor::from_ndarray(&Array4::<f32>::zeros((1, 1, 7, 7)));
    let result = x.avg_pool2d().kernel_size(&[2, 2]).stride(&[3, 3]).ceil_mode(true).call().unwrap();
    let shape = result.shape().unwrap();
    let dims: Vec<usize> = shape.iter().map(|s| s.as_const().unwrap()).collect();
    assert_eq!(dims, vec![1, 1, 3, 3]);
}

#[test]
fn test_max_pool2d_ceil_mode_shape() {
    // (1,1,7,7) with k=2 s=3 ceil_mode=true → output should be 3x3
    let x = Tensor::from_ndarray(&Array4::<f32>::zeros((1, 1, 7, 7)));
    let result = x.max_pool2d().kernel_size(&[2, 2]).stride(&[3, 3]).ceil_mode(true).call().unwrap();
    let shape = result.shape().unwrap();
    let dims: Vec<usize> = shape.iter().map(|s| s.as_const().unwrap()).collect();
    assert_eq!(dims, vec![1, 1, 3, 3]);
}

#[test]
fn test_avg_pool2d_ceil_mode_large_stride() {
    // Regression test for ceil_mode correction: input=3, kernel=2, stride=3
    // Without correction, apply_ceil_mode over-pads by 1.
    // Expected: ceildiv(3-2, 3)+1 = 2 output elements, but last window starts
    // past real data, so correction reduces padding.
    let x = Tensor::from_ndarray(&array![[[[1.0f32, 2.0, 3.0]]]]);
    let result = x.avg_pool2d().kernel_size(&[1, 2]).stride(&[1, 3]).ceil_mode(true).call().unwrap();
    let shape = result.shape().unwrap();
    let dims: Vec<usize> = shape.iter().map(|s| s.as_const().unwrap()).collect();
    // With stride=3, kernel=2, input=3: floor output=1 ([1,2]), ceil output=1
    // (last window at offset 3 starts past data end, so correction removes it)
    assert_eq!(dims[3], 1);
}

// =========================================================================
// Linspace shape-only test
// =========================================================================

#[test]
fn test_linspace_zero() {
    let t = Tensor::linspace(0.0, 1.0, 0, svod_dtype::DType::Float32).unwrap();
    assert_eq!(get_shape(&t), vec![0]);
}

// =========================================================================
// Input validation tests
// =========================================================================

fn expect_err_msg<T>(result: crate::Result<T>, substr: &str) {
    let msg = result.err().expect("expected an error").to_string();
    assert!(msg.contains(substr), "error should contain '{substr}', got: {msg}");
}

#[test]
fn test_depth_to_space_rejects_3d() {
    let x = Tensor::from_slice([0.0f32; 24]).try_reshape([2, 3, 4]).unwrap();
    expect_err_msg(x.depth_to_space().blocksize(2).call(), "exactly 4D");
}

#[test]
fn test_depth_to_space_rejects_indivisible_channels() {
    // c=3, blocksize=2 → blocksize^2=4, 3 % 4 != 0
    let x = Tensor::from_ndarray(&Array4::<f32>::zeros((1, 3, 2, 2)));
    expect_err_msg(x.depth_to_space().blocksize(2).call(), "divisible");
}

#[test]
fn test_space_to_depth_rejects_indivisible_spatial() {
    // h=3, w=3, blocksize=2 → 3 % 2 != 0
    let x = Tensor::from_ndarray(&Array4::<f32>::zeros((1, 1, 3, 3)));
    expect_err_msg(x.space_to_depth(2), "divisible");
}

#[test]
fn test_dropout_rejects_invalid_p() {
    let x = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    expect_err_msg(x.dropout().p(1.5).call(), "p");
    expect_err_msg(x.dropout().p(-0.1).call(), "p");
}

#[test]
fn test_lp_pool_rejects_p_zero() {
    let x = Tensor::from_ndarray(&Array4::<f32>::zeros((1, 1, 4, 4)));
    expect_err_msg(x.lp_pool().kernel_shape(&[2, 2]).p(0).call(), "p");
}

#[test]
fn test_group_norm_rejects_1d() {
    let x = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);
    let scale = Tensor::from_slice([1.0f32]);
    let bias = Tensor::from_slice([0.0f32]);
    expect_err_msg(x.group_norm().scale(&scale).bias(&bias).num_groups(1).call(), "at least 2D");
}

#[test]
fn test_lrn_rejects_3d() {
    let x = Tensor::from_slice([0.0f32; 24]).try_reshape([2, 3, 4]).unwrap();
    expect_err_msg(x.lrn().size(5).call(), "exactly 4D");
}

// =========================================================================
// Codegen tests
// =========================================================================

crate::codegen_tests! {
    fn test_pad_value_neg_inf(config) {
        let x = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let padded = x.try_pad_value(&[(1, 1)], f64::NEG_INFINITY).unwrap();
        let result = padded.realize_with_and(&config).as_vec::<f32>().unwrap();
        assert_eq!(result.len(), 5);
        assert!(result[0].is_infinite() && result[0] < 0.0);
        assert_eq!(result[1], 1.0);
        assert_eq!(result[2], 2.0);
        assert_eq!(result[3], 3.0);
        assert!(result[4].is_infinite() && result[4] < 0.0);
    }

    fn test_pad_value_zero_delegates(config) {
        // pad_value with 0.0 should be identical to try_pad
        let x = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let padded = x.try_pad_value(&[(1, 1)], 0.0).unwrap();
        let result = padded.realize_with_and(&config).as_vec::<f32>().unwrap();
        assert_eq!(result.len(), 5);
        assert_eq!(result[0], 0.0);
        assert_eq!(result[1], 1.0);
        assert_eq!(result[3], 3.0);
        assert_eq!(result[4], 0.0);
    }

    fn test_conv2d_1x1(config) {
        // 1x1 convolution acts as a per-pixel linear transformation
        // Input: (1, 1, 3, 3), Weight: (1, 1, 1, 1) with value 2.0
        let x_data: Vec<f32> = (1..=9).map(|v| v as f32).collect();
        let x = Tensor::from_ndarray(&Array4::from_shape_vec((1, 1, 3, 3), x_data).unwrap());
        let w = Tensor::from_ndarray(&array![[[[2.0f32]]]]);
        let result = x.conv2d().weight(&w).call().unwrap();
        let result = result.contiguous();
        result.realize_with(&config).unwrap();
        let view = result.array_view::<f32>().unwrap();
        let expected: Vec<f32> = (1..=9).map(|v| v as f32 * 2.0).collect();
        assert_eq!(view.shape(), &[1, 1, 3, 3]);
        for (got, exp) in view.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-5, "got {got}, expected {exp}");
        }
    }

    fn test_conv2d_3x3(config) {
        // 3x3 all-ones kernel on 4x4 input
        // Output should be 2x2 with sums of 3x3 regions
        let x_data: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let x = Tensor::from_ndarray(&Array4::from_shape_vec((1, 1, 4, 4), x_data).unwrap());
        let w = Tensor::from_ndarray(&Array4::<f32>::ones((1, 1, 3, 3)));
        let result = x.conv2d().weight(&w).call().unwrap();
        let result = result.contiguous();
        result.realize_with(&config).unwrap();
        let view = result.array_view::<f32>().unwrap();
        assert_eq!(view.shape(), &[1, 1, 2, 2]);
        // Top-left 3x3: 0+1+2+4+5+6+8+9+10 = 45
        assert!((view[[0, 0, 0, 0]] - 45.0).abs() < 1e-4);
        // Top-right 3x3: 1+2+3+5+6+7+9+10+11 = 54
        assert!((view[[0, 0, 0, 1]] - 54.0).abs() < 1e-4);
        // Bottom-left: 4+5+6+8+9+10+12+13+14 = 81
        assert!((view[[0, 0, 1, 0]] - 81.0).abs() < 1e-4);
        // Bottom-right: 5+6+7+9+10+11+13+14+15 = 90
        assert!((view[[0, 0, 1, 1]] - 90.0).abs() < 1e-4);
    }

    fn test_conv2d_stride(config) {
        // 2x2 kernel, stride=2 on 4x4 → 2x2
        let x_data: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let x = Tensor::from_ndarray(&Array4::from_shape_vec((1, 1, 4, 4), x_data).unwrap());
        let w = Tensor::from_ndarray(&Array4::<f32>::ones((1, 1, 2, 2)));
        let result = x.conv2d().weight(&w).stride(&[2, 2]).call().unwrap();
        let result = result.contiguous();
        result.realize_with(&config).unwrap();
        let view = result.array_view::<f32>().unwrap();
        assert_eq!(view.shape(), &[1, 1, 2, 2]);
        // Top-left: 0+1+4+5 = 10
        assert!((view[[0, 0, 0, 0]] - 10.0).abs() < 1e-4);
        // Top-right: 2+3+6+7 = 18
        assert!((view[[0, 0, 0, 1]] - 18.0).abs() < 1e-4);
    }

    // NOTE: test_conv2d_groups is disabled — see root cause analysis below.
    // The failure is NOT specific to conv2d groups. It's a fundamental bug in the
    // CONTIGUOUS realization path: assign_ranges() creates separate RANGE nodes for
    // CONTIGUOUS realization that leak into the outer STORE scope when the inner
    // STAGE is removed. split_store() then rejects the END because it sees
    // non-OUTER ranges in scope. This affects ANY tensor with multiple non-trivial
    // dims that goes through CONTIGUOUS realization.
    // Minimal repro: Tensor::from_slice(&[1.0f32, 2.0]).contiguous().realize()
    #[ignore = "blocked by CONTIGUOUS realization range-leak bug in rangeify pipeline"]
    fn test_conv2d_groups(config) {
        // Depthwise conv: groups=2, input (1,2,3,3), weight (2,1,1,1)
        let x = Tensor::from_ndarray(&Array4::<f32>::ones((1, 2, 3, 3)));
        let w = Tensor::from_ndarray(&array![[[[2.0f32]]], [[[3.0f32]]]]);
        let result = x.conv2d().weight(&w).groups(2).call().unwrap();
        let result = result.contiguous();
        result.realize_with(&config).unwrap();
        let view = result.array_view::<f32>().unwrap();
        assert_eq!(view.shape(), &[1, 2, 3, 3]);
        // Channel 0: all 1.0 * 2.0 = 2.0
        assert!((view[[0, 0, 0, 0]] - 2.0).abs() < 1e-4);
        // Channel 1: all 1.0 * 3.0 = 3.0
        assert!((view[[0, 1, 0, 0]] - 3.0).abs() < 1e-4);
    }

    fn test_conv2d_bias(config) {
        let x = Tensor::from_ndarray(&Array4::<f32>::ones((1, 1, 2, 2)));
        let w = Tensor::from_ndarray(&array![[[[1.0f32]]]]);
        let b = Tensor::from_slice([10.0f32]);
        let result = x.conv2d().weight(&w).bias(&b).call().unwrap();
        let result = result.contiguous();
        result.realize_with(&config).unwrap();
        let view = result.array_view::<f32>().unwrap();
        assert_eq!(view.shape(), &[1, 1, 2, 2]);
        // 1.0 * 1.0 + 10.0 = 11.0
        assert!((view[[0, 0, 0, 0]] - 11.0).abs() < 1e-4);
    }

    fn test_conv2d_padding(config) {
        // 3x3 kernel with padding=1 on 3x3 → 3x3
        let x = Tensor::from_ndarray(&Array4::<f32>::ones((1, 1, 3, 3)));
        let w = Tensor::from_ndarray(&Array4::<f32>::ones((1, 1, 3, 3)));
        let result = x.conv2d().weight(&w).padding(&[(1, 1), (1, 1)]).call().unwrap();
        let shape = result.shape().unwrap();
        let dims: Vec<usize> = shape.iter().map(|s| s.as_const().unwrap()).collect();
        assert_eq!(dims, vec![1, 1, 3, 3]);
        let result = result.contiguous();
        result.realize_with(&config).unwrap();
        let view = result.array_view::<f32>().unwrap();
        // Center element: all 9 values = 9.0
        assert!((view[[0, 0, 1, 1]] - 9.0).abs() < 1e-4);
        // Corner: 4 elements = 4.0
        assert!((view[[0, 0, 0, 0]] - 4.0).abs() < 1e-4);
    }

    fn test_avg_pool2d(config) {
        // 2x2 kernel on 4x4 → 2x2
        let x_data: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let x = Tensor::from_ndarray(&Array4::from_shape_vec((1, 1, 4, 4), x_data).unwrap());
        let result = x.avg_pool2d().kernel_size(&[2, 2]).stride(&[2, 2]).call().unwrap();
        let result = result.contiguous();
        result.realize_with(&config).unwrap();
        let view = result.array_view::<f32>().unwrap();
        assert_eq!(view.shape(), &[1, 1, 2, 2]);
        // Top-left: mean(0,1,4,5) = 2.5
        assert!((view[[0, 0, 0, 0]] - 2.5).abs() < 1e-4);
        // Top-right: mean(2,3,6,7) = 4.5
        assert!((view[[0, 0, 0, 1]] - 4.5).abs() < 1e-4);
        // Bottom-left: mean(8,9,12,13) = 10.5
        assert!((view[[0, 0, 1, 0]] - 10.5).abs() < 1e-4);
        // Bottom-right: mean(10,11,14,15) = 12.5
        assert!((view[[0, 0, 1, 1]] - 12.5).abs() < 1e-4);
    }

    fn test_max_pool2d(config) {
        // 2x2 kernel on 4x4 with negative values
        let x_data: Vec<f32> =
            vec![-1.0, 2.0, 3.0, -4.0, 5.0, -6.0, 7.0, 8.0, 9.0, 10.0, -11.0, 12.0, 13.0, -14.0, 15.0, 16.0];
        let x = Tensor::from_ndarray(&Array4::from_shape_vec((1, 1, 4, 4), x_data).unwrap());
        let result = x.max_pool2d().kernel_size(&[2, 2]).stride(&[2, 2]).call().unwrap();
        let result = result.contiguous();
        result.realize_with(&config).unwrap();
        let view = result.array_view::<f32>().unwrap();
        assert_eq!(view.shape(), &[1, 1, 2, 2]);
        // Top-left: max(-1, 2, 5, -6) = 5
        assert!((view[[0, 0, 0, 0]] - 5.0).abs() < 1e-4);
        // Top-right: max(3, -4, 7, 8) = 8
        assert!((view[[0, 0, 0, 1]] - 8.0).abs() < 1e-4);
        // Bottom-left: max(9, 10, 13, -14) = 13
        assert!((view[[0, 0, 1, 0]] - 13.0).abs() < 1e-4);
        // Bottom-right: max(-11, 12, 15, 16) = 16
        assert!((view[[0, 0, 1, 1]] - 16.0).abs() < 1e-4);
    }

    fn test_max_pool2d_pad(config) {
        // Padding should fill with -inf, not 0
        // 3x3 kernel with padding=1 on 3x3 → 3x3, all values are negative
        let x = Tensor::from_ndarray(&Array4::from_elem((1, 1, 3, 3), -5.0f32));
        let result = x.max_pool2d().kernel_size(&[3, 3]).stride(&[1, 1]).padding(&[(1, 1), (1, 1)]).call().unwrap();
        result.realize_with(&config).unwrap();
        let result = result.as_vec::<f32>().unwrap();
        // All outputs should be -5.0 (not 0.0 which would happen with zero padding)
        for val in result.iter() {
            assert!((*val - (-5.0)).abs() < 1e-4, "max_pool2d with padding should use -inf fill, got {val}");
        }
    }

    fn test_max_pool2d_large_symmetric_pad(config) {
        // Padding as wide as the kernel's reach: every window is clamped to the
        // input, so out[i][j] is the element at (min(i, 2) + 2, min(j, 2) + 2).
        let x_data: Vec<f32> = (1..=25).map(|v| v as f32).collect();
        let x = Tensor::from_ndarray(&Array4::from_shape_vec((1, 1, 5, 5), x_data).unwrap());
        let result =
            x.max_pool2d().kernel_size(&[5, 5]).stride(&[1, 1]).padding(&[(2, 2), (2, 2)]).call().unwrap();
        result.realize_with(&config).unwrap();
        assert_eq!(get_shape(&result), vec![1, 1, 5, 5]);
        let expected: Vec<f32> = (0..5)
            .flat_map(|i: usize| (0..5).map(move |j: usize| ((i.min(2) + 2) * 5 + j.min(2) + 3) as f32))
            .collect();
        assert_eq!(result.as_vec::<f32>().unwrap(), expected);
    }

    fn test_max_pool2d_with_indices_basic(config) {
        // 2x2 kernel on 4x4 with stride 2 → 2x2 output
        let x_data: Vec<f32> =
            vec![-1.0, 2.0, 3.0, -4.0, 5.0, -6.0, 7.0, 8.0, 9.0, 10.0, -11.0, 12.0, 13.0, -14.0, 15.0, 16.0];
        let x = Tensor::from_ndarray(&Array4::from_shape_vec((1, 1, 4, 4), x_data).unwrap());
        let (values, indices) = x.max_pool2d_with_indices().kernel_size(&[2, 2]).stride(&[2, 2]).call().unwrap();
        let values = values.contiguous();
        values.realize_with(&config).unwrap();
        let vals = values.array_view::<f32>().unwrap();
        assert_eq!(vals.shape(), &[1, 1, 2, 2]);
        // Top-left: max(-1, 2, 5, -6) = 5 at flat index 4
        assert!((vals[[0, 0, 0, 0]] - 5.0).abs() < 1e-4);
        // Top-right: max(3, -4, 7, 8) = 8 at flat index 7
        assert!((vals[[0, 0, 0, 1]] - 8.0).abs() < 1e-4);

        let indices = indices.contiguous();
        indices.realize_with(&config).unwrap();
        let idx = indices.array_view::<i32>().unwrap();
        assert_eq!(idx.shape(), &[1, 1, 2, 2]);
        // Index of max=5 in flat 4x4: position (1,0) → index 4
        assert_eq!(idx[[0, 0, 0, 0]], 4);
        // Index of max=8 in flat 4x4: position (1,3) → index 7
        assert_eq!(idx[[0, 0, 0, 1]], 7);
    }

    fn test_layernorm(config) {
        // (2, 4), normalize over last axis
        let x = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]);
        let result = x.layernorm(-1, 1e-5).unwrap();
        let result = result.contiguous();
        result.realize_with(&config).unwrap();
        let view = result.array_view::<f32>().unwrap();
        assert_eq!(view.shape(), &[2, 4]);

        // For each row, mean should be ~0 and var should be ~1
        for row in 0..2 {
            let row_data: Vec<f32> = (0..4).map(|c| view[[row, c]]).collect();
            let mean: f32 = row_data.iter().sum::<f32>() / 4.0;
            let var: f32 = row_data.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / 4.0;
            assert!(mean.abs() < 1e-4, "mean should be ~0, got {mean}");
            assert!((var - 1.0).abs() < 0.1, "var should be ~1, got {var}");
        }
    }

    fn test_layernorm_2d(config) {
        // (2, 3, 4), normalize over last 2 axes
        let x_data: Vec<f32> = (0..24).map(|v| v as f32).collect();
        let x = Tensor::from_ndarray(&ndarray::Array3::from_shape_vec((2, 3, 4), x_data).unwrap());
        let result = x.layernorm(-2, 1e-5).unwrap();
        let result = result.contiguous();
        result.realize_with(&config).unwrap();
        let view = result.array_view::<f32>().unwrap();
        assert_eq!(view.shape(), &[2, 3, 4]);

        // For each batch, mean over last 2 dims should be ~0
        for b in 0..2 {
            let mut sum = 0.0f32;
            for h in 0..3 {
                for w in 0..4 {
                    sum += view[[b, h, w]];
                }
            }
            let mean = sum / 12.0;
            assert!(mean.abs() < 1e-3, "mean should be ~0, got {mean}");
        }
    }

    fn test_resize_nearest_upsample(config) {
        let t = Tensor::from_ndarray(&array![[[[1.0f32, 2.0], [3.0, 4.0]]]]);
        let result = t.resize().scales(&[1.0, 1.0, 2.0, 2.0]).mode(ResizeMode::Nearest).call().unwrap();
        result.realize_with(&config).unwrap();
        assert_eq!(get_shape(&result), vec![1, 1, 4, 4]);
    }

    fn test_resize_linear_upsample(config) {
        let t = Tensor::from_ndarray(&array![[[[1.0f32, 2.0], [3.0, 4.0]]]]);
        let result = t.resize().scales(&[1.0, 1.0, 2.0, 2.0]).mode(ResizeMode::Linear).call().unwrap();
        result.realize_with(&config).unwrap();
        assert_eq!(get_shape(&result), vec![1, 1, 4, 4]);
    }

    fn test_resize_nearest_downsample(config) {
        let x_data: Vec<f32> = (1..=9).map(|v| v as f32).collect();
        let t = Tensor::from_ndarray(&Array4::from_shape_vec((1, 1, 3, 3), x_data).unwrap());
        let result = t.resize().sizes(&[1, 1, 2, 2]).mode(ResizeMode::Nearest).call().unwrap();
        result.realize_with(&config).unwrap();
        assert_eq!(get_shape(&result), vec![1, 1, 2, 2]);
    }

    fn test_linspace_basic(config) {
        let t = Tensor::linspace(-1.0, 1.0, 5, svod_dtype::DType::Float32).unwrap();
        assert_eq!(get_shape(&t), vec![5]);
        t.realize_with(&config).unwrap();
        let result = t.as_vec::<f32>().unwrap();
        let expected = [-1.0f32, -0.5, 0.0, 0.5, 1.0];
        for (got, exp) in result.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-5, "got {got}, expected {exp}");
        }
    }

    fn test_linspace_single(config) {
        let t = Tensor::linspace(3.0, 7.0, 1, svod_dtype::DType::Float32).unwrap();
        assert_eq!(get_shape(&t), vec![1], "steps=1 must produce 1-D shape [1]");
        t.realize_with(&config).unwrap();
        let vals = t.as_vec::<f32>().unwrap();
        assert_eq!(vals.len(), 1);
        assert!((vals[0] - 3.0).abs() < 1e-5);
    }

    fn test_nll_loss_basic(config) {
        // 2 samples, 3 classes — mean reduction
        let log_probs = Tensor::from_ndarray(&array![
            [-0.5f32, -1.0, -2.0], // sample 0
            [-0.3, -1.5, -0.8],    // sample 1
        ]);
        let target = Tensor::from_slice([0i64, 2]); // class 0 for sample 0, class 2 for sample 1
        let loss = log_probs.nll_loss().target(&target).call().unwrap();
        let val = loss.realize_with_and(&config).as_vec::<f32>().unwrap()[0];
        // NLL = -log_probs[i, target[i]]: sample0=-(-0.5)=0.5, sample1=-(-0.8)=0.8
        // mean = (0.5 + 0.8) / 2 = 0.65
        assert!((val - 0.65).abs() < 1e-4, "got {val}");
    }

    fn test_nll_loss_none_reduction(config) {
        let log_probs = Tensor::from_ndarray(&array![
            [-0.5f32, -1.0, -2.0], // sample 0
            [-0.3, -1.5, -0.8],    // sample 1
        ]);
        let target = Tensor::from_slice([0i64, 2]);
        let loss = log_probs.nll_loss().target(&target).reduction(Reduction::None).call().unwrap();
        let vals = loss.realize_with_and(&config).as_vec::<f32>().unwrap();
        assert_eq!(vals.len(), 2);
        assert!((vals[0] - 0.5).abs() < 1e-4);
        assert!((vals[1] - 0.8).abs() < 1e-4);
    }

    fn test_nll_loss_weighted(config) {
        let log_probs = Tensor::from_ndarray(&array![
            [-0.5f32, -1.0, -2.0], // sample 0
            [-0.3, -1.5, -0.8],    // sample 1
        ]);
        let target = Tensor::from_slice([0i64, 2]);
        let weight = Tensor::from_slice([2.0f32, 1.0, 3.0]); // class weights
        let loss = log_probs.nll_loss().target(&target).weight(&weight).call().unwrap();
        let val = loss.realize_with_and(&config).as_vec::<f32>().unwrap()[0];
        // weighted: sample0=0.5*2.0=1.0, sample1=0.8*3.0=2.4
        // mean = (1.0 + 2.4) / (2.0 + 3.0) = 3.4 / 5.0 = 0.68
        assert!((val - 0.68).abs() < 1e-4, "got {val}");
    }

    fn test_nll_loss_ignore_index(config) {
        let log_probs = Tensor::from_ndarray(&array![
            [-0.5f32, -1.0, -2.0], // sample 0
            [-0.3, -1.5, -0.8],    // sample 1
        ]);
        let target = Tensor::from_slice([0i64, 2]);
        // Ignore class 2 — sample 1 is masked out
        let loss = log_probs.nll_loss().target(&target).ignore_index(2).call().unwrap();
        let val = loss.realize_with_and(&config).as_vec::<f32>().unwrap()[0];
        // Only sample 0 contributes: 0.5 / 1.0 = 0.5
        assert!((val - 0.5).abs() < 1e-4, "got {val}");
    }

    fn test_dropout_inference(config) {
        let x = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);
        let (output, mask) = x.dropout().p(0.5).call().unwrap();
        output.realize_with(&config).unwrap();
        assert_eq!(output.as_vec::<f32>().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        mask.realize_with(&config).unwrap();
        assert!(mask.as_vec::<bool>().unwrap().iter().all(|&v| v));
    }

    fn test_conv1d_module_matches_explicit_conv2d(config) {
        // Conv1d::forward must reproduce x.conv2d() with the stored stride/padding.
        let x_data: Vec<f32> = (0..8).map(|v| v as f32 * 0.1).collect();
        let x = Tensor::from_slice(&x_data).try_reshape([1isize, 2, 4]).unwrap();
        let w_data: Vec<f32> = (0..18).map(|v| (v as f32 * 0.05).sin()).collect();
        let w = Tensor::from_slice(&w_data).try_reshape([3isize, 2, 3]).unwrap();
        let b = Tensor::from_slice([0.1f32, 0.2, 0.3]);

        let conv = Conv1d::new(w.clone(), Some(b.clone())).with_stride(2).with_padding((1, 1));
        let got = conv.forward(&x).unwrap().contiguous();
        got.realize_with(&config).unwrap();

        let expected =
            x.conv2d().weight(&w).bias(&b).stride(&[2]).padding(&[(1, 1)]).call().unwrap().contiguous();
        expected.realize_with(&config).unwrap();

        assert_eq!(got.as_vec::<f32>().unwrap(), expected.as_vec::<f32>().unwrap());
    }

    fn test_conv1d_no_bias(config) {
        // Conv1d::new(_, None) must omit bias in the conv2d call.
        let x = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]).try_reshape([1isize, 1, 4]).unwrap();
        let w = Tensor::from_slice([0.5f32, 0.25]).try_reshape([1isize, 1, 2]).unwrap();

        let conv = Conv1d::new(w.clone(), None);
        let got = conv.forward(&x).unwrap().contiguous();
        got.realize_with(&config).unwrap();

        let expected = x.conv2d().weight(&w).call().unwrap().contiguous();
        expected.realize_with(&config).unwrap();

        assert_eq!(got.as_vec::<f32>().unwrap(), expected.as_vec::<f32>().unwrap());
    }

    fn test_lstm_cell_step_matches_explicit(config) {
        // LSTMCell::step must use PyTorch's [i, f, g, o] gate order.
        let input = 3usize;
        let hidden = 2usize;
        let four_hidden = 4 * hidden;

        let w_ih_data: Vec<f32> = (0..four_hidden * input).map(|i| (i as f32 * 0.1).sin()).collect();
        let w_ih = Tensor::from_slice(&w_ih_data).try_reshape([four_hidden as isize, input as isize]).unwrap();
        let w_hh_data: Vec<f32> = (0..four_hidden * hidden).map(|i| (i as f32 * 0.07).cos()).collect();
        let w_hh = Tensor::from_slice(&w_hh_data).try_reshape([four_hidden as isize, hidden as isize]).unwrap();
        let b_ih = Tensor::from_slice([0.01f32, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]);
        let b_hh = Tensor::from_slice([-0.01f32, -0.02, -0.03, -0.04, -0.05, -0.06, -0.07, -0.08]);

        let x = Tensor::from_slice([0.5f32, -0.25, 0.125]).try_reshape([1isize, input as isize]).unwrap();
        let h0 = Tensor::from_slice([0.1f32, -0.2]).try_reshape([1isize, hidden as isize]).unwrap();
        let c0 = Tensor::from_slice([0.3f32, 0.4]).try_reshape([1isize, hidden as isize]).unwrap();

        let cell = LSTMCell::new(w_ih.clone(), w_hh.clone(), b_ih.clone(), b_hh.clone());
        assert_eq!(cell.hidden_size(), hidden);
        let (new_h, new_c) = cell.step(&x, &h0, &c0).unwrap();
        let new_h = new_h.contiguous();
        let new_c = new_c.contiguous();
        new_h.realize_with(&config).unwrap();
        new_c.realize_with(&config).unwrap();

        // Reference: inline body with [i, f, g, o] gate order.
        let gates_x = x.linear().weight(&w_ih).bias(&b_ih).call().unwrap();
        let gates_h = h0.linear().weight(&w_hh).bias(&b_hh).call().unwrap();
        let gates = gates_x.try_add(&gates_h).unwrap();
        let parts = gates.split(&[hidden, hidden, hidden, hidden], 1).unwrap();
        let i = parts[0].sigmoid().unwrap();
        let f = parts[1].sigmoid().unwrap();
        let g = parts[2].tanh().unwrap();
        let o = parts[3].sigmoid().unwrap();
        let exp_c = f.try_mul(&c0).unwrap().try_add(&i.try_mul(&g).unwrap()).unwrap();
        let exp_h = o.try_mul(&exp_c.tanh().unwrap()).unwrap();
        let exp_h = exp_h.contiguous();
        let exp_c = exp_c.contiguous();
        exp_h.realize_with(&config).unwrap();
        exp_c.realize_with(&config).unwrap();

        assert_eq!(new_h.as_vec::<f32>().unwrap(), exp_h.as_vec::<f32>().unwrap());
        assert_eq!(new_c.as_vec::<f32>().unwrap(), exp_c.as_vec::<f32>().unwrap());
    }

    fn test_mean_variance_normalize_float16_constant_row(config) {
        // A float16 `eps` is subnormal and flushes to zero, so a constant row
        // divided 0 by 0 and produced NaN; the f32 path returns zeros.
        let x = Tensor::from_slice(vec![1000.0f32; 512]).try_reshape([1, 512]).unwrap().cast(DType::Float16).unwrap();
        let y = x.mean_variance_normalize(&[1], 1e-5).unwrap();
        assert_eq!(y.uop().dtype(), DType::Float16);
        let y = y.cast(DType::Float32).unwrap();
        let values = y.realize_with_and(&config).as_vec::<f32>().unwrap();
        assert!(values.iter().all(|v| *v == 0.0), "expected zeros, got {:?}", &values[..4]);
    }

    fn test_mean_variance_normalize_float16_matches_float32(config) {
        // Near-constant row: 1000 ± 4 in float16 must track the float32 result.
        let data: Vec<f32> = (0..512).map(|i| if i % 2 == 0 { 1004.0 } else { 996.0 }).collect();
        let x32 = Tensor::from_slice(data).try_reshape([1, 512]).unwrap();
        let x16 = x32.cast(DType::Float16).unwrap();

        let reference = x32.mean_variance_normalize(&[1], 1e-5).unwrap();
        let reference = reference.realize_with_and(&config).as_vec::<f32>().unwrap();
        let actual = x16.mean_variance_normalize(&[1], 1e-5).unwrap().cast(DType::Float32).unwrap();
        let actual = actual.realize_with_and(&config).as_vec::<f32>().unwrap();

        for (got, expected) in actual.iter().zip(reference.iter()) {
            assert!((got - expected).abs() < 1e-3, "got {got}, expected {expected}");
        }
    }

    fn test_qlinear_matmul_saturates_output(config) {
        // 3 x 200 = 600 accumulator: ONNX QuantizeLinear saturates to 255,
        // an int32 truncation would wrap it to 88.
        let a = Tensor::from_ndarray(&Array2::from_elem((2, 3), 200u8));
        let b = Tensor::from_ndarray(&Array2::from_elem((3, 1), 1u8));
        let one = Tensor::from_slice([1.0f32]);
        let zero_u8 = Tensor::from_slice([0u8]);

        let saturated = |b_zero_point: &Tensor| {
            let y = a
                .qlinear_matmul()
                .a_scale(&one)
                .a_zero_point(&zero_u8)
                .b(&b)
                .b_scale(&one)
                .b_zero_point(b_zero_point)
                .y_scale(&one)
                .y_zero_point(&zero_u8)
                .call()
                .unwrap();
            y.realize_with_and(&config).as_vec::<u8>().unwrap()
        };

        assert_eq!(saturated(&zero_u8), vec![255, 255]);
        // b_zero_point = 2 makes every product -200, so -600 clamps to 0.
        assert_eq!(saturated(&Tensor::from_slice([2u8])), vec![0, 0]);
    }

    fn test_qlinear_matmul_saturates_to_int8_range(config) {
        // Same 600 accumulator against a signed output dtype: clamps to 127.
        let a = Tensor::from_ndarray(&Array2::from_elem((1, 3), 200u8));
        let b = Tensor::from_ndarray(&Array2::from_elem((3, 1), 1u8));
        let one = Tensor::from_slice([1.0f32]);
        let y = a
            .qlinear_matmul()
            .a_scale(&one)
            .a_zero_point(&Tensor::from_slice([0u8]))
            .b(&b)
            .b_scale(&one)
            .b_zero_point(&Tensor::from_slice([0u8]))
            .y_scale(&one)
            .y_zero_point(&Tensor::from_slice([0i8]))
            .call()
            .unwrap();
        assert_eq!(y.realize_with_and(&config).as_vec::<i8>().unwrap(), vec![127]);
    }

    fn test_dynamic_quantized_linear_float16_small_activations(config) {
        // In float16 the 1e-6 epsilon and a ~3e-6 scale are subnormal, so
        // deriving the scale in the input dtype sent `x / scale` to infinity.
        let x = Tensor::from_slice([1e-4f32, 2e-4, 3e-4, 4e-4])
            .try_reshape([1, 4])
            .unwrap()
            .cast(DType::Float16)
            .unwrap();

        // Reference from the float16-rounded activations, quantized in f32.
        let widened = x.cast(DType::Float32).unwrap();
        let values = widened.realize_with_and(&config).as_vec::<f32>().unwrap();
        let scale = (values.iter().fold(0.0f32, |acc, v| acc.max(v.abs())) / 127.0).max(1e-6);
        let expected: f32 = values.iter().map(|v| (v / scale).round().clamp(-127.0, 127.0)).sum::<f32>() * scale;

        let weight = Tensor::from_slice([1i8, 1, 1, 1]).try_reshape([1, 4]).unwrap();
        let weight_scale = Tensor::from_slice([1.0f32]);
        let output = x.dynamic_quantized_linear().weight(&weight).weight_scale(&weight_scale).call().unwrap();
        assert_eq!(output.uop().dtype(), DType::Float16);
        let output = output.cast(DType::Float32).unwrap();
        let got = output.realize_with_and(&config).as_vec::<f32>().unwrap()[0];
        assert!(got.is_finite(), "float16 activation scale overflowed: {got}");
        assert!((got - expected).abs() <= expected * 2e-2, "got {got}, expected {expected}");
    }

    fn test_conv2d_int8_promotes_accumulator(config) {
        // 3x3 of 10 convolved with 3x3 of 5 sums to 450 — only representable
        // once the reduction promotes int8 to int32.
        let x = Tensor::from_ndarray(&Array4::from_elem((1, 1, 3, 3), 10.0f32)).cast(DType::Int8).unwrap();
        let w = Tensor::from_ndarray(&Array4::from_elem((1, 1, 3, 3), 5.0f32)).cast(DType::Int8).unwrap();
        let result = x.conv2d().weight(&w).call().unwrap().contiguous();
        result.realize_with(&config).unwrap();
        assert_eq!(result.uop().dtype(), DType::Int32);
        assert_eq!(result.as_vec::<i32>().unwrap(), vec![450]);
    }

    fn test_conv2d_int8_explicit_acc_dtype_wins(config) {
        // An explicit `acc_dtype` still selects the accumulator (and suppresses
        // promotion) rather than conflicting with it.
        let x = Tensor::from_ndarray(&Array4::from_elem((1, 1, 3, 3), 10.0f32)).cast(DType::Int8).unwrap();
        let w = Tensor::from_ndarray(&Array4::from_elem((1, 1, 3, 3), 5.0f32)).cast(DType::Int8).unwrap();
        let result =
            x.conv2d().weight(&w).acc_dtype(DType::Int64).call().unwrap().contiguous();
        result.realize_with(&config).unwrap();
        assert_eq!(result.uop().dtype(), DType::Int64);
        assert_eq!(result.as_vec::<i64>().unwrap(), vec![450]);
    }

    fn test_avg_pool2d_int8_promotes_accumulator(config) {
        // 2x2 window of 100 sums to 400: wraps to -112 (mean -28) in int8.
        let x = Tensor::from_ndarray(&Array4::from_elem((1, 1, 2, 2), 100.0f32)).cast(DType::Int8).unwrap();

        // count_include_pad=false divides two promoted int32 sums.
        let counted = x.avg_pool2d().kernel_size(&[2, 2]).count_include_pad(false).call().unwrap().contiguous();
        counted.realize_with(&config).unwrap();
        assert_eq!(counted.as_vec::<i32>().unwrap(), vec![100]);

        // ceil_mode=true takes the third path (sum / pooled-ones sum).
        let ceil = x
            .avg_pool2d()
            .kernel_size(&[2, 2])
            .count_include_pad(true)
            .ceil_mode(true)
            .call()
            .unwrap()
            .contiguous();
        ceil.realize_with(&config).unwrap();
        assert_eq!(ceil.as_vec::<i32>().unwrap(), vec![100]);

        // count_include_pad=true without ceil_mode goes through `mean` (float32).
        let plain = x.avg_pool2d().kernel_size(&[2, 2]).call().unwrap().contiguous();
        plain.realize_with(&config).unwrap();
        assert_eq!(plain.as_vec::<f32>().unwrap(), vec![100.0]);
    }
}

/// DenseNet 2-layer kernel structure regression test.
/// Verifies rangeify produces 6 kernels matching Tinygrad's kernel splitting.
#[test]
fn test_densenet_two_layer_kernel_count() {
    use ndarray::Array4;

    let mk_bn_params = |ch: usize| {
        let mean = Tensor::from_slice(vec![0.0f32; ch]);
        let var = Tensor::from_slice(vec![1.0f32; ch]);
        let gamma = Tensor::from_slice(vec![1.0f32; ch]);
        let beta = Tensor::from_slice(vec![0.0f32; ch]);
        let invstd =
            (&var + Tensor::const_(1e-5f64, svod_dtype::DType::Float32)).try_sqrt().unwrap().reciprocal().unwrap();
        (mean, invstd, gamma, beta)
    };

    let x0 = Tensor::from_ndarray(&Array4::<f32>::ones((1, 64, 14, 14)));

    // Layer 1: BN+ReLU → Conv1x1(128) → BN+ReLU → Conv3x3(32) → Cat
    let (m, inv, g, b) = mk_bn_params(64);
    let bn1 = x0.batchnorm().mean(&m).invstd(&inv).scale(&g).bias(&b).call().unwrap().relu().unwrap();
    let w1x1 = Tensor::from_ndarray(&Array4::<f32>::ones((128, 64, 1, 1)));
    let conv1x1 = bn1.conv2d().weight(&w1x1).call().unwrap();
    let (m, inv, g, b) = mk_bn_params(128);
    let bn2 = conv1x1.batchnorm().mean(&m).invstd(&inv).scale(&g).bias(&b).call().unwrap().relu().unwrap();
    let w3x3 = Tensor::from_ndarray(&Array4::<f32>::ones((32, 128, 3, 3)));
    let conv3x3 = bn2.conv2d().weight(&w3x3).padding(&[(1, 1), (1, 1)]).call().unwrap();
    let cat1 = Tensor::cat(&[&x0, &conv3x3], 1).unwrap();

    // Layer 2: same pattern
    let (m, inv, g, b) = mk_bn_params(96);
    let bn3 = cat1.batchnorm().mean(&m).invstd(&inv).scale(&g).bias(&b).call().unwrap().relu().unwrap();
    let w1x1_2 = Tensor::from_ndarray(&Array4::<f32>::ones((128, 96, 1, 1)));
    let conv1x1_2 = bn3.conv2d().weight(&w1x1_2).call().unwrap();
    let (m, inv, g, b) = mk_bn_params(128);
    let bn4 = conv1x1_2.batchnorm().mean(&m).invstd(&inv).scale(&g).bias(&b).call().unwrap().relu().unwrap();
    let w3x3_2 = Tensor::from_ndarray(&Array4::<f32>::ones((32, 128, 3, 3)));
    let conv3x3_2 = bn4.conv2d().weight(&w3x3_2).padding(&[(1, 1), (1, 1)]).call().unwrap();
    let result = Tensor::cat(&[&cat1, &conv3x3_2], 1).unwrap();

    let uop = result.uop();
    let sink = svod_ir::UOp::sink(vec![uop.contiguous()]);
    // Normalize Buffer→Param before rangeify (matches real pipeline)
    let normalization = crate::realize::normalize_for_schedule_cache(&sink).expect("normalize schedule cache");
    let (rangeified, _ctx) = svod_schedule::rangeify::rangeify(normalization.normalized).unwrap();
    let (kernels_root, _kctx) = svod_schedule::rangeify::try_get_kernel_graph(rangeified)
        .expect("kernel split pipeline should succeed for dense layer kernel count");

    let kernels: Vec<_> =
        kernels_root.toposort().into_iter().filter(|n| matches!(n.op(), svod_ir::Op::Call(..))).collect();

    // 6 kernels matching Tinygrad: BN+ReLU, Conv1x1+BN+ReLU, Conv3x3+Cat (×2 layers)
    assert_eq!(kernels.len(), 6, "Expected 6 kernels for 2 dense layers, got {}", kernels.len());
}

// Full ONNX model kernel count test: onnx/src/test/unit/nn.rs::test_rnnt_encoder_kernel_count
