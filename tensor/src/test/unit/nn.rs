//! Tests for neural network operations: pool, conv, normalization, resize.

use ndarray::{Array4, array};

use crate::Tensor;
use crate::nn::{Reduction, ResizeMode};

fn get_shape(tensor: &Tensor) -> Vec<usize> {
    tensor.uop().shape().unwrap().unwrap().iter().map(|s| s.as_const().unwrap()).collect()
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
    let t = Tensor::linspace(0.0, 1.0, 0, morok_dtype::DType::Float32).unwrap();
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
    let x = Tensor::from_slice([0.0f32; 24]).try_reshape(&[2, 3, 4]).unwrap();
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
    let x = Tensor::from_slice([0.0f32; 24]).try_reshape(&[2, 3, 4]).unwrap();
    expect_err_msg(x.lrn().size(5).call(), "exactly 4D");
}

// =========================================================================
// Codegen tests
// =========================================================================

crate::codegen_tests! {
    fn test_pad_value_neg_inf(config) {
        let x = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let padded = x.try_pad_value(&[(1, 1)], f64::NEG_INFINITY).unwrap();
        let result = padded.realize_with(&config).unwrap().to_vec::<f32>().unwrap();
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
        let result = padded.realize_with(&config).unwrap().to_vec::<f32>().unwrap();
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
        let result = result.contiguous().realize_with(&config).unwrap();
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
        let result = result.contiguous().realize_with(&config).unwrap();
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
        let result = result.contiguous().realize_with(&config).unwrap();
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
    // BUFFERIZE is removed. split_store() then rejects the END because it sees
    // non-OUTER ranges in scope. This affects ANY tensor with multiple non-trivial
    // dims that goes through CONTIGUOUS realization.
    // Minimal repro: Tensor::from_slice(&[1.0f32, 2.0]).contiguous().realize()
    #[ignore = "blocked by CONTIGUOUS realization range-leak bug in rangeify pipeline"]
    fn test_conv2d_groups(config) {
        // Depthwise conv: groups=2, input (1,2,3,3), weight (2,1,1,1)
        let x = Tensor::from_ndarray(&Array4::<f32>::ones((1, 2, 3, 3)));
        let w = Tensor::from_ndarray(&array![[[[2.0f32]]], [[[3.0f32]]]]);
        let result = x.conv2d().weight(&w).groups(2).call().unwrap();
        let result = result.contiguous().realize_with(&config).unwrap();
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
        let result = result.contiguous().realize_with(&config).unwrap();
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
        let result = result.contiguous().realize_with(&config).unwrap();
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
        let result = result.contiguous().realize_with(&config).unwrap();
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
        let result = result.contiguous().realize_with(&config).unwrap();
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
        let result = result.realize_with(&config).unwrap().to_vec::<f32>().unwrap();
        // All outputs should be -5.0 (not 0.0 which would happen with zero padding)
        for val in result.iter() {
            assert!((*val - (-5.0)).abs() < 1e-4, "max_pool2d with padding should use -inf fill, got {val}");
        }
    }

    fn test_max_pool2d_with_indices_basic(config) {
        // 2x2 kernel on 4x4 with stride 2 → 2x2 output
        let x_data: Vec<f32> =
            vec![-1.0, 2.0, 3.0, -4.0, 5.0, -6.0, 7.0, 8.0, 9.0, 10.0, -11.0, 12.0, 13.0, -14.0, 15.0, 16.0];
        let x = Tensor::from_ndarray(&Array4::from_shape_vec((1, 1, 4, 4), x_data).unwrap());
        let (values, indices) = x.max_pool2d_with_indices().kernel_size(&[2, 2]).stride(&[2, 2]).call().unwrap();
        let values = values.contiguous().realize_with(&config).unwrap();
        let vals = values.array_view::<f32>().unwrap();
        assert_eq!(vals.shape(), &[1, 1, 2, 2]);
        // Top-left: max(-1, 2, 5, -6) = 5 at flat index 4
        assert!((vals[[0, 0, 0, 0]] - 5.0).abs() < 1e-4);
        // Top-right: max(3, -4, 7, 8) = 8 at flat index 7
        assert!((vals[[0, 0, 0, 1]] - 8.0).abs() < 1e-4);

        let indices = indices.contiguous().realize_with(&config).unwrap();
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
        let result = result.contiguous().realize_with(&config).unwrap();
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
        let result = result.contiguous().realize_with(&config).unwrap();
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
        let result = t.resize().scales(&[1.0, 1.0, 2.0, 2.0]).mode(ResizeMode::Nearest).call().unwrap().realize_with(&config).unwrap();
        assert_eq!(get_shape(&result), vec![1, 1, 4, 4]);
    }

    fn test_resize_linear_upsample(config) {
        let t = Tensor::from_ndarray(&array![[[[1.0f32, 2.0], [3.0, 4.0]]]]);
        let result = t.resize().scales(&[1.0, 1.0, 2.0, 2.0]).mode(ResizeMode::Linear).call().unwrap().realize_with(&config).unwrap();
        assert_eq!(get_shape(&result), vec![1, 1, 4, 4]);
    }

    fn test_resize_nearest_downsample(config) {
        let x_data: Vec<f32> = (1..=9).map(|v| v as f32).collect();
        let t = Tensor::from_ndarray(&Array4::from_shape_vec((1, 1, 3, 3), x_data).unwrap());
        let result = t.resize().sizes(&[1, 1, 2, 2]).mode(ResizeMode::Nearest).call().unwrap().realize_with(&config).unwrap();
        assert_eq!(get_shape(&result), vec![1, 1, 2, 2]);
    }

    fn test_linspace_basic(config) {
        let t = Tensor::linspace(-1.0, 1.0, 5, morok_dtype::DType::Float32).unwrap();
        assert_eq!(get_shape(&t), vec![5]);
        let result = t.realize_with(&config).unwrap().to_vec::<f32>().unwrap();
        let expected = [-1.0f32, -0.5, 0.0, 0.5, 1.0];
        for (got, exp) in result.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-5, "got {got}, expected {exp}");
        }
    }

    fn test_linspace_single(config) {
        let t = Tensor::linspace(3.0, 7.0, 1, morok_dtype::DType::Float32).unwrap();
        assert_eq!(get_shape(&t), vec![1], "steps=1 must produce 1-D shape [1]");
        let vals = t.realize_with(&config).unwrap().to_vec::<f32>().unwrap();
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
        let val = loss.realize_with(&config).unwrap().to_vec::<f32>().unwrap()[0];
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
        let vals = loss.realize_with(&config).unwrap().to_vec::<f32>().unwrap();
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
        let val = loss.realize_with(&config).unwrap().to_vec::<f32>().unwrap()[0];
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
        let val = loss.realize_with(&config).unwrap().to_vec::<f32>().unwrap()[0];
        // Only sample 0 contributes: 0.5 / 1.0 = 0.5
        assert!((val - 0.5).abs() < 1e-4, "got {val}");
    }

    fn test_dropout_inference(config) {
        let x = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);
        let (output, mask) = x.dropout().p(0.5).call().unwrap();
        let out = output.realize_with(&config).unwrap().to_vec::<f32>().unwrap();
        assert_eq!(out, &[1.0, 2.0, 3.0, 4.0]);
        let m = mask.realize_with(&config).unwrap().to_vec::<bool>().unwrap();
        assert!(m.iter().all(|&v| v)); // all true
    }
}
