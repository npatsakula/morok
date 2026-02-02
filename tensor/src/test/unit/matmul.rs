use crate::*;
use morok_dtype::DType;
use morok_schedule::{OptStrategy, OptimizerConfig};

// ========== Basic 2D x 2D Tests ==========

#[test]
fn test_matmul_2d_basic() {
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]).try_reshape(&[2, 2]).unwrap();
    let b = Tensor::from_slice([5.0f32, 6.0, 7.0, 8.0]).try_reshape(&[2, 2]).unwrap();
    let c = a.dot(&b).unwrap();

    let c_shape = c.shape().unwrap();
    assert_eq!(c_shape.len(), 2);
    assert_eq!(c_shape[0].as_const().unwrap(), 2);
    assert_eq!(c_shape[1].as_const().unwrap(), 2);
}

#[test]
fn test_matmul_2d_non_square() {
    // [2, 3] @ [3, 4] → [2, 4]
    let a = Tensor::from_slice([1.0f32; 6]).try_reshape(&[2, 3]).unwrap();
    let b = Tensor::from_slice([1.0f32; 12]).try_reshape(&[3, 4]).unwrap();
    let c = a.dot(&b).unwrap();

    let c_shape = c.shape().unwrap();
    assert_eq!(c_shape.len(), 2);
    assert_eq!(c_shape[0].as_const().unwrap(), 2);
    assert_eq!(c_shape[1].as_const().unwrap(), 4);
}

#[test]
fn test_matmul_alias() {
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]).try_reshape(&[2, 2]).unwrap();
    let b = Tensor::from_slice([5.0f32, 6.0, 7.0, 8.0]).try_reshape(&[2, 2]).unwrap();

    // Test that matmul is an alias for dot
    let c1 = a.dot(&b).unwrap();
    let c2 = a.matmul(&b).unwrap();

    assert_eq!(c1.shape().unwrap().len(), c2.shape().unwrap().len());
}

// ========== 1D x 1D Tests (Dot Product) ==========

#[test]
fn test_dot_product_1d() {
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let b = Tensor::from_slice([4.0f32, 5.0, 6.0]);
    let c = a.dot(&b).unwrap();

    // Result should be scalar (0D tensor)
    let c_shape = c.shape().unwrap();
    assert_eq!(c_shape.len(), 0);
}

#[test]
fn test_dot_product_orthogonal() {
    let a = Tensor::from_slice([1.0f32, 0.0, 0.0]);
    let b = Tensor::from_slice([0.0f32, 1.0, 0.0]);
    let c = a.dot(&b).unwrap();

    // Orthogonal vectors → dot product = 0
    let c_shape = c.shape().unwrap();
    assert_eq!(c_shape.len(), 0);
}

// ========== 1D x 2D and 2D x 1D Tests ==========

#[test]
fn test_vector_matrix() {
    // [3] @ [3, 4] → [4]
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let b = Tensor::from_slice([1.0f32; 12]).try_reshape(&[3, 4]).unwrap();
    let c = a.dot(&b).unwrap();

    let c_shape = c.shape().unwrap();
    assert_eq!(c_shape.len(), 1);
    assert_eq!(c_shape[0].as_const().unwrap(), 4);
}

#[test]
fn test_matrix_vector() {
    // [2, 3] @ [3] → [2]
    let a = Tensor::from_slice([1.0f32; 6]).try_reshape(&[2, 3]).unwrap();
    let b = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let c = a.dot(&b).unwrap();

    let c_shape = c.shape().unwrap();
    assert_eq!(c_shape.len(), 1);
    assert_eq!(c_shape[0].as_const().unwrap(), 2);
}

// ========== Batched Matmul Tests ==========

#[test]
fn test_batched_matmul_3d() {
    // [2, 3, 4] @ [2, 4, 5] → [2, 3, 5]
    let a = Tensor::from_slice([1.0f32; 24]).try_reshape(&[2, 3, 4]).unwrap();
    let b = Tensor::from_slice([1.0f32; 40]).try_reshape(&[2, 4, 5]).unwrap();
    let c = a.dot(&b).unwrap();

    let c_shape = c.shape().unwrap();
    assert_eq!(c_shape.len(), 3);
    assert_eq!(c_shape[0].as_const().unwrap(), 2);
    assert_eq!(c_shape[1].as_const().unwrap(), 3);
    assert_eq!(c_shape[2].as_const().unwrap(), 5);
}

// ========== Edge Cases ==========

#[test]
fn test_matmul_error_0d() {
    let scalar = Tensor::from_slice([1.0f32]).try_reshape(&[]).unwrap();
    let vector = Tensor::from_slice([1.0f32, 2.0, 3.0]);

    // 0D tensors not supported
    assert!(scalar.dot(&vector).is_err());
    assert!(vector.dot(&scalar).is_err());
}

#[test]
fn test_matmul_error_shape_mismatch() {
    // [2, 3] @ [4, 5] - inner dimensions don't match
    let a = Tensor::from_slice([1.0f32; 6]).try_reshape(&[2, 3]).unwrap();
    let b = Tensor::from_slice([1.0f32; 20]).try_reshape(&[4, 5]).unwrap();

    let result = a.dot(&b);
    assert!(result.is_err());
}

#[test]
fn test_matmul_identity() {
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]).try_reshape(&[2, 2]).unwrap();
    let identity = Tensor::from_slice([1.0f32, 0.0, 0.0, 1.0]).try_reshape(&[2, 2]).unwrap();

    let result = a.dot(&identity).unwrap();

    // Result shape should match input
    let result_shape = result.shape().unwrap();
    assert_eq!(result_shape.len(), 2);
    assert_eq!(result_shape[0].as_const().unwrap(), 2);
    assert_eq!(result_shape[1].as_const().unwrap(), 2);
}

// ========== Dtype Tests ==========

#[test]
fn test_matmul_dtype_promotion() {
    let a = Tensor::from_slice([1i32, 2, 3, 4]).try_reshape(&[2, 2]).unwrap();
    let b = Tensor::from_slice([5.0f32, 6.0, 7.0, 8.0]).try_reshape(&[2, 2]).unwrap();

    let c = a.dot(&b).unwrap();
    // Result should be promoted to float32
    assert_eq!(c.uop().dtype(), DType::Float32);
}

#[test]
fn test_matmul_explicit_dtype() {
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]).try_reshape(&[2, 2]).unwrap();
    let b = Tensor::from_slice([5.0f32, 6.0, 7.0, 8.0]).try_reshape(&[2, 2]).unwrap();

    // Use float64 accumulation
    let c = a.matmul_with().other(&b).dtype(DType::Float64).call().unwrap();
    assert_eq!(c.uop().dtype(), DType::Float64);
}

#[test]
#[ignore] // Run with: cargo test -p morok-tensor test_print_matmul_ir -- --ignored --nocapture
fn test_print_matmul_ir() {
    // Create 4x4 matmul to see generated IR
    let a = Tensor::from_slice((0..16).map(|i| i as f32).collect::<Vec<_>>()).try_reshape(&[4, 4]).unwrap();
    let b = Tensor::from_slice((0..16).map(|i| i as f32).collect::<Vec<_>>()).try_reshape(&[4, 4]).unwrap();
    let c = a.matmul(&b).unwrap();

    let plan = c.prepare().expect("prepare should succeed");

    println!("\n=== Generated Kernels ===\n");
    for kernel in plan.kernels() {
        println!("--- {} ({}) ---", kernel.entry_point, kernel.device);
        println!("{}", kernel.code);
        println!();
    }
}

#[test]
#[ignore] // Run with: cargo test -p morok-tensor test_print_matmul_512x512_ir -- --ignored --nocapture
// #[tracing_test::traced_test]
fn test_print_matmul_512x512_ir() {
    const SIZE: usize = 512;
    let a = Tensor::from_slice((0..SIZE * SIZE).map(|i| (i as f32) * 0.01).collect::<Vec<_>>())
        .try_reshape(&[SIZE as _, SIZE as _])
        .unwrap();
    let b = Tensor::from_slice((0..SIZE * SIZE).map(|i| (i as f32) * 0.01).collect::<Vec<_>>())
        .try_reshape(&[SIZE as _, SIZE as _])
        .unwrap();
    let c = a.matmul(&b).unwrap();

    // Use Heuristic strategy (Beam has a pre-existing bug with horizontal reduction)
    let config = OptimizerConfig::builder().strategy(OptStrategy::Heuristic).build();
    let plan = c.prepare_with(&config).expect("prepare should succeed");

    println!("\n=== Generated Kernels (64x64 with output upcast) ===\n");
    for kernel in plan.kernels() {
        println!("--- {} ({}) ---", kernel.entry_point, kernel.device);
        println!("{}", kernel.code);
        println!();
    }
}

#[test]
// #[ignore] // Run with: cargo test -p morok-tensor test_beam_search_matmul -- --ignored --nocapture
// #[tracing_test::traced_test]
fn test_beam_search_matmul() {
    // Test beam search optimization for matmul - reproduces float vector index bug
    let size = 512; // Original size that triggered the bug
    let a = Tensor::from_slice((0..size * size).map(|i| (i as f32) * 0.01).collect::<Vec<_>>())
        .try_reshape(&[size as isize, size as isize])
        .unwrap();
    let b = Tensor::from_slice((0..size * size).map(|i| (i as f32) * 0.01).collect::<Vec<_>>())
        .try_reshape(&[size as isize, size as isize])
        .unwrap();
    let c = a.matmul(&b).unwrap();

    // Use width=2 for reasonable test time
    let beam_config = OptimizerConfig::builder().strategy(OptStrategy::Beam { width: 2 }).build();

    let plan = c.prepare_with(&beam_config).expect("beam search prepare should succeed");

    println!("\n=== Beam Search Kernels ({}x{}) ===\n", size, size);
    for kernel in plan.kernels() {
        println!("--- {} ({}) ---", kernel.entry_point, kernel.device);
        println!("{}", kernel.code);
        println!();
    }
}

// ========== Linear Layer Tests ==========

#[test]
fn test_linear_basic() {
    // input: [1, 3], weight: [2, 3], bias: [2]
    let input = Tensor::from_slice([1.0f32, 2.0, 3.0]).try_reshape(&[1, 3]).unwrap();
    let weight = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).try_reshape(&[2, 3]).unwrap();
    let bias = Tensor::from_slice([0.1f32, 0.2]);

    let result = input.linear().weight(&weight).bias(&bias).call().unwrap();

    let result_shape = result.shape().unwrap();
    assert_eq!(result_shape.len(), 2);
    assert_eq!(result_shape[0].as_const().unwrap(), 1);
    assert_eq!(result_shape[1].as_const().unwrap(), 2);
}

#[test]
fn test_linear_no_bias() {
    let input = Tensor::from_slice([1.0f32, 2.0, 3.0]).try_reshape(&[1, 3]).unwrap();
    let weight = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).try_reshape(&[2, 3]).unwrap();

    let result = input.linear().weight(&weight).call().unwrap();

    let result_shape = result.shape().unwrap();
    assert_eq!(result_shape.len(), 2);
    assert_eq!(result_shape[0].as_const().unwrap(), 1);
    assert_eq!(result_shape[1].as_const().unwrap(), 2);
}

#[test]
fn test_linear_batched() {
    // input: [4, 3], weight: [2, 3] → output: [4, 2]
    let input = Tensor::from_slice([1.0f32; 12]).try_reshape(&[4, 3]).unwrap();
    let weight = Tensor::from_slice([1.0f32; 6]).try_reshape(&[2, 3]).unwrap();

    let result = input.linear().weight(&weight).call().unwrap();

    let result_shape = result.shape().unwrap();
    assert_eq!(result_shape.len(), 2);
    assert_eq!(result_shape[0].as_const().unwrap(), 4);
    assert_eq!(result_shape[1].as_const().unwrap(), 2);
}

#[test]
fn test_linear_1d_weight() {
    // Test 1D weight case (element-wise multiply)
    let input = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let weight = Tensor::from_slice([2.0f32, 3.0, 4.0]);

    let result = input.linear().weight(&weight).call().unwrap();

    let result_shape = result.shape().unwrap();
    assert_eq!(result_shape.len(), 1);
    assert_eq!(result_shape[0].as_const().unwrap(), 3);
}

// ========== Minimal VECTORIZE Normalization Test ==========

#[test]
// #[tracing_test::traced_test]
fn test_vectorize_normalize_minimal() {
    // Test 64x64 matmul with vectorization enabled
    let a = Tensor::from_slice([1.0f32; 64 * 64]).try_reshape(&[64, 64]).unwrap();
    let b = Tensor::from_slice([1.0f32; 64 * 64]).try_reshape(&[64, 64]).unwrap();
    let c = a.matmul(&b).unwrap();

    // Explicit config to avoid test pollution from shared global state
    // Note: default config has devectorize_alu=true (converts vector ALU to scalar)
    let config = OptimizerConfig::builder().strategy(OptStrategy::Heuristic).build();
    let result = c.realize_with(&config);
    assert!(result.is_ok(), "realize failed: {:?}", result.err());
}

// ========== 512x512 Vectorized Test (for UPCAST debugging) ==========

#[test]
// #[tracing_test::traced_test]
fn test_matmul_512x512_vectorized() {
    // Create 512x512 matrices filled with 1.0
    const SIZE: usize = 512;
    let a = Tensor::from_slice(vec![1.0f32; SIZE * SIZE]).try_reshape(&[SIZE as _, SIZE as _]).unwrap();
    let b = Tensor::from_slice(vec![1.0f32; SIZE * SIZE]).try_reshape(&[SIZE as _, SIZE as _]).unwrap();
    let c = a.matmul(&b).unwrap();

    // Use from_env() to respect MOROK_OUTPUT_UPCAST and other env vars
    // Note: Beam search has a pre-existing bug with horizontal reduction, using Heuristic
    let config = OptimizerConfig::from_env();
    let c = c.realize_with(&config).unwrap();
    let result = c.to_ndarray::<f32>().unwrap();

    // Each element should be 512 (sum of 512 ones)
    assert_eq!(result.len(), SIZE * SIZE);
    assert!((result[[0, 0]] - SIZE as f32).abs() < 0.01, "Expected {}, got {}", SIZE, result[[0, 0]]);
}

#[test]
fn test_matmul_64x64_vectorized() {
    // Create 64x64 matrices filled with 1.0
    const SIZE: usize = 64;
    let a = Tensor::from_slice(vec![1.0f32; SIZE * SIZE]).try_reshape(&[SIZE as _, SIZE as _]).unwrap();
    let b = Tensor::from_slice(vec![1.0f32; SIZE * SIZE]).try_reshape(&[SIZE as _, SIZE as _]).unwrap();
    let c = a.matmul(&b).unwrap();

    let config = OptimizerConfig::from_env();
    let c = c.realize_with(&config).unwrap();
    let result = c.to_ndarray::<f32>().unwrap();

    // Each element should be 64 (sum of 64 ones)
    assert_eq!(result.len(), SIZE * SIZE);
    assert!((result[[0, 0]] - SIZE as f32).abs() < 0.01, "Expected {}, got {}", SIZE, result[[0, 0]]);
}

#[test]
#[ignore] // Run with: cargo test -p morok-tensor test_print_matmul_64x64_ir -- --ignored --nocapture
// #[tracing_test::traced_test]
fn test_print_matmul_64x64_ir() {
    const SIZE: usize = 64;
    let a = Tensor::from_slice(vec![1.0f32; SIZE * SIZE]).try_reshape(&[SIZE as _, SIZE as _]).unwrap();
    let b = Tensor::from_slice(vec![1.0f32; SIZE * SIZE]).try_reshape(&[SIZE as _, SIZE as _]).unwrap();
    let c = a.matmul(&b).unwrap();

    let config = OptimizerConfig::from_env();
    let plan = c.prepare_with(&config).expect("prepare should succeed");

    println!("\n=== Generated Kernels (64x64 matmul) ===\n");
    for kernel in plan.kernels() {
        println!("--- {} ({}) ---", kernel.entry_point, kernel.device);
        println!("{}", kernel.code);
        println!();
    }
}

// ========== Validated Matmul Tests (against ndarray reference) ==========

use ndarray::{Array2, ArrayD};

/// Helper to compare morok result against ndarray reference with tolerance.
fn assert_matmul_close(actual: &ArrayD<f32>, expected: &Array2<f32>, tol: f32) {
    let expected_shape: Vec<usize> = expected.shape().to_vec();
    assert_eq!(actual.shape(), expected_shape.as_slice(), "Shape mismatch");

    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!((a - e).abs() < tol, "Mismatch at index {}: morok={} vs ndarray={} (diff: {})", i, a, e, (a - e).abs());
    }
}

#[test]
// #[tracing_test::traced_test]
fn test_matmul_validated_2x2() {
    // Simple 2x2 matmul with known values
    let a_data = [1.0f32, 2.0, 3.0, 4.0];
    let b_data = [5.0f32, 6.0, 7.0, 8.0];

    // Compute with morok
    let a = Tensor::from_slice(a_data).try_reshape(&[2, 2]).unwrap();
    let b = Tensor::from_slice(b_data).try_reshape(&[2, 2]).unwrap();
    let c = a.matmul(&b).unwrap().realize().unwrap();
    let morok_result = c.to_ndarray::<f32>().unwrap();

    // Compute reference with ndarray
    let a_nd = Array2::from_shape_vec((2, 2), a_data.to_vec()).unwrap();
    let b_nd = Array2::from_shape_vec((2, 2), b_data.to_vec()).unwrap();
    let expected = a_nd.dot(&b_nd);

    // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
    assert_matmul_close(&morok_result, &expected, 1e-5);
}

#[test]
fn test_matmul_validated_3x3() {
    // 3x3 matmul with sequential values
    let a_data: Vec<f32> = (1..=9).map(|x| x as f32).collect();
    let b_data: Vec<f32> = (10..=18).map(|x| x as f32).collect();

    let a = Tensor::from_slice(&a_data).try_reshape(&[3, 3]).unwrap();
    let b = Tensor::from_slice(&b_data).try_reshape(&[3, 3]).unwrap();
    let c = a.matmul(&b).unwrap().realize().unwrap();
    let morok_result = c.to_ndarray::<f32>().unwrap();

    let a_nd = Array2::from_shape_vec((3, 3), a_data).unwrap();
    let b_nd = Array2::from_shape_vec((3, 3), b_data).unwrap();
    let expected = a_nd.dot(&b_nd);

    assert_matmul_close(&morok_result, &expected, 1e-4);
}

#[test]
fn test_matmul_validated_2x3_3x4() {
    // [2, 3] @ [3, 4] -> [2, 4]
    let a_data: Vec<f32> = (1..=6).map(|x| x as f32).collect();
    let b_data: Vec<f32> = (1..=12).map(|x| x as f32).collect();

    let a = Tensor::from_slice(&a_data).try_reshape(&[2, 3]).unwrap();
    let b = Tensor::from_slice(&b_data).try_reshape(&[3, 4]).unwrap();
    let c = a.matmul(&b).unwrap().realize().unwrap();
    let morok_result = c.to_ndarray::<f32>().unwrap();

    let a_nd = Array2::from_shape_vec((2, 3), a_data).unwrap();
    let b_nd = Array2::from_shape_vec((3, 4), b_data).unwrap();
    let expected = a_nd.dot(&b_nd);

    assert_matmul_close(&morok_result, &expected, 1e-4);
}

#[test]
fn test_matmul_validated_tall_wide() {
    // [4, 2] @ [2, 5] -> [4, 5]
    let a_data: Vec<f32> = (1..=8).map(|x| x as f32 * 0.5).collect();
    let b_data: Vec<f32> = (1..=10).map(|x| x as f32 * 0.3).collect();

    let a = Tensor::from_slice(&a_data).try_reshape(&[4, 2]).unwrap();
    let b = Tensor::from_slice(&b_data).try_reshape(&[2, 5]).unwrap();
    let c = a.matmul(&b).unwrap().realize().unwrap();
    let morok_result = c.to_ndarray::<f32>().unwrap();

    let a_nd = Array2::from_shape_vec((4, 2), a_data).unwrap();
    let b_nd = Array2::from_shape_vec((2, 5), b_data).unwrap();
    let expected = a_nd.dot(&b_nd);

    assert_matmul_close(&morok_result, &expected, 1e-5);
}

#[test]
// #[tracing_test::traced_test]
fn test_matmul_validated_16x16() {
    // Larger matrix to test vectorization paths
    const SIZE: usize = 16;
    let a_data: Vec<f32> = (0..SIZE * SIZE).map(|x| (x as f32) * 0.1).collect();
    let b_data: Vec<f32> = (0..SIZE * SIZE).map(|x| (x as f32) * 0.05 + 1.0).collect();

    let a = Tensor::from_slice(&a_data).try_reshape(&[SIZE as _, SIZE as _]).unwrap();
    let b = Tensor::from_slice(&b_data).try_reshape(&[SIZE as _, SIZE as _]).unwrap();
    let c = a.matmul(&b).unwrap().realize().unwrap();
    let morok_result = c.to_ndarray::<f32>().unwrap();

    let a_nd = Array2::from_shape_vec((SIZE, SIZE), a_data).unwrap();
    let b_nd = Array2::from_shape_vec((SIZE, SIZE), b_data).unwrap();
    let expected = a_nd.dot(&b_nd);

    assert_matmul_close(&morok_result, &expected, 1e-3);
}

#[test]
#[tracing_test::traced_test]
fn test_matmul_validated_32x32() {
    // Test with 32x32 to exercise more optimization paths
    const SIZE: usize = 32;
    let a_data: Vec<f32> = (0..SIZE * SIZE).map(|x| ((x % 17) as f32) * 0.1 - 0.8).collect();
    let b_data: Vec<f32> = (0..SIZE * SIZE).map(|x| ((x % 13) as f32) * 0.15 - 0.5).collect();

    let a = Tensor::from_slice(&a_data).try_reshape(&[SIZE as _, SIZE as _]).unwrap();
    let b = Tensor::from_slice(&b_data).try_reshape(&[SIZE as _, SIZE as _]).unwrap();
    let c = a.matmul(&b).unwrap().realize().unwrap();
    let morok_result = c.to_ndarray::<f32>().unwrap();

    let a_nd = Array2::from_shape_vec((SIZE, SIZE), a_data).unwrap();
    let b_nd = Array2::from_shape_vec((SIZE, SIZE), b_data).unwrap();
    let expected = a_nd.dot(&b_nd);

    assert_matmul_close(&morok_result, &expected, 1e-2);
}

#[test]
fn test_matmul_validated_64x64() {
    // 64x64 test with varied data
    const SIZE: usize = 64;
    let a_data: Vec<f32> = (0..SIZE * SIZE).map(|x| ((x as f32) * 0.01).sin()).collect();
    let b_data: Vec<f32> = (0..SIZE * SIZE).map(|x| ((x as f32) * 0.02).cos()).collect();

    let a = Tensor::from_slice(&a_data).try_reshape(&[SIZE as _, SIZE as _]).unwrap();
    let b = Tensor::from_slice(&b_data).try_reshape(&[SIZE as _, SIZE as _]).unwrap();

    let config = OptimizerConfig::from_env();
    let c = a.matmul(&b).unwrap().realize_with(&config).unwrap();
    let morok_result = c.to_ndarray::<f32>().unwrap();

    let a_nd = Array2::from_shape_vec((SIZE, SIZE), a_data).unwrap();
    let b_nd = Array2::from_shape_vec((SIZE, SIZE), b_data).unwrap();
    let expected = a_nd.dot(&b_nd);

    // Larger tolerance for accumulated floating point error
    assert_matmul_close(&morok_result, &expected, 1e-1);
}

#[test]
fn test_dot_product_validated() {
    // 1D @ 1D dot product
    let a_data = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let b_data = [2.0f32, 3.0, 4.0, 5.0, 6.0];

    let a = Tensor::from_slice(a_data);
    let b = Tensor::from_slice(b_data);
    let c = a.dot(&b).unwrap().realize().unwrap();
    let morok_result = c.to_ndarray::<f32>().unwrap();

    // Expected: 1*2 + 2*3 + 3*4 + 4*5 + 5*6 = 2 + 6 + 12 + 20 + 30 = 70
    let expected: f32 = a_data.iter().zip(b_data.iter()).map(|(a, b)| a * b).sum();

    assert_eq!(morok_result.ndim(), 0, "Dot product should be scalar");
    assert!((morok_result[[]] - expected).abs() < 1e-5, "Expected {}, got {}", expected, morok_result[[]]);
}

#[test]
fn test_vector_matrix_validated() {
    // [4] @ [4, 3] -> [3]
    let v_data = [1.0f32, 2.0, 3.0, 4.0];
    let m_data: Vec<f32> = (1..=12).map(|x| x as f32).collect();

    let v = Tensor::from_slice(v_data);
    let m = Tensor::from_slice(&m_data).try_reshape(&[4, 3]).unwrap();
    let c = v.dot(&m).unwrap().realize().unwrap();
    let morok_result = c.to_ndarray::<f32>().unwrap();

    // ndarray: need to treat vector as [1, 4] @ [4, 3] -> [1, 3], then squeeze
    let v_nd = ndarray::Array1::from_vec(v_data.to_vec());
    let m_nd = Array2::from_shape_vec((4, 3), m_data).unwrap();
    let expected = v_nd.dot(&m_nd);

    assert_eq!(morok_result.shape(), &[3]);
    for (i, (a, e)) in morok_result.iter().zip(expected.iter()).enumerate() {
        assert!((a - e).abs() < 1e-5, "Mismatch at index {}: {} != {}", i, a, e);
    }
}

#[test]
fn test_matrix_vector_validated() {
    // [3, 4] @ [4] -> [3]
    let m_data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let v_data = [1.0f32, 2.0, 3.0, 4.0];

    let m = Tensor::from_slice(&m_data).try_reshape(&[3, 4]).unwrap();
    let v = Tensor::from_slice(v_data);
    let c = m.dot(&v).unwrap().realize().unwrap();
    let morok_result = c.to_ndarray::<f32>().unwrap();

    let m_nd = Array2::from_shape_vec((3, 4), m_data).unwrap();
    let v_nd = ndarray::Array1::from_vec(v_data.to_vec());
    let expected = m_nd.dot(&v_nd);

    assert_eq!(morok_result.shape(), &[3]);
    for (i, (a, e)) in morok_result.iter().zip(expected.iter()).enumerate() {
        assert!((a - e).abs() < 1e-5, "Mismatch at index {}: {} != {}", i, a, e);
    }
}

#[test]
fn test_matmul_identity_validated() {
    // A @ I = A
    let a_data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let identity = [1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];

    let a = Tensor::from_slice(&a_data).try_reshape(&[4, 4]).unwrap();
    let i = Tensor::from_slice(identity).try_reshape(&[4, 4]).unwrap();
    let c = a.matmul(&i).unwrap().realize().unwrap();
    let morok_result = c.to_ndarray::<f32>().unwrap();

    // Result should equal original A
    for (i, (actual, expected)) in morok_result.iter().zip(a_data.iter()).enumerate() {
        assert!((actual - expected).abs() < 1e-5, "Mismatch at index {}: {} != {}", i, actual, expected);
    }
}

#[test]
fn test_matmul_negative_values_validated() {
    // Test with negative values to ensure sign handling
    let a_data = [-1.0f32, 2.0, -3.0, 4.0, -5.0, 6.0];
    let b_data = [1.0f32, -2.0, 3.0, -4.0, 5.0, -6.0]; // [3, 2] = 6 elements

    let a = Tensor::from_slice(a_data).try_reshape(&[2, 3]).unwrap();
    let b = Tensor::from_slice(b_data).try_reshape(&[3, 2]).unwrap().try_transpose(0, 1).unwrap();
    let b = b.try_transpose(0, 1).unwrap(); // Back to [3, 2] but contiguous
    let c = a.matmul(&b).unwrap().realize().unwrap();
    let morok_result = c.to_ndarray::<f32>().unwrap();

    let a_nd = Array2::from_shape_vec((2, 3), a_data.to_vec()).unwrap();
    let b_nd = Array2::from_shape_vec((3, 2), b_data.to_vec()).unwrap();
    let expected = a_nd.dot(&b_nd);

    assert_matmul_close(&morok_result, &expected, 1e-5);
}

// ========== Large Dimension Validated Tests ==========

use test_case::test_case;

/// Helper to run validated square matmul test for a given size.
fn run_validated_square_matmul(size: usize, tol: f32) {
    // Use prime modulos to create varied but reproducible data
    let a_data: Vec<f32> = (0..size * size).map(|x| ((x % 31) as f32) * 0.05 - 0.8).collect();
    let b_data: Vec<f32> = (0..size * size).map(|x| ((x % 37) as f32) * 0.04 - 0.7).collect();

    let a = Tensor::from_slice(&a_data).try_reshape(&[size as isize, size as isize]).unwrap();
    let b = Tensor::from_slice(&b_data).try_reshape(&[size as isize, size as isize]).unwrap();

    let config = OptimizerConfig::from_env();
    let c = a.matmul(&b).unwrap().realize_with(&config).unwrap();
    let morok_result = c.to_ndarray::<f32>().unwrap();

    let a_nd = Array2::from_shape_vec((size, size), a_data).unwrap();
    let b_nd = Array2::from_shape_vec((size, size), b_data).unwrap();
    let expected = a_nd.dot(&b_nd);

    assert_matmul_close(&morok_result, &expected, tol);
}

/// Helper to run validated non-square matmul test.
fn run_validated_matmul(m: usize, k: usize, n: usize, tol: f32) {
    let a_data: Vec<f32> = (0..m * k).map(|x| ((x % 41) as f32) * 0.04 - 0.8).collect();
    let b_data: Vec<f32> = (0..k * n).map(|x| ((x % 43) as f32) * 0.035 - 0.7).collect();

    let a = Tensor::from_slice(&a_data).try_reshape(&[m as isize, k as isize]).unwrap();
    let b = Tensor::from_slice(&b_data).try_reshape(&[k as isize, n as isize]).unwrap();

    let config = OptimizerConfig::from_env();
    let c = a.matmul(&b).unwrap().realize_with(&config).unwrap();
    let morok_result = c.to_ndarray::<f32>().unwrap();

    let a_nd = Array2::from_shape_vec((m, k), a_data).unwrap();
    let b_nd = Array2::from_shape_vec((k, n), b_data).unwrap();
    let expected = a_nd.dot(&b_nd);

    assert_eq!(morok_result.shape(), &[m, n], "Output shape mismatch");
    assert_matmul_close(&morok_result, &expected, tol);
}

// Square matrix tests with increasing sizes
#[test_case(128, 0.5; "128x128")]
#[test_case(256, 1.0; "256x256")]
#[test_case(500, 1.5; "500x500 non-power-of-2")]
#[test_case(512, 2.0; "512x512")]
#[test_case(1024, 3.0; "1024x1024")]
fn test_matmul_validated_square(size: usize, tol: f32) {
    run_validated_square_matmul(size, tol);
}

// Non-square matrix tests
#[test_case(512, 256, 384, 2.0; "512x256 @ 256x384")]
#[test_case(1024, 64, 128, 1.0; "1024x64 @ 64x128 tall-skinny")]
#[test_case(64, 512, 64, 1.5; "64x512 @ 512x64 wide")]
#[test_case(256, 1024, 256, 2.5; "256x1024 @ 1024x256 large-K")]
fn test_matmul_validated_non_square(m: usize, k: usize, n: usize, tol: f32) {
    run_validated_matmul(m, k, n, tol);
}
