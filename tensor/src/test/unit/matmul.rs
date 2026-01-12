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
#[ignore] // Run with: cargo test -p morok-tensor test_print_matmul_64x64_ir -- --ignored --nocapture
#[tracing_test::traced_test]
fn test_print_matmul_64x64_ir() {
    // Create 64x64 matmul to see vectorized IR (output upcast)
    let a =
        Tensor::from_slice((0..64 * 64).map(|i| (i as f32) * 0.01).collect::<Vec<_>>()).try_reshape(&[64, 64]).unwrap();
    let b =
        Tensor::from_slice((0..64 * 64).map(|i| (i as f32) * 0.01).collect::<Vec<_>>()).try_reshape(&[64, 64]).unwrap();
    let c = a.matmul(&b).unwrap();

    let plan = c.prepare().expect("prepare should succeed");

    println!("\n=== Generated Kernels (64x64 with output upcast) ===\n");
    for kernel in plan.kernels() {
        println!("--- {} ({}) ---", kernel.entry_point, kernel.device);
        println!("{}", kernel.code);
        println!();
    }
}

#[test]
#[ignore] // Run with: cargo test -p morok-tensor test_beam_search_matmul -- --ignored --nocapture
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

// ========== 64x64 Vectorized Test (for UPCAST debugging) ==========

#[test]
fn test_matmul_64x64_vectorized() {
    // Create 64x64 matrices filled with 1.0
    let a = Tensor::from_slice([1.0f32; 64 * 64]).try_reshape(&[64, 64]).unwrap();
    let b = Tensor::from_slice([1.0f32; 64 * 64]).try_reshape(&[64, 64]).unwrap();
    let c = a.matmul(&b).unwrap();
    let result = c.realize().unwrap().to_ndarray::<f32>().unwrap();

    // Each element should be 64 (sum of 64 ones)
    assert_eq!(result.len(), 64 * 64);
    assert!((result[[0, 0]] - 64.0).abs() < 0.01, "Expected 64.0, got {}", result[[0, 0]]);
}
