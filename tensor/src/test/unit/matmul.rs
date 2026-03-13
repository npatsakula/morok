use crate::*;
use morok_dtype::DType;
use morok_schedule::{BeamConfig, OptStrategy, OptimizerConfig};
use ndarray::{Array2, array};

fn prep_config(optimizer: OptimizerConfig) -> PrepareConfig {
    optimizer.into()
}
fn env_config() -> PrepareConfig {
    PrepareConfig::from_env()
}

/// Helper to compare morok result against ndarray reference with tolerance.
fn assert_matmul_close(actual: &[f32], expected: &Array2<f32>, tol: f32) {
    let expected_flat: Vec<f32> = expected.iter().copied().collect();
    assert_eq!(actual.len(), expected_flat.len(), "Length mismatch: {} != {}", actual.len(), expected_flat.len());

    for (i, (a, e)) in actual.iter().zip(expected_flat.iter()).enumerate() {
        assert!((a - e).abs() < tol, "Mismatch at index {}: morok={} vs ndarray={} (diff: {})", i, a, e, (a - e).abs());
    }
}

/// Helper to run validated square matmul test for a given size.
fn run_validated_square_matmul(size: usize, tol: f32) {
    // Use prime modulos to create varied but reproducible data
    let a_data: Vec<f32> = (0..size * size).map(|x| ((x % 31) as f32) * 0.05 - 0.8).collect();
    let b_data: Vec<f32> = (0..size * size).map(|x| ((x % 37) as f32) * 0.04 - 0.7).collect();

    let a_nd = Array2::from_shape_vec((size, size), a_data).unwrap();
    let b_nd = Array2::from_shape_vec((size, size), b_data).unwrap();
    let a = Tensor::from_ndarray(&a_nd);
    let b = Tensor::from_ndarray(&b_nd);

    let config = env_config();
    let c = a.matmul(&b).unwrap().realize_with(&config).unwrap();

    let expected = a_nd.dot(&b_nd);

    assert_matmul_close(&c.to_vec::<f32>().unwrap(), &expected, tol);
}

/// Helper to run validated non-square matmul test.
fn run_validated_matmul(m: usize, k: usize, n: usize, tol: f32) {
    let a_data: Vec<f32> = (0..m * k).map(|x| ((x % 41) as f32) * 0.04 - 0.8).collect();
    let b_data: Vec<f32> = (0..k * n).map(|x| ((x % 43) as f32) * 0.035 - 0.7).collect();

    let a_nd = Array2::from_shape_vec((m, k), a_data).unwrap();
    let b_nd = Array2::from_shape_vec((k, n), b_data).unwrap();
    let a = Tensor::from_ndarray(&a_nd);
    let b = Tensor::from_ndarray(&b_nd);

    let config = env_config();
    let c = a.matmul(&b).unwrap().realize_with(&config).unwrap();

    let c_shape = c.shape().unwrap();
    assert_eq!(c_shape[0].as_const().unwrap(), m, "Output shape mismatch");
    assert_eq!(c_shape[1].as_const().unwrap(), n, "Output shape mismatch");

    let expected = a_nd.dot(&b_nd);

    assert_matmul_close(&c.to_vec::<f32>().unwrap(), &expected, tol);
}

// =========================================================================
// Validated matmul tests (codegen required)
// =========================================================================

crate::codegen_tests! {
    fn test_matmul_validated_2x2(config) {
        // Simple 2x2 matmul with known values
        let a_nd = Array2::from_shape_vec((2, 2), vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let b_nd = Array2::from_shape_vec((2, 2), vec![5.0f32, 6.0, 7.0, 8.0]).unwrap();

        // Compute with morok
        let a = Tensor::from_ndarray(&a_nd);
        let b = Tensor::from_ndarray(&b_nd);
        let c = a.matmul(&b).unwrap().realize_with(&config).unwrap();

        // Compute reference with ndarray
        let expected = a_nd.dot(&b_nd);

        // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
        assert_matmul_close(&c.to_vec::<f32>().unwrap(), &expected, 1e-5);
    }

    fn test_matmul_validated_3x3(config) {
        // 3x3 matmul with sequential values
        let a_data: Vec<f32> = (1..=9).map(|x| x as f32).collect();
        let b_data: Vec<f32> = (10..=18).map(|x| x as f32).collect();

        let a_nd = Array2::from_shape_vec((3, 3), a_data).unwrap();
        let b_nd = Array2::from_shape_vec((3, 3), b_data).unwrap();
        let a = Tensor::from_ndarray(&a_nd);
        let b = Tensor::from_ndarray(&b_nd);
        let c = a.matmul(&b).unwrap().realize_with(&config).unwrap();

        let expected = a_nd.dot(&b_nd);

        assert_matmul_close(&c.to_vec::<f32>().unwrap(), &expected, 1e-4);
    }

    fn test_matmul_validated_2x3_3x4(config) {
        // [2, 3] @ [3, 4] -> [2, 4]
        let a_data: Vec<f32> = (1..=6).map(|x| x as f32).collect();
        let b_data: Vec<f32> = (1..=12).map(|x| x as f32).collect();

        let a_nd = Array2::from_shape_vec((2, 3), a_data).unwrap();
        let b_nd = Array2::from_shape_vec((3, 4), b_data).unwrap();
        let a = Tensor::from_ndarray(&a_nd);
        let b = Tensor::from_ndarray(&b_nd);
        let c = a.matmul(&b).unwrap().realize_with(&config).unwrap();

        let expected = a_nd.dot(&b_nd);

        assert_matmul_close(&c.to_vec::<f32>().unwrap(), &expected, 1e-4);
    }

    fn test_matmul_validated_tall_wide(config) {
        // [4, 2] @ [2, 5] -> [4, 5]
        let a_data: Vec<f32> = (1..=8).map(|x| x as f32 * 0.5).collect();
        let b_data: Vec<f32> = (1..=10).map(|x| x as f32 * 0.3).collect();

        let a_nd = Array2::from_shape_vec((4, 2), a_data).unwrap();
        let b_nd = Array2::from_shape_vec((2, 5), b_data).unwrap();
        let a = Tensor::from_ndarray(&a_nd);
        let b = Tensor::from_ndarray(&b_nd);
        let c = a.matmul(&b).unwrap().realize_with(&config).unwrap();

        let expected = a_nd.dot(&b_nd);

        assert_matmul_close(&c.to_vec::<f32>().unwrap(), &expected, 1e-5);
    }

    fn test_matmul_validated_16x16(config) {
        // Larger matrix to test vectorization paths
        const SIZE: usize = 16;
        let a_data: Vec<f32> = (0..SIZE * SIZE).map(|x| (x as f32) * 0.1).collect();
        let b_data: Vec<f32> = (0..SIZE * SIZE).map(|x| (x as f32) * 0.05 + 1.0).collect();

        let a_nd = Array2::from_shape_vec((SIZE, SIZE), a_data).unwrap();
        let b_nd = Array2::from_shape_vec((SIZE, SIZE), b_data).unwrap();
        let a = Tensor::from_ndarray(&a_nd);
        let b = Tensor::from_ndarray(&b_nd);
        let c = a.matmul(&b).unwrap().realize_with(&config).unwrap();

        let expected = a_nd.dot(&b_nd);

        assert_matmul_close(&c.to_vec::<f32>().unwrap(), &expected, 1e-3);
    }

    fn test_matmul_validated_32x32(config) {
        // Test with 32x32 to exercise more optimization paths
        const SIZE: usize = 32;
        let a_data: Vec<f32> = (0..SIZE * SIZE).map(|x| ((x % 17) as f32) * 0.1 - 0.8).collect();
        let b_data: Vec<f32> = (0..SIZE * SIZE).map(|x| ((x % 13) as f32) * 0.15 - 0.5).collect();

        let a_nd = Array2::from_shape_vec((SIZE, SIZE), a_data).unwrap();
        let b_nd = Array2::from_shape_vec((SIZE, SIZE), b_data).unwrap();
        let a = Tensor::from_ndarray(&a_nd);
        let b = Tensor::from_ndarray(&b_nd);
        let c = a.matmul(&b).unwrap().realize_with(&config).unwrap();

        let expected = a_nd.dot(&b_nd);

        assert_matmul_close(&c.to_vec::<f32>().unwrap(), &expected, 1e-2);
    }

    fn test_dot_product_validated(config) {
        // 1D @ 1D dot product
        let a_data = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b_data = [2.0f32, 3.0, 4.0, 5.0, 6.0];

        let a = Tensor::from_slice(a_data);
        let b = Tensor::from_slice(b_data);
        let c = a.dot(&b).unwrap().realize_with(&config).unwrap();

        // Expected: 1*2 + 2*3 + 3*4 + 4*5 + 5*6 = 2 + 6 + 12 + 20 + 30 = 70
        let expected: f32 = a_data.iter().zip(b_data.iter()).map(|(a, b)| a * b).sum();

        assert_eq!(c.shape().unwrap().len(), 0, "Dot product should be scalar");
        let result = c.to_vec::<f32>().unwrap();
        assert!((result[0] - expected).abs() < 1e-5, "Expected {}, got {}", expected, result[0]);
    }

    fn test_vector_matrix_validated(config) {
        // [4] @ [4, 3] -> [3]
        let v_data = [1.0f32, 2.0, 3.0, 4.0];
        let m_data: Vec<f32> = (1..=12).map(|x| x as f32).collect();

        let v = Tensor::from_slice(v_data);
        let m_nd = Array2::from_shape_vec((4, 3), m_data).unwrap();
        let m = Tensor::from_ndarray(&m_nd);
        let c = v.dot(&m).unwrap().realize_with(&config).unwrap();

        // ndarray: need to treat vector as [1, 4] @ [4, 3] -> [1, 3], then squeeze
        let v_nd = ndarray::Array1::from_vec(v_data.to_vec());
        let expected = v_nd.dot(&m_nd);

        assert_eq!(c.shape().unwrap()[0].as_const().unwrap(), 3);
        let morok_result = c.to_vec::<f32>().unwrap();
        for (i, (a, e)) in morok_result.iter().zip(expected.iter()).enumerate() {
            assert!((a - e).abs() < 1e-5, "Mismatch at index {}: {} != {}", i, a, e);
        }
    }

    fn test_matrix_vector_validated(config) {
        // [3, 4] @ [4] -> [3]
        let m_data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let v_data = [1.0f32, 2.0, 3.0, 4.0];

        let m_nd = Array2::from_shape_vec((3, 4), m_data).unwrap();
        let m = Tensor::from_ndarray(&m_nd);
        let v = Tensor::from_slice(v_data);
        let c = m.dot(&v).unwrap().realize_with(&config).unwrap();

        let v_nd = ndarray::Array1::from_vec(v_data.to_vec());
        let expected = m_nd.dot(&v_nd);

        assert_eq!(c.shape().unwrap()[0].as_const().unwrap(), 3);
        let morok_result = c.to_vec::<f32>().unwrap();
        for (i, (a, e)) in morok_result.iter().zip(expected.iter()).enumerate() {
            assert!((a - e).abs() < 1e-5, "Mismatch at index {}: {} != {}", i, a, e);
        }
    }

    fn test_matmul_identity_validated(config) {
        // A @ I = A
        let a_data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let identity_data = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];

        let a_nd = Array2::from_shape_vec((4, 4), a_data.clone()).unwrap();
        let i_nd = Array2::from_shape_vec((4, 4), identity_data).unwrap();
        let a = Tensor::from_ndarray(&a_nd);
        let i = Tensor::from_ndarray(&i_nd);
        let c = a.matmul(&i).unwrap().realize_with(&config).unwrap();
        let morok_result = c.to_vec::<f32>().unwrap();

        // Result should equal original A
        for (i, (actual, expected)) in morok_result.iter().zip(a_data.iter()).enumerate() {
            assert!((actual - expected).abs() < 1e-5, "Mismatch at index {}: {} != {}", i, actual, expected);
        }
    }

    fn test_matmul_negative_values_validated(config) {
        // Test with negative values to ensure sign handling
        let a_nd = Array2::from_shape_vec((2, 3), vec![-1.0f32, 2.0, -3.0, 4.0, -5.0, 6.0]).unwrap();
        let b_nd = Array2::from_shape_vec((3, 2), vec![1.0f32, -2.0, 3.0, -4.0, 5.0, -6.0]).unwrap();

        let a = Tensor::from_ndarray(&a_nd);
        let b = Tensor::from_ndarray(&b_nd).try_transpose(0, 1).unwrap();
        let b = b.try_transpose(0, 1).unwrap(); // Back to [3, 2] but contiguous
        let c = a.matmul(&b).unwrap().realize_with(&config).unwrap();

        let expected = a_nd.dot(&b_nd);

        assert_matmul_close(&c.to_vec::<f32>().unwrap(), &expected, 1e-5);
    }
}

// ========== Basic 2D x 2D Tests ==========

#[test]
fn test_matmul_2d_basic() {
    let a = Tensor::from_ndarray(&array![[1.0f32, 2.0], [3.0, 4.0]]);
    let b = Tensor::from_ndarray(&array![[5.0f32, 6.0], [7.0, 8.0]]);
    let c = a.dot(&b).unwrap();

    let c_shape = c.shape().unwrap();
    assert_eq!(c_shape.len(), 2);
    assert_eq!(c_shape[0].as_const().unwrap(), 2);
    assert_eq!(c_shape[1].as_const().unwrap(), 2);
}

#[test]
fn test_matmul_2d_non_square() {
    // [2, 3] @ [3, 4] → [2, 4]
    let a = Tensor::from_ndarray(&Array2::<f32>::ones((2, 3)));
    let b = Tensor::from_ndarray(&Array2::<f32>::ones((3, 4)));
    let c = a.dot(&b).unwrap();

    let c_shape = c.shape().unwrap();
    assert_eq!(c_shape.len(), 2);
    assert_eq!(c_shape[0].as_const().unwrap(), 2);
    assert_eq!(c_shape[1].as_const().unwrap(), 4);
}

#[test]
fn test_matmul_alias() {
    let a = Tensor::from_ndarray(&array![[1.0f32, 2.0], [3.0, 4.0]]);
    let b = Tensor::from_ndarray(&array![[5.0f32, 6.0], [7.0, 8.0]]);

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
    let b = Tensor::from_ndarray(&Array2::<f32>::ones((3, 4)));
    let c = a.dot(&b).unwrap();

    let c_shape = c.shape().unwrap();
    assert_eq!(c_shape.len(), 1);
    assert_eq!(c_shape[0].as_const().unwrap(), 4);
}

#[test]
fn test_matrix_vector() {
    // [2, 3] @ [3] → [2]
    let a = Tensor::from_ndarray(&Array2::<f32>::ones((2, 3)));
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
    let a = Tensor::from_ndarray(&ndarray::Array3::<f32>::ones((2, 3, 4)));
    let b = Tensor::from_ndarray(&ndarray::Array3::<f32>::ones((2, 4, 5)));
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
    let scalar = Tensor::from_ndarray(&ndarray::Array0::<f32>::from_elem((), 1.0));
    let vector = Tensor::from_slice([1.0f32, 2.0, 3.0]);

    // 0D tensors not supported
    assert!(scalar.dot(&vector).is_err());
    assert!(vector.dot(&scalar).is_err());
}

#[test]
fn test_matmul_error_shape_mismatch() {
    // [2, 3] @ [4, 5] - inner dimensions don't match
    let a = Tensor::from_ndarray(&Array2::<f32>::ones((2, 3)));
    let b = Tensor::from_ndarray(&Array2::<f32>::ones((4, 5)));

    let result = a.dot(&b);
    assert!(result.is_err());
}

#[test]
fn test_matmul_identity() {
    let a = Tensor::from_ndarray(&array![[1.0f32, 2.0], [3.0, 4.0]]);
    let identity = Tensor::from_ndarray(&array![[1.0f32, 0.0], [0.0, 1.0]]);

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
    let a = Tensor::from_ndarray(&array![[1i32, 2], [3, 4]]);
    let b = Tensor::from_ndarray(&array![[5.0f32, 6.0], [7.0, 8.0]]);

    let c = a.dot(&b).unwrap();
    // Result should be promoted to float32
    assert_eq!(c.uop().dtype(), DType::Float32);
}

#[test]
fn test_matmul_explicit_dtype() {
    let a = Tensor::from_ndarray(&array![[1.0f32, 2.0], [3.0, 4.0]]);
    let b = Tensor::from_ndarray(&array![[5.0f32, 6.0], [7.0, 8.0]]);

    // Use float64 accumulation
    let c = a.matmul_with().other(&b).dtype(DType::Float64).call().unwrap();
    assert_eq!(c.uop().dtype(), DType::Float64);
}

#[test]
#[ignore] // Run with: cargo test -p morok-tensor test_print_matmul_ir -- --ignored --nocapture
fn test_print_matmul_ir() {
    // Create 4x4 matmul to see generated IR
    let a = Tensor::from_ndarray(&Array2::from_shape_vec((4, 4), (0..16).map(|i| i as f32).collect()).unwrap());
    let b = Tensor::from_ndarray(&Array2::from_shape_vec((4, 4), (0..16).map(|i| i as f32).collect()).unwrap());
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
fn test_print_matmul_512x512_ir() {
    const SIZE: usize = 512;
    let a = Tensor::from_ndarray(
        &Array2::from_shape_vec((SIZE, SIZE), (0..SIZE * SIZE).map(|i| (i as f32) * 0.01).collect()).unwrap(),
    );
    let b = Tensor::from_ndarray(
        &Array2::from_shape_vec((SIZE, SIZE), (0..SIZE * SIZE).map(|i| (i as f32) * 0.01).collect()).unwrap(),
    );
    let c = a.matmul(&b).unwrap();

    // Use Heuristic strategy (Beam has a pre-existing bug with horizontal reduction)
    let config = prep_config(OptimizerConfig::builder().strategy(OptStrategy::Heuristic).build());
    let plan = c.prepare_with(&config).expect("prepare should succeed");

    println!("\n=== Generated Kernels (64x64 with output upcast) ===\n");
    for kernel in plan.kernels() {
        println!("--- {} ({}) ---", kernel.entry_point, kernel.device);
        println!("{}", kernel.code);
        println!();
    }
}

#[test]
fn test_beam_search_matmul() {
    // Test beam search optimization for matmul - reproduces float vector index bug
    let size = 512; // Original size that triggered the bug
    let a = Tensor::from_ndarray(
        &Array2::from_shape_vec((size, size), (0..size * size).map(|i| (i as f32) * 0.01).collect()).unwrap(),
    );
    let b = Tensor::from_ndarray(
        &Array2::from_shape_vec((size, size), (0..size * size).map(|i| (i as f32) * 0.01).collect()).unwrap(),
    );
    let c = a.matmul(&b).unwrap();

    // Use width=2 for reasonable test time. Disable disk cache to avoid stale results
    // from previous runs affecting correctness (beam cache is keyed by AST hash, but
    // the post-optimization pipeline may have changed).
    let beam_config = prep_config(
        OptimizerConfig::builder()
            .strategy(OptStrategy::Beam { width: 2 })
            .beam(BeamConfig::builder().disable_cache(true).build())
            .build(),
    );

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
    let input = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0]]);
    let weight = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let bias = Tensor::from_slice([0.1f32, 0.2]);

    let result = input.linear().weight(&weight).bias(&bias).call().unwrap();

    let result_shape = result.shape().unwrap();
    assert_eq!(result_shape.len(), 2);
    assert_eq!(result_shape[0].as_const().unwrap(), 1);
    assert_eq!(result_shape[1].as_const().unwrap(), 2);
}

#[test]
fn test_linear_no_bias() {
    let input = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0]]);
    let weight = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);

    let result = input.linear().weight(&weight).call().unwrap();

    let result_shape = result.shape().unwrap();
    assert_eq!(result_shape.len(), 2);
    assert_eq!(result_shape[0].as_const().unwrap(), 1);
    assert_eq!(result_shape[1].as_const().unwrap(), 2);
}

#[test]
fn test_linear_batched() {
    // input: [4, 3], weight: [2, 3] → output: [4, 2]
    let input = Tensor::from_ndarray(&Array2::<f32>::ones((4, 3)));
    let weight = Tensor::from_ndarray(&Array2::<f32>::ones((2, 3)));

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
fn test_vectorize_normalize_minimal() {
    // Test 64x64 matmul with vectorization enabled
    let a = Tensor::from_ndarray(&Array2::<f32>::ones((64, 64)));
    let b = Tensor::from_ndarray(&Array2::<f32>::ones((64, 64)));
    let c = a.matmul(&b).unwrap();

    // Explicit config to avoid test pollution from shared global state
    let config = prep_config(OptimizerConfig::builder().strategy(OptStrategy::Heuristic).build());
    let result = c.realize_with(&config);
    assert!(result.is_ok(), "realize failed: {:?}", result.err());
}

// ========== 512x512 Vectorized Test (for UPCAST debugging) ==========

#[test]
fn test_matmul_512x512_vectorized() {
    // Create 512x512 matrices filled with 1.0
    const SIZE: usize = 512;
    let a = Tensor::from_ndarray(&Array2::<f32>::ones((SIZE, SIZE)));
    let b = Tensor::from_ndarray(&Array2::<f32>::ones((SIZE, SIZE)));
    let c = a.matmul(&b).unwrap();

    // Use from_env() to respect MOROK_OUTPUT_UPCAST and other env vars
    // Note: Beam search has a pre-existing bug with horizontal reduction, using Heuristic
    let config = env_config();
    let c = c.realize_with(&config).unwrap();
    let result = c.to_vec::<f32>().unwrap();

    // Each element should be 512 (sum of 512 ones)
    assert_eq!(result.len(), SIZE * SIZE);
    assert!((result[0] - SIZE as f32).abs() < 0.01, "Expected {}, got {}", SIZE, result[0]);
}

#[test]
fn test_matmul_64x64_vectorized() {
    // Create 64x64 matrices filled with 1.0
    const SIZE: usize = 64;
    let a = Tensor::from_ndarray(&Array2::<f32>::ones((SIZE, SIZE)));
    let b = Tensor::from_ndarray(&Array2::<f32>::ones((SIZE, SIZE)));
    let c = a.matmul(&b).unwrap();

    let config = env_config();
    let c = c.realize_with(&config).unwrap();
    let result = c.to_vec::<f32>().unwrap();

    // Each element should be 64 (sum of 64 ones)
    assert_eq!(result.len(), SIZE * SIZE);
    assert!((result[0] - SIZE as f32).abs() < 0.01, "Expected {}, got {}", SIZE, result[0]);
}

#[test]
#[ignore] // Run with: cargo test -p morok-tensor test_print_matmul_64x64_ir -- --ignored --nocapture
fn test_print_matmul_64x64_ir() {
    const SIZE: usize = 64;
    let a = Tensor::from_ndarray(&Array2::<f32>::ones((SIZE, SIZE)));
    let b = Tensor::from_ndarray(&Array2::<f32>::ones((SIZE, SIZE)));
    let c = a.matmul(&b).unwrap();

    let config = env_config();
    let plan = c.prepare_with(&config).expect("prepare should succeed");

    println!("\n=== Generated Kernels (64x64 matmul) ===\n");
    for kernel in plan.prepared_kernels() {
        println!("--- {} ({}) ---", kernel.kernel.entry_point, kernel.kernel.device);
        println!("{}", kernel.ast.tree());
        println!("{}", kernel.kernel.code);
        println!();
    }
}

// ========== Validated Matmul Tests (64x64 with env_config) ==========

#[test]
fn test_matmul_validated_64x64() {
    // 64x64 test with varied data
    const SIZE: usize = 64;
    let a_data: Vec<f32> = (0..SIZE * SIZE).map(|x| ((x as f32) * 0.01).sin()).collect();
    let b_data: Vec<f32> = (0..SIZE * SIZE).map(|x| ((x as f32) * 0.02).cos()).collect();

    let a_nd = Array2::from_shape_vec((SIZE, SIZE), a_data).unwrap();
    let b_nd = Array2::from_shape_vec((SIZE, SIZE), b_data).unwrap();
    let a = Tensor::from_ndarray(&a_nd);
    let b = Tensor::from_ndarray(&b_nd);

    let config = env_config();
    let c = a.matmul(&b).unwrap().realize_with(&config).unwrap();

    let expected = a_nd.dot(&b_nd);

    // Larger tolerance for accumulated floating point error
    assert_matmul_close(&c.to_vec::<f32>().unwrap(), &expected, 1e-1);
}

// ========== Large Dimension Validated Tests ==========

use test_case::test_case;

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
