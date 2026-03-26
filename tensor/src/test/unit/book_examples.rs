use crate::nn::Layer;
use crate::test::helpers::{assert_close_f32, test_setup};
use crate::*;
use ndarray::array;

// =========================================================================
// Example 1: Hello Tensor
// =========================================================================

crate::codegen_tests! {
    fn test_example_1_hello_tensor(config) {
        test_setup();

        let a = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);
        let b = Tensor::from_slice([10.0f32, 20.0, 30.0, 40.0]);

        let sum = &a + &b;
        let scaled = sum * Tensor::from_slice([0.1f32]);

        let mut result = scaled;
        result.realize_with(&config).unwrap();
        let data = result.as_vec::<f32>().unwrap();

        // [1+10, 2+20, 3+30, 4+40] * 0.1 = [1.1, 2.2, 3.3, 4.4]
        assert_close_f32(&data, &[1.1, 2.2, 3.3, 4.4], 1e-6);
    }
}

// =========================================================================
// Example 2: Shape Gymnastics
// =========================================================================

crate::codegen_tests! {
    fn test_example_2_shape_reshape(_config) {
        test_setup();

        let data = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(data.shape().unwrap().len(), 1);
        assert_eq!(data.shape().unwrap()[0].as_const(), Some(6));

        let matrix = data.try_reshape([2, 3]).unwrap();
        assert_eq!(matrix.shape().unwrap().len(), 2);
    }

    fn test_example_2_shape_transpose(config) {
        test_setup();

        let data = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let transposed = data.try_transpose(0, 1).unwrap();

        assert_eq!(transposed.shape().unwrap()[0].as_const(), Some(3));
        assert_eq!(transposed.shape().unwrap()[1].as_const(), Some(2));

        // Verify transposition is correct: [[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]]
        let mut result = transposed;
        result.realize_with(&config).unwrap();
        let vals = result.as_vec::<f32>().unwrap();
        assert_close_f32(&vals, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 1e-6);
    }

    fn test_example_2_broadcast(config) {
        test_setup();

        // [3, 2] + [1, 2] → [3, 2]
        let transposed = Tensor::from_ndarray(&array![[1.0f32, 4.0], [2.0, 5.0], [3.0, 6.0]]);
        let bias = Tensor::from_ndarray(&array![[100.0f32, 200.0]]);
        let biased = &transposed + &bias;

        let mut result = biased;
        result.realize_with(&config).unwrap();
        let vals = result.as_vec::<f32>().unwrap();

        // [[1+100, 4+200], [2+100, 5+200], [3+100, 6+200]]
        assert_close_f32(&vals, &[101.0, 204.0, 102.0, 205.0, 103.0, 206.0], 1e-6);
    }
}

// =========================================================================
// Example 3: Matrix Multiply
// =========================================================================

crate::codegen_tests! {
    fn test_example_3_matmul(config) {
        test_setup();

        // Input: [4, 3], Weights: [3, 2] → Output: [4, 2]
        let input = Tensor::from_ndarray(&array![
            [1.0f32, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ]);

        let weights = Tensor::from_ndarray(&array![
            [0.1f32, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
        ]);

        let output = input.dot(&weights).unwrap();

        // Verify shape before realize
        let shape = output.shape().unwrap();
        assert_eq!(shape[0].as_const(), Some(4));
        assert_eq!(shape[1].as_const(), Some(2));

        let mut result = output;
        result.realize_with(&config).unwrap();
        let vals = result.as_vec::<f32>().unwrap();

        // Row 0: [1,2,3] @ [0.1,0.3,0.5] = 0.1+0.6+1.5=2.2, [1,2,3] @ [0.2,0.4,0.6] = 0.2+0.8+1.8=2.8
        // Row 1: [4,5,6] @ [0.1,0.3,0.5] = 0.4+1.5+3.0=4.9, [4,5,6] @ [0.2,0.4,0.6] = 0.8+2.0+3.6=6.4
        // Row 2: [7,8,9] @ [0.1,0.3,0.5] = 0.7+2.4+4.5=7.6, [7,8,9] @ [0.2,0.4,0.6] = 1.4+3.2+5.4=10.0
        // Row 3: [10,11,12] @ [0.1,0.3,0.5] = 1.0+3.3+6.0=10.3, [10,11,12] @ [0.2,0.4,0.6] = 2.0+4.4+7.2=13.6
        assert_close_f32(
            &vals,
            &[2.2, 2.8, 4.9, 6.4, 7.6, 10.0, 10.3, 13.6],
            1e-4,
        );
    }
}

// =========================================================================
// Example 4: Building a Linear Layer
// =========================================================================

crate::codegen_tests! {
    fn test_example_4_linear_layer(config) {
        test_setup();

        // Weight: [2, 4], Bias: [2]
        let weight = Tensor::from_ndarray(&array![
            [0.0f32, 0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6, 0.7],
        ]);
        let bias = Tensor::from_slice([0.0f32, 0.0]);

        // Input: [1, 4]
        let input = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0, 4.0]]);

        // y = input @ weight.T + bias
        let weight_t = weight.try_transpose(0, 1).unwrap();
        let out = input.dot(&weight_t).unwrap();
        let result_tensor = &out + &bias;

        // Verify shape
        let shape = result_tensor.shape().unwrap();
        assert_eq!(shape[0].as_const(), Some(1));
        assert_eq!(shape[1].as_const(), Some(2));

        let mut result = result_tensor;
        result.realize_with(&config).unwrap();
        let vals = result.as_vec::<f32>().unwrap();

        // [1,2,3,4] @ [0.0,0.4; 0.1,0.5; 0.2,0.6; 0.3,0.7] = [0*1+0.1*2+0.2*3+0.3*4, 0.4*1+0.5*2+0.6*3+0.7*4]
        // = [0+0.2+0.6+1.2, 0.4+1.0+1.8+2.8] = [2.0, 6.0]
        assert_close_f32(&vals, &[2.0, 6.0], 1e-4);
    }
}

// =========================================================================
// Example 5: MNIST Classifier
// =========================================================================

crate::codegen_tests! {
    fn test_example_5_mnist_forward(config) {
        test_setup();

        // Use small dimensions for fast test: 4 → 3 → 2
        // fc1: [3, 4] weights + [3] bias
        let w1_data: Vec<f32> = (0..12).map(|i| (i as f32) * 0.1 - 0.5).collect();
        let w1 = Tensor::from_ndarray(&ndarray::Array2::from_shape_vec((3, 4), w1_data).unwrap());
        let b1 = Tensor::from_slice([0.0f32; 3]);

        // fc2: [2, 3] weights + [2] bias
        let w2_data: Vec<f32> = (0..6).map(|i| (i as f32) * 0.1 - 0.3).collect();
        let w2 = Tensor::from_ndarray(&ndarray::Array2::from_shape_vec((2, 3), w2_data).unwrap());
        let b2 = Tensor::from_slice([0.0f32; 2]);

        // Input: [1, 4]
        let input = Tensor::from_ndarray(&array![[0.5f32, -0.3, 0.8, 0.1]]);

        // Layer 1: linear + relu
        let h = input.dot(&w1.try_transpose(0, 1).unwrap()).unwrap();
        let h = &h + &b1;
        let h = h.relu().unwrap();

        // Layer 2: linear
        let logits = h.dot(&w2.try_transpose(0, 1).unwrap()).unwrap();
        let logits = &logits + &b2;

        // Verify shape: [1, 2]
        let shape = logits.shape().unwrap();
        assert_eq!(shape[0].as_const(), Some(1));
        assert_eq!(shape[1].as_const(), Some(2));

        // Softmax
        let probs = logits.softmax(-1).unwrap();

        let mut result = probs;
        result.realize_with(&config).unwrap();
        let vals = result.as_vec::<f32>().unwrap();

        // Probabilities should sum to 1.0
        let sum: f32 = vals.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "Probabilities should sum to 1.0, got {}", sum);

        // All probabilities should be positive
        for (i, &p) in vals.iter().enumerate() {
            assert!(p > 0.0, "Probability at index {} should be positive, got {}", i, p);
            assert!(p <= 1.0, "Probability at index {} should be <= 1.0, got {}", i, p);
        }
    }

    fn test_example_5_argmax(config) {
        test_setup();

        // Simple test: [1, 4] with clear argmax
        let input = Tensor::from_ndarray(&array![[0.1f32, 0.9, 0.3, 0.5]]);
        let mut result = input.argmax(Some(-1)).unwrap();

        result.realize_with(&config).unwrap();
        let vals = result.as_vec::<i32>().unwrap();

        assert_eq!(vals, &[1], "Argmax should be 1 (0.9 is largest)");
    }
}

// =========================================================================
// Example 6: Under the Hood
// =========================================================================

// =========================================================================
// Example 4b: Linear layer via library API
// =========================================================================

crate::codegen_tests! {
    fn test_example_4_linear_layer_sequential(config) {
        test_setup();

        // Use library Linear with known weights: [2, 4]
        let weight = Tensor::from_ndarray(&array![
            [0.0f32, 0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6, 0.7],
        ]);
        let bias = Tensor::from_slice([0.0f32, 0.0]);
        let layer = nn::Linear::new(weight, bias);

        let input = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);

        let output = layer.forward(&input).unwrap();
        let shape = output.shape().unwrap();
        assert_eq!(shape.len(), 1);
        assert_eq!(shape[0].as_const(), Some(2));

        let mut result = output;
        result.realize_with(&config).unwrap();
        let vals = result.as_vec::<f32>().unwrap();

        // Same computation as test_example_4_linear_layer
        assert_close_f32(&vals, &[2.0, 6.0], 1e-4);
    }
}

// =========================================================================
// Example 5b: MNIST via sequential
// =========================================================================

crate::codegen_tests! {
    fn test_example_5_mnist_sequential(config) {
        test_setup();

        // Small 4 -> 3 -> 2 network via sequential
        let w1_data: Vec<f32> = (0..12).map(|i| (i as f32) * 0.1 - 0.5).collect();
        let w1 = Tensor::from_ndarray(&ndarray::Array2::from_shape_vec((3, 4), w1_data).unwrap());
        let b1 = Tensor::from_slice([0.0f32; 3]);
        let fc1 = nn::Linear::new(w1, b1);

        let w2_data: Vec<f32> = (0..6).map(|i| (i as f32) * 0.1 - 0.3).collect();
        let w2 = Tensor::from_ndarray(&ndarray::Array2::from_shape_vec((2, 3), w2_data).unwrap());
        let b2 = Tensor::from_slice([0.0f32; 2]);
        let fc2 = nn::Linear::new(w2, b2);

        let input = Tensor::from_slice([0.5f32, -0.3, 0.8, 0.1]);

        let logits = input.sequential(&[&fc1, &nn::Relu, &fc2]).unwrap();

        // Verify shape: [2]
        let shape = logits.shape().unwrap();
        assert_eq!(shape.len(), 1);
        assert_eq!(shape[0].as_const(), Some(2));

        // Softmax
        let probs = logits.softmax(-1).unwrap();

        let mut result = probs;
        result.realize_with(&config).unwrap();
        let vals = result.as_vec::<f32>().unwrap();

        let sum: f32 = vals.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "Probabilities should sum to 1.0, got {}", sum);

        for (i, &p) in vals.iter().enumerate() {
            assert!(p > 0.0, "Probability at index {} should be positive, got {}", i, p);
            assert!(p <= 1.0, "Probability at index {} should be <= 1.0, got {}", i, p);
        }
    }
}

#[test]
fn test_example_6_ir_graph() {
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let b = Tensor::from_slice([4.0f32, 5.0, 6.0]);
    let c = &a + &b;

    // Verify we can print the IR tree without crashing
    let tree = c.uop().tree();
    assert!(!tree.is_empty(), "IR tree should not be empty");

    // The tree should contain Add operation
    assert!(tree.contains("Add"), "IR tree should contain Add: {}", tree);
}
