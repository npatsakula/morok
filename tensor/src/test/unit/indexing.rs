use crate::*;
use morok_dtype::DType;
use ndarray::array;

fn get_shape(tensor: &Tensor) -> Vec<usize> {
    tensor.uop().shape().unwrap().unwrap().iter().map(|s| s.as_const().unwrap()).collect()
}

// =========================================================================
// One-Hot Tests (codegen required)
// =========================================================================

crate::codegen_tests! {
    fn test_one_hot_along_dim_basic(config) {
        // [0, 1, 2] with 3 classes → 3x3 identity-like mask
        let idx = Tensor::from_slice([0i32, 1, 2]).try_unsqueeze(-1).unwrap();
        let result = idx.one_hot_along_dim(3, -1).unwrap();
        let shape = get_shape(&result);
        assert_eq!(shape, vec![3, 3]);
        let realized = result.contiguous().realize_with(&config).unwrap();
        let view = realized.array_view::<bool>().unwrap();
        // Row 0: [true, false, false]
        assert!(view[[0, 0]]);
        assert!(!view[[0, 1]]);
        assert!(!view[[0, 2]]);
        // Row 1: [false, true, false]
        assert!(!view[[1, 0]]);
        assert!(view[[1, 1]]);
        assert!(!view[[1, 2]]);
        // Row 2: [false, false, true]
        assert!(!view[[2, 0]]);
        assert!(!view[[2, 1]]);
        assert!(view[[2, 2]]);
    }

    // =========================================================================
    // Scatter Tests
    // =========================================================================

    fn test_scatter_1d_basic(config) {
        // Create [0, 0, 0, 0, 0], scatter [10, 20, 30] at indices [1, 3, 0]
        let x = Tensor::from_slice([0.0f32, 0.0, 0.0, 0.0, 0.0]);
        let idx = Tensor::from_slice([1i32, 3, 0]);
        let src = Tensor::from_slice([10.0f32, 20.0, 30.0]);
        let result = x.scatter(0, &idx, &src).unwrap().contiguous().realize_with(&config).unwrap();
        let view = result.array_view::<f32>().unwrap();
        assert_eq!(view[[0]], 30.0); // index 0 got 30
        assert_eq!(view[[1]], 10.0); // index 1 got 10
        assert_eq!(view[[3]], 20.0); // index 3 got 20
    }

    fn test_scatter_reduce_sum(config) {
        let x = Tensor::from_slice([0.0f32, 0.0, 0.0]);
        let idx = Tensor::from_slice([0i32, 0, 1]);
        let src = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let result = x
            .scatter_reduce(0, &idx, &src, crate::indexing::ScatterReduction::Sum, true)
            .unwrap()
            .contiguous()
            .realize_with(&config)
            .unwrap();
        let view = result.array_view::<f32>().unwrap();
        // index 0: 0 + 1 + 2 = 3, index 1: 0 + 3 = 3, index 2: 0
        assert_eq!(view[[0]], 3.0);
        assert_eq!(view[[1]], 3.0);
        assert_eq!(view[[2]], 0.0);
    }

    fn test_scatter_2d(config) {
        let x = Tensor::from_ndarray(&ndarray::Array2::<f32>::zeros((3, 2)));
        let idx = Tensor::from_ndarray(&array![[0i32, 1]]);
        let src = Tensor::from_ndarray(&array![[10.0f32, 20.0]]);
        let result = x.scatter(0, &idx, &src).unwrap().contiguous().realize_with(&config).unwrap();
        assert_eq!(get_shape(&result), vec![3, 2]);
        let view = result.array_view::<f32>().unwrap();
        assert_eq!(view[[0, 0]], 10.0);
        assert_eq!(view[[1, 1]], 20.0);
    }

    // =========================================================================
    // TopK Tests
    // =========================================================================

    fn test_topk_basic(config) {
        // 4 elements = n_stages=2 (power of 2) — larger sizes are very slow in debug builds
        let t = Tensor::from_slice([1.0f32, 4.0, 2.0, 3.0]);
        let (values, indices) = t.topk(2, -1, true).unwrap();
        let values = values.contiguous().realize_with(&config).unwrap();
        let indices = indices.realize_with(&config).unwrap();
        assert_eq!(get_shape(&values), vec![2]);
        assert_eq!(get_shape(&indices), vec![2]);
        let view = values.array_view::<f32>().unwrap();
        assert_eq!(view[[0]], 4.0);
        assert_eq!(view[[1]], 3.0);
    }

    fn test_topk_smallest(config) {
        let t = Tensor::from_slice([1.0f32, 4.0, 2.0, 3.0]);
        let (values, _) = t.topk(2, -1, false).unwrap();
        let values = values.contiguous().realize_with(&config).unwrap();
        let view = values.array_view::<f32>().unwrap();
        assert_eq!(view[[0]], 1.0);
        assert_eq!(view[[1]], 2.0);
    }

    // =========================================================================
    // Masked Select Tests
    // =========================================================================

    fn test_masked_select_basic(config) {
        let t = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0]);
        let mask = Tensor::from_slice([true, false, true, false, true]);
        let result = t.masked_select(&mask).unwrap().realize_with(&config).unwrap();
        assert_eq!(get_shape(&result), vec![3]);
        assert_eq!(result.to_vec::<f32>().unwrap(), [1.0, 3.0, 5.0]);
    }

    // =========================================================================
    // NonZero Tests
    // =========================================================================

    fn test_nonzero_1d(config) {
        let t = Tensor::from_slice([1i32, 0, 2, 0, 3]);
        let result = t.nonzero().unwrap().contiguous().realize_with(&config).unwrap();
        assert_eq!(get_shape(&result), vec![3, 1]);
        let view = result.array_view::<i32>().unwrap();
        assert_eq!(view[[0, 0]], 0); // index of 1
        assert_eq!(view[[1, 0]], 2); // index of 2
        assert_eq!(view[[2, 0]], 4); // index of 3
    }

    fn test_nonzero_2d_debug_coords(config) {
        // Test the coordinate building for nonzero on [2, 2]
        // coord0: arange(2) → [0, 1], reshape [2, 1], expand [2, 2], flatten → [0, 0, 1, 1]
        let coord0 = Tensor::arange(0, Some(2), None)
            .unwrap()
            .try_reshape(&[2, 1])
            .unwrap()
            .try_expand(&[2, 2])
            .unwrap()
            .flatten()
            .unwrap();
        let c0 = coord0.realize_with(&config).unwrap().to_vec::<i32>().unwrap();
        eprintln!("coord0 len: {}, vals: {:?}", c0.len(), c0);

        // coord1: arange(2) → [0, 1], reshape [1, 2], expand [2, 2], flatten → [0, 1, 0, 1]
        let coord1 = Tensor::arange(0, Some(2), None)
            .unwrap()
            .try_reshape(&[1, 2])
            .unwrap()
            .try_expand(&[2, 2])
            .unwrap()
            .flatten()
            .unwrap();
        let c1 = coord1.realize_with(&config).unwrap().to_vec::<i32>().unwrap();
        eprintln!("coord1 len: {}, vals: {:?}", c1.len(), c1);

        assert_eq!(c0, [0, 0, 1, 1]);
        assert_eq!(c1, [0, 1, 0, 1]);
    }

    fn test_nonzero_2d_debug_stack(config) {
        // Test stack with lazy coordinate tensors
        let coord0 = Tensor::arange(0, Some(2), None)
            .unwrap()
            .try_reshape(&[2, 1])
            .unwrap()
            .try_expand(&[2, 2])
            .unwrap()
            .flatten()
            .unwrap(); // [0, 0, 1, 1]

        let coord1 = Tensor::arange(0, Some(2), None)
            .unwrap()
            .try_reshape(&[1, 2])
            .unwrap()
            .try_expand(&[2, 2])
            .unwrap()
            .flatten()
            .unwrap(); // [0, 1, 0, 1]

        let stacked = Tensor::stack(&[&coord0, &coord1], -1).unwrap();
        eprintln!("stacked uop tree:\n{}", stacked.uop().tree());
        let stacked_realized = stacked.realize_with(&config).unwrap();
        let stacked_vec = stacked_realized.to_vec::<i32>().unwrap();
        let stacked_shape = get_shape(&stacked_realized);
        eprintln!("stacked shape: {:?}", stacked_shape);
        eprintln!("stacked values: {:?}", stacked_vec);
        assert_eq!(stacked_shape, [4, 2]);
        // Expected: [[0, 0], [0, 1], [1, 0], [1, 1]]
        assert_eq!(stacked_vec, [0, 0, 0, 1, 1, 0, 1, 1]);
    }

    fn test_nonzero_2d(config) {
        // [[1, 0], [1, 1]] — nonzero at (0,0), (1,0), (1,1)
        let t = Tensor::from_ndarray(&array![[1i32, 0], [1, 1]]);
        let result = t.nonzero().unwrap().contiguous().realize_with(&config).unwrap();
        assert_eq!(get_shape(&result), vec![3, 2]);
        let view = result.array_view::<i32>().unwrap();
        eprintln!("nonzero shape: {:?}", view.shape());
        eprintln!("nonzero values: {:?}", view.as_slice().unwrap());
        assert_eq!(view[[0, 0]], 0);
        assert_eq!(view[[0, 1]], 0);
        assert_eq!(view[[1, 0]], 1);
        assert_eq!(view[[1, 1]], 0);
        assert_eq!(view[[2, 0]], 1);
        assert_eq!(view[[2, 1]], 1);
    }
}

// =========================================================================
// Gather Tests (shape/error/dtype only — no codegen)
// =========================================================================

#[test]
fn test_gather_1d_basic() {
    // Gather from 1D tensor
    let t = Tensor::from_slice([10.0f32, 20.0, 30.0, 40.0, 50.0]);
    let idx = Tensor::from_slice([2i64, 0, 4]); // Gather elements 2, 0, 4

    // Need to expand index to same rank as input (1D)
    let result = t.gather(0, &idx).unwrap();

    // Result shape should match index shape
    assert_eq!(get_shape(&result), vec![3]);
}

#[test]
fn test_gather_2d_dim0() {
    // Input shape [3, 4], index shape [2, 4]
    let t = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]);

    // Index must have same non-gather dim sizes
    let idx = Tensor::from_ndarray(&array![[0i64, 1, 2, 0], [1, 0, 1, 2]]);

    let result = t.gather(0, &idx).unwrap();
    assert_eq!(get_shape(&result), vec![2, 4]);
}

#[test]
fn test_gather_2d_dim1() {
    // Input shape [2, 5], index shape [2, 3]
    let t = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]]);

    let idx = Tensor::from_ndarray(&array![[0i64, 2, 4], [1, 3, 0]]);

    let result = t.gather(1, &idx).unwrap();
    assert_eq!(get_shape(&result), vec![2, 3]);
}

#[test]
fn test_gather_negative_axis() {
    let t = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);

    let idx = Tensor::from_ndarray(&array![[0i64], [2]]);

    // -1 = last axis
    let result = t.gather(-1, &idx).unwrap();
    assert_eq!(get_shape(&result), vec![2, 1]);
}

#[test]
fn test_gather_error_rank_mismatch() {
    let t = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);

    // Index has different rank (1D vs 2D)
    let idx = Tensor::from_slice([0i64, 1]);

    let result = t.gather(0, &idx);
    assert!(result.is_err());
}

#[test]
fn test_gather_error_dim_mismatch() {
    // Input [2, 3], gather along dim=1
    let t = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);

    // Index [3, 2] - non-gather dim 0 has size 3 > input size 2
    let idx = Tensor::from_ndarray(&array![[0i64, 1], [0, 1], [0, 1]]);

    let result = t.gather(1, &idx);
    assert!(result.is_err());
}

#[test]
fn test_gather_dtype_preserved() {
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0]);
    let idx = Tensor::from_slice([0i64, 2, 4]);

    let result = t.gather(0, &idx).unwrap();

    // Result should have same dtype as input
    assert_eq!(result.uop().dtype(), DType::Float32);
}

// =========================================================================
// Shrink Tests (shape only — no codegen)
// =========================================================================

#[test]
fn test_shrink_1d() {
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0]);

    let sliced = t.try_shrink(&[(1, 4)]).unwrap();
    assert_eq!(get_shape(&sliced), vec![3]);
}

#[test]
fn test_shrink_2d() {
    let t = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);

    let sliced = t.try_shrink(&[(0, 1), (1, 3)]).unwrap();
    assert_eq!(get_shape(&sliced), vec![1, 2]);
}

#[test]
fn test_shrink_negative_indices() {
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0]);

    // -3 to -1 should give elements [3, 4]
    let sliced = t.try_shrink(&[(-3, -1)]).unwrap();
    assert_eq!(get_shape(&sliced), vec![2]);
}

#[test]
fn test_shrink_full_dimension() {
    let t = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);

    // Keep full first dim, slice second
    let sliced = t.try_shrink(&[(0, 2), (1, 3)]).unwrap();
    assert_eq!(get_shape(&sliced), vec![2, 2]);
}

#[test]
fn test_shrink_empty_is_identity() {
    let t = Tensor::from_slice([1.0f32]);
    let sliced = t.try_shrink(&[]).unwrap();
    assert_eq!(get_shape(&sliced), vec![1]);
}
