use crate::*;
use morok_dtype::DType;

fn get_shape(tensor: &Tensor) -> Vec<usize> {
    tensor.uop().shape().unwrap().unwrap().iter().map(|s| s.as_const().unwrap()).collect()
}

// =========================================================================
// One-Hot Tests
// =========================================================================

#[test]
fn test_one_hot_along_dim_basic() {
    // [0, 1, 2] with 3 classes → 3x3 identity-like mask
    let idx = Tensor::from_slice([0i32, 1, 2]).try_unsqueeze(-1).unwrap();
    let result = idx.one_hot_along_dim(3, -1).unwrap();
    let shape = get_shape(&result);
    assert_eq!(shape, vec![3, 3]);
    let arr = result.to_ndarray::<bool>().unwrap();
    // Row 0: [true, false, false]
    assert!(arr[[0, 0]]);
    assert!(!arr[[0, 1]]);
    assert!(!arr[[0, 2]]);
    // Row 1: [false, true, false]
    assert!(!arr[[1, 0]]);
    assert!(arr[[1, 1]]);
    assert!(!arr[[1, 2]]);
    // Row 2: [false, false, true]
    assert!(!arr[[2, 0]]);
    assert!(!arr[[2, 1]]);
    assert!(arr[[2, 2]]);
}

// =========================================================================
// Gather Tests
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
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
        .try_reshape(&[3, 4])
        .unwrap();

    // Index must have same non-gather dim sizes
    let idx = Tensor::from_slice([0i64, 1, 2, 0, 1, 0, 1, 2]).try_reshape(&[2, 4]).unwrap();

    let result = t.gather(0, &idx).unwrap();
    assert_eq!(get_shape(&result), vec![2, 4]);
}

#[test]
fn test_gather_2d_dim1() {
    // Input shape [2, 5], index shape [2, 3]
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]).try_reshape(&[2, 5]).unwrap();

    let idx = Tensor::from_slice([0i64, 2, 4, 1, 3, 0]).try_reshape(&[2, 3]).unwrap();

    let result = t.gather(1, &idx).unwrap();
    assert_eq!(get_shape(&result), vec![2, 3]);
}

#[test]
fn test_gather_negative_axis() {
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).try_reshape(&[2, 3]).unwrap();

    let idx = Tensor::from_slice([0i64, 2]).try_reshape(&[2, 1]).unwrap();

    // -1 = last axis
    let result = t.gather(-1, &idx).unwrap();
    assert_eq!(get_shape(&result), vec![2, 1]);
}

#[test]
fn test_gather_error_rank_mismatch() {
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).try_reshape(&[2, 3]).unwrap();

    // Index has different rank (1D vs 2D)
    let idx = Tensor::from_slice([0i64, 1]);

    let result = t.gather(0, &idx);
    assert!(result.is_err());
}

#[test]
fn test_gather_error_dim_mismatch() {
    // Input [2, 3], gather along dim=1
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).try_reshape(&[2, 3]).unwrap();

    // Index [3, 2] - non-gather dim 0 has size 3 > input size 2
    let idx = Tensor::from_slice([0i64, 1, 0, 1, 0, 1]).try_reshape(&[3, 2]).unwrap();

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
// Shrink Tests
// =========================================================================

#[test]
fn test_shrink_1d() {
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0]);

    let sliced = t.try_shrink(&[(1, 4)]).unwrap();
    assert_eq!(get_shape(&sliced), vec![3]);
}

#[test]
fn test_shrink_2d() {
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).try_reshape(&[2, 3]).unwrap();

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
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).try_reshape(&[2, 3]).unwrap();

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

// =========================================================================
// Scatter Tests
// =========================================================================

#[test]
fn test_scatter_1d_basic() {
    // Create [0, 0, 0, 0, 0], scatter [10, 20, 30] at indices [1, 3, 0]
    let x = Tensor::from_slice([0.0f32, 0.0, 0.0, 0.0, 0.0]);
    let idx = Tensor::from_slice([1i32, 3, 0]);
    let src = Tensor::from_slice([10.0f32, 20.0, 30.0]);
    let result = x.scatter(0, &idx, &src).unwrap().realize().unwrap();
    let vals = result.to_ndarray::<f32>().unwrap();
    assert_eq!(vals[[0]], 30.0); // index 0 got 30
    assert_eq!(vals[[1]], 10.0); // index 1 got 10
    assert_eq!(vals[[3]], 20.0); // index 3 got 20
}

#[test]
fn test_scatter_reduce_sum() {
    let x = Tensor::from_slice([0.0f32, 0.0, 0.0]);
    let idx = Tensor::from_slice([0i32, 0, 1]);
    let src = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let result = x.scatter_reduce(0, &idx, &src, "sum", true).unwrap().realize().unwrap();
    let vals = result.to_ndarray::<f32>().unwrap();
    // index 0: 0 + 1 + 2 = 3, index 1: 0 + 3 = 3, index 2: 0
    assert_eq!(vals[[0]], 3.0);
    assert_eq!(vals[[1]], 3.0);
    assert_eq!(vals[[2]], 0.0);
}

#[test]
fn test_scatter_2d() {
    let x = Tensor::from_slice([0.0f32; 6]).try_reshape(&[3, 2]).unwrap();
    let idx = Tensor::from_slice([0i32, 1]).try_reshape(&[1, 2]).unwrap();
    let src = Tensor::from_slice([10.0f32, 20.0]).try_reshape(&[1, 2]).unwrap();
    let result = x.scatter(0, &idx, &src).unwrap().realize().unwrap();
    assert_eq!(get_shape(&result), vec![3, 2]);
    let vals = result.to_ndarray::<f32>().unwrap();
    assert_eq!(vals[[0, 0]], 10.0);
    assert_eq!(vals[[1, 1]], 20.0);
}

// =========================================================================
// TopK Tests
// =========================================================================

#[test]
fn test_topk_basic() {
    // 4 elements = n_stages=2 (power of 2) — larger sizes are very slow in debug builds
    let t = Tensor::from_slice([1.0f32, 4.0, 2.0, 3.0]);
    let (values, indices) = t.topk(2, -1, true).unwrap();
    let values = values.realize().unwrap();
    let indices = indices.realize().unwrap();
    assert_eq!(get_shape(&values), vec![2]);
    assert_eq!(get_shape(&indices), vec![2]);
    let vals = values.to_ndarray::<f32>().unwrap();
    assert_eq!(vals[[0]], 4.0);
    assert_eq!(vals[[1]], 3.0);
}

#[test]
fn test_topk_smallest() {
    let t = Tensor::from_slice([1.0f32, 4.0, 2.0, 3.0]);
    let (values, _) = t.topk(2, -1, false).unwrap();
    let values = values.realize().unwrap();
    let vals = values.to_ndarray::<f32>().unwrap();
    assert_eq!(vals[[0]], 1.0);
    assert_eq!(vals[[1]], 2.0);
}

// =========================================================================
// Masked Select Tests
// =========================================================================

#[test]
fn test_masked_select_basic() {
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0]);
    let mask = Tensor::from_slice([true, false, true, false, true]);
    let result = t.masked_select(&mask).unwrap().realize().unwrap();
    assert_eq!(get_shape(&result), vec![3]);
    let vals = result.to_ndarray::<f32>().unwrap();
    assert_eq!(vals.as_slice().unwrap(), &[1.0, 3.0, 5.0]);
}

// =========================================================================
// NonZero Tests
// =========================================================================

#[test]
fn test_nonzero_1d() {
    let t = Tensor::from_slice([1i32, 0, 2, 0, 3]);
    let result = t.nonzero().unwrap().realize().unwrap();
    assert_eq!(get_shape(&result), vec![3, 1]);
    let vals = result.to_ndarray::<i32>().unwrap();
    assert_eq!(vals[[0, 0]], 0); // index of 1
    assert_eq!(vals[[1, 0]], 2); // index of 2
    assert_eq!(vals[[2, 0]], 4); // index of 3
}
