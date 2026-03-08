use crate::Tensor;

// =========================================================================
// Triu Tests
// =========================================================================

#[test]
fn test_triu_basic() {
    // 3x3 matrix → upper triangular (zero below diagonal)
    let x = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).try_reshape(&[3, 3]).unwrap();
    let result = x.triu(0).unwrap();
    let arr = result.to_ndarray::<f32>().unwrap();
    let vals: Vec<f32> = arr.iter().copied().collect();
    // Expected: [1, 2, 3, 0, 5, 6, 0, 0, 9]
    assert_eq!(vals, vec![1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0]);
}

#[test]
fn test_triu_diagonal_positive() {
    // k=1: exclude main diagonal
    let x = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).try_reshape(&[3, 3]).unwrap();
    let result = x.triu(1).unwrap();
    let arr = result.to_ndarray::<f32>().unwrap();
    let vals: Vec<f32> = arr.iter().copied().collect();
    // Expected: [0, 2, 3, 0, 0, 6, 0, 0, 0]
    assert_eq!(vals, vec![0.0, 2.0, 3.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0]);
}

#[test]
fn test_triu_diagonal_negative() {
    // k=-1: include one subdiagonal
    let x = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).try_reshape(&[3, 3]).unwrap();
    let result = x.triu(-1).unwrap();
    let arr = result.to_ndarray::<f32>().unwrap();
    let vals: Vec<f32> = arr.iter().copied().collect();
    // Expected: [1, 2, 3, 4, 5, 6, 0, 8, 9]
    assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 8.0, 9.0]);
}

// =========================================================================
// Tril Tests
// =========================================================================

#[test]
fn test_tril_basic() {
    // 3x3 matrix → lower triangular (zero above diagonal)
    let x = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).try_reshape(&[3, 3]).unwrap();
    let result = x.tril(0).unwrap();
    let arr = result.to_ndarray::<f32>().unwrap();
    let vals: Vec<f32> = arr.iter().copied().collect();
    // Expected: [1, 0, 0, 4, 5, 0, 7, 8, 9]
    assert_eq!(vals, vec![1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0]);
}

#[test]
fn test_tril_diagonal_negative() {
    // k=-1: exclude main diagonal
    let x = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).try_reshape(&[3, 3]).unwrap();
    let result = x.tril(-1).unwrap();
    let arr = result.to_ndarray::<f32>().unwrap();
    let vals: Vec<f32> = arr.iter().copied().collect();
    // Expected: [0, 0, 0, 4, 0, 0, 7, 8, 0]
    assert_eq!(vals, vec![0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 7.0, 8.0, 0.0]);
}

#[test]
fn test_triu_non_square() {
    // 2x4 matrix
    let x = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).try_reshape(&[2, 4]).unwrap();
    let result = x.triu(0).unwrap();
    let arr = result.to_ndarray::<f32>().unwrap();
    let vals: Vec<f32> = arr.iter().copied().collect();
    // Expected: [1, 2, 3, 4, 0, 6, 7, 8]
    assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0, 0.0, 6.0, 7.0, 8.0]);
}
