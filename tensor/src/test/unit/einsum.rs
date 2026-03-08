use crate::*;

fn get_shape(tensor: &Tensor) -> Vec<usize> {
    tensor.uop().shape().unwrap().unwrap().iter().map(|s| s.as_const().unwrap()).collect()
}

#[test]
fn test_einsum_matrix_multiply() {
    // ij,jk->ik (matrix multiply)
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]).try_reshape(&[2, 2]).unwrap();
    let b = Tensor::from_slice([5.0f32, 6.0, 7.0, 8.0]).try_reshape(&[2, 2]).unwrap();
    let result = Tensor::einsum("ij,jk->ik", &[&a, &b]).unwrap();
    let result = result.realize().unwrap();
    assert_eq!(get_shape(&result), vec![2, 2]);
    let vals = result.to_ndarray::<f32>().unwrap();
    // [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
    assert_eq!(vals[[0, 0]], 19.0);
    assert_eq!(vals[[0, 1]], 22.0);
    assert_eq!(vals[[1, 0]], 43.0);
    assert_eq!(vals[[1, 1]], 50.0);
}

#[test]
fn test_einsum_trace() {
    // ii-> (trace)
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]).try_reshape(&[2, 2]).unwrap();
    let result = Tensor::einsum("ii->", &[&a]).unwrap();
    let result = result.realize().unwrap();
    let vals = result.to_ndarray::<f32>().unwrap();
    // trace = 1 + 4 = 5
    assert!((vals[[]] - 5.0).abs() < 1e-5);
}

#[test]
fn test_einsum_transpose() {
    // ij->ji (transpose)
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).try_reshape(&[2, 3]).unwrap();
    let result = Tensor::einsum("ij->ji", &[&a]).unwrap();
    let result = result.realize().unwrap();
    assert_eq!(get_shape(&result), vec![3, 2]);
}

#[test]
fn test_einsum_outer_product() {
    // i,j->ij (outer product)
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let b = Tensor::from_slice([4.0f32, 5.0]);
    let result = Tensor::einsum("i,j->ij", &[&a, &b]).unwrap();
    let result = result.realize().unwrap();
    assert_eq!(get_shape(&result), vec![3, 2]);
    let vals = result.to_ndarray::<f32>().unwrap();
    assert_eq!(vals[[0, 0]], 4.0);
    assert_eq!(vals[[0, 1]], 5.0);
    assert_eq!(vals[[1, 0]], 8.0);
    assert_eq!(vals[[1, 1]], 10.0);
    assert_eq!(vals[[2, 0]], 12.0);
    assert_eq!(vals[[2, 1]], 15.0);
}

#[test]
fn test_einsum_batch_matmul() {
    // bij,bjk->bik (batch matmul)
    let a = Tensor::from_slice([1.0f32, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0]).try_reshape(&[2, 2, 2]).unwrap();
    let b = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).try_reshape(&[2, 2, 2]).unwrap();
    let result = Tensor::einsum("bij,bjk->bik", &[&a, &b]).unwrap();
    let result = result.realize().unwrap();
    assert_eq!(get_shape(&result), vec![2, 2, 2]);
}

#[test]
fn test_einsum_sum_all() {
    // ij-> (sum all elements)
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]).try_reshape(&[2, 2]).unwrap();
    let result = Tensor::einsum("ij->", &[&a]).unwrap();
    let result = result.realize().unwrap();
    let vals = result.to_ndarray::<f32>().unwrap();
    assert!((vals[[]] - 10.0).abs() < 1e-5);
}
