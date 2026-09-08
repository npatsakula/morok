use crate::bgem3::{colbert_score, dense_score, sparse_score};
use svod_tensor::Tensor;

#[test]
fn dense_score_identity() {
    let q = Tensor::from_slice([1.0f32, 0.0, 0.0]).try_reshape([1, 3]).unwrap();
    let p = Tensor::from_slice([1.0f32, 0.0, 0.0]).try_reshape([1, 3]).unwrap();
    let s = dense_score(&q, &p).unwrap();
    s.realize().unwrap();
    assert!((s.as_vec::<f32>().unwrap()[0] - 1.0).abs() < 1e-6);
}

#[test]
fn sparse_score_basic() {
    let q = Tensor::from_slice([0.0f32, 2.0, 0.0, 3.0]).try_reshape([1, 4]).unwrap();
    let p = Tensor::from_slice([0.0f32, 1.0, 0.0, 1.0]).try_reshape([1, 4]).unwrap();
    let s = sparse_score(&q, &p).unwrap();
    s.realize().unwrap();
    assert!((s.as_vec::<f32>().unwrap()[0] - 5.0).abs() < 1e-6);
}

#[test]
fn colbert_score_basic() {
    let q = Tensor::from_slice([1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0]).try_reshape([2, 3]).unwrap();
    let p = Tensor::from_slice([1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).try_reshape([3, 3]).unwrap();
    let s = colbert_score(&q, &p).unwrap();
    s.realize().unwrap();
    assert!((s.as_vec::<f32>().unwrap()[0] - 1.0).abs() < 1e-5);
}
