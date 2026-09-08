use crate::bgem3::{ColbertHead, SparseHead};
use svod_dtype::DType;
use svod_tensor::Tensor;

#[test]
fn sparse_head_forward_shape() {
    let head = SparseHead::empty(32, 100, DType::Float32);
    let hidden = Tensor::zeros(&[2, 5, 32], DType::Float32).unwrap();
    let ids = Tensor::from_slice([0i64, 10, 20, 2, 1, 0, 10, 20, 2, 1]).try_reshape([2, 5]).unwrap();
    let mut out = head.forward(&hidden, &ids).unwrap();
    out.realize().unwrap();
    let s = out.dims().unwrap();
    assert_eq!(s[0], 2);
    assert_eq!(s[1], 100);
}

#[test]
fn sparse_head_zeros_special_tokens() {
    let head = SparseHead::empty(32, 100, DType::Float32);
    let hidden = Tensor::ones(&[1, 3, 32], DType::Float32).unwrap();
    let ids = Tensor::from_slice([0i64, 50, 2]).try_reshape([1, 3]).unwrap();
    let mut out = head.forward(&hidden, &ids).unwrap();
    out.realize().unwrap();
    let vals = out.as_vec::<f32>().unwrap();
    assert_eq!(vals[0], 0.0); // CLS
    assert_eq!(vals[1], 0.0); // PAD
    assert_eq!(vals[2], 0.0); // EOS
    assert_eq!(vals[3], 0.0); // UNK
}

#[test]
fn colbert_head_forward_shape() {
    let head = ColbertHead::empty(32, 32, DType::Float32);
    let hidden = Tensor::zeros(&[2, 6, 32], DType::Float32).unwrap();
    let mask = Tensor::from_slice([1i64, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0]).try_reshape([2, 6]).unwrap();
    let mut out = head.forward(&hidden, Some(&mask)).unwrap();
    out.realize().unwrap();
    let s = out.dims().unwrap();
    assert_eq!(s[0], 2);
    assert_eq!(s[1], 5);
    assert_eq!(s[2], 32);
}
