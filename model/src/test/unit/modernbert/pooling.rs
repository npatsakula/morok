use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::modernbert::{cls, masked_mean};

/// Masked mean: padded positions must contribute nothing. With
/// hidden = [[[1,1],[2,2],[3,3]]] (B=1, L=3, D=2) and mask [[1,1,0]] (token 2
/// padded), the mean is over the two real tokens only → (1.5, 1.5). Without
/// masking it would be (2.0, 2.0) — the divergence proves the mask is honored.
#[test]
fn masked_mean_excludes_padding() {
    let h = Tensor::from_slice([1.0f32, 1.0, 2.0, 2.0, 3.0, 3.0]).try_reshape([1isize, 3, 2]).unwrap();
    let mask = Tensor::from_slice([1.0f32, 1.0, 0.0]).try_reshape([1isize, 3]).unwrap().cast(DType::Bool).unwrap();

    let out = masked_mean(&h, &mask).unwrap();
    out.realize().unwrap();
    let v = out.as_vec::<f32>().unwrap();
    assert_eq!(v.len(), 2, "(B, D) = (1, 2)");
    assert!((v[0] - 1.5).abs() < 1e-5, "mean over real tokens: got {}", v[0]);
    assert!((v[1] - 1.5).abs() < 1e-5);
}

/// Masked mean: all tokens real (mask all ones) == plain mean.
#[test]
fn masked_mean_all_real_is_plain_mean() {
    let h = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).try_reshape([1isize, 3, 2]).unwrap();
    let mask = Tensor::from_slice([1.0f32, 1.0, 1.0]).try_reshape([1isize, 3]).unwrap().cast(DType::Bool).unwrap();

    let out = masked_mean(&h, &mask).unwrap();
    out.realize().unwrap();
    let v = out.as_vec::<f32>().unwrap();
    // mean over (1,2),(3,4),(5,6) = (3.0, 4.0)
    assert!((v[0] - 3.0).abs() < 1e-5, "got {}", v[0]);
    assert!((v[1] - 4.0).abs() < 1e-5);
}

/// Masked mean over a real batch (B=2): each row pools independently using its
/// own mask, exercising the batch dimension.
#[test]
fn masked_mean_two_rows() {
    // row 0: tokens (1,1),(2,2),(3,3); row 1: tokens (4,4),(5,5),(6,6)
    let h = Tensor::from_slice([1.0f32, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0])
        .try_reshape([2isize, 3, 2])
        .unwrap();
    // row 0: last token padded; row 1: all real.
    let mask = Tensor::from_slice([1.0f32, 1.0, 0.0, 1.0, 1.0, 1.0])
        .try_reshape([2isize, 3])
        .unwrap()
        .cast(DType::Bool)
        .unwrap();

    let out = masked_mean(&h, &mask).unwrap();
    out.realize().unwrap();
    let v = out.as_vec::<f32>().unwrap();
    assert_eq!(v.len(), 4, "(B, D) = (2, 2)");
    // row 0: mean of (1,1),(2,2) = (1.5, 1.5)
    assert!((v[0] - 1.5).abs() < 1e-5, "row 0: {}", v[0]);
    assert!((v[1] - 1.5).abs() < 1e-5);
    // row 1: mean of (4,4),(5,5),(6,6) = (5.0, 5.0)
    assert!((v[2] - 5.0).abs() < 1e-5, "row 1: {}", v[2]);
    assert!((v[3] - 5.0).abs() < 1e-5);
}

/// CLS pooling: takes the first token's embedding verbatim.
#[test]
fn cls_takes_first_token() {
    let h = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).try_reshape([1isize, 3, 2]).unwrap();
    let out = cls(&h).unwrap();
    out.realize().unwrap();
    let v = out.as_vec::<f32>().unwrap();
    assert_eq!(v.len(), 2);
    assert_eq!(v[0], 1.0);
    assert_eq!(v[1], 2.0);
}
