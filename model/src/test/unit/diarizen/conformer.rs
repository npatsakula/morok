use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::{Layer, Linear, Module};

use crate::diarizen::{ConformerBlock, ConformerEncoder, PlainMultiHeadSelfAttention};

/// `(B, T, C) → (B, T, C)` for a single block.
#[test]
fn conformer_block_preserves_shape() {
    let block = ConformerBlock::empty(256, 1024, 4, 31);
    let x = Tensor::zeros(&[1, 7, 256], DType::Float32);
    let y = block.forward(&x).expect("symbolic forward");
    let shape = y.dims().unwrap();
    assert_eq!(shape, vec![1, 7, 256]);
}

/// 4-layer stack preserves shape.
#[test]
fn conformer_encoder_4_blocks_preserves_shape() {
    let enc = ConformerEncoder::empty(256, 1024, 4, 4, 31);
    let x = Tensor::zeros(&[1, 5, 256], DType::Float32);
    let y = enc.forward(&x).expect("symbolic forward");
    let shape = y.dims().unwrap();
    assert_eq!(shape, vec![1, 5, 256]);
}

/// State-dict keys match the upstream Python module names.
#[test]
fn conformer_block_state_dict_keys() {
    let block = ConformerBlock::empty(32, 64, 4, 7);
    let sd = block.state_dict("block");
    for key in [
        "block.ffn1.ln_norm.weight",
        "block.ffn1.w_1.weight",
        "block.ffn1.w_2.bias",
        "block.mha.ln_norm.weight",
        "block.mha.mha.linearQ.weight",
        "block.mha.mha.linearK.bias",
        "block.mha.mha.linearV.weight",
        "block.mha.mha.linearO.bias",
        "block.conv.ln_norm.weight",
        "block.conv.pointwise_conv1.weight",
        "block.conv.depthwise_conv.weight",
        // bn_norm reads PyTorch's own BatchNorm keys.
        "block.conv.bn_norm.weight",
        "block.conv.bn_norm.running_mean",
        "block.conv.bn_norm.running_var",
        "block.conv.pointwise_conv2.weight",
        "block.ffn2.w_1.weight",
        "block.ln_norm.weight",
    ] {
        assert!(sd.contains_key(key), "missing key: {key}");
    }
}

/// State-dict round-trip on a 4-layer encoder.
#[test]
fn conformer_encoder_state_dict_round_trip() {
    let enc = ConformerEncoder::empty(32, 64, 4, 4, 7);
    let sd = enc.state_dict("conformer");
    for i in 0..4 {
        assert!(sd.contains_key(&format!("conformer.conformer_layer.{i}.ffn1.w_1.weight")));
        assert!(sd.contains_key(&format!("conformer.conformer_layer.{i}.ln_norm.weight")));
    }
    let mut empty = ConformerEncoder::empty(32, 64, 4, 4, 7);
    empty.load_state_dict(&sd, "conformer").expect("round-trip");
}

/// The Conformer MHSA runs through `scaled_dot_product_attention`; it must
/// still equal the plain `softmax(scale · q @ kᵀ) @ v` the port started from.
#[test]
fn plain_mhsa_matches_reference_formula() {
    let (l, units, heads) = (6isize, 16isize, 4isize);
    let d_k = units / heads;
    let mha = PlainMultiHeadSelfAttention::empty(units as usize, heads as usize);

    let n = (l * units) as usize;
    let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.29).cos()).collect();
    let x = Tensor::from_slice(&data).try_reshape([1, l, units]).unwrap();

    let split = |lin: &Linear| {
        lin.forward(&x).unwrap().try_reshape([1, l, heads, d_k]).unwrap().try_permute(&[0, 2, 1, 3]).unwrap()
    };
    let (q, k, v) = (split(&mha.q), split(&mha.k), split(&mha.v));
    let scaling = Tensor::const_((d_k as f32).powf(-0.5), DType::Float32);
    let weights = q.try_mul(&scaling).unwrap().matmul(&k.try_transpose(-2, -1).unwrap()).unwrap().softmax(-1).unwrap();
    let out = weights.matmul(&v).unwrap().try_permute(&[0, 2, 1, 3]).unwrap().try_reshape([1, l, units]).unwrap();

    let want = mha.o.forward(&out).unwrap().to_vec::<f32>().unwrap();
    let got = mha.forward(&x).unwrap().to_vec::<f32>().unwrap();
    assert_eq!(got.len(), want.len());
    for (a, b) in got.iter().zip(&want) {
        assert!((a - b).abs() < 1e-5, "MHSA mismatch: {a} vs {b}");
    }
}
