use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::diarizen::{ConformerBlock, ConformerEncoder};
use crate::state::HasStateDict;

/// `(B, T, C) → (B, T, C)` for a single block.
#[test]
fn conformer_block_preserves_shape() {
    let block = ConformerBlock::empty(256, 1024, 4, 31);
    let x = Tensor::zeros(&[1, 7, 256], DType::Float32).unwrap();
    let y = block.forward(&x).expect("symbolic forward");
    let shape = y.dims().unwrap();
    assert_eq!(shape, vec![1, 7, 256]);
}

/// 4-layer stack preserves shape.
#[test]
fn conformer_encoder_4_blocks_preserves_shape() {
    let enc = ConformerEncoder::empty(256, 1024, 4, 4, 31);
    let x = Tensor::zeros(&[1, 5, 256], DType::Float32).unwrap();
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
        // bn_norm uses PyTorch BN keys via blocks::BatchNormWeights.
        "block.conv.bn_norm.weight",
        "block.conv.bn_norm.running_mean",
        "block.conv.bn_norm.invstd",
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
