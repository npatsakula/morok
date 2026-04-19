use morok_tensor::Tensor;

use crate::gigaam::remap::remap_pytorch;
use crate::gigaam::{ConvNormType, GigaAmConfig, SubsamplingMode};
use crate::state::StateDict;

fn make_config(conv_norm: ConvNormType) -> GigaAmConfig {
    GigaAmConfig {
        max_batch_size: 8,
        n_mels: 64,
        d_model: 768,
        n_heads: 16,
        n_layers: 2,
        d_ff: 3072,
        conv_kernel: 5,
        subsampling_factor: 4,
        subsampling_mode: SubsamplingMode::Conv1d,
        subs_kernel_size: 5,
        conv_norm_type: conv_norm,
        vocab_size: 34,
        sample_rate: 16000,
        n_fft: 320,
        hop_length: 160,
        win_length: 320,
        mel_center: false,
        max_seq_len: 5000,
    }
}

fn fake_tensor() -> Tensor {
    Tensor::from_slice(&[0.0f32; 8])
}

#[test]
fn test_remap_subsampling() {
    let config = make_config(ConvNormType::LayerNorm);
    let mut sd = StateDict::new();
    sd.insert("encoder.pre_encode.conv.0.weight".into(), fake_tensor());
    sd.insert("encoder.pre_encode.conv.0.bias".into(), fake_tensor());
    sd.insert("encoder.pre_encode.conv.2.weight".into(), fake_tensor());
    sd.insert("encoder.pre_encode.conv.2.bias".into(), fake_tensor());

    let out = remap_pytorch(sd, &config).unwrap();
    assert!(out.contains_key("subsampling.conv1_weight"));
    assert!(out.contains_key("subsampling.conv1_bias"));
    assert!(out.contains_key("subsampling.conv2_weight"));
    assert!(out.contains_key("subsampling.conv2_bias"));
}

#[test]
fn test_remap_mhsa() {
    let config = make_config(ConvNormType::LayerNorm);
    let mut sd = StateDict::new();
    sd.insert("encoder.layers.0.norm_self_att.weight".into(), fake_tensor());
    sd.insert("encoder.layers.0.self_attn.linear_q.weight".into(), fake_tensor());
    sd.insert("encoder.layers.0.self_attn.linear_k.weight".into(), fake_tensor());
    sd.insert("encoder.layers.0.self_attn.linear_v.weight".into(), fake_tensor());
    sd.insert("encoder.layers.0.self_attn.linear_out.weight".into(), fake_tensor());
    sd.insert("encoder.layers.0.self_attn.linear_pos.weight".into(), fake_tensor());

    let out = remap_pytorch(sd, &config).unwrap();
    assert!(out.contains_key("layers.0.mhsa.norm.weight"));
    assert!(out.contains_key("layers.0.mhsa.q_proj"));
    assert!(out.contains_key("layers.0.mhsa.k_proj"));
    assert!(out.contains_key("layers.0.mhsa.v_proj"));
    assert!(out.contains_key("layers.0.mhsa.out_proj"));
    assert!(!out.contains_key("layers.0.mhsa.linear_pos.weight"));
}

#[test]
fn test_remap_conv_layernorm() {
    let config = make_config(ConvNormType::LayerNorm);
    let mut sd = StateDict::new();
    sd.insert("encoder.layers.0.conv.batch_norm.weight".into(), fake_tensor());
    sd.insert("encoder.layers.0.conv.batch_norm.bias".into(), fake_tensor());
    sd.insert("encoder.layers.0.conv.pointwise_conv1.weight".into(), fake_tensor());
    sd.insert("encoder.layers.0.conv.depthwise_conv.weight".into(), fake_tensor());

    let out = remap_pytorch(sd, &config).unwrap();
    assert!(out.contains_key("layers.0.conv.conv_norm.weight"));
    assert!(out.contains_key("layers.0.conv.conv_norm.bias"));
    assert!(out.contains_key("layers.0.conv.pw1_weight"));
    assert!(out.contains_key("layers.0.conv.dw_weight"));
    assert!(!out.contains_key("layers.0.conv.bn_scale"));
}

#[test]
fn test_remap_conv_batchnorm() {
    let config = make_config(ConvNormType::BatchNorm);
    let mut sd = StateDict::new();
    sd.insert("encoder.layers.0.conv.batch_norm.weight".into(), fake_tensor());
    sd.insert("encoder.layers.0.conv.batch_norm.bias".into(), fake_tensor());
    sd.insert("encoder.layers.0.conv.batch_norm.running_mean".into(), fake_tensor());
    sd.insert("encoder.layers.0.conv.batch_norm.running_var".into(), fake_tensor());
    sd.insert("encoder.layers.0.conv.batch_norm.num_batches_tracked".into(), fake_tensor());

    let out = remap_pytorch(sd, &config).unwrap();
    assert!(out.contains_key("layers.0.conv.bn_scale"));
    assert!(out.contains_key("layers.0.conv.bn_bias"));
    assert!(out.contains_key("layers.0.conv.bn_mean"));
    assert!(out.contains_key("layers.0.conv.bn_invstd"));
    assert!(!out.contains_key("layers.0.conv.conv_norm.weight"));
    assert!(!out.contains_key("layers.0.conv.batch_norm.num_batches_tracked"));
}

#[test]
fn test_remap_ffn() {
    let config = make_config(ConvNormType::LayerNorm);
    let mut sd = StateDict::new();
    sd.insert("encoder.layers.0.norm_feed_forward1.weight".into(), fake_tensor());
    sd.insert("encoder.layers.0.feed_forward1.linear1.weight".into(), fake_tensor());
    sd.insert("encoder.layers.0.feed_forward1.linear2.weight".into(), fake_tensor());
    sd.insert("encoder.layers.0.norm_out.weight".into(), fake_tensor());

    let out = remap_pytorch(sd, &config).unwrap();
    assert!(out.contains_key("layers.0.ffn1.norm.weight"));
    assert!(out.contains_key("layers.0.ffn1.linear1.weight"));
    assert!(out.contains_key("layers.0.ffn1.linear2.weight"));
    assert!(out.contains_key("layers.0.final_norm.weight"));
}

#[test]
fn test_remap_head() {
    let config = make_config(ConvNormType::LayerNorm);
    let mut sd = StateDict::new();
    sd.insert("head.decoder_layers.0.weight".into(), fake_tensor());
    sd.insert("head.decoder_layers.0.bias".into(), fake_tensor());

    let out = remap_pytorch(sd, &config).unwrap();
    assert!(out.contains_key("head.weight"));
    assert!(out.contains_key("head.bias"));
}
