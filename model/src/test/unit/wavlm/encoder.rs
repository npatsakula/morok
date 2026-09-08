use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::state::HasStateDict;
use crate::wavlm::{Encoder, EncoderLayer, WavLm, WavLmConfig, wavlm_large_s80_md};

// ---------------------------------------------------------------------------
// EncoderLayer
// ---------------------------------------------------------------------------

#[test]
fn encoder_layer_state_dict_with_attention() {
    let cfg = small_cfg();
    let layer = EncoderLayer::empty(&cfg, 0);
    assert!(layer.attention.is_some());
    let sd = layer.state_dict("layers.0");
    for key in [
        "layers.0.layer_norm.weight",
        "layers.0.layer_norm.bias",
        "layers.0.attention.q_proj.weight",
        "layers.0.attention.gru_rel_pos_const",
        "layers.0.final_layer_norm.weight",
        "layers.0.feed_forward.intermediate_dense.weight",
        "layers.0.feed_forward.output_dense.weight",
    ] {
        assert!(sd.contains_key(key), "missing: {key}");
    }
}

#[test]
fn encoder_layer_state_dict_without_attention() {
    // Force attention to be absent on layer 1 by both flags.
    let mut cfg = small_cfg();
    cfg.encoder_use_attention[1] = false;
    cfg.encoder_remaining_heads[1] = vec![];
    let layer = EncoderLayer::empty(&cfg, 1);
    assert!(layer.attention.is_none());

    let sd = layer.state_dict("layers.1");
    assert!(sd.contains_key("layers.1.layer_norm.weight"), "layer_norm always present");
    assert!(sd.contains_key("layers.1.final_layer_norm.weight"), "final_layer_norm always present");
    let attn_keys: Vec<&String> = sd.keys().filter(|k| k.contains("attention")).collect();
    assert!(attn_keys.is_empty(), "no attention keys expected, found: {attn_keys:?}");
}

#[test]
fn encoder_layer_forward_skips_attention_when_none() {
    let mut cfg = small_cfg();
    cfg.encoder_use_attention[0] = false;
    cfg.encoder_remaining_heads[0] = vec![];
    let layer = EncoderLayer::empty(&cfg, 0);
    assert!(layer.attention.is_none());

    let x = Tensor::zeros(&[1, 4, cfg.encoder_embed_dim], DType::Float32);
    // No position_bias needed when attention is None.
    let out = layer.forward(&x, None).expect("forward without attention");
    let shape = out.dims().unwrap();
    assert_eq!(shape, vec![1, 4, cfg.encoder_embed_dim]);
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

#[test]
fn encoder_extract_features_returns_n_plus_one() {
    let cfg = small_cfg();
    let enc = Encoder::empty(&cfg);

    let l = 4;
    let features = Tensor::zeros(&[1, l, cfg.extractor_out_dim()], DType::Float32);
    let out = enc.extract_features(&features).expect("symbolic forward");
    assert_eq!(out.len(), cfg.encoder_num_layers + 1, "extract_features must return num_layers + 1 tensors");
    for t in &out {
        let shape = t.dims().unwrap();
        assert_eq!(shape, vec![1, l, cfg.encoder_embed_dim]);
    }
}

#[test]
fn encoder_state_dict_keys_present() {
    let cfg = small_cfg();
    let enc = Encoder::empty(&cfg);
    let sd = enc.state_dict("encoder");

    for key in [
        "encoder.feature_projection.layer_norm.weight",
        "encoder.feature_projection.projection.weight",
        "encoder.transformer.pos_conv_embed.conv.weight",
        "encoder.transformer.pos_conv_embed.conv.bias",
        "encoder.transformer.layer_norm.weight",
        "encoder.transformer.layer_norm.bias",
        "encoder.transformer.layers.0.attention.q_proj.weight",
        "encoder.transformer.layers.0.attention.rel_attn_embed.weight",
        "encoder.transformer.layers.0.feed_forward.intermediate_dense.weight",
    ] {
        assert!(sd.contains_key(key), "missing key: {key}");
    }
}

#[test]
fn encoder_state_dict_round_trip() {
    let cfg = small_cfg();
    let enc = Encoder::empty(&cfg);
    let sd = enc.state_dict("encoder");
    let mut empty = Encoder::empty(&cfg);
    empty.load_state_dict(&sd, "encoder").expect("round-trip");
}

// ---------------------------------------------------------------------------
// WavLm root
// ---------------------------------------------------------------------------

#[test]
fn wavlm_extract_features_returns_correct_count() {
    let cfg = small_cfg();
    let model = WavLm::empty(cfg.clone());

    // 4 frames after the feature extractor → make sure we have enough input
    // samples to produce >= 1 output frame. Each block needs kernel_size
    // samples minimum; cumulative kernel ≈ sum of inflated kernels.
    let samples = 4096;
    let wav = Tensor::zeros(&[1, samples], DType::Float32);
    let intermediates = model.extract_features(&wav).expect("forward");

    assert_eq!(intermediates.len(), cfg.encoder_num_layers + 1);
}

#[test]
fn wavlm_state_dict_round_trip() {
    let cfg = small_cfg();
    let model = WavLm::empty(cfg.clone());
    let sd = model.state_dict("");

    // Spot-check coverage at the boundaries.
    assert!(sd.contains_key("feature_extractor.conv_layers.0.conv.weight"));
    assert!(sd.contains_key("encoder.feature_projection.projection.weight"));
    assert!(sd.contains_key("encoder.transformer.pos_conv_embed.conv.weight"));
    assert!(sd.contains_key("encoder.transformer.layers.0.attention.rel_attn_embed.weight"));
    assert!(
        sd.contains_key(&format!("encoder.transformer.layers.{}.final_layer_norm.weight", cfg.encoder_num_layers - 1))
    );

    let mut empty = WavLm::empty(cfg);
    empty.load_state_dict(&sd, "").expect("round-trip");
}

#[test]
fn wavlm_state_dict_round_trip_with_skipped_attention() {
    // Cfg with attention skipped on some layers — verify the loader tolerates
    // the *absence* of attention keys on those layers (full-scale s80_md_v2
    // has 4 such layers at indices 9/12/16/17).
    let cfg = wavlm_large_s80_md();
    let model = WavLm::empty(cfg.clone());
    let sd = model.state_dict("");

    // Skipped layer (index 9): attention keys must NOT appear.
    assert!(!sd.contains_key("encoder.transformer.layers.9.attention.q_proj.weight"));
    // But layer_norm + final_layer_norm + feed_forward must.
    assert!(sd.contains_key("encoder.transformer.layers.9.layer_norm.weight"));
    assert!(sd.contains_key("encoder.transformer.layers.9.final_layer_norm.weight"));
    assert!(sd.contains_key("encoder.transformer.layers.9.feed_forward.intermediate_dense.weight"));

    let mut empty = WavLm::empty(cfg);
    empty.load_state_dict(&sd, "").expect("round-trip with skipped attention");
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Small WavLM config (4 layers, embed_dim 32, head_dim 8, 4 heads) used for
/// shape tests. Keeps lazy graphs small and avoids 24-layer compilation cost.
fn small_cfg() -> WavLmConfig {
    let mut cfg = wavlm_large_s80_md();
    cfg.encoder_embed_dim = 32;
    cfg.encoder_head_dim = 8;
    cfg.encoder_num_layers = 4;
    cfg.encoder_use_attention = vec![true; 4];
    cfg.encoder_use_feed_forward = vec![true; 4];
    cfg.encoder_total_num_heads = vec![4; 4];
    cfg.encoder_remaining_heads = vec![vec![0, 1, 2, 3]; 4];
    cfg.encoder_ff_interm_features = vec![64; 4];
    cfg
}
