use svod_dtype::DType;
use svod_tensor::Tensor;

use svod_tensor::nn::{Module, StateDict};

use crate::diarizen::{DiariZenConfig, DiariZenSegmentationModel};

/// State-dict round-trip on a tiny config (2 WavLM layers, 1 Conformer block).
#[test]
fn segmentation_state_dict_round_trip() {
    let cfg = tiny_cfg();
    let model = DiariZenSegmentationModel::empty(cfg.clone());
    let sd = model.state_dict("");

    // The published checkpoint layout: WavLM keys carry `wavlm_model.` prefix.
    assert!(sd.contains_key("wavlm_model.feature_extractor.conv_layers.0.conv.weight"));
    assert!(sd.contains_key("wavlm_model.encoder.transformer.pos_conv_embed.conv.weight"));
    assert!(sd.contains_key("wavlm_model.encoder.transformer.layers.0.attention.rel_attn_embed.weight"));
    // Head keys live at the top.
    assert!(sd.contains_key("weight_sum.weight"));
    assert!(sd.contains_key("proj.weight"));
    assert!(sd.contains_key("proj.bias"));
    assert!(sd.contains_key("lnorm.weight"));
    assert!(sd.contains_key("conformer.conformer_layer.0.ffn1.w_1.weight"));
    assert!(sd.contains_key("classifier.weight"));
    assert!(sd.contains_key("classifier.bias"));

    let mut empty = DiariZenSegmentationModel::empty(cfg);
    empty.load_state_dict(&sd, "").expect("round-trip");
}

/// `from_state_dict` loads the published key layout straight through the
/// derived [`Module`] impl. Inert PyTorch buffers must be dropped silently.
#[test]
fn from_state_dict_drops_inert_buffers() {
    let cfg = tiny_cfg();
    let model = DiariZenSegmentationModel::empty(cfg.clone());
    let mut sd: StateDict = model.state_dict("");

    sd.insert(
        "conformer.conformer_layer.0.conv.bn_norm.num_batches_tracked".to_string(),
        Tensor::zeros(&[1], DType::Float32),
    );
    sd.insert(
        "wavlm_model.encoder.transformer.layers.0.attention.hard_concrete_for_heads.log_alpha".to_string(),
        Tensor::zeros(&[1], DType::Float32),
    );

    DiariZenSegmentationModel::from_state_dict(&sd, cfg).expect("from_state_dict");
}

/// Forward returns `(B, T, K)` log-probabilities. Uses tiny config to keep
/// the symbolic graph small (no `.realize()`).
#[test]
fn segmentation_forward_shape() {
    let cfg = tiny_cfg();
    let model = DiariZenSegmentationModel::empty(cfg.clone());

    // (B, channels=1, samples) — small enough to stay cheap.
    let wav = Tensor::zeros(&[1, 1, 4096], DType::Float32);
    let out = model.forward(&wav).expect("forward");
    let shape = out.dims().unwrap();
    assert_eq!(shape.len(), 3);
    assert_eq!(shape[0], 1);
    assert!(shape[1] > 0);
    assert_eq!(shape[2], cfg.powerset_class_count(), "last axis should be powerset count K");
}

/// `forward_logits` skips the final log-softmax — useful for stage-level
/// parity tests.
#[test]
fn segmentation_forward_logits_shape() {
    let cfg = tiny_cfg();
    let model = DiariZenSegmentationModel::empty(cfg.clone());
    let wav = Tensor::zeros(&[1, 1, 4096], DType::Float32);
    let logits = model.forward_logits(&wav).expect("forward_logits");
    let shape = logits.dims().unwrap();
    assert_eq!(shape[0], 1);
    assert_eq!(shape[2], cfg.powerset_class_count());
}

pub(super) fn tiny_cfg() -> DiariZenConfig {
    let mut cfg = DiariZenConfig::diarizen_wavlm_large_s80_md_v2();
    cfg.wavlm.encoder_embed_dim = 32;
    cfg.wavlm.encoder_head_dim = 8;
    cfg.wavlm.encoder_num_layers = 2;
    cfg.wavlm.encoder_use_attention = vec![true; 2];
    cfg.wavlm.encoder_use_feed_forward = vec![true; 2];
    cfg.wavlm.encoder_total_num_heads = vec![4; 2];
    cfg.wavlm.encoder_remaining_heads = vec![vec![0, 1, 2, 3]; 2];
    cfg.wavlm.encoder_ff_interm_features = vec![64; 2];
    cfg.wavlm.max_batch_size = 1;
    cfg.attention_in = 16;
    cfg.ffn_hidden = 32;
    cfg.num_head = 2;
    cfg.num_layer = 1;
    cfg.kernel_size = 7;
    cfg
}
