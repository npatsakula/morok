use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::diarizen::{DiariZenConfig, DiariZenSegmentationModel, split_diarizen_state_dict};
use crate::state::{HasStateDict, StateDict};

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

/// `split_diarizen_state_dict` peels the WavLM prefix and folds BN
/// `running_var → invstd`.
#[test]
fn remap_splits_wavlm_and_head() {
    let cfg = tiny_cfg();
    let model = DiariZenSegmentationModel::empty(cfg);
    let sd = model.state_dict("");

    let (wavlm_sd, head_sd) = split_diarizen_state_dict(sd).expect("split");
    // WavLm keys lose the `wavlm_model.` prefix.
    assert!(wavlm_sd.contains_key("feature_extractor.conv_layers.0.conv.weight"));
    assert!(wavlm_sd.contains_key("encoder.transformer.pos_conv_embed.conv.weight"));
    // Head keys untouched.
    assert!(head_sd.contains_key("weight_sum.weight"));
    assert!(head_sd.contains_key("classifier.weight"));
    // BN invstd key is present through the split (BN folding happens
    // separately in the production loader; the round-trip path skips it).
    let bn_key = "conformer.conformer_layer.0.conv.bn_norm.invstd";
    assert!(head_sd.contains_key(bn_key), "BN invstd key present");
}

/// `from_state_dict` exercises split + BN fold + load. The fold reads
/// `running_var` via `as_vec::<f32>`, which requires realized data — so
/// simulate a PyTorch checkpoint by renaming `invstd` keys back to
/// `running_var` and realize them. Also inject inert PyTorch buffers and
/// confirm they're dropped silently.
#[test]
fn from_state_dict_via_remap() {
    let cfg = tiny_cfg();
    let model = DiariZenSegmentationModel::empty(cfg.clone());
    let mut sd: StateDict = model.state_dict("");

    // Simulate a PyTorch checkpoint: rename `invstd` keys back to
    // `running_var` and materialize so `fold_batchnorm` can read the bytes.
    let invstd_keys: Vec<String> = sd.keys().filter(|k| k.ends_with(".invstd")).cloned().collect();
    for key in invstd_keys {
        let t = sd.remove(&key).unwrap();
        t.realize().expect("realize invstd");
        let var_key = key.replace(".invstd", ".running_var");
        sd.insert(var_key, t);
    }

    // Inert PyTorch buffers must be dropped silently.
    sd.insert("wavlm_model.feature_extractor.dummy_weight".to_string(), Tensor::zeros(&[1], DType::Float32).unwrap());
    sd.insert(
        "conformer.conformer_layer.0.conv.bn_norm.num_batches_tracked".to_string(),
        Tensor::zeros(&[1], DType::Float32).unwrap(),
    );

    let _model = DiariZenSegmentationModel::from_state_dict(&sd, cfg).expect("from_state_dict");
}

/// Forward returns `(B, T, K)` log-probabilities. Uses tiny config to keep
/// the symbolic graph small (no `.realize()`).
#[test]
fn segmentation_forward_shape() {
    let cfg = tiny_cfg();
    let model = DiariZenSegmentationModel::empty(cfg.clone());

    // (B, channels=1, samples) — small enough to stay cheap.
    let wav = Tensor::zeros(&[1, 1, 4096], DType::Float32).unwrap();
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
    let wav = Tensor::zeros(&[1, 1, 4096], DType::Float32).unwrap();
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
