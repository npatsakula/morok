//! State-dict round-trip tests for GigaAM.
//!
//! Cheap — no `.realize()`, no JIT. Build the model via `with_random_weights`,
//! emit its state dict via the per-component `HasStateDict` impls (Encoder
//! itself doesn't implement the trait; we compose its sub-modules the same
//! way `Encoder::from_state_dict` does at `gigaam/encoder.rs:764`), assert
//! representative keys cover the encoder + head surface, then reload into a
//! fresh model.
//!
//! Catches: a sub-module rename, a forgotten field in `state_dict()`, a
//! prefix mismatch between emit and load. Mirrors mexus's WeSpeaker round-trip
//! at `test/unit/wespeaker/model.rs`.

use crate::gigaam::{GigaAm, GigaAmConfig, Head, TransducerConfig};
use crate::state::{HasStateDict, StateDict};
use svod_dtype::DType;
use svod_tensor::Tensor;

use super::batch::test_config;

fn rnnt_test_config() -> GigaAmConfig {
    let mut cfg = test_config();
    // Mirror an `v3_e2e_rnnt`-style RN-T head: small predictor + joint, blank
    // token at the end of the vocabulary.
    let vocabulary: Vec<String> = (0..8).map(|i| format!("p{i}")).collect();
    cfg.transducer = Some(TransducerConfig {
        pred_hidden: 16,
        pred_rnn_layers: 1,
        joint_hidden: 16,
        num_classes: vocabulary.len() + 1,
        max_symbols_per_step: 10,
        vocabulary,
        sentencepiece: false,
    });
    cfg
}

/// Compose the model's full state dict from its component-level `HasStateDict`
/// impls. The encoder itself doesn't implement the trait (only its
/// sub-modules), so we mirror `Encoder::from_state_dict`'s key convention:
/// `subsampling.*` and `layers.{i}.*` (no `encoder.` prefix on disk — the
/// `remap::remap_pytorch` step at `gigaam/model.rs:146` strips it for
/// PyTorch checkpoints).
fn compose_state_dict(model: &GigaAm) -> StateDict {
    let mut sd = model.encoder.subsampling.state_dict("subsampling");
    for (i, layer) in model.encoder.layers.iter().enumerate() {
        sd.extend(layer.state_dict(&format!("layers.{i}")));
    }
    match &model.head {
        Head::Ctc(h) => sd.extend(h.state_dict("head")),
        Head::Rnnt { head, .. } => sd.extend(head.state_dict("head")),
    }
    sd
}

/// Reload the same model in-place from a state dict. Matches the per-component
/// `load_state_dict` flow used by `GigaAm::from_state_dict`.
fn reload(model: &mut GigaAm, sd: &StateDict) {
    model.encoder.subsampling.load_state_dict(sd, "subsampling").expect("subsampling reload");
    for (i, layer) in model.encoder.layers.iter_mut().enumerate() {
        layer.load_state_dict(sd, &format!("layers.{i}")).expect("conformer layer reload");
    }
    match &mut model.head {
        Head::Ctc(h) => h.load_state_dict(sd, "head").expect("ctc head reload"),
        Head::Rnnt { head, .. } => head.load_state_dict(sd, "head").expect("rnnt head reload"),
    }
}

#[test]
fn gigaam_state_dict_round_trip_ctc() {
    let cfg = test_config();
    let model = GigaAm::with_random_weights(cfg.clone());

    let sd = compose_state_dict(&model);

    // Representative keys covering subsampling, every conformer-layer sub-
    // module (ffn1 / mhsa / conv / ffn2 / final_norm), and the CTC head.
    for key in [
        "subsampling.conv1_weight",
        "layers.0.ffn1.norm.weight",
        "layers.0.ffn1.linear1.weight",
        "layers.0.ffn1.linear2.bias",
        "layers.0.mhsa.norm.weight",
        "layers.0.conv.norm.weight",
        "layers.0.ffn2.linear1.weight",
        "layers.0.final_norm.weight",
        "layers.1.ffn1.norm.weight",
        "head.weight",
        "head.bias",
    ] {
        assert!(sd.contains_key(key), "missing key: {key}");
    }

    // RN-T-specific keys must NOT appear for a CTC model.
    assert!(!sd.contains_key("head.predictor.embed"), "CTC head must not emit predictor.embed");
    assert!(!sd.contains_key("head.joint.enc_w"), "CTC head must not emit joint.enc_w");

    let mut empty = GigaAm::with_random_weights(cfg);
    reload(&mut empty, &sd);
}

#[test]
fn gigaam_state_dict_round_trip_rnnt() {
    let cfg = rnnt_test_config();
    let model = GigaAm::with_random_weights(cfg.clone());

    let sd = compose_state_dict(&model);

    for key in [
        "subsampling.conv1_weight",
        "layers.0.ffn1.norm.weight",
        "layers.1.final_norm.weight",
        "head.predictor.embed",
        "head.predictor.lstm.0.w_ih",
        "head.predictor.lstm.0.b_hh",
        "head.joint.enc_w",
        "head.joint.enc_b",
        "head.joint.pred_w",
        "head.joint.out_w",
        "head.joint.out_b",
    ] {
        assert!(sd.contains_key(key), "missing key: {key}");
    }

    // CTC-specific direct projection keys must NOT appear.
    assert!(!sd.contains_key("head.weight"), "RN-T head must not emit a bare `head.weight`");
    assert!(!sd.contains_key("head.bias"), "RN-T head must not emit a bare `head.bias`");

    let mut empty = GigaAm::with_random_weights(cfg);
    reload(&mut empty, &sd);
}

#[test]
fn gigaam_encoder_dtype_conversion_leaves_head_fp32() {
    let cfg = test_config();
    let source = GigaAm::with_random_weights(cfg.clone());
    let sd = compose_state_dict(&source);

    let default_model = GigaAm::from_state_dict(&sd, cfg.clone(), None).expect("load default encoder");
    assert_eq!(default_model.encoder.input_dtype(), DType::Float16);

    let model = GigaAm::from_state_dict_with_encoder_dtype(&sd, cfg, None, DType::BFloat16).expect("load BF16 encoder");

    assert_eq!(model.encoder.input_dtype(), DType::BFloat16);
    assert_eq!(model.encoder.layers[0].mhsa.q_proj.dtype(), DType::BFloat16);
    assert_eq!(model.encoder.layers[0].mhsa.q_bias.dtype(), DType::BFloat16);
    assert_eq!(model.encoder.layers[0].final_norm.weight.dtype(), DType::BFloat16);
    let head = model.head.as_ctc().expect("CTC head");
    assert_eq!(head.weight.dtype(), DType::Float32);
    assert_eq!(head.bias.dtype(), DType::Float32);
}

#[test]
fn gigaam_quantization_scales_keep_checkpoint_dtype() {
    // Int8 encoder weights keep their scales in the state dict (they are applied
    // at matmul time, not folded), so the encoder-dtype coercion must exempt both
    // scale spellings: `<x>.weight_scale` (FFN) and `<x>_weight_scale` (MHSA).
    let cfg = test_config();
    let source = GigaAm::with_random_weights(cfg.clone());
    let mut sd = compose_state_dict(&source);
    for (weight, scale) in [
        ("layers.0.mhsa.q_proj", "layers.0.mhsa.q_weight_scale"),
        ("layers.0.ffn1.linear1.weight", "layers.0.ffn1.linear1.weight_scale"),
    ] {
        let quantized = sd[weight].cast(DType::Int8);
        let out = quantized.dim_const(0).unwrap();
        sd.insert(weight.into(), quantized);
        sd.insert(scale.into(), Tensor::full(&[out], 1.0f32, DType::Float32));
    }

    let model = GigaAm::from_state_dict_with_encoder_dtype(&sd, cfg, None, DType::Float16).expect("load FP16 encoder");

    let mhsa = &model.encoder.layers[0].mhsa;
    let ffn1 = &model.encoder.layers[0].ffn1;
    for scale in [
        &mhsa.q_quantization.as_ref().expect("q quantization").weight_scale,
        &ffn1.linear1_quantization.as_ref().expect("linear1 quantization").weight_scale,
    ] {
        assert_eq!(scale.dtype(), DType::Float32);
    }
}

#[test]
fn gigaam_rejects_ctc_head_shape_mismatched_with_config() {
    let cfg = test_config();
    let source = GigaAm::with_random_weights(cfg.clone());
    let mut sd = compose_state_dict(&source);
    sd.insert("head.weight".into(), Tensor::full(&[257, cfg.d_model, 1], 0.0f32, DType::Float32));
    sd.insert("head.bias".into(), Tensor::full(&[257], 0.0f32, DType::Float32));

    let err = match GigaAm::from_state_dict(&sd, cfg, None) {
        Ok(_) => panic!("mismatched CTC head must fail"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("CTC head shapes"), "unexpected error: {err}");
    assert!(err.to_string().contains("num_classes=34"), "unexpected error: {err}");
}
