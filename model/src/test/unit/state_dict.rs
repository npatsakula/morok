//! State-dict round-trip tests for GigaAM.
//!
//! Build the model via `with_random_weights`, emit its state dict via the
//! per-component `Module` impls (Encoder itself doesn't implement the trait;
//! we compose its sub-modules the same way `Encoder::from_state_dict` does),
//! assert the keys cover the encoder + head surface, then reload into a fresh
//! model. Only the BatchNorm-fold test realizes anything.
//!
//! Catches: a sub-module rename, a forgotten field in `state_dict()`, a
//! prefix mismatch between emit and load. Mirrors mexus's WeSpeaker round-trip
//! at `test/unit/wespeaker/model.rs`.

use crate::gigaam::{ConvNormType, GigaAm, GigaAmConfig, Head, TransducerConfig};
use crate::state::StateDict;
use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::Module as _;

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

/// Compose the model's full state dict from its component-level `Module`
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

/// The exact key set every module emits, spelled out. The `#[derive(Module)]`
/// migration must not move a single parameter, so this pins the checkpoint
/// contract independently of the impls that produce it — and asserts that a
/// non-empty prefix simply prepends `<prefix>.` (no leading dot, no drift).
#[test]
fn gigaam_state_dict_keys_are_exactly_the_checkpoint_contract() {
    let cfg = rnnt_test_config();
    let model = GigaAm::with_random_weights(cfg.clone());

    let ffn = |slot: &str| {
        ["norm.weight", "norm.bias", "linear1.weight", "linear1.bias", "linear2.weight", "linear2.bias"]
            .map(|leaf| format!("{slot}.{leaf}"))
            .to_vec()
    };
    let mut expected: Vec<String> = ["conv1_weight", "conv1_bias", "conv2_weight", "conv2_bias"]
        .iter()
        .map(|k| format!("subsampling.{k}"))
        .collect();
    for layer in 0..cfg.n_layers {
        let mut leaves = ffn("ffn1");
        leaves.extend(ffn("ffn2"));
        leaves.extend(["norm.weight", "norm.bias"].map(|l| format!("mhsa.{l}")));
        leaves.extend(["q", "k", "v", "out"].iter().flat_map(|p| [format!("mhsa.{p}_proj"), format!("mhsa.{p}_bias")]));
        leaves.extend(
            [
                "norm.weight",
                "norm.bias",
                "conv_norm.weight",
                "conv_norm.bias",
                "pw1_weight",
                "pw1_bias",
                "dw_weight",
                "dw_bias",
                "pw2_weight",
                "pw2_bias",
            ]
            .map(|l| format!("conv.{l}")),
        );
        leaves.extend(["final_norm.weight".to_string(), "final_norm.bias".to_string()]);
        expected.extend(leaves.into_iter().map(|leaf| format!("layers.{layer}.{leaf}")));
    }
    expected.extend(
        ["embed", "lstm.0.w_ih", "lstm.0.w_hh", "lstm.0.b_ih", "lstm.0.b_hh"].map(|k| format!("head.predictor.{k}")),
    );
    expected.extend(["enc_w", "enc_b", "pred_w", "pred_b", "out_w", "out_b"].map(|k| format!("head.joint.{k}")));
    expected.sort();

    let mut actual: Vec<String> = compose_state_dict(&model).into_keys().collect();
    actual.sort();
    assert_eq!(actual, expected);

    // A non-empty prefix only prepends; the root prefix grows no leading dot.
    let mut nested: Vec<String> = model.encoder.layers[0].state_dict("m").into_keys().collect();
    nested.sort();
    let mut bare: Vec<String> = model.encoder.layers[0].state_dict("").into_keys().map(|k| format!("m.{k}")).collect();
    bare.sort();
    assert_eq!(nested, bare);
    assert!(bare.iter().all(|k| !k.starts_with("m..")), "empty-prefix keys grew a leading dot");
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
        mhsa.q_weight_scale.as_ref().expect("q weight scale"),
        ffn1.linear1_scale.as_ref().expect("linear1 weight scale"),
    ] {
        assert_eq!(scale.dtype(), DType::Float32);
    }
}

/// The Conformer BatchNorm collapses into the depthwise conv at load — the
/// checkpoint carries `running_var`, and `1/sqrt(var + eps)` is computed in the
/// graph (it used to be a host `as_vec` loop in `remap`). This pins the folded
/// weights against the closed form and asserts the fold is one-shot: the
/// reloaded model emits no BN keys, and re-loading its own dict is a no-op.
#[test]
fn gigaam_batchnorm_folds_into_the_depthwise_conv() {
    const EPS: f32 = 1e-5;
    let mut cfg = test_config();
    cfg.conv_norm_type = ConvNormType::BatchNorm;
    let model = GigaAm::with_random_weights(cfg.clone());
    let sd = compose_state_dict(&model);

    let conv = &model.encoder.layers[0].conv;
    let host = |t: &Tensor| t.to_vec::<f32>().expect("realize");
    let (dw_w, dw_b) = (host(&conv.dw_weight), host(&conv.dw_bias));
    let param = |name: &str| host(&sd[&format!("layers.0.conv.{name}")]);
    let (scale, bias, mean, var) = (param("bn_scale"), param("bn_bias"), param("bn_mean"), param("bn_var"));

    let mut folded = GigaAm::with_random_weights(cfg.clone());
    reload(&mut folded, &sd);
    let folded_conv = &folded.encoder.layers[0].conv;
    let (got_w, got_b) = (host(&folded_conv.dw_weight), host(&folded_conv.dw_bias));

    let kernel = cfg.conv_kernel;
    for channel in 0..cfg.d_model {
        let s = scale[channel] / (var[channel] + EPS).sqrt();
        for k in 0..kernel {
            let (want, got) = (dw_w[channel * kernel + k] * s, got_w[channel * kernel + k]);
            assert!((want - got).abs() < 1e-5, "channel {channel} tap {k}: want {want}, got {got}");
        }
        let want = dw_b[channel] * s + bias[channel] - mean[channel] * s;
        assert!((want - got_b[channel]).abs() < 1e-5, "bias {channel}: want {want}, got {}", got_b[channel]);
    }

    // Folded: the BN parameters are gone from the dict, and a second load of
    // that dict must not fold a second time.
    let folded_sd = compose_state_dict(&folded);
    for key in ["bn_scale", "bn_bias", "bn_mean", "bn_var"] {
        assert!(!folded_sd.contains_key(&format!("layers.0.conv.{key}")), "folded model still emits {key}");
    }
    let mut again = folded.clone();
    reload(&mut again, &folded_sd);
    let twice = host(&again.encoder.layers[0].conv.dw_weight);
    assert_eq!(twice, got_w, "re-loading a folded dict folded again");
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
