use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::state::HasStateDict;
use crate::wavlm::{FeatureExtractor, wavlm_base, wavlm_large_s80_md};

/// On `(1, 256000)` waveform, the s80-md-v2 feature extractor produces
/// `(1, T, 211)` with T derived from `num_frames(256000)`. Symbolic forward
/// only (no `.realize()`).
#[test]
fn s80_md_v2_feature_extractor_shape() {
    let cfg = wavlm_large_s80_md();
    let fe = FeatureExtractor::empty(&cfg);

    let n_samples = 256_000;
    let wav = Tensor::zeros(&[1, n_samples], DType::Float32);
    let out = fe.forward(&wav).expect("symbolic forward");

    let shape = out.dims().unwrap();
    let expected_t = fe.num_frames(n_samples);
    assert_eq!(shape, vec![1, expected_t, 211]);

    // Sanity: cumulative stride is 5*2*2*2*2*2*2 = 320.
    assert_eq!(fe.total_stride(), 320);
    // 256000 samples through this stack lands at 799 frames (matches the
    // 16 s @ 16 kHz convention used by DiariZen).
    assert_eq!(expected_t, 799);
}

/// Base config uses GroupNorm mode: only block 0 carries a norm; blocks 1-6
/// are conv + GELU. Confirm both via the state-dict layout.
#[test]
fn base_feature_extractor_norm_mode() {
    let cfg = wavlm_base();
    let fe = FeatureExtractor::empty(&cfg);
    let sd = fe.state_dict("feature_extractor");

    // Block 0 has both conv and layer_norm keys.
    assert!(sd.contains_key("feature_extractor.conv_layers.0.conv.weight"));
    assert!(sd.contains_key("feature_extractor.conv_layers.0.layer_norm.weight"));
    assert!(sd.contains_key("feature_extractor.conv_layers.0.layer_norm.bias"));

    // Blocks 1..7 have conv only, no layer_norm keys.
    for i in 1..7 {
        assert!(sd.contains_key(&format!("feature_extractor.conv_layers.{i}.conv.weight")));
        assert!(
            !sd.contains_key(&format!("feature_extractor.conv_layers.{i}.layer_norm.weight")),
            "GroupNorm mode should not emit layer_norm key for block {i}"
        );
    }
    // No bias on convs (extractor_conv_bias=false).
    for i in 0..7 {
        assert!(
            !sd.contains_key(&format!("feature_extractor.conv_layers.{i}.conv.bias")),
            "extractor_conv_bias=false should not emit bias for block {i}"
        );
    }
}

/// LayerNorm mode: every block carries a layer_norm.
#[test]
fn large_s80_md_feature_extractor_norm_mode() {
    let cfg = wavlm_large_s80_md();
    let fe = FeatureExtractor::empty(&cfg);
    let sd = fe.state_dict("feature_extractor");

    for i in 0..7 {
        assert!(sd.contains_key(&format!("feature_extractor.conv_layers.{i}.conv.weight")));
        assert!(sd.contains_key(&format!("feature_extractor.conv_layers.{i}.layer_norm.weight")));
        assert!(sd.contains_key(&format!("feature_extractor.conv_layers.{i}.layer_norm.bias")));
    }
}

/// Round-trip: state_dict → load_state_dict on a fresh model preserves all keys.
#[test]
fn feature_extractor_state_dict_round_trip() {
    let cfg = wavlm_large_s80_md();
    let fe = FeatureExtractor::empty(&cfg);
    let sd = fe.state_dict("feature_extractor");

    let mut empty = FeatureExtractor::empty(&cfg);
    empty.load_state_dict(&sd, "feature_extractor").expect("round-trip");
}
