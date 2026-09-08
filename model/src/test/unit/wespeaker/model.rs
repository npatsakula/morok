use crate::state::HasStateDict;
use crate::wespeaker::{WeSpeakerConfig, WeSpeakerResNet34};

/// State-dict round-trip: `with_zero_weights` produces a model whose
/// `state_dict()` keys cover every layer the loader expects, and
/// `load_state_dict` accepts that dict back without error.
#[test]
fn state_dict_round_trip() {
    let cfg = WeSpeakerConfig::new();
    let model = WeSpeakerResNet34::with_zero_weights(cfg.clone());

    let sd = model.state_dict("");

    // Critical keys for the WeSpeaker ResNet34 LM checkpoint shape.
    for key in [
        "conv1.weight",
        "bn1.weight",
        "bn1.bias",
        "bn1.running_mean",
        "bn1.invstd",
        "layer1.0.conv1.weight",
        "layer1.0.bn1.weight",
        "layer1.2.conv2.weight",
        "layer2.0.downsample.0.weight",
        "layer2.0.downsample.1.invstd",
        "layer3.0.downsample.0.weight",
        "layer4.0.downsample.0.weight",
        "layer4.2.bn2.weight",
        "seg_1.weight",
        "seg_1.bias",
    ] {
        assert!(sd.contains_key(key), "missing key: {key}");
    }

    // No stray num_batches_tracked or other PyTorch metadata — we strip those
    // on load. Round-tripping our own state_dict is the pre-fold layout, so
    // no stripping needed.

    // Load it back.
    let mut empty = WeSpeakerResNet34::with_zero_weights(cfg);
    empty.load_state_dict(&sd, "").expect("load round-trip");
}

#[test]
fn config_max_batch_size_with() {
    let cfg = WeSpeakerConfig::new().with_max_batch_size(16);
    assert_eq!(cfg.max_batch_size, 16);
}

#[test]
fn shapes_match_pyannote_reference() {
    // Spot-check that the precomputed Linear input dim matches pyannote:
    // stats_dim = m_channels * 8 * (num_mel_bins / 8) = 32 * 8 * 10 = 2560
    // seg_1 in_features = stats_dim * 2 = 5120
    let model = WeSpeakerResNet34::with_zero_weights(WeSpeakerConfig::new());
    let sd = model.state_dict("");

    let seg_w = sd.get("seg_1.weight").unwrap();
    let shape = seg_w.dims().unwrap();
    assert_eq!(shape, vec![256, 5120], "seg_1 weight must be (embed_dim=256, stats_dim*2=5120)");

    let seg_b = sd.get("seg_1.bias").unwrap();
    let bias_shape = seg_b.dims().unwrap();
    assert_eq!(bias_shape, vec![256]);
}
