use svod_dtype::DType;
use svod_tensor::{Tensor, Variable};
use test_case::test_case;

use crate::resnet::{OutputMode, ResNet, ResNetConfig, ResNetDepth};
use crate::state::HasStateDict;

#[test]
fn feature_channels_matches_depth_expansion() {
    let r18 = ResNet::with_zero_weights(ResNetConfig::new(ResNetDepth::R18, OutputMode::Features));
    let r50 = ResNet::with_zero_weights(ResNetConfig::new(ResNetDepth::R50, OutputMode::Features));
    assert_eq!(r18.feature_channels(), 512);
    assert_eq!(r50.feature_channels(), 2048);
}

/// State-dict round-trip on a BasicBlock-stack R18 with a Classification head.
/// Catches: forgetting a `state_dict()` entry, renaming a sub-module on either
/// the emit or the load side, breaking the timm-key prefix convention.
#[test]
fn state_dict_round_trip_r18_classification() {
    let cfg = ResNetConfig::new(ResNetDepth::R18, OutputMode::Classification { num_classes: 10 });
    let model = ResNet::with_zero_weights(cfg.clone());

    let sd = model.state_dict("");

    // Representative keys: stem, every stage's first block, downsample on
    // stages 2/3/4 (R18 stage 1 has no downsample), classification head.
    for key in [
        "conv1.weight",
        "bn1.weight",
        "bn1.bias",
        "bn1.running_mean",
        "bn1.invstd",
        "layer1.0.conv1.weight",
        "layer1.0.bn1.weight",
        "layer1.1.conv2.weight",
        "layer2.0.conv1.weight",
        "layer2.0.downsample.0.weight",
        "layer2.0.downsample.1.invstd",
        "layer3.0.downsample.0.weight",
        "layer4.0.downsample.0.weight",
        "layer4.1.bn2.running_mean",
        "fc.weight",
        "fc.bias",
    ] {
        assert!(sd.contains_key(key), "missing key: {key}");
    }

    // Bottleneck-only keys must NOT appear for a BasicBlock-stack model.
    assert!(!sd.contains_key("layer1.0.conv3.weight"), "BasicBlock must not emit conv3");
    assert!(!sd.contains_key("layer1.0.bn3.weight"), "BasicBlock must not emit bn3");

    // Reload it back into a fresh model.
    let mut empty = ResNet::with_zero_weights(cfg);
    empty.load_state_dict(&sd, "").expect("load round-trip");
}

/// State-dict round-trip on a Bottleneck-stack R50 in Features mode. Asserts
/// the Bottleneck-specific `conv3`/`bn3` keys are present and the `fc.*` keys
/// are absent (no head in Features mode).
#[test]
fn state_dict_round_trip_r50_features() {
    let cfg = ResNetConfig::new(ResNetDepth::R50, OutputMode::Features);
    let model = ResNet::with_zero_weights(cfg.clone());

    let sd = model.state_dict("");

    for key in [
        "conv1.weight",
        "bn1.weight",
        "layer1.0.conv1.weight",
        "layer1.0.conv3.weight",
        "layer1.0.bn3.invstd",
        "layer1.0.downsample.0.weight", // R50 stage1 downsamples (channel expansion).
        "layer2.0.downsample.0.weight",
        "layer4.2.conv3.weight",
        "layer4.2.bn3.weight",
    ] {
        assert!(sd.contains_key(key), "missing key: {key}");
    }

    assert!(!sd.contains_key("fc.weight"), "Features mode must not emit fc.weight");
    assert!(!sd.contains_key("fc.bias"), "Features mode must not emit fc.bias");

    let mut empty = ResNet::with_zero_weights(cfg);
    empty.load_state_dict(&sd, "").expect("load round-trip");
}

/// Build the symbolic forward graph and assert the output shape, without ever
/// calling `.realize()`. Catches bad axes / strides / broadcasts anywhere in
/// stem → 4 stages → (head | identity) — milliseconds, not the 30 s a full
/// compile would take.
fn forward_zero_weights_shape_check(depth: ResNetDepth, output: OutputMode, expected: &[usize]) {
    let cfg = ResNetConfig::new(depth, output).with_max_batch_size(1);
    let model = ResNet::with_zero_weights(cfg);

    let images = Tensor::zeros(&[1, 3, 32, 32], DType::Float32).unwrap();
    let var = Variable::new("b", 1, 1);
    let b = var.bind(1).unwrap();

    let out = model.forward(&images, &b).unwrap();
    let shape = crate::test::max_dims(&out);
    assert_eq!(shape, expected);
}

#[test]
fn forward_shape_r18_classification() {
    forward_zero_weights_shape_check(ResNetDepth::R18, OutputMode::Classification { num_classes: 10 }, &[1, 10]);
}

#[test]
fn forward_shape_r18_features() {
    // 32×32 → stem(/2) → 16×16 → maxpool(/2) → 8×8 → stage1(s1) → 8×8 →
    // stage2(s2) → 4×4 → stage3(s2) → 2×2 → stage4(s2) → 1×1.
    forward_zero_weights_shape_check(ResNetDepth::R18, OutputMode::Features, &[1, 512, 1, 1]);
}

#[test]
fn forward_shape_r50_classification() {
    forward_zero_weights_shape_check(ResNetDepth::R50, OutputMode::Classification { num_classes: 10 }, &[1, 10]);
}

#[test]
fn forward_shape_r50_features() {
    forward_zero_weights_shape_check(ResNetDepth::R50, OutputMode::Features, &[1, 2048, 1, 1]);
}

/// `fc.weight` is `[num_classes, 512 * expansion]`. Catches off-by-one in the
/// stage4→head arithmetic for every depth.
#[test_case(ResNetDepth::R18,  512;  "r18")]
#[test_case(ResNetDepth::R34,  512;  "r34")]
#[test_case(ResNetDepth::R50,  2048; "r50")]
#[test_case(ResNetDepth::R101, 2048; "r101")]
#[test_case(ResNetDepth::R152, 2048; "r152")]
fn head_dimensions_per_depth(depth: ResNetDepth, expected_in: usize) {
    let cfg = ResNetConfig::new(depth, OutputMode::Classification { num_classes: 1000 });
    let model = ResNet::with_zero_weights(cfg);

    let sd = model.state_dict("");
    let fc_w = sd.get("fc.weight").expect("fc.weight present in classification mode");
    let shape = fc_w.dims().unwrap();
    assert_eq!(shape, vec![1000, expected_in]);

    let fc_b = sd.get("fc.bias").unwrap();
    let bias_shape = fc_b.dims().unwrap();
    assert_eq!(bias_shape, vec![1000]);
}

/// Realize-based smoke test gated behind `--ignored`. Build a small ResNet-18,
/// forward zeros through it, assert the output spatial map is `[1, 512, 1, 1]`.
/// Exercise this when actively touching layer code; the default suite skips it
/// because the tensor-level `codegen_tests!` conv2d / linear / batchnorm tests
/// cover the same compile path more directly.
#[test]
#[ignore = "heavy: full ResNet-18 graph compile through the CPU backend"]
fn features_r18_returns_512_channel_map() {
    use svod_dtype::DType;
    use svod_tensor::{Tensor, Variable};

    let max_batch = 1;
    let config = ResNetConfig::new(ResNetDepth::R18, OutputMode::Features).with_max_batch_size(max_batch);
    let model = ResNet::with_zero_weights(config);

    let images = Tensor::zeros(&[max_batch, 3, 32, 32], DType::Float32).unwrap();
    let var = Variable::new("b", 1, max_batch as i64);
    let b1 = var.bind(1).unwrap();

    let mut out = model.forward(&images, &b1).unwrap();
    out.realize().unwrap();

    let shape = crate::test::max_dims(&out);
    assert_eq!(shape, vec![1, 512, 1, 1]);
}

#[test]
#[ignore = "heavy: full ResNet-50 graph compile through the CPU backend"]
fn features_r50_returns_2048_channel_map() {
    use svod_dtype::DType;
    use svod_tensor::{Tensor, Variable};

    let max_batch = 1;
    let config = ResNetConfig::new(ResNetDepth::R50, OutputMode::Features).with_max_batch_size(max_batch);
    let model = ResNet::with_zero_weights(config);

    let images = Tensor::zeros(&[max_batch, 3, 32, 32], DType::Float32).unwrap();
    let var = Variable::new("b", 1, max_batch as i64);
    let b1 = var.bind(1).unwrap();

    let mut out = model.forward(&images, &b1).unwrap();
    out.realize().unwrap();

    let shape = crate::test::max_dims(&out);
    assert_eq!(shape, vec![1, 2048, 1, 1]);
}
