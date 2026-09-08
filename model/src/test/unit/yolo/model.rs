use svod_dtype::DType;
use svod_tensor::{Tensor, Variable};

use crate::state::HasStateDict;
use crate::yolo::{Yolo26Detect, YoloConfig, YoloScale};

/// State-dict round-trip: build a yolo26n, emit its state dict, verify
/// representative keys exist (matching Ultralytics layout), then reload
/// into a fresh instance.
#[test]
fn state_dict_round_trip_nano() {
    let cfg = YoloConfig::new(YoloScale::Nano, 80);
    let model = Yolo26Detect::with_zero_weights(cfg.clone());

    let sd = model.state_dict("");

    for key in [
        "0.conv.weight",
        "0.bn.weight",
        "2.cv1.conv.weight",
        "2.cv2.conv.weight",
        "2.m.0.cv1.conv.weight",
        "2.m.0.cv2.conv.weight",
        "9.cv1.conv.weight",
        "9.cv2.conv.weight",
        "10.cv1.conv.weight",
        "10.cv2.conv.weight",
        "10.m.0.attn.qkv.conv.weight",
        "10.m.0.attn.proj.conv.weight",
        "10.m.0.attn.pe.conv.weight",
        "10.m.0.ffn.0.conv.weight",
        "10.m.0.ffn.1.conv.weight",
        "13.cv1.conv.weight",
        "16.cv1.conv.weight",
        "19.cv1.conv.weight",
        "22.cv1.conv.weight",
        "23.one2one_cv2.0.0.conv.weight",
        "23.one2one_cv2.0.2.weight",
        "23.one2one_cv2.0.2.bias",
        "23.one2one_cv3.0.0.0.conv.weight",
        "23.one2one_cv3.0.2.weight",
        "23.one2one_cv3.0.2.bias",
        "22.m.0.1.attn.qkv.conv.weight",
    ] {
        assert!(sd.contains_key(key), "missing key: {key}");
    }

    let mut empty = Yolo26Detect::with_zero_weights(cfg);
    empty.load_state_dict(&sd, "").expect("load round-trip");
}

/// Build the symbolic forward graph and check the output shape.
/// Verifies symbolic batch `b` propagates through the full FPN+PAN DAG.
#[test]
fn forward_shape_nano() {
    let cfg = YoloConfig::new(YoloScale::Nano, 80);
    let model = Yolo26Detect::with_zero_weights(cfg);

    let images = Tensor::zeros(&[1, 3, 320, 320], DType::Float32);
    let var = Variable::new("b", 1, 1);
    let b = var.bind(1).unwrap();

    let out = model.forward(&images, &b).unwrap();
    let shape = crate::test::max_dims(&out);

    // 320×320: P3=40×40, P4=20×20, P5=10×10 → A=2100; out=[1, 84, 2100]
    assert_eq!(shape, vec![1, 84, 2100]);
}

/// Heavy: realize the full forward through zero weights.
#[test]
#[ignore = "heavy: full Yolo26n graph compile through the CPU backend"]
fn forward_realize_nano() {
    let cfg = YoloConfig::new(YoloScale::Nano, 80);
    let model = Yolo26Detect::with_zero_weights(cfg);

    let images = Tensor::zeros(&[1, 3, 320, 320], DType::Float32);
    let var = Variable::new("b", 1, 1);
    let b = var.bind(1).unwrap();

    let out = model.forward(&images, &b).unwrap();
    out.realize().unwrap();

    let shape = crate::test::max_dims(&out);
    assert_eq!(shape, vec![1, 84, 2100]);
}
