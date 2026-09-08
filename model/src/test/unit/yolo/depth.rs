use svod_dtype::DType;
use svod_tensor::nn::Module;
use svod_tensor::{Tensor, Variable};

use crate::yolo::{Yolo26Depth, YoloConfig, YoloScale};

#[test]
fn state_dict_round_trip_depth() {
    let cfg = YoloConfig::new(YoloScale::Nano, 1);
    let model = Yolo26Depth::with_zero_weights(cfg.clone());
    let sd = model.state_dict("");
    for key in
        ["0.conv.weight", "23.proj.0.conv.weight", "23.head.0.conv.weight", "23.head.1.weight", "23.head.3.weight"]
    {
        assert!(sd.contains_key(key), "missing key: {key}");
    }
    for key in ["23.cal_a", "23.cal_b"] {
        assert!(sd.contains_key(key), "missing key: {key}");
    }

    let mut empty = Yolo26Depth::with_zero_weights(cfg);
    empty.load_state_dict(&sd, "").expect("load round-trip");
}

/// An uncalibrated checkpoint carries no `cal_a` / `cal_b`; the head must load
/// anyway and clear the fields instead of silently substituting a default.
#[test]
fn calibration_is_optional() {
    let cfg = YoloConfig::new(YoloScale::Nano, 1);
    let model = Yolo26Depth::with_zero_weights(cfg.clone());

    let mut sd = model.state_dict("");
    sd.remove("23.cal_a");
    sd.remove("23.cal_b");

    let mut loaded = Yolo26Depth::with_zero_weights(cfg);
    loaded.load_state_dict(&sd, "").expect("load without calibration");
    assert!(loaded.head.cal_a.is_none());
    assert!(loaded.head.cal_b.is_none());
    assert!(!loaded.state_dict("").contains_key("23.cal_a"));
}

#[test]
fn forward_shape_depth() {
    let cfg = YoloConfig::new(YoloScale::Nano, 1);
    let model = Yolo26Depth::with_zero_weights(cfg);
    let images = Tensor::zeros(&[1, 3, 320, 320], DType::Float32);
    let var = Variable::new("b", 1, 1);
    let b = var.bind(1).unwrap();
    let out = model.forward(&images, &b).unwrap();
    let shape = crate::test::max_dims(&out);
    assert_eq!(shape, vec![1, 1, 80, 80]);
}
