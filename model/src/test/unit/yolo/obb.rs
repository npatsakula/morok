use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::Module;

use crate::yolo::{Yolo26Obb, YoloConfig, YoloScale};

#[test]
fn state_dict_round_trip_obb() {
    let cfg = YoloConfig::new(YoloScale::Nano, 80);
    let model = Yolo26Obb::with_zero_weights(cfg.clone());
    let sd = model.state_dict("");
    for key in [
        "0.conv.weight",
        "23.one2one_cv2.0.0.conv.weight",
        "23.one2one_cv4.0.0.conv.weight",
        "23.one2one_cv4.0.2.weight",
        "23.one2one_cv4.0.2.bias",
    ] {
        assert!(sd.contains_key(key), "missing key: {key}");
    }
    let mut empty = Yolo26Obb::with_zero_weights(cfg);
    empty.load_state_dict(&sd, "").expect("load round-trip");
}

#[test]
fn forward_shape_obb() {
    let cfg = YoloConfig::new(YoloScale::Nano, 80);
    let model = Yolo26Obb::with_zero_weights(cfg);
    let images = Tensor::zeros(&[1, 3, 320, 320], DType::Float32);
    let out = model.forward(&images).unwrap();
    let shape = crate::test::max_dims(&out);
    // 4 + 80 + 1 = 85 channels, 2100 anchors
    assert_eq!(shape, vec![1, 85, 2100]);
}
