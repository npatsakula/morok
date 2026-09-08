use svod_dtype::DType;
use svod_tensor::{Tensor, Variable};

use crate::state::HasStateDict;
use crate::yolo::{Yolo26Pose, YoloConfig, YoloScale};

#[test]
fn state_dict_round_trip_pose() {
    let cfg = YoloConfig::new(YoloScale::Nano, 1);
    let model = Yolo26Pose::with_zero_weights(cfg.clone());
    let sd = model.state_dict("");
    for key in [
        "0.conv.weight",
        "23.one2one_cv4.0.0.conv.weight",
        "23.one2one_cv4_kpts.0.weight",
        "23.one2one_cv4_kpts.0.bias",
    ] {
        assert!(sd.contains_key(key), "missing key: {key}");
    }
    let mut empty = Yolo26Pose::with_zero_weights(cfg);
    empty.load_state_dict(&sd, "").expect("load round-trip");
}

#[test]
fn forward_shape_pose() {
    let cfg = YoloConfig::new(YoloScale::Nano, 1);
    let model = Yolo26Pose::with_zero_weights(cfg);
    let images = Tensor::zeros(&[1, 3, 320, 320], DType::Float32);
    let var = Variable::new("b", 1, 1);
    let b = var.bind(1).unwrap();
    let out = model.forward(&images, &b).unwrap();
    let shape = crate::test::max_dims(&out);
    // 4 + 1 + 51 = 56 (boxes + cls + 17*3 keypoints), 2100 anchors
    assert_eq!(shape, vec![1, 56, 2100]);
}
