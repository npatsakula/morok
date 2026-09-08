use svod_dtype::DType;
use svod_tensor::{Tensor, Variable};

use crate::state::HasStateDict;
use crate::yolo::{Yolo26DetectP2, YoloConfig, YoloScale};

#[test]
fn state_dict_round_trip_p2() {
    let cfg = YoloConfig::new(YoloScale::Nano, 80);
    let model = Yolo26DetectP2::with_zero_weights(cfg.clone());
    let sd = model.state_dict("");
    for key in [
        "0.conv.weight",
        "13.cv1.conv.weight",
        "19.cv1.conv.weight",
        "29.one2one_cv2.0.0.conv.weight",
        "29.one2one_cv3.0.2.weight",
    ] {
        assert!(sd.contains_key(key), "missing key: {key}");
    }
    let mut empty = Yolo26DetectP2::with_zero_weights(cfg);
    empty.load_state_dict(&sd, "").expect("load round-trip");
}

#[test]
fn forward_shape_p2() {
    let cfg = YoloConfig::new(YoloScale::Nano, 80);
    let model = Yolo26DetectP2::with_zero_weights(cfg);
    let images = Tensor::zeros(&[1, 3, 320, 320], DType::Float32).unwrap();
    let var = Variable::new("b", 1, 1);
    let b = var.bind(1).unwrap();
    let out = model.forward(&images, &b).unwrap();
    let shape = crate::test::max_dims(&out);
    // 4 + 80 = 84 channels
    // P2: 80×80=6400, P3: 40×40=1600, P4: 20×20=400, P5: 10×10=100 → 8500 anchors
    assert_eq!(shape, vec![1, 84, 8500]);
}
