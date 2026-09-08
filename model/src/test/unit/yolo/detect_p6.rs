use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::Module;

use crate::yolo::{Yolo26DetectP6, YoloConfig, YoloScale};

#[test]
fn state_dict_round_trip_p6() {
    let cfg = YoloConfig::new(YoloScale::Nano, 80);
    let model = Yolo26DetectP6::with_zero_weights(cfg.clone());
    let sd = model.state_dict("");
    for key in [
        "0.conv.weight",
        "9.conv.weight",
        "12.cv1.conv.weight",
        "15.cv1.conv.weight",
        "31.one2one_cv2.0.0.conv.weight",
        "31.one2one_cv3.0.2.weight",
    ] {
        assert!(sd.contains_key(key), "missing key: {key}");
    }
    let mut empty = Yolo26DetectP6::with_zero_weights(cfg);
    empty.load_state_dict(&sd, "").expect("load round-trip");
}

#[test]
fn forward_shape_p6() {
    let cfg = YoloConfig::new(YoloScale::Nano, 80);
    let model = Yolo26DetectP6::with_zero_weights(cfg);
    let images = Tensor::zeros(&[1, 3, 320, 320], DType::Float32);
    let out = model.forward(&images).unwrap();
    let shape = crate::test::max_dims(&out);
    // 4 + 80 = 84 channels
    // P3: 40×40=1600, P4: 20×20=400, P5: 10×10=100, P6: 5×5=25 → 2125 anchors
    assert_eq!(shape, vec![1, 84, 2125]);
}
