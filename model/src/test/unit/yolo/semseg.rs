use svod_dtype::DType;
use svod_tensor::nn::Module;
use svod_tensor::{Tensor, Variable};

use crate::yolo::{Yolo26SemSeg, YoloConfig, YoloScale};

#[test]
fn state_dict_round_trip_semseg() {
    let cfg = YoloConfig::new(YoloScale::Nano, 19);
    let model = Yolo26SemSeg::with_zero_weights(cfg.clone());

    let sd = model.state_dict("");

    for key in [
        "0.conv.weight",
        "10.cv1.conv.weight",
        "13.cv1.conv.weight",
        "16.cv1.conv.weight",
        "17.classifier.0.conv.weight",
        "17.classifier.2.weight",
        "17.classifier.2.bias",
    ] {
        assert!(sd.contains_key(key), "missing key: {key}");
    }

    let mut empty = Yolo26SemSeg::with_zero_weights(cfg);
    empty.load_state_dict(&sd, "").expect("load round-trip");
}

#[test]
fn forward_shape_semseg() {
    let cfg = YoloConfig::new(YoloScale::Nano, 19);
    let model = Yolo26SemSeg::with_zero_weights(cfg);

    let images = Tensor::zeros(&[1, 3, 320, 320], DType::Float32);
    let var = Variable::new("b", 1, 1);
    let b = var.bind(1).unwrap();

    let out = model.forward(&images, &b).unwrap();
    let shape = crate::test::max_dims(&out);

    // 320×320 → P3 at stride 8 → 40×40, nc=19
    assert_eq!(shape, vec![1, 19, 40, 40]);
}
