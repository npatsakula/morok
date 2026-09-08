use svod_dtype::DType;
use svod_tensor::{Tensor, Variable};

use crate::state::HasStateDict;
use crate::yolo::{Yolo26Classify, YoloConfig, YoloScale};

#[test]
fn state_dict_round_trip_cls() {
    let cfg = YoloConfig::new(YoloScale::Nano, 1000);
    let model = Yolo26Classify::with_zero_weights(cfg.clone());

    let sd = model.state_dict("");

    for key in ["0.conv.weight", "9.cv1.conv.weight", "10.conv.conv.weight", "10.linear.weight", "10.linear.bias"] {
        assert!(sd.contains_key(key), "missing key: {key}");
    }

    let mut empty = Yolo26Classify::with_zero_weights(cfg);
    empty.load_state_dict(&sd, "").expect("load round-trip");
}

#[test]
fn forward_shape_cls() {
    let cfg = YoloConfig::new(YoloScale::Nano, 1000);
    let model = Yolo26Classify::with_zero_weights(cfg);

    let images = Tensor::zeros(&[1, 3, 224, 224], DType::Float32);
    let var = Variable::new("b", 1, 1);
    let b = var.bind(1).unwrap();

    let out = model.forward(&images, &b).unwrap();
    let shape = crate::test::max_dims(&out);

    assert_eq!(shape, vec![1, 1000]);
}
