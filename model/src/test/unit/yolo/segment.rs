use svod_dtype::DType;
use svod_tensor::{Tensor, Variable};

use crate::state::HasStateDict;
use crate::yolo::{Yolo26Segment, YoloConfig, YoloScale};

#[test]
fn state_dict_round_trip_segment() {
    let cfg = YoloConfig::new(YoloScale::Nano, 80);
    let model = Yolo26Segment::with_zero_weights(cfg.clone());
    let sd = model.state_dict("");
    for key in [
        "0.conv.weight",
        "23.one2one_cv4.0.0.conv.weight",
        "23.proto.cv1.conv.weight",
        "23.proto.upsample.weight",
        "23.proto.cv3.conv.weight",
    ] {
        assert!(sd.contains_key(key), "missing key: {key}");
    }
    let mut empty = Yolo26Segment::with_zero_weights(cfg);
    empty.load_state_dict(&sd, "").expect("load round-trip");
}

#[test]
fn forward_shape_segment() {
    let cfg = YoloConfig::new(YoloScale::Nano, 80);
    let model = Yolo26Segment::with_zero_weights(cfg);
    let images = Tensor::zeros(&[1, 3, 320, 320], DType::Float32);
    let var = Variable::new("b", 1, 1);
    let b = var.bind(1).unwrap();
    let (preds, protos) = model.forward(&images, &b).unwrap();
    let ps = crate::test::max_dims(&preds);
    // 4 + 80 + 32 = 116 channels, 2100 anchors
    assert_eq!(ps, vec![1, 116, 2100]);
    let prs = crate::test::max_dims(&protos);
    // nm=32 protos at H/4=80, W/4=80
    assert_eq!(prs, vec![1, 32, 80, 80]);
}
