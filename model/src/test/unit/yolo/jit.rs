use crate::jit::InputSpec;
use crate::yolo::{Yolo26Classify, Yolo26ClassifyJit, YoloConfig, YoloScale};

/// The YOLO wrappers all share one shape: `images` shrunk on dim 0 by
/// `batch_var b`, one `predictions` output whose live shape follows `b`.
/// Classify is the cheapest of them to compile, so it stands in for the family.
#[test]
#[ignore = "heavy: full Yolo26n-cls graph compile through the CPU backend"]
fn predictions_shape_tracks_the_bound_batch() {
    let max_batch = 2;
    let cfg = YoloConfig::new(YoloScale::Nano, 10).with_max_batch_size(max_batch);
    let mut jit = Yolo26ClassifyJit::new(Yolo26Classify::with_zero_weights(cfg));
    jit.prepare(InputSpec::f32(&[max_batch, 3, 64, 64])).unwrap();

    for b in [1, 2] {
        jit.execute_bound(b as i64).unwrap();
        assert_eq!(jit.predictions_shape().unwrap(), vec![b, 10], "predictions shape for b={b}");
        assert_eq!(jit.predictions_to_vec::<f32>().unwrap().len(), b * 10);
    }
}
