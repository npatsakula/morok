use svod_tensor::Tensor;

use crate::blocks::remap::strip_metadata;
use crate::state::StateDict;

/// The only entry no layer reads is dropped; everything a `BatchNorm2d` loads
/// survives untouched.
#[test]
fn strip_metadata_drops_only_num_batches_tracked() {
    let keep = ["bn1.weight", "bn1.bias", "bn1.running_mean", "bn1.running_var"];
    let mut sd: StateDict = keep.iter().map(|k| ((*k).to_string(), Tensor::from_slice([1.0f32]))).collect();
    sd.insert("bn1.num_batches_tracked".into(), Tensor::from_slice([42.0f32]));

    let stripped = strip_metadata(sd);

    assert!(!stripped.contains_key("bn1.num_batches_tracked"));
    for key in keep {
        assert!(stripped.contains_key(key), "missing key: {key}");
    }
}
