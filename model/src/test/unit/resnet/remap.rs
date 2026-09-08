use svod_tensor::Tensor;

use crate::blocks::remap::{fold_batchnorm, strip_metadata};
use crate::state::StateDict;

fn bn_dict(prefixes: &[&str], var: f32) -> StateDict {
    let mut sd = StateDict::new();
    for prefix in prefixes {
        sd.insert(format!("{prefix}.weight"), Tensor::from_slice([1.0f32]));
        sd.insert(format!("{prefix}.bias"), Tensor::from_slice([0.0f32]));
        sd.insert(format!("{prefix}.running_mean"), Tensor::from_slice([0.0f32]));
        sd.insert(format!("{prefix}.running_var"), Tensor::from_slice([var]));
        sd.insert(format!("{prefix}.num_batches_tracked"), Tensor::from_slice([42.0f32]));
    }
    sd
}

/// The only entry no layer reads is dropped; everything a `BatchNorm2d` loads
/// survives untouched.
#[test]
fn strip_metadata_drops_only_num_batches_tracked() {
    let stripped = strip_metadata(bn_dict(&["bn1"], 0.25));

    assert!(!stripped.contains_key("bn1.num_batches_tracked"));
    for key in ["bn1.weight", "bn1.bias", "bn1.running_mean", "bn1.running_var"] {
        assert!(stripped.contains_key(key), "missing key: {key}");
    }
}

/// The `BatchNormWeights` compatibility shim adds `invstd` beside the variance
/// it was computed from, so one dict feeds both layer types.
#[test]
fn fold_adds_invstd_and_keeps_running_var() {
    let folded = fold_batchnorm(bn_dict(&["bn1", "layer1.0.bn1", "layer4.2.downsample.1"], 0.25)).expect("fold");

    for prefix in ["bn1", "layer1.0.bn1", "layer4.2.downsample.1"] {
        assert!(!folded.contains_key(&format!("{prefix}.num_batches_tracked")));
        assert!(folded.contains_key(&format!("{prefix}.running_var")), "running_var must survive the fold");

        let invstd = folded[&format!("{prefix}.invstd")].to_vec::<f32>().expect("read invstd");
        let expected = 1.0 / (0.25_f32 + 1e-5).sqrt();
        assert!((invstd[0] - expected).abs() < 1e-6, "got {invstd:?}, expected {expected}");
    }
}
