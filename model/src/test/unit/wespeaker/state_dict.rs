//! Key-set equivalence for the WeSpeaker ResNet34: torchvision-style block
//! naming with pyannote's `seg_1` head, all under PyTorch's own parameter names.

use std::collections::BTreeSet;

use crate::test::unit::wavlm::state_dict::assert_layout;
use crate::wespeaker::{M_CHANNELS, NUM_BLOCKS, WeSpeakerConfig, WeSpeakerResNet34};

fn batchnorm(prefix: &str) -> [String; 4] {
    ["weight", "bias", "running_mean", "running_var"].map(|p| format!("{prefix}.{p}"))
}

fn expected_keys() -> BTreeSet<String> {
    let mut keys = vec!["conv1.weight".to_string(), "seg_1.weight".to_string(), "seg_1.bias".to_string()];
    keys.extend(batchnorm("bn1"));

    let mut in_planes = M_CHANNELS;
    for (stage, &blocks) in NUM_BLOCKS.iter().enumerate() {
        let planes = M_CHANNELS << stage;
        for block in 0..blocks {
            let p = format!("layer{}.{block}", stage + 1);
            keys.push(format!("{p}.conv1.weight"));
            keys.push(format!("{p}.conv2.weight"));
            keys.extend(batchnorm(&format!("{p}.bn1")));
            keys.extend(batchnorm(&format!("{p}.bn2")));
            // Only the first block of a width- or stride-changing stage
            // projects the shortcut.
            if block == 0 && in_planes != planes {
                keys.push(format!("{p}.downsample.0.weight"));
                keys.extend(batchnorm(&format!("{p}.downsample.1")));
            }
        }
        in_planes = planes;
    }
    keys.into_iter().collect()
}

#[test]
fn resnet34_keys_match_pyannote_layout() {
    let model = WeSpeakerResNet34::with_zero_weights(WeSpeakerConfig::new());
    assert_layout(&model, &expected_keys());
}
