//! Key-set equivalence for the DiariZen segmentation model: the WavLM backbone
//! nested under `wavlm_model.` plus the head, and nothing else.

use std::collections::BTreeSet;

use crate::diarizen::{DiariZenConfig, DiariZenSegmentationModel};
use crate::test::unit::wavlm::state_dict::{assert_layout, expected_keys as wavlm_keys};

use super::model::tiny_cfg;

fn expected_keys(cfg: &DiariZenConfig) -> BTreeSet<String> {
    let mut keys: Vec<String> = wavlm_keys(&cfg.wavlm).iter().map(|k| format!("wavlm_model.{k}")).collect();
    keys.push("weight_sum.weight".to_string());
    for module in ["proj", "lnorm", "classifier"] {
        keys.push(format!("{module}.weight"));
        keys.push(format!("{module}.bias"));
    }
    for i in 0..cfg.num_layer {
        let block = format!("conformer.conformer_layer.{i}");
        for ffn in ["ffn1", "ffn2"] {
            for part in ["ln_norm", "w_1", "w_2"] {
                keys.push(format!("{block}.{ffn}.{part}.weight"));
                keys.push(format!("{block}.{ffn}.{part}.bias"));
            }
        }
        for part in ["ln_norm", "mha.linearQ", "mha.linearK", "mha.linearV", "mha.linearO"] {
            keys.push(format!("{block}.mha.{part}.weight"));
            keys.push(format!("{block}.mha.{part}.bias"));
        }
        for part in ["ln_norm", "pointwise_conv1", "depthwise_conv", "pointwise_conv2"] {
            keys.push(format!("{block}.conv.{part}.weight"));
            keys.push(format!("{block}.conv.{part}.bias"));
        }
        for part in ["weight", "bias", "running_mean", "running_var"] {
            keys.push(format!("{block}.conv.bn_norm.{part}"));
        }
        keys.push(format!("{block}.ln_norm.weight"));
        keys.push(format!("{block}.ln_norm.bias"));
    }
    keys.into_iter().collect()
}

#[test]
fn segmentation_keys_match_upstream_layout() {
    let cfg = tiny_cfg();
    assert_layout(&DiariZenSegmentationModel::empty(cfg.clone()), &expected_keys(&cfg));
}
