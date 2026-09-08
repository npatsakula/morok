//! Key-set equivalence for the derived [`Module`] impls: the emitted keys must
//! be exactly the upstream WavLM checkpoint layout, and a prefix must shift
//! every key by exactly one segment (no leading dot at the root).

use std::collections::BTreeSet;

use svod_tensor::nn::Module;
use test_case::test_case;

use crate::wavlm::{ExtractorMode, WavLm, WavLmConfig, wavlm_base, wavlm_large, wavlm_large_s80_md};

fn affine(prefix: &str) -> [String; 2] {
    [format!("{prefix}.weight"), format!("{prefix}.bias")]
}

/// The upstream layout, spelled out from the Python module tree rather than
/// read back off the Rust structs.
pub(crate) fn expected_keys(cfg: &WavLmConfig) -> BTreeSet<String> {
    let mut keys: Vec<String> = Vec::new();

    for i in 0..cfg.extractor_conv_layer_config.len() {
        let block = format!("feature_extractor.conv_layers.{i}");
        keys.push(format!("{block}.conv.weight"));
        if cfg.extractor_conv_bias {
            keys.push(format!("{block}.conv.bias"));
        }
        // GroupNorm mode normalizes block 0 only; LayerNorm mode every block.
        if cfg.extractor_mode == ExtractorMode::LayerNorm || i == 0 {
            keys.extend(affine(&format!("{block}.layer_norm")));
        }
    }

    keys.extend(affine("encoder.feature_projection.layer_norm"));
    keys.extend(affine("encoder.feature_projection.projection"));
    keys.extend(affine("encoder.transformer.pos_conv_embed.conv"));
    keys.extend(affine("encoder.transformer.layer_norm"));
    // The shared bucket table is stored under layer 0's attention.
    keys.push("encoder.transformer.layers.0.attention.rel_attn_embed.weight".to_string());

    for i in 0..cfg.encoder_num_layers {
        let layer = format!("encoder.transformer.layers.{i}");
        keys.extend(affine(&format!("{layer}.layer_norm")));
        keys.extend(affine(&format!("{layer}.final_layer_norm")));
        if cfg.encoder_use_attention[i] && !cfg.encoder_remaining_heads[i].is_empty() {
            for proj in ["q_proj", "k_proj", "v_proj", "out_proj", "gru_rel_pos_linear"] {
                keys.extend(affine(&format!("{layer}.attention.{proj}")));
            }
            keys.push(format!("{layer}.attention.gru_rel_pos_const"));
        }
        if cfg.encoder_use_feed_forward[i] {
            keys.extend(affine(&format!("{layer}.feed_forward.intermediate_dense")));
            keys.extend(affine(&format!("{layer}.feed_forward.output_dense")));
        }
    }

    keys.into_iter().collect()
}

/// Assert `m.state_dict("")` is `expected` and that a prefix only shifts it.
pub(crate) fn assert_layout<M: Module>(model: &M, expected: &BTreeSet<String>) {
    let keys: BTreeSet<String> = model.state_dict("").into_keys().collect();
    assert_eq!(&keys, expected);

    let prefixed: BTreeSet<String> = model.state_dict("m").into_keys().collect();
    let shifted: BTreeSet<String> = keys.iter().map(|k| format!("m.{k}")).collect();
    assert_eq!(prefixed, shifted);
}

#[test_case(wavlm_base(); "base")]
#[test_case(wavlm_large(); "large")]
#[test_case(wavlm_large_s80_md(); "large_s80_md")]
fn wavlm_keys_match_upstream_layout(cfg: WavLmConfig) {
    assert_layout(&WavLm::empty(cfg.clone()), &expected_keys(&cfg));
}
