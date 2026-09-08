use crate::state::StateDict;

use super::GigaAmConfig;
use super::error::Result;

use super::ConvNormType;

/// Rename a PyTorch GigaAM checkpoint's keys to svod's layout. Pure renaming:
/// the BatchNorm affine is folded into the depthwise conv in the graph, by
/// `ConvModule`'s loader, so `running_var` passes through untouched.
pub fn remap_pytorch(sd: StateDict, config: &GigaAmConfig) -> Result<StateDict> {
    Ok(sd.into_iter().filter_map(|(key, tensor)| Some((remap_key(&key, config)?, tensor))).collect())
}

fn remap_key(key: &str, config: &GigaAmConfig) -> Option<String> {
    let key = key.strip_prefix("model.").unwrap_or(key);
    let parts: Vec<&str> = key.split('.').collect();

    if parts.len() == 5 && parts[..3] == ["encoder", "pre_encode", "conv"] {
        let idx = parts[3];
        let param = parts[4];
        let conv_map = match idx {
            "0" => "conv1",
            "2" => "conv2",
            _ => return None,
        };
        return Some(format!("subsampling.{conv_map}_{param}"));
    }

    if parts.len() == 4 && parts[..3] == ["encoder", "pre_encode", "out"] {
        return Some(format!("subsampling.linear_{}", parts[3]));
    }

    if parts.len() >= 4 && parts[..2] == ["encoder", "layers"] {
        let i = parts[2];
        let rest = &parts[3..];
        return remap_encoder_layer(i, rest, config);
    }

    if parts.len() >= 4 && parts[..2] == ["head", "decoder_layers"] {
        return Some(format!("head.{}", parts[3..].join(".")));
    }

    // RNN-T predictor: head.decoder.embed.weight, head.decoder.lstm.{w,b}_{ih,hh}_l{N}.
    if parts.len() >= 3 && parts[..2] == ["head", "decoder"] {
        if parts[2] == "embed" && parts.len() == 4 && parts[3] == "weight" {
            return Some("head.predictor.embed".to_string());
        }
        if parts[2] == "lstm" && parts.len() == 4 {
            return remap_rnnt_lstm_param(parts[3]);
        }
        return None;
    }

    // RNN-T joint: head.joint.{enc,pred}.{weight,bias},
    // head.joint.joint_net.1.{weight,bias} (joint_net.0 = ReLU, no params).
    if parts.len() >= 4 && parts[..2] == ["head", "joint"] {
        let sub = parts[2];
        let last = parts.last().unwrap();
        if (sub == "enc" || sub == "pred") && parts.len() == 4 {
            let suffix = match *last {
                "weight" => "w",
                "bias" => "b",
                _ => return None,
            };
            return Some(format!("head.joint.{sub}_{suffix}"));
        }
        if sub == "joint_net" && parts.len() == 5 && parts[3] == "1" {
            let suffix = match *last {
                "weight" => "out_w",
                "bias" => "out_b",
                _ => return None,
            };
            return Some(format!("head.joint.{suffix}"));
        }
        return None;
    }

    None
}

/// Encoder-layer key remapping. The 7 "passthrough" sub-modules (4 norms + 2
/// feed-forwards + final_norm) all share the same rename shape: replace the
/// `rest[0]` prefix with the svod target prefix and keep the remaining
/// path segments verbatim. `self_attn` and `conv` need bespoke logic.
fn remap_encoder_layer(i: &str, rest: &[&str], config: &GigaAmConfig) -> Option<String> {
    // (rest[0] prefix → svod target prefix)
    const PASSTHROUGH: &[(&str, &str)] = &[
        ("norm_feed_forward1", "ffn1.norm"),
        ("feed_forward1", "ffn1"),
        ("norm_self_att", "mhsa.norm"),
        ("norm_conv", "conv.norm"),
        ("norm_feed_forward2", "ffn2.norm"),
        ("feed_forward2", "ffn2"),
        ("norm_out", "final_norm"),
    ];
    if let Some(&(_, target)) = PASSTHROUGH.iter().find(|(src, _)| *src == rest[0]) {
        return Some(format!("layers.{i}.{target}.{}", rest[1..].join(".")));
    }

    if rest[0] == "self_attn" && rest.len() == 3 {
        let base = match rest[1] {
            "linear_q" => "q",
            "linear_k" => "k",
            "linear_v" => "v",
            "linear_out" => "out",
            _ => return None,
        };
        let suffix = match rest[2] {
            "weight" => "proj",
            "bias" => "bias",
            "weight_scale" => "weight_scale",
            _ => return None,
        };
        return Some(format!("layers.{i}.mhsa.{base}_{suffix}"));
    }

    if rest[0] == "conv" && rest.len() == 3 {
        let param = rest[2];
        return match rest[1] {
            "pointwise_conv1" => Some(format!("layers.{i}.conv.pw1_{param}")),
            "depthwise_conv" => Some(format!("layers.{i}.conv.dw_{param}")),
            "pointwise_conv2" => Some(format!("layers.{i}.conv.pw2_{param}")),
            "batch_norm" => remap_bn_key(i, param, config),
            _ => None,
        };
    }

    None
}

/// Decode a PyTorch LSTM parameter token like `weight_ih_l0` or `bias_hh_l2`
/// into svod's `head.predictor.lstm.{layer}.{w_ih,w_hh,b_ih,b_hh}` shape.
fn remap_rnnt_lstm_param(token: &str) -> Option<String> {
    // Strip `_l{N}` suffix.
    let (base, layer) = token.rsplit_once("_l")?;
    if !layer.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let mapped = match base {
        "weight_ih" => "w_ih",
        "weight_hh" => "w_hh",
        "bias_ih" => "b_ih",
        "bias_hh" => "b_hh",
        _ => return None,
    };
    Some(format!("head.predictor.lstm.{layer}.{mapped}"))
}

fn remap_bn_key(layer: &str, param: &str, config: &GigaAmConfig) -> Option<String> {
    match &config.conv_norm_type {
        ConvNormType::LayerNorm => match param {
            "weight" => Some(format!("layers.{layer}.conv.conv_norm.weight")),
            "bias" => Some(format!("layers.{layer}.conv.conv_norm.bias")),
            _ => None,
        },
        ConvNormType::BatchNorm => match param {
            "weight" => Some(format!("layers.{layer}.conv.bn_scale")),
            "bias" => Some(format!("layers.{layer}.conv.bn_bias")),
            "running_mean" => Some(format!("layers.{layer}.conv.bn_mean")),
            "running_var" => Some(format!("layers.{layer}.conv.bn_var")),
            "num_batches_tracked" => None,
            _ => None,
        },
    }
}
