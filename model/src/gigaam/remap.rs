use morok_tensor::Tensor;

use crate::state::StateDict;

use super::error::{Error, Result};
use super::GigaAmConfig;

use super::ConvNormType;

pub fn remap_pytorch(sd: StateDict, config: &GigaAmConfig) -> Result<StateDict> {
    let mut out = StateDict::new();
    let mut bn_var_keys: Vec<(String, Tensor)> = Vec::new();

    for (key, tensor) in sd {
        let Some(mapped) = remap_key(&key, config) else {
            continue;
        };
        if mapped.starts_with("__bn_var__.") {
            let layer_idx = mapped.strip_prefix("__bn_var__.").unwrap().to_string();
            bn_var_keys.push((layer_idx, tensor));
            continue;
        }
        out.insert(mapped, tensor);
    }

    if matches!(config.conv_norm_type, ConvNormType::BatchNorm) {
        for (layer_idx, var_tensor) in bn_var_keys {
            let data = var_tensor.as_vec::<f32>().map_err(|e| Error::Tensor { source: Box::new(e) })?;
            let invstd: Vec<f32> = data.iter().map(|&v| 1.0 / (v + 1e-5).sqrt()).collect();
            let invstd_tensor = Tensor::from_slice(&invstd);
            out.insert(format!("layers.{layer_idx}.conv.bn_invstd"), invstd_tensor);
        }
    }

    Ok(out)
}

fn remap_key(key: &str, config: &GigaAmConfig) -> Option<String> {
    let key = key.strip_prefix("model.").unwrap_or(key);
    let parts: Vec<&str> = key.split('.').collect();

    if parts[..3] == ["encoder", "pre_encode", "conv"] && parts.len() == 5 {
        let idx = parts[3];
        let param = parts[4];
        let conv_map = match idx {
            "0" => "conv1",
            "2" => "conv2",
            _ => return None,
        };
        return Some(format!("subsampling.{conv_map}_{param}"));
    }

    if parts[..3] == ["encoder", "pre_encode", "out"] && parts.len() == 4 {
        return Some(format!("subsampling.linear_{}", parts[3]));
    }

    if parts[..2] == ["encoder", "layers"] && parts.len() >= 4 {
        let i = parts[2];
        let rest = &parts[3..];

        if rest[0] == "norm_feed_forward1" {
            return Some(format!("layers.{i}.ffn1.norm.{}", &rest[1..].join(".")));
        }
        if rest[0] == "feed_forward1" && rest.len() == 3 {
            let sub = rest[1];
            let param = rest[2];
            return Some(format!("layers.{i}.ffn1.{sub}.{param}"));
        }

        if rest[0] == "norm_self_att" {
            return Some(format!("layers.{i}.mhsa.norm.{}", &rest[1..].join(".")));
        }
        if rest[0] == "self_attn" && rest.len() == 3 {
            let linear = rest[1];
            let param = rest[2];
            let base = match linear {
                "linear_q" => "q",
                "linear_k" => "k",
                "linear_v" => "v",
                "linear_out" => "out",
                "linear_pos" => return None,
                _ => return None,
            };
            let suffix = match param {
                "weight" => "proj",
                "bias" => "bias",
                _ => return None,
            };
            return Some(format!("layers.{i}.mhsa.{base}_{suffix}"));
        }

        if rest[0] == "norm_conv" {
            return Some(format!("layers.{i}.conv.norm.{}", &rest[1..].join(".")));
        }
        if rest[0] == "conv" && rest.len() == 3 {
            let sub = rest[1];
            let param = rest[2];
            match sub {
                "pointwise_conv1" => return Some(format!("layers.{i}.conv.pw1_{param}")),
                "depthwise_conv" => return Some(format!("layers.{i}.conv.dw_{param}")),
                "pointwise_conv2" => return Some(format!("layers.{i}.conv.pw2_{param}")),
                "batch_norm" => return remap_bn_key(i, param, config),
                _ => return None,
            }
        }

        if rest[0] == "norm_feed_forward2" {
            return Some(format!("layers.{i}.ffn2.norm.{}", &rest[1..].join(".")));
        }
        if rest[0] == "feed_forward2" && rest.len() == 3 {
            let sub = rest[1];
            let param = rest[2];
            return Some(format!("layers.{i}.ffn2.{sub}.{param}"));
        }

        if rest[0] == "norm_out" {
            return Some(format!("layers.{i}.final_norm.{}", &rest[1..].join(".")));
        }

        return None;
    }

    if parts[..2] == ["head", "decoder_layers"] && parts.len() >= 4 {
        return Some(format!("head.{}", &parts[3..].join(".")));
    }

    None
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
            "running_var" => Some(format!("__bn_var__.{layer}")),
            "num_batches_tracked" => None,
            _ => None,
        },
    }
}
