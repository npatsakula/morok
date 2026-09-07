//! Checkpoint loading: safetensors from HuggingFace Hub or local directory.
//!
//! Handles two checkpoint formats:
//! - **Original OpenAI** (`.pt` converted): `encoder.conv1.weight`, `decoder.blocks.0.attn.query.weight`
//! - **HF Transformers** (`model.safetensors`): `model.encoder.conv1.weight`, `model.decoder.layers.0.self_attn.q_proj.weight`

use std::path::Path;

use snafu::ResultExt;

use crate::state::{self, HasStateDict, StateDict};

use super::config::{ModelDimensions, WhisperSize};
use super::error::{Error, Result};
use super::model::Whisper;

impl Whisper {
    /// Load from a safetensors state dict.
    ///
    /// HuggingFace Transformers checkpoints commonly store weights in Float32.
    /// Linear and convolution weights use `dims.dtype`, while embeddings and
    /// LayerNorm affine parameters retain checkpoint precision to match
    /// OpenAI's mixed-precision forward pass. Pre-converted checkpoints should
    /// already use these storage dtypes so loading does not create lazy cast
    /// graphs.
    pub fn from_state_dict(sd: &StateDict, dims: ModelDimensions) -> Result<Self> {
        let dtype = dims.dtype.clone();
        let mut remapped = remap_hf_keys(sd);
        let scales: Vec<_> = remapped
            .keys()
            .filter_map(|key| key.strip_suffix(".weight_scale").map(|weight_key| (key.clone(), weight_key.to_string())))
            .collect();
        for (scale_key, weight_key) in scales {
            let scale = remapped.remove(&scale_key).expect("scale key came from the state dict");
            let weight = remapped
                .get(&weight_key)
                .ok_or_else(|| Error::State { source: state::Error::MissingKey { key: weight_key.clone() } })?;
            // The scale is per output channel: reshape to `[out, 1, ..]` so it
            // broadcasts along the weight's input (and kernel) axes.
            let shape = weight.shape().context(super::error::TensorSnafu)?;
            let mut scale_shape = vec![1isize; shape.len()];
            scale_shape[0] = shape[0].as_const().ok_or_else(|| Error::Checkpoint {
                msg: format!("quantized weight {weight_key} has a symbolic output dimension"),
            })? as isize;
            let scale = scale
                .cast(dtype.clone())
                .context(super::error::TensorSnafu)?
                .try_reshape(scale_shape)
                .context(super::error::TensorSnafu)?;
            let dequantized = weight
                .cast(dtype.clone())
                .context(super::error::TensorSnafu)?
                .try_mul(&scale)
                .context(super::error::TensorSnafu)?;
            remapped.insert(weight_key, dequantized);
        }
        let mut sd = state::cast_all(&remapped, dtype);
        for (key, tensor) in &remapped {
            if keeps_checkpoint_dtype(key) {
                sd.insert(key.clone(), tensor.clone());
            }
        }
        let mut model = Self::empty(dims);
        model.load_state_dict(&sd, "").map_err(|e| Error::State { source: e })?;
        Ok(model)
    }

    /// Load from a local safetensors file or directory.
    pub fn from_dir(dir: &Path, dims: ModelDimensions) -> Result<Self> {
        Self::from_dir_with_weights(dir, "model.safetensors", dims)
    }

    /// Load a named safetensors checkpoint from a local directory.
    pub fn from_dir_with_weights(dir: &Path, weights: &str, dims: ModelDimensions) -> Result<Self> {
        let sd = if weights == "model.safetensors" {
            state::load_safetensors_dir(dir)
        } else {
            state::load_safetensors(&dir.join(weights))
        }
        .map_err(|e| Error::State { source: e })?;
        Self::from_state_dict(&sd, dims)
    }

    /// Load from HuggingFace Hub (`openai/whisper-{name}` or custom repo).
    pub fn from_hub(model_id: &str, revision: &str, dims: ModelDimensions) -> Result<Self> {
        Self::from_hub_with_weights(model_id, revision, "model.safetensors", dims)
    }

    /// Load a named safetensors checkpoint from Hugging Face Hub.
    pub fn from_hub_with_weights(model_id: &str, revision: &str, weights: &str, dims: ModelDimensions) -> Result<Self> {
        let repo =
            crate::hub::HubRepo::open(model_id, revision).map_err(|e| Error::Checkpoint { msg: e.to_string() })?;
        let path = repo.get(weights).map_err(|e| Error::Checkpoint { msg: e.to_string() })?;
        let sd = state::load_safetensors(&path).map_err(|e| Error::State { source: e })?;
        Self::from_state_dict(&sd, dims)
    }

    /// Convenience: load a known size from `openai/whisper-{name}`.
    pub fn from_size(size: WhisperSize) -> Result<Self> {
        let dims = ModelDimensions::for_size(size);
        let repo = format!("openai/whisper-{}", size.name());
        Self::from_hub(&repo, "main", dims)
    }
}

fn keeps_checkpoint_dtype(key: &str) -> bool {
    key == "encoder.positional_embedding"
        || key == "decoder.positional_embedding"
        || key == "decoder.token_embedding.weight"
        || key.starts_with("encoder.ln_post.")
        || key.starts_with("decoder.ln.")
        || [".attn_ln.", ".cross_attn_ln.", ".mlp_ln."].iter().any(|part| key.contains(part))
}

/// Remap HuggingFace Transformers keys to the original OpenAI naming
/// convention used by our model structs. If keys already match (e.g.
/// loading from an original-format checkpoint), they pass through unchanged.
fn remap_hf_keys(sd: &StateDict) -> StateDict {
    // Detect format: if any key starts with "encoder." → already OpenAI format
    if sd.keys().any(|k| k.starts_with("encoder.")) {
        return sd.clone();
    }

    sd.iter().map(|(k, v)| (remap_key(k), v.clone())).collect()
}

/// Map a single HF Transformers key to the OpenAI original key name.
fn remap_key(key: &str) -> String {
    let k = key.strip_prefix("model.").unwrap_or(key);

    // Positional/token embeddings: HF Embedding params have `.weight` suffix;
    // OpenAI buffers don't (except token_embedding which does).
    let k = match k {
        "encoder.embed_positions.weight" => return "encoder.positional_embedding".into(),
        "decoder.embed_positions.weight" => return "decoder.positional_embedding".into(),
        _ => k,
    };

    // Token embedding: keep .weight
    let k = k.replacen("decoder.embed_tokens", "decoder.token_embedding", 1);

    // ── Encoder ──────────────────────────────────────────────────────────
    let k = k.replacen("encoder.layer_norm", "encoder.ln_post", 1);
    let k = k.replacen("encoder.layers.", "encoder.blocks.", 1);

    // ── Decoder ──────────────────────────────────────────────────────────
    let k = k.replacen("decoder.layer_norm", "decoder.ln", 1);
    let k = k.replacen("decoder.layers.", "decoder.blocks.", 1);

    // ── Per-layer projection names (applies to both encoder and decoder) ─
    let k = k.replace("self_attn_layer_norm", "attn_ln");
    let k = k.replace("encoder_attn_layer_norm", "cross_attn_ln");
    let k = k.replace("encoder_attn", "cross_attn");
    let k = k.replace("self_attn", "attn");
    // Projection names are now uniform: {attn,cross_attn}.{q,k,v,out}_proj
    let k = k.replace("q_proj", "query");
    let k = k.replace("k_proj", "key");
    let k = k.replace("v_proj", "value");
    let k = k.replace("out_proj", "out");
    let k = k.replace("fc1", "mlp.0");
    let k = k.replace("fc2", "mlp.2");

    k.replace("final_layer_norm", "mlp_ln")
}
