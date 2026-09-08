//! Shared loader helpers for YOLO models.

use std::path::Path;

use svod_tensor::Tensor;

use crate::state::{self, StateDict};

use super::error::Result;

/// Download `model.safetensors` from HuggingFace Hub.
pub fn download_safetensors(model_id: &str, revision: &str) -> Result<std::path::PathBuf> {
    let repo = crate::hub::HubRepo::open(model_id, revision)?;
    Ok(repo.get("model.safetensors")?)
}

/// Load a checkpoint and strip the `model.` prefix, returning a clean state
/// dict. The layers read PyTorch's own keys, so nothing else is renamed.
pub fn prepare_state_dict(path: &Path) -> Result<StateDict> {
    Ok(strip_model_prefix(&state::load_safetensors(path)?))
}

/// Strip the `model.` prefix from all keys if present (Ultralytics wraps
/// everything in `self.model`).
pub fn strip_model_prefix(sd: &StateDict) -> StateDict {
    if sd.keys().any(|k| k.starts_with("model.")) {
        sd.iter()
            .map(|(k, v)| {
                let k2 = k.strip_prefix("model.").unwrap_or(k);
                (k2.to_string(), v.clone())
            })
            .collect()
    } else {
        sd.clone()
    }
}

/// Shrink the batch dimension of a 4D NCHW tensor to a bound variable.
pub fn shrink_batch(images: &Tensor, batch: &svod_tensor::BoundVariable) -> Result<Tensor> {
    Ok(images.narrow(0, 0usize, batch.as_sint())?)
}
