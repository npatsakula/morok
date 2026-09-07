//! Shared loader helpers for YOLO models.

use std::path::Path;

use snafu::ResultExt;
use svod_tensor::Tensor;

use crate::state::{self, StateDict};

use super::error::{HubSnafu, Result, StateSnafu, TensorSnafu};

/// Download `model.safetensors` from HuggingFace Hub.
pub fn download_safetensors(model_id: &str, revision: &str) -> Result<std::path::PathBuf> {
    let repo = crate::hub::HubRepo::open(model_id, revision).context(HubSnafu)?;
    repo.get("model.safetensors").context(HubSnafu)
}

/// Load + fold BN + strip `model.` prefix, returning a clean state dict.
pub fn prepare_state_dict(path: &Path) -> Result<StateDict> {
    let sd = state::load_safetensors(path).context(StateSnafu)?;
    prepare_state_dict_from_sd(&sd)
}

/// Same as [`prepare_state_dict`] but from a pre-loaded state dict.
pub fn prepare_state_dict_from_sd(sd: &StateDict) -> Result<StateDict> {
    let sd = crate::blocks::remap::fold_batchnorm(sd.clone()).map_err(super::error::Error::from)?;
    Ok(strip_model_prefix(&sd))
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
    use svod_ir::SInt;
    images.try_shrink([Some((SInt::Const(0), batch.as_sint())), None, None, None]).context(TensorSnafu)
}
