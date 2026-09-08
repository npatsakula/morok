//! State-dict pre-passes for PyTorch checkpoints.
//!
//! [`BatchNorm2d`](svod_tensor::nn::BatchNorm2d) reads PyTorch's own keys, so
//! nothing has to be renamed for it: the only entry a checkpoint carries that
//! no layer consumes is `num_batches_tracked`, which [`strip_metadata`] drops.
//!
//! [`fold_batchnorm`] additionally pre-computes `invstd` for the models still
//! on [`BatchNormWeights`](super::batchnorm::BatchNormWeights); it keeps
//! `running_var` in place so a dict can feed both.

use svod_tensor::Tensor;

use crate::state::StateDict;

use super::batchnorm::BN_EPS;
use super::error::Result;

/// Drop `num_batches_tracked` — training metadata with no inference use.
pub fn strip_metadata(mut sd: StateDict) -> StateDict {
    sd.retain(|k, _| !k.ends_with("num_batches_tracked"));
    sd
}

/// [`strip_metadata`] plus an `*.invstd = 1 / sqrt(running_var + eps)` entry
/// beside every `*.running_var`, computed on the host.
///
/// Compatibility shim for the models still loading
/// [`BatchNormWeights`](super::batchnorm::BatchNormWeights); delete it with
/// them.
pub fn fold_batchnorm(sd: StateDict) -> Result<StateDict> {
    let mut sd = strip_metadata(sd);
    let var_keys: Vec<String> = sd.keys().filter(|k| k.ends_with("running_var")).cloned().collect();
    for key in var_keys {
        let var = sd[&key].to_vec::<f32>()?;
        let invstd: Vec<f32> = var.iter().map(|&v| 1.0 / (v + BN_EPS as f32).sqrt()).collect();
        sd.insert(key.replace(".running_var", ".invstd"), Tensor::from_slice(&invstd));
    }
    Ok(sd)
}
