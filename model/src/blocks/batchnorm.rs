use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::BatchNorm2d;

use crate::init::{ones, zeros};
use crate::state::{self, HasStateDict, StateDict, get_tensor, prefixed};

use super::error::Result;

/// Default PyTorch BatchNorm epsilon. The timm, Ultralytics and WeSpeaker
/// checkpoints we target do not override it.
pub const BN_EPS: f64 = 1e-5;

/// Identity-initialized inference batch norm over the channel axis, keyed with
/// PyTorch's `weight` / `bias` / `running_mean` / `running_var` names.
pub fn batchnorm2d(channels: usize) -> BatchNorm2d {
    BatchNorm2d::with_dims(channels, BN_EPS, DType::Float32)
}

/// Compatibility shim for callers not yet ported to [`BatchNorm2d`], holding
/// the variance pre-folded into `invstd` as
/// [`remap::fold_batchnorm`](super::remap::fold_batchnorm) writes it. Delete
/// once `wespeaker`, `diarizen` and `gtcrn` are migrated.
#[derive(Clone)]
pub struct BatchNormWeights {
    pub scale: Tensor,
    pub bias: Tensor,
    pub mean: Tensor,
    pub invstd: Tensor,
}

impl BatchNormWeights {
    pub fn empty(channels: usize) -> Self {
        Self {
            scale: ones(&[channels], DType::Float32),
            bias: zeros(&[channels], DType::Float32),
            mean: zeros(&[channels], DType::Float32),
            invstd: ones(&[channels], DType::Float32),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.batchnorm().scale(&self.scale).bias(&self.bias).mean(&self.mean).invstd(&self.invstd).call()?)
    }
}

impl HasStateDict for BatchNormWeights {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.insert(prefixed(prefix, "weight"), self.scale.clone());
        sd.insert(prefixed(prefix, "bias"), self.bias.clone());
        sd.insert(prefixed(prefix, "running_mean"), self.mean.clone());
        sd.insert(prefixed(prefix, "invstd"), self.invstd.clone());
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.scale = get_tensor(sd, &prefixed(prefix, "weight"))?;
        self.bias = get_tensor(sd, &prefixed(prefix, "bias"))?;
        self.mean = get_tensor(sd, &prefixed(prefix, "running_mean"))?;
        self.invstd = get_tensor(sd, &prefixed(prefix, "invstd"))?;
        Ok(())
    }
}
