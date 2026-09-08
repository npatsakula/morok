use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::init::{ones, zeros};
use crate::state::{self, HasStateDict, StateDict, get_tensor, prefixed};

use super::error::Result;

/// BN with the running variance pre-folded into `invstd`. State-dict keys:
/// `weight` (→ `scale`), `bias`, `running_mean` (→ `mean`), `invstd`.
/// The `invstd` key diverges from PyTorch's `running_var` intentionally: it
/// matches the actual data stored and makes a raw-PyTorch dict fail with a
/// "missing key" error rather than silently loading variance into `invstd`.
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
