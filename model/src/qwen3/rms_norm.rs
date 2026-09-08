//! RMSNorm weights for Qwen3.
//!
//! Used in three roles:
//! - Per-head Q/K norm (weight shape `[head_dim]`, applied to `(B, H, L, head_dim)`)
//! - Layer norm before attention / MLP (weight shape `[hidden_size]`)
//! - Final norm (weight shape `[hidden_size]`)
//!
//! `Tensor::rms_norm` upcasts to f32 internally and normalizes over all
//! trailing axes from `axis` onward. With `axis = -1` only the last dim is
//! the normalization axis, which is correct for all three roles.

use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::init::ones;
use crate::state::{self, HasStateDict, StateDict, get_tensor, prefixed};

use super::error::Result;

#[derive(Clone)]
pub struct RmsNormWeights {
    pub weight: Tensor,
    pub eps: f64,
}

impl RmsNormWeights {
    pub fn empty(size: usize, eps: f64, dtype: DType) -> Self {
        Self { weight: ones(&[size], dtype), eps }
    }

    pub fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let normed = x.rms_norm(-1, self.eps)?;
        Ok(normed.try_mul(&self.weight)?)
    }
}

impl HasStateDict for RmsNormWeights {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.insert(prefixed(prefix, "weight"), self.weight.clone());
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.weight = get_tensor(sd, &prefixed(prefix, "weight"))?;
        Ok(())
    }
}
