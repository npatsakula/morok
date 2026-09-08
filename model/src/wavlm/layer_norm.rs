//! Affine layer normalization with a state-dict-friendly wrapper. Kept local
//! to `wavlm` (rather than reused from `gigaam::encoder::LayerNormWeights`,
//! which lives in a private module) so the two modules don't develop coupling.

use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::init::{ones, zeros};
use crate::state::{HasStateDict, StateDict};
use crate::{load_state_field, state_field};

use super::error::Result;

#[derive(Clone)]
pub struct LayerNormWeights {
    pub weight: Tensor,
    pub bias: Tensor,
    pub eps: f64,
}

impl LayerNormWeights {
    pub fn empty(size: usize) -> Self {
        Self { weight: ones(&[size], DType::Float32), bias: zeros(&[size], DType::Float32), eps: 1e-5 }
    }

    pub fn with_eps(size: usize, eps: f64) -> Self {
        Self { weight: ones(&[size], DType::Float32), bias: zeros(&[size], DType::Float32), eps }
    }

    /// Apply LayerNorm over the last axis.
    pub fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let normed = x.layernorm(-1, self.eps)?;
        Ok(normed.try_mul(&self.weight)?.try_add(&self.bias)?)
    }
}

impl HasStateDict for LayerNormWeights {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        state_field!(sd, prefix, self, [weight, bias]);
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), crate::state::Error> {
        load_state_field!(self, sd, prefix, [weight, bias]);
        Ok(())
    }
}
