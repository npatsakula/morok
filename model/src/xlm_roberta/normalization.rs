//! Affine layer normalization with an optional bias.
//!
//! The `Tensor::layernorm` op upcasts to f32 internally before casting back,
//! so this is numerically exact in bf16.

use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::init::ones;
use crate::state::{self, HasStateDict, StateDict, get_tensor, prefixed};

use super::error::Result;

#[derive(Clone)]
pub struct LayerNormWeights {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub eps: f64,
}

impl LayerNormWeights {
    pub fn empty(size: usize, dtype: DType) -> Self {
        Self { weight: ones(&[size], dtype), bias: None, eps: 1e-5 }
    }

    pub fn with_eps(size: usize, eps: f64, dtype: DType) -> Self {
        Self { weight: ones(&[size], dtype), bias: None, eps }
    }

    /// Apply LayerNorm over the last axis then affine:
    /// `(x - μ)/√(σ²+ε) * γ + β?`.
    pub fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let normed = x.layernorm(-1, self.eps)?;
        let scaled = normed.try_mul(&self.weight)?;
        match &self.bias {
            Some(b) => Ok(scaled.try_add(b)?),
            None => Ok(scaled),
        }
    }
}

impl HasStateDict for LayerNormWeights {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.insert(prefixed(prefix, "weight"), self.weight.clone());
        if let Some(b) = &self.bias {
            sd.insert(prefixed(prefix, "bias"), b.clone());
        }
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.weight = get_tensor(sd, &prefixed(prefix, "weight"))?;
        let bias_key = prefixed(prefix, "bias");
        self.bias = sd.get(&bias_key).cloned();
        Ok(())
    }
}
