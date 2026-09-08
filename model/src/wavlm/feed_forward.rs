//! WavLM per-layer feed-forward block: `Linear(d, dff) → GELU → Linear(dff, d)`.
//!
//! Mirrors `FeedForward` from `components.py:762-820`. Does NOT include a
//! LayerNorm — the pre-FFN norm is the encoder layer's `final_layer_norm`.

use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::init::{fan_in_uniform, zeros};
use crate::state::{self, HasStateDict, StateDict, get_tensor, prefixed};

use super::error::Result;

#[derive(Clone)]
pub struct FeedForwardWeights {
    pub io_features: usize,
    pub intermediate_features: usize,
    pub intermediate_weight: Tensor,
    pub intermediate_bias: Tensor,
    pub output_weight: Tensor,
    pub output_bias: Tensor,
}

impl FeedForwardWeights {
    pub fn empty(io_features: usize, intermediate_features: usize) -> Self {
        Self {
            io_features,
            intermediate_features,
            intermediate_weight: fan_in_uniform(&[intermediate_features, io_features], io_features, DType::Float32),
            intermediate_bias: zeros(&[intermediate_features], DType::Float32),
            output_weight: fan_in_uniform(&[io_features, intermediate_features], intermediate_features, DType::Float32),
            output_bias: zeros(&[io_features], DType::Float32),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = x.linear().weight(&self.intermediate_weight).bias(&self.intermediate_bias).call()?;
        // PyTorch's `nn.functional.gelu(x)` defaults to the EXACT erf-based
        // GELU; morok's `.gelu()` is the tanh approximation. Use `gelu_exact`
        // for numerical parity with the published checkpoint.
        let y = y.gelu_exact()?;
        Ok(y.linear().weight(&self.output_weight).bias(&self.output_bias).call()?)
    }
}

impl HasStateDict for FeedForwardWeights {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.insert(prefixed(prefix, "intermediate_dense.weight"), self.intermediate_weight.clone());
        sd.insert(prefixed(prefix, "intermediate_dense.bias"), self.intermediate_bias.clone());
        sd.insert(prefixed(prefix, "output_dense.weight"), self.output_weight.clone());
        sd.insert(prefixed(prefix, "output_dense.bias"), self.output_bias.clone());
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.intermediate_weight = get_tensor(sd, &prefixed(prefix, "intermediate_dense.weight"))?;
        self.intermediate_bias = get_tensor(sd, &prefixed(prefix, "intermediate_dense.bias"))?;
        self.output_weight = get_tensor(sd, &prefixed(prefix, "output_dense.weight"))?;
        self.output_bias = get_tensor(sd, &prefixed(prefix, "output_dense.bias"))?;
        Ok(())
    }
}
