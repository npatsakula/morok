//! WavLM per-layer feed-forward block: `Linear(d, dff) → GELU → Linear(dff, d)`.
//!
//! Mirrors `FeedForward` from `components.py:762-820`. Does NOT include a
//! LayerNorm — the pre-FFN norm is the encoder layer's `final_layer_norm`.

use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::{Layer, Linear, Module};

use crate::init::{Bias, linear};

use super::error::Result;

#[derive(Clone, Module)]
pub struct FeedForward {
    #[module(key = "intermediate_dense")]
    pub intermediate: Linear,
    #[module(key = "output_dense")]
    pub output: Linear,
}

impl FeedForward {
    pub fn empty(io_features: usize, intermediate_features: usize) -> Self {
        let linear = |inp: usize, out: usize| linear(inp, out, Bias::Zero, DType::Float32);
        Self {
            intermediate: linear(io_features, intermediate_features),
            output: linear(intermediate_features, io_features),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // PyTorch's `nn.functional.gelu(x)` defaults to the EXACT erf-based
        // GELU; `.gelu()` is the tanh approximation. Use `gelu_exact` for
        // numerical parity with the published checkpoint.
        let y = self.intermediate.forward(x)?.gelu_exact()?;
        Ok(self.output.forward(&y)?)
    }
}
