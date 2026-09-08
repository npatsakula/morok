//! Qwen3 gated feed-forward (SwiGLU): `down(silu(gate(x)) * up(x))`.
//!
//! No biases. Three separate projections (gate, up, down) — not a fused
//! 2×intermediate matrix like ModernBERT.

use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::Module;

use crate::init::fan_in_uniform;

use super::error::Result;

#[derive(Clone, Module)]
pub struct Qwen3MLP {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    #[module(key = "gate_proj.weight")]
    pub gate_weight: Tensor,
    #[module(key = "up_proj.weight")]
    pub up_weight: Tensor,
    #[module(key = "down_proj.weight")]
    pub down_weight: Tensor,
}

impl Qwen3MLP {
    pub fn empty(hidden_size: usize, intermediate_size: usize, dtype: DType) -> Self {
        let gate_weight = fan_in_uniform(&[intermediate_size, hidden_size], hidden_size, dtype.clone());
        let up_weight = fan_in_uniform(&[intermediate_size, hidden_size], hidden_size, dtype.clone());
        let down_weight = fan_in_uniform(&[hidden_size, intermediate_size], intermediate_size, dtype);
        Self { hidden_size, intermediate_size, gate_weight, up_weight, down_weight }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = x.linear().weight(&self.gate_weight).call()?;
        let up = x.linear().weight(&self.up_weight).call()?;
        let act = gate.silu()?.try_mul(&up)?;
        Ok(act.linear().weight(&self.down_weight).call()?)
    }
}
