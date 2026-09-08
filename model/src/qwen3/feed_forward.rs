//! Qwen3 gated feed-forward (SwiGLU): `down(silu(gate(x)) * up(x))`.
//!
//! No biases. Three separate projections (gate, up, down) — not a fused
//! 2×intermediate matrix like ModernBERT.

use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::init::fan_in_uniform;
use crate::state::{self, HasStateDict, StateDict, get_tensor, prefixed};

use super::error::Result;

#[derive(Clone)]
pub struct Qwen3MLP {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub gate_weight: Tensor,
    pub up_weight: Tensor,
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

impl HasStateDict for Qwen3MLP {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.insert(prefixed(prefix, "gate_proj.weight"), self.gate_weight.clone());
        sd.insert(prefixed(prefix, "up_proj.weight"), self.up_weight.clone());
        sd.insert(prefixed(prefix, "down_proj.weight"), self.down_weight.clone());
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.gate_weight = get_tensor(sd, &prefixed(prefix, "gate_proj.weight"))?;
        self.up_weight = get_tensor(sd, &prefixed(prefix, "up_proj.weight"))?;
        self.down_weight = get_tensor(sd, &prefixed(prefix, "down_proj.weight"))?;
        Ok(())
    }
}
