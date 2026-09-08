//! ModernBERT gated feed-forward (SwiGLU-style): `Linear(D, 2I)` → split
//! `[input | gate]` → `GELU(input) * gate` → `Linear(I, D)`. No biases.
//!
//! Mirrors `FlexBertGLU`. `Wi` output dim is `2*intermediate` (input and gate
//! branches concatenated along the last axis); `Wo` takes the gated `I`-dim
//! result back to `D`.

use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::init::fan_in_uniform;
use crate::state::{self, HasStateDict, StateDict, get_tensor, prefixed};

use super::error::Result;

#[derive(Clone)]
pub struct ModernBertGlu {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub wi_weight: Tensor,
    pub wo_weight: Tensor,
}

impl ModernBertGlu {
    pub fn empty(hidden_size: usize, intermediate_size: usize, dtype: DType) -> Self {
        let wi_weight = fan_in_uniform(&[2 * intermediate_size, hidden_size], hidden_size, dtype.clone());
        let wo_weight = fan_in_uniform(&[hidden_size, intermediate_size], intermediate_size, dtype);
        Self { hidden_size, intermediate_size, wi_weight, wo_weight }
    }

    /// Forward. `x`: `(B, L, D)` → `(B, L, D)`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // (., 2I) → split into [input (., I), gate (., I)].
        let h = x.linear().weight(&self.wi_weight).call()?;
        let mut parts = h.chunk(2, -1)?;
        let gate = parts.pop().expect("chunk(2) yields 2 parts");
        let input = parts.pop().expect("chunk(2) yields 2 parts");
        // GELU(input) * gate — exact (erf) GELU matches PyTorch's nn.GELU default.
        let gated = input.gelu_exact()?.try_mul(&gate)?;
        Ok(gated.linear().weight(&self.wo_weight).call()?)
    }
}

impl HasStateDict for ModernBertGlu {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.insert(prefixed(prefix, "Wi.weight"), self.wi_weight.clone());
        sd.insert(prefixed(prefix, "Wo.weight"), self.wo_weight.clone());
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.wi_weight = get_tensor(sd, &prefixed(prefix, "Wi.weight"))?;
        self.wo_weight = get_tensor(sd, &prefixed(prefix, "Wo.weight"))?;
        Ok(())
    }
}
