//! XLM-RoBERTa feed-forward block: `Linear(D, I, bias) → GELU(exact) → Linear(I, D, bias)`.

use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::init::{fan_in_uniform, zeros};
use crate::state::{self, HasStateDict, StateDict, get_tensor, prefixed};

use super::error::Result;

#[derive(Clone)]
pub struct FeedForwardWeights {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub intermediate_weight: Tensor,
    pub intermediate_bias: Tensor,
    pub output_weight: Tensor,
    pub output_bias: Tensor,
}

impl FeedForwardWeights {
    pub fn empty(hidden_size: usize, intermediate_size: usize, dtype: DType) -> Self {
        Self {
            hidden_size,
            intermediate_size,
            intermediate_weight: fan_in_uniform(&[intermediate_size, hidden_size], hidden_size, dtype.clone()),
            intermediate_bias: zeros(&[intermediate_size], dtype.clone()),
            output_weight: fan_in_uniform(&[hidden_size, intermediate_size], intermediate_size, dtype.clone()),
            output_bias: zeros(&[hidden_size], dtype),
        }
    }

    /// Forward. `x`: `(B, L, D)` → `(B, L, D)`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = x.linear().weight(&self.intermediate_weight).bias(&self.intermediate_bias).call()?;
        let y = y.gelu_exact()?;
        Ok(y.linear().weight(&self.output_weight).bias(&self.output_bias).call()?)
    }
}

impl HasStateDict for FeedForwardWeights {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.insert(prefixed(prefix, "intermediate.dense.weight"), self.intermediate_weight.clone());
        sd.insert(prefixed(prefix, "intermediate.dense.bias"), self.intermediate_bias.clone());
        sd.insert(prefixed(prefix, "output.dense.weight"), self.output_weight.clone());
        sd.insert(prefixed(prefix, "output.dense.bias"), self.output_bias.clone());
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.intermediate_weight = get_tensor(sd, &prefixed(prefix, "intermediate.dense.weight"))?;
        self.intermediate_bias = get_tensor(sd, &prefixed(prefix, "intermediate.dense.bias"))?;
        self.output_weight = get_tensor(sd, &prefixed(prefix, "output.dense.weight"))?;
        self.output_bias = get_tensor(sd, &prefixed(prefix, "output.dense.bias"))?;
        Ok(())
    }
}
