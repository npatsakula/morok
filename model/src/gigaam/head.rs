use morok_tensor::Tensor;
use snafu::ResultExt;

use crate::state::{HasStateDict, StateDict, get_tensor, prefixed};

use super::GigaAmConfig;
use super::error::TensorSnafu;

type Result<T> = super::Result<T>;

/// CTC decoder head: Conv1d(d_model, vocab_size, k=1) + LogSoftmax.
pub struct CTCHead {
    pub weight: Tensor, // [vocab_size, d_model, 1]
    pub bias: Tensor,   // [vocab_size]
}

impl CTCHead {
    pub fn empty(config: &GigaAmConfig) -> Self {
        Self {
            weight: Tensor::full(&[config.vocab_size, config.d_model, 1], 0.0, morok_dtype::DType::Float32).unwrap(),
            bias: Tensor::full(&[config.vocab_size], 0.0, morok_dtype::DType::Float32).unwrap(),
        }
    }

    /// Forward pass. Input: `[B, d_model, T]`, output: `[B, T, vocab_size]` log-probs.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Conv1d(d_model, vocab_size, kernel_size=1)
        let y = x.conv2d().weight(&self.weight).bias(&self.bias).call().context(TensorSnafu)?;
        // [B, vocab_size, T] -> [B, T, vocab_size]
        let y = y.try_transpose(-1, -2).context(TensorSnafu)?;
        y.log_softmax(-1isize).context(TensorSnafu)
    }
}

impl HasStateDict for CTCHead {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.insert(prefixed(prefix, "weight"), self.weight.clone());
        sd.insert(prefixed(prefix, "bias"), self.bias.clone());
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), crate::state::Error> {
        self.weight = get_tensor(sd, &prefixed(prefix, "weight"))?;
        self.bias = get_tensor(sd, &prefixed(prefix, "bias"))?;
        Ok(())
    }
}
