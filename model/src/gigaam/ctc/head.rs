//! CTC projection head: `Conv1d(d_model, vocab_size, k=1)` + transpose +
//! `LogSoftmax`. Produces the `[B, T, vocab_size]` log-probabilities consumed
//! by `svod_arch::ctc` decoders — the head itself is just the final
//! projection layer, not the decoder.

use svod_dtype::DType;
use svod_tensor::Tensor;

use svod_tensor::nn::Module;

use crate::init::fan_in_uniform;

use crate::gigaam::{GigaAmConfig, Result};

#[derive(Clone, Module)]
pub struct CTCHead {
    pub weight: Tensor, // [vocab_size, d_model, 1]
    pub bias: Tensor,   // [vocab_size]
}

impl CTCHead {
    pub fn empty(config: &GigaAmConfig) -> Self {
        let fan_in = config.d_model;
        Self {
            weight: fan_in_uniform(&[config.vocab_size, config.d_model, 1], fan_in, DType::Float32),
            bias: fan_in_uniform(&[config.vocab_size], fan_in, DType::Float32),
        }
    }

    /// Forward pass. Input: `[B, d_model, T]`, output: `[B, T, vocab_size]` log-probs.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = x.conv1d().weight(&self.weight).bias(&self.bias).call()?;
        let y = y.try_transpose(-1, -2)?;
        Ok(y.log_softmax(-1isize)?)
    }
}
