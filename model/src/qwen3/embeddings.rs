//! Qwen3 token embeddings — plain `nn.Embedding` lookup with no norm.
//!
//! Position information enters via RoPE in attention, not via embedding
//! tables. Unlike ModernBERT (which LayerNorms after embedding), Qwen3
//! applies the first norm inside the decoder layer.

use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::init::fan_in_uniform;
use crate::state::{self, HasStateDict, StateDict, get_tensor, prefixed};

use super::error::Result;

#[derive(Clone)]
pub struct Qwen3Embeddings {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub embed_weight: Tensor,
}

impl Qwen3Embeddings {
    pub fn empty(vocab_size: usize, hidden_size: usize, dtype: DType) -> Self {
        Self { vocab_size, hidden_size, embed_weight: fan_in_uniform(&[vocab_size, hidden_size], hidden_size, dtype) }
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let _l = input_ids.dim_const(1)?;
        Ok(self.embed_weight.embedding(input_ids)?)
    }
}

impl HasStateDict for Qwen3Embeddings {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.insert(prefixed(prefix, "weight"), self.embed_weight.clone());
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.embed_weight = get_tensor(sd, &prefixed(prefix, "weight"))?;
        Ok(())
    }
}
