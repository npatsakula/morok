//! ModernBERT token embeddings (`FlexBertSansPositionEmbeddings`): a plain
//! `nn.Embedding` lookup followed by a LayerNorm. There are **no position
//! embeddings** — position information enters via RoPE in attention.

use snafu::ResultExt;
use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::init::fan_in_uniform;
use crate::state::{self, HasStateDict, StateDict, get_tensor, prefixed};

use super::error::{Result, TensorSnafu};
use super::normalization::LayerNormWeights;

#[derive(Clone)]
pub struct Embeddings {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub tok_embeddings: Tensor,
    pub norm: LayerNormWeights,
}

impl Embeddings {
    pub fn empty(vocab_size: usize, hidden_size: usize, eps: f64, dtype: DType) -> Self {
        Self {
            vocab_size,
            hidden_size,
            tok_embeddings: fan_in_uniform(&[vocab_size, hidden_size], hidden_size, dtype.clone()),
            norm: LayerNormWeights::with_eps(hidden_size, eps, dtype),
        }
    }

    /// Forward. `input_ids`: `(B, L)` int64 → `(B, L, D)`.
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let x = self.tok_embeddings.embedding(input_ids).context(TensorSnafu)?;
        self.norm.apply(&x)
    }
}

impl HasStateDict for Embeddings {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.insert(prefixed(prefix, "tok_embeddings.weight"), self.tok_embeddings.clone());
        sd.extend(self.norm.state_dict(&format!("{prefix}.norm")));
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.tok_embeddings = get_tensor(sd, &prefixed(prefix, "tok_embeddings.weight"))?;
        self.norm.load_state_dict(sd, &format!("{prefix}.norm"))?;
        Ok(())
    }
}
