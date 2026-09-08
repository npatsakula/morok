//! XLM-RoBERTa embeddings: word + position + token_type → LayerNorm.
//!
//! Unlike ModernBERT (which uses RoPE and has no position embeddings),
//! XLM-RoBERTa uses **absolute learned position embeddings** with the fairseq
//! `make_positions` offset (see [`super::position_ids`]).
//!
//! Token-type embeddings: `type_vocab_size = 1`, so only row 0 exists and is
//! always selected (token_type_ids are implicitly zeros). We broadcast row 0
//! directly instead of constructing a zeros index tensor — simpler and avoids
//! a symbolic-shape dependency on the index.

use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::init::fan_in_uniform;
use crate::state::{self, HasStateDict, StateDict, get_tensor, prefixed};

use super::error::Result;

use super::normalization::LayerNormWeights;
use super::position_ids::position_ids_from_input_ids;

#[derive(Clone)]
pub struct XlmRobertaEmbeddings {
    pub word_embeddings: Tensor,
    pub position_embeddings: Tensor,
    pub token_type_embeddings: Tensor,
    pub norm: LayerNormWeights,
    pub pad_token_id: usize,
}

impl XlmRobertaEmbeddings {
    pub fn empty(
        vocab_size: usize,
        hidden_size: usize,
        max_position_embeddings: usize,
        type_vocab_size: usize,
        eps: f64,
        pad_token_id: usize,
        dtype: DType,
    ) -> Self {
        Self {
            word_embeddings: fan_in_uniform(&[vocab_size, hidden_size], hidden_size, dtype.clone()),
            position_embeddings: fan_in_uniform(&[max_position_embeddings, hidden_size], hidden_size, dtype.clone()),
            token_type_embeddings: fan_in_uniform(&[type_vocab_size, hidden_size], hidden_size, dtype.clone()),
            norm: LayerNormWeights::with_eps(hidden_size, eps, dtype),
            pad_token_id,
        }
    }

    /// Forward. `input_ids`: `(B, L)` int → `(B, L, D)`.
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let _l = input_ids.dim_const(1)?;

        let word = self.word_embeddings.embedding(input_ids)?;

        let pos_ids = position_ids_from_input_ids(input_ids, self.pad_token_id)?;
        let pos = self.position_embeddings.embedding(&pos_ids)?;

        let tok_type = self.token_type_embeddings.try_unsqueeze(0)?;

        let h = word.try_add(&pos)?.try_add(&tok_type)?;
        self.norm.apply(&h)
    }
}

impl HasStateDict for XlmRobertaEmbeddings {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.insert(prefixed(prefix, "word_embeddings.weight"), self.word_embeddings.clone());
        sd.insert(prefixed(prefix, "position_embeddings.weight"), self.position_embeddings.clone());
        sd.insert(prefixed(prefix, "token_type_embeddings.weight"), self.token_type_embeddings.clone());
        sd.extend(self.norm.state_dict(&format!("{prefix}.LayerNorm")));
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.word_embeddings = get_tensor(sd, &prefixed(prefix, "word_embeddings.weight"))?;
        self.position_embeddings = get_tensor(sd, &prefixed(prefix, "position_embeddings.weight"))?;
        self.token_type_embeddings = get_tensor(sd, &prefixed(prefix, "token_type_embeddings.weight"))?;
        self.norm.load_state_dict(sd, &format!("{prefix}.LayerNorm"))?;
        Ok(())
    }
}
