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
use svod_tensor::nn::{Embedding, Layer, LayerNorm, Module};

use crate::init::embedding;

use super::error::Result;

use super::position_ids::position_ids_from_input_ids;

#[derive(Clone, Module)]
pub struct XlmRobertaEmbeddings {
    pub word_embeddings: Embedding,
    pub position_embeddings: Embedding,
    pub token_type_embeddings: Embedding,
    #[module(key = "LayerNorm")]
    pub norm: LayerNorm,
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
        let table = |rows: usize| embedding(rows, hidden_size, dtype.clone());
        Self {
            word_embeddings: table(vocab_size),
            position_embeddings: table(max_position_embeddings),
            token_type_embeddings: table(type_vocab_size),
            norm: LayerNorm::with_dims(hidden_size, false, eps, dtype),
            pad_token_id,
        }
    }

    /// Forward. `input_ids`: `(B, L)` int → `(B, L, D)`.
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let word = self.word_embeddings.forward(input_ids)?;

        let pos_ids = position_ids_from_input_ids(input_ids, self.pad_token_id)?;
        let pos = self.position_embeddings.forward(&pos_ids)?;

        let tok_type = self.token_type_embeddings.weight.try_unsqueeze(0)?;

        let h = word.try_add(&pos)?.try_add(&tok_type)?;
        Ok(self.norm.forward(&h)?)
    }
}
