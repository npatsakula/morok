//! BGE-M3 sparse (lexical) embedding head.
//!
//! `Linear(D, 1, bias) → ReLU → scatter_reduce(input_ids, vocab, amax)`.
//! Produces a `(B, vocab_size)` sparse embedding vector where each position
//! holds the maximum ReLU activation across all tokens with that ID.
//!
//! Unused token positions (CLS, EOS, PAD, UNK) are zeroed post-scatter to
//! match the FlagEmbedding convention.
//!
//! Weights are loaded from `sparse_linear.pt` (PyTorch pickle with bare
//! `weight` / `bias` keys).

use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::indexing::ScatterReduction;

use crate::init::{fan_in_uniform, zeros};
use crate::state::{self, HasStateDict, StateDict, get_tensor, prefixed};
use crate::xlm_roberta::error::Result;

/// XLM-RoBERTa special token IDs — always zeroed in sparse output.
/// `<s>`=0, `<pad>`=1, `</s>`=2, `<unk>`=3.
const XLM_R_SPECIAL_IDS: &[usize] = &[0, 1, 2, 3];

#[derive(Clone)]
pub struct SparseHead {
    pub weight: Tensor,
    pub bias: Tensor,
    pub vocab_size: usize,
}

impl SparseHead {
    pub fn empty(hidden_size: usize, vocab_size: usize, dtype: DType) -> Self {
        Self {
            weight: fan_in_uniform(&[1, hidden_size], hidden_size, dtype.clone()),
            bias: zeros(&[1], dtype),
            vocab_size,
        }
    }

    /// Load from a `sparse_linear.pt` checkpoint (bare `weight`/`bias` keys).
    pub fn from_pytorch_bin(path: &std::path::Path, vocab_size: usize, dtype: DType) -> Result<Self> {
        let sd = crate::wespeaker::pickle::load_flat_pytorch_bin(path, "")
            .map_err(|e| crate::xlm_roberta::Error::Pickle { source: Box::new(e) })?;
        Self::from_state_dict(&sd, "", vocab_size, dtype)
    }

    /// Build from a preloaded state dict, casting to `dtype`.
    pub fn from_state_dict(sd: &StateDict, prefix: &str, vocab_size: usize, dtype: DType) -> Result<Self> {
        let sd = state::cast_all(sd, dtype.clone());
        let mut head = Self::empty(0, vocab_size, dtype);
        head.load_state_dict(&sd, prefix).map_err(|e| crate::xlm_roberta::Error::State { source: Box::new(e) })?;
        Ok(head)
    }

    /// Forward. `hidden`: `(B, L, D)`, `input_ids`: `(B, L)` int.
    /// Returns `(B, vocab_size)` sparse embedding with special-token positions zeroed.
    pub fn forward(&self, hidden: &Tensor, input_ids: &Tensor) -> Result<Tensor> {
        let b = hidden.dim_const(0)?;

        let token_weights = hidden.linear().weight(&self.weight).bias(&self.bias).call()?.relu()?;

        let token_weights = token_weights.try_squeeze(Some(-1))?;

        let zeros = Tensor::zeros(&[b, self.vocab_size], hidden.dtype());
        let sparse = zeros.scatter_reduce(-1, input_ids, &token_weights, ScatterReduction::Amax, true)?;

        let mask = self.unused_token_mask(hidden.dtype())?;
        Ok(sparse.try_mul(&mask)?)
    }

    fn unused_token_mask(&self, dtype: DType) -> Result<Tensor> {
        let mut vals = vec![1.0f32; self.vocab_size];
        for &id in XLM_R_SPECIAL_IDS {
            if id < self.vocab_size {
                vals[id] = 0.0;
            }
        }
        let mask = Tensor::from_slice(&vals).try_reshape([1isize, self.vocab_size as isize])?.cast(dtype);
        Ok(mask)
    }
}

impl HasStateDict for SparseHead {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.insert(prefixed(prefix, "weight"), self.weight.clone());
        sd.insert(prefixed(prefix, "bias"), self.bias.clone());
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.weight = get_tensor(sd, &prefixed(prefix, "weight"))?;
        self.bias = get_tensor(sd, &prefixed(prefix, "bias"))?;
        Ok(())
    }
}
