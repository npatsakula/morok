//! Qwen3 embedding model: decoder backbone + last-token pooling + L2 normalize.
//!
//! The pipeline (matching `sentence-transformers` `modules.json`):
//! 1. Qwen3 decoder forward → `(B, L, D)` last hidden states
//! 2. Last-token pooling → `(B, D)` (select position `L-1`)
//! 3. L2 normalize → `(B, D)` unit-norm embeddings
//!
//! **Padding convention**: requires **left-padding** (standard for decoder
//! embedding inference). With left-padding the last real token is at position
//! `L-1`, a compile-time constant — so last-token pooling works in JIT
//! (identical to CLS pooling in ModernBERT).
//!
//! Loads from the same `model.safetensors` as [`Qwen3Model`] (bare keys,
//! no `model.` prefix).

use std::path::Path;

use svod_tensor::Tensor;
use svod_tensor::nn::Module;

use crate::state::{self, StateDict};

use super::config::Qwen3Config;
use super::error::Result;

use super::model::Qwen3Model;

#[derive(Clone, Module)]
pub struct Qwen3Embedding {
    #[module(key = "")]
    pub model: Qwen3Model,
    pub normalize: bool,
}

impl Qwen3Embedding {
    pub fn empty(config: Qwen3Config) -> Self {
        let model = Qwen3Model::empty(config);
        Self { model, normalize: true }
    }

    /// Eager forward: `input_ids` `(B, L)` + `attention_mask` `(B, L)` →
    /// embeddings `(B, D)`.
    pub fn encode(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let hidden = self.model.forward(input_ids, Some(attention_mask))?;
        self.pool_and_normalize(&hidden)
    }

    fn pool_and_normalize(&self, hidden: &Tensor) -> Result<Tensor> {
        // Last-token pooling: take position L-1 (requires left-padding).
        let pooled = hidden.take_index(1, -1)?;
        if self.normalize { Ok(pooled.lp_normalize(-1, 2)?) } else { Ok(pooled) }
    }

    pub fn from_hub(model_id: &str, mut config: Qwen3Config) -> Result<Self> {
        Self::from_hub_with_revision(model_id, "main", &mut config)
    }

    pub fn from_hub_with_revision(model_id: &str, revision: &str, config: &mut Qwen3Config) -> Result<Self> {
        let repo = crate::hub::HubRepo::open(model_id, revision)?;
        let cfg_path = repo.get("config.json")?;
        let parsed = Qwen3Config::from_json(&cfg_path)?;
        config.merge_structural_from(&parsed);

        let dir = crate::qwen3::download_safetensors(&repo)?;
        Self::from_safetensors_dir(&dir, config.clone())
    }

    pub fn from_safetensors(path: &Path, config: Qwen3Config) -> Result<Self> {
        let sd = state::load_safetensors(path)?;
        Self::from_state_dict(&sd, config)
    }

    /// Load from a directory containing `model.safetensors` or multi-shard files.
    pub fn from_safetensors_dir(dir: &Path, config: Qwen3Config) -> Result<Self> {
        let sd = state::load_safetensors_dir(dir)?;
        Self::from_state_dict(&sd, config)
    }

    pub fn from_state_dict(sd: &StateDict, config: Qwen3Config) -> Result<Self> {
        let dtype = config.dtype.clone();
        let mut model = Self::empty(config);
        model.load_state_dict(&state::cast_all(sd, dtype), "")?;
        Ok(model)
    }
}
