//! Root Qwen3 decoder backbone: token embeddings → RoPE decoder stack →
//! final RMSNorm. Exposes `forward` returning the `(B, L, D)` last-hidden-state.
//!
//! Loads from a HuggingFace `model.safetensors` checkpoint with bare keys
//! (`embed_tokens.weight`, `layers.N.*`, `norm.weight` — no `model.` prefix).
//! Compute dtype is `config.dtype` (bf16 by default, f32 for CPU parity).

use std::path::Path;

use svod_tensor::Tensor;
use svod_tensor::nn::{Embedding, Layer, Module, RmsNorm};

use crate::state::{self, StateDict};

use super::config::Qwen3Config;
use super::decoder_layer::Qwen3DecoderLayer;
use super::error::Result;

#[derive(Clone, Module)]
pub struct Qwen3Model {
    #[module(skip)]
    pub config: Qwen3Config,
    #[module(key = "embed_tokens")]
    pub embeddings: Embedding,
    pub layers: Vec<Qwen3DecoderLayer>,
    pub norm: RmsNorm,
}

impl Qwen3Model {
    pub fn empty(config: Qwen3Config) -> Self {
        let dtype = config.dtype.clone();
        let embeddings = crate::init::embedding(config.vocab_size, config.hidden_size, dtype.clone());
        let layers = (0..config.num_hidden_layers).map(|_| Qwen3DecoderLayer::empty(&config)).collect();
        let norm = RmsNorm::with_dims(config.hidden_size, config.rms_norm_eps, dtype);
        Self { config, embeddings, layers, norm }
    }

    /// Eager forward: `input_ids` `(B, L)` + optional `padding_mask` `(B, L)`
    /// bool (`true` = real token) → last-hidden-state `(B, L, D)`.
    pub fn forward(&self, input_ids: &Tensor, padding_mask: Option<&Tensor>) -> Result<Tensor> {
        let x = self.embeddings.forward(input_ids)?;
        let seq_len = x.dim_const(1)?;

        // Build the rotary table once — shared across all layers.
        let rope =
            Tensor::rope_table(self.config.rope_theta, seq_len, self.config.head_dim, self.config.dtype.clone())?;

        let mut h = x;
        for layer in &self.layers {
            h = layer.forward(&h, &rope, padding_mask)?;
        }
        Ok(self.norm.forward(&h)?)
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

    /// Load from a directory containing `model.safetensors` (single-file) or
    /// `model.safetensors.index.json` + shards (multi-shard).
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
