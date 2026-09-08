//! Root ModernBERT backbone: token embeddings → RoPE encoder stack → final
//! LayerNorm. Exposes `forward` returning the `(B, L, D)` last-hidden-state.
//!
//! Loads from a HuggingFace `model.safetensors` checkpoint (the published
//! `answerdotai/ModernBERT-{base,large}` layout). Compute dtype is
//! `config.dtype` (bf16 by default, f32 for CPU parity tests); weights load at
//! their stored dtype and are cast to `config.dtype` on read.

use std::path::Path;

use snafu::ResultExt;
use svod_ir::SInt;
use svod_tensor::{BoundVariable, Tensor};

use crate::state::{self, HasStateDict, StateDict};

use super::config::ModernBertConfig;
use super::embeddings::Embeddings;
use super::encoder::Encoder;
use super::error::{HubSnafu, Result, StateSnafu, TensorSnafu};
use super::normalization::LayerNormWeights;

#[derive(Clone)]
pub struct ModernBert {
    pub config: ModernBertConfig,
    pub embeddings: Embeddings,
    pub encoder: Encoder,
    pub final_norm: LayerNormWeights,
}

impl ModernBert {
    pub fn empty(config: ModernBertConfig) -> Self {
        let eps = config.layer_norm_eps;
        let dtype = config.dtype.clone();
        let embeddings = Embeddings::empty(config.vocab_size, config.hidden_size, eps, dtype.clone());
        let encoder = Encoder::empty(&config);
        let final_norm = LayerNormWeights::with_eps(config.hidden_size, eps, dtype);
        Self { config, embeddings, encoder, final_norm }
    }

    /// Eager forward: `input_ids` `(B, L)` int64 + optional `padding_mask`
    /// `(B, L)` bool → last-hidden-state `(B, L, D)`.
    pub fn forward(&self, input_ids: &Tensor, padding_mask: Option<&Tensor>) -> Result<Tensor> {
        let x = self.embeddings.forward(input_ids)?;
        let x = self.encoder.forward(&x, padding_mask)?;
        self.final_norm.apply(&x)
    }

    /// JIT-path variant: `input_ids` / `padding_mask` are sized for the JIT
    /// plan's `max_batch`; `b` shrinks the leading batch dim to the live value
    /// at execute time. The symbolic batch survives the embedding op (which
    /// carries index dims through as `SInt`), so one compiled plan serves all
    /// batch sizes up to `max_batch_size`.
    pub fn forward_batch(
        &self,
        input_ids: &Tensor,
        padding_mask: Option<&Tensor>,
        b: &BoundVariable,
    ) -> Result<Tensor> {
        let bv = b.as_sint();
        let input_ids = input_ids.try_shrink([Some((SInt::Const(0), bv.clone())), None]).context(TensorSnafu)?;
        let padding_mask = match padding_mask {
            Some(m) => Some(m.try_shrink([Some((SInt::Const(0), bv)), None]).context(TensorSnafu)?),
            None => None,
        };
        self.forward(&input_ids, padding_mask.as_ref())
    }

    /// Download `config.json` + `model.safetensors` from a HuggingFace Hub
    /// repository and load the backbone. Config is parsed from `config.json`,
    /// then overridden with the caller's `dtype`/`max_batch_size` from `config`.
    pub fn from_hub(model_id: &str, mut config: ModernBertConfig) -> Result<Self> {
        Self::from_hub_with_revision(model_id, "main", &mut config)
    }

    pub fn from_hub_with_revision(model_id: &str, revision: &str, config: &mut ModernBertConfig) -> Result<Self> {
        let repo = crate::hub::HubRepo::open(model_id, revision).context(HubSnafu)?;
        // Parse the published config.json, then splice back the caller-chosen
        // dtype / max_batch_size (those aren't in the on-disk config).
        let cfg_path = repo.get("config.json").context(HubSnafu)?;
        let parsed = ModernBertConfig::from_json(&cfg_path)?;
        config.merge_structural_from(&parsed);

        let weights_path = repo.get("model.safetensors").context(HubSnafu)?;
        Self::from_safetensors(&weights_path, config.clone())
    }

    /// Load from a `model.safetensors` checkpoint. Weights are cast to
    /// `config.dtype` as they are read.
    pub fn from_safetensors(path: &Path, config: ModernBertConfig) -> Result<Self> {
        let sd = state::load_safetensors(path).context(StateSnafu)?;
        Self::from_state_dict(&sd, config)
    }

    /// Build from a preloaded state dict. Each weight is cast to
    /// `config.dtype`; keys absent from the checkpoint (e.g. layer-0
    /// `attn_norm`, the optional norm biases) are tolerated by the per-block
    /// loaders.
    pub fn from_state_dict(sd: &StateDict, config: ModernBertConfig) -> Result<Self> {
        let dtype = config.dtype.clone();
        let mut model = Self::empty(config);
        model.load_state_dict(&state::cast_all(sd, dtype), "").context(StateSnafu)?;
        Ok(model)
    }
}

impl HasStateDict for ModernBert {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.extend(self.embeddings.state_dict(&prefix_or(prefix, "model.embeddings")));
        sd.extend(self.encoder.state_dict(&prefix_or(prefix, "model.layers")));
        sd.extend(self.final_norm.state_dict(&prefix_or(prefix, "model.final_norm")));
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.embeddings.load_state_dict(sd, &prefix_or(prefix, "model.embeddings"))?;
        self.encoder.load_state_dict(sd, &prefix_or(prefix, "model.layers"))?;
        self.final_norm.load_state_dict(sd, &prefix_or(prefix, "model.final_norm"))?;
        Ok(())
    }
}

fn prefix_or(prefix: &str, suffix: &str) -> String {
    if prefix.is_empty() { suffix.to_string() } else { format!("{prefix}.{suffix}") }
}
