//! ModernBERT MLM (fill-mask) head — `FlexBertPredictionHead` + decoder,
//! mirroring `FlexBertForMaskedLM`.
//!
//! The published `answerdotai/ModernBERT-{base,large}` head is **LayerNorm +
//! GELU**: `head_pred_act: gelu` and `normalization: layernorm` in the pretrain
//! YAML (HF's `modeling_modernbert.py` also hardcodes GELU). RMSNorm + SiLU is
//! only the FlexBERT *code default*, overridden for the released checkpoints,
//! so this reuses [`LayerNormWeights`] and [`Tensor::gelu_exact`] with no new
//! normalization/activation ops.
//!
//! **Weight tying.** When `config.tie_word_embeddings` is `true` (the published
//! checkpoints), the decoder weight is the token embedding table. The head
//! resolves this at load time — cloning `model.embeddings.tok_embeddings.weight`
//! out of the same state dict — so [`MlmHead::forward`] is fully self-contained
//! (no call-time tensor argument), matching the `gigaam::CTCHead` idiom. When
//! tying is disabled, the head loads a standalone `decoder.weight` instead.

use std::path::Path;

use snafu::ResultExt;
use svod_tensor::{BoundVariable, Tensor};

use crate::init::{fan_in_uniform, zeros};
use crate::state::{self, HasStateDict, StateDict, get_tensor, prefixed};

use super::config::ModernBertConfig;
use super::error::{HubSnafu, Result, StateSnafu, TensorSnafu};
use super::model::ModernBert;
use super::normalization::LayerNormWeights;

/// The token-embedding key the tied decoder aliases. Lives on the backbone but
/// is present in the same `model.safetensors` checkpoint the head loads from.
const TIED_EMBEDDING_KEY: &str = "model.embeddings.tok_embeddings.weight";

/// `FlexBertPredictionHead` + the decoder. Owns all its weights: `head.dense`
/// (square Linear, no bias), `head.norm` (gamma-only LayerNorm), the decoder
/// weight `(V, D)` (tied-to-embeddings or standalone), and `decoder.bias`
/// (present iff `config.decoder_bias`).
#[derive(Clone)]
pub struct MlmHead {
    pub dense_weight: Tensor,
    pub norm: LayerNormWeights,
    /// `(V, D)`. When `tie_word_embeddings`, this is a clone of the embedding
    /// table (aliased under `model.embeddings.tok_embeddings.weight` in the
    /// checkpoint); otherwise a standalone `decoder.weight`.
    pub decoder_weight: Tensor,
    /// `decoder.bias (V,)` — `None` when `config.decoder_bias` is false.
    pub decoder_bias: Option<Tensor>,
}

impl MlmHead {
    pub fn empty(config: &ModernBertConfig) -> Self {
        let dtype = config.dtype.clone();
        let dense_weight = fan_in_uniform(&[config.hidden_size, config.hidden_size], config.hidden_size, dtype.clone());
        let norm = LayerNormWeights::with_eps(config.hidden_size, config.layer_norm_eps, dtype.clone());
        let decoder_weight =
            fan_in_uniform(&[config.vocab_size, config.hidden_size], config.hidden_size, dtype.clone());
        let decoder_bias = if config.decoder_bias { Some(zeros(&[config.vocab_size], dtype)) } else { None };
        Self { dense_weight, norm, decoder_weight, decoder_bias }
    }

    /// `hidden (B,L,D)` → logits `(B,L,V)`.
    ///
    /// Order matches `FlexBertPredictionHead.forward` (`norm(act(dense(x))`)
    /// followed by the decoder Linear `logits = h @ decoder_weight.T + bias`.
    /// The decoder weight is owned by the head (tied or standalone), so this is
    /// self-contained — no external tensor argument, mirroring `CTCHead`.
    pub fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let h = hidden.linear().weight(&self.dense_weight).call().context(TensorSnafu)?;
        let h = h.gelu_exact().context(TensorSnafu)?;
        let h = self.norm.apply(&h)?;
        let logits = h.linear().weight(&self.decoder_weight).call().context(TensorSnafu)?;
        match &self.decoder_bias {
            Some(b) => logits.try_add(b).context(TensorSnafu),
            None => Ok(logits),
        }
    }
}

impl HasStateDict for MlmHead {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.insert(prefixed(prefix, "head.dense.weight"), self.dense_weight.clone());
        sd.extend(self.norm.state_dict(&prefixed(prefix, "head.norm")));
        // Emit the decoder weight under its own key. For a tied checkpoint this
        // duplicates `model.embeddings.tok_embeddings.weight`; loaders dedup on
        // load by re-aliasing, and `ModernBertForMaskedLm::state_dict` drops the
        // copy (see below).
        sd.insert(prefixed(prefix, "decoder.weight"), self.decoder_weight.clone());
        if let Some(b) = &self.decoder_bias {
            sd.insert(prefixed(prefix, "decoder.bias"), b.clone());
        }
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.dense_weight = get_tensor(sd, &prefixed(prefix, "head.dense.weight"))?;
        self.norm.load_state_dict(sd, &prefixed(prefix, "head.norm"))?;
        // Resolve the decoder weight. Published (tied) checkpoints store it
        // only as the embedding table, so alias from there when the standalone
        // `decoder.weight` key is absent. `model.safetensors` carries the
        // embedding weight at its unprefixed backbone key.
        self.decoder_weight = match sd.get(&prefixed(prefix, "decoder.weight")) {
            Some(w) => w.clone(),
            None => get_tensor(sd, TIED_EMBEDDING_KEY).or_else(|_| {
                // Fall back to the prefixed embedding key (head loaded under a
                // non-empty prefix).
                get_tensor(sd, &prefixed(prefix, TIED_EMBEDDING_KEY))
            })?,
        };
        let bias_key = prefixed(prefix, "decoder.bias");
        self.decoder_bias = sd.get(&bias_key).cloned();
        Ok(())
    }
}

/// `FlexBertForMaskedLM`: the backbone + the MLM head. Loads from the same
/// `model.safetensors` as [`ModernBert`] (the backbone keys `model.*` and the
/// head keys `head.*` / `decoder.*` are disjoint, so one checkpoint populates
/// both). Returns `(B, L, V)` logits.
#[derive(Clone)]
pub struct ModernBertForMaskedLm {
    pub bert: ModernBert,
    pub head: MlmHead,
}

impl ModernBertForMaskedLm {
    pub fn empty(config: ModernBertConfig) -> Self {
        let head = MlmHead::empty(&config);
        let bert = ModernBert::empty(config);
        Self { bert, head }
    }

    /// Eager forward: `input_ids` `(B, L)` + optional `padding_mask` `(B, L)`
    /// bool → logits `(B, L, V)`.
    pub fn forward(&self, input_ids: &Tensor, padding_mask: Option<&Tensor>) -> Result<Tensor> {
        let hidden = self.bert.forward(input_ids, padding_mask)?;
        self.head.forward(&hidden)
    }

    /// JIT-path variant: shrinks the leading batch dim to the live value at
    /// execute time. See [`ModernBert::forward_batch`].
    pub fn forward_batch(
        &self,
        input_ids: &Tensor,
        padding_mask: Option<&Tensor>,
        b: &BoundVariable,
    ) -> Result<Tensor> {
        let hidden = self.bert.forward_batch(input_ids, padding_mask, b)?;
        self.head.forward(&hidden)
    }

    /// Download `config.json` + `model.safetensors` from a HuggingFace Hub
    /// repository and load the full MLM model. Mirrors [`ModernBert::from_hub`].
    pub fn from_hub(model_id: &str, mut config: ModernBertConfig) -> Result<Self> {
        Self::from_hub_with_revision(model_id, "main", &mut config)
    }

    pub fn from_hub_with_revision(model_id: &str, revision: &str, config: &mut ModernBertConfig) -> Result<Self> {
        let api = hf_hub::api::sync::Api::new().context(HubSnafu)?;
        let repo =
            api.repo(hf_hub::Repo::with_revision(model_id.to_string(), hf_hub::RepoType::Model, revision.to_string()));
        let cfg_path = repo.get("config.json").context(HubSnafu)?;
        let parsed = ModernBertConfig::from_json(&cfg_path)?;
        config.apply_checkpoint(&parsed);

        let weights_path = repo.get("model.safetensors").context(HubSnafu)?;
        Self::from_safetensors(&weights_path, config.clone())
    }

    /// Load from a `model.safetensors` checkpoint. Weights are cast to
    /// `config.dtype` as they are read.
    pub fn from_safetensors(path: &Path, config: ModernBertConfig) -> Result<Self> {
        let sd = state::load_safetensors(path).context(StateSnafu)?;
        Self::from_state_dict(&sd, config)
    }

    /// Build from a preloaded state dict. Each weight is cast to `config.dtype`.
    pub fn from_state_dict(sd: &StateDict, config: ModernBertConfig) -> Result<Self> {
        let dtype = config.dtype.clone();
        let mut model = Self::empty(config);
        model.load_state_dict(&state::cast_all(sd, dtype), "").context(StateSnafu)?;
        Ok(model)
    }
}

impl HasStateDict for ModernBertForMaskedLm {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = self.bert.state_dict(prefix);
        let head_sd = self.head.state_dict(prefix);
        // The backbone already emits the tied embedding weight under
        // `model.embeddings.tok_embeddings.weight`; drop the head's duplicate
        // `decoder.weight` copy so the composite round-trips to the published
        // (tied) checkpoint shape — exactly one `(V, D)` weight, under the
        // embedding key.
        let decoder_key = prefixed(prefix, "decoder.weight");
        for (k, v) in head_sd {
            if k != decoder_key {
                sd.insert(k, v);
            }
        }
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.bert.load_state_dict(sd, prefix)?;
        self.head.load_state_dict(sd, prefix)?;
        Ok(())
    }
}
