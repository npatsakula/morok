//! Root Qwen3 decoder backbone: token embeddings → RoPE decoder stack →
//! final RMSNorm. Exposes `forward` returning the `(B, L, D)` last-hidden-state.
//!
//! Loads from a HuggingFace `model.safetensors` checkpoint with bare keys
//! (`embed_tokens.weight`, `layers.N.*`, `norm.weight` — no `model.` prefix).
//! Compute dtype is `config.dtype` (bf16 by default, f32 for CPU parity).

use std::path::Path;

use snafu::{OptionExt, ResultExt};
use svod_ir::SInt;
use svod_tensor::{BoundVariable, Tensor};

use crate::state::{self, HasStateDict, StateDict};

use super::config::Qwen3Config;
use super::decoder_layer::Qwen3DecoderLayer;
use super::embeddings::Qwen3Embeddings;
use super::error::{HubSnafu, Result, StateSnafu, SymbolicShapeSnafu, TensorSnafu};
use super::rms_norm::RmsNormWeights;
use super::rotary::RotaryTable;

#[derive(Clone)]
pub struct Qwen3Model {
    pub config: Qwen3Config,
    pub embeddings: Qwen3Embeddings,
    pub layers: Vec<Qwen3DecoderLayer>,
    pub norm: RmsNormWeights,
}

impl Qwen3Model {
    pub fn empty(config: Qwen3Config) -> Self {
        let dtype = config.dtype.clone();
        let embeddings = Qwen3Embeddings::empty(config.vocab_size, config.hidden_size, dtype.clone());
        let layers = (0..config.num_hidden_layers).map(|_| Qwen3DecoderLayer::empty(&config)).collect();
        let norm = RmsNormWeights::empty(config.hidden_size, config.rms_norm_eps, dtype);
        Self { config, embeddings, layers, norm }
    }

    /// Eager forward: `input_ids` `(B, L)` + optional `padding_mask` `(B, L)`
    /// bool → last-hidden-state `(B, L, D)`.
    pub fn forward(&self, input_ids: &Tensor, padding_mask: Option<&Tensor>) -> Result<Tensor> {
        let x = self.embeddings.forward(input_ids)?;
        let shape = x.shape().context(TensorSnafu)?;
        let b_dim = shape[0].clone();
        let seq_len: usize = shape[1].as_const().context(SymbolicShapeSnafu { what: "qwen3 forward seq_len" })?;

        // Build the rotary table once — shared across all layers.
        let rotary =
            RotaryTable::new(self.config.rope_theta, seq_len, self.config.head_dim, self.config.dtype.clone())?;

        // Build SDPA key-axis mask: bool (B,1,1,L), True = masked out.
        let mask_4d = match padding_mask {
            Some(m) => {
                let m2 = m.try_reshape([b_dim.clone(), SInt::from(seq_len)]).context(TensorSnafu)?;
                let inverted = m2.logical_not().context(TensorSnafu)?;
                Some(inverted.try_unsqueeze(1).context(TensorSnafu)?.try_unsqueeze(1).context(TensorSnafu)?)
            }
            None => None,
        };

        let mut h = x;
        for layer in &self.layers {
            h = layer.forward(&h, &rotary, mask_4d.as_ref())?;
        }
        self.norm.apply(&h)
    }

    /// JIT-path variant: shrinks the leading batch dim to the live value.
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

    pub fn from_hub(model_id: &str, mut config: Qwen3Config) -> Result<Self> {
        Self::from_hub_with_revision(model_id, "main", &mut config)
    }

    pub fn from_hub_with_revision(model_id: &str, revision: &str, config: &mut Qwen3Config) -> Result<Self> {
        let repo = crate::hub::HubRepo::open(model_id, revision).context(HubSnafu)?;
        let cfg_path = repo.get("config.json").context(HubSnafu)?;
        let parsed = Qwen3Config::from_json(&cfg_path)?;
        config.merge_structural_from(&parsed);

        let dir = crate::qwen3::download_safetensors(&repo)?;
        Self::from_safetensors_dir(&dir, config.clone())
    }

    pub fn from_safetensors(path: &Path, config: Qwen3Config) -> Result<Self> {
        let sd = state::load_safetensors(path).context(StateSnafu)?;
        Self::from_state_dict(&sd, config)
    }

    /// Load from a directory containing `model.safetensors` (single-file) or
    /// `model.safetensors.index.json` + shards (multi-shard).
    pub fn from_safetensors_dir(dir: &Path, config: Qwen3Config) -> Result<Self> {
        let sd = state::load_safetensors_dir(dir).context(StateSnafu)?;
        Self::from_state_dict(&sd, config)
    }

    pub fn from_state_dict(sd: &StateDict, config: Qwen3Config) -> Result<Self> {
        let dtype = config.dtype.clone();
        let mut model = Self::empty(config);
        model.load_state_dict(&state::cast_all(sd, dtype), "").context(StateSnafu)?;
        Ok(model)
    }
}

impl HasStateDict for Qwen3Model {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.extend(self.embeddings.state_dict(&prefix_or(prefix, "embed_tokens")));
        for (i, layer) in self.layers.iter().enumerate() {
            sd.extend(layer.state_dict(&prefixed_index(prefix, "layers", i)));
        }
        sd.extend(self.norm.state_dict(&prefix_or(prefix, "norm")));
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.embeddings.load_state_dict(sd, &prefix_or(prefix, "embed_tokens"))?;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.load_state_dict(sd, &prefixed_index(prefix, "layers", i))?;
        }
        self.norm.load_state_dict(sd, &prefix_or(prefix, "norm"))?;
        Ok(())
    }
}

fn prefix_or(prefix: &str, suffix: &str) -> String {
    if prefix.is_empty() { suffix.to_string() } else { format!("{prefix}.{suffix}") }
}

fn prefixed_index(prefix: &str, name: &str, i: usize) -> String {
    if prefix.is_empty() { format!("{name}.{i}") } else { format!("{prefix}.{name}.{i}") }
}
