//! ModernBERT pre-norm encoder layer.
//!
//! Direct port of `FlexBertUnpadPreNormLayer` (padded path). Standard pre-norm:
//! ```text
//! h = x + attn(attn_norm(x))      // attn_norm is Identity for layer 0
//! y = h + mlp(mlp_norm(h))
//! ```
//!
//! `skip_first_prenorm = true` makes layer 0's `attn_norm` an identity (its
//! state-dict key is absent), so `attn_norm` is `Option` and `None` for layer 0.

use svod_tensor::Tensor;

use crate::state::{self, HasStateDict, StateDict};

use super::attention::ModernBertAttention;
use super::error::Result;

use super::mlp::ModernBertGlu;
use super::normalization::LayerNormWeights;
use super::rotary::RotaryTable;

#[derive(Clone)]
pub struct EncoderLayer {
    /// `None` for layer 0 (`skip_first_prenorm`).
    pub attn_norm: Option<LayerNormWeights>,
    pub attention: ModernBertAttention,
    pub mlp_norm: LayerNormWeights,
    pub mlp: ModernBertGlu,
}

impl EncoderLayer {
    pub fn empty(config: &super::ModernBertConfig, layer_id: usize) -> Self {
        let hidden = config.hidden_size;
        let eps = config.layer_norm_eps;
        let dtype = config.dtype.clone();
        let window = if config.is_global_layer(layer_id) { None } else { Some(config.local_window()) };
        let attn_norm = (layer_id != 0).then(|| LayerNormWeights::with_eps(hidden, eps, dtype.clone()));
        Self {
            attn_norm,
            attention: ModernBertAttention::empty(
                hidden,
                config.num_attention_heads,
                config.head_dim(),
                window,
                dtype.clone(),
            ),
            mlp_norm: LayerNormWeights::with_eps(hidden, eps, dtype.clone()),
            mlp: ModernBertGlu::empty(hidden, config.intermediate_size, dtype),
        }
    }

    /// Forward. `x`: `(B, L, D)` → `(B, L, D)`.
    pub fn forward(&self, x: &Tensor, rotary: &RotaryTable, padding_mask: Option<&Tensor>) -> Result<Tensor> {
        let normed = match &self.attn_norm {
            Some(ln) => ln.apply(x)?,
            None => x.clone(),
        };
        let delta = self.attention.forward(&normed, rotary, padding_mask)?;
        let mut h = x.try_add(&delta)?;

        let normed = self.mlp_norm.apply(&h)?;
        let delta = self.mlp.forward(&normed)?;
        h = h.try_add(&delta)?;
        Ok(h)
    }
}

impl HasStateDict for EncoderLayer {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        if let Some(ln) = &self.attn_norm {
            sd.extend(ln.state_dict(&format!("{prefix}.attn_norm")));
        }
        sd.extend(self.attention.state_dict(&format!("{prefix}.attn")));
        sd.extend(self.mlp_norm.state_dict(&format!("{prefix}.mlp_norm")));
        sd.extend(self.mlp.state_dict(&format!("{prefix}.mlp")));
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        if let Some(ln) = self.attn_norm.as_mut() {
            ln.load_state_dict(sd, &format!("{prefix}.attn_norm"))?;
        }
        self.attention.load_state_dict(sd, &format!("{prefix}.attn"))?;
        self.mlp_norm.load_state_dict(sd, &format!("{prefix}.mlp_norm"))?;
        self.mlp.load_state_dict(sd, &format!("{prefix}.mlp"))?;
        Ok(())
    }
}
