//! WavLM transformer encoder layer.
//!
//! Direct port of `EncoderLayer` from `components.py:879-942`.
//!
//! The Python `wavlm_model` factory inverts the config flag
//! (`layer_norm_first = not encoder_layer_norm_first`), so:
//!
//! - **Post-norm** (`layer_norm_first = false`, WavLM Large): LayerNorm is
//!   applied *after* each residual add.
//!   ```text
//!   if attention is Some:
//!       residual = x
//!       x        = LayerNorm(residual + Dropout(Attention(x, pos_bias)))
//!   if feed_forward is Some:
//!       x = FinalLayerNorm(x + FF(x))
//!   ```
//!
//! - **Pre-norm** (`layer_norm_first = true`, WavLM Base): LayerNorm is
//!   applied *before* each sub-layer.
//!   ```text
//!   if attention is Some:
//!       residual = x
//!       x        = residual + Dropout(Attention(LayerNorm(x), pos_bias))
//!   if feed_forward is Some:
//!       x = x + FF(FinalLayerNorm(x))
//!   ```
//!
//! Both `layer_norm` and `final_layer_norm` are *always* present in the state
//! dict (Python instantiates them unconditionally in `__init__`); only the
//! `attention` and `feed_forward` sub-modules are optional.

use svod_tensor::Tensor;

use crate::state::{self, HasStateDict, StateDict};

use super::attention::GatedRelPosAttention;
use super::config::WavLmConfig;
use super::error::Result;

use super::feed_forward::FeedForwardWeights;
use super::layer_norm::LayerNormWeights;

#[derive(Clone)]
pub struct EncoderLayer {
    pub embed_dim: usize,
    /// Runtime `layer_norm_first` flag — the *inverted* config value.
    /// `true` = pre-norm, `false` = post-norm.
    pub layer_norm_first: bool,
    pub layer_norm: LayerNormWeights,
    pub attention: Option<GatedRelPosAttention>,
    pub final_layer_norm: LayerNormWeights,
    pub feed_forward: Option<FeedForwardWeights>,
}

impl EncoderLayer {
    pub fn empty(config: &WavLmConfig, layer_index: usize) -> Self {
        let embed_dim = config.encoder_embed_dim;
        let use_attn =
            config.encoder_use_attention[layer_index] && !config.encoder_remaining_heads[layer_index].is_empty();
        let attention = use_attn.then(|| GatedRelPosAttention::empty(config, layer_index));
        let feed_forward = config.encoder_use_feed_forward[layer_index]
            .then(|| FeedForwardWeights::empty(embed_dim, config.encoder_ff_interm_features[layer_index]));
        Self {
            embed_dim,
            layer_norm_first: config.encoder_layer_norm_first,
            layer_norm: LayerNormWeights::empty(embed_dim),
            attention,
            final_layer_norm: LayerNormWeights::empty(embed_dim),
            feed_forward,
        }
    }

    pub fn forward(&self, x: &Tensor, position_bias: Option<&Tensor>) -> Result<Tensor> {
        if self.layer_norm_first {
            self.forward_pre_norm(x, position_bias)
        } else {
            self.forward_post_norm(x, position_bias)
        }
    }

    fn forward_pre_norm(&self, x: &Tensor, position_bias: Option<&Tensor>) -> Result<Tensor> {
        let mut x = x.clone();
        if let Some(attn) = &self.attention {
            let residual = x.clone();
            let normed = self.layer_norm.apply(&x)?;
            let bias = position_bias.expect("attention layer requires position_bias");
            let delta = attn.forward(&normed, bias)?;
            x = residual.try_add(&delta)?;
        }
        if let Some(ff) = &self.feed_forward {
            let normed = self.final_layer_norm.apply(&x)?;
            let delta = ff.forward(&normed)?;
            x = x.try_add(&delta)?;
        }
        Ok(x)
    }

    fn forward_post_norm(&self, x: &Tensor, position_bias: Option<&Tensor>) -> Result<Tensor> {
        let mut x = x.clone();
        if let Some(attn) = &self.attention {
            let residual = x.clone();
            let bias = position_bias.expect("attention layer requires position_bias");
            let delta = attn.forward(&x, bias)?;
            x = residual.try_add(&delta)?;
        }
        x = self.layer_norm.apply(&x)?;
        if let Some(ff) = &self.feed_forward {
            let residual = x.clone();
            let delta = ff.forward(&x)?;
            x = residual.try_add(&delta)?;
        }
        x = self.final_layer_norm.apply(&x)?;
        Ok(x)
    }
}

impl HasStateDict for EncoderLayer {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = self.layer_norm.state_dict(&format!("{prefix}.layer_norm"));
        if let Some(attn) = &self.attention {
            sd.extend(attn.state_dict(&format!("{prefix}.attention")));
        }
        sd.extend(self.final_layer_norm.state_dict(&format!("{prefix}.final_layer_norm")));
        if let Some(ff) = &self.feed_forward {
            sd.extend(ff.state_dict(&format!("{prefix}.feed_forward")));
        }
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.layer_norm.load_state_dict(sd, &format!("{prefix}.layer_norm"))?;
        if let Some(attn) = self.attention.as_mut() {
            attn.load_state_dict(sd, &format!("{prefix}.attention"))?;
        }
        self.final_layer_norm.load_state_dict(sd, &format!("{prefix}.final_layer_norm"))?;
        if let Some(ff) = self.feed_forward.as_mut() {
            ff.load_state_dict(sd, &format!("{prefix}.feed_forward"))?;
        }
        Ok(())
    }
}
