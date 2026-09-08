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

use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::{Layer, LayerNorm, Module};

use crate::init::layer_norm;

use super::attention::GatedRelPosAttention;
use super::config::WavLmConfig;
use super::error::Result;
use super::feed_forward::FeedForward;

#[derive(Clone, Module)]
pub struct EncoderLayer {
    /// Runtime `layer_norm_first` flag — the *inverted* config value.
    /// `true` = pre-norm, `false` = post-norm.
    pub layer_norm_first: bool,
    pub layer_norm: LayerNorm,
    pub attention: Option<GatedRelPosAttention>,
    pub final_layer_norm: LayerNorm,
    pub feed_forward: Option<FeedForward>,
}

impl EncoderLayer {
    pub fn empty(config: &WavLmConfig, layer_index: usize) -> Self {
        let embed_dim = config.encoder_embed_dim;
        let use_attn =
            config.encoder_use_attention[layer_index] && !config.encoder_remaining_heads[layer_index].is_empty();
        Self {
            layer_norm_first: config.encoder_layer_norm_first,
            layer_norm: layer_norm(embed_dim, DType::Float32),
            attention: use_attn.then(|| GatedRelPosAttention::empty(config, layer_index)),
            final_layer_norm: layer_norm(embed_dim, DType::Float32),
            feed_forward: config.encoder_use_feed_forward[layer_index]
                .then(|| FeedForward::empty(embed_dim, config.encoder_ff_interm_features[layer_index])),
        }
    }

    pub fn forward(&self, x: &Tensor, position_bias: Option<&Tensor>) -> Result<Tensor> {
        if self.layer_norm_first {
            self.forward_pre_norm(x, position_bias)
        } else {
            self.forward_post_norm(x, position_bias)
        }
    }

    fn attend(&self, x: &Tensor, position_bias: Option<&Tensor>) -> Result<Tensor> {
        let attn = self.attention.as_ref().expect("caller checked `attention.is_some()`");
        attn.forward(x, position_bias.expect("attention layer requires position_bias"))
    }

    fn forward_pre_norm(&self, x: &Tensor, position_bias: Option<&Tensor>) -> Result<Tensor> {
        let mut x = x.clone();
        if self.attention.is_some() {
            let delta = self.attend(&self.layer_norm.forward(&x)?, position_bias)?;
            x = x.try_add(&delta)?;
        }
        if let Some(ff) = &self.feed_forward {
            let delta = ff.forward(&self.final_layer_norm.forward(&x)?)?;
            x = x.try_add(&delta)?;
        }
        Ok(x)
    }

    fn forward_post_norm(&self, x: &Tensor, position_bias: Option<&Tensor>) -> Result<Tensor> {
        let mut x = x.clone();
        if self.attention.is_some() {
            let delta = self.attend(&x, position_bias)?;
            x = x.try_add(&delta)?;
        }
        x = self.layer_norm.forward(&x)?;
        if let Some(ff) = &self.feed_forward {
            let delta = ff.forward(&x)?;
            x = x.try_add(&delta)?;
        }
        Ok(self.final_layer_norm.forward(&x)?)
    }
}
