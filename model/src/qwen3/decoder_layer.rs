//! Qwen3 pre-norm decoder layer.
//!
//! Standard pre-norm residual structure:
//! ```text
//! h = x + attn(input_layernorm(x))
//! y = h + mlp(post_attention_layernorm(h))
//! ```

use svod_tensor::Tensor;
use svod_tensor::nn::{Layer, Module, RmsNorm};

use super::attention::Qwen3Attention;
use super::error::Result;

use super::feed_forward::Qwen3MLP;

#[derive(Clone, Module)]
pub struct Qwen3DecoderLayer {
    pub input_layernorm: RmsNorm,
    #[module(key = "self_attn")]
    pub attention: Qwen3Attention,
    pub post_attention_layernorm: RmsNorm,
    pub mlp: Qwen3MLP,
}

impl Qwen3DecoderLayer {
    pub fn empty(config: &super::Qwen3Config) -> Self {
        let dtype = config.dtype.clone();
        Self {
            input_layernorm: RmsNorm::with_dims(config.hidden_size, config.rms_norm_eps, dtype.clone()),
            attention: Qwen3Attention::empty(
                config.hidden_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                config.rms_norm_eps,
                dtype.clone(),
            ),
            post_attention_layernorm: RmsNorm::with_dims(config.hidden_size, config.rms_norm_eps, dtype.clone()),
            mlp: Qwen3MLP::empty(config.hidden_size, config.intermediate_size, dtype),
        }
    }

    pub fn forward(&self, x: &Tensor, rope: &(Tensor, Tensor), padding_mask: Option<&Tensor>) -> Result<Tensor> {
        let normed = self.input_layernorm.forward(x)?;
        let h = x.try_add(&self.attention.forward(&normed, rope, padding_mask)?)?;

        let delta = self.mlp.forward(&self.post_attention_layernorm.forward(&h)?)?;
        Ok(h.try_add(&delta)?)
    }
}
