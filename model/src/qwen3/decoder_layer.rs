//! Qwen3 pre-norm decoder layer.
//!
//! Standard pre-norm residual structure:
//! ```text
//! h = x + attn(input_layernorm(x))
//! y = h + mlp(post_attention_layernorm(h))
//! ```

use svod_tensor::Tensor;

use crate::state::{self, HasStateDict, StateDict};

use super::attention::Qwen3Attention;
use super::error::Result;

use super::feed_forward::Qwen3MLP;
use super::rms_norm::RmsNormWeights;
use super::rotary::RotaryTable;

#[derive(Clone)]
pub struct Qwen3DecoderLayer {
    pub input_layernorm: RmsNormWeights,
    pub attention: Qwen3Attention,
    pub post_attention_layernorm: RmsNormWeights,
    pub mlp: Qwen3MLP,
}

impl Qwen3DecoderLayer {
    pub fn empty(config: &super::Qwen3Config) -> Self {
        let dtype = config.dtype.clone();
        Self {
            input_layernorm: RmsNormWeights::empty(config.hidden_size, config.rms_norm_eps, dtype.clone()),
            attention: Qwen3Attention::empty(
                config.hidden_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                config.rms_norm_eps,
                dtype.clone(),
            ),
            post_attention_layernorm: RmsNormWeights::empty(config.hidden_size, config.rms_norm_eps, dtype.clone()),
            mlp: Qwen3MLP::empty(config.hidden_size, config.intermediate_size, dtype),
        }
    }

    pub fn forward(&self, x: &Tensor, rotary: &RotaryTable, padding_mask: Option<&Tensor>) -> Result<Tensor> {
        let normed = self.input_layernorm.apply(x)?;
        let delta = self.attention.forward(&normed, rotary, padding_mask)?;
        let h = x.try_add(&delta)?;

        let normed = self.post_attention_layernorm.apply(&h)?;
        let delta = self.mlp.forward(&normed)?;
        Ok(h.try_add(&delta)?)
    }
}

impl HasStateDict for Qwen3DecoderLayer {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.extend(self.input_layernorm.state_dict(&format!("{prefix}.input_layernorm")));
        sd.extend(self.attention.state_dict(&format!("{prefix}.self_attn")));
        sd.extend(self.post_attention_layernorm.state_dict(&format!("{prefix}.post_attention_layernorm")));
        sd.extend(self.mlp.state_dict(&format!("{prefix}.mlp")));
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.input_layernorm.load_state_dict(sd, &format!("{prefix}.input_layernorm"))?;
        self.attention.load_state_dict(sd, &format!("{prefix}.self_attn"))?;
        self.post_attention_layernorm.load_state_dict(sd, &format!("{prefix}.post_attention_layernorm"))?;
        self.mlp.load_state_dict(sd, &format!("{prefix}.mlp"))?;
        Ok(())
    }
}
