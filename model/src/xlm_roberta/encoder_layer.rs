//! XLM-RoBERTa post-norm encoder layer.
//!
//! Standard BERT/RoBERTa post-norm:
//! ```text
//! attn_out = LayerNorm_attn(x + Attention(x))
//! layer_out = LayerNorm_ffn(attn_out + FFN(attn_out))
//! ```

use svod_tensor::Tensor;

use crate::state::{self, HasStateDict, StateDict};

use super::attention::XlmRobertaAttention;
use super::config::XlmRobertaConfig;
use super::error::Result;

use super::feed_forward::FeedForwardWeights;
use super::normalization::LayerNormWeights;

#[derive(Clone)]
pub struct EncoderLayer {
    pub attention: XlmRobertaAttention,
    pub attention_norm: LayerNormWeights,
    pub feed_forward: FeedForwardWeights,
    pub ffn_norm: LayerNormWeights,
}

impl EncoderLayer {
    pub fn empty(config: &XlmRobertaConfig) -> Self {
        let hidden = config.hidden_size;
        let eps = config.layer_norm_eps;
        let dtype = config.dtype.clone();
        Self {
            attention: XlmRobertaAttention::empty(hidden, config.num_attention_heads, config.head_dim(), dtype.clone()),
            attention_norm: LayerNormWeights::with_eps(hidden, eps, dtype.clone()),
            feed_forward: FeedForwardWeights::empty(hidden, config.intermediate_size, dtype.clone()),
            ffn_norm: LayerNormWeights::with_eps(hidden, eps, dtype),
        }
    }

    /// Forward. `x`: `(B, L, D)` → `(B, L, D)`. Post-norm.
    pub fn forward(&self, x: &Tensor, padding_mask: Option<&Tensor>) -> Result<Tensor> {
        let attn_delta = self.attention.forward(x, padding_mask)?;
        let x = x.try_add(&attn_delta)?;
        let x = self.attention_norm.apply(&x)?;

        let ffn_delta = self.feed_forward.forward(&x)?;
        let x = x.try_add(&ffn_delta)?;
        self.ffn_norm.apply(&x)
    }
}

impl HasStateDict for EncoderLayer {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.extend(self.attention.state_dict(&format!("{prefix}.attention")));
        sd.extend(self.attention_norm.state_dict(&format!("{prefix}.attention.output.LayerNorm")));
        sd.extend(self.feed_forward.state_dict(prefix));
        sd.extend(self.ffn_norm.state_dict(&format!("{prefix}.output.LayerNorm")));
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.attention.load_state_dict(sd, &format!("{prefix}.attention"))?;
        self.attention_norm.load_state_dict(sd, &format!("{prefix}.attention.output.LayerNorm"))?;
        self.feed_forward.load_state_dict(sd, prefix)?;
        self.ffn_norm.load_state_dict(sd, &format!("{prefix}.output.LayerNorm"))?;
        Ok(())
    }
}
