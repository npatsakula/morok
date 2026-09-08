//! XLM-RoBERTa transformer encoder: a stack of post-norm [`EncoderLayer`]s.

use svod_tensor::Tensor;
use svod_tensor::nn::Module;

use super::config::XlmRobertaConfig;
use super::encoder_layer::EncoderLayer;
use super::error::Result;

#[derive(Clone, Module)]
pub struct XlmRobertaEncoder {
    #[module(skip)]
    pub config: XlmRobertaConfig,
    #[module(key = "layer")]
    pub layers: Vec<EncoderLayer>,
}

impl XlmRobertaEncoder {
    pub fn empty(config: &XlmRobertaConfig) -> Self {
        let layers = (0..config.num_hidden_layers).map(|_| EncoderLayer::empty(config)).collect();
        Self { config: config.clone(), layers }
    }

    /// Run the encoder stack. `x`: `(B, L, D)` → `(B, L, D)`.
    /// `padding_mask`: optional bool `(B, L)` where `true` = real token.
    pub fn forward(&self, x: &Tensor, padding_mask: Option<&Tensor>) -> Result<Tensor> {
        let mut h = x.clone();
        for layer in &self.layers {
            h = layer.forward(&h, padding_mask)?;
        }
        Ok(h)
    }
}
