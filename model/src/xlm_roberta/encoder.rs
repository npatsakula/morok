//! XLM-RoBERTa transformer encoder: a stack of post-norm [`EncoderLayer`]s.

use svod_ir::SInt;
use svod_tensor::Tensor;

use crate::state::{self, HasStateDict, StateDict};

use super::config::XlmRobertaConfig;
use super::encoder_layer::EncoderLayer;
use super::error::Result;

#[derive(Clone)]
pub struct XlmRobertaEncoder {
    pub config: XlmRobertaConfig,
    pub layers: Vec<EncoderLayer>,
}

impl XlmRobertaEncoder {
    pub fn empty(config: &XlmRobertaConfig) -> Self {
        let layers = (0..config.num_hidden_layers).map(|_| EncoderLayer::empty(config)).collect();
        Self { config: config.clone(), layers }
    }

    /// Run the encoder stack. `x`: `(B, L, D)` → `(B, L, D)`.
    pub fn forward(&self, x: &Tensor, padding_mask: Option<&Tensor>) -> Result<Tensor> {
        let seq_len = x.dim_const(1)?;

        let mask_4d = match padding_mask {
            Some(m) => {
                let b_dim = x.dim(0)?;
                let m2 = m.try_reshape([b_dim, SInt::from(seq_len)])?;
                let inverted = m2.logical_not()?;
                Some(inverted.try_unsqueeze(1)?.try_unsqueeze(1)?)
            }
            None => None,
        };

        let mut h = x.clone();
        for layer in &self.layers {
            h = layer.forward(&h, mask_4d.as_ref())?;
        }
        Ok(h)
    }
}

impl HasStateDict for XlmRobertaEncoder {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        for (i, layer) in self.layers.iter().enumerate() {
            sd.extend(layer.state_dict(&format!("{prefix}.layer.{i}")));
        }
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.load_state_dict(sd, &format!("{prefix}.layer.{i}"))?;
        }
        Ok(())
    }
}
