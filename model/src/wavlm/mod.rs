//! WavLM speech-representation backbone — a 7-layer Conv1d feature extractor +
//! N-layer Transformer with gated relative-position attention, supporting the
//! per-layer pruning structure used by `wavlm-large-s80-md-v2`.

mod attention;
mod config;
mod encoder;
mod encoder_layer;
mod error;
mod feature_extractor;
mod feed_forward;
mod jit;
mod model;
mod pos_conv;

use svod_dtype::DType;
use svod_tensor::nn::LayerNorm;

use crate::init::{ones, zeros};

pub use attention::{GatedRelPosAttention, bucket_index_tensor, compute_bucket_indices, compute_position_bias};
pub use config::{ConvLayerConfig, ExtractorMode, WavLmConfig, wavlm_base, wavlm_large, wavlm_large_s80_md};
pub use encoder::Encoder;
pub use encoder_layer::EncoderLayer;
pub use error::{Error, Result};
pub use feature_extractor::{BlockNorm, ConvLayerBlock, FeatureExtractor, GroupNorm};
pub use feed_forward::FeedForward;
pub use jit::WavLmJit;
pub use model::WavLm;
pub(crate) use model::drop_inert_keys;
pub use pos_conv::ConvolutionalPositionalEmbedding;

/// The identity-affine f32 [`LayerNorm`] every WavLM sub-module starts from:
/// PyTorch's default `eps` over the last axis.
pub(crate) fn layer_norm(size: usize) -> LayerNorm {
    LayerNorm::new(ones(&[size], DType::Float32), Some(zeros(&[size], DType::Float32)), 1e-5)
}
