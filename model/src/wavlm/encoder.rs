//! WavLM transformer encoder: feature projection → conv positional embedding →
//! stack of [`EncoderLayer`] → final LayerNorm.
//!
//! Mirrors `Encoder` + `Transformer` from `components.py:960-1160` plus
//! `FeatureProjection` at 272-315. Notable design choices:
//!
//! - The shared bucketed `rel_attn_embed: Embedding(num_buckets, total_num_heads)`
//!   is owned by this encoder (one per backbone). The upstream checkpoint
//!   stores it under `transformer.layers.0.attention.rel_attn_embed.weight`;
//!   we read/write that key but the runtime tensor lives on the encoder so
//!   every layer sees the same `position_bias`.
//! - **Final LayerNorm placement**: the Python `wavlm_model` factory passes
//!   `not layer_norm_first` to the internal `Transformer`, so
//!   `Transformer.layer_norm_first = False` for our config (factory
//!   `layer_norm_first=True`). Python's `forward` applies the final LN
//!   after the loop (`if not self.layer_norm_first`), but
//!   `get_intermediate_outputs` does **not**. Our `extract_features`
//!   mirrors `get_intermediate_outputs`, so the last intermediate is
//!   **pre-LN**. Call [`Encoder::final_layer_norm_apply`] to get the
//!   post-LN output.

use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::{Layer, LayerNorm, Linear, Module};

use crate::init::{Bias, fan_in_uniform, layer_norm, linear};

use super::attention::compute_position_bias;
use super::config::WavLmConfig;
use super::encoder_layer::EncoderLayer;
use super::error::Result;
use super::pos_conv::ConvolutionalPositionalEmbedding;

#[derive(Clone, Module)]
pub struct Encoder {
    /// Runtime `layer_norm_first` — the *inverted* config value. `true` =
    /// pre-norm (WavLM Base), `false` = post-norm (WavLM Large).
    pub layer_norm_first: bool,
    pub num_buckets: usize,
    pub max_distance: usize,

    // FeatureProjection: LayerNorm + Linear + Dropout (dropout no-op in eval).
    #[module(key = "feature_projection.layer_norm")]
    pub feature_projection_norm: LayerNorm,
    #[module(key = "feature_projection.projection")]
    pub feature_projection: Linear,

    #[module(key = "transformer.pos_conv_embed")]
    pub pos_conv_embed: ConvolutionalPositionalEmbedding,
    #[module(key = "transformer.layer_norm")]
    pub layer_norm: LayerNorm,
    #[module(key = "transformer.layers")]
    pub layers: Vec<EncoderLayer>,

    /// Shared bucketed relative-position table; in the saved checkpoint it
    /// lives under layer 0's attention.
    #[module(key = "transformer.layers.0.attention.rel_attn_embed.weight")]
    pub rel_attn_embed: Tensor,
}

impl Encoder {
    pub fn empty(config: &WavLmConfig) -> Self {
        let embed_dim = config.encoder_embed_dim;
        let extractor_out = config.extractor_out_dim();
        let total_num_heads = config.encoder_total_num_heads[0];
        Self {
            layer_norm_first: !config.encoder_layer_norm_first,
            num_buckets: config.encoder_num_buckets,
            max_distance: config.encoder_max_distance,

            feature_projection_norm: layer_norm(extractor_out, DType::Float32),
            feature_projection: linear(extractor_out, embed_dim, Bias::Zero, DType::Float32),

            pos_conv_embed: ConvolutionalPositionalEmbedding::empty(
                embed_dim,
                config.encoder_pos_conv_kernel,
                config.encoder_pos_conv_groups,
            ),
            layer_norm: layer_norm(embed_dim, DType::Float32),
            layers: (0..config.encoder_num_layers).map(|i| EncoderLayer::empty(config, i)).collect(),

            rel_attn_embed: fan_in_uniform(
                &[config.encoder_num_buckets, total_num_heads],
                total_num_heads,
                DType::Float32,
            ),
        }
    }

    /// Run feature projection, add positional encoding, then the transformer
    /// loop. Returns `num_layers + 1` intermediates matching Python's
    /// `Transformer.get_intermediate_outputs`: index 0 is the
    /// post-feature-projection-plus-pos-conv input, indices `1..=N` are the
    /// per-layer outputs. **No final LayerNorm is applied.** Python's
    /// `forward` (separate method) applies it; `get_intermediate_outputs`
    /// does not — and our `extract_features` mirrors the latter, which is
    /// what the published Python dump uses.
    pub fn extract_features(&self, features: &Tensor) -> Result<Vec<Tensor>> {
        let l = features.dim_const(1)?;

        // FeatureProjection: LN → Linear → (dropout no-op)
        let h = self.feature_projection.forward(&self.feature_projection_norm.forward(features)?)?;

        // Pos-conv add.
        let mut x = h.try_add(&self.pos_conv_embed.forward(&h)?)?;

        // Pre-norm: _preprocess applies LayerNorm after pos-conv.
        if self.layer_norm_first {
            x = self.layer_norm.forward(&x)?;
        }

        // Compute the un-gated position bias once for this L and share it
        // lazily across all layers. Its `.contiguous()` marks a shared-buffer
        // boundary, so the scheduler materializes the small L*L lookup once and
        // every layer's attention graph reads that buffer.
        let pb = compute_position_bias(&self.rel_attn_embed, l, l, self.num_buckets, self.max_distance)?;
        let mut intermediates = Vec::with_capacity(self.layers.len() + 1);
        intermediates.push(x.clone());
        for layer in &self.layers {
            x = layer.forward(&x, Some(&pb))?;
            intermediates.push(x.clone());
        }
        Ok(intermediates)
    }

    /// Stand-alone final LayerNorm matching Python's `Transformer.forward`
    /// (which applies `self.layer_norm` after the last block). Exposed for
    /// downstream consumers that want the "post-norm output" instead of the
    /// raw last intermediate.
    pub fn final_layer_norm_apply(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.layer_norm.forward(x)?)
    }
}
