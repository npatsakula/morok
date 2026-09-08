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

use crate::init::{fan_in_uniform, zeros};
use crate::state::{self, HasStateDict, StateDict, get_tensor, prefixed};

use super::attention::compute_position_bias;
use super::config::WavLmConfig;
use super::encoder_layer::EncoderLayer;
use super::error::Result;

use super::layer_norm::LayerNormWeights;
use super::pos_conv::ConvolutionalPositionalEmbedding;

#[derive(Clone)]
pub struct Encoder {
    pub embed_dim: usize,
    /// Runtime `layer_norm_first` — the *inverted* config value. `true` =
    /// pre-norm (WavLM Base), `false` = post-norm (WavLM Large).
    pub layer_norm_first: bool,
    pub total_num_heads_at_layer0: usize,
    pub num_buckets: usize,
    pub max_distance: usize,

    // FeatureProjection: LayerNorm + Linear + Dropout (dropout no-op in eval).
    pub feature_projection_norm: LayerNormWeights,
    pub feature_projection_weight: Tensor,
    pub feature_projection_bias: Tensor,

    pub pos_conv_embed: ConvolutionalPositionalEmbedding,
    pub layer_norm: LayerNormWeights,

    /// Shared bucketed relative-position table; in the saved checkpoint it
    /// lives at `transformer.layers.0.attention.rel_attn_embed.weight`.
    pub rel_attn_embed: Tensor,

    pub layers: Vec<EncoderLayer>,
}

impl Encoder {
    pub fn empty(config: &WavLmConfig) -> Self {
        let embed_dim = config.encoder_embed_dim;
        let extractor_out = config.extractor_out_dim();
        let total_num_heads = config.encoder_total_num_heads[0];
        Self {
            embed_dim,
            layer_norm_first: !config.encoder_layer_norm_first,
            total_num_heads_at_layer0: total_num_heads,
            num_buckets: config.encoder_num_buckets,
            max_distance: config.encoder_max_distance,

            feature_projection_norm: LayerNormWeights::empty(extractor_out),
            feature_projection_weight: fan_in_uniform(&[embed_dim, extractor_out], extractor_out, DType::Float32),
            feature_projection_bias: zeros(&[embed_dim], DType::Float32),

            pos_conv_embed: ConvolutionalPositionalEmbedding::empty(
                embed_dim,
                config.encoder_pos_conv_kernel,
                config.encoder_pos_conv_groups,
            ),
            layer_norm: LayerNormWeights::empty(embed_dim),

            rel_attn_embed: fan_in_uniform(
                &[config.encoder_num_buckets, total_num_heads],
                total_num_heads,
                DType::Float32,
            ),

            layers: (0..config.encoder_num_layers).map(|i| EncoderLayer::empty(config, i)).collect(),
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
        let h = self.feature_projection_norm.apply(features)?;
        let h = h.linear().weight(&self.feature_projection_weight).bias(&self.feature_projection_bias).call()?;

        // Pos-conv add.
        let pe = self.pos_conv_embed.forward(&h)?;
        let mut x = h.try_add(&pe)?;

        // Pre-norm: _preprocess applies LayerNorm after pos-conv.
        if self.layer_norm_first {
            x = self.layer_norm.apply(&x)?;
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
        self.layer_norm.apply(x)
    }
}

impl HasStateDict for Encoder {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        let fp = format!("{prefix}.feature_projection");
        sd.extend(self.feature_projection_norm.state_dict(&format!("{fp}.layer_norm")));
        sd.insert(prefixed(&fp, "projection.weight"), self.feature_projection_weight.clone());
        sd.insert(prefixed(&fp, "projection.bias"), self.feature_projection_bias.clone());

        let tr = format!("{prefix}.transformer");
        sd.extend(self.pos_conv_embed.state_dict(&format!("{tr}.pos_conv_embed")));
        sd.extend(self.layer_norm.state_dict(&format!("{tr}.layer_norm")));

        for (i, layer) in self.layers.iter().enumerate() {
            sd.extend(layer.state_dict(&format!("{tr}.layers.{i}")));
        }
        // `rel_attn_embed.weight` lives under layer 0's attention in upstream.
        sd.insert(format!("{tr}.layers.0.attention.rel_attn_embed.weight"), self.rel_attn_embed.clone());
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        let fp = format!("{prefix}.feature_projection");
        self.feature_projection_norm.load_state_dict(sd, &format!("{fp}.layer_norm"))?;
        self.feature_projection_weight = get_tensor(sd, &prefixed(&fp, "projection.weight"))?;
        self.feature_projection_bias = get_tensor(sd, &prefixed(&fp, "projection.bias"))?;

        let tr = format!("{prefix}.transformer");
        self.pos_conv_embed.load_state_dict(sd, &format!("{tr}.pos_conv_embed"))?;
        self.layer_norm.load_state_dict(sd, &format!("{tr}.layer_norm"))?;

        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.load_state_dict(sd, &format!("{tr}.layers.{i}"))?;
        }
        self.rel_attn_embed = get_tensor(sd, &format!("{tr}.layers.0.attention.rel_attn_embed.weight"))?;
        Ok(())
    }
}
