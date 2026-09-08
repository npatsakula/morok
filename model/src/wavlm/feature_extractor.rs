//! WavLM convolutional feature extractor: a stack of 7 strided `Conv1d` blocks
//! that turn a raw 16 kHz waveform `(B, samples)` into framed features
//! `(B, T, C_out)`. Mirrors `ConvLayerBlock` / `_get_feature_extractor` from
//! `submodules/DiariZen/diarizen/models/module/wav2vec2/components.py`.
//!
//! Two normalization modes, both stored under the block's `layer_norm.*` keys:
//! - [`ExtractorMode::GroupNorm`]: only block 0 carries a per-channel norm
//!   (`num_groups == out_channels`, i.e. instance-norm over time, in `NCT`).
//! - [`ExtractorMode::LayerNorm`]: every block normalizes over the channel
//!   axis between conv and GELU.
//!
//! Each block: `Conv1d → Norm? → GELU`. Block 0 takes input channels = 1
//! (the raw mono waveform is unsqueezed at the channel axis).

use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::{Conv1d, Layer, LayerNorm, Module};

use crate::init::{fan_in_uniform, ones, zeros};

use super::config::{ConvLayerConfig, ExtractorMode, WavLmConfig};
use super::error::Result;

const NORM_EPS: f64 = 1e-5;

/// Group normalization with `num_groups` groups over the channel axis of an
/// `NCT` input. State-dict keys: `weight`, `bias`.
#[derive(Clone, Module)]
pub struct GroupNorm {
    pub weight: Tensor,
    pub bias: Tensor,
    pub num_groups: usize,
    pub eps: f64,
}

impl Layer for GroupNorm {
    fn forward(&self, x: &Tensor) -> svod_tensor::error::Result<Tensor> {
        x.group_norm().scale(&self.weight).bias(&self.bias).num_groups(self.num_groups).eps(self.eps).call()
    }
}

/// A feature-extractor block's normalization. Both variants carry `weight` and
/// `bias` at the block's `layer_norm` prefix, matching upstream.
#[derive(Clone, Module)]
pub enum BlockNorm {
    Layer(LayerNorm),
    Group(GroupNorm),
}

impl BlockNorm {
    fn layer(channels: usize) -> Self {
        Self::Layer(LayerNorm::new(
            ones(&[channels], DType::Float32),
            Some(zeros(&[channels], DType::Float32)),
            NORM_EPS,
        ))
    }

    fn group(channels: usize) -> Self {
        Self::Group(GroupNorm {
            weight: ones(&[channels], DType::Float32),
            bias: zeros(&[channels], DType::Float32),
            num_groups: channels,
            eps: NORM_EPS,
        })
    }

    /// Normalize an `NCT` activation. The LayerNorm variant normalizes over the
    /// channel axis, so it round-trips through `NTC`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(match self {
            Self::Group(norm) => norm.forward(x)?,
            Self::Layer(norm) => norm.forward(&x.try_permute(&[0, 2, 1])?)?.try_permute(&[0, 2, 1])?,
        })
    }
}

/// One feature-extractor conv block. The norm shape is `out_channels`; the
/// conv weight shape is `(out_channels, in_channels, kernel_size)`.
#[derive(Clone, Module)]
pub struct ConvLayerBlock {
    pub conv: Conv1d,
    /// `None` for blocks that carry no normalization (GroupNorm mode, blocks > 0).
    #[module(key = "layer_norm")]
    pub norm: Option<BlockNorm>,
    /// Kept alongside the weight so the frame-count arithmetic stays infallible.
    pub kernel_size: usize,
}

impl ConvLayerBlock {
    pub fn empty(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        has_bias: bool,
        norm: Option<BlockNorm>,
    ) -> Self {
        let fan_in = in_channels * kernel_size;
        let weight = fan_in_uniform(&[out_channels, in_channels, kernel_size], fan_in, DType::Float32);
        let bias = has_bias.then(|| zeros(&[out_channels], DType::Float32));
        Self { conv: Conv1d::new(weight, bias).with_stride(stride), norm, kernel_size }
    }

    /// Forward in `NCT` layout: input `(B, C_in, T_in)` → output `(B, C_out, T_out)`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.conv.forward(x)?;
        let y = match &self.norm {
            Some(norm) => norm.forward(&y)?,
            None => y,
        };
        // Exact (erf-based) GELU matches PyTorch's `nn.functional.gelu`.
        Ok(y.gelu_exact()?)
    }
}

// ---------------------------------------------------------------------------
// FeatureExtractor
// ---------------------------------------------------------------------------

#[derive(Clone, Module)]
pub struct FeatureExtractor {
    #[module(key = "conv_layers")]
    pub blocks: Vec<ConvLayerBlock>,
    #[module(optional)]
    pub dummy_weight: Option<Tensor>,
}

impl FeatureExtractor {
    pub fn empty(config: &WavLmConfig) -> Self {
        let layers: &[ConvLayerConfig] = &config.extractor_conv_layer_config;
        let has_bias = config.extractor_conv_bias;
        let blocks: Vec<ConvLayerBlock> = layers
            .iter()
            .enumerate()
            .scan(1usize, |in_ch, (i, &(out_ch, k, stride))| {
                let norm = match config.extractor_mode {
                    ExtractorMode::LayerNorm => Some(BlockNorm::layer(out_ch)),
                    ExtractorMode::GroupNorm if i == 0 => Some(BlockNorm::group(out_ch)),
                    ExtractorMode::GroupNorm => None,
                };
                let block = ConvLayerBlock::empty(*in_ch, out_ch, k, stride, has_bias, norm);
                *in_ch = out_ch;
                Some(block)
            })
            .collect();
        Self { blocks, dummy_weight: None }
    }

    /// Forward on `(B, samples)` → `(B, T, C_out)`. Unsqueezes the channel
    /// dim, runs each block in `NCT`, then transposes the result to `NTC` for
    /// downstream consumers (matches upstream `FeatureExtractor.forward`).
    pub fn forward(&self, waveform: &Tensor) -> Result<Tensor> {
        let mut x = waveform.try_unsqueeze(1)?; // (B, 1, samples)
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        // (B, C_out, T) → (B, T, C_out)
        let x = x.try_permute(&[0, 2, 1])?;
        match &self.dummy_weight {
            Some(dw) => Ok(x.try_mul(dw)?),
            None => Ok(x),
        }
    }

    /// Cumulative downsampling factor of all blocks (product of strides).
    pub fn total_stride(&self) -> usize {
        self.blocks.iter().map(|b| b.conv.stride).product()
    }

    /// Output time-frames given an input sample count: applies each block's
    /// `(L - k) // stride + 1` rule (assumes `padding=0`, `dilation=1`).
    pub fn num_frames(&self, num_samples: usize) -> usize {
        self.blocks
            .iter()
            .fold(num_samples, |t, b| if t < b.kernel_size { 0 } else { (t - b.kernel_size) / b.conv.stride + 1 })
    }
}
