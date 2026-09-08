//! WavLM convolutional feature extractor: a stack of 7 strided `Conv1d` blocks
//! that turn a raw 16 kHz waveform `(B, samples)` into framed features
//! `(B, T, C_out)`. Mirrors `ConvLayerBlock` / `_get_feature_extractor` from
//! `submodules/DiariZen/diarizen/models/module/wav2vec2/components.py`.
//!
//! Two normalization modes:
//! - [`ExtractorMode::GroupNorm`]: only block 0 carries a per-channel norm
//!   (modeled as `num_groups == out_channels`, i.e. instance-norm over time).
//! - [`ExtractorMode::LayerNorm`]: every block carries a `LayerNorm` over the
//!   channel axis between conv and GELU.
//!
//! Each block: `Conv1d → Norm? → GELU`. Block 0 takes input channels = 1
//! (the raw mono waveform is unsqueezed at the channel axis).

use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::init::{fan_in_uniform, ones, zeros};
use crate::state::{self, HasStateDict, StateDict, get_tensor, prefixed};

use super::config::{ConvLayerConfig, ExtractorMode, WavLmConfig};
use super::error::Result;

/// One feature-extractor conv block. The norm shape is `out_channels`; the
/// conv weight shape is `(out_channels, in_channels, kernel_size)`.
#[derive(Clone)]
pub struct ConvLayerBlock {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub conv_weight: Tensor,
    pub conv_bias: Option<Tensor>,
    /// `None` for blocks that carry no normalization (GroupNorm mode, blocks > 0).
    pub norm: Option<BlockNorm>,
}

#[derive(Clone)]
pub struct BlockNorm {
    pub kind: NormKind,
    pub weight: Tensor,
    pub bias: Tensor,
    pub eps: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormKind {
    /// `LayerNorm` over the channel axis (applied after transposing to `NTC`).
    LayerNorm,
    /// `GroupNorm` with `num_groups == out_channels` (instance-norm over time)
    /// applied in `NCT` layout.
    GroupNorm,
}

impl ConvLayerBlock {
    pub fn empty(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        has_bias: bool,
        norm: Option<NormKind>,
    ) -> Self {
        let fan_in = in_channels * kernel_size;
        let conv_weight = fan_in_uniform(&[out_channels, in_channels, kernel_size], fan_in, DType::Float32);
        let conv_bias = has_bias.then(|| zeros(&[out_channels], DType::Float32));
        let norm = norm.map(|kind| BlockNorm {
            kind,
            weight: ones(&[out_channels], DType::Float32),
            bias: zeros(&[out_channels], DType::Float32),
            eps: 1e-5,
        });
        Self { in_channels, out_channels, kernel_size, stride, conv_weight, conv_bias, norm }
    }

    /// Forward in `NCT` layout: input `(B, C_in, T_in)` → output `(B, C_out, T_out)`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Conv1d via the generic conv2d builder (weight rank decides ndim).
        let y = x
            .conv2d()
            .weight(&self.conv_weight)
            .maybe_bias(self.conv_bias.as_ref())
            .stride(&[self.stride])
            .padding(&[(0, 0)])
            .call()?;

        // Norm (if any).
        let y = match &self.norm {
            None => y,
            Some(BlockNorm { kind: NormKind::GroupNorm, weight, bias, eps }) => {
                // GroupNorm with num_groups = out_channels, NCT layout.
                y.group_norm().scale(weight).bias(bias).num_groups(self.out_channels).eps(*eps).call()?
            }
            Some(BlockNorm { kind: NormKind::LayerNorm, weight, bias, eps }) => {
                // LayerNorm over channel axis: transpose to NTC, normalize, transpose back.
                let yt = y.try_permute(&[0, 2, 1])?;
                let normed = yt.layernorm(-1, *eps)?;
                let yt = normed.try_mul(weight)?.try_add(bias)?;
                yt.try_permute(&[0, 2, 1])?
            }
        };

        // Exact (erf-based) GELU matches PyTorch's `nn.functional.gelu`.
        Ok(y.gelu_exact()?)
    }
}

impl HasStateDict for ConvLayerBlock {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.insert(prefixed(prefix, "conv.weight"), self.conv_weight.clone());
        if let Some(b) = &self.conv_bias {
            sd.insert(prefixed(prefix, "conv.bias"), b.clone());
        }
        if let Some(norm) = &self.norm {
            sd.insert(prefixed(prefix, "layer_norm.weight"), norm.weight.clone());
            sd.insert(prefixed(prefix, "layer_norm.bias"), norm.bias.clone());
        }
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.conv_weight = get_tensor(sd, &prefixed(prefix, "conv.weight"))?;
        if self.conv_bias.is_some() {
            self.conv_bias = Some(get_tensor(sd, &prefixed(prefix, "conv.bias"))?);
        }
        if let Some(norm) = self.norm.as_mut() {
            norm.weight = get_tensor(sd, &prefixed(prefix, "layer_norm.weight"))?;
            norm.bias = get_tensor(sd, &prefixed(prefix, "layer_norm.bias"))?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// FeatureExtractor
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct FeatureExtractor {
    pub blocks: Vec<ConvLayerBlock>,
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
                let norm_kind = match config.extractor_mode {
                    ExtractorMode::LayerNorm => Some(NormKind::LayerNorm),
                    ExtractorMode::GroupNorm if i == 0 => Some(NormKind::GroupNorm),
                    ExtractorMode::GroupNorm => None,
                };
                let block = ConvLayerBlock::empty(*in_ch, out_ch, k, stride, has_bias, norm_kind);
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
        if let Some(dw) = &self.dummy_weight { Ok(x.try_mul(dw)?) } else { Ok(x) }
    }

    /// Cumulative downsampling factor of all blocks (product of strides).
    pub fn total_stride(&self) -> usize {
        self.blocks.iter().map(|b| b.stride).product()
    }

    /// Output time-frames given an input sample count: applies each block's
    /// `(L - k) // stride + 1` rule (assumes `padding=0`, `dilation=1`).
    pub fn num_frames(&self, num_samples: usize) -> usize {
        self.blocks.iter().fold(
            num_samples,
            |t, b| {
                if t < b.kernel_size { 0 } else { (t - b.kernel_size) / b.stride + 1 }
            },
        )
    }
}

impl HasStateDict for FeatureExtractor {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        for (i, block) in self.blocks.iter().enumerate() {
            let p = format!("{prefix}.conv_layers.{i}");
            sd.extend(block.state_dict(&p));
        }
        if let Some(dw) = &self.dummy_weight {
            sd.insert(prefixed(prefix, "dummy_weight"), dw.clone());
        }
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        for (i, block) in self.blocks.iter_mut().enumerate() {
            let p = format!("{prefix}.conv_layers.{i}");
            block.load_state_dict(sd, &p)?;
        }
        let dw_key = prefixed(prefix, "dummy_weight");
        self.dummy_weight = sd.get(&dw_key).cloned();
        Ok(())
    }
}
