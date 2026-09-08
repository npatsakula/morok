//! [`ResNet`] — the unified depth-parameterised ResNet model. Construction is
//! driven by [`ResNetDepth`] and [`OutputMode`]; the forward pass is identical
//! for every variant and only the loader's key probing is depth-aware.
//!
//! Layout matches `timm` / `torchvision`:
//!
//! ```text
//! conv1.weight              # stem 7x7
//! bn1.{...}                 # stem BN
//! layer{1..4}.{i}.{...}     # stage blocks
//!   conv{1..N}.weight
//!   bn{1..N}.{...}
//!   downsample.0.weight     # 1x1 downsample conv (when first block downsamples)
//!   downsample.1.{...}      # downsample BN
//! fc.weight, fc.bias        # classification head (optional)
//! ```

use std::path::Path;

use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::{BatchNorm2d, Conv2d, Layer, Linear, Module};

use crate::blocks::{ResidualStage, batchnorm2d, conv2d};
use crate::init::{Bias, linear};
use crate::state::{self, StateDict};

use super::config::{OutputMode, ResNetConfig, ResNetDepth};
use super::error::Result;

/// Image classification / feature backbone. Construct via one of the loaders
/// ([`ResNet::from_hub`], [`ResNet::from_safetensors`], or
/// [`ResNet::from_state_dict`]) — the empty-tensor placeholders in the layer
/// structs are not usable until weights are loaded.
#[derive(Clone, Module)]
pub struct ResNet {
    #[module(skip)]
    pub config: ResNetConfig,
    #[module(key = "conv1")]
    stem_conv: Conv2d,
    #[module(key = "bn1")]
    stem_bn: BatchNorm2d,
    #[module(key = "layer1")]
    stage1: ResidualStage,
    #[module(key = "layer2")]
    stage2: ResidualStage,
    #[module(key = "layer3")]
    stage3: ResidualStage,
    #[module(key = "layer4")]
    stage4: ResidualStage,
    /// Present only in [`OutputMode::Classification`].
    #[module(key = "fc")]
    head: Option<Linear>,
}

impl ResNet {
    /// Build with all-zero weight placeholders. Used by every loader before a
    /// `load_state_dict` call, and exposed publicly for round-trip tests.
    pub fn with_zero_weights(config: ResNetConfig) -> Self {
        let depth = config.depth;
        let block = depth.block();
        let expansion = depth.expansion();
        let layers = depth.layers();

        // timm/torchvision channel schedule: stem emits 64, each stage doubles.
        // Block-internal expansion (×4 for Bottleneck) multiplies the next
        // stage's in_planes.
        let stage1 = ResidualStage::empty(block, 64, 64, layers[0], 1);
        let stage2 = ResidualStage::empty(block, 64 * expansion, 128, layers[1], 2);
        let stage3 = ResidualStage::empty(block, 128 * expansion, 256, layers[2], 2);
        let stage4 = ResidualStage::empty(block, 256 * expansion, 512, layers[3], 2);

        let head = match &config.output {
            OutputMode::Classification { num_classes } => {
                let fan_in = 512 * expansion;
                Some(linear(fan_in, *num_classes, Bias::FanIn, DType::Float32))
            }
            OutputMode::Features => None,
        };

        Self {
            config,
            stem_conv: conv2d(64, 3, 7, 2, 3),
            stem_bn: batchnorm2d(64),
            stage1,
            stage2,
            stage3,
            stage4,
            head,
        }
    }

    /// Number of output channels after stage 4 (before any FC head). Useful
    /// when consumers want to pre-allocate downstream buffers.
    pub fn feature_channels(&self) -> usize {
        512 * self.config.depth.expansion()
    }

    // -----------------------------------------------------------------------
    // Loaders
    // -----------------------------------------------------------------------

    /// Download `model.safetensors` from a HuggingFace Hub repository at the
    /// `main` revision and load it. The repo must publish a flat timm /
    /// torchvision-style state dict.
    pub fn from_hub(model_id: &str, depth: ResNetDepth, output: OutputMode) -> Result<Self> {
        Self::from_hub_with_revision(model_id, "main", depth, output)
    }

    pub fn from_hub_with_revision(
        model_id: &str,
        revision: &str,
        depth: ResNetDepth,
        output: OutputMode,
    ) -> Result<Self> {
        let repo = crate::hub::HubRepo::open(model_id, revision)?;
        let weights_path = repo.get("model.safetensors")?;
        Self::from_safetensors(&weights_path, depth, output)
    }

    /// Load from a local `model.safetensors`. The file must use the timm /
    /// torchvision key layout (see the module-level docs for the keys).
    pub fn from_safetensors(path: &Path, depth: ResNetDepth, output: OutputMode) -> Result<Self> {
        let sd = state::load_safetensors(path)?;
        Self::from_state_dict(&sd, ResNetConfig::new(depth, output))
    }

    /// Build from a preloaded state dict in the timm / torchvision layout. The
    /// keys are PyTorch's own, `num_batches_tracked` included — it is simply
    /// not read.
    pub fn from_state_dict(sd: &StateDict, config: ResNetConfig) -> Result<Self> {
        let mut model = Self::with_zero_weights(config);
        model.load_state_dict(sd, "")?;
        Ok(model)
    }

    // -----------------------------------------------------------------------
    // Forward
    // -----------------------------------------------------------------------

    /// Run the full network on `images` `[B, 3, H, W]`. Returns either
    /// classification logits `[B, num_classes]` or the final feature map
    /// `[B, 512*exp, H/32, W/32]`, depending on [`ResNetConfig::output`].
    pub fn forward(&self, images: &Tensor) -> Result<Tensor> {
        let x = self.stem_bn.forward(&self.stem_conv.forward(images)?)?.relu()?;
        let x = x.max_pool2d().kernel_size(&[3, 3]).stride(&[2, 2]).padding(&[(1, 1), (1, 1)]).call()?;

        let x = self.stage1.forward(&x)?;
        let x = self.stage2.forward(&x)?;
        let x = self.stage3.forward(&x)?;
        let x = self.stage4.forward(&x)?;

        match (&self.head, &self.config.output) {
            (Some(fc), OutputMode::Classification { .. }) => {
                // Global average pool over the two spatial axes.
                let pooled = x.mean_with().axes(vec![2isize, 3]).keepdim(false).call()?;
                Ok(fc.forward(&pooled)?)
            }
            _ => Ok(x),
        }
    }
}
