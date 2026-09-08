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
use svod_ir::SInt;
use svod_tensor::{BoundVariable, Tensor};

use crate::blocks::{BatchNormWeights, Conv2dWeights, ResidualStage, remap};
use crate::init::fan_in_uniform;
use crate::state::{self, HasStateDict, StateDict, get_tensor};

use super::config::{OutputMode, ResNetConfig, ResNetDepth};
use super::error::Result;

/// Image classification / feature backbone. Construct via one of the loaders
/// ([`ResNet::from_hub`], [`ResNet::from_safetensors`], or
/// [`ResNet::from_state_dict`]) — the empty-tensor placeholders in the layer
/// structs are not usable until weights are loaded.
#[derive(Clone)]
pub struct ResNet {
    pub config: ResNetConfig,
    stem_conv: Conv2dWeights,
    stem_bn: BatchNormWeights,
    stage1: ResidualStage,
    stage2: ResidualStage,
    stage3: ResidualStage,
    stage4: ResidualStage,
    head: Option<HeadWeights>,
}

#[derive(Clone)]
struct HeadWeights {
    weight: Tensor,
    bias: Tensor,
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
                Some(HeadWeights {
                    weight: fan_in_uniform(&[*num_classes, fan_in], fan_in, DType::Float32),
                    bias: fan_in_uniform(&[*num_classes], fan_in, DType::Float32),
                })
            }
            OutputMode::Features => None,
        };

        Self {
            config,
            stem_conv: Conv2dWeights::empty(64, 3, 7, 2, 3),
            stem_bn: BatchNormWeights::empty(64),
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

    /// Build from a preloaded state dict. Runs [`remap::fold_batchnorm`] first
    /// to translate `running_var` keys into `invstd` keys (and compute the
    /// value transform) and drop `num_batches_tracked` — the loaded layer
    /// structs read directly from the post-fold layout.
    pub fn from_state_dict(sd: &StateDict, config: ResNetConfig) -> Result<Self> {
        let sd = remap::fold_batchnorm(sd.clone())?;
        let mut model = Self::with_zero_weights(config);
        model.stem_conv.load_state_dict(&sd, "conv1")?;
        model.stem_bn.load_state_dict(&sd, "bn1")?;
        model.stage1.load_state_dict(&sd, "layer1")?;
        model.stage2.load_state_dict(&sd, "layer2")?;
        model.stage3.load_state_dict(&sd, "layer3")?;
        model.stage4.load_state_dict(&sd, "layer4")?;

        if let Some(head) = model.head.as_mut() {
            head.weight = get_tensor(&sd, "fc.weight")?;
            head.bias = get_tensor(&sd, "fc.bias")?;
        }
        Ok(model)
    }

    // -----------------------------------------------------------------------
    // Forward
    // -----------------------------------------------------------------------

    /// Run the full network on `images` `[max_b, 3, H, W]`, shrunk to the
    /// `batch` variable's bound value before the stem. Returns either
    /// classification logits `[B, num_classes]` or the final feature map
    /// `[B, 512*exp, H/32, W/32]`, depending on
    /// [`ResNetConfig::output`].
    pub fn forward(&self, images: &Tensor, batch: &BoundVariable) -> Result<Tensor> {
        let b = batch.as_sint();

        let x = images.try_shrink([Some((SInt::Const(0), b)), None, None, None])?;
        let x = self.stem_bn.forward(&self.stem_conv.forward(&x)?)?.relu()?;
        let x = x.max_pool2d().kernel_size(&[3, 3]).stride(&[2, 2]).padding(&[(1, 1), (1, 1)]).call()?;

        let x = self.stage1.forward(&x)?;
        let x = self.stage2.forward(&x)?;
        let x = self.stage3.forward(&x)?;
        let x = self.stage4.forward(&x)?;

        match (&self.head, &self.config.output) {
            (Some(fc), OutputMode::Classification { .. }) => {
                // Global average pool over the two spatial axes.
                let pooled = x.mean_with().axes(vec![2isize, 3]).keepdim(false).call()?;
                Ok(pooled.linear().weight(&fc.weight).bias(&fc.bias).call()?)
            }
            _ => Ok(x),
        }
    }
}

impl HasStateDict for ResNet {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = self.stem_conv.state_dict(&state::prefixed(prefix, "conv1"));
        sd.extend(self.stem_bn.state_dict(&state::prefixed(prefix, "bn1")));
        sd.extend(self.stage1.state_dict(&state::prefixed(prefix, "layer1")));
        sd.extend(self.stage2.state_dict(&state::prefixed(prefix, "layer2")));
        sd.extend(self.stage3.state_dict(&state::prefixed(prefix, "layer3")));
        sd.extend(self.stage4.state_dict(&state::prefixed(prefix, "layer4")));
        if let Some(head) = &self.head {
            sd.insert(state::prefixed(prefix, "fc.weight"), head.weight.clone());
            sd.insert(state::prefixed(prefix, "fc.bias"), head.bias.clone());
        }
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.stem_conv.load_state_dict(sd, &state::prefixed(prefix, "conv1"))?;
        self.stem_bn.load_state_dict(sd, &state::prefixed(prefix, "bn1"))?;
        self.stage1.load_state_dict(sd, &state::prefixed(prefix, "layer1"))?;
        self.stage2.load_state_dict(sd, &state::prefixed(prefix, "layer2"))?;
        self.stage3.load_state_dict(sd, &state::prefixed(prefix, "layer3"))?;
        self.stage4.load_state_dict(sd, &state::prefixed(prefix, "layer4"))?;
        if let Some(head) = self.head.as_mut() {
            head.weight = get_tensor(sd, &state::prefixed(prefix, "fc.weight"))?;
            head.bias = get_tensor(sd, &state::prefixed(prefix, "fc.bias"))?;
        }
        Ok(())
    }
}
