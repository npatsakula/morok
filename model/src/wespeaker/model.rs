//! [`WeSpeakerResNet34`] — pyannote's 256-d speaker-embedding model.
//!
//! Input contract: Kaldi-fbank features `[B, T=1598, F=80]` (f32) and per-frame
//! attention weights `[B, T_w=799]` (f32). Output: `[B, 256]` embeddings.
//!
//! State-dict layout (after stripping the `resnet.` prefix that pyannote's
//! `WeSpeakerResNet34` wrapper adds):
//!
//! ```text
//! conv1.weight             # stem 3x3
//! bn1.{...}                # stem BN
//! layer{1..4}.{i}.{...}    # 4 BasicBlock stages with [3, 4, 6, 3] schedule
//!   conv{1,2}.weight
//!   bn{1,2}.{...}
//!   downsample.0.weight    # 1x1 downsample conv on stages 2/3/4 first block
//!   downsample.1.{...}     # downsample BN
//! seg_1.weight             # final projection 5120 -> 256
//! seg_1.bias
//! ```

use std::path::Path;

use snafu::ResultExt;
use svod_dtype::DType;
use svod_ir::SInt;
use svod_tensor::nn::{BatchNorm2d, Conv2d, Layer, Linear, Module, StateDict};
use svod_tensor::{BoundVariable, Tensor};

use crate::blocks::{BlockKind, ResidualStage, batchnorm2d, conv2d, remap};

use super::error::{PickleSnafu, Result};
use super::pickle;
use super::tstp::tstp_forward;

/// Fixed mel-bin count of the pyannote LM checkpoint.
pub const NUM_MEL_BINS: usize = 80;
/// Fixed embedding output dimension.
pub const EMBED_DIM: usize = 256;
/// `m_channels` from the WeSpeaker reference (stem output channels).
pub const M_CHANNELS: usize = 32;
/// Per-stage block count: ResNet34 schedule.
pub const NUM_BLOCKS: [usize; 4] = [3, 4, 6, 3];

#[derive(Clone, Debug)]
pub struct WeSpeakerConfig {
    /// Upper bound on the symbolic `b` variable exposed by the JIT wrapper.
    pub max_batch_size: usize,
}

impl WeSpeakerConfig {
    pub fn new() -> Self {
        Self { max_batch_size: 1 }
    }

    pub fn with_max_batch_size(mut self, max_batch_size: usize) -> Self {
        self.max_batch_size = max_batch_size;
        self
    }
}

impl Default for WeSpeakerConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Module)]
pub struct WeSpeakerResNet34 {
    #[module(skip)]
    pub config: WeSpeakerConfig,
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
    seg_1: Linear,
}

impl WeSpeakerResNet34 {
    /// Build with all-zero weights. Used by every loader before
    /// `from_state_dict` and exposed publicly for round-trip tests.
    pub fn with_zero_weights(config: WeSpeakerConfig) -> Self {
        let stage1 = ResidualStage::empty(BlockKind::Basic, M_CHANNELS, M_CHANNELS, NUM_BLOCKS[0], 1);
        let stage2 = ResidualStage::empty(BlockKind::Basic, M_CHANNELS, M_CHANNELS * 2, NUM_BLOCKS[1], 2);
        let stage3 = ResidualStage::empty(BlockKind::Basic, M_CHANNELS * 2, M_CHANNELS * 4, NUM_BLOCKS[2], 2);
        let stage4 = ResidualStage::empty(BlockKind::Basic, M_CHANNELS * 4, M_CHANNELS * 8, NUM_BLOCKS[3], 2);

        // Stats dim = (m_channels * 8) * (num_mel_bins / 8) = 256 * 10 = 2560
        // seg_1: Linear(stats_dim * 2 -> embed_dim) = Linear(5120 -> 256)
        let stats_dim = M_CHANNELS * 8 * (NUM_MEL_BINS / 8);
        let zeros = |shape: &[usize]| Tensor::zeros(shape, DType::Float32);

        Self {
            config,
            stem_conv: conv2d(M_CHANNELS, 1, 3, 1, 1),
            stem_bn: batchnorm2d(M_CHANNELS),
            stage1,
            stage2,
            stage3,
            stage4,
            seg_1: Linear::new(zeros(&[EMBED_DIM, stats_dim * 2]), Some(zeros(&[EMBED_DIM]))),
        }
    }

    // -----------------------------------------------------------------------
    // Loaders
    // -----------------------------------------------------------------------

    /// Download `pytorch_model.bin` from a HuggingFace Hub repository at the
    /// `main` revision and load it. Expects the pyannote-style checkpoint
    /// (`torch.save({"state_dict": ..., "pytorch-lightning_version": ...})`)
    /// with `resnet.`-prefixed keys.
    pub fn from_hub(model_id: &str, config: WeSpeakerConfig) -> Result<Self> {
        Self::from_hub_with_revision(model_id, "main", config)
    }

    pub fn from_hub_with_revision(model_id: &str, revision: &str, config: WeSpeakerConfig) -> Result<Self> {
        let repo = crate::hub::HubRepo::open(model_id, revision)?;
        let weights_path = repo.get("pytorch_model.bin")?;
        Self::from_pytorch_bin(&weights_path, config)
    }

    /// Load from a local `pytorch_model.bin` (pyannote/PyTorch-Lightning
    /// format). Strips the `resnet.` key prefix and renames pyannote's
    /// `shortcut.{0,1}` BN downsample naming to torchvision-style
    /// `downsample.{0,1}` on the way in.
    pub fn from_pytorch_bin(path: &Path, config: WeSpeakerConfig) -> Result<Self> {
        let sd = pickle::load_pyannote_pytorch_bin(path, "resnet.").context(PickleSnafu)?;
        let sd: StateDict = sd.into_iter().map(|(k, v)| (rename_shortcut_to_downsample(&k), v)).collect();
        Self::from_state_dict(&sd, config)
    }

    /// Build from a preloaded state dict. The state dict must use the
    /// `resnet.`-stripped WeSpeaker key layout (see module docs); every layer
    /// reads PyTorch's own key names, so the only pre-pass is dropping
    /// `num_batches_tracked`.
    pub fn from_state_dict(sd: &StateDict, config: WeSpeakerConfig) -> Result<Self> {
        let mut model = Self::with_zero_weights(config);
        model.load_state_dict(&remap::strip_metadata(sd.clone()), "")?;
        Ok(model)
    }

    // -----------------------------------------------------------------------
    // Forward
    // -----------------------------------------------------------------------

    /// Run the full network on `feats` `[max_b, 1598, 80]` and `weights`
    /// `[max_b, 799]`, shrunk to `batch` before the stem. Returns
    /// `[B, 256]` embeddings.
    pub fn forward(&self, feats: &Tensor, weights: &Tensor, batch: &BoundVariable) -> Result<Tensor> {
        let b = batch.as_sint();

        // Shrink batch dim to live value
        let feats = feats.try_shrink([Some((SInt::Const(0), b.clone())), None, None])?;
        let weights = weights.try_shrink([Some((SInt::Const(0), b)), None])?;

        // (B, T=1598, F=80) -> (B, F, T) -> (B, 1, F, T)
        let x = feats.try_permute(&[0, 2, 1])?;
        let x = x.try_unsqueeze(1)?;

        // Stem
        let x = self.stem_bn.forward(&self.stem_conv.forward(&x)?)?.relu()?;

        // Stages
        let x = self.stage1.forward(&x)?;
        let x = self.stage2.forward(&x)?;
        let x = self.stage3.forward(&x)?;
        let x = self.stage4.forward(&x)?;
        // x is now (B, 256, 10, T_back) with T_back determined by the strided convs.

        // TSTP head: (B, C, H, T) + (B, T_w) -> (B, 2*C*H = 5120)
        let stats = tstp_forward(&x, &weights)?;

        // seg_1: Linear(5120 -> 256)
        Ok(self.seg_1.forward(&stats)?)
    }
}

/// pyannote's WeSpeaker module names the residual-shortcut conv `shortcut.0`
/// and the BN `shortcut.1`. The svod [`BasicBlock`] expects torchvision's
/// `downsample.0` / `downsample.1` naming for the same slots. Rewrite the
/// substring `.shortcut.` to `.downsample.` whenever it appears in a key.
fn rename_shortcut_to_downsample(key: &str) -> String {
    key.replace(".shortcut.", ".downsample.")
}
