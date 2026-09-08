use svod_tensor::Tensor;
use svod_tensor::nn::{BatchNorm2d, Conv2d, Layer, Module};

use super::batchnorm::batchnorm2d;
use super::conv::conv2d;
use super::error::Result;

/// Which residual block class a stage uses.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum BlockKind {
    /// Two 3×3 convs per block, no channel expansion.
    Basic,
    /// 1×1 → 3×3 → 1×1 bottleneck, 4× channel expansion.
    Bottleneck,
}

impl BlockKind {
    pub fn expansion(self) -> usize {
        match self {
            BlockKind::Basic => 1,
            BlockKind::Bottleneck => 4,
        }
    }
}

/// The 1×1 conv + BN projection on a residual shortcut that changes shape.
pub(super) type Downsample = Option<(Conv2d, BatchNorm2d)>;

/// Build the shortcut projection a block needs when it changes stride or width.
pub(super) fn downsample(in_planes: usize, out_ch: usize, stride: usize) -> Downsample {
    (stride != 1 || in_planes != out_ch).then(|| (conv2d(out_ch, in_planes, 1, stride, 0), batchnorm2d(out_ch)))
}

/// Run the shortcut branch: the projection when there is one, else identity.
pub(super) fn shortcut(downsample: &Downsample, x: &Tensor) -> Result<Tensor> {
    match downsample {
        Some((conv, bn)) => Ok(bn.forward(&conv.forward(x)?)?),
        None => Ok(x.clone()),
    }
}

#[derive(Clone, Module)]
pub struct BasicBlock {
    pub conv1: Conv2d,
    pub bn1: BatchNorm2d,
    pub conv2: Conv2d,
    pub bn2: BatchNorm2d,
    pub downsample: Downsample,
}

impl BasicBlock {
    pub fn empty(in_planes: usize, planes: usize, stride: usize) -> Self {
        let out_ch = planes * BlockKind::Basic.expansion();
        Self {
            conv1: conv2d(planes, in_planes, 3, stride, 1),
            bn1: batchnorm2d(planes),
            conv2: conv2d(planes, planes, 3, 1, 1),
            bn2: batchnorm2d(planes),
            downsample: downsample(in_planes, out_ch, stride),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = self.bn1.forward(&self.conv1.forward(x)?)?.relu()?;
        let out = self.bn2.forward(&self.conv2.forward(&out)?)?;
        Ok(out.try_add(&shortcut(&self.downsample, x)?)?.relu()?)
    }
}
