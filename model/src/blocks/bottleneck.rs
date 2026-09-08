use svod_tensor::Tensor;
use svod_tensor::nn::{BatchNorm2d, Conv2d, Layer, Module};

use super::basic_block::{BlockKind, Downsample, downsample, shortcut};
use super::batchnorm::batchnorm2d;
use super::conv::conv2d;
use super::error::Result;

#[derive(Clone, Module)]
pub struct Bottleneck {
    pub conv1: Conv2d,
    pub bn1: BatchNorm2d,
    pub conv2: Conv2d,
    pub bn2: BatchNorm2d,
    pub conv3: Conv2d,
    pub bn3: BatchNorm2d,
    pub downsample: Downsample,
}

impl Bottleneck {
    pub fn empty(in_planes: usize, planes: usize, stride: usize) -> Self {
        let out_ch = planes * BlockKind::Bottleneck.expansion();
        Self {
            conv1: conv2d(planes, in_planes, 1, 1, 0),
            bn1: batchnorm2d(planes),
            conv2: conv2d(planes, planes, 3, stride, 1),
            bn2: batchnorm2d(planes),
            conv3: conv2d(out_ch, planes, 1, 1, 0),
            bn3: batchnorm2d(out_ch),
            downsample: downsample(in_planes, out_ch, stride),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = self.bn1.forward(&self.conv1.forward(x)?)?.relu()?;
        let out = self.bn2.forward(&self.conv2.forward(&out)?)?.relu()?;
        let out = self.bn3.forward(&self.conv3.forward(&out)?)?;
        Ok(out.try_add(&shortcut(&self.downsample, x)?)?.relu()?)
    }
}
