use svod_tensor::Tensor;

use crate::state::{self, HasStateDict, StateDict, prefixed};

use super::basic_block::BlockKind;
use super::batchnorm::BatchNormWeights;
use super::conv::Conv2dWeights;
use super::error::Result;

#[derive(Clone)]
pub struct Bottleneck {
    pub conv1: Conv2dWeights,
    pub bn1: BatchNormWeights,
    pub conv2: Conv2dWeights,
    pub bn2: BatchNormWeights,
    pub conv3: Conv2dWeights,
    pub bn3: BatchNormWeights,
    pub downsample: Option<(Conv2dWeights, BatchNormWeights)>,
}

impl Bottleneck {
    pub fn empty(in_planes: usize, planes: usize, stride: usize) -> Self {
        let expansion = BlockKind::Bottleneck.expansion();
        let out_ch = planes * expansion;
        let downsample = if stride != 1 || in_planes != out_ch {
            Some((Conv2dWeights::empty(out_ch, in_planes, 1, stride, 0), BatchNormWeights::empty(out_ch)))
        } else {
            None
        };
        Self {
            conv1: Conv2dWeights::empty(planes, in_planes, 1, 1, 0),
            bn1: BatchNormWeights::empty(planes),
            conv2: Conv2dWeights::empty(planes, planes, 3, stride, 1),
            bn2: BatchNormWeights::empty(planes),
            conv3: Conv2dWeights::empty(out_ch, planes, 1, 1, 0),
            bn3: BatchNormWeights::empty(out_ch),
            downsample,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = self.bn1.forward(&self.conv1.forward(x)?)?.relu()?;
        let out = self.bn2.forward(&self.conv2.forward(&out)?)?.relu()?;
        let out = self.bn3.forward(&self.conv3.forward(&out)?)?;
        let shortcut = match &self.downsample {
            Some((c, b)) => b.forward(&c.forward(x)?)?,
            None => x.clone(),
        };
        Ok(out.try_add(&shortcut)?.relu()?)
    }
}

impl HasStateDict for Bottleneck {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = self.conv1.state_dict(&prefixed(prefix, "conv1"));
        sd.extend(self.bn1.state_dict(&prefixed(prefix, "bn1")));
        sd.extend(self.conv2.state_dict(&prefixed(prefix, "conv2")));
        sd.extend(self.bn2.state_dict(&prefixed(prefix, "bn2")));
        sd.extend(self.conv3.state_dict(&prefixed(prefix, "conv3")));
        sd.extend(self.bn3.state_dict(&prefixed(prefix, "bn3")));
        if let Some((c, b)) = &self.downsample {
            sd.extend(c.state_dict(&prefixed(prefix, "downsample.0")));
            sd.extend(b.state_dict(&prefixed(prefix, "downsample.1")));
        }
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.conv1.load_state_dict(sd, &prefixed(prefix, "conv1"))?;
        self.bn1.load_state_dict(sd, &prefixed(prefix, "bn1"))?;
        self.conv2.load_state_dict(sd, &prefixed(prefix, "conv2"))?;
        self.bn2.load_state_dict(sd, &prefixed(prefix, "bn2"))?;
        self.conv3.load_state_dict(sd, &prefixed(prefix, "conv3"))?;
        self.bn3.load_state_dict(sd, &prefixed(prefix, "bn3"))?;
        if let Some((c, b)) = &mut self.downsample {
            c.load_state_dict(sd, &prefixed(prefix, "downsample.0"))?;
            b.load_state_dict(sd, &prefixed(prefix, "downsample.1"))?;
        }
        Ok(())
    }
}
