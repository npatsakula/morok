use svod_tensor::Tensor;

use crate::state::{self, HasStateDict, StateDict, prefixed};

use super::batchnorm::BatchNormWeights;
use super::conv::Conv2dWeights;
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

#[derive(Clone)]
pub struct BasicBlock {
    pub conv1: Conv2dWeights,
    pub bn1: BatchNormWeights,
    pub conv2: Conv2dWeights,
    pub bn2: BatchNormWeights,
    pub downsample: Option<(Conv2dWeights, BatchNormWeights)>,
}

impl BasicBlock {
    pub fn empty(in_planes: usize, planes: usize, stride: usize) -> Self {
        let expansion = BlockKind::Basic.expansion();
        let downsample = if stride != 1 || in_planes != planes * expansion {
            Some((
                Conv2dWeights::empty(planes * expansion, in_planes, 1, stride, 0),
                BatchNormWeights::empty(planes * expansion),
            ))
        } else {
            None
        };
        Self {
            conv1: Conv2dWeights::empty(planes, in_planes, 3, stride, 1),
            bn1: BatchNormWeights::empty(planes),
            conv2: Conv2dWeights::empty(planes, planes, 3, 1, 1),
            bn2: BatchNormWeights::empty(planes),
            downsample,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = self.bn1.forward(&self.conv1.forward(x)?)?;
        let out = out.relu()?;
        let out = self.bn2.forward(&self.conv2.forward(&out)?)?;
        let shortcut = match &self.downsample {
            Some((c, b)) => b.forward(&c.forward(x)?)?,
            None => x.clone(),
        };
        Ok(out.try_add(&shortcut)?.relu()?)
    }
}

impl HasStateDict for BasicBlock {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = self.conv1.state_dict(&prefixed(prefix, "conv1"));
        sd.extend(self.bn1.state_dict(&prefixed(prefix, "bn1")));
        sd.extend(self.conv2.state_dict(&prefixed(prefix, "conv2")));
        sd.extend(self.bn2.state_dict(&prefixed(prefix, "bn2")));
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
        if let Some((c, b)) = &mut self.downsample {
            c.load_state_dict(sd, &prefixed(prefix, "downsample.0"))?;
            b.load_state_dict(sd, &prefixed(prefix, "downsample.1"))?;
        }
        Ok(())
    }
}
