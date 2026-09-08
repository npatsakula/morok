use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::init::fan_in_uniform;
use crate::state::{self, HasStateDict, StateDict, get_tensor, prefixed};

use crate::yolo::error::Result;

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Conv2d(bias=False) + BatchNorm2d + SiLU — the universal YOLO building block.
/// When `act` is `false` the activation is skipped (used by SPPF.cv1,
/// Attention projections, and PSABlock FFN output conv).
///
/// State-dict keys: `conv.weight`, `bn.{weight,bias,running_mean,invstd}`.
#[derive(Clone)]
pub struct YoloConv {
    pub conv: crate::blocks::Conv2dWeights,
    pub bn: crate::blocks::BatchNormWeights,
    pub act: bool,
}

impl YoloConv {
    pub fn empty(in_ch: usize, out_ch: usize, kernel: usize, stride: usize, act: bool) -> Self {
        let padding = kernel / 2;
        Self {
            conv: crate::blocks::Conv2dWeights::empty(out_ch, in_ch, kernel, stride, padding),
            bn: crate::blocks::BatchNormWeights::empty(out_ch),
            act,
        }
    }

    /// Depthwise variant: `groups = gcd(in_ch, out_ch)`.
    pub fn empty_dw(in_ch: usize, out_ch: usize, kernel: usize, stride: usize, act: bool) -> Self {
        let groups = gcd(in_ch, out_ch);
        let padding = kernel / 2;
        Self {
            conv: crate::blocks::Conv2dWeights::empty_grouped(out_ch, in_ch, kernel, stride, padding, groups),
            bn: crate::blocks::BatchNormWeights::empty(out_ch),
            act,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv.forward(x)?;
        let x = self.bn.forward(&x)?;
        if self.act { Ok(x.silu()?) } else { Ok(x) }
    }
}

impl HasStateDict for YoloConv {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = self.conv.state_dict(&prefixed(prefix, "conv"));
        sd.extend(self.bn.state_dict(&prefixed(prefix, "bn")));
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.conv.load_state_dict(sd, &prefixed(prefix, "conv"))?;
        self.bn.load_state_dict(sd, &prefixed(prefix, "bn"))?;
        Ok(())
    }
}

/// Raw Conv2d with bias (no BN, no activation). Used by the Detect head's
/// final 1×1 classification and box-regression layers.
///
/// State-dict keys: `weight`, `bias`.
#[derive(Clone)]
pub struct Conv2dBias {
    pub weight: Tensor,
    pub bias: Tensor,
    pub stride: usize,
    pub padding: usize,
}

impl Conv2dBias {
    pub fn empty(in_ch: usize, out_ch: usize, kernel: usize, stride: usize) -> Self {
        let padding = kernel / 2;
        let fan_in = in_ch * kernel * kernel;
        Self {
            weight: fan_in_uniform(&[out_ch, in_ch, kernel, kernel], fan_in, DType::Float32),
            bias: fan_in_uniform(&[out_ch], fan_in, DType::Float32),
            stride,
            padding,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let p = self.padding as isize;
        Ok(x.conv2d()
            .weight(&self.weight)
            .bias(&self.bias)
            .stride(&[self.stride, self.stride])
            .padding(&[(p, p), (p, p)])
            .call()?)
    }
}

impl HasStateDict for Conv2dBias {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.insert(prefixed(prefix, "weight"), self.weight.clone());
        sd.insert(prefixed(prefix, "bias"), self.bias.clone());
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.weight = get_tensor(sd, &prefixed(prefix, "weight"))?;
        self.bias = get_tensor(sd, &prefixed(prefix, "bias"))?;
        Ok(())
    }
}

/// ConvTranspose2d with bias (no BN). Doubles spatial resolution.
///
/// State-dict keys: `weight`, `bias`.
#[derive(Clone)]
pub struct ConvTranspose2dBias {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl ConvTranspose2dBias {
    pub fn empty(in_ch: usize, out_ch: usize, kernel: usize) -> Self {
        let fan_in = in_ch * kernel * kernel;
        Self {
            weight: fan_in_uniform(&[in_ch, out_ch, kernel, kernel], fan_in, DType::Float32),
            bias: fan_in_uniform(&[out_ch], fan_in, DType::Float32),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.conv_transpose2d().weight(&self.weight).maybe_bias(Some(&self.bias)).stride(&[2, 2]).call()?)
    }
}

impl HasStateDict for ConvTranspose2dBias {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.insert(prefixed(prefix, "weight"), self.weight.clone());
        sd.insert(prefixed(prefix, "bias"), self.bias.clone());
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.weight = get_tensor(sd, &prefixed(prefix, "weight"))?;
        self.bias = get_tensor(sd, &prefixed(prefix, "bias"))?;
        Ok(())
    }
}
