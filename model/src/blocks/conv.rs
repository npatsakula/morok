use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::init::fan_in_uniform;
use crate::state::{self, HasStateDict, StateDict, get_tensor, prefixed};

use super::error::Result;

/// Bias-less 2D convolution wrapper. `weight` has layout
/// `[out_ch, in_ch / groups, kH, kW]` (same as PyTorch / timm).
#[derive(Clone)]
pub struct Conv2dWeights {
    pub weight: Tensor,
    pub stride: usize,
    pub padding: usize,
    pub groups: usize,
}

impl Conv2dWeights {
    pub fn empty(out_ch: usize, in_ch: usize, kernel: usize, stride: usize, padding: usize) -> Self {
        let fan_in = in_ch * kernel * kernel;
        let weight = fan_in_uniform(&[out_ch, in_ch, kernel, kernel], fan_in, DType::Float32);
        Self { weight, stride, padding, groups: 1 }
    }

    /// Like [`empty`](Self::empty) but with grouped convolution. Weight shape
    /// is `[out_ch, in_ch / groups, kernel, kernel]`.
    pub fn empty_grouped(
        out_ch: usize,
        in_ch: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        groups: usize,
    ) -> Self {
        let cin_per_g = in_ch / groups;
        let fan_in = cin_per_g * kernel * kernel;
        let weight = fan_in_uniform(&[out_ch, cin_per_g, kernel, kernel], fan_in, DType::Float32);
        Self { weight, stride, padding, groups }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let p = self.padding as isize;
        Ok(x.conv2d()
            .weight(&self.weight)
            .groups(self.groups)
            .stride(&[self.stride, self.stride])
            .padding(&[(p, p), (p, p)])
            .call()?)
    }
}

impl HasStateDict for Conv2dWeights {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.insert(prefixed(prefix, "weight"), self.weight.clone());
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.weight = get_tensor(sd, &prefixed(prefix, "weight"))?;
        Ok(())
    }
}
