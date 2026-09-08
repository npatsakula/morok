use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::Conv2d;

use crate::init::fan_in_uniform;
use crate::state::{self, HasStateDict, StateDict, get_tensor, prefixed};

use super::error::Result;

/// A bias-less `kernel×kernel` convolution with square stride and padding,
/// weights drawn from the fan-in uniform distribution.
pub fn conv2d(out_ch: usize, in_ch: usize, kernel: usize, stride: usize, padding: usize) -> Conv2d {
    conv2d_grouped(out_ch, in_ch, kernel, stride, padding, 1)
}

/// [`conv2d`] split into `groups`; each filter sees `in_ch / groups` channels,
/// so the weight is `[out_ch, in_ch / groups, kernel, kernel]`.
pub fn conv2d_grouped(
    out_ch: usize,
    in_ch: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    groups: usize,
) -> Conv2d {
    let cin = in_ch / groups;
    let weight = fan_in_uniform(&[out_ch, cin, kernel, kernel], cin * kernel * kernel, DType::Float32);
    let p = padding as isize;
    Conv2d::new(weight, None).with_stride((stride, stride)).with_padding(((p, p), (p, p))).with_groups(groups)
}

/// Compatibility shim for callers not yet ported to [`Conv2d`]; the blocks in
/// this module use [`conv2d`] instead. Delete once `wespeaker`, `diarizen` and
/// `gtcrn` are migrated.
#[derive(Clone)]
pub struct Conv2dWeights {
    pub weight: Tensor,
    pub stride: usize,
    pub padding: usize,
    pub groups: usize,
}

impl Conv2dWeights {
    pub fn empty(out_ch: usize, in_ch: usize, kernel: usize, stride: usize, padding: usize) -> Self {
        Self::empty_grouped(out_ch, in_ch, kernel, stride, padding, 1)
    }

    pub fn empty_grouped(
        out_ch: usize,
        in_ch: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        groups: usize,
    ) -> Self {
        let conv = conv2d_grouped(out_ch, in_ch, kernel, stride, padding, groups);
        Self { weight: conv.weight, stride, padding, groups }
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
