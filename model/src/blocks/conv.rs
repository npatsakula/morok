use svod_dtype::DType;
use svod_tensor::nn::Conv2d;

use crate::init::fan_in_uniform;

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
