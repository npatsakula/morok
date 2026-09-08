//! Weight initializers and layer constructors shared by every model's
//! `empty` / `with_random_weights` constructor.
//!
//! Each helper ends in `contiguous()`, as Tinygrad's `rand` does, so an
//! initialized parameter materializes into its own buffer instead of being
//! fused into every consuming kernel: consumers then compile the same kernels
//! for `empty()` weights as for loaded ones.
//!
//! [`svod_tensor::nn`]'s own `with_dims` constructors draw a Kaiming-uniform
//! weight; the ports here draw PyTorch's fan-in uniform, which is what the
//! checkpoints we target were trained under.

use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::{Conv1d, Embedding, LayerNorm, Linear};

/// PyTorch / Tinygrad fan-in uniform: samples from `[-1/√fan_in, +1/√fan_in)`.
/// Used for Conv/Linear weights, their biases, embeddings, and LSTM gates.
pub(crate) fn fan_in_uniform(shape: &[usize], fan_in: usize, dtype: DType) -> Tensor {
    let bound = (fan_in.max(1) as f64).powf(-0.5);
    Tensor::uniform_with_dtype(shape, -bound, bound, dtype).expect("non-empty shape with finite bounds").contiguous()
}

pub(crate) fn ones(shape: &[usize], dtype: DType) -> Tensor {
    Tensor::ones(shape, dtype).contiguous()
}

pub(crate) fn zeros(shape: &[usize], dtype: DType) -> Tensor {
    Tensor::zeros(shape, dtype).contiguous()
}

/// How a [`linear`] or [`conv1d`] initializes its bias. PyTorch's own default
/// is [`Bias::FanIn`]; the ports whose upstream passes an explicit zero
/// initializer use [`Bias::Zero`].
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Bias {
    None,
    Zero,
    FanIn,
}

impl Bias {
    fn build(self, out: usize, fan_in: usize, dtype: DType) -> Option<Tensor> {
        match self {
            Self::None => Option::None,
            Self::Zero => Some(zeros(&[out], dtype)),
            Self::FanIn => Some(fan_in_uniform(&[out], fan_in, dtype)),
        }
    }
}

/// `nn.Linear(in_features, out_features)`; the weight is `[out, in]`.
pub(crate) fn linear(in_features: usize, out_features: usize, bias: Bias, dtype: DType) -> Linear {
    Linear::new(
        fan_in_uniform(&[out_features, in_features], in_features, dtype.clone()),
        bias.build(out_features, in_features, dtype),
    )
}

/// `nn.Conv1d(in_channels, out_channels, kernel)` at unit stride, dilation and
/// group count; `in_channels` is per group, as the weight shape is.
pub(crate) fn conv1d(in_channels: usize, out_channels: usize, kernel: usize, bias: Bias, dtype: DType) -> Conv1d {
    let fan_in = in_channels * kernel;
    Conv1d::new(
        fan_in_uniform(&[out_channels, in_channels, kernel], fan_in, dtype.clone()),
        bias.build(out_channels, fan_in, dtype),
    )
}

/// Identity-affine `nn.LayerNorm(size)` over the last axis, at PyTorch's
/// default epsilon — the only one any of these checkpoints uses.
pub(crate) fn layer_norm(size: usize, dtype: DType) -> LayerNorm {
    LayerNorm::with_dims(size, true, 1e-5, dtype)
}

/// `nn.Embedding(rows, cols)`: a fan-in uniform lookup table.
pub(crate) fn embedding(rows: usize, cols: usize, dtype: DType) -> Embedding {
    Embedding::new(fan_in_uniform(&[rows, cols], cols, dtype))
}
