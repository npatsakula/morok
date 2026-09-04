//! Weight initializers shared by every model's `empty` / `with_random_weights`
//! constructor.
//!
//! Each helper ends in `contiguous()`, as Tinygrad's `rand` does, so an
//! initialized parameter materializes into its own buffer instead of being
//! fused into every consuming kernel: consumers then compile the same kernels
//! for `empty()` weights as for loaded ones.

use svod_dtype::DType;
use svod_tensor::Tensor;

/// PyTorch / Tinygrad fan-in uniform: samples from `[-1/√fan_in, +1/√fan_in)`.
/// Used for Conv/Linear weights, their biases, embeddings, and LSTM gates.
pub(crate) fn fan_in_uniform(shape: &[usize], fan_in: usize, dtype: DType) -> Tensor {
    let bound = (fan_in.max(1) as f64).powf(-0.5);
    Tensor::uniform_with_dtype(shape, -bound, bound, dtype).expect("non-empty shape with finite bounds").contiguous()
}

pub(crate) fn ones(shape: &[usize], dtype: DType) -> Tensor {
    Tensor::ones(shape, dtype).expect("non-empty shape").contiguous()
}

pub(crate) fn zeros(shape: &[usize], dtype: DType) -> Tensor {
    Tensor::zeros(shape, dtype).expect("non-empty shape").contiguous()
}
