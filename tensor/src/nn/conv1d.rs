use svod_dtype::DType;

use crate::Tensor;
use crate::nn::Layer;

type Result<T> = crate::Result<T>;

/// 1D convolution: `y = conv1d(x, weight) + bias`.
///
/// Weight shape: `[out_channels, in_channels, kernel]`, optional bias shape: `[out_channels]`.
/// `stride` and `padding` are stored on the module so [`Layer::forward`] stays parameter-free.
pub struct Conv1d {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub stride: usize,
    pub padding: (isize, isize),
}

impl Conv1d {
    /// Create a Conv1d from existing weight (and optional bias) tensors.
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias, stride: 1, padding: (0, 0) }
    }

    /// Create a Conv1d with deterministic `sin()` initialization, zero bias.
    #[track_caller]
    pub fn with_dims(in_channels: usize, out_channels: usize, kernel: usize, dtype: DType) -> Self {
        origin_call!("Conv1d::with_dims");
        let weight_data: Vec<f32> =
            (0..in_channels * out_channels * kernel).map(|i| ((i as f32) * 0.1).sin() * 0.1).collect();
        let weight = Tensor::from_slice(&weight_data)
            .try_reshape([out_channels as isize, in_channels as isize, kernel as isize])
            .expect("conv1d weight reshape failed");
        let bias = Tensor::full(&[out_channels], 0.0, dtype);
        Self { weight, bias: Some(bias), stride: 1, padding: (0, 0) }
    }

    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }

    pub fn with_padding(mut self, padding: (isize, isize)) -> Self {
        self.padding = padding;
        self
    }
}

impl Layer for Conv1d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.conv1d().weight(&self.weight).maybe_bias(self.bias.as_ref()).stride(self.stride).padding(self.padding).call()
    }
}
