use svod_dtype::DType;

use crate::Tensor;
use crate::nn::{Layer, Module};

type Result<T> = crate::Result<T>;

/// 1D convolution: `y = conv1d(x, weight) + bias`.
///
/// Weight shape: `[out_channels, in_channels / groups, kernel]`, optional bias
/// shape: `[out_channels]`. The hyper-parameters live on the module so
/// [`Layer::forward`] stays parameter-free.
/// State-dict keys: `weight`, and `bias` when the layer has one.
#[derive(Clone, Module)]
#[module(crate = "crate")]
pub struct Conv1d {
    pub weight: Tensor,
    #[module(optional)]
    pub bias: Option<Tensor>,
    pub stride: usize,
    pub padding: (isize, isize),
    pub dilation: usize,
    pub groups: usize,
}

impl Conv1d {
    /// Create a Conv1d from existing weight (and optional bias) tensors,
    /// with unit stride and dilation, no padding and one group.
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias, stride: 1, padding: (0, 0), dilation: 1, groups: 1 }
    }

    /// Create a Conv1d with a Kaiming-uniform weight and a zero bias.
    ///
    /// Both parameters are [`contiguous`](Tensor::contiguous), so they
    /// materialize into their own buffers instead of being fused into every
    /// consumer.
    #[track_caller]
    pub fn with_dims(in_channels: usize, out_channels: usize, kernel: usize, bias: bool, dtype: DType) -> Self {
        origin_call!("Conv1d::with_dims");
        let weight = Tensor::kaiming_uniform_with_dtype(&[out_channels, in_channels, kernel], 0.0, dtype.clone())
            .expect("non-empty shape")
            .contiguous();
        Self { bias: bias.then(|| Tensor::zeros(&[out_channels], dtype).contiguous()), ..Self::new(weight, None) }
    }

    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }

    pub fn with_padding(mut self, padding: (isize, isize)) -> Self {
        self.padding = padding;
        self
    }

    pub fn with_dilation(mut self, dilation: usize) -> Self {
        self.dilation = dilation;
        self
    }

    pub fn with_groups(mut self, groups: usize) -> Self {
        self.groups = groups;
        self
    }
}

impl Layer for Conv1d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.conv1d()
            .weight(&self.weight)
            .maybe_bias(self.bias.as_ref())
            .stride(self.stride)
            .padding(self.padding)
            .dilation(self.dilation)
            .groups(self.groups)
            .call()
    }
}
