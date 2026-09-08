use svod_dtype::DType;

use crate::Tensor;
use crate::nn::{Layer, Module};

type Result<T> = crate::Result<T>;

/// Fully connected layer: `y = x @ weight.T + bias`.
///
/// Weight shape: `[out_features, in_features]`, bias shape: `[out_features]`.
/// State-dict keys: `weight`, and `bias` when the layer has one.
#[derive(Clone, Module)]
#[module(crate = "crate")]
pub struct Linear {
    pub weight: Tensor,
    #[module(optional)]
    pub bias: Option<Tensor>,
}

impl Linear {
    /// Create a linear layer from existing weight and optional bias tensors.
    ///
    /// Weight must have shape `[out_features, in_features]`, bias must have shape `[out_features]`.
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }

    /// Create a linear layer with a Kaiming-uniform weight and a zero bias.
    ///
    /// Weight shape: `[out_features, in_features]`. Both parameters are
    /// [`contiguous`](Tensor::contiguous), so they materialize into their own
    /// buffers instead of being fused into every consumer.
    #[track_caller]
    pub fn with_dims(in_features: usize, out_features: usize, bias: bool, dtype: DType) -> Self {
        origin_call!("Linear::with_dims");
        let weight = Tensor::kaiming_uniform_with_dtype(&[out_features, in_features], 0.0, dtype.clone())
            .expect("non-empty shape")
            .contiguous();
        Self { weight, bias: bias.then(|| Tensor::zeros(&[out_features], dtype).contiguous()) }
    }
}

impl Layer for Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.linear().weight(&self.weight).maybe_bias(self.bias.as_ref()).call()
    }
}
