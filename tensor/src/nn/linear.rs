use morok_dtype::DType;

use crate::Tensor;
use crate::nn::Layer;

type Result<T> = crate::Result<T>;

/// Fully connected layer: `y = x @ weight.T + bias`.
///
/// Weight shape: `[out_features, in_features]`, bias shape: `[out_features]`.
pub struct Linear {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl Linear {
    /// Create a linear layer from existing weight and bias tensors.
    ///
    /// Weight must have shape `[out_features, in_features]`, bias must have shape `[out_features]`.
    pub fn new(weight: Tensor, bias: Tensor) -> Self {
        Self { weight, bias }
    }

    /// Create a linear layer with deterministic initialization using `sin()`.
    ///
    /// Weight shape: `[out_features, in_features]`, bias: zeros.
    pub fn with_dims(in_features: usize, out_features: usize, dtype: DType) -> Self {
        let weight_data: Vec<f32> = (0..in_features * out_features).map(|i| ((i as f32) * 0.1).sin() * 0.1).collect();
        let weight = Tensor::from_slice(&weight_data)
            .try_reshape([out_features as isize, in_features as isize])
            .expect("linear weight reshape failed");
        let bias = Tensor::full(&[out_features], 0.0, dtype).expect("linear bias creation failed");
        Self { weight, bias }
    }
}

impl Layer for Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.linear().weight(&self.weight).bias(&self.bias).call()
    }
}
