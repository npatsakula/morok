//! Shared building blocks: LinearWeights, LayerNormWeights, Conv1dWeights, sinusoids.

use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::init::{fan_in_uniform, ones, zeros};
use crate::state::{self, HasStateDict, StateDict, get_tensor, prefixed};
use crate::{load_state_field, state_field};

use super::error::Result;

// ─── LinearWeights ──────────────────────────────────────────────────────────

/// Linear layer weights matching PyTorch `nn.Linear(in, out, bias=...)`.
/// Weight shape `[out, in]`. Bias may be absent (e.g. Whisper's key projection).
#[derive(Clone)]
pub struct LinearWeights {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

impl LinearWeights {
    pub fn empty(in_features: usize, out_features: usize, has_bias: bool) -> Self {
        Self::empty_dtype(in_features, out_features, has_bias, DType::Float32)
    }

    pub fn empty_dtype(in_features: usize, out_features: usize, has_bias: bool, dtype: DType) -> Self {
        let weight = fan_in_uniform(&[out_features, in_features], in_features, dtype.clone());
        let bias = has_bias.then(|| fan_in_uniform(&[out_features], in_features, dtype));
        Self { weight, bias }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match &self.bias {
            Some(bias) => linear_with_bias(x, &self.weight, bias),
            None => Ok(x.linear().weight(&self.weight).call()?),
        }
    }
}

pub(crate) fn linear_with_bias(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let output_dtype = x.dtype();
    let is_low_precision = |dtype: &DType| dtype == &DType::Float16 || dtype == &DType::BFloat16;
    let low_precision = is_low_precision(&output_dtype) && is_low_precision(&weight.dtype());
    if !low_precision {
        return Ok(x.linear().weight(weight).bias(bias).call()?);
    }

    Ok(x.linear()
        .weight(weight)
        .dtype(DType::Float32)
        .call()?
        .try_add(&bias.cast(DType::Float32)?)?
        .cast(output_dtype)?)
}

impl HasStateDict for LinearWeights {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.insert(prefixed(prefix, "weight"), self.weight.clone());
        if let Some(b) = &self.bias {
            sd.insert(prefixed(prefix, "bias"), b.clone());
        }
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.weight = get_tensor(sd, &prefixed(prefix, "weight"))?;
        let bias_key = prefixed(prefix, "bias");
        self.bias = sd.get(&bias_key).cloned();
        Ok(())
    }
}

// ─── LayerNormWeights ───────────────────────────────────────────────────────

/// Affine layer normalization: `layernorm(x) * weight + bias`.
#[derive(Clone)]
pub struct LayerNormWeights {
    pub weight: Tensor,
    pub bias: Tensor,
    pub eps: f64,
}

impl LayerNormWeights {
    pub fn empty(size: usize) -> Self {
        Self::empty_dtype(size, DType::Float32)
    }

    pub fn empty_dtype(size: usize, dtype: DType) -> Self {
        Self { weight: ones(&[size], dtype.clone()), bias: zeros(&[size], dtype), eps: 1e-5 }
    }

    pub fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let output_dtype = x.dtype();
        if output_dtype == DType::Float16 || output_dtype == DType::BFloat16 {
            let x = x.cast(DType::Float32)?;
            let weight = self.weight.cast(DType::Float32)?;
            let bias = self.bias.cast(DType::Float32)?;
            return Ok(x.layernorm(-1, self.eps)?.try_mul(&weight)?.try_add(&bias)?.cast(output_dtype)?);
        }

        let normed = x.layernorm(-1, self.eps)?;
        Ok(normed.try_mul(&self.weight)?.try_add(&self.bias)?)
    }
}

impl HasStateDict for LayerNormWeights {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        state_field!(sd, prefix, self, [weight, bias]);
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        load_state_field!(self, sd, prefix, [weight, bias]);
        Ok(())
    }
}

// ─── Conv1dWeights ──────────────────────────────────────────────────────────

/// 1D convolution with optional bias. Weight shape `[out_ch, in_ch, kernel]`.
#[derive(Clone)]
pub struct Conv1dWeights {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub stride: usize,
    pub padding: usize,
}

impl Conv1dWeights {
    pub fn empty(in_ch: usize, out_ch: usize, kernel: usize, stride: usize, padding: usize, has_bias: bool) -> Self {
        Self::empty_dtype(in_ch, out_ch, kernel, stride, padding, has_bias, DType::Float32)
    }

    pub fn empty_dtype(
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        has_bias: bool,
        dtype: DType,
    ) -> Self {
        let fan_in = in_ch * kernel;
        Self {
            weight: fan_in_uniform(&[out_ch, in_ch, kernel], fan_in, dtype.clone()),
            bias: has_bias.then(|| fan_in_uniform(&[out_ch], fan_in, dtype)),
            stride,
            padding,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let p = self.padding as isize;
        Ok(x.conv2d()
            .weight(&self.weight)
            .maybe_bias(self.bias.as_ref())
            .stride(&[self.stride])
            .padding(&[(p, p)])
            .call()?)
    }
}

impl HasStateDict for Conv1dWeights {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.insert(prefixed(prefix, "weight"), self.weight.clone());
        if let Some(b) = &self.bias {
            sd.insert(prefixed(prefix, "bias"), b.clone());
        }
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.weight = get_tensor(sd, &prefixed(prefix, "weight"))?;
        let bias_key = prefixed(prefix, "bias");
        self.bias = sd.get(&bias_key).cloned();
        Ok(())
    }
}

// ─── Sinusoidal positional embedding ────────────────────────────────────────

/// Compute sinusoidal positional embeddings matching `whisper.model.sinusoids()`.
/// Returns a `[length, channels]` f32 tensor (constant, not learned).
pub fn sinusoids(length: usize, channels: usize, max_timescale: f64) -> Result<Tensor> {
    assert!(channels.is_multiple_of(2), "sinusoids require even channel count");
    let half = channels / 2;
    let log_inc = max_timescale.ln() / (half - 1) as f64;
    let inv_data: Vec<f32> = (0..half).map(|i| (-log_inc * i as f64).exp() as f32).collect();
    let inv = Tensor::from_slice(&inv_data);
    let scaled_time =
        Tensor::arange(0, Some(length as i64), None)?.cast(DType::Float32)?.try_unsqueeze(-1)?.try_mul(&inv)?;
    let sin = scaled_time.sin()?;
    let cos = scaled_time.cos()?;
    Ok(Tensor::cat(&[&sin, &cos], -1)?)
}
