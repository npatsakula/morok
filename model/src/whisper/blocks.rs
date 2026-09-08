//! Whisper building blocks: the mixed-precision linear epilogue and the
//! sinusoidal positional embedding. The layer constructors live in
//! [`crate::init`].

use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::Linear;

use super::error::Result;

/// Whisper's linear forward. OpenAI keeps the matmul accumulator *and* the bias
/// addition in FP32 when activation and weight are both half precision, so the
/// result rounds exactly once, at the final cast. [`svod_tensor::nn::Layer`]'s
/// `forward` has no accumulator-dtype knob, so this stays a free function.
pub(crate) fn linear_forward(layer: &Linear, x: &Tensor) -> Result<Tensor> {
    let half = |dtype: &DType| *dtype == DType::Float16 || *dtype == DType::BFloat16;
    let output_dtype = x.dtype();
    match &layer.bias {
        Some(bias) if half(&output_dtype) && half(&layer.weight.dtype()) => Ok(x
            .linear()
            .weight(&layer.weight)
            .dtype(DType::Float32)
            .call()?
            .try_add(bias.cast(DType::Float32))?
            .cast(output_dtype)),
        _ => Ok(x.linear().weight(&layer.weight).maybe_bias(layer.bias.as_ref()).call()?),
    }
}

/// Sinusoidal positional embeddings matching `whisper.model.sinusoids()`:
/// a `[length, channels]` f32 constant, built in-graph.
pub fn sinusoids(length: usize, channels: usize, max_timescale: f64) -> Result<Tensor> {
    assert!(channels.is_multiple_of(2), "sinusoids require even channel count");
    let half = channels / 2;
    let log_increment = max_timescale.ln() / (half - 1) as f64;
    let inv = Tensor::arange(0, Some(half as i64), None)?.cast(DType::Float32).try_mul(-log_increment)?.try_exp()?;
    let scaled_time =
        Tensor::arange(0, Some(length as i64), None)?.cast(DType::Float32).try_unsqueeze(-1)?.try_mul(&inv)?;
    Ok(Tensor::cat(&[&scaled_time.sin()?, &scaled_time.cos()?], -1)?)
}
