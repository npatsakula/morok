//! Distribution wrappers around `Tensor::rand`.
//!
//! | Method | Formula |
//! |---|---|
//! | `uniform(shape, low, high)` | `(high - low) * rand + low` |
//! | `randn(shape)` | Box-Muller: `cos(2π·u₁) · √(-2·ln(1 - u₂))` |
//! | `normal(shape, mean, std)` | `std · randn + mean` |
//! | `randint(shape, low, high)` | `((high-low)·rand).cast(int32) + low` |
//! | `scaled_uniform(shape)` | `uniform(-1, 1) · prod(shape)^-½` |
//! | `glorot_uniform(shape)` | `uniform(-b, b)`, `b = √(6 / (shape[0] + prod(shape[1..])))` |
//! | `kaiming_uniform(shape, a)` | `uniform(-b, b)`, `b = √(6 / ((1+a²) · prod(shape[1..])))` |
//! | `kaiming_normal(shape, a)` | `randn · √(2 / ((1+a²) · prod(shape[1..])))` |

use svod_dtype::DType;
use svod_ir::ConstValue;

use crate::{ErrorKind, Result, Tensor};

const TWO_PI: f64 = 2.0 * std::f64::consts::PI;

fn fan_in(shape: &[usize]) -> usize {
    // fan_in = prod(shape[1..]). For 1D inputs (e.g., bias) the product over
    // the empty slice is 1.
    shape.iter().skip(1).copied().product::<usize>().max(1)
}

impl Tensor {
    /// Uniform `[low, high)` random tensor, float32, on the default (CPU) device.
    ///
    /// Convenience wrapper around [`Tensor::uniform_with_dtype`] with f32 output.
    #[track_caller]
    pub fn uniform(shape: &[usize], low: f64, high: f64) -> Result<Tensor> {
        origin_call!("uniform");
        Self::uniform_with_dtype(shape, low, high, DType::Float32)
    }

    /// Uniform `[low, high)` random tensor with explicit float dtype.
    ///
    /// Generates a `[0, 1)` sample at f32, scales by `(high - low)`, **casts
    /// to the target dtype**, then adds `low`. Casting before the offset
    /// keeps the addition honest in low-precision targets (f16/bf16) where
    /// `low` might otherwise be lost to rounding if applied at f32.
    #[track_caller]
    pub fn uniform_with_dtype(shape: &[usize], low: f64, high: f64, dtype: DType) -> Result<Tensor> {
        origin_call!("uniform");
        if low >= high {
            return Err(ErrorKind::ParamRange {
                op: "Tensor::uniform",
                param: "low/high",
                value: format!("low={low}, high={high}"),
                constraint: "low < high",
            }
            .into());
        }
        let u = Tensor::rand(shape)?;
        let scale = u.broadcast_scalar(ConstValue::Float(high - low))?;
        let scaled = u.try_mul(&scale)?.cast(dtype);
        let offset = scaled.broadcast_scalar(ConstValue::Float(low))?;
        scaled.try_add(&offset)
    }

    /// Standard normal `N(0, 1)` random tensor (float32, Box-Muller).
    ///
    /// Each output element draws from two `[0, 1)` uniforms via one combined
    /// `rand([2, *shape])` call, so the RNG counter advances exactly once per
    /// `randn` invocation regardless of `shape`.
    #[track_caller]
    pub fn randn(shape: &[usize]) -> Result<Tensor> {
        origin_call!("randn");
        // src = rand([2, *shape])  →  one counter advance for two halves.
        let mut combined_shape: Vec<usize> = Vec::with_capacity(shape.len() + 1);
        combined_shape.push(2);
        combined_shape.extend_from_slice(shape);
        let src = Tensor::rand(&combined_shape)?;

        // u1 = src[0:1, …] reshaped to shape; u2 = src[1:2, …] reshaped to shape.
        let mut shrink_u1: Vec<Option<(isize, isize)>> = Vec::with_capacity(combined_shape.len());
        shrink_u1.push(Some((0, 1)));
        shrink_u1.extend(std::iter::repeat_n(None, shape.len()));
        let mut shrink_u2: Vec<Option<(isize, isize)>> = Vec::with_capacity(combined_shape.len());
        shrink_u2.push(Some((1, 2)));
        shrink_u2.extend(std::iter::repeat_n(None, shape.len()));
        let target_shape: Vec<isize> = shape.iter().map(|&d| d as isize).collect();
        let u1 = src.try_shrink(shrink_u1)?.try_reshape(&target_shape)?;
        let u2 = src.try_shrink(shrink_u2)?.try_reshape(&target_shape)?;

        // Box-Muller: cos(2π·u1) · √(-2·ln(1 - u2))
        let two_pi = u1.broadcast_scalar(ConstValue::Float(TWO_PI))?;
        let theta = u1.try_mul(&two_pi)?.cos()?;
        let one = u2.broadcast_scalar(ConstValue::Float(1.0))?;
        let neg_two = u2.broadcast_scalar(ConstValue::Float(-2.0))?;
        let r = one.try_sub(&u2)?.try_log()?.try_mul(&neg_two)?.try_sqrt()?;
        theta.try_mul(&r)
    }

    /// Normal `N(mean, std)` random tensor. Requires `std >= 0`.
    #[track_caller]
    pub fn normal(shape: &[usize], mean: f64, std: f64) -> Result<Tensor> {
        origin_call!("normal");
        if std < 0.0 {
            return Err(ErrorKind::ParamRange {
                op: "Tensor::normal",
                param: "std",
                value: format!("{std}"),
                constraint: ">= 0",
            }
            .into());
        }
        let z = Tensor::randn(shape)?;
        let std_t = z.broadcast_scalar(ConstValue::Float(std))?;
        let mean_t = z.broadcast_scalar(ConstValue::Float(mean))?;
        z.try_mul(&std_t)?.try_add(&mean_t)
    }

    /// Uniform integer tensor `[low, high)`, dtype `int32`. Requires `low < high`.
    ///
    /// Truncates `(high - low) · rand` to int32 **before** adding `low`.
    /// Casting after the add would truncate-toward-zero asymmetrically for
    /// negative `low` (e.g. `low=-3, rand≈0.005` would yield `-2` instead of
    /// the correct `-3`).
    #[track_caller]
    pub fn randint(shape: &[usize], low: i64, high: i64) -> Result<Tensor> {
        origin_call!("randint");
        if low >= high {
            return Err(ErrorKind::ParamRange {
                op: "Tensor::randint",
                param: "low/high",
                value: format!("low={low}, high={high}"),
                constraint: "low < high",
            }
            .into());
        }
        let scaled = Tensor::rand(shape)?;
        let range = scaled.broadcast_scalar(ConstValue::Float((high - low) as f64))?;
        let truncated = scaled.try_mul(&range)?.cast(DType::Int32);
        let offset = truncated.broadcast_scalar(ConstValue::Int(low))?;
        truncated.try_add(&offset)
    }

    /// `uniform(-1, 1) · prod(shape)^(-½)`. Same dtype contract as `uniform`.
    #[track_caller]
    pub fn scaled_uniform(shape: &[usize]) -> Result<Tensor> {
        origin_call!("scaled_uniform");
        let numel: usize = shape.iter().copied().product::<usize>().max(1);
        let scale = (numel as f64).powf(-0.5);
        let u = Tensor::uniform(shape, -1.0, 1.0)?;
        let scale_t = u.broadcast_scalar(ConstValue::Float(scale))?;
        u.try_mul(&scale_t)
    }

    /// Glorot/Xavier uniform initializer, float32 output.
    #[track_caller]
    pub fn glorot_uniform(shape: &[usize]) -> Result<Tensor> {
        origin_call!("glorot_uniform");
        Self::glorot_uniform_with_dtype(shape, DType::Float32)
    }

    /// Glorot/Xavier uniform initializer with explicit dtype.
    /// `bound = √(6 / (shape[0] + prod(shape[1..]))); uniform(-bound, bound)`.
    #[track_caller]
    pub fn glorot_uniform_with_dtype(shape: &[usize], dtype: DType) -> Result<Tensor> {
        origin_call!("glorot_uniform");
        if shape.is_empty() {
            return Err(ErrorKind::ParamRange {
                op: "Tensor::glorot_uniform",
                param: "shape",
                value: "[]".to_string(),
                constraint: "at least 1D",
            }
            .into());
        }
        let fan_in_v = fan_in(shape);
        let fan_out_v = shape[0];
        let bound = (6.0 / (fan_out_v + fan_in_v) as f64).sqrt();
        Self::uniform_with_dtype(shape, -bound, bound, dtype)
    }

    /// Kaiming/He uniform initializer for ReLU-family activations, float32 output.
    #[track_caller]
    pub fn kaiming_uniform(shape: &[usize], a: f64) -> Result<Tensor> {
        origin_call!("kaiming_uniform");
        Self::kaiming_uniform_with_dtype(shape, a, DType::Float32)
    }

    /// Kaiming/He uniform initializer with explicit dtype.
    ///
    /// `bound = √(6 / ((1 + a²) · prod(shape[1..]))); uniform(-bound, bound)`.
    ///
    /// `a` is the negative slope of the activation:
    /// - `0.0` — plain ReLU (PyTorch default).
    /// - `0.01` — leaky-ReLU with default slope.
    #[track_caller]
    pub fn kaiming_uniform_with_dtype(shape: &[usize], a: f64, dtype: DType) -> Result<Tensor> {
        origin_call!("kaiming_uniform");
        if shape.is_empty() {
            return Err(ErrorKind::ParamRange {
                op: "Tensor::kaiming_uniform",
                param: "shape",
                value: "[]".to_string(),
                constraint: "at least 1D",
            }
            .into());
        }
        let bound = (6.0 / ((1.0 + a * a) * fan_in(shape) as f64)).sqrt();
        Self::uniform_with_dtype(shape, -bound, bound, dtype)
    }

    /// Kaiming/He normal initializer for ReLU-family activations.
    /// `std = √(2 / ((1 + a²) · prod(shape[1..]))); randn · std`.
    #[track_caller]
    pub fn kaiming_normal(shape: &[usize], a: f64) -> Result<Tensor> {
        origin_call!("kaiming_normal");
        if shape.is_empty() {
            return Err(ErrorKind::ParamRange {
                op: "Tensor::kaiming_normal",
                param: "shape",
                value: "[]".to_string(),
                constraint: "at least 1D",
            }
            .into());
        }
        let std = (2.0 / ((1.0 + a * a) * fan_in(shape) as f64)).sqrt();
        let z = Tensor::randn(shape)?;
        let std_t = z.broadcast_scalar(ConstValue::Float(std))?;
        z.try_mul(&std_t)
    }
}
