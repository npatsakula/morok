//! Activation functions for neural networks.
//!
//! This module provides common activation functions used in deep learning,
//! including relu, sigmoid, tanh, softmax, and their variants.

use morok_ir::{ConstValue, UOp};

use crate::{Result, Tensor, reduce::AxisSpec};

impl Tensor {
    /// Helper to broadcast a scalar constant to match this tensor's shape.
    ///
    /// Uses the broadcast_to method which supports symbolic shapes.
    fn broadcast_scalar(&self, value: ConstValue) -> Result<Self> {
        let shape = self.shape()?;
        let scalar = Self::new(UOp::const_(self.uop.dtype(), value));
        scalar.broadcast_to(&shape)
    }

    /// Helper to broadcast this tensor to a target shape.
    ///
    /// Uses the broadcast_to method which supports symbolic shapes.
    fn broadcast_to_shape(&self, target_shape: &morok_ir::shape::Shape) -> Result<Self> {
        self.broadcast_to(target_shape)
    }

    /// Rectified Linear Unit: `max(0, x)`.
    ///
    /// ReLU is one of the most common activation functions in deep learning.
    /// It's simple, efficient, and helps mitigate the vanishing gradient problem.
    ///
    /// # Examples
    /// ```ignore
    /// let x = Tensor::from_slice(&[-2.0f32, -1.0, 0.0, 1.0, 2.0]);
    /// let y = x.relu()?;
    /// // y = [0.0, 0.0, 0.0, 1.0, 2.0]
    /// ```
    pub fn relu(&self) -> Result<Self> {
        // relu(x) = where(x > 0, x, 0)
        let zero = self.broadcast_scalar(ConstValue::Int(0))?;
        let condition = self.try_gt(&zero)?;
        self.where_(&condition, &zero)
    }

    /// Sigmoid activation: `1 / (1 + exp(-x))`.
    ///
    /// Maps input to range (0, 1), commonly used for binary classification.
    ///
    /// # Examples
    /// ```ignore
    /// let x = Tensor::from_slice(&[-2.0f32, -1.0, 0.0, 1.0, 2.0]);
    /// let y = x.sigmoid()?;
    /// // y ≈ [0.119, 0.268, 0.5, 0.731, 0.880]
    /// ```
    pub fn sigmoid(&self) -> Result<Self> {
        // sigmoid(x) = 1 / (1 + exp(-x))
        // Equivalent to: (1 + exp(-x)).reciprocal()
        let neg_x = self.try_neg()?;
        let exp_neg_x = neg_x.try_exp()?;
        let one = exp_neg_x.broadcast_scalar(ConstValue::Int(1))?;
        let denominator = one.try_add(&exp_neg_x)?;

        // Use reciprocal for 1 / denominator
        let recip = Self::new(UOp::reciprocal_op(denominator.uop));
        Ok(recip)
    }

    /// Hyperbolic tangent: `tanh(x)`.
    ///
    /// Maps input to range (-1, 1), centered at zero.
    ///
    /// # Examples
    /// ```ignore
    /// let x = Tensor::from_slice(&[-2.0f32, -1.0, 0.0, 1.0, 2.0]);
    /// let y = x.tanh()?;
    /// // y ≈ [-0.964, -0.762, 0.0, 0.762, 0.964]
    /// ```
    pub fn tanh(&self) -> Result<Self> {
        // Check if tanh is a UOp primitive
        // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
        // Or: tanh(x) = 2*sigmoid(2x) - 1
        let two = self.broadcast_scalar(ConstValue::Int(2))?;
        let two_x = self.try_mul(&two)?;
        let sig = two_x.sigmoid()?;
        let two_sig = two.try_mul(&sig)?;
        let one = sig.broadcast_scalar(ConstValue::Int(1))?;
        two_sig.try_sub(&one)
    }

    /// Softmax activation: `exp(x - max(x)) / sum(exp(x - max(x)))`.
    ///
    /// Converts logits to probability distribution over specified axis.
    /// Numerically stable implementation using max subtraction.
    ///
    /// # Arguments
    /// * `axis` - Axis along which to compute softmax (default: -1, last axis)
    ///
    /// # Examples
    /// ```ignore
    /// let logits = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0]);
    /// let probs = logits.softmax(-1)?;
    /// // sum(probs) = 1.0, probs[i] > 0 for all i
    /// ```
    pub fn softmax(&self, axis: impl Into<AxisSpec>) -> Result<Self> {
        let axis = axis.into();

        // For 1D tensors (most common case), we can simplify
        // softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
        let max_val = self.max(axis.clone())?; // Scalar
        let max_broadcasted = max_val.broadcast_to_shape(&self.shape()?)?;
        let shifted = self.try_sub(&max_broadcasted)?;
        let exp_shifted = shifted.try_exp()?;
        let sum_exp = exp_shifted.sum(axis)?; // Scalar
        let sum_broadcasted = sum_exp.broadcast_to_shape(&self.shape()?)?;

        exp_shifted.try_div(&sum_broadcasted)
    }

    /// Log-softmax activation: `log(softmax(x))`.
    ///
    /// Numerically stable implementation: `x - max(x) - log(sum(exp(x - max(x))))`.
    ///
    /// More numerically stable than computing `log(softmax(x))` separately.
    ///
    /// # Arguments
    /// * `axis` - Axis along which to compute log-softmax (default: -1, last axis)
    ///
    /// # Examples
    /// ```ignore
    /// let logits = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0]);
    /// let log_probs = logits.log_softmax(-1)?;
    /// // More numerically stable than logits.softmax(-1)?.try_log()
    /// ```
    pub fn log_softmax(&self, axis: impl Into<AxisSpec>) -> Result<Self> {
        let axis = axis.into();

        // log_softmax(x) = x - logsumexp(x, axis)
        let logsumexp = self.logsumexp(axis)?;
        self.try_sub(&logsumexp)
    }

    /// Log-sum-exp: `log(sum(exp(x)))`.
    ///
    /// Numerically stable implementation: `max(x) + log(sum(exp(x - max(x))))`.
    ///
    /// # Arguments
    /// * `axis` - Axis along which to compute logsumexp
    ///
    /// # Examples
    /// ```ignore
    /// let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0]);
    /// let lse = x.logsumexp(-1)?;
    /// ```
    pub fn logsumexp(&self, axis: impl Into<AxisSpec>) -> Result<Self> {
        let axis = axis.into();

        // logsumexp(x) = max(x) + log(sum(exp(x - max(x))))
        let max_val = self.max(axis.clone())?;
        let max_broadcasted = max_val.broadcast_to_shape(&self.shape()?)?;
        let shifted = self.try_sub(&max_broadcasted)?;
        let exp_shifted = shifted.try_exp()?;
        let sum_exp = exp_shifted.sum(axis)?;
        let log_sum = sum_exp.try_log()?;

        // Result is scalar, broadcast back if needed
        let result = max_val.try_add(&log_sum)?;
        result.broadcast_to_shape(&self.shape()?)
    }

    /// GELU activation (Gaussian Error Linear Unit).
    ///
    /// Smooth approximation: `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))`.
    ///
    /// GELU is the standard activation for Transformer models (BERT, GPT, etc.).
    ///
    /// # Examples
    /// ```ignore
    /// let x = Tensor::from_slice(&[-2.0f32, -1.0, 0.0, 1.0, 2.0]);
    /// let y = x.gelu()?;
    /// ```
    pub fn gelu(&self) -> Result<Self> {
        // gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        // sqrt(2/π) ≈ 0.7978845608

        let half = self.broadcast_scalar(ConstValue::Float(0.5))?;
        let one = self.broadcast_scalar(ConstValue::Float(1.0))?;
        let coef1 = self.broadcast_scalar(ConstValue::Float(0.7978845608))?;
        let coef2 = self.broadcast_scalar(ConstValue::Float(0.044715))?;

        // x^3
        let x_squared = self.try_mul(self)?;
        let x_cubed = x_squared.try_mul(self)?;

        // 0.044715 * x^3
        let cubic_term = coef2.try_mul(&x_cubed)?;

        // x + 0.044715 * x^3
        let inner = self.try_add(&cubic_term)?;

        // sqrt(2/π) * (x + 0.044715 * x^3)
        let scaled = coef1.try_mul(&inner)?;

        // tanh(...)
        let tanh_part = scaled.tanh()?;

        // 1 + tanh(...)
        let one_plus_tanh = one.try_add(&tanh_part)?;

        // x * (1 + tanh(...))
        let x_times = self.try_mul(&one_plus_tanh)?;

        // 0.5 * x * (1 + tanh(...))
        half.try_mul(&x_times)
    }

    /// Swish/SiLU activation: `x * sigmoid(x)`.
    ///
    /// Also known as SiLU (Sigmoid Linear Unit).
    /// Used in modern CNN architectures and some Transformers.
    ///
    /// # Examples
    /// ```ignore
    /// let x = Tensor::from_slice(&[-2.0f32, -1.0, 0.0, 1.0, 2.0]);
    /// let y = x.swish()?;
    /// ```
    pub fn swish(&self) -> Result<Self> {
        // swish(x) = x * sigmoid(x)
        let sig = self.sigmoid()?;
        self.try_mul(&sig)
    }

    /// Alias for `swish` (matches PyTorch naming).
    pub fn silu(&self) -> Result<Self> {
        self.swish()
    }
}
