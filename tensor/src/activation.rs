//! Activation functions for neural networks.
//!
//! This module provides common activation functions used in deep learning,
//! including relu, sigmoid, tanh, softmax, and their variants.

use bon::bon;
use morok_ir::{ConstValue, UOp};
use snafu::ResultExt;

use crate::reduce::AxisSpec;
use crate::{Result, Tensor, error::UOpSnafu};

#[bon]
impl Tensor {
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
        let zero = self.zero()?;
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
        // sigmoid(x) = 1 / (1 + exp(-x)) = 1 / (1 + 2^(-x/ln2))
        // Using exp2 matches Tinygrad's implementation for better hardware mapping.
        let scale = self.broadcast_scalar(ConstValue::Float(-1.0 / std::f64::consts::LN_2))?;
        let scaled = self.try_mul(&scale)?;
        let exp2_val = scaled.try_exp2()?;
        let one = exp2_val.one()?;
        let denominator = one.try_add(&exp2_val)?;
        let recip = Self::new(UOp::try_reciprocal(&denominator.uop()).context(UOpSnafu)?);
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
        let one = sig.one()?;
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

        // softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
        // keepdim preserves the reduced axis as size 1 for correct broadcasting
        let max_val = self.max_with().axes(axis.clone()).keepdim(true).call()?;
        let shifted = self.try_sub(&max_val)?;
        let exp_shifted = shifted.try_exp()?;
        let sum_exp = exp_shifted.sum_with().axes(axis).keepdim(true).call()?;

        exp_shifted.try_div(&sum_exp)
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

        // log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))
        // keepdim preserves the reduced axis as size 1 for correct broadcasting
        let max_val = self.max_with().axes(axis.clone()).keepdim(true).call()?;
        let shifted = self.try_sub(&max_val)?;
        let exp_shifted = shifted.try_exp()?;
        let sum_exp = exp_shifted.sum_with().axes(axis).keepdim(true).call()?;
        let log_sum_exp = sum_exp.try_log()?;
        shifted.try_sub(&log_sum_exp)
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
        // Use keepdim internally for correct broadcasting, then drop via max
        let max_keepdim = self.max_with().axes(axis.clone()).keepdim(true).call()?;
        let shifted = self.try_sub(&max_keepdim)?;
        let exp_shifted = shifted.try_exp()?;
        let sum_exp = exp_shifted.sum_with().axes(axis.clone()).keepdim(true).call()?;
        let log_sum = sum_exp.try_log()?;
        let result_keepdim = max_keepdim.try_add(&log_sum)?;

        // Drop the keepdim axis — max over size-1 dim is effectively a squeeze
        result_keepdim.max(axis)
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

    /// Exact GELU: `0.5 * x * (1 + erf(x / sqrt(2)))`.
    pub fn gelu_exact(&self) -> Result<Self> {
        let dtype = self.uop().dtype();
        let half = Tensor::const_(0.5f64, dtype.clone());
        let one = Tensor::const_(1.0f64, dtype.clone());
        let sqrt2 = Tensor::const_(std::f64::consts::SQRT_2, dtype);
        half.try_mul(self)?.try_mul(&one.try_add(&self.try_div(&sqrt2)?.erf()?)?)
    }

    /// Hard Sigmoid: `clamp(alpha * x + beta, 0, 1)`.
    ///
    /// Piecewise linear approximation of sigmoid. Faster to compute.
    ///
    /// # Arguments
    /// * `alpha` - Slope (default 0.2 in ONNX)
    /// * `beta` - Offset (default 0.5 in ONNX)
    pub fn hard_sigmoid(&self, alpha: f64, beta: f64) -> Result<Self> {
        let alpha_t = self.broadcast_scalar(ConstValue::Float(alpha))?;
        let beta_t = self.broadcast_scalar(ConstValue::Float(beta))?;
        let zero = self.broadcast_scalar(ConstValue::Float(0.0))?;
        let one = self.broadcast_scalar(ConstValue::Float(1.0))?;
        let ax = alpha_t.try_mul(self)?;
        let axb = ax.try_add(&beta_t)?;
        // clamp(axb, 0, 1) = max(0, min(1, axb))
        let clamped_low = axb.maximum(&zero)?;
        clamped_low.minimum(&one)
    }

    /// Leaky ReLU: `x if x > 0, alpha * x otherwise`.
    ///
    /// # Arguments
    /// * `alpha` - Negative slope (default 0.01 in ONNX)
    pub fn leaky_relu(&self, alpha: f64) -> Result<Self> {
        let zero = self.zero()?;
        let alpha_t = self.broadcast_scalar(ConstValue::Float(alpha))?;
        let condition = self.try_gt(&zero)?;
        let neg_branch = alpha_t.try_mul(self)?;
        self.where_(&condition, &neg_branch)
    }

    /// PReLU: `x if x > 0, slope * x otherwise`.
    ///
    /// Like LeakyReLU but with a learned per-channel slope.
    pub fn prelu(&self, slope: &Tensor) -> Result<Self> {
        let zero = self.zero()?;
        let condition = self.try_gt(&zero)?;
        let neg_branch = self.try_mul(slope)?;
        self.where_(&condition, &neg_branch)
    }

    /// Thresholded ReLU: `x if x > alpha, 0 otherwise`.
    ///
    /// # Arguments
    /// * `alpha` - Threshold (default 1.0 in ONNX)
    pub fn thresholded_relu(&self, alpha: f64) -> Result<Self> {
        let alpha_t = self.broadcast_scalar(ConstValue::Float(alpha))?;
        let zero = self.zero()?;
        let condition = self.try_gt(&alpha_t)?;
        self.where_(&condition, &zero)
    }

    /// ELU: `x if x > 0, alpha * (exp(x) - 1) otherwise`.
    ///
    /// # Arguments
    /// * `alpha` - Scale for negative part (default 1.0 in ONNX)
    pub fn elu(&self, alpha: f64) -> Result<Self> {
        let zero = self.zero()?;
        let one = self.one()?;
        let alpha_t = self.broadcast_scalar(ConstValue::Float(alpha))?;
        let condition = self.try_gt(&zero)?;
        let exp_minus_1 = self.try_exp()?.try_sub(&one)?;
        let neg_branch = alpha_t.try_mul(&exp_minus_1)?;
        self.where_(&condition, &neg_branch)
    }

    /// SELU: `gamma * (alpha * exp(x) - alpha) if x <= 0, gamma * x if x > 0`.
    ///
    /// Self-normalizing activation with fixed constants.
    ///
    /// # Arguments
    /// * `alpha` - Default 1.6732632...
    /// * `gamma` - Default 1.0507010...
    pub fn selu(&self, alpha: f64, gamma: f64) -> Result<Self> {
        let zero = self.zero()?;
        let alpha_t = self.broadcast_scalar(ConstValue::Float(alpha))?;
        let gamma_t = self.broadcast_scalar(ConstValue::Float(gamma))?;
        let condition = self.try_ge(&zero)?;
        // neg: alpha * exp(x) - alpha
        let neg_branch =
            alpha_t.try_mul(&self.try_exp()?)?.try_sub(&self.broadcast_scalar(ConstValue::Float(alpha))?)?;
        let selected = self.where_(&condition, &neg_branch)?;
        gamma_t.try_mul(&selected)
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

    /// Gated Linear Unit: splits `self` along `dim` into two halves,
    /// returns `first_half * sigmoid(second_half)`.
    pub fn glu(&self, dim: isize) -> Result<Self> {
        let shape = self.shape()?;
        let ndim = shape.len();
        let axis = if dim < 0 { (ndim as isize + dim) as usize } else { dim as usize };
        let full_size = shape[axis].as_const().expect("GLU dim must be concrete");
        assert!(full_size % 2 == 0, "GLU dimension must be even, got {full_size}");
        let half = full_size / 2;
        let halves = self.split(&[half, half], dim)?;
        let gate = halves[1].sigmoid()?;
        halves[0].try_mul(&gate)
    }

    /// Softplus: `log(1 + exp(beta*x)) / beta`, numerically stable via logaddexp.
    pub fn softplus(&self, beta: f64) -> Result<Self> {
        let beta_t = self.broadcast_scalar(ConstValue::Float(beta))?;
        let scaled = self.try_mul(&beta_t)?;
        let zero = self.zero()?;
        let inv_beta = self.broadcast_scalar(ConstValue::Float(1.0 / beta))?;
        // logaddexp(scaled, 0) = max(scaled, 0) + log(exp(scaled - max) + exp(0 - max))
        let m = scaled.maximum(&zero)?;
        let stable = scaled.try_sub(&m)?.try_exp()?.try_add(&zero.try_sub(&m)?.try_exp()?)?.try_log()?.try_add(&m)?;
        stable.try_mul(&inv_beta)
    }

    /// Mish: `x * tanh(softplus(x))`.
    pub fn mish(&self) -> Result<Self> {
        self.try_mul(&self.softplus(1.0)?.tanh()?)
    }

    /// ReLU6: `relu(x) - relu(x-6)` = `clamp(x, 0, 6)`.
    pub fn relu6(&self) -> Result<Self> {
        let six = self.broadcast_scalar(ConstValue::Int(6))?;
        let relu_x = self.relu()?;
        let relu_x6 = self.try_sub(&six)?.relu()?;
        relu_x.try_sub(&relu_x6)
    }

    /// HardSwish: `x * relu6(x+3) / 6`.
    pub fn hardswish(&self) -> Result<Self> {
        let three = self.broadcast_scalar(ConstValue::Int(3))?;
        let six = self.broadcast_scalar(ConstValue::Int(6))?;
        let r6 = self.try_add(&three)?.relu6()?;
        self.try_mul(&r6)?.try_div(&six)
    }

    /// Softsign: `x / (1 + |x|)`.
    pub fn softsign(&self) -> Result<Self> {
        let one = self.one()?;
        let denom = one.try_add(&self.try_abs()?)?;
        self.try_div(&denom)
    }

    /// CELU: `max(0, x) + min(0, alpha*(exp(x/alpha)-1))`.
    pub fn celu(&self, alpha: f64) -> Result<Self> {
        let zero = self.zero()?;
        let one = self.one()?;
        let alpha_t = self.broadcast_scalar(ConstValue::Float(alpha))?;
        let pos = self.maximum(&zero)?;
        let neg = alpha_t.try_mul(&self.try_div(&alpha_t)?.try_exp()?.try_sub(&one)?)?.minimum(&zero)?;
        pos.try_add(&neg)
    }

    /// Batch Normalization.
    ///
    /// Applies: `y = scale * (x - mean) * invstd + bias`
    /// where `invstd = 1 / sqrt(var + epsilon)`
    ///
    /// This is the inference mode batchnorm (no running stats update).
    /// The caller provides pre-computed mean and inverse standard deviation.
    ///
    /// # Arguments
    /// * `scale` - Gamma/weight parameter (optional, defaults to 1)
    /// * `bias` - Beta parameter (optional, defaults to 0)
    /// * `mean` - Running mean
    /// * `invstd` - Inverse standard deviation (1 / sqrt(var + eps))
    /// * `axis` - Axis/axes to normalize over (default: 1 for NCHW)
    ///
    /// # Examples
    /// ```ignore
    /// let x = Tensor::randn(&[8, 4, 16, 16]);
    /// let mean = x.mean(AxisSpec::Multiple(vec![0, 2, 3]))?;
    /// let var = x.var(AxisSpec::Multiple(vec![0, 2, 3]))?;
    /// let eps = Tensor::from_slice([1e-5]);
    /// let invstd = var.try_add(&eps)?.try_rsqrt()?;
    /// let normalized = x.batchnorm().mean(&mean).invstd(&invstd).call()?;
    /// ```
    #[builder]
    pub fn batchnorm(
        &self,
        scale: Option<&Tensor>,
        bias: Option<&Tensor>,
        mean: &Tensor,
        invstd: &Tensor,
        #[builder(default = AxisSpec::Single(1))] axis: AxisSpec,
    ) -> Result<Self> {
        let shape = self.shape()?;

        // Build broadcast shape: keep axis dimensions, others become 1
        let axis_indices: std::collections::HashSet<usize> = match &axis {
            AxisSpec::All => (0..shape.len()).collect(),
            AxisSpec::Single(a) => {
                let a = if *a < 0 { (shape.len() as isize + *a) as usize } else { *a as usize };
                std::iter::once(a).collect()
            }
            AxisSpec::Multiple(axes) => {
                axes.iter().map(|&a| if a < 0 { (shape.len() as isize + a) as usize } else { a as usize }).collect()
            }
        };

        let broadcast_shape: Vec<isize> = shape
            .iter()
            .enumerate()
            .map(|(i, dim)| if axis_indices.contains(&i) { dim.as_const().unwrap_or(1) as isize } else { 1 })
            .collect();

        // x - mean (reshape mean to broadcast shape, like Tinygrad does)
        let mean_reshaped = mean.try_reshape(&broadcast_shape)?;
        let x_centered = self.try_sub(&mean_reshaped)?;

        // (x - mean) * invstd
        let invstd_reshaped = invstd.try_reshape(&broadcast_shape)?;
        let mut result = x_centered.try_mul(&invstd_reshaped)?;

        // scale * (x - mean) * invstd
        if let Some(w) = scale {
            let w_reshaped = w.try_reshape(&broadcast_shape)?;
            result = result.try_mul(&w_reshaped)?;
        }

        // scale * (x - mean) * invstd + bias
        if let Some(b) = bias {
            let b_reshaped = b.try_reshape(&broadcast_shape)?;
            result = result.try_add(&b_reshaped)?;
        }

        Ok(result)
    }
}
