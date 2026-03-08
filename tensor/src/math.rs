//! Mathematical operations for tensors.
//!
//! This module provides:
//! - Trigonometric functions: sin, cos, tan
//! - Rounding functions: floor, ceil, round, trunc
//! - Advanced math: erf (error function), reciprocal, square, sign

use morok_ir::ConstValue;

use super::*;

/// Horner's method for polynomial evaluation: `coeffs[0]*x^(n-1) + ... + coeffs[n-1]`.
fn poly_n(x: &Tensor, coefficients: &[f64]) -> Result<Tensor> {
    let mut acc = x.broadcast_scalar(ConstValue::Float(coefficients[0]))?;
    for &c in &coefficients[1..] {
        let c_t = x.broadcast_scalar(ConstValue::Float(c))?;
        acc = acc.try_mul(x)?.try_add(&c_t)?;
    }
    Ok(acc)
}

impl Tensor {
    // =========================================================================
    // Trigonometric Functions
    // =========================================================================

    /// Sine function: sin(x).
    ///
    /// Computes the sine of each element. Requires float dtype.
    ///
    /// # Examples
    /// ```ignore
    /// use std::f32::consts::PI;
    /// let t = Tensor::from_slice(&[0.0f32, PI/2.0, PI]);
    /// let result = t.sin()?;  // [0, 1, 0]
    /// ```
    ///
    /// # Errors
    /// Returns error if dtype is not float.
    #[track_caller]
    pub fn sin(&self) -> Result<Tensor> {
        self.uop().try_sin().map(Self::new).context(UOpSnafu)
    }

    /// Cosine function: cos(x).
    ///
    /// Computes the cosine of each element. Requires float dtype.
    ///
    /// # Examples
    /// ```ignore
    /// use std::f32::consts::PI;
    /// let t = Tensor::from_slice(&[0.0f32, PI/2.0, PI]);
    /// let result = t.cos()?;  // [1, 0, -1]
    /// ```
    ///
    /// # Errors
    /// Returns error if dtype is not float.
    #[track_caller]
    pub fn cos(&self) -> Result<Tensor> {
        self.uop().try_cos().map(Self::new).context(UOpSnafu)
    }

    /// Tangent function: tan(x).
    ///
    /// Computes the tangent of each element. Requires float dtype.
    ///
    /// # Examples
    /// ```ignore
    /// use std::f32::consts::PI;
    /// let t = Tensor::from_slice(&[0.0f32, PI/4.0]);
    /// let result = t.tan()?;  // [0, 1]
    /// ```
    ///
    /// # Errors
    /// Returns error if dtype is not float.
    #[track_caller]
    pub fn tan(&self) -> Result<Tensor> {
        self.uop().try_tan().map(Self::new).context(UOpSnafu)
    }

    // =========================================================================
    // Rounding Functions
    // =========================================================================

    /// Floor function: round towards -∞.
    ///
    /// Returns the largest integer less than or equal to each element.
    /// For integer dtypes, returns the tensor unchanged.
    ///
    /// # Examples
    /// ```ignore
    /// let t = Tensor::from_slice(&[1.2f32, -1.2, 2.8, -2.8]);
    /// let result = t.floor()?;  // [1.0, -2.0, 2.0, -3.0]
    /// ```
    #[track_caller]
    pub fn floor(&self) -> Result<Tensor> {
        Ok(Self::new(UOp::floor(self.uop())))
    }

    /// Ceiling function: round towards +∞.
    ///
    /// Returns the smallest integer greater than or equal to each element.
    /// For integer dtypes, returns the tensor unchanged.
    ///
    /// # Examples
    /// ```ignore
    /// let t = Tensor::from_slice(&[1.2f32, -1.2, 2.8, -2.8]);
    /// let result = t.ceil()?;  // [2.0, -1.0, 3.0, -2.0]
    /// ```
    #[track_caller]
    pub fn ceil(&self) -> Result<Tensor> {
        Ok(Self::new(UOp::ceil(self.uop())))
    }

    /// Round function: round to nearest integer (half to even).
    ///
    /// Rounds each element to the nearest integer. Ties are rounded to the nearest even number.
    /// For integer dtypes, returns the tensor unchanged.
    ///
    /// # Examples
    /// ```ignore
    /// let t = Tensor::from_slice(&[1.2f32, 1.5, 2.5, -1.5]);
    /// let result = t.round()?;  // [1.0, 2.0, 2.0, -2.0]
    /// ```
    #[track_caller]
    pub fn round(&self) -> Result<Tensor> {
        Ok(Self::new(UOp::round(self.uop())))
    }

    /// Truncate function: round towards zero.
    ///
    /// Removes the fractional part, rounding towards zero.
    /// For integer dtypes, returns the tensor unchanged.
    ///
    /// # Examples
    /// ```ignore
    /// let t = Tensor::from_slice(&[1.2f32, -1.2, 2.8, -2.8]);
    /// let result = t.trunc()?;  // [1.0, -1.0, 2.0, -2.0]
    /// ```
    #[track_caller]
    pub fn trunc(&self) -> Result<Tensor> {
        Ok(Self::new(UOp::trunc(self.uop())))
    }

    // =========================================================================
    // Advanced Math Functions
    // =========================================================================

    /// Error function: erf(x).
    ///
    /// Computes the error function (Gauss error function) of each element.
    /// Requires float dtype. Critical for GELU activation.
    ///
    /// # Examples
    /// ```ignore
    /// let t = Tensor::from_slice(&[-1.0f32, 0.0, 1.0]);
    /// let result = t.erf()?;  // [-0.8427, 0, 0.8427]
    /// ```
    ///
    /// # Errors
    /// Returns error if dtype is not float.
    #[track_caller]
    pub fn erf(&self) -> Result<Tensor> {
        self.uop().erf().map(Self::new).context(UOpSnafu)
    }

    /// Reciprocal: 1/x.
    ///
    /// Computes the reciprocal of each element.
    ///
    /// # Examples
    /// ```ignore
    /// let t = Tensor::from_slice(&[1.0f32, 2.0, 4.0]);
    /// let result = t.reciprocal()?;  // [1.0, 0.5, 0.25]
    /// ```
    #[track_caller]
    pub fn reciprocal(&self) -> Result<Tensor> {
        UOp::try_reciprocal(&self.uop()).map(Self::new).context(UOpSnafu)
    }

    /// Square: x².
    ///
    /// Computes the square of each element.
    ///
    /// # Examples
    /// ```ignore
    /// let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, -4.0]);
    /// let result = t.square()?;  // [1.0, 4.0, 9.0, 16.0]
    /// ```
    #[track_caller]
    pub fn square(&self) -> Result<Tensor> {
        Ok(Self::new(self.uop().square()))
    }

    /// Sign function: -1 for negative, 0 for zero, 1 for positive.
    ///
    /// Returns the sign of each element.
    ///
    /// # Examples
    /// ```ignore
    /// let t = Tensor::from_slice(&[-5.0f32, 0.0, 3.0, -0.0]);
    /// let result = t.sign()?;  // [-1.0, 0.0, 1.0, 0.0]
    /// ```
    #[track_caller]
    pub fn sign(&self) -> Result<Tensor> {
        Ok(Self::new(self.uop().sign()))
    }

    /// Linear interpolation: `self + (end - self) * weight`.
    #[track_caller]
    pub fn lerp(&self, end: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let diff = end.try_sub(self)?;
        self.try_add(&diff.try_mul(weight)?)
    }

    // =========================================================================
    // NaN / Infinity Detection
    // =========================================================================

    /// Returns `true` where elements are NaN: `self != self`.
    #[track_caller]
    pub fn isnan(&self) -> Result<Tensor> {
        self.try_ne(self)
    }

    /// Returns `true` where elements are infinite.
    ///
    /// Uses `(x - x).isnan() & !x.isnan()` to detect infinity without direct
    /// equality comparison (which our range analyzer handles conservatively).
    /// For directional detection, further filters by sign.
    #[track_caller]
    pub fn isinf(&self, detect_positive: bool, detect_negative: bool) -> Result<Tensor> {
        // inf - inf = NaN, finite - finite = 0, NaN - NaN = NaN
        // So (x-x).isnan() is true for both inf and NaN; exclude NaN.
        let is_inf = self.try_sub(self)?.isnan()?.try_sub(&self.isnan()?)?;
        let zero = self.zero()?;
        match (detect_positive, detect_negative) {
            (true, true) => Ok(is_inf),
            (true, false) => is_inf.try_mul(&self.try_gt(&zero)?),
            (false, true) => is_inf.try_mul(&self.try_lt(&zero)?),
            (false, false) => Ok(zero.cast(morok_dtype::DType::Bool)?),
        }
    }

    // =========================================================================
    // Hyperbolic Functions
    // =========================================================================

    /// Hyperbolic sine: `(exp(x) - exp(-x)) / 2`.
    #[track_caller]
    pub fn sinh(&self) -> Result<Tensor> {
        let exp_pos = self.try_exp()?;
        let exp_neg = self.try_neg()?.try_exp()?;
        let two = self.broadcast_scalar(ConstValue::Int(2))?;
        exp_pos.try_sub(&exp_neg)?.try_div(&two)
    }

    /// Hyperbolic cosine: `(exp(x) + exp(-x)) / 2`.
    #[track_caller]
    pub fn cosh(&self) -> Result<Tensor> {
        let exp_pos = self.try_exp()?;
        let exp_neg = self.try_neg()?.try_exp()?;
        let two = self.broadcast_scalar(ConstValue::Int(2))?;
        exp_pos.try_add(&exp_neg)?.try_div(&two)
    }

    // =========================================================================
    // Inverse Hyperbolic Functions
    // =========================================================================

    /// Inverse hyperbolic sine: `log(x + sqrt(x² + 1))`.
    #[track_caller]
    pub fn asinh(&self) -> Result<Tensor> {
        let one = self.one()?;
        let inner = self.square()?.try_add(&one)?.try_sqrt()?;
        self.try_add(&inner)?.try_log()
    }

    /// Inverse hyperbolic cosine: `log(x + sqrt(x² - 1))`.
    #[track_caller]
    pub fn acosh(&self) -> Result<Tensor> {
        let one = self.one()?;
        let inner = self.square()?.try_sub(&one)?.try_sqrt()?;
        self.try_add(&inner)?.try_log()
    }

    /// Inverse hyperbolic tangent: `0.5 * log((1+x)/(1-x))`.
    #[track_caller]
    pub fn atanh(&self) -> Result<Tensor> {
        let one = self.one()?;
        let half = self.broadcast_scalar(ConstValue::Float(0.5))?;
        let num = one.try_add(self)?;
        let den = one.try_sub(self)?;
        half.try_mul(&num.try_div(&den)?.try_log()?)
    }

    // =========================================================================
    // Inverse Trigonometric Functions
    // =========================================================================

    /// Arcsine using polynomial approximation (Abramowitz & Stegun 4.4.46).
    #[track_caller]
    pub fn asin(&self) -> Result<Tensor> {
        let coefficients = [
            -0.0012624911,
            0.0066700901,
            -0.0170881256,
            0.0308918810,
            -0.0501743046,
            0.0889789874,
            -0.2145988016,
            1.5707963050,
        ];
        let abs_x = self.try_abs()?;
        let one = self.one()?;
        let half_pi = self.broadcast_scalar(ConstValue::Float(std::f64::consts::FRAC_PI_2))?;
        let sqrt_part = one.try_sub(&abs_x)?.try_sqrt()?;
        let poly = poly_n(&abs_x, &coefficients)?;
        let x = half_pi.try_sub(&sqrt_part.try_mul(&poly)?)?;
        self.sign()?.try_mul(&x)
    }

    /// Arccosine: `π/2 - asin(x)`.
    #[track_caller]
    pub fn acos(&self) -> Result<Tensor> {
        let half_pi = self.broadcast_scalar(ConstValue::Float(std::f64::consts::FRAC_PI_2))?;
        half_pi.try_sub(&self.asin()?)
    }

    /// Arctangent: `asin(x / sqrt(1 + x²))`.
    #[track_caller]
    pub fn atan(&self) -> Result<Tensor> {
        let one = self.one()?;
        let denom = one.try_add(&self.square()?)?.try_sqrt()?;
        self.try_div(&denom)?.asin()
    }

    // =========================================================================
    // Shrinkage / Thresholding
    // =========================================================================

    /// Shrinkage operator: applies soft/hard thresholding.
    ///
    /// `(x < -λ)*(x+bias) + (x > λ)*(x-bias)`
    #[track_caller]
    pub fn shrink(&self, bias: f64, lambd: f64) -> Result<Tensor> {
        let dtype = self.uop().dtype();
        let neg_lambd = Tensor::const_(-lambd, dtype.clone());
        let pos_lambd = Tensor::const_(lambd, dtype.clone());
        let bias_t = Tensor::const_(bias, dtype.clone());
        let neg_bias = Tensor::const_(-bias, dtype.clone());
        let neg_part = self.try_lt(&neg_lambd)?.cast(dtype.clone())?.try_mul(&self.try_add(&bias_t)?)?;
        let pos_part = self.try_gt(&pos_lambd)?.cast(dtype)?.try_mul(&self.try_add(&neg_bias)?)?;
        neg_part.try_add(&pos_part)
    }
}
