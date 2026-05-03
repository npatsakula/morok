//! Mathematical operations for tensors.
//!
//! This module provides:
//! - Trigonometric functions: sin, cos, tan
//! - Rounding functions: floor, ceil, round, trunc
//! - Advanced math: erf (error function), reciprocal, square, sign

use morok_ir::ConstValue;
use snafu::ResultExt;

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
    /// Detects ±∞ via bitcast to the corresponding unsigned integer type and a
    /// bit-pattern compare. Operating in integer space sidesteps Morok's float
    /// range analysis, which folds `x == ±inf` to false because `dtype_bounds`
    /// returns finite ±max for floats. Tinygrad gets away with the float compare
    /// because its `dtype.min/max` are ±inf.
    #[track_caller]
    pub fn isinf(&self, detect_positive: bool, detect_negative: bool) -> Result<Tensor> {
        use morok_dtype::{DType, ScalarDType};
        let dtype = self.uop().dtype();
        // (uint_bitcast_dtype, +inf bit pattern, -inf bit pattern, abs-mask)
        let (uint_dt, pos_bits, neg_bits, abs_mask): (DType, i64, i64, i64) = match dtype {
            DType::Scalar(ScalarDType::Float16) => (DType::UInt16, 0x7C00, 0xFC00, 0x7FFF),
            DType::Scalar(ScalarDType::BFloat16) => (DType::UInt16, 0x7F80, 0xFF80, 0x7FFF),
            DType::Scalar(ScalarDType::Float32) => (DType::UInt32, 0x7F800000, 0xFF800000_u32 as i64, 0x7FFFFFFF),
            DType::Scalar(ScalarDType::Float64) => {
                (DType::UInt64, 0x7FF0000000000000, 0xFFF0000000000000_u64 as i64, 0x7FFFFFFFFFFFFFFF)
            }
            // Non-float dtypes never have inf.
            _ => return self.zero()?.cast(DType::Bool),
        };

        let bits = self.bitcast(uint_dt)?;
        let pos_pat = bits.broadcast_scalar(ConstValue::Int(pos_bits))?;
        match (detect_positive, detect_negative) {
            (true, true) => {
                // (bits & abs_mask) == +inf bits → matches both +inf and -inf
                let mask = bits.broadcast_scalar(ConstValue::Int(abs_mask))?;
                bits.bitwise_and(&mask)?.try_eq(&pos_pat)
            }
            (true, false) => bits.try_eq(&pos_pat),
            (false, true) => {
                let neg_pat = bits.broadcast_scalar(ConstValue::Int(neg_bits))?;
                bits.try_eq(&neg_pat)
            }
            (false, false) => self.zero()?.cast(DType::Bool),
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

    // =========================================================================
    // Linear Algebra
    // =========================================================================

    /// Matrix determinant via LU decomposition with partial pivoting.
    ///
    /// Input shape: `[..., n, n]`. Output shape: `[...]`.
    /// Batch dimensions are preserved. Uses O(n³) computation with O(n)
    /// graph construction steps (unrolled at compile time).
    #[track_caller]
    pub fn det(&self) -> Result<Tensor> {
        let shape = self.shape()?;
        let ndim = shape.len();
        snafu::ensure!(
            ndim >= 2,
            crate::error::ShapeMismatchSnafu {
                context: "det",
                expected: "at least 2D".to_string(),
                actual: format!("{ndim}D"),
            }
        );
        let n = shape[ndim - 1].as_const().unwrap();
        let m = shape[ndim - 2].as_const().unwrap();
        snafu::ensure!(
            n == m,
            crate::error::ShapeMismatchSnafu {
                context: "det",
                expected: format!("square last two dims, got [{m}, {n}]"),
                actual: format!("[{m}, {n}]"),
            }
        );

        let dtype = self.uop().dtype();
        let float_dt = if dtype.is_float() { dtype.clone() } else { DType::Float32 };

        if n == 0 {
            let batch: Vec<usize> = shape[..ndim - 2].iter().map(|s| s.as_const().unwrap()).collect();
            return if batch.is_empty() {
                Ok(Tensor::const_(1.0, float_dt))
            } else {
                Tensor::full(&batch, 1.0, float_dt)
            };
        }

        // Cast to float for correct division in Gaussian elimination
        let mut a = if dtype.is_float() { self.clone() } else { self.cast(float_dt.clone())? };
        let mut det_val: Option<Tensor> = None;
        let neg_one = Tensor::const_(-1.0, float_dt.clone());
        let one = Tensor::const_(1.0, float_dt.clone());
        let zero_i = Tensor::const_(ConstValue::Int(0), DType::Int32);

        for step in 0..n {
            let cur_n = n - step;
            let cni = cur_n as isize;

            if cur_n > 1 {
                // Partial pivoting: find row with max |a[..., :, 0]|
                let col0 = shrink_last2(&a, ndim, (0, cni), (0, 1))?;
                let max_idx = col0.try_abs()?.argmax_with().axis(Some(-2)).keepdim(true).call()?;

                // Extract max_row via gather
                let mut gather_shape: Vec<isize> = vec![-1; ndim - 2];
                gather_shape.push(1);
                gather_shape.push(cur_n as isize);
                let max_idx_gather = max_idx.try_expand(&gather_shape)?;
                let max_row = a.gather(-2, &max_idx_gather)?;

                // Extract row 0
                let row_0 = shrink_last2(&a, ndim, (0, 1), (0, cni))?;

                // Build row-index mask: shape [1, ..., 1, cur_n, 1] for broadcasting
                let mut row_idx = Tensor::arange(0, Some(cur_n as i64), None)?.try_unsqueeze(-1)?;
                for _ in 0..ndim - 2 {
                    row_idx = row_idx.try_unsqueeze(0)?;
                }
                let mask_0 = row_idx.try_eq(&zero_i)?;
                let mask_max = row_idx.try_eq(&max_idx)?;

                // Swap: row_0 → max_idx position, max_row → row 0 position
                let temp = row_0.where_(&mask_max, &a)?;
                a = max_row.where_(&mask_0, &temp)?;

                // Track sign: flip when a swap actually happened
                let max_idx_scalar = max_idx.try_squeeze(Some(-1))?.try_squeeze(Some(-1))?;
                let swapped = max_idx_scalar.try_ne(&zero_i)?;
                let swap_sign = neg_one.where_(&swapped, &one)?;
                det_val = Some(match det_val {
                    None => swap_sign,
                    Some(d) => d.try_mul(&swap_sign)?,
                });
            }

            // Extract pivot a[..., 0, 0] and accumulate
            let pivot = shrink_last2(&a, ndim, (0, 1), (0, 1))?;
            let pivot_scalar = pivot.try_squeeze(Some(-1))?.try_squeeze(Some(-1))?;
            det_val = Some(match det_val {
                None => pivot_scalar,
                Some(d) => d.try_mul(&pivot_scalar)?,
            });

            if cur_n <= 1 {
                break;
            }

            // Gaussian elimination on the submatrix.
            // Use safe pivot: replace 0 with 1 to avoid div-by-zero NaN.
            // When pivot is 0 the matrix is singular (det=0), already captured
            // in det_val; the elimination result doesn't matter.
            let pivot_is_zero = pivot.try_eq(&Tensor::const_(0.0, float_dt.clone()))?;
            let pivot_safe = one.where_(&pivot_is_zero, &pivot)?;
            let col_below = shrink_last2(&a, ndim, (1, cni), (0, 1))?;
            let factors = col_below.try_div(&pivot_safe)?;
            let row_0_rest = shrink_last2(&a, ndim, (0, 1), (1, cni))?;
            let sub = shrink_last2(&a, ndim, (1, cni), (1, cni))?;
            a = sub.try_sub(&factors.try_mul(&row_0_rest)?)?;
        }

        Ok(det_val.unwrap())
    }
}

/// Shrink only the last two dimensions of a tensor, preserving batch dims.
fn shrink_last2(tensor: &Tensor, ndim: usize, row_range: (isize, isize), col_range: (isize, isize)) -> Result<Tensor> {
    let shape = tensor.shape()?;
    let mut ranges: Vec<(isize, isize)> =
        shape[..ndim - 2].iter().map(|s| (0, s.as_const().unwrap() as isize)).collect();
    ranges.push(row_range);
    ranges.push(col_range);
    tensor.try_shrink(&ranges)
}
