//! Mathematical operations for tensors.
//!
//! This module provides:
//! - Trigonometric functions: sin, cos, tan
//! - Rounding functions: floor, ceil, round, trunc
//! - Advanced math: erf (error function), reciprocal, square, sign

use super::*;

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
}
