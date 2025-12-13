//! Conditional and selection operations for tensors.
//!
//! This module provides element-wise conditional operations like where, maximum,
//! minimum, and clamp that are fundamental for many ML operations.

use bon::bon;
use morok_ir::UOp;
use snafu::ResultExt;

use crate::{Result, Tensor, error::UOpSnafu};

#[bon]
impl Tensor {
    /// Element-wise conditional selection: `condition ? self : other`.
    ///
    /// For each element, returns `self[i]` if `condition[i]` is true, else `other[i]`.
    ///
    /// # Arguments
    /// * `condition` - Boolean tensor (dtype should be Bool or will be treated as boolean)
    /// * `other` - Alternative value tensor
    ///
    /// # Shape Requirements
    /// All three tensors (self, condition, other) must be broadcastable to the same shape.
    ///
    /// # Examples
    /// ```ignore
    /// let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0]);
    /// let condition = &x.gt(&Tensor::from_slice(&[2.0f32]))?; // [false, false, true, true]
    /// let zeros = Tensor::from_slice(&[0.0f32]);
    ///
    /// // Replace values > 2.0 with the original value, else 0
    /// let result = x.where_(condition, &zeros)?;
    /// // result = [0.0, 0.0, 3.0, 4.0]
    /// ```
    pub fn where_(&self, condition: &Tensor, other: &Tensor) -> Result<Self> {
        let result = UOp::try_where(condition.uop(), self.uop(), other.uop()).context(UOpSnafu)?;
        Ok(Self::new(result))
    }

    /// Element-wise maximum: `max(self, other)`.
    ///
    /// Returns the element-wise maximum of two tensors.
    /// This is NOT a reduction - it returns a tensor of the same shape.
    ///
    /// # Shape Requirements
    /// Both tensors must be broadcastable to the same shape.
    ///
    /// # Examples
    /// ```ignore
    /// let a = Tensor::from_slice(&[1.0f32, 5.0, 3.0]);
    /// let b = Tensor::from_slice(&[2.0f32, 3.0, 4.0]);
    /// let result = a.maximum(&b)?;
    /// // result = [2.0, 5.0, 4.0]
    /// ```
    pub fn maximum(&self, other: &Tensor) -> Result<Self> {
        let result = self.uop().try_max(&other.uop()).context(UOpSnafu)?;
        Ok(Self::new(result))
    }

    /// Element-wise minimum: `min(self, other)`.
    ///
    /// Returns the element-wise minimum of two tensors.
    /// This is NOT a reduction - it returns a tensor of the same shape.
    ///
    /// # Shape Requirements
    /// Both tensors must be broadcastable to the same shape.
    ///
    /// # Examples
    /// ```ignore
    /// let a = Tensor::from_slice(&[1.0f32, 5.0, 3.0]);
    /// let b = Tensor::from_slice(&[2.0f32, 3.0, 4.0]);
    /// let result = a.minimum(&b)?;
    /// // result = [1.0, 3.0, 3.0]
    /// ```
    pub fn minimum(&self, other: &Tensor) -> Result<Self> {
        // Minimum is not a primitive, we implement it as: -max(-a, -b)
        // Or equivalently: where(a < b, a, b)
        let condition = self.try_lt(other)?;
        self.where_(&condition, other)
    }

    /// Clamp values to a range: `max(min_val, min(self, max_val))`.
    ///
    /// Constrains all elements to be within [min_val, max_val].
    ///
    /// # Examples
    /// ```ignore
    /// let x = Tensor::from_slice(&[-1.0f32, 0.0, 1.0, 2.0, 3.0]);
    /// let min = Tensor::from_slice(&[0.0f32, 0.0, 0.0, 0.0, 0.0]);
    /// let max = Tensor::from_slice(&[2.0f32, 2.0, 2.0, 2.0, 2.0]);
    ///
    /// // Clamp to [0, 2]
    /// let result = x.clamp().min(&min).max(&max).call()?;
    /// // result = [0.0, 0.0, 1.0, 2.0, 2.0]
    ///
    /// // Clamp only lower bound
    /// let result = x.clamp().min(&min).call()?;
    /// // result = [0.0, 0.0, 1.0, 2.0, 3.0]
    ///
    /// // Clamp only upper bound
    /// let result = x.clamp().max(&max).call()?;
    /// // result = [-1.0, 0.0, 1.0, 2.0, 2.0]
    /// ```
    #[builder]
    pub fn clamp(&self, min: Option<&Tensor>, max: Option<&Tensor>) -> Result<Self> {
        let mut result = self.clone();

        if let Some(min_val) = min {
            result = result.maximum(min_val)?;
        }

        if let Some(max_val) = max {
            result = result.minimum(max_val)?;
        }

        Ok(result)
    }

    /// Alias for `clamp` (matches NumPy/PyTorch naming).
    ///
    /// # Examples
    /// ```ignore
    /// let x = Tensor::from_slice(&[-1.0f32, 0.0, 1.0, 2.0, 3.0]);
    /// let min = Tensor::from_slice(&[0.0f32, 0.0, 0.0, 0.0, 0.0]);
    /// let max = Tensor::from_slice(&[2.0f32, 2.0, 2.0, 2.0, 2.0]);
    ///
    /// // Clip to [0, 2]
    /// let result = x.clip().min(&min).max(&max).call()?;
    /// ```
    #[builder]
    pub fn clip(&self, min: Option<&Tensor>, max: Option<&Tensor>) -> Result<Self> {
        self.clamp().maybe_min(min).maybe_max(max).call()
    }
}
