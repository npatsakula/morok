//! Bitwise operations for integer tensors.
//!
//! This module provides bitwise operations: AND, OR, XOR, left shift, right shift.
//! All operations require integer or boolean dtypes.

use super::*;

impl Tensor {
    /// Bitwise AND operation.
    ///
    /// Performs element-wise bitwise AND between two tensors.
    /// Both tensors must have integer or boolean dtype.
    ///
    /// # Examples
    /// ```ignore
    /// let a = Tensor::from_slice(&[0b1010i32, 0b1100]);
    /// let b = Tensor::from_slice(&[0b1100i32, 0b0011]);
    /// let result = a.bitwise_and(&b)?;  // [0b1000, 0b0000]
    /// ```
    ///
    /// # Errors
    /// Returns error if:
    /// - Dtypes are not integer or boolean
    /// - Shapes don't match (broadcasting not implemented)
    #[track_caller]
    pub fn bitwise_and(&self, other: &Tensor) -> Result<Tensor> {
        self.uop.try_and_op(&other.uop).map(Self::new).context(UOpSnafu)
    }

    /// Bitwise OR operation.
    ///
    /// Performs element-wise bitwise OR between two tensors.
    /// Both tensors must have integer or boolean dtype.
    ///
    /// # Examples
    /// ```ignore
    /// let a = Tensor::from_slice(&[0b1010i32, 0b1100]);
    /// let b = Tensor::from_slice(&[0b1100i32, 0b0011]);
    /// let result = a.bitwise_or(&b)?;  // [0b1110, 0b1111]
    /// ```
    ///
    /// # Errors
    /// Returns error if:
    /// - Dtypes are not integer or boolean
    /// - Shapes don't match (broadcasting not implemented)
    #[track_caller]
    pub fn bitwise_or(&self, other: &Tensor) -> Result<Tensor> {
        self.uop.try_or_op(&other.uop).map(Self::new).context(UOpSnafu)
    }

    /// Bitwise XOR operation.
    ///
    /// Performs element-wise bitwise XOR between two tensors.
    /// Both tensors must have integer or boolean dtype.
    ///
    /// # Examples
    /// ```ignore
    /// let a = Tensor::from_slice(&[0b1010i32, 0b1100]);
    /// let b = Tensor::from_slice(&[0b1100i32, 0b0011]);
    /// let result = a.bitwise_xor(&b)?;  // [0b0110, 0b1111]
    /// ```
    ///
    /// # Errors
    /// Returns error if:
    /// - Dtypes are not integer or boolean
    /// - Shapes don't match (broadcasting not implemented)
    #[track_caller]
    pub fn bitwise_xor(&self, other: &Tensor) -> Result<Tensor> {
        self.uop.try_xor_op(&other.uop).map(Self::new).context(UOpSnafu)
    }

    /// Left shift operation.
    ///
    /// Shifts bits of the tensor to the left by the specified amount.
    /// The tensor must have integer or boolean dtype.
    ///
    /// # Examples
    /// ```ignore
    /// let t = Tensor::from_slice(&[1i32, 2, 3]);
    /// let shift = Tensor::from_slice(&[1i32, 2, 3]);
    /// let result = t.lshift(&shift)?;  // [2, 8, 24]
    /// ```
    ///
    /// # Errors
    /// Returns error if:
    /// - Dtype is not integer or boolean
    /// - Shapes don't match (broadcasting not implemented)
    #[track_caller]
    pub fn lshift(&self, other: &Tensor) -> Result<Tensor> {
        self.uop.try_shl_op(&other.uop).map(Self::new).context(UOpSnafu)
    }

    /// Right shift operation.
    ///
    /// Shifts bits of the tensor to the right by the specified amount.
    /// The tensor must have integer or boolean dtype.
    ///
    /// # Examples
    /// ```ignore
    /// let t = Tensor::from_slice(&[8i32, 16, 24]);
    /// let shift = Tensor::from_slice(&[1i32, 2, 3]);
    /// let result = t.rshift(&shift)?;  // [4, 4, 3]
    /// ```
    ///
    /// # Errors
    /// Returns error if:
    /// - Dtype is not integer or boolean
    /// - Shapes don't match (broadcasting not implemented)
    #[track_caller]
    pub fn rshift(&self, other: &Tensor) -> Result<Tensor> {
        self.uop.try_shr_op(&other.uop).map(Self::new).context(UOpSnafu)
    }
}

