//! Bitwise operations for integer tensors.
//!
//! This module provides bitwise operations: AND, OR, XOR, left shift, right shift.
//! All operations require integer or boolean dtypes.

use snafu::ResultExt;

use super::*;

impl Tensor {
    /// Bitwise AND operation.
    ///
    /// Performs element-wise bitwise AND between two tensors with broadcasting.
    /// Both tensors must have integer or boolean dtype.
    #[track_caller]
    pub fn bitwise_and<'o>(&self, other: impl Into<Operand<'o>>) -> Result<Tensor> {
        origin_call!("bitwise_and");
        let other = self.operand(other);
        let (lhs, rhs) = self.broadcast_for_binop(&other)?;
        lhs.uop().try_and_op(&rhs.uop()).map(Self::new).context(UOpSnafu).map_err(Into::into)
    }

    /// Bitwise OR operation.
    ///
    /// Performs element-wise bitwise OR between two tensors with broadcasting.
    /// Both tensors must have integer or boolean dtype.
    #[track_caller]
    pub fn bitwise_or<'o>(&self, other: impl Into<Operand<'o>>) -> Result<Tensor> {
        origin_call!("bitwise_or");
        let other = self.operand(other);
        let (lhs, rhs) = self.broadcast_for_binop(&other)?;
        lhs.uop().try_or_op(&rhs.uop()).map(Self::new).context(UOpSnafu).map_err(Into::into)
    }

    /// Bitwise XOR operation.
    ///
    /// Performs element-wise bitwise XOR between two tensors with broadcasting.
    /// Both tensors must have integer or boolean dtype.
    #[track_caller]
    pub fn bitwise_xor<'o>(&self, other: impl Into<Operand<'o>>) -> Result<Tensor> {
        origin_call!("bitwise_xor");
        let other = self.operand(other);
        let (lhs, rhs) = self.broadcast_for_binop(&other)?;
        lhs.uop().try_xor_op(&rhs.uop()).map(Self::new).context(UOpSnafu).map_err(Into::into)
    }

    /// Left shift operation.
    ///
    /// Shifts bits of the tensor to the left by the specified amount with broadcasting.
    /// The tensor must have integer or boolean dtype.
    #[track_caller]
    pub fn lshift<'o>(&self, other: impl Into<Operand<'o>>) -> Result<Tensor> {
        origin_call!("lshift");
        let other = self.operand(other);
        let (lhs, rhs) = self.broadcast_for_binop(&other)?;
        lhs.uop().try_shl_op(&rhs.uop()).map(Self::new).context(UOpSnafu).map_err(Into::into)
    }

    /// Right shift operation.
    ///
    /// Shifts bits of the tensor to the right by the specified amount with broadcasting.
    /// The tensor must have integer or boolean dtype.
    #[track_caller]
    pub fn rshift<'o>(&self, other: impl Into<Operand<'o>>) -> Result<Tensor> {
        origin_call!("rshift");
        let other = self.operand(other);
        let (lhs, rhs) = self.broadcast_for_binop(&other)?;
        lhs.uop().try_shr_op(&rhs.uop()).map(Self::new).context(UOpSnafu).map_err(Into::into)
    }
}
