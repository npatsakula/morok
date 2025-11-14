//! Operation implementations for UOps.
//!
//! This module contains all operation constructors organized by category:
//! - `arithmetic`: Basic arithmetic operations (add, sub, mul, div, rem)
//! - `bitwise`: Bitwise and shift operations
//! - `scalar`: Scalar convenience wrappers
//! - `movement`: Tensor reshaping and movement operations
//! - `reduction`: Reduction and aggregation operations
//! - `control`: Control flow operations
//! - `vector`: SIMD/vector operations
//! - `advanced`: Specialized operations (wmma, contract, etc.)

use std::rc::Rc;

use super::{DType, Op, Result, UOp};
use crate::error::Error;

// Submodules
pub mod advanced;
pub mod arithmetic;
pub mod bitwise;
pub mod control;
pub mod movement;
pub mod reduction;
pub mod scalar;
pub mod vector;

// =========================================================================
// Common Helper Functions
// =========================================================================
//
// These helpers are used across multiple operation categories and are kept
// centralized to avoid duplication.

impl UOp {
    /// Promote two dtypes to a common type and cast operands if needed.
    ///
    /// This function implements automatic type promotion following the same rules
    /// as tinygrad's `least_upper_dtype`. It finds the smallest common type that
    /// can represent both operands and casts them to that type.
    ///
    /// # Errors
    /// - Returns `VoidTypeInOp` if either operand has Void dtype
    /// - Returns `TypePromotionFailed` if no common type exists
    pub(super) fn promote_and_cast(lhs: Rc<Self>, rhs: Rc<Self>) -> Result<(Rc<Self>, Rc<Self>, DType)> {
        let lhs_dtype = lhs.dtype();
        let rhs_dtype = rhs.dtype();

        // Check for Void type
        if lhs_dtype == DType::Void || rhs_dtype == DType::Void {
            return Err(Error::VoidTypeInOp);
        }

        // Try to find common type
        let target_dtype = DType::least_upper_dtype(&[lhs_dtype.clone(), rhs_dtype.clone()])
            .ok_or(Error::TypePromotionFailed { lhs: lhs_dtype.clone(), rhs: rhs_dtype.clone() })?;

        // Cast if needed
        let lhs = if lhs_dtype != target_dtype { Self::cast(lhs, target_dtype.clone()) } else { lhs };
        let rhs = if rhs_dtype != target_dtype { Self::cast(rhs, target_dtype.clone()) } else { rhs };

        Ok((lhs, rhs, target_dtype))
    }

    /// Check that dtype is int or bool for bitwise operations.
    ///
    /// Bitwise operations (and, or, xor, not, shl, shr) require integer or boolean types.
    ///
    /// # Errors
    /// Returns `InvalidDTypeForOp` if dtype is not int or bool
    pub(super) fn check_bitwise_dtype(dtype: DType, operation: &'static str) -> Result<()> {
        let is_valid = dtype.is_bool() || dtype.is_signed() || dtype.is_unsigned();
        if !is_valid { Err(Error::InvalidDTypeForOp { operation, dtype }) } else { Ok(()) }
    }

    /// Check for division by zero when divisor is a constant.
    ///
    /// This validation only applies when the divisor is a compile-time constant.
    /// Runtime division by zero cannot be detected at IR construction time.
    ///
    /// # Errors
    /// Returns `DivisionByZero` if divisor is a constant zero
    pub(super) fn check_division_by_zero(divisor: &Rc<Self>) -> Result<()> {
        use super::ConstValue;
        use crate::error::DivisionByZeroSnafu;
        use snafu::ensure;

        // Only check if divisor is a constant
        if let Op::Const(const_hash) = divisor.op() {
            let is_zero = match const_hash.0 {
                ConstValue::Int(v) => v == 0,
                ConstValue::UInt(v) => v == 0,
                ConstValue::Float(v) => v == 0.0,
                ConstValue::Bool(_) => false,
            };
            ensure!(!is_zero, DivisionByZeroSnafu);
        }

        Ok(())
    }

    /// Validate permutation is valid (all indices 0..n, no duplicates).
    ///
    /// A valid permutation must:
    /// - Have exactly `expected_dims` elements
    /// - Contain each index from 0 to expected_dims-1 exactly once
    ///
    /// # Errors
    /// Returns `PermuteInvalidPermutation` if permutation is invalid
    pub(super) fn validate_permutation(axes: &[usize], expected_dims: usize) -> Result<()> {
        use crate::error::PermuteInvalidPermutationSnafu;
        use snafu::ensure;

        // Check length first
        ensure!(
            axes.len() == expected_dims,
            PermuteInvalidPermutationSnafu { permutation: axes.to_vec(), expected_dims }
        );

        // O(n) validation: check each index appears exactly once
        // Use bitset for small dims (≤64), boolean array for larger
        if expected_dims <= 64 {
            let mut seen = 0u64;
            for &axis in axes {
                ensure!(
                    axis < expected_dims,
                    PermuteInvalidPermutationSnafu { permutation: axes.to_vec(), expected_dims }
                );
                let bit = 1u64 << axis;
                ensure!(seen & bit == 0, PermuteInvalidPermutationSnafu { permutation: axes.to_vec(), expected_dims });
                seen |= bit;
            }
            // All bits must be set
            ensure!(
                seen == (1u64 << expected_dims) - 1,
                PermuteInvalidPermutationSnafu { permutation: axes.to_vec(), expected_dims }
            );
        } else {
            let mut seen = vec![false; expected_dims];
            for &axis in axes {
                ensure!(
                    axis < expected_dims,
                    PermuteInvalidPermutationSnafu { permutation: axes.to_vec(), expected_dims }
                );
                ensure!(!seen[axis], PermuteInvalidPermutationSnafu { permutation: axes.to_vec(), expected_dims });
                seen[axis] = true;
            }
        }

        Ok(())
    }

    /// Validate reduce axes are within bounds.
    ///
    /// Each reduction axis must be a valid dimension index (< shape_dims).
    ///
    /// # Errors
    /// Returns `ReduceAxisInvalid` if any axis is out of bounds
    pub(super) fn validate_reduce_axes(axes: &[usize], shape_dims: usize) -> Result<()> {
        use crate::error::ReduceAxisInvalidSnafu;
        use snafu::ensure;

        for &axis in axes {
            ensure!(axis < shape_dims, ReduceAxisInvalidSnafu { axis: axis as i32, shape_dims });
        }

        Ok(())
    }

    /// Validate flip axes specification.
    ///
    /// The flip specification must have exactly one boolean per dimension.
    ///
    /// # Errors
    /// Returns `FlipInvalidSpec` if specification length doesn't match expected dimensions
    pub(super) fn validate_flip_axes(axes: &[bool], expected_dims: usize) -> Result<()> {
        use crate::error::FlipInvalidSpecSnafu;
        use snafu::ensure;

        ensure!(axes.len() == expected_dims, FlipInvalidSpecSnafu { expected_dims, got_dims: axes.len() });

        Ok(())
    }
}
