//! Movement and reshape operations.
//!
//! These operations manipulate tensor shapes and layouts without changing
//! the underlying data (except for padding which may add values).

use std::rc::Rc;

use super::super::{Op, Result, UOp};

// These constructors are not yet used but will be needed for future optimization passes
#[allow(dead_code)]
impl UOp {
    /// Reshape tensor to new shape (low-level, UOp-based constructor).
    ///
    /// Takes a UOp for the shape parameter (used internally by compiler passes).
    /// For the public API with validation, use `try_reshape`.
    pub(crate) fn reshape(src: Rc<Self>, new_shape: Rc<Self>) -> Rc<Self> {
        let dtype = src.dtype();
        Self::new(Op::Reshape { src, new_shape }, dtype)
    }

    /// Permute dimensions (low-level, UOp-based constructor).
    ///
    /// For the public API with validation, use `try_permute`.
    pub(crate) fn permute(src: Rc<Self>, axes: Vec<usize>) -> Rc<Self> {
        let dtype = src.dtype();
        Self::new(Op::Permute { src, axes }, dtype)
    }

    /// Expand (broadcast) dimensions (low-level, UOp-based constructor).
    ///
    /// Takes a UOp for the shape parameter (used internally by compiler passes).
    /// For the public API with validation, use `try_expand`.
    pub(crate) fn expand(src: Rc<Self>, new_shape: Rc<Self>) -> Rc<Self> {
        let dtype = src.dtype();
        Self::new(Op::Expand { src, new_shape }, dtype)
    }

    /// Pad tensor (low-level, UOp-based constructor).
    ///
    /// Takes UOps for padding parameters (used internally by compiler passes).
    /// For the public API with validation, use `try_pad`.
    pub(crate) fn pad(src: Rc<Self>, begin_pads: Rc<Self>, end_pads: Rc<Self>) -> Rc<Self> {
        let dtype = src.dtype();
        Self::new(Op::Pad { src, begin_pads, end_pads }, dtype)
    }

    /// Shrink (slice) tensor (low-level, UOp-based constructor).
    ///
    /// Takes UOps for range parameters (used internally by compiler passes).
    /// For the public API with validation, use `try_shrink`.
    pub(crate) fn shrink(src: Rc<Self>, begins: Rc<Self>, ends: Rc<Self>) -> Rc<Self> {
        let dtype = src.dtype();
        Self::new(Op::Shrink { src, begins, ends }, dtype)
    }

    /// Flip (reverse) axes (low-level, UOp-based constructor).
    ///
    /// For the public API with validation, use `try_flip`.
    pub(crate) fn flip(src: Rc<Self>, axes: Vec<bool>) -> Rc<Self> {
        let dtype = src.dtype();
        Self::new(Op::Flip { src, axes }, dtype)
    }

    /// Multi-device split along axis.
    pub fn multi(src: Rc<Self>, axis: usize) -> Rc<Self> {
        let dtype = src.dtype();
        Self::new(Op::Multi { src, axis }, dtype)
    }
}

// =========================================================================
// Primary Movement Operation Constructors (with validation)
// =========================================================================

impl UOp {
    /// Reshape with strict validation (fail-fast).
    ///
    /// Validates:
    /// - No negative dimensions in new_shape
    /// - Product of input shape == product of output shape
    ///
    /// # Errors
    /// Returns error if validation fails.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use morok_ir::{UOp, ConstValue, SInt, shape};
    /// # use morok_dtype::DType;
    /// # use smallvec::SmallVec;
    /// let src = UOp::const_(DType::Float32, ConstValue::Float(1.0)); // Scalar
    /// let new_shape: SmallVec<[SInt; 4]> = SmallVec::from_vec(vec![SInt::from(1), SInt::from(1)]);
    /// // This would work: scalar (product=1) -> [1,1] (product=1)
    /// // let reshaped = src.try_reshape(&new_shape);
    /// ```
    pub fn try_reshape(self: &Rc<Self>, new_shape: &crate::shape::Shape) -> Result<Rc<Self>> {
        use crate::error::ReshapeNegativeDimensionSnafu;
        use crate::error::ReshapeSizeMismatchSnafu;
        use crate::shape::shape_to_uop;
        use snafu::ensure;

        // Check for negative dimensions
        for dim in new_shape {
            if let Some(val) = dim.as_const() {
                ensure!(val > 0, ReshapeNegativeDimensionSnafu { shape: vec![val as isize] });
            }
            // Symbolic dimensions are assumed positive (can't validate at compile time)
        }

        // Validate product equality if source shape is known
        if let Some(src_shape) = self.shape()? {
            let src_product = crate::sint_prod(src_shape);
            let dst_product = crate::sint_prod(new_shape);

            // If both are concrete, validate equality
            if let (Some(src_prod), Some(dst_prod)) = (src_product.as_const(), dst_product.as_const()) {
                ensure!(src_prod == dst_prod, ReshapeSizeMismatchSnafu { input_size: src_prod, output_size: dst_prod });
            }
            // Symbolic products can't be validated at compile time
        }

        let shape_uop = shape_to_uop(new_shape);
        let dtype = self.dtype();
        Ok(Self::new(Op::Reshape { src: self.clone(), new_shape: shape_uop }, dtype))
    }

    /// Expand (broadcast) with strict validation.
    ///
    /// Validates:
    /// - Number of dimensions matches
    /// - Each dimension either matches or src dimension is 1
    ///
    /// # Errors
    /// Returns error if validation fails.
    pub fn try_expand(self: &Rc<Self>, new_shape: &crate::shape::Shape) -> Result<Rc<Self>> {
        use crate::error::ExpandDimensionMismatchSnafu;
        use crate::error::ExpandInvalidDimensionSnafu;
        use crate::shape::shape_to_uop;
        use snafu::ensure;

        if let Some(src_shape) = self.shape()? {
            // Check dimension count
            ensure!(
                src_shape.len() == new_shape.len(),
                ExpandDimensionMismatchSnafu { input_dims: src_shape.len(), output_dims: new_shape.len() }
            );

            // Check each dimension can be expanded
            for (dim_idx, (src_dim, new_dim)) in src_shape.iter().zip(new_shape.iter()).enumerate() {
                // If both are concrete, validate expand rule
                if let (Some(s), Some(ns)) = (src_dim.as_const(), new_dim.as_const()) {
                    ensure!(
                        s == ns || (s == 1 && ns >= 1),
                        ExpandInvalidDimensionSnafu { dim: dim_idx, input: s, output: ns }
                    );
                }
                // Symbolic dimensions assumed compatible
            }
        }

        let shape_uop = shape_to_uop(new_shape);
        let dtype = self.dtype();
        Ok(Self::new(Op::Expand { src: self.clone(), new_shape: shape_uop }, dtype))
    }

    /// Permute with strict validation.
    ///
    /// Validates:
    /// - Permutation is valid (contains each index 0..n exactly once)
    ///
    /// # Errors
    /// Returns error if permutation is invalid.
    pub fn try_permute(self: &Rc<Self>, axes: Vec<usize>) -> Result<Rc<Self>> {
        // Validate permutation if source shape is known
        if let Some(src_shape) = self.shape()? {
            Self::validate_permutation(&axes, src_shape.len())?;
        }

        let dtype = self.dtype();
        Ok(Self::new(Op::Permute { src: self.clone(), axes }, dtype))
    }

    /// Pad with strict validation.
    ///
    /// Validates:
    /// - Padding values are concrete (not symbolic)
    /// - Number of padding pairs matches dimensions
    ///
    /// # Errors
    /// Returns error if validation fails or if padding contains symbolic values.
    pub fn try_pad(self: &Rc<Self>, padding: &[(crate::SInt, crate::SInt)]) -> Result<Rc<Self>> {
        use crate::error::{PadDimensionMismatchSnafu, SymbolicPaddingUnsupportedSnafu};
        use crate::shape::ranges_to_uops;
        use snafu::ensure;

        // Check for symbolic padding values
        for (begin, end) in padding {
            ensure!(begin.is_const(), SymbolicPaddingUnsupportedSnafu);
            ensure!(end.is_const(), SymbolicPaddingUnsupportedSnafu);
        }

        if let Some(src_shape) = self.shape()? {
            // Check dimension count
            ensure!(
                padding.len() == src_shape.len(),
                PadDimensionMismatchSnafu { padding_dims: padding.len(), shape_dims: src_shape.len() }
            );
        }

        let (begin_pads, end_pads) = ranges_to_uops(padding);
        let dtype = self.dtype();
        Ok(Self::new(Op::Pad { src: self.clone(), begin_pads, end_pads }, dtype))
    }

    /// Shrink (slice) with strict validation.
    ///
    /// Validates:
    /// - Range values are concrete (not symbolic)
    /// - begin <= end for each dimension
    /// - 0 <= begin, end <= dimension_size
    ///
    /// # Errors
    /// Returns error if validation fails or if ranges contain symbolic values.
    pub fn try_shrink(self: &Rc<Self>, ranges: &[(crate::SInt, crate::SInt)]) -> Result<Rc<Self>> {
        use crate::error::{ShrinkBoundsViolationSnafu, SymbolicShrinkingUnsupportedSnafu};
        use crate::shape::ranges_to_uops;
        use snafu::ensure;

        // Check for symbolic range values
        for (begin, end) in ranges {
            ensure!(begin.is_const(), SymbolicShrinkingUnsupportedSnafu);
            ensure!(end.is_const(), SymbolicShrinkingUnsupportedSnafu);
        }

        if let Some(src_shape) = self.shape()? {
            // Validate each range
            for (dim_idx, ((begin, end), dim_size)) in ranges.iter().zip(src_shape.iter()).enumerate() {
                // All are concrete now (checked above), validate bounds
                if let (Some(b), Some(e), Some(s)) = (begin.as_const(), end.as_const(), dim_size.as_const()) {
                    ensure!(
                        b <= e && e <= s,
                        ShrinkBoundsViolationSnafu { dim: dim_idx, begin: b, end: e, shape_size: s }
                    );
                }
            }
        }

        let (begins, ends) = ranges_to_uops(ranges);
        let dtype = self.dtype();
        Ok(Self::new(Op::Shrink { src: self.clone(), begins, ends }, dtype))
    }

    /// Flip with strict validation.
    ///
    /// Validates:
    /// - Flip specification length matches shape dimensions
    ///
    /// # Errors
    /// Returns error if specification is invalid.
    pub fn try_flip(self: &Rc<Self>, axes: Vec<bool>) -> Result<Rc<Self>> {
        if let Some(src_shape) = self.shape()? {
            Self::validate_flip_axes(&axes, src_shape.len())?;
        }

        let dtype = self.dtype();
        Ok(Self::new(Op::Flip { src: self.clone(), axes }, dtype))
    }
}
