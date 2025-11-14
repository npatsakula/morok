//! Movement and reshape operations.
//!
//! These operations manipulate tensor shapes and layouts without changing
//! the underlying data (except for padding which may add values).

use std::rc::Rc;

use super::super::{Op, Result, UOp};

impl UOp {
    /// Reshape tensor to new shape.
    ///
    /// # Shape Parameters
    /// The `new_shape` parameter should contain Index-typed values representing
    /// the target dimensions. This is validated at runtime or during codegen.
    pub fn reshape(src: Rc<Self>, new_shape: Rc<Self>) -> Rc<Self> {
        let dtype = src.dtype();
        Self::new(Op::Reshape { src, new_shape }, dtype)
    }

    /// Permute dimensions according to axes.
    ///
    /// # Errors
    /// Returns error if permutation is invalid (not a valid permutation of 0..n).
    ///
    /// Note: Validation only occurs if source shape can be inferred.
    pub fn try_permute(src: Rc<Self>, axes: Vec<usize>) -> Result<Rc<Self>> {
        // TODO: Validate permutation using symbolic shape system
        // Shape validation will be done when shape inference is implemented
        // if let Some(src_shape) = src.shape() {
        //     Self::validate_permutation(&axes, src_shape.len())?;
        // }
        let dtype = src.dtype();
        Ok(Self::new(Op::Permute { src, axes }, dtype))
    }

    /// Expand (broadcast) dimensions.
    ///
    /// # Shape Parameters
    /// The `new_shape` parameter should contain Index-typed values representing
    /// the target dimensions. This is validated at runtime or during codegen.
    pub fn expand(src: Rc<Self>, new_shape: Rc<Self>) -> Rc<Self> {
        let dtype = src.dtype();
        Self::new(Op::Expand { src, new_shape }, dtype)
    }

    /// Pad tensor with begin and end padding for each dimension.
    ///
    /// # Shape Parameters
    /// The `begin_pads` and `end_pads` parameters should contain Index-typed
    /// values. This is validated at runtime or during codegen.
    pub fn pad(src: Rc<Self>, begin_pads: Rc<Self>, end_pads: Rc<Self>) -> Rc<Self> {
        let dtype = src.dtype();
        Self::new(Op::Pad { src, begin_pads, end_pads }, dtype)
    }

    /// Shrink (slice) tensor with begin and end indices for each dimension.
    ///
    /// # Shape Parameters
    /// The `begins` and `ends` parameters should contain Index-typed values
    /// representing the slice ranges. This is validated at runtime or during codegen.
    pub fn shrink(src: Rc<Self>, begins: Rc<Self>, ends: Rc<Self>) -> Rc<Self> {
        let dtype = src.dtype();
        Self::new(Op::Shrink { src, begins, ends }, dtype)
    }

    /// Flip (reverse) specified axes.
    ///
    /// # Errors
    /// Returns error if flip specification length doesn't match shape dimensions.
    ///
    /// Note: Validation only occurs if source shape can be inferred.
    pub fn try_flip(src: Rc<Self>, axes: Vec<bool>) -> Result<Rc<Self>> {
        // TODO: Validate flip axes using symbolic shape system
        // Shape validation will be done when shape inference is implemented
        // if let Some(src_shape) = src.shape() {
        //     Self::validate_flip_axes(&axes, src_shape.len())?;
        // }
        let dtype = src.dtype();
        Ok(Self::new(Op::Flip { src, axes }, dtype))
    }

    /// Multi-device split along axis.
    pub fn multi(src: Rc<Self>, axis: usize) -> Rc<Self> {
        let dtype = src.dtype();
        Self::new(Op::Multi { src, axis }, dtype)
    }
}

// =========================================================================
// Validated Movement Operation Constructors (Tinygrad-style)
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
    /// let src = UOp::const_(DType::Float32, ConstValue::Float(1.0)); // Scalar
    /// let new_shape = vec![SInt::from(1), SInt::from(1)];
    /// let shape_uop = shape::shape_to_uop(&new_shape);
    /// // This would work: scalar (product=1) -> [1,1] (product=1)
    /// // let reshaped = UOp::try_reshape_validated(src, &new_shape);
    /// ```
    pub fn try_reshape_validated(src: Rc<Self>, new_shape: &crate::shape::Shape) -> Result<Rc<Self>> {
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
        if let Some(src_shape) = src.shape() {
            let src_product = crate::sint_prod(src_shape);
            let dst_product = crate::sint_prod(new_shape);

            // If both are concrete, validate equality
            if let (Some(src_prod), Some(dst_prod)) = (src_product.as_const(), dst_product.as_const()) {
                ensure!(src_prod == dst_prod, ReshapeSizeMismatchSnafu { input_size: src_prod, output_size: dst_prod });
            }
            // Symbolic products can't be validated at compile time
        }

        let shape_uop = shape_to_uop(new_shape);
        let dtype = src.dtype();
        Ok(Self::new(Op::Reshape { src, new_shape: shape_uop }, dtype))
    }

    /// Expand (broadcast) with strict validation.
    ///
    /// Validates:
    /// - Number of dimensions matches
    /// - Each dimension either matches or src dimension is 1
    ///
    /// # Errors
    /// Returns error if validation fails.
    pub fn try_expand_validated(src: Rc<Self>, new_shape: &crate::shape::Shape) -> Result<Rc<Self>> {
        use crate::error::ExpandDimensionMismatchSnafu;
        use crate::error::ExpandInvalidDimensionSnafu;
        use crate::shape::shape_to_uop;
        use snafu::ensure;

        if let Some(src_shape) = src.shape() {
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
        let dtype = src.dtype();
        Ok(Self::new(Op::Expand { src, new_shape: shape_uop }, dtype))
    }

    /// Permute with strict validation.
    ///
    /// Validates:
    /// - Permutation is valid (contains each index 0..n exactly once)
    ///
    /// # Errors
    /// Returns error if permutation is invalid.
    pub fn try_permute_validated(src: Rc<Self>, axes: Vec<usize>) -> Result<Rc<Self>> {
        // Validate permutation if source shape is known
        if let Some(src_shape) = src.shape() {
            Self::validate_permutation(&axes, src_shape.len())?;
        }

        let dtype = src.dtype();
        Ok(Self::new(Op::Permute { src, axes }, dtype))
    }

    /// Pad with strict validation.
    ///
    /// Validates:
    /// - Padding values are non-negative
    /// - Number of padding pairs matches dimensions
    ///
    /// # Errors
    /// Returns error if validation fails.
    pub fn try_pad_validated(src: Rc<Self>, padding: &[(crate::SInt, crate::SInt)]) -> Result<Rc<Self>> {
        use crate::error::PadDimensionMismatchSnafu;
        use crate::shape::ranges_to_uops;
        use snafu::ensure;

        if let Some(src_shape) = src.shape() {
            // Check dimension count
            ensure!(
                padding.len() == src_shape.len(),
                PadDimensionMismatchSnafu { padding_dims: padding.len(), shape_dims: src_shape.len() }
            );
            // Note: No need to validate non-negative padding since SInt::Const uses usize
        }

        let (begin_pads, end_pads) = ranges_to_uops(padding);
        let dtype = src.dtype();
        Ok(Self::new(Op::Pad { src, begin_pads, end_pads }, dtype))
    }

    /// Shrink (slice) with strict validation.
    ///
    /// Validates:
    /// - begin <= end for each dimension
    /// - 0 <= begin, end <= dimension_size
    ///
    /// # Errors
    /// Returns error if validation fails.
    pub fn try_shrink_validated(src: Rc<Self>, ranges: &[(crate::SInt, crate::SInt)]) -> Result<Rc<Self>> {
        use crate::error::ShrinkBoundsViolationSnafu;
        use crate::shape::ranges_to_uops;
        use snafu::ensure;

        if let Some(src_shape) = src.shape() {
            // Validate each range
            for (dim_idx, ((begin, end), dim_size)) in ranges.iter().zip(src_shape.iter()).enumerate() {
                // If all are concrete, validate bounds
                if let (Some(b), Some(e), Some(s)) = (begin.as_const(), end.as_const(), dim_size.as_const()) {
                    ensure!(
                        b <= e && e <= s,
                        ShrinkBoundsViolationSnafu { dim: dim_idx, begin: b, end: e, shape_size: s }
                    );
                }
                // Symbolic values can't be validated at compile time
            }
        }

        let (begins, ends) = ranges_to_uops(ranges);
        let dtype = src.dtype();
        Ok(Self::new(Op::Shrink { src, begins, ends }, dtype))
    }

    /// Flip with strict validation.
    ///
    /// Validates:
    /// - Flip specification length matches shape dimensions
    ///
    /// # Errors
    /// Returns error if specification is invalid.
    pub fn try_flip_validated(src: Rc<Self>, axes: Vec<bool>) -> Result<Rc<Self>> {
        if let Some(src_shape) = src.shape() {
            Self::validate_flip_axes(&axes, src_shape.len())?;
        }

        let dtype = src.dtype();
        Ok(Self::new(Op::Flip { src, axes }, dtype))
    }
}
