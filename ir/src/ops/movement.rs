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
        // Validate permutation if we can infer the source shape
        if let Some(src_shape) = crate::shape::infer_shape(&src) {
            Self::validate_permutation(&axes, src_shape.len())?;
        }
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
        // Validate flip axes if we can infer the source shape
        if let Some(src_shape) = crate::shape::infer_shape(&src) {
            Self::validate_flip_axes(&axes, src_shape.len())?;
        }
        let dtype = src.dtype();
        Ok(Self::new(Op::Flip { src, axes }, dtype))
    }

    /// Multi-device split along axis.
    pub fn multi(src: Rc<Self>, axis: usize) -> Rc<Self> {
        let dtype = src.dtype();
        Self::new(Op::Multi { src, axis }, dtype)
    }
}
