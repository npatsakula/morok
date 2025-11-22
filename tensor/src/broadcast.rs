//! Broadcasting operations for tensors.
//!
//! Implements NumPy-style broadcasting rules:
//! - Shapes are aligned from the right (trailing dimensions)
//! - Missing dimensions are treated as 1
//! - For each dimension, sizes must either match or one must be 1
//!
//! This module provides the infrastructure for automatic broadcasting
//! in binary operations, matching Tinygrad's architecture.

use super::*;
use morok_ir::shape::{align_shapes_left, broadcast_shape};

impl Tensor {
    /// Broadcast two tensors to a common shape for binary operations.
    ///
    /// This method implements automatic broadcasting similar to NumPy/PyTorch.
    /// It aligns the shapes, computes the broadcast result shape, and broadcasts
    /// each tensor to that shape.
    ///
    /// # Broadcasting Rules
    ///
    /// - Shapes are aligned from the right (trailing dimensions)
    /// - Missing dimensions are padded with 1 on the left
    /// - For each dimension, sizes must either match or one must be 1
    /// - The result dimension is the maximum of the two
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Scalar + Vector: [] + [3] -> [3]
    /// let scalar = Tensor::from_slice([5.0f32]);
    /// let vector = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    /// let (a, b) = scalar.broadcast_for_binop(&vector)?;
    ///
    /// // Matrix + Row: [2, 3] + [1, 3] -> [2, 3]
    /// let matrix = Tensor::from_slice([1.0f32; 6]).try_reshape(&[2, 3])?;
    /// let row = Tensor::from_slice([1.0f32; 3]).try_reshape(&[1, 3])?;
    /// let (a, b) = matrix.broadcast_for_binop(&row)?;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if shapes are incompatible for broadcasting.
    pub(crate) fn broadcast_for_binop(&self, other: &Tensor) -> Result<(Tensor, Tensor)> {
        let self_shape = self.shape()?;
        let other_shape = other.shape()?;

        // Early return if shapes already match
        if self_shape == other_shape {
            return Ok((self.clone(), other.clone()));
        }

        // Handle scalar cases (empty shape means scalar in morok)
        // Actually, in morok scalars have shape [1], but let's handle both
        if self_shape.is_empty() && other_shape.is_empty() {
            return Ok((self.clone(), other.clone()));
        }

        // Align shapes (pad with 1s on left)
        let aligned = align_shapes_left(&[self_shape.clone(), other_shape.clone()]);

        // Compute broadcast result shape
        let result_shape = broadcast_shape(&aligned[0], &aligned[1]).context(UOpSnafu)?;

        // Broadcast each tensor to result shape
        let self_broadcast = self.broadcast_to(&result_shape)?;
        let other_broadcast = other.broadcast_to(&result_shape)?;

        Ok((self_broadcast, other_broadcast))
    }

    /// Broadcast tensor to a target shape.
    ///
    /// This is the low-level broadcast operation that reshapes (adds explicit 1 dimensions)
    /// and then expands (replicates data along size-1 dimensions).
    ///
    /// # Algorithm
    ///
    /// 1. If shape already matches, return self
    /// 2. Pad shape with 1s on the left to match rank
    /// 3. Reshape to add explicit 1 dimensions
    /// 4. Expand size-1 dimensions to target size
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // [3] -> [2, 3]
    /// let t = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    /// let target = vec![SInt::from(2), SInt::from(3)];
    /// let broadcasted = t.broadcast_to(&target)?;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Shape has more dimensions than target
    /// - Dimension sizes are incompatible (not 1 and not equal to target)
    pub(crate) fn broadcast_to(&self, target_shape: &morok_ir::shape::Shape) -> Result<Tensor> {
        let self_shape = self.shape()?;

        // Early return if already correct shape
        if &self_shape == target_shape {
            return Ok(self.clone());
        }

        // Cannot broadcast to fewer dimensions
        if self_shape.len() > target_shape.len() {
            return Err(Error::BroadcastFewerDimensions { from_dims: self_shape.len(), to_dims: target_shape.len() });
        }

        // Pad shape with 1s on left if needed
        let aligned_shape = if self_shape.len() < target_shape.len() {
            let padding = target_shape.len() - self_shape.len();
            let mut new_shape = morok_ir::shape::Shape::new();
            new_shape.extend(std::iter::repeat_n(morok_ir::SInt::from(1), padding));
            new_shape.extend(self_shape.iter().cloned());
            new_shape
        } else {
            self_shape.clone()
        };

        // Validate broadcast compatibility
        for (i, (aligned_dim, target_dim)) in aligned_shape.iter().zip(target_shape.iter()).enumerate() {
            if let (Some(aligned_size), Some(target_size)) = (aligned_dim.as_const(), target_dim.as_const())
                && aligned_size != 1
                && aligned_size != target_size
            {
                return Err(Error::BroadcastIncompatible { dim: i, from_size: aligned_size, to_size: target_size });
            }
            // For symbolic dimensions, conservatively assume they're compatible
        }

        // Reshape to add explicit 1 dimensions (if needed)
        let reshaped = if aligned_shape != self_shape {
            // Call IR layer directly to support symbolic dimensions
            self.uop.try_reshape(&aligned_shape).map(Self::new).context(UOpSnafu)?
        } else {
            self.clone()
        };

        // Check if expansion is actually needed
        if &aligned_shape == target_shape {
            return Ok(reshaped);
        }

        // Expand to target shape - call IR layer directly to support symbolic dimensions
        reshaped.uop.try_expand(target_shape).map(Self::new).context(UOpSnafu)
    }
}

