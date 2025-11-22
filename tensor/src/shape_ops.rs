//! Shape manipulation operations for Tensors.
//!
//! This module provides operations that change tensor shapes without copying data:
//! - Reshape: Change shape while preserving total elements
//! - Permute: Reorder dimensions
//! - Transpose: Swap two dimensions (convenience wrapper for permute)
//! - Expand: Broadcast dimensions from size 1
//! - Squeeze: Remove dimensions of size 1
//! - Unsqueeze: Add dimensions of size 1

use super::*;

impl Tensor {
    /// Reshape tensor to a new shape.
    ///
    /// The total number of elements must remain the same.
    /// Supports negative indices: -1 means "infer this dimension".
    ///
    /// # Examples
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// let reshaped = t.try_reshape(&[2, 3]).unwrap();  // [6] -> [2, 3]
    /// let inferred = t.try_reshape(&[-1, 2]).unwrap(); // [6] -> [3, 2]
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Shape contains negative values other than -1
    /// - Multiple -1 dimensions specified
    /// - Total elements don't match
    #[track_caller]
    pub fn try_reshape(&self, new_shape: &[isize]) -> Result<Tensor> {
        // Convert to Shape, handling -1 inference
        let shape = self.resolve_shape_with_inference(new_shape)?;

        self.uop.try_reshape(&shape).map(Self::new).context(UOpSnafu)
    }

    /// Permute (reorder) tensor dimensions.
    ///
    /// The axes parameter specifies the new order of dimensions.
    /// Each axis index 0..ndim must appear exactly once.
    ///
    /// # Examples
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// // Tensor with shape [2, 3, 4]
    /// // t.try_permute(&[2, 0, 1]) -> shape [4, 2, 3]
    /// // t.try_permute(&[1, 0, 2]) -> shape [3, 2, 4]
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Axes is not a valid permutation
    /// - Axis indices out of range
    #[track_caller]
    pub fn try_permute(&self, axes: &[isize]) -> Result<Tensor> {
        let shape = self.shape()?;
        let ndim = shape.len();

        // Normalize negative indices and validate
        let normalized_axes = self.normalize_axes(axes, ndim)?;

        self.uop.try_permute(normalized_axes).map(Self::new).context(UOpSnafu)
    }

    /// Transpose two dimensions.
    ///
    /// Convenience method for swapping two dimensions.
    /// Equivalent to permute with the two dimensions swapped.
    ///
    /// # Examples
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// // Tensor with shape [2, 3, 4]
    /// // t.try_transpose(0, 1) -> shape [3, 2, 4]
    /// // t.try_transpose(-1, 0) -> shape [4, 3, 2]  (negative indices supported)
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if axis indices are out of range.
    #[track_caller]
    pub fn try_transpose(&self, dim0: isize, dim1: isize) -> Result<Tensor> {
        let shape = self.shape()?;
        let ndim = shape.len();

        // Normalize negative indices
        let d0 = Self::normalize_axis(dim0, ndim)?;
        let d1 = Self::normalize_axis(dim1, ndim)?;

        // Build permutation with swapped dimensions
        let mut axes: Vec<usize> = (0..ndim).collect();
        axes.swap(d0, d1);

        self.uop.try_permute(axes).map(Self::new).context(UOpSnafu)
    }

    /// Expand (broadcast) dimensions.
    ///
    /// Dimensions of size 1 can be expanded to larger sizes.
    /// Use -1 to keep the current dimension size.
    ///
    /// # Examples
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// // Tensor with shape [1, 3, 1]
    /// // t.try_expand(&[4, -1, 5]) -> shape [4, 3, 5]
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Number of dimensions doesn't match
    /// - Trying to expand non-1 dimension to different size
    #[track_caller]
    pub fn try_expand(&self, new_shape: &[isize]) -> Result<Tensor> {
        let shape = self.resolve_expand_shape(new_shape)?;

        self.uop.try_expand(&shape).map(Self::new).context(UOpSnafu)
    }

    /// Squeeze dimensions of size 1.
    ///
    /// If dim is None, removes all dimensions of size 1.
    /// If dim is Some(axis), removes only that dimension if it's size 1.
    ///
    /// # Examples
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// // Tensor with shape [1, 3, 1, 4]
    /// // t.try_squeeze(None) -> shape [3, 4]
    /// // t.try_squeeze(Some(0)) -> shape [3, 1, 4]
    /// // t.try_squeeze(Some(2)) -> shape [1, 3, 4]
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Specified dimension is not size 1
    /// - Axis index out of range
    #[track_caller]
    pub fn try_squeeze(&self, dim: Option<isize>) -> Result<Tensor> {
        let shape = self.shape()?;

        let new_shape = match dim {
            None => {
                // Remove all dimensions of size 1
                shape
                    .iter()
                    .filter_map(|s| s.as_const().and_then(|v| if v != 1 { Some(SInt::Const(v)) } else { None }))
                    .collect()
            }
            Some(axis) => {
                let ndim = shape.len();
                let normalized_axis = Self::normalize_axis(axis, ndim)?;

                // Check if dimension is size 1
                let dim_size = shape[normalized_axis]
                    .as_const()
                    .ok_or_else(|| Error::SymbolicShapeUnsupported { operation: "squeeze".to_string() })?;

                snafu::ensure!(dim_size == 1, SqueezeDimensionNotOneSnafu { dim: normalized_axis, size: dim_size });

                // Remove the specified dimension
                shape
                    .iter()
                    .enumerate()
                    .filter_map(|(i, s)| if i != normalized_axis { Some(s.clone()) } else { None })
                    .collect()
            }
        };

        self.uop.try_reshape(&new_shape).map(Self::new).context(UOpSnafu)
    }

    /// Add a dimension of size 1.
    ///
    /// Inserts a new dimension at the specified position.
    /// Supports negative indices: -1 means after the last dimension.
    ///
    /// # Examples
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// // Tensor with shape [3, 4]
    /// // t.try_unsqueeze(0) -> shape [1, 3, 4]
    /// // t.try_unsqueeze(1) -> shape [3, 1, 4]
    /// // t.try_unsqueeze(-1) -> shape [3, 4, 1]
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if axis index is out of range.
    #[track_caller]
    pub fn try_unsqueeze(&self, dim: isize) -> Result<Tensor> {
        let shape = self.shape()?;
        let ndim = shape.len();

        // For unsqueeze, valid range is [0, ndim] (can insert at end)
        // Normalize negative indices: -1 means ndim (after last), -2 means ndim-1, etc.
        let normalized_dim = if dim < 0 {
            let positive = (ndim as isize + 1 + dim) as usize;
            snafu::ensure!(dim >= -(ndim as isize + 1), AxisOutOfRangeSnafu { axis: dim, ndim });
            positive
        } else {
            let pos = dim as usize;
            snafu::ensure!(pos <= ndim, AxisOutOfRangeSnafu { axis: dim, ndim });
            pos
        };

        // Insert dimension of size 1
        let mut new_shape = shape.clone();
        new_shape.insert(normalized_dim, SInt::Const(1));

        self.uop.try_reshape(&new_shape).map(Self::new).context(UOpSnafu)
    }

    /// Flatten tensor to 1D.
    ///
    /// Reshapes tensor to have a single dimension containing all elements.
    /// Equivalent to `try_reshape(&[-1])`.
    ///
    /// # Examples
    /// ```ignore
    /// let t = Tensor::from_slice(&[[1, 2], [3, 4]]);  // Shape [2, 2]
    /// let flattened = t.flatten()?;  // Shape [4]
    /// ```
    #[track_caller]
    pub fn flatten(&self) -> Result<Tensor> {
        self.try_reshape(&[-1])
    }

    // =========================================================================
    // Helper Methods
    // =========================================================================

    /// Get the shape of this tensor.
    pub(crate) fn shape(&self) -> Result<Shape> {
        self.uop.shape().context(UOpSnafu)?.cloned().ok_or(Error::ShapeUnknown)
    }

    /// Get the number of dimensions (rank) of this tensor.
    ///
    /// This is equivalent to `len(tensor.shape)` in NumPy/PyTorch.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let scalar = Tensor::from_slice([5.0f32]);  // Shape [1]
    /// assert_eq!(scalar.ndim()?, 1);
    ///
    /// let matrix = Tensor::from_slice([1.0f32; 6]).try_reshape(&[2, 3])?;
    /// assert_eq!(matrix.ndim()?, 2);
    /// ```
    pub(crate) fn ndim(&self) -> Result<usize> {
        Ok(self.shape()?.len())
    }

    /// Get the size of a specific dimension.
    ///
    /// Supports negative indexing (e.g., -1 for last dimension).
    /// Returns a SInt which can be either concrete (Const) or symbolic.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::from_slice([1.0f32; 6]).try_reshape(&[2, 3])?;
    /// assert_eq!(t.dim(0)?.as_const(), Some(2));   // First dimension
    /// assert_eq!(t.dim(1)?.as_const(), Some(3));   // Second dimension
    /// assert_eq!(t.dim(-1)?.as_const(), Some(3));  // Last dimension (negative indexing)
    /// assert_eq!(t.dim(-2)?.as_const(), Some(2));  // Second-to-last dimension
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if axis is out of range.
    pub(crate) fn dim(&self, axis: isize) -> Result<morok_ir::SInt> {
        let shape = self.shape()?;
        let idx = Self::normalize_axis(axis, shape.len())?;
        Ok(shape[idx].clone())
    }

    /// Get the concrete size of a specific dimension.
    ///
    /// Like `dim()`, but returns an error if the dimension is symbolic.
    /// Use this when you need a concrete usize value (e.g., for buffer allocation).
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Axis is out of range
    /// - Dimension is symbolic (non-concrete)
    pub(crate) fn dim_concrete(&self, axis: isize) -> Result<usize> {
        self.dim(axis)?
            .as_const()
            .ok_or_else(|| Error::SymbolicShapeUnsupported { operation: format!("dim_concrete({})", axis) })
    }

    /// Normalize a single axis index (handle negative indices).
    pub(crate) fn normalize_axis(axis: isize, ndim: usize) -> Result<usize> {
        if axis < 0 {
            let positive = (ndim as isize + axis) as usize;
            snafu::ensure!(axis >= -(ndim as isize), AxisOutOfRangeSnafu { axis, ndim });
            Ok(positive)
        } else {
            let pos = axis as usize;
            snafu::ensure!(pos < ndim, AxisOutOfRangeSnafu { axis, ndim });
            Ok(pos)
        }
    }

    /// Normalize axes list and validate it's a valid permutation.
    fn normalize_axes(&self, axes: &[isize], ndim: usize) -> Result<Vec<usize>> {
        snafu::ensure!(axes.len() == ndim, PermutationLengthMismatchSnafu { expected: ndim, got: axes.len() });

        let mut normalized = Vec::with_capacity(ndim);
        for &axis in axes {
            normalized.push(Self::normalize_axis(axis, ndim)?);
        }

        // Validate it's a permutation (each index appears exactly once)
        let mut seen = vec![false; ndim];
        for &idx in &normalized {
            snafu::ensure!(!seen[idx], InvalidPermutationSnafu { axes: axes.to_vec() });
            seen[idx] = true;
        }

        Ok(normalized)
    }

    /// Resolve reshape shape with -1 inference.
    fn resolve_shape_with_inference(&self, shape_spec: &[isize]) -> Result<Shape> {
        let current_shape = self.shape()?;

        // Find -1 position and validate
        let minus_one_pos = shape_spec.iter().position(|&s| s == -1);
        let has_multiple_minus_one = shape_spec.iter().filter(|&&s| s == -1).count() > 1;

        snafu::ensure!(!has_multiple_minus_one, MultipleInferDimensionsSnafu);

        // Validate no other negative values
        for &dim in shape_spec {
            snafu::ensure!(dim > 0 || dim == -1, NegativeDimensionSnafu { dim });
        }

        // Calculate new shape
        let new_shape: Shape = if let Some(_infer_pos) = minus_one_pos {
            // Inference requires concrete shape to compute total elements
            let total_elements = current_shape
                .iter()
                .try_fold(1usize, |acc, dim| dim.as_const().map(|v| acc * v))
                .ok_or_else(|| Error::SymbolicShapeUnsupported { operation: "reshape with -1 inference".to_string() })?;

            // Calculate product of known dimensions
            let known_product: usize = shape_spec.iter().filter(|&&s| s > 0).map(|&s| s as usize).product();

            snafu::ensure!(
                total_elements % known_product == 0,
                ReshapeSizeMismatchSnafu { operation: "reshape with inference".to_string() }
            );

            let inferred_dim = total_elements / known_product;

            shape_spec
                .iter()
                .map(|&s| if s == -1 { SInt::Const(inferred_dim) } else { SInt::Const(s as usize) })
                .collect()
        } else {
            // No inference, direct conversion - allow symbolic shapes to pass through
            shape_spec.iter().map(|&s| SInt::Const(s as usize)).collect()
        };

        Ok(new_shape)
    }

    /// Resolve expand shape with -1 meaning "keep current dimension".
    fn resolve_expand_shape(&self, shape_spec: &[isize]) -> Result<Shape> {
        let current_shape = self.shape()?;

        snafu::ensure!(
            shape_spec.len() == current_shape.len(),
            ExpandDimensionMismatchSnafu { current_dims: current_shape.len(), target_dims: shape_spec.len() }
        );

        let new_shape: Shape = shape_spec
            .iter()
            .zip(current_shape.iter())
            .map(|(&spec, current)| {
                if spec == -1 {
                    Ok(current.clone())
                } else {
                    snafu::ensure!(spec > 0, NegativeDimensionSnafu { dim: spec });
                    Ok(SInt::Const(spec as usize))
                }
            })
            .collect::<Result<_>>()?;

        Ok(new_shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use morok_ir::Op;

    // Helper to get concrete shape as Vec<usize>
    fn get_shape(tensor: &Tensor) -> Vec<usize> {
        tensor.uop.shape().unwrap().unwrap().iter().map(|s| s.as_const().unwrap()).collect()
    }

    // =========================================================================
    // Reshape Tests
    // =========================================================================

    #[test]
    fn test_reshape_basic() {
        let t = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let reshaped = t.try_reshape(&[2, 3]).unwrap();

        assert_eq!(get_shape(&reshaped), vec![2, 3]);
        if let Op::Reshape { .. } = reshaped.uop.op() {
            // Correct operation type
        } else {
            panic!("Expected Reshape operation");
        }
    }

    #[test]
    fn test_reshape_with_inference() {
        let t = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // -1 should infer dimension: 6 elements / 2 = 3
        let reshaped = t.try_reshape(&[-1, 2]).unwrap();
        assert_eq!(get_shape(&reshaped), vec![3, 2]);

        // -1 at different position
        let reshaped2 = t.try_reshape(&[3, -1]).unwrap();
        assert_eq!(get_shape(&reshaped2), vec![3, 2]);
    }

    #[test]
    fn test_reshape_flatten() {
        let t = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let reshaped = t.try_reshape(&[2, 3]).unwrap();
        let flattened = reshaped.try_reshape(&[6]).unwrap();

        assert_eq!(get_shape(&flattened), vec![6]);
    }

    #[test]
    fn test_reshape_identity() {
        let t = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let reshaped = t.try_reshape(&[3]).unwrap();

        assert_eq!(get_shape(&reshaped), vec![3]);
    }

    #[test]
    fn test_reshape_error_size_mismatch() {
        let t = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // 6 elements cannot be reshaped to [2, 4] = 8 elements
        let result = t.try_reshape(&[2, 4]);
        assert!(result.is_err());
    }

    #[test]
    fn test_reshape_error_multiple_inference() {
        let t = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // Multiple -1 dimensions not allowed
        let result = t.try_reshape(&[-1, -1]);
        assert!(result.is_err());
    }

    #[test]
    fn test_reshape_error_invalid_negative() {
        let t = Tensor::from_slice([1.0f32, 2.0, 3.0]);

        // Negative dimension other than -1 not allowed
        let result = t.try_reshape(&[-2, 3]);
        assert!(result.is_err());
    }

    // =========================================================================
    // Permute Tests
    // =========================================================================

    #[test]
    fn test_permute_basic() {
        let t = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let reshaped = t.try_reshape(&[2, 3]).unwrap();

        // Swap dimensions [2, 3] -> [3, 2]
        let permuted = reshaped.try_permute(&[1, 0]).unwrap();
        assert_eq!(get_shape(&permuted), vec![3, 2]);
    }

    #[test]
    fn test_permute_3d() {
        let t = Tensor::from_slice([1.0f32; 24]);
        let reshaped = t.try_reshape(&[2, 3, 4]).unwrap();

        // Permute [2, 3, 4] -> [4, 2, 3]
        let permuted = reshaped.try_permute(&[2, 0, 1]).unwrap();
        assert_eq!(get_shape(&permuted), vec![4, 2, 3]);
    }

    #[test]
    fn test_permute_identity() {
        let t = Tensor::from_slice([1.0f32; 6]);
        let reshaped = t.try_reshape(&[2, 3]).unwrap();

        // Identity permutation
        let permuted = reshaped.try_permute(&[0, 1]).unwrap();
        assert_eq!(get_shape(&permuted), vec![2, 3]);
    }

    #[test]
    fn test_permute_negative_indices() {
        let t = Tensor::from_slice([1.0f32; 6]);
        let reshaped = t.try_reshape(&[2, 3]).unwrap();

        // Negative indices: -1 = last axis, -2 = second to last
        let permuted = reshaped.try_permute(&[-1, -2]).unwrap();
        assert_eq!(get_shape(&permuted), vec![3, 2]);
    }

    #[test]
    fn test_permute_error_invalid_permutation() {
        let t = Tensor::from_slice([1.0f32; 6]);
        let reshaped = t.try_reshape(&[2, 3]).unwrap();

        // Duplicate axis
        let result = reshaped.try_permute(&[0, 0]);
        assert!(result.is_err());

        // Missing axis
        let result2 = reshaped.try_permute(&[0, 2]);
        assert!(result2.is_err());
    }

    #[test]
    fn test_permute_error_wrong_length() {
        let t = Tensor::from_slice([1.0f32; 6]);
        let reshaped = t.try_reshape(&[2, 3]).unwrap();

        // Wrong number of axes
        let result = reshaped.try_permute(&[0, 1, 2]);
        assert!(result.is_err());
    }

    // =========================================================================
    // Transpose Tests
    // =========================================================================

    #[test]
    fn test_transpose_basic() {
        let t = Tensor::from_slice([1.0f32; 6]);
        let reshaped = t.try_reshape(&[2, 3]).unwrap();

        let transposed = reshaped.try_transpose(0, 1).unwrap();
        assert_eq!(get_shape(&transposed), vec![3, 2]);
    }

    #[test]
    fn test_transpose_3d() {
        let t = Tensor::from_slice([1.0f32; 24]);
        let reshaped = t.try_reshape(&[2, 3, 4]).unwrap();

        // Swap first and last dimensions
        let transposed = reshaped.try_transpose(0, 2).unwrap();
        assert_eq!(get_shape(&transposed), vec![4, 3, 2]);
    }

    #[test]
    fn test_transpose_negative_indices() {
        let t = Tensor::from_slice([1.0f32; 6]);
        let reshaped = t.try_reshape(&[2, 3]).unwrap();

        // -1 = last axis (1), -2 = second to last (0)
        let transposed = reshaped.try_transpose(-1, -2).unwrap();
        assert_eq!(get_shape(&transposed), vec![3, 2]);
    }

    #[test]
    fn test_transpose_same_dimension() {
        let t = Tensor::from_slice([1.0f32; 6]);
        let reshaped = t.try_reshape(&[2, 3]).unwrap();

        // Transpose dimension with itself = identity
        let transposed = reshaped.try_transpose(0, 0).unwrap();
        assert_eq!(get_shape(&transposed), vec![2, 3]);
    }

    // =========================================================================
    // Expand Tests
    // =========================================================================

    #[test]
    fn test_expand_basic() {
        let t = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let reshaped = t.try_reshape(&[1, 3]).unwrap();

        // Expand first dimension from 1 to 4
        let expanded = reshaped.try_expand(&[4, -1]).unwrap();
        assert_eq!(get_shape(&expanded), vec![4, 3]);
    }

    #[test]
    fn test_expand_keep_dimension() {
        let t = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let reshaped = t.try_reshape(&[1, 3]).unwrap();

        // -1 keeps current dimension
        let expanded = reshaped.try_expand(&[-1, -1]).unwrap();
        assert_eq!(get_shape(&expanded), vec![1, 3]);
    }

    #[test]
    fn test_expand_multiple_dims() {
        let t = Tensor::from_slice([1.0f32]);
        let reshaped = t.try_reshape(&[1, 1, 1]).unwrap();

        // Expand all dimensions
        let expanded = reshaped.try_expand(&[4, 5, 6]).unwrap();
        assert_eq!(get_shape(&expanded), vec![4, 5, 6]);
    }

    #[test]
    fn test_expand_error_non_one_dimension() {
        let t = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let reshaped = t.try_reshape(&[1, 3]).unwrap();

        // Cannot expand dimension of size 3 to 5
        let result = reshaped.try_expand(&[1, 5]);
        assert!(result.is_err());
    }

    #[test]
    fn test_expand_error_dimension_mismatch() {
        let t = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let reshaped = t.try_reshape(&[1, 3]).unwrap();

        // Wrong number of dimensions
        let result = reshaped.try_expand(&[4, 5, 6]);
        assert!(result.is_err());
    }

    // =========================================================================
    // Squeeze Tests
    // =========================================================================

    #[test]
    fn test_squeeze_all() {
        let t = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let reshaped = t.try_reshape(&[1, 3, 1]).unwrap();

        // Remove all dimensions of size 1
        let squeezed = reshaped.try_squeeze(None).unwrap();
        assert_eq!(get_shape(&squeezed), vec![3]);
    }

    #[test]
    fn test_squeeze_specific_dim() {
        let t = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let reshaped = t.try_reshape(&[1, 3, 1]).unwrap();

        // Remove only first dimension
        let squeezed = reshaped.try_squeeze(Some(0)).unwrap();
        assert_eq!(get_shape(&squeezed), vec![3, 1]);

        // Remove last dimension
        let squeezed2 = reshaped.try_squeeze(Some(-1)).unwrap();
        assert_eq!(get_shape(&squeezed2), vec![1, 3]);
    }

    #[test]
    fn test_squeeze_no_effect() {
        let t = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let reshaped = t.try_reshape(&[3]).unwrap();

        // No dimensions of size 1 to squeeze
        let squeezed = reshaped.try_squeeze(None).unwrap();
        assert_eq!(get_shape(&squeezed), vec![3]);
    }

    #[test]
    fn test_squeeze_error_not_size_one() {
        let t = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let reshaped = t.try_reshape(&[1, 3]).unwrap();

        // Cannot squeeze dimension of size 3
        let result = reshaped.try_squeeze(Some(1));
        assert!(result.is_err());
    }

    // =========================================================================
    // Unsqueeze Tests
    // =========================================================================

    #[test]
    fn test_unsqueeze_at_start() {
        let t = Tensor::from_slice([1.0f32, 2.0, 3.0]);

        let unsqueezed = t.try_unsqueeze(0).unwrap();
        assert_eq!(get_shape(&unsqueezed), vec![1, 3]);
    }

    #[test]
    fn test_unsqueeze_at_end() {
        let t = Tensor::from_slice([1.0f32, 2.0, 3.0]);

        let unsqueezed = t.try_unsqueeze(1).unwrap();
        assert_eq!(get_shape(&unsqueezed), vec![3, 1]);
    }

    #[test]
    fn test_unsqueeze_negative_index() {
        let t = Tensor::from_slice([1.0f32, 2.0, 3.0]);

        // -1 means after last dimension
        let unsqueezed = t.try_unsqueeze(-1).unwrap();
        assert_eq!(get_shape(&unsqueezed), vec![3, 1]);
    }

    #[test]
    fn test_unsqueeze_middle() {
        let t = Tensor::from_slice([1.0f32; 6]);
        let reshaped = t.try_reshape(&[2, 3]).unwrap();

        let unsqueezed = reshaped.try_unsqueeze(1).unwrap();
        assert_eq!(get_shape(&unsqueezed), vec![2, 1, 3]);
    }

    #[test]
    fn test_unsqueeze_multiple() {
        let t = Tensor::from_slice([1.0f32, 2.0, 3.0]);

        let unsqueezed1 = t.try_unsqueeze(0).unwrap();
        let unsqueezed2 = unsqueezed1.try_unsqueeze(0).unwrap();
        assert_eq!(get_shape(&unsqueezed2), vec![1, 1, 3]);
    }

    // =========================================================================
    // Combined Operations Tests
    // =========================================================================

    #[test]
    fn test_reshape_then_transpose() {
        let t = Tensor::from_slice([1.0f32; 6]);
        let reshaped = t.try_reshape(&[2, 3]).unwrap();
        let transposed = reshaped.try_transpose(0, 1).unwrap();

        assert_eq!(get_shape(&transposed), vec![3, 2]);
    }

    #[test]
    fn test_unsqueeze_then_expand() {
        let t = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let unsqueezed = t.try_unsqueeze(0).unwrap();
        let expanded = unsqueezed.try_expand(&[4, -1]).unwrap();

        assert_eq!(get_shape(&expanded), vec![4, 3]);
    }

    #[test]
    fn test_expand_then_squeeze() {
        let t = Tensor::from_slice([1.0f32]);
        let reshaped = t.try_reshape(&[1, 1]).unwrap();
        let expanded = reshaped.try_expand(&[4, -1]).unwrap();
        let squeezed = expanded.try_squeeze(Some(1)).unwrap();

        assert_eq!(get_shape(&squeezed), vec![4]);
    }

    #[test]
    fn test_lazy_evaluation() {
        let t = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let reshaped = t.try_reshape(&[2, 3]).unwrap();
        let permuted = reshaped.try_permute(&[1, 0]).unwrap();
        let unsqueezed = reshaped.try_unsqueeze(0).unwrap();

        // Movement operations share the same underlying buffer via .base()
        // They all point to the same buffer (t's buffer)
        assert!(reshaped.buffer().is_some());
        assert!(permuted.buffer().is_some());
        assert!(unsqueezed.buffer().is_some());

        // All should share the same buffer ID (same base)
        assert_eq!(reshaped.uop.base().id, t.uop.base().id);
        assert_eq!(permuted.uop.base().id, t.uop.base().id);
        assert_eq!(unsqueezed.uop.base().id, t.uop.base().id);
    }

    #[test]
    fn test_dtype_preservation() {
        let t_f32 = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let t_i32 = Tensor::from_slice([1i32, 2, 3]);

        let reshaped_f32 = t_f32.try_reshape(&[3, 1]).unwrap();
        let reshaped_i32 = t_i32.try_reshape(&[3, 1]).unwrap();

        assert_eq!(reshaped_f32.uop.dtype(), morok_dtype::DType::Float32);
        assert_eq!(reshaped_i32.uop.dtype(), morok_dtype::DType::Int32);
    }

    // =========================================================================
    // Symbolic Shape Tests
    // =========================================================================

    #[test]
    fn test_symbolic_shape_support() {
        use morok_ir::{Op, DType, ConstValue};

        // Create a tensor with a symbolic dimension using DefineVar
        let batch_var = UOp::new(Op::DefineVar { name: "batch".to_string(), min_val: 1, max_val: 128 }, DType::Index);
        let batch_dim = morok_ir::SInt::Symbolic(batch_var);

        // Create shape: [batch, 3, 4] where batch is symbolic
        let symbolic_shape: morok_ir::shape::Shape =
            vec![batch_dim.clone(), morok_ir::SInt::from(3), morok_ir::SInt::from(4)].into();

        // Create a tensor with this symbolic shape using a const value
        let const_val = UOp::const_(DType::Float32, ConstValue::Float(1.0));
        let tensor_with_symbolic_shape = const_val.try_reshape(&symbolic_shape).unwrap();
        let tensor = Tensor::new(tensor_with_symbolic_shape);

        // Test 1: dim() returns SInt (can be symbolic or concrete)
        let dim0 = tensor.dim(0).unwrap();
        let dim1 = tensor.dim(1).unwrap();
        let dim2 = tensor.dim(2).unwrap();

        // First dimension is symbolic
        assert!(dim0.as_const().is_none()); // Symbolic, no concrete value
        assert_eq!(dim1.as_const(), Some(3)); // Concrete
        assert_eq!(dim2.as_const(), Some(4)); // Concrete

        // Test 2: ndim() works with symbolic shapes
        assert_eq!(tensor.ndim().unwrap(), 3);

        // Test 3: Reshape preserving symbolic dimension
        let new_shape: morok_ir::shape::Shape =
            vec![batch_dim.clone(), morok_ir::SInt::from(12)].into();
        let reshaped = tensor.uop.try_reshape(&new_shape).map(Tensor::new).unwrap();
        assert_eq!(reshaped.ndim().unwrap(), 2);

        // Test 4: Permute works with symbolic shapes
        let permuted = tensor.try_permute(&[1, 0, 2]).unwrap();
        let perm_shape = permuted.shape().unwrap();
        assert_eq!(perm_shape[0].as_const(), Some(3)); // Was dim 1
        assert!(perm_shape[1].as_const().is_none()); // Was dim 0 (symbolic)
        assert_eq!(perm_shape[2].as_const(), Some(4)); // Was dim 2
    }

    #[test]
    fn test_symbolic_shape_broadcast() {
        use morok_ir::{Op, DType, ConstValue};

        // Create symbolic batch dimension
        let batch_var = UOp::new(Op::DefineVar { name: "N".to_string(), min_val: 1, max_val: 1024 }, DType::Index);
        let batch_dim = morok_ir::SInt::Symbolic(batch_var);

        // Create tensor with shape [N, 4]
        let symbolic_shape: morok_ir::shape::Shape = vec![batch_dim.clone(), morok_ir::SInt::from(4)].into();

        let const_val = UOp::const_(DType::Float32, ConstValue::Float(1.0));
        let tensor_symbolic = const_val.try_reshape(&symbolic_shape).unwrap();
        let tensor = Tensor::new(tensor_symbolic);

        // Create a concrete tensor to broadcast against: [1, 4]
        let concrete = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]).try_reshape(&[1, 4]).unwrap();

        // Broadcasting should work with symbolic shapes
        let (broadcasted_symbolic, broadcasted_concrete) = tensor.broadcast_for_binop(&concrete).unwrap();

        // Both should have shape [N, 4]
        let result_shape = broadcasted_symbolic.shape().unwrap();
        assert_eq!(result_shape.len(), 2);
        assert!(result_shape[0].as_const().is_none()); // Symbolic N
        assert_eq!(result_shape[1].as_const(), Some(4)); // Concrete 4
    }

    #[test]
    fn test_symbolic_shape_binary_ops() {
        use morok_ir::{Op, DType, ConstValue};

        // Create symbolic dimensions
        let dim_var = UOp::new(Op::DefineVar { name: "D".to_string(), min_val: 1, max_val: 512 }, DType::Index);
        let dim_sym = morok_ir::SInt::Symbolic(dim_var);

        // Create two tensors with symbolic shape [D]
        let shape: morok_ir::shape::Shape = vec![dim_sym.clone()].into();

        let const1 = UOp::const_(DType::Float32, ConstValue::Float(2.0));
        let tensor1_uop = const1.try_reshape(&shape).unwrap();
        let tensor1 = Tensor::new(tensor1_uop);

        let const2 = UOp::const_(DType::Float32, ConstValue::Float(3.0));
        let tensor2_uop = const2.try_reshape(&shape).unwrap();
        let tensor2 = Tensor::new(tensor2_uop);

        // Binary operations should work with matching symbolic shapes
        let sum = tensor1.try_add(&tensor2).unwrap();
        let product = tensor1.try_mul(&tensor2).unwrap();

        // Results should preserve symbolic shape
        let sum_shape = sum.shape().unwrap();
        let product_shape = product.shape().unwrap();

        assert_eq!(sum_shape.len(), 1);
        assert!(sum_shape[0].as_const().is_none()); // Still symbolic

        assert_eq!(product_shape.len(), 1);
        assert!(product_shape[0].as_const().is_none()); // Still symbolic
    }
}
