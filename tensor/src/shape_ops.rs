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
            let total_elements =
                current_shape.iter().try_fold(1usize, |acc, dim| dim.as_const().map(|v| acc * v)).ok_or_else(|| {
                    Error::SymbolicShapeUnsupported { operation: "reshape with -1 inference".to_string() }
                })?;

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
