//! Shape manipulation operations for Tensors.
//!
//! This module provides operations that change tensor shapes without copying data:
//! - Reshape: Change shape while preserving total elements
//! - Permute: Reorder dimensions
//! - Transpose: Swap two dimensions (convenience wrapper for permute)
//! - Expand: Broadcast dimensions from size 1
//! - Squeeze: Remove dimensions of size 1
//! - Unsqueeze: Add dimensions of size 1

use bon::bon;
use snafu::ResultExt;
use strum::{Display, EnumString};

use super::*;

/// Indexing convention for meshgrid.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, EnumString, Display)]
pub enum MeshgridIndexing {
    #[default]
    #[strum(serialize = "ij")]
    Ij,
    #[strum(serialize = "xy")]
    Xy,
}

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

        self.uop().try_reshape(&shape).map(Self::new).context(UOpSnafu)
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

        self.uop().try_permute(normalized_axes).map(Self::new).context(UOpSnafu)
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

        self.uop().try_permute(axes).map(Self::new).context(UOpSnafu)
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

        self.uop().try_expand(&shape).map(Self::new).context(UOpSnafu)
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

        self.uop().try_reshape(&new_shape).map(Self::new).context(UOpSnafu)
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

        self.uop().try_reshape(&new_shape).map(Self::new).context(UOpSnafu)
    }

    /// Reverse elements along specified axes.
    ///
    /// Each axis in the list is flipped (reversed). Supports negative indexing.
    ///
    /// # Examples
    /// ```ignore
    /// let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0]).try_reshape(&[2, 2])?;
    /// let flipped = t.flip(&[0])?;  // Flip along axis 0
    /// ```
    #[track_caller]
    pub fn flip(&self, axes: &[isize]) -> Result<Tensor> {
        let shape = self.shape()?;
        let ndim = shape.len();
        let flip_spec: Vec<bool> =
            (0..ndim).map(|d| axes.iter().any(|&a| Self::normalize_axis(a, ndim).is_ok_and(|na| na == d))).collect();
        self.uop().try_flip(flip_spec).map(Self::new).context(UOpSnafu)
    }

    /// Split tensor into chunks along a dimension.
    ///
    /// Returns a vector of tensors, each with the specified size along the split dimension.
    ///
    /// # Examples
    /// ```ignore
    /// let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0]);
    /// let parts = t.split(&[2, 3], 0)?;  // [2] and [3]
    /// ```
    #[track_caller]
    pub fn split(&self, sizes: &[usize], dim: isize) -> Result<Vec<Tensor>> {
        let shape = self.shape()?;
        let ndim = shape.len();
        let dim = Self::normalize_axis(dim, ndim)?;
        let mut results = Vec::with_capacity(sizes.len());
        let mut offset = 0usize;
        for &size in sizes {
            let ranges: Vec<(isize, isize)> = (0..ndim)
                .map(|d| {
                    if d == dim {
                        (offset as isize, (offset + size) as isize)
                    } else {
                        (0, shape[d].as_const().unwrap() as isize)
                    }
                })
                .collect();
            results.push(self.try_shrink(&ranges)?);
            offset += size;
        }
        Ok(results)
    }

    /// Repeat tensor along each dimension.
    ///
    /// `repeats[i]` is the number of times to repeat along dimension `i`.
    ///
    /// # Examples
    /// ```ignore
    /// let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0]).try_reshape(&[1, 3])?;
    /// let tiled = t.repeat(&[3, 2])?;  // Shape [3, 6]
    /// ```
    #[track_caller]
    pub fn repeat(&self, repeats: &[usize]) -> Result<Tensor> {
        let shape = self.shape()?;
        let ndim = shape.len();
        snafu::ensure!(
            repeats.len() == ndim,
            ShapeMismatchSnafu {
                context: "repeat",
                expected: format!("{} dimensions", ndim),
                actual: format!("{} repeats", repeats.len())
            }
        );
        let mut result = self.clone();
        for (dim, &rep) in repeats.iter().enumerate() {
            if rep == 1 {
                continue;
            }
            let current_shape = result.shape()?;
            let dim_size = current_shape[dim]
                .as_const()
                .ok_or_else(|| Error::SymbolicShapeUnsupported { operation: "repeat".to_string() })?
                as isize;
            // Unsqueeze at dim, expand rep times, then reshape to merge
            result = result.try_unsqueeze(dim as isize)?;
            let mut expand_shape = morok_ir::shape::to_vec_isize(&current_shape).context(UOpSnafu)?;
            expand_shape.insert(dim, rep as isize);
            result = result.try_expand(&expand_shape)?;
            expand_shape[dim] = rep as isize * dim_size;
            expand_shape.remove(dim + 1);
            result = result.try_reshape(&expand_shape)?;
        }
        Ok(result)
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

    /// Pad tensor with zeros (or other padding value).
    ///
    /// Each tuple in `padding` specifies (begin, end) padding for a dimension.
    /// Use 0 for no padding on that side.
    ///
    /// # Examples
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);  // Shape [3]
    /// let padded = t.try_pad(&[(1, 2)]).unwrap();  // Shape [6]: [0, 1, 2, 3, 0, 0]
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Padding values are symbolic (not concrete)
    /// - Number of padding pairs doesn't match dimensions
    #[track_caller]
    pub fn try_pad(&self, padding: &[(isize, isize)]) -> Result<Tensor> {
        let shape = self.shape()?;

        // Empty padding (scalar) → identity
        if padding.is_empty() {
            return Ok(self.clone());
        }

        // Convert to SInt and validate
        snafu::ensure!(
            padding.len() == shape.len(),
            ShapeMismatchSnafu {
                context: "pad",
                expected: format!("{} dimensions", shape.len()),
                actual: format!("{} padding pairs", padding.len())
            }
        );

        // Phase 1: shrink for negative padding (negative padding = cropping)
        let needs_shrink = padding.iter().any(|(b, e)| *b < 0 || *e < 0);
        let base = if needs_shrink {
            let shrink_ranges: Vec<(isize, isize)> = padding
                .iter()
                .zip(shape.iter())
                .map(|((b, e), s)| {
                    let dim = s.as_const().expect("pad with negative values requires concrete shape") as isize;
                    let begin = (-*b).max(0);
                    let end = (dim + *e).min(dim);
                    (begin, end)
                })
                .collect();
            self.try_shrink(&shrink_ranges)?
        } else {
            self.clone()
        };

        // Phase 2: pad with positive-only values
        let pos_padding: Vec<(isize, isize)> = padding.iter().map(|(b, e)| ((*b).max(0), (*e).max(0))).collect();
        if pos_padding.iter().all(|(b, e)| *b == 0 && *e == 0) {
            return Ok(base);
        }

        let padding_sint: Vec<(SInt, SInt)> =
            pos_padding.iter().map(|(begin, end)| (SInt::Const(*begin as usize), SInt::Const(*end as usize))).collect();

        base.uop().try_pad(&padding_sint).map(Self::new).context(UOpSnafu)
    }

    /// Concatenate tensors along an axis.
    ///
    /// All tensors must have the same shape except in the concatenating dimension.
    ///
    /// # Examples
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0]).try_reshape(&[3]).unwrap();
    /// let b = Tensor::from_slice(&[4.0f32, 5.0]).try_reshape(&[2]).unwrap();
    /// let c = Tensor::cat(&[&a, &b], 0).unwrap();  // Shape [5]: [1, 2, 3, 4, 5]
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Tensors have different number of dimensions
    /// - Non-concat dimensions don't match
    #[track_caller]
    pub fn cat(tensors: &[&Tensor], dim: isize) -> Result<Tensor> {
        if tensors.is_empty() {
            return Err(IrConstructionSnafu { details: "cat requires at least one tensor".to_string() }.build());
        }

        let first = tensors[0];
        let first_shape = first.shape()?;
        let ndim = first_shape.len();
        let dim = Self::normalize_axis(dim, ndim)?;

        // Validate all tensors have compatible shapes
        for (i, t) in tensors.iter().enumerate().skip(1) {
            let t_shape = t.shape()?;
            snafu::ensure!(
                t_shape.len() == ndim,
                ShapeMismatchSnafu {
                    context: "cat",
                    expected: format!("{} dimensions", ndim),
                    actual: format!("{} dimensions for tensor {}", t_shape.len(), i)
                }
            );
            for (d, (s1, s2)) in first_shape.iter().zip(t_shape.iter()).enumerate() {
                if d != dim {
                    snafu::ensure!(
                        s1 == s2,
                        ShapeMismatchSnafu {
                            context: format!("cat dimension {}", d),
                            expected: format!("{:?}", s1),
                            actual: format!("{:?}", s2)
                        }
                    );
                }
            }
        }

        // Compute cumulative sizes along concat dimension
        let dim_sizes: Vec<usize> = tensors.iter().map(|t| t.shape().unwrap()[dim].as_const().unwrap_or(0)).collect();
        let total_dim: usize = dim_sizes.iter().sum();

        // Pad each tensor to final size and add
        let mut cumsum = 0usize;
        let padded: Vec<Tensor> = tensors
            .iter()
            .zip(dim_sizes.iter())
            .map(|(t, &sz)| {
                let begin_pad = cumsum;
                let end_pad = total_dim - cumsum - sz;
                cumsum += sz;

                let mut padding = vec![(0isize, 0isize); ndim];
                padding[dim] = (begin_pad as isize, end_pad as isize);
                t.try_pad(&padding)
            })
            .collect::<Result<Vec<_>>>()?;

        // Sum all padded tensors
        let mut result = padded[0].clone();
        for t in padded.iter().skip(1) {
            result = result.try_add(t)?;
        }
        Ok(result)
    }

    /// Stack tensors along a new dimension.
    ///
    /// Creates a new axis at `dim` by unsqueezing each tensor, then concatenating.
    #[track_caller]
    pub fn stack(tensors: &[&Tensor], dim: isize) -> Result<Tensor> {
        let unsqueezed: Vec<Tensor> = tensors.iter().map(|t| t.try_unsqueeze(dim)).collect::<Result<_>>()?;
        Tensor::cat(&unsqueezed.iter().collect::<Vec<_>>(), dim)
    }

    /// Replace a single dimension with multiple dimensions.
    ///
    /// Inverse of flatten: splits dimension `dim` into the shape given by `sizes`.
    #[track_caller]
    pub fn unflatten(&self, dim: isize, sizes: &[isize]) -> Result<Tensor> {
        let shape = self.shape()?;
        let dim = Self::normalize_axis(dim, shape.len())?;
        let mut new_shape = morok_ir::shape::to_vec_isize(&shape).context(UOpSnafu)?;
        new_shape.splice(dim..=dim, sizes.iter().copied());
        self.try_reshape(&new_shape)
    }

    /// Create coordinate grids from 1D tensors.
    ///
    /// `indexing`: `Ij` (matrix/default) or `Xy` (Cartesian, swaps first two inputs).
    #[track_caller]
    pub fn meshgrid(tensors: &[&Tensor], indexing: MeshgridIndexing) -> Result<Vec<Tensor>> {
        let n = tensors.len();
        let sizes: Vec<usize> = tensors.iter().map(|t| t.numel().unwrap()).collect();
        // For "xy" indexing, swap the first two inputs
        let swapped: Vec<usize> = if indexing == MeshgridIndexing::Xy && n >= 2 {
            let mut s: Vec<usize> = (0..n).collect();
            s.swap(0, 1);
            s
        } else {
            (0..n).collect()
        };
        // Output shape is [sizes[swapped[0]], sizes[swapped[1]], ...]
        let out_shape: Vec<isize> = swapped.iter().map(|&i| sizes[i] as isize).collect();
        tensors
            .iter()
            .enumerate()
            .map(|(i, t)| {
                // Position of this tensor's dimension in the output
                let pos = swapped.iter().position(|&s| s == i).unwrap();
                let mut shape = vec![1isize; n];
                shape[pos] = sizes[i] as isize;
                t.flatten()?.try_reshape(&shape)?.try_expand(&out_shape)
            })
            .collect()
    }

    /// Get the shape of this tensor as a new tensor.
    ///
    /// Returns a 1D tensor of int64 containing the shape dimensions.
    /// This is useful for ONNX Shape operator compatibility.
    ///
    /// # Examples
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// let t = Tensor::from_slice(&[1.0f32; 6]).try_reshape(&[2, 3]).unwrap();
    /// let shape_tensor = t.shape_tensor().unwrap();  // Tensor([2, 3]) with dtype int64
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if shape is unknown or contains symbolic dimensions.
    #[track_caller]
    pub fn shape_tensor(&self) -> Result<Tensor> {
        let shape = self.shape()?;

        // Extract concrete dimensions
        let dims: Vec<i64> = shape
            .iter()
            .map(|d| {
                d.as_const()
                    .map(|v| v as i64)
                    .ok_or_else(|| Error::SymbolicShapeUnsupported { operation: "shape_tensor".to_string() })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Tensor::from_slice(&dims))
    }

    /// Shrink (slice) tensor along each dimension.
    ///
    /// Each tuple in `ranges` specifies (begin, end) for a dimension.
    /// Use (0, size) to keep full dimension.
    ///
    /// # Examples
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0]);
    /// let sliced = t.try_shrink(&[(1, 4)]).unwrap();  // Elements [2, 3, 4]
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if negative indices are used with symbolic shape dimensions.
    #[track_caller]
    pub fn try_shrink(&self, ranges: &[(isize, isize)]) -> Result<Tensor> {
        let shape = self.shape()?;

        // Empty ranges (scalar) → identity
        if ranges.is_empty() {
            return Ok(self.clone());
        }

        // Convert to SInt and handle negative indices
        let ranges_sint: Vec<(SInt, SInt)> = ranges
            .iter()
            .enumerate()
            .map(|(dim_idx, &(begin, end))| {
                // Only need dimension size if we have negative indices
                let (normalized_begin, normalized_end) = if begin < 0 || end < 0 {
                    let dim_size = shape[dim_idx].as_const().ok_or_else(|| Error::SymbolicShapeUnsupported {
                        operation: "shrink with negative indices".to_string(),
                    })? as isize;

                    let nb = if begin < 0 { dim_size + begin } else { begin };
                    let ne = if end < 0 { dim_size + end } else { end };
                    (nb, ne)
                } else {
                    (begin, end)
                };

                Ok((SInt::Const(normalized_begin as usize), SInt::Const(normalized_end as usize)))
            })
            .collect::<Result<Vec<_>>>()?;

        self.uop().try_shrink(&ranges_sint).map(Self::new).context(UOpSnafu)
    }

    /// Center-crop or center-pad each specified axis to the target size.
    ///
    /// For axes where `target < current`, crops from the center.
    /// For axes where `target > current`, pads symmetrically around the center.
    /// Axes where `target == current` are unchanged.
    ///
    /// `axes` specifies which dimensions to apply (default: all).
    pub fn center_crop_pad(&self, target_shape: &[usize], axes: Option<&[usize]>) -> Result<Tensor> {
        let shape = self.shape()?;
        let ndim = shape.len();
        let default_axes: Vec<usize> = (0..ndim).collect();
        let axes = axes.unwrap_or(&default_axes);

        let mut shrink_arg: Vec<(isize, isize)> =
            (0..ndim).map(|i| (0, shape[i].as_const().unwrap_or(1) as isize)).collect();
        let mut pad_arg: Vec<(isize, isize)> = vec![(0, 0); ndim];

        for (&s, &ax) in target_shape.iter().zip(axes.iter()) {
            let s = s as isize;
            let tx = shape[ax].as_const().unwrap_or(1) as isize;
            if s < tx {
                shrink_arg[ax] = (tx / 2 - (s + 1) / 2, tx / 2 + s / 2);
            } else if s > tx {
                pad_arg[ax] = ((s - tx) / 2, (s - tx + 1) / 2);
            }
        }

        self.try_shrink(&shrink_arg)?.try_pad(&pad_arg)
    }

    // =========================================================================
    // Helper Methods
    // =========================================================================

    /// Get the concrete shape of this tensor.
    pub fn shape(&self) -> Result<Shape> {
        self.uop().shape().context(UOpSnafu)?.cloned().ok_or(Error::ShapeUnknown)
    }

    /// Get the number of dimensions (rank).
    pub fn ndim(&self) -> Result<usize> {
        Ok(self.shape()?.len())
    }

    /// Total number of elements. Fails if any dimension is symbolic.
    pub fn numel(&self) -> Result<usize> {
        self.shape()?.iter().try_fold(1usize, |acc, d| {
            d.as_const().map(|v| acc * v).ok_or(Error::SymbolicShapeUnsupported { operation: "numel".into() })
        })
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

        // Validate no other negative values (0 is valid for zero-size tensors)
        for &dim in shape_spec {
            snafu::ensure!(dim >= 0 || dim == -1, NegativeDimensionSnafu { dim });
        }

        // Calculate new shape
        let new_shape: Shape = if let Some(_infer_pos) = minus_one_pos {
            // Inference requires concrete shape to compute total elements
            let total_elements =
                current_shape.iter().try_fold(1usize, |acc, dim| dim.as_const().map(|v| acc * v)).ok_or_else(|| {
                    Error::SymbolicShapeUnsupported { operation: "reshape with -1 inference".to_string() }
                })?;

            // Calculate product of known (non -1) dimensions.
            // Tinygrad: -prod(source) // prod(target_with_-1). When target has both 0 and -1,
            // prod is 0 → ZeroDivisionError. We match that: known_product=0 is a size mismatch.
            let known_product: usize = shape_spec.iter().filter(|&&s| s != -1).map(|&s| s as usize).product();

            snafu::ensure!(
                known_product > 0 && total_elements % known_product == 0,
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

    /// Upper triangular mask: row + diagonal <= col.
    fn tri(rows: i64, cols: i64, diagonal: i64) -> Result<Tensor> {
        let row = Tensor::arange(0, Some(rows), None)?.try_unsqueeze(-1)?;
        let col = Tensor::arange(0, Some(cols), None)?;
        let diag = Tensor::const_(ConstValue::Int(diagonal), DType::Int32);
        row.try_add(&diag)?.try_le(&col)
    }

    /// Keep upper triangle, zero below. Matches Tinygrad `Tensor.triu(diagonal)`.
    pub fn triu(&self, diagonal: i64) -> Result<Tensor> {
        let shape = self.shape()?;
        let ndim = shape.len();
        let r = shape[ndim - 2].as_const().unwrap() as i64;
        let c = shape[ndim - 1].as_const().unwrap() as i64;
        let mask = Self::tri(r, c, diagonal)?;
        let zero = Tensor::new(self.uop().const_like(ConstValue::zero(self.uop().dtype().scalar().unwrap())));
        self.where_(&mask, &zero)
    }

    /// Keep lower triangle, zero above. Matches Tinygrad `Tensor.tril(diagonal)`.
    pub fn tril(&self, diagonal: i64) -> Result<Tensor> {
        let shape = self.shape()?;
        let ndim = shape.len();
        let r = shape[ndim - 2].as_const().unwrap() as i64;
        let c = shape[ndim - 1].as_const().unwrap() as i64;
        let mask = Self::tri(r, c, diagonal + 1)?;
        let zero = Tensor::new(self.uop().const_like(ConstValue::zero(self.uop().dtype().scalar().unwrap())));
        zero.where_(&mask, self)
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

#[bon]
impl Tensor {
    /// Slice tensor with Python-style indexing: negative indices, steps, and axis selection.
    #[builder]
    pub fn slice_with(
        &self,
        starts: &[i64],
        ends: &[i64],
        axes: Option<&[i64]>,
        steps: Option<&[i64]>,
    ) -> Result<Tensor> {
        let shape = self.shape()?;
        let ndim = shape.len();

        let axes: Vec<usize> = axes
            .map(|v| v.iter().map(|&a| if a < 0 { (ndim as i64 + a) as usize } else { a as usize }).collect())
            .unwrap_or_else(|| (0..starts.len()).collect());

        let default_steps;
        let steps = match steps {
            Some(s) => s,
            None => {
                default_steps = vec![1i64; starts.len()];
                &default_steps
            }
        };

        let mut ranges: Vec<(isize, isize)> =
            (0..ndim).map(|d| (0isize, shape[d].as_const().unwrap() as isize)).collect();
        let mut flip_axes: Vec<isize> = Vec::new();

        for (i, &axis) in axes.iter().enumerate() {
            let d = shape[axis].as_const().unwrap() as i64;
            let step = steps[i];
            if step == 0 {
                return Err(crate::error::Error::IrConstruction { details: "Slice step cannot be 0".into() });
            }

            let (lower, upper) = if step > 0 { (0i64, d) } else { (-1i64, d - 1) };
            let mut s = starts[i].clamp(-d, d);
            if s < 0 {
                s += d;
            }
            let s = s.clamp(lower, upper);

            let mut e = ends[i].clamp(-d - 1, d);
            if e < 0 {
                e += d;
            }
            let e = e.clamp(lower, upper);

            if step * (e - s) < 0 {
                ranges[axis] = (0, 0);
            } else if step < 0 {
                flip_axes.push(axis as isize);
                ranges[axis] = ((e + 1) as isize, (s + 1) as isize);
            } else {
                ranges[axis] = (s as isize, e as isize);
            }
        }

        let mut result = self.try_shrink(&ranges)?;
        if !flip_axes.is_empty() {
            result = result.flip(&flip_axes)?;
        }

        for (i, &axis) in axes.iter().enumerate() {
            let abs_step = steps[i].unsigned_abs() as usize;
            if abs_step <= 1 {
                continue;
            }
            let cur = result.shape()?;
            let size = cur[axis].as_const().unwrap();
            let padded = size.div_ceil(abs_step) * abs_step;
            if padded > size {
                let mut p = vec![(0isize, 0isize); cur.len()];
                p[axis] = (0, (padded - size) as isize);
                result = result.try_pad(&p)?;
            }
            let n = padded / abs_step;
            let cs = result.shape()?;
            let mut rs: Vec<isize> = Vec::new();
            for (d, dim) in cs.iter().enumerate() {
                if d == axis {
                    rs.push(n as isize);
                    rs.push(abs_step as isize);
                } else {
                    rs.push(dim.as_const().unwrap() as isize);
                }
            }
            result = result.try_reshape(&rs)?;
            let ss = result.shape()?;
            let sr: Vec<(isize, isize)> = ss
                .iter()
                .enumerate()
                .map(|(d, dim)| if d == axis + 1 { (0, 1) } else { (0, dim.as_const().unwrap() as isize) })
                .collect();
            result = result.try_shrink(&sr)?;
            let fs: Vec<isize> = result
                .shape()?
                .iter()
                .enumerate()
                .filter(|&(d, _)| d != axis + 1)
                .map(|(_, dim)| dim.as_const().unwrap() as isize)
                .collect();
            result = result.try_reshape(&fs)?;
        }

        if !flip_axes.is_empty() || steps.iter().any(|&s| s.unsigned_abs() > 1) {
            result = result.contiguous();
        }

        Ok(result)
    }
}
