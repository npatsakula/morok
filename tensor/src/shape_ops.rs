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
use snafu::{OptionExt, ResultExt};
use strum::{Display, EnumString};
use svod_ir::IntoShrinkRange;

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

/// End bounds for [`Tensor::slice_with`].
///
/// Built through `From` in the builder setter, so `&[i64]` (ONNX-style, where
/// the `i64::MAX` sentinel still reads as "to the end"), `&[Option<i64>]`
/// (`None` is "to the end") and owned `Vec`s of either all work. Omitting the
/// `ends` setter slices every listed axis to its end.
#[derive(Debug, Clone, Copy)]
pub enum SliceEnds<'a> {
    /// Plain bounds; values above `i64::MAX / 2` mean "to the end".
    Bounds(&'a [i64]),
    /// Per-axis bounds where `None` means "to the end".
    Optional(&'a [Option<i64>]),
}

impl SliceEnds<'_> {
    /// End bound of axis `i`, or `None` for "to the end".
    fn get(&self, i: usize) -> Option<i64> {
        match self {
            Self::Bounds(b) => b.get(i).copied().filter(|&e| e <= i64::MAX / 2),
            Self::Optional(o) => o.get(i).copied().flatten(),
        }
    }
}

macro_rules! impl_slice_ends_from {
    ($variant:ident, $elem:ty) => {
        impl<'a> From<&'a [$elem]> for SliceEnds<'a> {
            fn from(v: &'a [$elem]) -> Self {
                Self::$variant(v)
            }
        }

        impl<'a> From<&'a Vec<$elem>> for SliceEnds<'a> {
            fn from(v: &'a Vec<$elem>) -> Self {
                Self::$variant(v)
            }
        }

        impl<'a, const N: usize> From<&'a [$elem; N]> for SliceEnds<'a> {
            fn from(v: &'a [$elem; N]) -> Self {
                Self::$variant(v)
            }
        }
    };
}

impl_slice_ends_from!(Bounds, i64);
impl_slice_ends_from!(Optional, Option<i64>);

impl Tensor {
    /// Reshape tensor to a new shape.
    ///
    /// The total number of elements must remain the same.
    /// Supports negative indices: -1 means "infer this dimension".
    ///
    /// # Examples
    ///
    /// ```
    /// # use svod_tensor::Tensor;
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
    pub fn try_reshape(&self, new_shape: impl IntoIterator<Item = impl Into<SInt>>) -> Result<Tensor> {
        origin_call!("reshape");
        let dims: Vec<SInt> = new_shape.into_iter().map(Into::into).collect();

        // Handle Infer (-1) if present
        let infer_count = dims.iter().filter(|d| d.is_infer()).count();
        snafu::ensure!(infer_count <= 1, MultipleInferDimensionsSnafu);

        let shape: Shape = if infer_count == 1 {
            let current_shape = self.shape()?;
            let total_elements =
                current_shape.iter().try_fold(1usize, |acc, dim| dim.as_const().map(|v| acc * v)).ok_or_else(|| {
                    ErrorKind::SymbolicShapeUnsupported { operation: "reshape with -1 inference".to_string() }
                })?;
            let known_product: usize = dims
                .iter()
                .filter(|d| !d.is_infer())
                .map(|d| d.as_const().context(SymbolicShapeUnsupportedSnafu { operation: "reshape with -1 inference" }))
                .product::<KindResult<usize>>()?;
            snafu::ensure!(
                known_product > 0 && total_elements % known_product == 0,
                ReshapeSizeMismatchSnafu { operation: "reshape with inference".to_string() }
            );
            let inferred = total_elements / known_product;
            dims.iter().map(|d| if d.is_infer() { SInt::Const(inferred) } else { d.clone() }).collect()
        } else {
            dims.into()
        };

        self.uop().try_reshape(&shape).map(Self::new).context(UOpSnafu).map_err(Into::into)
    }

    /// Expand tensor to a new shape with mixed concrete/symbolic dimensions.
    #[track_caller]
    pub fn try_expand(&self, new_shape: impl IntoIterator<Item = impl Into<SInt>>) -> Result<Tensor> {
        origin_call!("expand");
        let requested: Vec<SInt> = new_shape.into_iter().map(Into::into).collect();
        // Resolve Infer (-1) to current dimension (expand's "keep" semantics)
        let current_shape = self.shape()?;
        let ndim = current_shape.len();
        let shape: Shape = requested
            .into_iter()
            .enumerate()
            .map(|(i, s)| match s.is_infer() {
                // `-1` keeps the current extent, so it needs one to keep.
                true => current_shape.get(i).cloned().context(AxisOutOfRangeSnafu { axis: i as isize, ndim }),
                false => Ok(s),
            })
            .collect::<KindResult<_>>()?;
        self.uop().try_expand(&shape).map(Self::new).context(UOpSnafu).map_err(Into::into)
    }

    /// Permute (reorder) tensor dimensions.
    ///
    /// The axes parameter specifies the new order of dimensions.
    /// Each axis index 0..ndim must appear exactly once.
    ///
    /// # Examples
    ///
    /// ```
    /// # use svod_tensor::Tensor;
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
        origin_call!("permute");
        let shape = self.shape()?;
        let ndim = shape.len();

        // Normalize negative indices and validate
        let normalized_axes = self.normalize_axes(axes, ndim)?;

        self.uop().try_permute(normalized_axes).map(Self::new).context(UOpSnafu).map_err(Into::into)
    }

    /// Transpose two dimensions.
    ///
    /// Convenience method for swapping two dimensions.
    /// Equivalent to permute with the two dimensions swapped.
    ///
    /// # Examples
    ///
    /// ```
    /// # use svod_tensor::Tensor;
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
        origin_call!("transpose");
        let shape = self.shape()?;
        let ndim = shape.len();

        // Normalize negative indices
        let d0 = Self::normalize_axis(dim0, ndim)?;
        let d1 = Self::normalize_axis(dim1, ndim)?;

        // Build permutation with swapped dimensions
        let mut axes: Vec<usize> = (0..ndim).collect();
        axes.swap(d0, d1);

        self.uop().try_permute(axes).map(Self::new).context(UOpSnafu).map_err(Into::into)
    }

    /// Expand (broadcast) dimensions.
    ///
    /// Dimensions of size 1 can be expanded to larger sizes.
    /// Use -1 to keep the current dimension size.
    ///
    /// # Examples
    ///
    /// ```
    /// # use svod_tensor::Tensor;
    /// // Tensor with shape [1, 3, 1]
    /// // t.try_expand(&[4, -1, 5]) -> shape [4, 3, 5]
    /// ```
    ///
    /// Squeeze dimensions of size 1.
    ///
    /// If dim is None, removes all dimensions of size 1.
    /// If dim is Some(axis), removes only that dimension if it's size 1.
    ///
    /// # Examples
    ///
    /// ```
    /// # use svod_tensor::Tensor;
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
        origin_call!("squeeze");
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
                    .ok_or_else(|| ErrorKind::SymbolicShapeUnsupported { operation: "squeeze".to_string() })?;

                snafu::ensure!(dim_size == 1, SqueezeDimensionNotOneSnafu { dim: normalized_axis, size: dim_size });

                // Remove the specified dimension
                shape
                    .iter()
                    .enumerate()
                    .filter_map(|(i, s)| if i != normalized_axis { Some(s.clone()) } else { None })
                    .collect()
            }
        };

        self.uop().try_reshape(&new_shape).map(Self::new).context(UOpSnafu).map_err(Into::into)
    }

    /// Add a dimension of size 1.
    ///
    /// Inserts a new dimension at the specified position.
    /// Supports negative indices: -1 means after the last dimension.
    ///
    /// # Examples
    ///
    /// ```
    /// # use svod_tensor::Tensor;
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
        origin_call!("unsqueeze");
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

        self.uop().try_reshape(&new_shape).map(Self::new).context(UOpSnafu).map_err(Into::into)
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
        origin_call!("flip");
        let shape = self.shape()?;
        let ndim = shape.len();
        let flip_spec: Vec<bool> =
            (0..ndim).map(|d| axes.iter().any(|&a| Self::normalize_axis(a, ndim).is_ok_and(|na| na == d))).collect();
        self.uop().try_flip(flip_spec).map(Self::new).context(UOpSnafu).map_err(Into::into)
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
        origin_call!("split");
        let shape = self.shape()?;
        let ndim = shape.len();
        let dim = Self::normalize_axis(dim, ndim)?;
        let mut results = Vec::with_capacity(sizes.len());
        let mut offset = 0usize;
        for &size in sizes {
            let ranges: Vec<Option<(isize, isize)>> = (0..ndim)
                .map(|d| {
                    if d == dim {
                        Some((offset as isize, (offset + size) as isize))
                    } else {
                        None // keep entire dim (supports symbolic)
                    }
                })
                .collect();
            results.push(self.try_shrink(ranges)?);
            offset += size;
        }
        Ok(results)
    }

    /// Repeat tensor along each dimension.
    ///
    /// `repeats[i]` is the number of times to repeat along dimension `i`.
    /// Accepts `&[SInt]` — supports both concrete and symbolic repeat counts.
    ///
    /// # Examples
    /// ```ignore
    /// use svod_ir::SInt;
    /// let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0]).try_reshape(&[1, 3])?;
    /// let tiled = t.repeat(&[SInt::from(3), SInt::from(2)])?;  // Shape [3, 6]
    /// ```
    #[track_caller]
    pub fn repeat(&self, repeats: &[SInt]) -> Result<Tensor> {
        origin_call!("repeat");
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
        for (dim, rep) in repeats.iter().enumerate() {
            if rep.as_const() == Some(1) {
                continue;
            }
            let current_shape = result.shape()?;
            let dim_size = &current_shape[dim];
            // Unsqueeze at dim, expand rep times, then reshape to merge.
            result = result.try_unsqueeze(dim as isize)?;
            let mut expand_shape: Vec<SInt> = current_shape.iter().cloned().collect();
            expand_shape.insert(dim, rep.clone());
            result = result.try_expand(&expand_shape)?;
            expand_shape[dim] = rep * dim_size;
            expand_shape.remove(dim + 1);
            result = result.try_reshape(expand_shape)?;
        }
        Ok(result)
    }

    /// Split tensor into approximately equal chunks along a dimension.
    ///
    /// Attempts to split the tensor into `chunks` parts of roughly equal size.
    /// If the dimension is not evenly divisible, earlier chunks may be larger.
    /// Returns at most `chunks` parts (may be fewer if the dimension is smaller).
    ///
    /// # Examples
    /// ```ignore
    /// let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).try_reshape(&[2, 3])?;
    /// let parts = t.chunk(2, 1)?;  // Two tensors of shape [2, 2] and [2, 1]
    /// ```
    #[track_caller]
    pub fn chunk(&self, chunks: usize, dim: isize) -> Result<Vec<Tensor>> {
        origin_call!("chunk");
        snafu::ensure!(
            chunks > 0,
            ParamRangeSnafu { op: "chunk", param: "chunks", value: chunks.to_string(), constraint: "> 0" }
        );
        let shape = self.shape()?;
        let ndim = shape.len();
        let dim = Self::normalize_axis(dim, ndim)?;
        let dim_size = shape[dim].as_const().context(SymbolicShapeUnsupportedSnafu { operation: "chunk" })?;
        // Empty dim → `chunks` zero-sized tensors.
        if dim_size == 0 {
            return self.split(&vec![0; chunks], dim as isize);
        }
        let actual_chunks = chunks.min(dim_size);
        let chunk_size = dim_size.div_ceil(actual_chunks);
        let mut sizes = Vec::with_capacity(actual_chunks);
        let mut remaining = dim_size;
        while remaining > 0 {
            let sz = chunk_size.min(remaining);
            sizes.push(sz);
            remaining -= sz;
        }
        self.split(&sizes, dim as isize)
    }

    /// Alias for [`try_reshape`](Self::try_reshape) — `view` does not enforce
    /// contiguity, since the lazy IR backend picks copy-vs-view per kernel.
    #[track_caller]
    pub fn view(&self, new_shape: impl IntoIterator<Item = impl Into<SInt>>) -> Result<Tensor> {
        origin_call!("view");
        self.try_reshape(new_shape)
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
        origin_call!("flatten");
        self.try_reshape([-1])
    }

    /// Pad tensor with zeros (or other padding value).
    ///
    /// Each tuple in `padding` specifies (begin, end) padding for a dimension.
    /// Use 0 for no padding on that side.
    ///
    /// # Examples
    ///
    /// ```
    /// # use svod_tensor::Tensor;
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
        origin_call!("pad");
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
                    let dim =
                        s.as_const().context(SymbolicShapeUnsupportedSnafu { operation: "pad with negative values" })?
                            as isize;
                    let begin = (-*b).max(0);
                    let end = (dim + *e).min(dim);
                    Ok((begin, end))
                })
                .collect::<Result<Vec<_>>>()?;
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

        base.uop().try_pad(&padding_sint).map(Self::new).context(UOpSnafu).map_err(Into::into)
    }

    /// Concatenate tensors along an axis.
    ///
    /// All tensors must have the same shape except in the concatenating dimension.
    ///
    /// # Examples
    ///
    /// ```
    /// # use svod_tensor::Tensor;
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
        origin_call!("cat");
        if tensors.is_empty() {
            return Err(IrConstructionSnafu { details: "cat requires at least one tensor".to_string() }.build().into());
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

        // Equal-length inputs concatenate through a stacked axis instead of
        // interleaved pads (Tinygrad `Tensor.cat`). Every input is then read at
        // the same offset within `dim`, so shared producers hash-cons into a
        // single node rather than one copy per slice.
        if tensors[1..].iter().all(|t| t.shape().is_ok_and(|s| s[dim] == first_shape[dim])) {
            let stacked = Self::stack(tensors, dim as isize)?;
            let stacked_shape = stacked.shape()?;
            let merged: Vec<SInt> = stacked_shape[..dim]
                .iter()
                .cloned()
                .chain([&stacked_shape[dim] * &stacked_shape[dim + 1]])
                .chain(stacked_shape[dim + 2..].iter().cloned())
                .collect();
            return stacked.try_reshape(merged);
        }

        Self::cat_padded(tensors, dim, ndim)
    }

    /// Concatenate by padding each input to the output extent and summing.
    ///
    /// Shapes are assumed validated by the caller.
    fn cat_padded(tensors: &[&Tensor], dim: usize, ndim: usize) -> Result<Tensor> {
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
        padded[1..].iter().try_fold(padded[0].clone(), |acc, t| acc.try_add(t))
    }

    /// Stack tensors along a new dimension.
    ///
    /// Creates a new axis at `dim` by unsqueezing each tensor, then concatenating.
    ///
    /// # Errors
    ///
    /// Returns error if `tensors` is empty or the shapes are not all identical.
    #[track_caller]
    pub fn stack(tensors: &[&Tensor], dim: isize) -> Result<Tensor> {
        origin_call!("stack");
        let first = tensors
            .first()
            .ok_or_else(|| IrConstructionSnafu { details: "stack requires at least one tensor".to_string() }.build())?;
        let first_shape = first.shape()?;
        for (i, t) in tensors.iter().enumerate().skip(1) {
            let t_shape = t.shape()?;
            snafu::ensure!(
                t_shape == first_shape,
                ShapeMismatchSnafu {
                    context: "stack",
                    expected: format!("{:?}", first_shape),
                    actual: format!("{:?} for tensor {}", t_shape, i)
                }
            );
        }

        let ndim = first_shape.len() + 1;
        let dim = Self::normalize_axis(dim, ndim)?;
        let stacked = Self::new(UOp::stack(tensors.iter().map(|t| t.uop().clone()).collect()));
        if dim == 0 {
            return Ok(stacked);
        }
        // Move the new leading axis into position `dim`.
        let axes: Vec<isize> = (1..=dim).chain([0]).chain(dim + 1..ndim).map(|a| a as isize).collect();
        stacked.try_permute(&axes)
    }

    /// Replace a single dimension with multiple dimensions.
    ///
    /// Inverse of flatten: splits dimension `dim` into the shape given by `sizes`.
    #[track_caller]
    pub fn unflatten(&self, dim: isize, sizes: &[isize]) -> Result<Tensor> {
        origin_call!("unflatten");
        let shape = self.shape()?;
        let dim = Self::normalize_axis(dim, shape.len())?;
        // Kept as `SInt` so untouched symbolic dims pass straight through.
        let mut new_shape: Vec<SInt> = shape.iter().cloned().collect();
        new_shape.splice(dim..=dim, sizes.iter().map(SInt::from));
        self.try_reshape(new_shape)
    }

    /// Create coordinate grids from 1D tensors.
    ///
    /// `indexing`: `Ij` (matrix/default) or `Xy` (Cartesian, swaps first two inputs).
    #[track_caller]
    pub fn meshgrid(tensors: &[&Tensor], indexing: MeshgridIndexing) -> Result<Vec<Tensor>> {
        origin_call!("meshgrid");
        let n = tensors.len();
        let sizes: Vec<usize> = tensors.iter().map(|t| t.numel()).collect::<Result<_>>()?;
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
    /// # use svod_tensor::Tensor;
    /// let t = Tensor::from_slice(&[1.0f32; 6]).try_reshape(&[2, 3]).unwrap();
    /// let shape_tensor = t.shape_tensor().unwrap();  // Tensor([2, 3]) with dtype int64
    /// ```
    ///
    /// # Errors
    ///
    /// Supports symbolic dimensions — symbolic dims produce scalar UOp tensors.
    #[track_caller]
    pub fn shape_tensor(&self) -> Result<Tensor> {
        origin_call!("shape_tensor");
        let shape = self.shape()?;

        // If all concrete, fast path
        if shape.iter().all(|d| d.is_const()) {
            let dims: Vec<i64> = shape.iter().map(|d| d.as_const().unwrap() as i64).collect();
            return Ok(Tensor::from_slice(&dims));
        }

        // Mixed concrete/symbolic: create scalar tensors and cat
        let shape_sint: smallvec::SmallVec<[SInt; 4]> = smallvec::smallvec![SInt::from(1usize)];
        let scalars: KindResult<Vec<Tensor>> = shape
            .iter()
            .map(|d| {
                let uop = d.to_uop(svod_dtype::DType::Int64);
                uop.try_reshape(&shape_sint).map(Tensor::new).context(UOpSnafu)
            })
            .collect();
        let scalars = scalars?;
        let refs: Vec<&Tensor> = scalars.iter().collect();
        Tensor::cat(&refs, 0)
    }

    /// Shrink (slice) tensor along each dimension.
    ///
    /// Each tuple in `ranges` specifies (begin, end) for a dimension.
    /// Use (0, size) to keep full dimension.
    ///
    /// # Examples
    ///
    /// ```
    /// # use svod_tensor::Tensor;
    /// let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0]);
    /// let sliced = t.try_shrink(&[(1, 4)]).unwrap();  // Elements [2, 3, 4]
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if negative indices are used with symbolic shape dimensions.
    #[track_caller]
    pub fn try_shrink<R: IntoShrinkRange>(&self, ranges: impl IntoIterator<Item = R>) -> Result<Tensor> {
        origin_call!("shrink");
        use svod_ir::ShrinkRange;

        let shape = self.shape()?;
        let resolved: Vec<ShrinkRange> = ranges.into_iter().map(|r| r.into_shrink_range()).collect();

        // Empty ranges (scalar) → identity
        if resolved.is_empty() {
            return Ok(self.clone());
        }

        // Check if all ranges are None (no-op)
        if resolved.iter().all(|r| matches!(r, ShrinkRange::None)) {
            return Ok(self.clone());
        }

        // Convert to (SInt, SInt), resolving negative isize indices.
        // `ShrinkRange::None` means "keep entire dim".
        let ranges_sint: Vec<(SInt, SInt)> = resolved
            .into_iter()
            .enumerate()
            .map(|(dim_idx, range)| match range {
                ShrinkRange::None => Ok((SInt::Const(0), shape[dim_idx].clone())),
                ShrinkRange::Sint(begin, end) => Ok((begin, end)),
                ShrinkRange::Isize(begin, end) => {
                    let (nb, ne) = if begin < 0 || end < 0 {
                        let dim_size = shape[dim_idx].as_const().ok_or_else(|| ErrorKind::SymbolicShapeUnsupported {
                            operation: "shrink with negative indices".to_string(),
                        })? as isize;
                        (if begin < 0 { dim_size + begin } else { begin }, if end < 0 { dim_size + end } else { end })
                    } else {
                        (begin, end)
                    };
                    // Still negative after normalization: `as usize` would wrap
                    // to a huge extent instead of reporting the bad index.
                    snafu::ensure!(nb >= 0, NegativeDimensionSnafu { dim: begin });
                    snafu::ensure!(ne >= 0, NegativeDimensionSnafu { dim: end });
                    Ok((SInt::Const(nb as usize), SInt::Const(ne as usize)))
                }
            })
            .collect::<Result<Vec<_>>>()?;

        self.uop().try_shrink(&ranges_sint).map(Self::new).context(UOpSnafu).map_err(Into::into)
    }

    /// Slice `len` elements starting at `start` along `dim` (torch `narrow`).
    ///
    /// Negative `dim` counts from the end; `start` and `len` may be symbolic.
    ///
    /// # Examples
    ///
    /// ```
    /// # use svod_tensor::Tensor;
    /// let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0]);
    /// let mid = t.narrow(0, 1usize, 3usize).unwrap();  // Elements [2, 3, 4]
    /// assert_eq!(mid.dims().unwrap(), vec![3]);
    /// ```
    #[track_caller]
    pub fn narrow(&self, dim: isize, start: impl Into<SInt>, len: impl Into<SInt>) -> Result<Tensor> {
        origin_call!("narrow");
        let ndim = self.ndim()?;
        let dim = Self::normalize_axis(dim, ndim)?;
        let start: SInt = start.into();
        let end = &start + len.into();
        let ranges: Vec<Option<(SInt, SInt)>> =
            (0..ndim).map(|d| (d == dim).then(|| (start.clone(), end.clone()))).collect();
        self.try_shrink(ranges)
    }

    /// Center-crop or center-pad each specified axis to the target size.
    ///
    /// For axes where `target < current`, crops from the center.
    /// For axes where `target > current`, pads symmetrically around the center.
    /// Axes where `target == current` are unchanged.
    ///
    /// `axes` selects the dimensions to apply (default: all); negative axes
    /// count from the end. Untouched axes keep their extent, symbolic included.
    #[track_caller]
    pub fn center_crop_pad(&self, target_shape: &[usize], axes: Option<&[isize]>) -> Result<Tensor> {
        origin_call!("center_crop_pad");
        let ndim = self.ndim()?;
        let axes: Vec<usize> = match axes {
            Some(a) => a.iter().map(|&ax| Self::normalize_axis(ax, ndim)).collect::<Result<_>>()?,
            None => (0..ndim).collect(),
        };

        // `None` keeps the dim as-is, so symbolic axes outside `axes` survive.
        let mut shrink_arg: Vec<Option<(isize, isize)>> = vec![None; ndim];
        let mut pad_arg: Vec<(isize, isize)> = vec![(0, 0); ndim];

        for (&s, &ax) in target_shape.iter().zip(axes.iter()) {
            let s = s as isize;
            let tx = self.dim_const(ax as isize)? as isize;
            if s < tx {
                shrink_arg[ax] = Some((tx / 2 - (s + 1) / 2, tx / 2 + s / 2));
            } else if s > tx {
                pad_arg[ax] = ((s - tx) / 2, (s - tx + 1) / 2);
            }
        }

        self.try_shrink(shrink_arg)?.try_pad(&pad_arg)
    }

    // =========================================================================
    // Helper Methods
    // =========================================================================

    /// Get the concrete shape of this tensor.
    pub fn shape(&self) -> Result<Shape> {
        self.uop().shape().context(UOpSnafu)?.cloned().ok_or(ErrorKind::ShapeUnknown.into())
    }

    /// Get the number of dimensions (rank).
    pub fn ndim(&self) -> Result<usize> {
        Ok(self.shape()?.len())
    }

    /// Total number of elements. Fails if any dimension is symbolic.
    pub fn numel(&self) -> Result<usize> {
        self.shape()?
            .iter()
            .try_fold(1usize, |acc, d| {
                d.as_const().map(|v| acc * v).ok_or(ErrorKind::SymbolicShapeUnsupported { operation: "numel".into() })
            })
            .map_err(Into::into)
    }

    /// Total number of elements, keeping symbolic dimensions in the product.
    pub fn numel_sint(&self) -> Result<SInt> {
        Ok(self.shape()?.iter().fold(SInt::Const(1), |acc, d| acc * d))
    }

    /// Concrete dimensions of this tensor.
    ///
    /// # Errors
    ///
    /// Fails if any dimension is symbolic; use [`shape`](Self::shape) to keep
    /// symbolic dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// # use svod_tensor::Tensor;
    /// let t = Tensor::from_slice(&[1.0f32; 6]).try_reshape(&[2, 3]).unwrap();
    /// assert_eq!(t.dims().unwrap(), vec![2, 3]);
    /// ```
    pub fn dims(&self) -> Result<Vec<usize>> {
        svod_ir::shape::to_vec_usize(&self.shape()?).context(UOpSnafu).map_err(Into::into)
    }

    /// Get the size of a specific dimension.
    ///
    /// Supports negative indexing (e.g., -1 for last dimension).
    /// Returns a SInt which can be either concrete (Const) or symbolic.
    ///
    /// # Examples
    ///
    /// ```
    /// # use svod_tensor::Tensor;
    /// let t = Tensor::from_slice(&[1.0f32; 6]).try_reshape(&[2, 3]).unwrap();
    /// assert_eq!(t.dim(0).unwrap().as_const(), Some(2)); // First dimension
    /// assert_eq!(t.dim(-1).unwrap().as_const(), Some(3)); // Last dimension
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if axis is out of range.
    pub fn dim(&self, axis: isize) -> Result<svod_ir::SInt> {
        let shape = self.shape()?;
        let idx = Self::normalize_axis(axis, shape.len())?;
        Ok(shape[idx].clone())
    }

    /// Get the size of a specific dimension, requiring it to be concrete.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::AxisOutOfRange`] if `axis` is out of range,
    /// [`ErrorKind::NonConstDim`] if the dimension is symbolic.
    pub fn dim_const(&self, axis: isize) -> Result<usize> {
        let dim = self.dim(axis)?;
        dim.as_const().ok_or(ErrorKind::NonConstDim { axis, dim }.into())
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

    /// Upper triangular mask: row + diagonal <= col.
    fn tri(rows: i64, cols: i64, diagonal: i64) -> Result<Tensor> {
        let row = Tensor::arange(0, Some(rows), None)?.try_unsqueeze(-1)?;
        let col = Tensor::arange(0, Some(cols), None)?;
        let diag = Tensor::const_(ConstValue::Int(diagonal), DType::Int32);
        row.try_add(&diag)?.try_le(&col)
    }

    /// Keep upper triangle, zero below.
    #[track_caller]
    pub fn triu(&self, diagonal: isize) -> Result<Tensor> {
        origin_call!("triu");
        let shape = self.shape()?;
        let ndim = shape.len();
        snafu::ensure!(ndim >= 2, NdimMinimumSnafu { op: "triu", min: 2usize, actual: ndim });
        let r = shape[ndim - 2].as_const().context(SymbolicShapeUnsupportedSnafu { operation: "triu" })? as i64;
        let c = shape[ndim - 1].as_const().context(SymbolicShapeUnsupportedSnafu { operation: "triu" })? as i64;
        let mask = Self::tri(r, c, diagonal as i64)?;
        let zero = Tensor::new(self.uop().const_like(ConstValue::zero(self.uop().dtype().scalar().unwrap())));
        self.where_(&mask, &zero)
    }

    /// Keep lower triangle, zero above.
    #[track_caller]
    pub fn tril(&self, diagonal: isize) -> Result<Tensor> {
        origin_call!("tril");
        let shape = self.shape()?;
        let ndim = shape.len();
        snafu::ensure!(ndim >= 2, NdimMinimumSnafu { op: "tril", min: 2usize, actual: ndim });
        let r = shape[ndim - 2].as_const().context(SymbolicShapeUnsupportedSnafu { operation: "tril" })? as i64;
        let c = shape[ndim - 1].as_const().context(SymbolicShapeUnsupportedSnafu { operation: "tril" })? as i64;
        let mask = Self::tri(r, c, diagonal as i64 + 1)?;
        let zero = Tensor::new(self.uop().const_like(ConstValue::zero(self.uop().dtype().scalar().unwrap())));
        zero.where_(&mask, self)
    }
}

#[bon]
impl Tensor {
    /// Slice tensor with Python-style indexing: negative indices, steps, and axis selection.
    ///
    /// `ends` is optional: an omitted setter, a `None` entry, or the ONNX
    /// `i64::MAX` sentinel all mean "to the end of that axis".
    #[builder]
    #[track_caller]
    pub fn slice_with(
        &self,
        starts: &[i64],
        #[builder(into)] ends: Option<SliceEnds<'_>>,
        axes: Option<&[i64]>,
        steps: Option<&[i64]>,
    ) -> Result<Tensor> {
        origin_call!("slice");
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

        let mut ranges: Vec<Option<(isize, isize)>> = vec![None; ndim];
        let mut flip_axes: Vec<isize> = Vec::new();

        for (i, &axis) in axes.iter().enumerate() {
            let step = steps[i];
            if step == 0 {
                return Err(crate::error::ErrorKind::IrConstruction { details: "Slice step cannot be 0".into() }.into());
            }

            let end = ends.and_then(|e| e.get(i));

            let d = match shape[axis].as_const() {
                Some(d) => d as i64,
                None => {
                    if starts[i] <= 0 && end.is_none() {
                        continue;
                    }
                    return Err(crate::error::ErrorKind::SymbolicShapeUnsupported {
                        operation: "slice_with on a symbolic dim".to_string(),
                    }
                    .into());
                }
            };

            let (lower, upper) = if step > 0 { (0i64, d) } else { (-1i64, d - 1) };
            let mut s = starts[i].clamp(-d, d);
            if s < 0 {
                s += d;
            }
            let s = s.clamp(lower, upper);

            // Missing end → the full extent (or one past index 0 when stepping down).
            let mut e = end.unwrap_or(if step > 0 { d } else { -d - 1 }).clamp(-d - 1, d);
            if e < 0 {
                e += d;
            }
            let e = e.clamp(lower, upper);

            if step * (e - s) < 0 {
                ranges[axis] = Some((0, 0));
            } else if step < 0 {
                flip_axes.push(axis as isize);
                ranges[axis] = Some(((e + 1) as isize, (s + 1) as isize));
            } else {
                ranges[axis] = Some((s as isize, e as isize));
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
            let size = cur[axis].as_const().ok_or_else(|| crate::error::ErrorKind::SymbolicShapeUnsupported {
                operation: "slice_with with step on a symbolic dim".to_string(),
            })?;
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
