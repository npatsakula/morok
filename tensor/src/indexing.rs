//! Indexing operations for Tensors.

use super::*;
use crate::error::ShapeMismatchSnafu;

impl Tensor {
    /// Gather values along an axis specified by `dim`, using `index` for element selection.
    #[track_caller]
    pub fn gather(&self, dim: isize, index: &Tensor) -> Result<Self> {
        let self_shape = self.shape()?;
        let index_shape = index.shape()?;
        let ndim = self_shape.len();
        let dim = Self::normalize_axis(dim, ndim)?;

        snafu::ensure!(
            index_shape.len() == ndim,
            ShapeMismatchSnafu {
                context: "gather",
                expected: format!("{ndim}D"),
                actual: format!("{}D index", index_shape.len())
            }
        );

        let self_dims = morok_ir::shape::to_vec_usize(&self_shape).context(UOpSnafu)?;
        let index_dims = morok_ir::shape::to_vec_usize(&index_shape).context(UOpSnafu)?;

        snafu::ensure!(
            self_dims.iter().zip(&index_dims).enumerate().all(|(d, (s, i))| d == dim || s >= i),
            ShapeMismatchSnafu {
                context: "gather",
                expected: "self[d] >= index[d] for d != dim".to_string(),
                actual: format!("self={self_dims:?}, index={index_dims:?}")
            }
        );

        let shrink: Vec<_> =
            (0..ndim).map(|d| (0, (if d == dim { self_dims[d] } else { index_dims[d] }) as isize)).collect();
        let x = self.try_shrink(&shrink)?.try_unsqueeze(-1)?.try_transpose(-1, dim as isize)?;

        let arange = Tensor::arange(0, Some(self_dims[dim] as i64), None)?.cast(index.uop().dtype())?;
        let mask = index.try_unsqueeze(-1)?.try_eq(&arange)?;

        x.where_(&mask, &Self::new(x.uop().const_like(0)))?.sum_with().axes(-1).dtype(self.uop().dtype()).call()
    }

    /// One-hot encoding: self == arange(num_classes) broadcast along dim.
    /// Returns a boolean tensor with True at the class positions.
    pub fn one_hot_along_dim(&self, num_classes: usize, dim: isize) -> Result<Tensor> {
        let ndim = self.ndim()?;
        let norm_dim = Self::normalize_axis(dim, ndim)?;
        let offset = ndim - norm_dim - 1;
        let arange = Tensor::arange(0, Some(num_classes as i64), None)?;
        let mut ar_shape = vec![1isize; 1 + offset];
        ar_shape[0] = num_classes as isize;
        self.try_eq(&arange.try_reshape(&ar_shape)?)
    }

    // =========================================================================
    // Scatter Operations (Tinygrad tensor.py:2641-2728)
    // =========================================================================

    /// Internal: prepare src and mask for scatter operations.
    ///
    /// Validates shapes, shrinks src to index.shape, then:
    ///  - src: unsqueeze(-1), expand(self.shape[dim]), transpose(-1, dim)
    ///  - mask: one_hot_along_dim(self.shape[dim]), transpose(-1, dim)
    ///
    /// Both are padded to self.shape on non-dim axes.
    fn _pre_scatter(&self, dim: isize, index: &Tensor, src: &Tensor) -> Result<(Tensor, Tensor)> {
        let self_shape = self.shape()?;
        let index_shape = index.shape()?;
        let src_shape = src.shape()?;
        let ndim = self_shape.len();
        let dim = Self::normalize_axis(dim, ndim)?;

        let self_dims = morok_ir::shape::to_vec_usize(&self_shape).context(UOpSnafu)?;
        let index_dims = morok_ir::shape::to_vec_usize(&index_shape).context(UOpSnafu)?;
        let src_dims = morok_ir::shape::to_vec_usize(&src_shape).context(UOpSnafu)?;

        snafu::ensure!(
            index_shape.len() == ndim && src_shape.len() == ndim,
            ShapeMismatchSnafu {
                context: "scatter",
                expected: format!("{ndim}D"),
                actual: format!("index={}D, src={}D", index_shape.len(), src_shape.len())
            }
        );
        snafu::ensure!(
            self_dims
                .iter()
                .zip(&index_dims)
                .zip(&src_dims)
                .enumerate()
                .all(|(d, ((s, i), sr))| { (d == dim || s >= i) && sr >= i }),
            ShapeMismatchSnafu {
                context: "scatter",
                expected: "valid scatter shape constraints".to_string(),
                actual: format!("self={self_dims:?}, index={index_dims:?}, src={src_dims:?}")
            }
        );

        // Shrink src to index shape
        let shrink_ranges: Vec<(isize, isize)> = index_dims.iter().map(|&d| (0, d as isize)).collect();
        let src = src.try_shrink(&shrink_ranges)?;

        // src: unsqueeze(-1) → expand(... self.shape[dim]) → transpose(-1, dim)
        let mut expand_shape: Vec<isize> = index_dims.iter().map(|&d| d as isize).collect();
        expand_shape.push(self_dims[dim] as isize);
        let src = src.try_unsqueeze(-1)?.try_expand(&expand_shape)?.try_transpose(-1, dim as isize)?;

        // mask: one_hot_along_dim(self.shape[dim]) → transpose(-1, dim)
        let mask = index.try_unsqueeze(-1)?.one_hot_along_dim(self_dims[dim], -1)?.try_transpose(-1, dim as isize)?;

        // Pad both to self.shape on non-dim axes
        let src_cur = src.shape()?;
        let src_cur_dims = morok_ir::shape::to_vec_usize(&src_cur).context(UOpSnafu)?;
        let padding: Vec<(isize, isize)> =
            (0..ndim).map(|d| (0, (self_dims[d] as isize - src_cur_dims[d] as isize).max(0))).collect();
        let needs_pad = padding.iter().any(|&(_, e)| e > 0);
        let src = if needs_pad { src.try_pad(&padding)? } else { src };
        let mask = if needs_pad { mask.try_pad(&padding)? } else { mask };

        Ok((src, mask))
    }

    /// Scatter values along dim using index positions.
    ///
    /// For each position in index, places the corresponding src value into self at
    /// the specified index along dim.
    #[track_caller]
    pub fn scatter(&self, dim: isize, index: &Tensor, src: &Tensor) -> Result<Tensor> {
        let (src_p, mask_p) = self._pre_scatter(dim, index, src)?;
        let zero = Tensor::new(src_p.uop().const_like(0));
        let gathered = src_p.where_(&mask_p, &zero)?.sum_with().axes(-1isize).call()?;
        let any_mask = mask_p.any(-1isize)?;
        gathered.where_(&any_mask, self)
    }

    /// Scatter with reduction. Applies reduce (sum/prod/amax/amin) at scatter positions.
    #[track_caller]
    pub fn scatter_reduce(
        &self,
        dim: isize,
        index: &Tensor,
        src: &Tensor,
        reduce: &str,
        include_self: bool,
    ) -> Result<Tensor> {
        let (src_p, mask_p) = self._pre_scatter(dim, index, src)?;
        let dtype = src_p.uop().dtype();
        let inv_mask = |a: &Tensor, b: &Tensor| -> Result<Tensor> {
            let no_hit = mask_p.any(-1isize)?.logical_not()?;
            a.where_(&no_hit, b)
        };
        let self_or = |identity_val: ConstValue| -> Result<Tensor> {
            if include_self { Ok(self.clone()) } else { inv_mask(self, &Tensor::const_(identity_val, dtype.clone())) }
        };

        match reduce {
            "sum" => {
                let zero = Tensor::const_(ConstValue::Int(0), dtype.clone());
                let reduced = src_p.where_(&mask_p, &zero)?.sum_with().axes(-1isize).call()?;
                reduced.try_add(&self_or(ConstValue::Int(0))?)
            }
            "prod" => {
                let one = Tensor::const_(ConstValue::Int(1), dtype.clone());
                let reduced = src_p.where_(&mask_p, &one)?.prod_with().axes(-1isize).call()?;
                reduced.try_mul(&self_or(ConstValue::Int(1))?)
            }
            "amax" => {
                let min_val =
                    if dtype.is_float() { ConstValue::Float(f64::NEG_INFINITY) } else { ConstValue::Int(i64::MIN) };
                let fill = Tensor::const_(min_val, dtype.clone());
                let reduced = src_p.where_(&mask_p, &fill)?.max(-1isize)?;
                reduced.maximum(&self_or(min_val)?)
            }
            "amin" => {
                let max_val =
                    if dtype.is_float() { ConstValue::Float(f64::INFINITY) } else { ConstValue::Int(i64::MAX) };
                let fill = Tensor::const_(max_val, dtype.clone());
                let reduced = src_p.where_(&mask_p, &fill)?.min(-1isize)?;
                reduced.minimum(&self_or(max_val)?)
            }
            _ => Err(crate::error::Error::IrConstruction {
                details: format!("scatter_reduce: unsupported reduce '{reduce}'"),
            }),
        }
    }

    // =========================================================================
    // Masked Select (Tinygrad tensor.py:1528-1547)
    // =========================================================================

    /// Select elements where mask is true, returning a flat tensor.
    ///
    /// Requires `realize()` internally (data-dependent output size).
    #[track_caller]
    pub fn masked_select(&self, mask: &Tensor) -> Result<Tensor> {
        let x = self.flatten()?;
        let mask_flat = mask.broadcast_to(&self.shape()?)?.flatten()?;
        let mask_cumsum = mask_flat.cast(morok_dtype::DType::Int32)?.cumsum(0)?;
        // Realize to get output size (data-dependent shape)
        let n = mask_flat.numel()?;
        let count_t = mask_cumsum.try_shrink(&[((n - 1) as isize, n as isize)])?.realize()?.to_ndarray::<i32>()?;
        let count = count_t[[0]] as usize;
        if count == 0 {
            return Ok(Tensor::empty(self.uop().dtype()));
        }
        // Build gather indices: zeros.scatter(0, cumsum, 1).cumsum
        let zeros = Tensor::full(&[count], ConstValue::Int(0), morok_dtype::DType::Int32)?;
        let ones = Tensor::full(&[n], ConstValue::Int(1), morok_dtype::DType::Int32)?;
        let idxs = zeros.scatter_reduce(0, &mask_cumsum, &ones, "sum", false)?.cumsum(0)?;
        x.gather(0, &idxs)
    }

    // =========================================================================
    // Sort (Bitonic) (Tinygrad tensor.py:2730-2779)
    // =========================================================================

    /// Bitonic sort along a dimension. Returns (sorted_values, indices).
    #[track_caller]
    pub fn sort(&self, dim: isize, descending: bool) -> Result<(Tensor, Tensor)> {
        let shape = self.shape()?;
        let ndim = shape.len();
        let dim = Self::normalize_axis(dim, ndim)?;
        let orig_len = shape[dim]
            .as_const()
            .ok_or_else(|| crate::error::Error::SymbolicShapeUnsupported { operation: "sort".into() })?;

        if orig_len <= 1 {
            let idx = Tensor::full(
                &morok_ir::shape::to_vec_usize(&shape).unwrap(),
                ConstValue::Int(0),
                morok_dtype::DType::Int32,
            )?;
            return Ok((self.clone(), idx));
        }

        let n_stages = (orig_len as u64 - 1).ilog2() as usize + 1;
        let padded_len = 1usize << n_stages;

        // Pad to power of 2
        let sentinel = if descending {
            if self.uop().dtype().is_float() { f64::NEG_INFINITY } else { i64::MIN as f64 }
        } else if self.uop().dtype().is_float() {
            f64::INFINITY
        } else {
            i64::MAX as f64
        };
        let mut padding = vec![(0isize, 0isize); ndim];
        padding[dim] = (0, (padded_len - orig_len) as isize);
        let mut x = self.try_pad_value(&padding, sentinel)?;

        // Unflatten dim into n_stages binary dimensions
        let unflatten_sizes: Vec<isize> = vec![2; n_stages];
        x = x.unflatten(dim as isize, &unflatten_sizes)?;

        // Bitonic sort network
        for stage in 1..=n_stages {
            if stage != n_stages {
                // Crossover: flip for green boxes
                let crossover_dim = (dim + n_stages - stage - 1) as isize;
                let halves = x.split(&[1, 1], crossover_dim)?;
                let (blue, green) = (&halves[0], &halves[1]);
                let flip_dims: Vec<isize> = (1..=(stage + (ndim - dim))).map(|i| -(i as isize)).collect();
                x = Tensor::cat(&[blue, &green.flip(&flip_dims)?], crossover_dim)?.contiguous();
            }

            for substage in (0..stage).rev() {
                let partner_dim = (dim + n_stages - substage - 1) as isize;
                let parts = x.split(&[1, 1], partner_dim)?;
                let (x_top, x_bottom) = (&parts[0], &parts[1]);
                let x_larger = x_top.maximum(x_bottom)?;
                let x_smaller = x_top.minimum(x_bottom)?;
                x = if descending {
                    Tensor::cat(&[&x_larger, &x_smaller], partner_dim)?
                } else {
                    Tensor::cat(&[&x_smaller, &x_larger], partner_dim)?
                }
                .contiguous();
            }

            if stage != n_stages {
                // Undo crossover
                let crossover_dim = (dim + n_stages - stage - 1) as isize;
                let halves = x.split(&[1, 1], crossover_dim)?;
                let (blue, flipped_green) = (&halves[0], &halves[1]);
                let flip_dims: Vec<isize> = (1..=(stage + (ndim - dim))).map(|i| -(i as isize)).collect();
                x = Tensor::cat(&[blue, &flipped_green.flip(&flip_dims)?], crossover_dim)?;
            }
        }

        // Flatten back and shrink to original size
        let flatten_end = dim + n_stages - 1;
        // Flatten dims [dim..dim+n_stages] back to one
        let cur_shape = x.shape()?;
        let cur_dims = morok_ir::shape::to_vec_usize(&cur_shape).context(UOpSnafu)?;
        let mut flat_shape: Vec<isize> = Vec::new();
        for (i, &d) in cur_dims.iter().enumerate() {
            if i == dim {
                flat_shape.push(padded_len as isize);
            } else if i > dim && i <= flatten_end {
                continue;
            } else {
                flat_shape.push(d as isize);
            }
        }
        x = x.try_reshape(&flat_shape)?;

        // Shrink to original size
        let x_shape = x.shape()?;
        let x_dims = morok_ir::shape::to_vec_usize(&x_shape).context(UOpSnafu)?;
        let shrink_ranges: Vec<(isize, isize)> =
            x_dims.iter().enumerate().map(|(d, &s)| (0, if d == dim { orig_len } else { s } as isize)).collect();
        x = x.try_shrink(&shrink_ranges)?;

        // Compute indices via count-matching (matches Tinygrad's approach)
        // Use lazy tril mask from arange comparisons — no host buffer allocation
        let mut tril_shape: Vec<usize> = vec![1; ndim + 1];
        tril_shape[dim] = orig_len;
        tril_shape[dim + 1] = orig_len;
        let tril_mask = Tensor::full(&tril_shape, true, morok_dtype::DType::Bool)?.tril(0)?;

        // Count occurrences of each value up to current position
        let compute_counts = |t: &Tensor| -> Result<Tensor> {
            let eq = t.try_unsqueeze(dim as isize)?.try_eq(&t.try_unsqueeze((dim + 1) as isize)?)?;
            eq.bitwise_and(&tril_mask)?.sum((dim + 1) as isize)
        };

        let count_orig = compute_counts(self)?;
        let count_sorted = compute_counts(&x)?;

        // Match: original[unsqueeze(dim+1)] == sorted[unsqueeze(dim)] && counts match
        let val_match = self.try_unsqueeze((dim + 1) as isize)?.try_eq(&x.try_unsqueeze(dim as isize)?)?;
        let cnt_match =
            count_orig.try_unsqueeze((dim + 1) as isize)?.try_eq(&count_sorted.try_unsqueeze(dim as isize)?)?;
        let cond = val_match.bitwise_and(&cnt_match)?;

        // Build index arange and compute weighted sum
        let mut idx_shape = vec![1isize; ndim + 1];
        idx_shape[dim] = orig_len as isize;
        let idx = (cond
            .cast(morok_dtype::DType::Int32)?
            .try_mul(&Tensor::arange(0, Some(orig_len as i64), None)?.try_reshape(&idx_shape)?)?)
        .sum(dim as isize)?;

        Ok((x, idx))
    }

    // =========================================================================
    // TopK (Tinygrad tensor.py:2792-2812)
    // =========================================================================

    /// Top-k elements along a dimension. Returns (values, indices).
    #[track_caller]
    pub fn topk(&self, k: usize, dim: isize, largest: bool) -> Result<(Tensor, Tensor)> {
        let shape = self.shape()?;
        let ndim = shape.len();
        let norm_dim = Self::normalize_axis(dim, ndim)?;
        let (x, idx) = self.sort(dim, largest)?;
        // Shrink to first k along dim
        let x_shape = x.shape()?;
        let x_dims = morok_ir::shape::to_vec_usize(&x_shape).context(UOpSnafu)?;
        let shrink: Vec<(isize, isize)> =
            x_dims.iter().enumerate().map(|(d, &s)| (0, if d == norm_dim { k } else { s } as isize)).collect();
        Ok((x.try_shrink(&shrink)?, idx.try_shrink(&shrink)?))
    }

    // =========================================================================
    // NonZero (Tinygrad tensor.py:1549-1573)
    // =========================================================================

    /// Indices of non-zero elements. Returns [num_nonzero, ndim] tensor.
    #[track_caller]
    pub fn nonzero(&self) -> Result<Tensor> {
        let shape = self.shape()?;
        let ndim = shape.len();
        let dims = morok_ir::shape::to_vec_usize(&shape).context(UOpSnafu)?;
        let numel: usize = dims.iter().product();

        let mask = self.try_ne(&Tensor::const_(ConstValue::Int(0), self.uop().dtype()))?.flatten()?;

        // Build coordinate tensor: for each dim, arange → reshape to broadcast → flatten
        let coords: Vec<Tensor> = (0..ndim)
            .map(|i| {
                let ar = Tensor::arange(0, Some(dims[i] as i64), None)?;
                let mut rshape = vec![1isize; ndim];
                rshape[i] = dims[i] as isize;
                let expand_shape: Vec<isize> = dims.iter().map(|&d| d as isize).collect();
                ar.try_reshape(&rshape)?.try_expand(&expand_shape)?.flatten()
            })
            .collect::<Result<Vec<_>>>()?;

        let coords_refs: Vec<&Tensor> = coords.iter().collect();
        let indices = Tensor::stack(&coords_refs, -1)?; // [numel, ndim]

        // Select nonzero coordinates
        let expanded_mask = mask.try_unsqueeze(-1)?.try_expand(&[numel as isize, ndim as isize])?;
        let selected = indices.masked_select(&expanded_mask)?;
        selected.try_reshape(&[-1, ndim as isize])
    }
}
