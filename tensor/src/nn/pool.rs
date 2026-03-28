//! Sliding-window pooling: pool, avg_pool2d, max_pool2d, max_pool2d_with_indices.

use bon::bon;
use morok_dtype::DType;
use morok_ir::{ConstValue, SInt, UOp};

use crate::Tensor;
use crate::error::DivisibilitySnafu;
use crate::reduce::AxisSpec;

use super::pad::apply_ceil_mode;

type Result<T> = crate::Result<T>;

impl Tensor {
    /// Sliding window extraction via shape manipulation (Tinygrad's `_pool`).
    ///
    /// Input: `(..., *spatial)` &rarr; Output: `(..., *out_spatial, *kernel)`.
    ///
    /// This is a low-level building block for pooling and convolution. It extracts
    /// all sliding windows of the given kernel size, stride, and dilation from the
    /// spatial dimensions, appending the kernel dimensions at the end.
    pub fn pool(&self, kernel: &[usize], stride: &[usize], dilation: &[usize]) -> Result<Tensor> {
        let shape = self.shape()?;
        let ndim = shape.len();
        let n_spatial = kernel.len();
        let n_batch = ndim - n_spatial;

        if ndim < n_spatial {
            return Err(crate::error::Error::IrConstruction {
                details: format!("can't pool {ndim}D with {n_spatial}D kernel"),
            });
        }
        if kernel.len() != stride.len() {
            return Err(crate::error::Error::IrConstruction {
                details: format!("kernel/stride length mismatch: {} vs {}", kernel.len(), stride.len()),
            });
        }
        if kernel.len() != dilation.len() {
            return Err(crate::error::Error::IrConstruction {
                details: format!("kernel/dilation length mismatch: {} vs {}", kernel.len(), dilation.len()),
            });
        }

        // Spatial dims as SInt — works for both concrete and symbolic.
        let i_: Vec<SInt> = (0..n_spatial).map(|j| shape[n_batch + j].clone()).collect();

        // Validate: kernel must fit in input (concrete dims only — symbolic skips check).
        for j in 0..n_spatial {
            if let Some(i) = i_[j].as_const()
                && dilation[j] * (kernel[j] - 1) >= i
            {
                return Err(crate::error::Error::IrConstruction {
                    details: format!(
                        "kernel size {} (dilated {}) > input size {}",
                        kernel[j],
                        dilation[j] * (kernel[j] - 1) + 1,
                        i
                    ),
                });
            }
        }

        // Pool formulas — SInt arithmetic: concrete folds inline, symbolic creates UOp graph.
        // o_[j] = ceildiv(i_[j] - dilation[j] * (kernel[j] - 1), stride[j])
        let o_: Vec<SInt> =
            (0..n_spatial).map(|j| (&i_[j] - dilation[j] * (kernel[j] - 1)).ceildiv(&SInt::from(stride[j]))).collect();

        // f_[j] = max(1, ceildiv(o_[j] * stride[j] - dilation[j], i_[j]))
        let f_: Vec<SInt> = (0..n_spatial)
            .map(|j| SInt::from(1usize).smax(&(&o_[j] * stride[j] - dilation[j]).ceildiv(&i_[j])))
            .collect();

        // Batch dims: None in shrink (identity), SInt in reshape.
        let noop: Vec<Option<(SInt, SInt)>> = vec![None; n_batch];
        let batch_sint: Vec<SInt> = shape.iter().take(n_batch).cloned().collect();

        // Step 1: repeat
        // repeat_count = ceildiv(k * (i*f + d), i)
        let mut repeats: Vec<SInt> = vec![SInt::from(1usize); n_batch];
        for j in 0..n_spatial {
            repeats.push((kernel[j] * (&i_[j] * &f_[j] + dilation[j])).ceildiv(&i_[j]));
        }
        let mut x = self.repeat(&repeats)?;

        // Step 2: shrink to exact needed size
        let mut shrink: Vec<Option<(SInt, SInt)>> = noop.clone();
        for j in 0..n_spatial {
            shrink.push(Some((SInt::from(0usize), kernel[j] * (&i_[j] * &f_[j] + dilation[j]))));
        }
        x = x.try_shrink(shrink)?;

        // Step 3: reshape to interleave kernel and spatial dims
        let mut reshape_dims: Vec<SInt> = batch_sint.clone();
        for j in 0..n_spatial {
            reshape_dims.push(kernel[j].into());
            reshape_dims.push(&i_[j] * &f_[j] + dilation[j]);
        }
        x = x.try_reshape(reshape_dims)?;

        // Step 4: shrink for stride
        let mut shrink: Vec<Option<(SInt, SInt)>> = noop.clone();
        for j in 0..n_spatial {
            shrink.push(Some((SInt::from(0usize), SInt::from(kernel[j]))));
            shrink.push(Some((SInt::from(0usize), &o_[j] * stride[j])));
        }
        x = x.try_shrink(shrink)?;

        // Step 5: reshape to separate stride: K_j, o_j, S_j
        let mut reshape_dims: Vec<SInt> = batch_sint.clone();
        for j in 0..n_spatial {
            reshape_dims.push(kernel[j].into());
            reshape_dims.push(o_[j].clone());
            reshape_dims.push(stride[j].into());
        }
        x = x.try_reshape(reshape_dims)?;

        // Step 6: shrink stride dim to 1
        let mut shrink: Vec<Option<(SInt, SInt)>> = noop.clone();
        for j in 0..n_spatial {
            shrink.push(Some((SInt::from(0usize), SInt::from(kernel[j]))));
            shrink.push(Some((SInt::from(0usize), o_[j].clone())));
            shrink.push(Some((SInt::from(0usize), SInt::from(1usize))));
        }
        x = x.try_shrink(shrink)?;

        // Step 7: reshape to collapse stride dim
        let mut reshape_dims: Vec<SInt> = batch_sint.clone();
        for j in 0..n_spatial {
            reshape_dims.push(kernel[j].into());
            reshape_dims.push(o_[j].clone());
        }
        x = x.try_reshape(reshape_dims)?;

        // Step 8: permute to move kernel dims to end
        let mut perm: Vec<isize> = (0..n_batch as isize).collect();
        for j in 0..n_spatial {
            perm.push(n_batch as isize + j as isize * 2 + 1); // output spatial
        }
        for j in 0..n_spatial {
            perm.push(n_batch as isize + j as isize * 2); // kernel
        }
        x = x.try_permute(&perm)?;

        Ok(x)
    }
}

#[bon]
impl Tensor {
    /// Average pooling over spatial dimensions.
    ///
    /// Computes the mean of each sliding window. Supports padding, dilation,
    /// `count_include_pad` (whether padded zeros count in the denominator),
    /// and `ceil_mode` (round output size up instead of down).
    ///
    /// Stride defaults to `kernel_size` when not specified.
    ///
    /// # Examples
    ///
    /// Basic 2x2 average pooling:
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// # use ndarray::Array4;
    /// let x = Tensor::from_ndarray(&Array4::from_elem((1, 1, 4, 4), 1.0f32));
    /// let y = x.avg_pool2d().kernel_size(&[2, 2]).call().unwrap();
    /// let shape: Vec<_> = y.shape().unwrap().iter().map(|d| d.as_const().unwrap()).collect();
    /// assert_eq!(shape, vec![1, 1, 2, 2]);
    /// assert_eq!(y.to_vec::<f32>().unwrap(), vec![1.0; 4]);
    /// ```
    ///
    /// With explicit stride:
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// # use ndarray::Array4;
    /// let x = Tensor::from_ndarray(&Array4::from_elem((1, 1, 4, 4), 1.0f32));
    /// let y = x.avg_pool2d().kernel_size(&[2, 2]).stride(&[1, 1]).call().unwrap();
    /// let shape: Vec<_> = y.shape().unwrap().iter().map(|d| d.as_const().unwrap()).collect();
    /// assert_eq!(shape, vec![1, 1, 3, 3]);
    /// ```
    ///
    /// With padding and `count_include_pad` disabled:
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// # use ndarray::Array4;
    /// let x = Tensor::from_ndarray(&Array4::from_elem((1, 1, 2, 2), 1.0f32));
    /// let y = x.avg_pool2d()
    ///     .kernel_size(&[2, 2])
    ///     .stride(&[1, 1])
    ///     .padding(&[(1, 1), (1, 1)])
    ///     .count_include_pad(false)
    ///     .call()
    ///     .unwrap();
    /// // With count_include_pad=false, only non-padded elements count in the average
    /// assert_eq!(y.to_vec::<f32>().unwrap(), vec![1.0; 9]);
    /// ```
    #[builder]
    pub fn avg_pool2d(
        &self,
        kernel_size: &[usize],
        stride: Option<&[usize]>,
        dilation: Option<&[usize]>,
        padding: Option<&[(isize, isize)]>,
        #[builder(default = true)] count_include_pad: bool,
        #[builder(default = false)] ceil_mode: bool,
    ) -> Result<Tensor> {
        let n_spatial = kernel_size.len();
        let default_dilation: Vec<usize> = vec![1; n_spatial];
        let stride = stride.unwrap_or(kernel_size);
        let dilation = dilation.unwrap_or(&default_dilation);
        let no_pad: Vec<(isize, isize)> = vec![(0, 0); n_spatial];
        let padding = padding.unwrap_or(&no_pad);

        let reduce_axes: Vec<isize> = (0..n_spatial).map(|j| -(1 + j as isize)).collect();
        let axes = AxisSpec::Multiple(reduce_axes);

        let shape = self.shape()?;
        let n_batch = shape.len() - n_spatial;
        let input_spatial: Vec<SInt> = shape[n_batch..].to_vec();

        let reg_pads = padding.to_vec();
        let ceil_pads = if ceil_mode {
            apply_ceil_mode(&reg_pads, &input_spatial, kernel_size, stride, dilation)
        } else {
            reg_pads.clone()
        };

        let pad_and_pool = |x: &Tensor, pads: &[(isize, isize)]| -> Result<Tensor> {
            let mut out = x.clone();
            if pads.iter().any(|&(b, e)| b != 0 || e != 0) {
                let mut full_pad: Vec<(isize, isize)> = vec![(0, 0); n_batch];
                full_pad.extend_from_slice(pads);
                out = out.try_pad(&full_pad)?;
            }
            out.pool(kernel_size, stride, dilation)
        };

        if !count_include_pad {
            // Path 1: sum(pool(x, pads)) / sum(pool(ones, pads))
            let pads = if ceil_mode { &ceil_pads } else { &reg_pads };
            let pooled = pad_and_pool(self, pads)?;
            let sum_x = pooled.sum_with().axes(axes.clone()).keepdim(false).call()?;
            // Use input dtype for ones tensor (not hardcoded Float32)
            let dtype = self.uop().dtype();
            let ones = Tensor::new(UOp::const_(dtype, ConstValue::Float(1.0)));
            let ones = ones.broadcast_to(&self.shape()?)?;
            let pooled_ones = pad_and_pool(&ones, pads)?;
            let sum_ones = pooled_ones.sum_with().axes(axes).keepdim(false).call()?;
            return sum_x.try_div(&sum_ones);
        }

        if !ceil_mode {
            // Path 2: count_include_pad=true, ceil_mode=false → simple mean
            let pooled = pad_and_pool(self, &reg_pads)?;
            return pooled.mean(axes);
        }

        // Path 3: count_include_pad=true, ceil_mode=true
        // Regular padding counts in the average, but ceil-extra padding does NOT.
        // Tinygrad: pool(x, ceil_pads).sum / pool(pad(x, reg_pads).ones_like(), ceil-reg).sum
        let pooled = pad_and_pool(self, &ceil_pads)?;
        let sum_x = pooled.sum_with().axes(axes.clone()).keepdim(false).call()?;

        // ones_like of the regularly-padded input (all positions are 1, including reg pads),
        // then pool with only the extra ceil pads (which add zeros that don't count).
        let mut padded_self = self.clone();
        if reg_pads.iter().any(|&(b, e)| b != 0 || e != 0) {
            let mut full_pad: Vec<(isize, isize)> = vec![(0, 0); n_batch];
            full_pad.extend_from_slice(&reg_pads);
            padded_self = padded_self.try_pad(&full_pad)?;
        }
        let ones_reg = padded_self.one()?;
        let extra_pads: Vec<(isize, isize)> =
            ceil_pads.iter().zip(reg_pads.iter()).map(|(c, r)| (c.0 - r.0, c.1 - r.1)).collect();
        let pooled_ones = pad_and_pool(&ones_reg, &extra_pads)?;
        let sum_ones = pooled_ones.sum_with().axes(axes).keepdim(false).call()?;
        sum_x.try_div(&sum_ones)
    }

    /// Max pooling over spatial dimensions.
    ///
    /// Returns the maximum value in each sliding window. Padded positions are
    /// filled with `-inf` (float) or `i64::MIN` (integer) so they never win.
    ///
    /// Stride defaults to `kernel_size` when not specified.
    ///
    /// # Examples
    ///
    /// Basic 2x2 max pooling:
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// # use ndarray::Array4;
    /// let x = Tensor::from_ndarray(&Array4::from_elem((1, 1, 4, 4), 1.0f32));
    /// let y = x.max_pool2d().kernel_size(&[2, 2]).call().unwrap();
    /// let shape: Vec<_> = y.shape().unwrap().iter().map(|d| d.as_const().unwrap()).collect();
    /// assert_eq!(shape, vec![1, 1, 2, 2]);
    /// assert_eq!(y.to_vec::<f32>().unwrap(), vec![1.0; 4]);
    /// ```
    ///
    /// With stride and padding:
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// # use ndarray::Array4;
    /// let x = Tensor::from_ndarray(&Array4::from_elem((1, 1, 4, 4), 1.0f32));
    /// let y = x.max_pool2d()
    ///     .kernel_size(&[3, 3])
    ///     .stride(&[1, 1])
    ///     .padding(&[(1, 1), (1, 1)])
    ///     .call()
    ///     .unwrap();
    /// let shape: Vec<_> = y.shape().unwrap().iter().map(|d| d.as_const().unwrap()).collect();
    /// assert_eq!(shape, vec![1, 1, 4, 4]);
    /// assert_eq!(y.to_vec::<f32>().unwrap(), vec![1.0; 16]);
    /// ```
    #[builder]
    pub fn max_pool2d(
        &self,
        kernel_size: &[usize],
        stride: Option<&[usize]>,
        dilation: Option<&[usize]>,
        padding: Option<&[(isize, isize)]>,
        #[builder(default = false)] ceil_mode: bool,
    ) -> Result<Tensor> {
        let n_spatial = kernel_size.len();
        let default_dilation: Vec<usize> = vec![1; n_spatial];
        let stride = stride.unwrap_or(kernel_size);
        let dilation = dilation.unwrap_or(&default_dilation);
        let no_pad: Vec<(isize, isize)> = vec![(0, 0); n_spatial];
        let padding = padding.unwrap_or(&no_pad);

        let pads = if ceil_mode {
            let shape = self.shape()?;
            let n_batch = shape.len() - n_spatial;
            let input_spatial: Vec<SInt> = shape[n_batch..].to_vec();
            apply_ceil_mode(padding, &input_spatial, kernel_size, stride, dilation)
        } else {
            padding.to_vec()
        };

        let reduce_axes: Vec<isize> = (0..n_spatial).map(|j| -(1 + j as isize)).collect();
        let axes = AxisSpec::Multiple(reduce_axes);

        let mut x = self.clone();
        if pads.iter().any(|&(b, e)| b != 0 || e != 0) {
            let mut full_pad: Vec<(isize, isize)> = vec![(0, 0); self.ndim()? - n_spatial];
            full_pad.extend_from_slice(&pads);
            let fill = if self.uop().dtype().is_float() { f64::NEG_INFINITY } else { i64::MIN as f64 };
            x = x.try_pad_value(&full_pad, fill)?;
        }

        let pooled = x.pool(kernel_size, stride, dilation)?;
        pooled.max(axes)
    }

    /// Max pooling returning both values and flat indices.
    ///
    /// Returns `(values, indices)` where indices are flat offsets into the
    /// input spatial dimensions. Indices can be passed to
    /// [`max_unpool2d`](Tensor::max_unpool2d) to invert the operation.
    ///
    /// Uses a reverse-arange trick (from Tinygrad) to compute first-occurrence
    /// indices without explicit argmax.
    ///
    /// # Examples
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// # use ndarray::Array4;
    /// let x = Tensor::from_ndarray(&Array4::from_elem((1, 1, 4, 4), 1.0f32));
    /// let (values, indices) = x.max_pool2d_with_indices()
    ///     .kernel_size(&[2, 2])
    ///     .call()
    ///     .unwrap();
    /// let shape: Vec<_> = values.shape().unwrap().iter().map(|d| d.as_const().unwrap()).collect();
    /// assert_eq!(shape, vec![1, 1, 2, 2]);
    /// assert_eq!(values.to_vec::<f32>().unwrap(), vec![1.0; 4]);
    /// ```
    #[builder]
    pub fn max_pool2d_with_indices(
        &self,
        kernel_size: &[usize],
        stride: Option<&[usize]>,
        dilation: Option<&[usize]>,
        padding: Option<&[(isize, isize)]>,
        #[builder(default = false)] ceil_mode: bool,
    ) -> Result<(Tensor, Tensor)> {
        let n_spatial = kernel_size.len();
        let default_dilation: Vec<usize> = vec![1; n_spatial];
        let stride = stride.unwrap_or(kernel_size);
        let dilation = dilation.unwrap_or(&default_dilation);
        let no_pad: Vec<(isize, isize)> = vec![(0, 0); n_spatial];
        let padding = padding.unwrap_or(&no_pad);

        let shape = self.shape()?;
        let n_batch = shape.len() - n_spatial;

        let pads = if ceil_mode {
            let input_spatial: Vec<SInt> = shape[n_batch..].to_vec();
            apply_ceil_mode(padding, &input_spatial, kernel_size, stride, dilation)
        } else {
            padding.to_vec()
        };

        let reduce_axes: Vec<isize> = (0..n_spatial).map(|j| -(1 + j as isize)).collect();
        let axes = AxisSpec::Multiple(reduce_axes.clone());

        // Pool the data with dtype-minimum padding
        let mut x = self.clone();
        if pads.iter().any(|&(b, e)| b != 0 || e != 0) {
            let mut full_pad: Vec<(isize, isize)> = vec![(0, 0); n_batch];
            full_pad.extend_from_slice(&pads);
            let fill = if self.uop().dtype().is_float() { f64::NEG_INFINITY } else { i64::MIN as f64 };
            x = x.try_pad_value(&full_pad, fill)?;
        }
        let pooled = x.pool(kernel_size, stride, dilation)?;
        let values = pooled.max_with().axes(axes.clone()).keepdim(false).call()?;

        // Compute indices using reverse arange trick (Tinygrad approach)
        let spatial_sz: usize = (0..n_spatial).map(|j| shape[n_batch + j].as_const().unwrap()).product();

        // Create reverse arange: spatial_sz, spatial_sz-1, ..., 1
        let idx_range = Tensor::arange(spatial_sz as i64, Some(0), Some(-1))?;
        // Reshape to match spatial dims
        let spatial_dims: Vec<isize> =
            (0..n_spatial).map(|j| shape[n_batch + j].as_const().unwrap() as isize).collect();
        let mut idx_shape: Vec<isize> = vec![1; n_batch];
        idx_shape.extend_from_slice(&spatial_dims);
        let idx = idx_range.try_reshape(&idx_shape)?;

        // Pad and pool the index tensor identically
        let mut idx_padded = idx;
        if pads.iter().any(|&(b, e)| b != 0 || e != 0) {
            let mut full_pad: Vec<(isize, isize)> = vec![(0, 0); n_batch];
            full_pad.extend_from_slice(&pads);
            idx_padded = idx_padded.try_pad(&full_pad)?;
        }
        let pooled_idx = idx_padded.pool(kernel_size, stride, dilation)?;

        // Create mask: pooled == pooled.max(keepdim=True)
        let pooled_max = pooled.max_with().axes(axes.clone()).keepdim(true).call()?;
        let mask = pooled.try_eq(&pooled_max)?;

        // Multiply mask * pooled_indices, take max → first-occurrence (via reverse index)
        let masked_idx = mask.cast(DType::Int32)?.try_mul(&pooled_idx)?;
        let max_idx = masked_idx.max_with().axes(axes).keepdim(false).call()?;

        // spatial_sz - max_idx → convert reverse index to forward index
        let sz_t = Tensor::const_(ConstValue::Int(spatial_sz as i64), DType::Int32);
        let indices = sz_t.try_sub(&max_idx)?;

        Ok((values, indices))
    }

    /// Inverse of max pooling: scatter pooled values back to their original positions.
    ///
    /// Indices are flat offsets into the *inferred* output spatial shape (computed
    /// from kernel/stride/padding). When `output_size` exceeds the inferred shape,
    /// the result is zero-padded to match.
    ///
    /// Uses one-hot encoding of indices to scatter values: `one_hot(idx) * vals -> sum`.
    ///
    /// # Examples
    ///
    /// Round-trip with max_pool2d_with_indices:
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// # use ndarray::Array4;
    /// let x = Tensor::from_ndarray(&Array4::from_elem((1, 1, 4, 4), 1.0f32));
    /// let (values, indices) = x.max_pool2d_with_indices()
    ///     .kernel_size(&[2, 2])
    ///     .call()
    ///     .unwrap();
    /// let unpooled = values.max_unpool2d()
    ///     .indices(&indices)
    ///     .kernel_size(&[2, 2])
    ///     .call()
    ///     .unwrap();
    /// let shape: Vec<_> = unpooled.shape().unwrap().iter()
    ///     .map(|d| d.as_const().unwrap()).collect();
    /// assert_eq!(shape, vec![1, 1, 4, 4]);
    /// ```
    #[builder]
    pub fn max_unpool2d(
        &self,
        indices: &Tensor,
        kernel_size: &[usize],
        stride: Option<&[usize]>,
        padding: Option<&[(isize, isize)]>,
        output_size: Option<&[usize]>,
    ) -> Result<Tensor> {
        let shape = self.shape()?;
        let ndim = shape.len();
        let n_spatial = kernel_size.len();
        let n_batch = ndim - n_spatial;

        let spatial_shape: Vec<usize> = (0..n_spatial).map(|j| shape[n_batch + j].as_const().unwrap()).collect();

        // Inferred shape from inverse pooling formula: o = (i-1)*s - (pB+pA) + k
        let stride = stride.unwrap_or(kernel_size);
        let no_pad: Vec<(isize, isize)> = vec![(0, 0); n_spatial];
        let padding = padding.unwrap_or(&no_pad);
        let inferred_spatial: Vec<usize> = (0..n_spatial)
            .map(|j| {
                let (pa, pb) = padding[j];
                (spatial_shape[j] - 1) * stride[j] - (pa as usize + pb as usize) + kernel_size[j]
            })
            .collect();

        let inferred_numel: usize = inferred_spatial.iter().product();
        let bs: usize = (0..n_batch).map(|j| shape[j].as_const().unwrap()).product();

        // Flatten: (N, C, *spatial) → (N*C, 1, num_pooled)
        let num_pooled: usize = spatial_shape.iter().product();
        let vals_flat = self.try_reshape([bs as isize, 1, num_pooled as isize])?;
        let idx_flat = indices.try_reshape([bs as isize, 1, num_pooled as isize])?;

        // One-hot: compare indices against arange(inferred_numel)
        let arange = Tensor::arange(inferred_numel as i64, None, None)?.cast(indices.uop().dtype())?.try_reshape([
            1,
            inferred_numel as isize,
            1,
        ])?;
        let one_hot = idx_flat.try_eq(&arange)?;

        // Place values at one-hot positions, zero elsewhere, then sum over pooled dim
        let zero = Tensor::const_(0.0f64, self.uop().dtype());
        let placed = vals_flat.where_(&one_hot, &zero)?;
        let result = placed.sum(-1isize)?;

        // Reshape to (N, C, *inferred_spatial)
        let batch_dims: Vec<isize> = (0..n_batch).map(|j| shape[j].as_const().unwrap() as isize).collect();
        let mut inferred_shape: Vec<isize> = batch_dims.clone();
        inferred_shape.extend(inferred_spatial.iter().map(|&s| s as isize));
        let result = result.try_reshape(&inferred_shape)?;

        // If output_size is larger, zero-pad to match
        if let Some(os) = output_size {
            let out_spatial = &os[os.len() - n_spatial..];
            if out_spatial != inferred_spatial.as_slice() {
                let mut pad_spec: Vec<(isize, isize)> = vec![(0, 0); n_batch];
                for j in 0..n_spatial {
                    pad_spec.push((0, (out_spatial[j] - inferred_spatial[j]) as isize));
                }
                return result.try_pad(&pad_spec);
            }
        }
        Ok(result)
    }

    /// Col2Im: adjoint of im2col. Reconstructs an image from columns, summing overlaps.
    ///
    /// Input shape: `[N, C * prod(block_shape), L]` where `L` is the number of sliding positions.
    /// Output shape: `[N, C, *image_shape]`.
    ///
    /// Uses the adjoint of [`pool`](Tensor::pool): for each kernel position, stride-dilate
    /// the column data, pad to the correct offset, and accumulate. `O(output_size)` memory,
    /// `O(bl * output_size)` compute -- no large one-hot intermediates.
    ///
    /// # Examples
    ///
    /// Reconstruct a 4x4 image from 2x2 blocks with no overlap:
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// # use ndarray::Array3;
    /// // 1 batch, 1 channel, 2x2 block = 4 cols, 4 sliding positions
    /// let cols = Tensor::from_ndarray(&Array3::from_elem((1, 4, 4), 1.0f32));
    /// let img = cols.col2im()
    ///     .image_shape(&[4, 4])
    ///     .block_shape(&[2, 2])
    ///     .strides(&[2, 2])
    ///     .call()
    ///     .unwrap();
    /// let shape: Vec<_> = img.shape().unwrap().iter()
    ///     .map(|d| d.as_const().unwrap()).collect();
    /// assert_eq!(shape, vec![1, 1, 4, 4]);
    /// // Non-overlapping blocks of ones reconstruct to all ones
    /// assert_eq!(img.to_vec::<f32>().unwrap(), vec![1.0; 16]);
    /// ```
    #[builder]
    pub fn col2im(
        &self,
        image_shape: &[usize],
        block_shape: &[usize],
        strides: Option<&[usize]>,
        pads: Option<&[(isize, isize)]>,
        dilations: Option<&[usize]>,
    ) -> Result<Tensor> {
        let n_spatial = image_shape.len();
        let no_strides: Vec<usize> = vec![1; n_spatial];
        let no_pads: Vec<(isize, isize)> = vec![(0, 0); n_spatial];
        let no_dilations: Vec<usize> = vec![1; n_spatial];
        let strides = strides.unwrap_or(&no_strides);
        let pads = pads.unwrap_or(&no_pads);
        let dilations = dilations.unwrap_or(&no_dilations);

        let shape = self.shape()?;
        let n = shape[0].as_const().unwrap();
        let c_times_bl: usize = shape[1].as_const().unwrap();
        let bl: usize = block_shape.iter().product();
        snafu::ensure!(
            c_times_bl.is_multiple_of(bl),
            DivisibilitySnafu {
                op: "col2im",
                lhs_name: "C*block_size",
                lhs: c_times_bl,
                rhs_name: "block_size",
                rhs: bl
            }
        );
        let c = c_times_bl / bl;

        // Padded image shape (reconstruct in padded space, shrink at end)
        let padded_img: Vec<usize> =
            (0..n_spatial).map(|i| (image_shape[i] as isize + pads[i].0 + pads[i].1) as usize).collect();

        // Number of sliding positions per spatial dimension
        let l_spatial: Vec<usize> = (0..n_spatial)
            .map(|i| {
                let effective_k = dilations[i] * (block_shape[i] - 1) + 1;
                (padded_img[i] - effective_k) / strides[i] + 1
            })
            .collect();

        // Reshape input: [N, C*bl, L] → [N*C, *block_shape, *L_spatial]
        let nc = n * c;
        let mut data_shape: Vec<isize> = vec![nc as isize];
        data_shape.extend(block_shape.iter().map(|&s| s as isize));
        data_shape.extend(l_spatial.iter().map(|&s| s as isize));
        let data = self.try_reshape(&data_shape)?;

        // Initialize output: [N*C, *padded_img] with zeros
        let mut out_dims: Vec<usize> = vec![nc];
        out_dims.extend_from_slice(&padded_img);
        let mut result = Tensor::full(&out_dims, 0.0f64, self.uop().dtype())?;

        // Iterate over all kernel positions in block_shape
        for be in 0..bl {
            // Unravel be → (k0, k1, ..., k_{n-1})
            let mut kpos = vec![0usize; n_spatial];
            let mut rem = be;
            for i in (0..n_spatial).rev() {
                kpos[i] = rem % block_shape[i];
                rem /= block_shape[i];
            }

            // Extract slice for this kernel position: [N*C, *L_spatial]
            // Shrink block dims (dims 1..n_spatial) to singletons, keep L_spatial dims
            let mut shrink_ranges: Vec<(isize, isize)> = vec![(0, nc as isize)];
            for &k in kpos.iter().take(n_spatial) {
                shrink_ranges.push((k as isize, k as isize + 1));
            }
            for &l in l_spatial.iter().take(n_spatial) {
                shrink_ranges.push((0, l as isize));
            }
            let slice = data.try_shrink(&shrink_ranges)?;
            // Squeeze block dims → [N*C, *L_spatial]
            let mut sq_shape: Vec<isize> = vec![nc as isize];
            sq_shape.extend(l_spatial.iter().map(|&s| s as isize));
            let mut slice = slice.try_reshape(&sq_shape)?;

            // For each spatial dim: stride-dilate L_j, then pad to position
            for j in 0..n_spatial {
                let dim = 1 + j;
                let l_j = l_spatial[j];

                // Stride dilation: insert stride-1 zeros between elements
                if strides[j] > 1 {
                    let s = strides[j];
                    let ndim = slice.shape()?.len();
                    // [... L_j ...] → [... L_j, 1 ...] → pad → [... L_j, S ...] → [... L_j*S ...] → shrink
                    let mut sh: Vec<isize> = slice.shape()?.iter().map(|d| d.as_const().unwrap() as isize).collect();
                    sh.insert(dim + 1, 1);
                    slice = slice.try_reshape(&sh)?;

                    let mut pad_spec: Vec<(isize, isize)> = vec![(0, 0); ndim + 1];
                    pad_spec[dim + 1] = (0, (s - 1) as isize);
                    slice = slice.try_pad(&pad_spec)?;

                    sh[dim] = (l_j * s) as isize;
                    sh.remove(dim + 1);
                    slice = slice.try_reshape(&sh)?;

                    let dilated_l = (l_j - 1) * s + 1;
                    let mut sr: Vec<(isize, isize)> =
                        slice.shape()?.iter().map(|d| (0, d.as_const().unwrap() as isize)).collect();
                    sr[dim] = (0, dilated_l as isize);
                    slice = slice.try_shrink(&sr)?;
                }

                // Pad for kernel position offset: left = k*d, right = (K-1-k)*d
                let left = kpos[j] * dilations[j];
                let right = (block_shape[j] - 1 - kpos[j]) * dilations[j];
                if left > 0 || right > 0 {
                    let mut pad_spec: Vec<(isize, isize)> = vec![(0, 0); slice.shape()?.len()];
                    pad_spec[dim] = (left as isize, right as isize);
                    slice = slice.try_pad(&pad_spec)?;
                }
            }

            result = result.try_add(&slice)?;
        }

        // Shrink to remove padding → [N*C, *image_shape]
        let mut shrink_ranges: Vec<(isize, isize)> = vec![(0, nc as isize)];
        for j in 0..n_spatial {
            shrink_ranges.push((pads[j].0, pads[j].0 + image_shape[j] as isize));
        }
        let result = result.try_shrink(&shrink_ranges)?;

        // Reshape to [N, C, *image_shape]
        let mut final_shape: Vec<isize> = vec![n as isize, c as isize];
        final_shape.extend(image_shape.iter().map(|&s| s as isize));
        result.try_reshape(&final_shape)
    }
}
