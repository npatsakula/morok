//! Convolution operations: conv2d, conv_transpose2d.

use bon::bon;

use morok_ir::SInt;

use crate::Tensor;
use crate::reduce::AxisSpec;

type Result<T> = crate::Result<T>;

#[bon]
impl Tensor {
    /// N-d convolution. Input `(N, Cin, *spatial)`, Weight `(Cout, Cin/groups, *kernel)`.
    ///
    /// Computes cross-correlation (conv without kernel flip) by extracting sliding
    /// windows via [`pool`](Tensor::pool), then contracting against the weight tensor.
    /// Supports grouped convolution, strided/dilated kernels, and asymmetric padding.
    ///
    /// # Examples
    ///
    /// Basic 2D convolution with uniform data:
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// # use ndarray::Array4;
    /// let x = Tensor::from_ndarray(&Array4::from_elem((1, 1, 5, 5), 1.0f32));
    /// let w = Tensor::from_ndarray(&Array4::from_elem((1, 1, 3, 3), 1.0f32));
    /// let mut y = x.conv2d().weight(&w).call().unwrap();
    /// y.realize().unwrap();
    /// // 3x3 kernel of ones on input of ones => each output element is 9.0
    /// assert_eq!(y.as_vec::<f32>().unwrap(), vec![9.0; 9]);
    /// ```
    ///
    /// With stride:
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// # use ndarray::Array4;
    /// let x = Tensor::from_ndarray(&Array4::from_elem((1, 1, 5, 5), 1.0f32));
    /// let w = Tensor::from_ndarray(&Array4::from_elem((1, 1, 3, 3), 1.0f32));
    /// let mut y = x.conv2d().weight(&w).stride(&[2, 2]).call().unwrap();
    /// y.realize().unwrap();
    /// let shape: Vec<_> = y.shape().unwrap().iter().map(|d| d.as_const().unwrap()).collect();
    /// assert_eq!(shape, vec![1, 1, 2, 2]);
    /// assert_eq!(y.as_vec::<f32>().unwrap(), vec![9.0; 4]);
    /// ```
    ///
    /// With padding:
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// # use ndarray::Array4;
    /// let x = Tensor::from_ndarray(&Array4::from_elem((1, 1, 3, 3), 1.0f32));
    /// let w = Tensor::from_ndarray(&Array4::from_elem((1, 1, 3, 3), 1.0f32));
    /// // padding=1 on each side: output matches input spatial dims
    /// let mut y = x.conv2d().weight(&w).padding(&[(1, 1), (1, 1)]).call().unwrap();
    /// y.realize().unwrap();
    /// let vals = y.as_vec::<f32>().unwrap();
    /// assert_eq!(vals.len(), 9); // 3x3 output
    /// // Center element sees full 3x3 window of ones = 9.0
    /// assert_eq!(vals[4], 9.0);
    /// // Corner element sees 2x2 window = 4.0
    /// assert_eq!(vals[0], 4.0);
    /// ```
    ///
    /// With bias:
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// # use ndarray::Array4;
    /// let x = Tensor::from_ndarray(&Array4::from_elem((1, 1, 3, 3), 1.0f32));
    /// let w = Tensor::from_ndarray(&Array4::from_elem((1, 1, 3, 3), 1.0f32));
    /// let b = Tensor::from_slice([10.0f32]);
    /// let mut y = x.conv2d().weight(&w).bias(&b).call().unwrap();
    /// y.realize().unwrap();
    /// // Each output element: 9.0 + 10.0 = 19.0
    /// assert_eq!(y.as_vec::<f32>().unwrap(), vec![19.0]);
    /// ```
    #[builder]
    pub fn conv2d(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        #[builder(default = 1)] groups: usize,
        stride: Option<&[usize]>,
        dilation: Option<&[usize]>,
        padding: Option<&[(isize, isize)]>,
        acc_dtype: Option<morok_dtype::DType>,
    ) -> Result<Tensor> {
        let x_shape = self.shape()?;
        let w_shape = weight.shape()?;

        let bs = x_shape[0].clone(); // SInt — concrete or symbolic (Variable batch)
        let cin_ = x_shape[1].as_const().expect("channel dim must be concrete");
        let cout = w_shape[0].as_const().expect("cout must be concrete");
        let cin = w_shape[1].as_const().expect("cin/g must be concrete");

        let hw: Vec<usize> = w_shape[2..].iter().map(|s| s.as_const().expect("kernel dim must be concrete")).collect();
        let n_spatial = hw.len();

        if x_shape.len() != w_shape.len() {
            return Err(crate::error::Error::IrConstruction {
                details: format!("input and weight must have same ndim, got {} and {}", x_shape.len(), w_shape.len()),
            });
        }
        if groups * cin != cin_ {
            return Err(crate::error::Error::IrConstruction {
                details: format!("groups*cin/g ({}) != input channels ({cin_})", groups * cin),
            });
        }

        let default_ones: Vec<usize> = vec![1; n_spatial];
        let stride = stride.unwrap_or(&default_ones);
        let dilation = dilation.unwrap_or(&default_ones);
        let no_padding: Vec<(isize, isize)> = vec![(0, 0); n_spatial];
        let padding = padding.unwrap_or(&no_padding);

        let mut x = self.clone();
        if padding.iter().any(|&(b, e)| b != 0 || e != 0) {
            let mut full_pad: Vec<(isize, isize)> = vec![(0, 0); 2];
            full_pad.extend_from_slice(padding);
            x = x.try_pad(&full_pad)?;
        }

        x = x.pool(&hw, stride, dilation)?;

        let oyx: Vec<SInt> = {
            let xs = x.shape()?;
            xs[2..2 + n_spatial].to_vec()
        };

        let rcout = cout / groups;

        // Reshape: (bs, groups, cin, 1, *oyx, *hw)
        let mut reshape_dims: Vec<SInt> = vec![bs.clone(), groups.into(), cin.into(), 1usize.into()];
        reshape_dims.extend(oyx.iter().cloned());
        reshape_dims.extend(hw.iter().map(|&k| SInt::from(k)));
        x = x.try_reshape(&reshape_dims)?;

        // Expand: (bs, groups, cin, rcout, *oyx, *hw)
        let mut expand_dims: Vec<SInt> = vec![bs.clone(), groups.into(), cin.into(), rcout.into()];
        expand_dims.extend(oyx.iter().cloned());
        expand_dims.extend(hw.iter().map(|&k| SInt::from(k)));
        x = x.try_expand(&expand_dims)?;

        // Permute: (bs, groups, rcout, *oyx, cin, *hw)
        let mut perm: Vec<isize> = vec![0, 1, 3];
        for j in 0..n_spatial {
            perm.push(4 + j as isize);
        }
        perm.push(2);
        for j in 0..n_spatial {
            perm.push((4 + n_spatial + j) as isize);
        }
        x = x.try_permute(&perm)?;

        // Reshape weight: (1, groups, rcout, *[1]*n_spatial, cin, *hw)
        let mut w_reshape: Vec<isize> = vec![1, groups as isize, rcout as isize];
        w_reshape.extend(std::iter::repeat_n(1isize, n_spatial));
        w_reshape.push(cin as isize);
        w_reshape.extend(hw.iter().map(|&k| k as isize));
        let w = weight.try_reshape(&w_reshape)?;

        x = x.try_mul(&w)?;

        // Sum over last (1 + n_spatial) dims
        let total_dims = x.ndim()?;
        let reduce_axes: Vec<isize> = (0..(1 + n_spatial)).map(|i| (total_dims - 1 - i) as isize).collect();
        x = x.sum_with().axes(AxisSpec::Multiple(reduce_axes)).keepdim(true).maybe_dtype(acc_dtype).call()?;

        // Reshape to (bs, cout, *oyx)
        let mut final_shape: Vec<SInt> = vec![bs.clone(), cout.into()];
        final_shape.extend(oyx.iter().cloned());
        x = x.try_reshape(&final_shape)?;

        if let Some(bias) = bias {
            let mut bias_shape: Vec<isize> = vec![1, cout as isize];
            bias_shape.extend(std::iter::repeat_n(1isize, n_spatial));
            let bias = bias.try_reshape(&bias_shape)?;
            x = x.try_add(&bias)?;
        }

        Ok(x)
    }

    /// Transposed convolution (fractionally-strided convolution).
    ///
    /// Computes the gradient of a forward convolution, commonly used for upsampling.
    /// Internally flips the kernel, interleaves zeros for stride > 1, computes
    /// transposed padding, then delegates to [`conv2d`](Tensor::conv2d).
    ///
    /// Input `(N, Cin, *spatial)`, Weight `(Cin, Cout/groups, *kernel)`.
    ///
    /// # Examples
    ///
    /// Basic transposed convolution (upsampling):
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// # use ndarray::Array4;
    /// let x = Tensor::from_ndarray(&Array4::from_elem((1, 1, 2, 2), 1.0f32));
    /// let w = Tensor::from_ndarray(&Array4::from_elem((1, 1, 3, 3), 1.0f32));
    /// let mut y = x.conv_transpose2d().weight(&w).call().unwrap();
    /// y.realize().unwrap();
    /// let vals = y.as_vec::<f32>().unwrap();
    /// assert_eq!(vals.len(), 16); // 4x4 output
    /// // Center elements see full overlap of both input positions
    /// assert_eq!(vals[5], 4.0);
    /// ```
    ///
    /// With stride (stronger upsampling):
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// # use ndarray::Array4;
    /// let x = Tensor::from_ndarray(&Array4::from_elem((1, 1, 2, 2), 1.0f32));
    /// let w = Tensor::from_ndarray(&Array4::from_elem((1, 1, 3, 3), 1.0f32));
    /// let mut y = x.conv_transpose2d().weight(&w).stride(&[2, 2]).call().unwrap();
    /// y.realize().unwrap();
    /// let vals = y.as_vec::<f32>().unwrap();
    /// assert_eq!(vals.len(), 25); // 5x5 output
    /// ```
    ///
    /// With padding and output padding:
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// # use ndarray::Array4;
    /// let x = Tensor::from_ndarray(&Array4::from_elem((1, 1, 2, 2), 1.0f32));
    /// let w = Tensor::from_ndarray(&Array4::from_elem((1, 1, 3, 3), 1.0f32));
    /// let mut y = x.conv_transpose2d()
    ///     .weight(&w)
    ///     .stride(&[2, 2])
    ///     .padding(&[(1, 1), (1, 1)])
    ///     .output_padding(&[1, 1])
    ///     .call()
    ///     .unwrap();
    /// y.realize().unwrap();
    /// let vals = y.as_vec::<f32>().unwrap();
    /// assert_eq!(vals.len(), 16); // 4x4 output
    /// ```
    #[builder]
    pub fn conv_transpose2d(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        #[builder(default = 1)] groups: usize,
        stride: Option<&[usize]>,
        dilation: Option<&[usize]>,
        padding: Option<&[(isize, isize)]>,
        output_padding: Option<&[usize]>,
    ) -> Result<Tensor> {
        let w_shape = weight.shape()?;
        let hw: Vec<usize> = w_shape[2..].iter().map(|s| s.as_const().expect("kernel dim must be concrete")).collect();
        let n_spatial = hw.len();

        let default_ones: Vec<usize> = vec![1; n_spatial];
        let default_zeros: Vec<usize> = vec![0; n_spatial];
        let default_no_pad: Vec<(isize, isize)> = vec![(0, 0); n_spatial];
        let stride = stride.unwrap_or(&default_ones);
        let dilation = dilation.unwrap_or(&default_ones);
        let padding = padding.unwrap_or(&default_no_pad);
        let output_padding = output_padding.unwrap_or(&default_zeros);

        let cout_in = w_shape[0].as_const().unwrap();
        let cin_g = w_shape[1].as_const().unwrap();
        let rcout = cout_in / groups;

        // Reshape to (groups, rcout, cin_g, *HW)
        let mut unflatten_shape: Vec<isize> = vec![groups as isize, rcout as isize, cin_g as isize];
        unflatten_shape.extend(hw.iter().map(|&k| k as isize));
        let mut w = weight.try_reshape(&unflatten_shape)?;

        // Transpose dim 1 and 2: (groups, cin_g, rcout, *HW)
        w = w.try_transpose(1, 2)?;

        // Flip kernel dims
        let flip_axes: Vec<isize> = (3..(3 + n_spatial) as isize).collect();
        w = w.flip(&flip_axes)?;

        // Flatten back: (groups * cin_g, rcout, *HW)
        let mut flat_shape: Vec<isize> = vec![(groups * cin_g) as isize, rcout as isize];
        flat_shape.extend(hw.iter().map(|&k| k as isize));
        w = w.try_reshape(&flat_shape)?;

        // Handle stride > 1: interleave zeros across all spatial dims at once.
        // Matches Tinygrad: (k) -> reshape (k,1) -> pad (k,s) -> reshape (k*s) -> shrink (k-(s-1))
        // All spatial dims are processed in a single reshape/pad/reshape/shrink sequence
        // to avoid cascading PAD operations that create exponential boolean condition trees.
        let mut x = self.clone();
        if stride.iter().any(|&s| s > 1) {
            let x_shape = x.shape()?;
            let spatial: Vec<usize> = x_shape[2..].iter().map(|s| s.as_const().unwrap()).collect();

            // Step 1: reshape (N,C,h,w) -> (N,C,h,1,w,1)
            let mut rshape: Vec<SInt> = vec![x_shape[0].clone(), x_shape[1].clone()];
            for &k in &spatial {
                rshape.push(k.into());
                rshape.push(1usize.into());
            }
            x = x.try_reshape(&rshape)?;

            // Step 2: pad inserted dims by (0, s-1): (N,C,h,s,w,s)
            let mut pad_spec: Vec<(isize, isize)> = vec![(0, 0); 2];
            for &s in stride.iter() {
                pad_spec.push((0, 0));
                pad_spec.push((0, (s - 1) as isize));
            }
            x = x.try_pad(&pad_spec)?;

            // Step 3: reshape to merge pairs: (N,C,h*s,w*s)
            let x_shape = x.shape()?;
            let mut rshape: Vec<SInt> = vec![x_shape[0].clone(), x_shape[1].clone()];
            for j in 0..n_spatial {
                let a = x_shape[2 + j * 2].as_const().unwrap();
                let b = x_shape[2 + j * 2 + 1].as_const().unwrap();
                rshape.push((a * b).into());
            }
            x = x.try_reshape(&rshape)?;

            // Step 4: shrink to remove trailing stride-1
            // Use None for batch/channel dims (pass through).
            let mut ranges: Vec<Option<(isize, isize)>> = vec![None, None];
            for j in 0..n_spatial {
                let new_size = spatial[j] * stride[j] - (stride[j] - 1);
                ranges.push(Some((0, new_size as isize)));
            }
            x = x.try_shrink(&ranges)?;
        }

        // Compute transposed padding
        let conv_padding: Vec<(isize, isize)> = (0..n_spatial)
            .map(|j| {
                let pb = padding[j].0;
                let pa = padding[j].1;
                let begin = (hw[j] as isize - 1) * dilation[j] as isize - pb;
                let end = (hw[j] as isize - 1) * dilation[j] as isize - pa + output_padding[j] as isize;
                (begin, end)
            })
            .collect();

        x.conv2d().weight(&w).groups(groups).maybe_bias(bias).dilation(dilation).padding(&conv_padding).call()
    }
}
