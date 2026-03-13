//! GridSample: spatial sampling via coordinate grids (ONNX GridSample operator).

use bon::bon;
use morok_dtype::DType;
use morok_ir::ConstValue;
use snafu::ResultExt;

use crate::Tensor;
use crate::error::{NdimMinimumSnafu, UOpSnafu};
use crate::shape_ops::MeshgridIndexing;

use super::{GridSampleMode, GridSamplePaddingMode};

type Result<T> = crate::Result<T>;

#[bon]
impl Tensor {
    /// Generate an affine sampling grid from transformation parameters.
    ///
    /// Produces a grid of normalized coordinates suitable for [`grid_sample`](Tensor::grid_sample).
    /// `theta` holds affine matrices of shape `[N, spatial_dims, spatial_dims+1]`.
    /// `size` is the target output shape `[N, C, *spatial_dims]`.
    ///
    /// # Examples
    ///
    /// Identity transform producing a 4x4 grid:
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// # use ndarray::array;
    /// let theta = Tensor::from_ndarray(&array![[[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0]]]);
    /// let grid = Tensor::affine_grid().theta(&theta).size(&[1, 1, 4, 4]).call().unwrap();
    /// let shape: Vec<usize> = grid.shape().unwrap().iter()
    ///     .map(|d| d.as_const().unwrap()).collect();
    /// assert_eq!(shape, vec![1, 4, 4, 2]); // [N, H, W, 2]
    /// ```
    ///
    /// With `align_corners`:
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// # use ndarray::array;
    /// let theta = Tensor::from_ndarray(&array![[[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0]]]);
    /// let grid = Tensor::affine_grid()
    ///     .theta(&theta)
    ///     .size(&[1, 1, 4, 4])
    ///     .align_corners(true)
    ///     .call()
    ///     .unwrap();
    /// let shape: Vec<usize> = grid.shape().unwrap().iter()
    ///     .map(|d| d.as_const().unwrap()).collect();
    /// assert_eq!(shape, vec![1, 4, 4, 2]);
    /// ```
    #[builder]
    pub fn affine_grid(
        theta: &Tensor,
        size: &[i64],
        #[builder(default = false)] align_corners: bool,
    ) -> Result<Tensor> {
        snafu::ensure!(size.len() >= 3, NdimMinimumSnafu { op: "affine_grid", min: 3_usize, actual: size.len() });
        let n = size[0] as usize;
        let ndim = size.len() - 2; // spatial dims

        let spatial_dims: Vec<usize> = size[2..].iter().map(|&s| s as usize).collect();
        let mut grids = Vec::with_capacity(ndim);
        for &dim_size in &spatial_dims {
            let g = if align_corners {
                Tensor::linspace(-1.0, 1.0, dim_size, DType::Float32)?
            } else {
                let start = -1.0 + 1.0 / dim_size as f64;
                let end = 1.0 - 1.0 / dim_size as f64;
                Tensor::linspace(start, end, dim_size, DType::Float32)?
            };
            grids.push(g);
        }

        let grid_refs: Vec<&Tensor> = grids.iter().collect();
        let mesh = Tensor::meshgrid(&grid_refs, MeshgridIndexing::Ij)?;

        let total_elements: usize = spatial_dims.iter().product();
        let flat_shape = [total_elements as isize];
        let mut components: Vec<Tensor> = Vec::with_capacity(ndim + 1);
        for g in mesh.iter().rev() {
            components.push(g.try_reshape(&flat_shape)?);
        }
        components.push(Tensor::full(&[total_elements], 1.0, DType::Float32)?);

        let comp_refs: Vec<&Tensor> = components.iter().collect();
        let base_grid = Tensor::cat(&comp_refs, 0)?
            .try_reshape(&[(ndim + 1) as isize, total_elements as isize])?
            .try_transpose(0, 1)?;

        let base_grid =
            base_grid.try_unsqueeze(0)?.try_expand(&[n as isize, total_elements as isize, (ndim + 1) as isize])?;

        let theta_t = theta.try_transpose(1, 2)?;
        let output = base_grid.matmul(&theta_t)?;

        let mut out_shape: Vec<isize> = vec![n as isize];
        out_shape.extend(spatial_dims.iter().map(|&d| d as isize));
        out_shape.push(ndim as isize);
        output.try_reshape(&out_shape)
    }

    /// Sample input at positions specified by a coordinate grid.
    ///
    /// - `self`: Input tensor `[N, C, *spatial_dims]`
    /// - `grid`: Coordinate grid `[N, *output_spatial_dims, n_spatial]` with values in `[-1, 1]`
    /// - Returns: `[N, C, *output_spatial_dims]`
    ///
    /// # Examples
    ///
    /// Sample with a grid from `affine_grid`:
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// # use ndarray::{array, Array4};
    /// let theta = Tensor::from_ndarray(&array![[[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0]]]);
    /// let grid = Tensor::affine_grid().theta(&theta).size(&[1, 1, 4, 4]).call().unwrap();
    /// let x = Tensor::from_ndarray(&Array4::from_elem((1, 1, 4, 4), 1.0f32));
    /// let y = x.grid_sample().grid(&grid).call().unwrap();
    /// let shape: Vec<usize> = y.shape().unwrap().iter()
    ///     .map(|d| d.as_const().unwrap()).collect();
    /// assert_eq!(shape, vec![1, 1, 4, 4]);
    /// ```
    ///
    /// With nearest-mode interpolation:
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// # use morok_tensor::nn::GridSampleMode;
    /// # use ndarray::{array, Array4};
    /// let theta = Tensor::from_ndarray(&array![[[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0]]]);
    /// let grid = Tensor::affine_grid().theta(&theta).size(&[1, 1, 4, 4]).call().unwrap();
    /// let x = Tensor::from_ndarray(&Array4::from_elem((1, 1, 4, 4), 1.0f32));
    /// let y = x.grid_sample()
    ///     .grid(&grid)
    ///     .mode(GridSampleMode::Nearest)
    ///     .call()
    ///     .unwrap();
    /// let shape: Vec<usize> = y.shape().unwrap().iter()
    ///     .map(|d| d.as_const().unwrap()).collect();
    /// assert_eq!(shape, vec![1, 1, 4, 4]);
    /// ```
    #[builder]
    pub fn grid_sample(
        &self,
        grid: &Tensor,
        #[builder(default)] mode: GridSampleMode,
        #[builder(default)] padding_mode: GridSamplePaddingMode,
        #[builder(default = false)] align_corners: bool,
    ) -> Result<Tensor> {
        let x_ndim = self.ndim()?;
        snafu::ensure!(x_ndim >= 3, NdimMinimumSnafu { op: "grid_sample", min: 3_usize, actual: x_ndim });
        let x_shape = self.shape()?;
        let grid_shape = grid.shape()?;
        let x_dims = morok_ir::shape::to_vec_usize(&x_shape).context(UOpSnafu)?;
        let grid_dims = morok_ir::shape::to_vec_usize(&grid_shape).context(UOpSnafu)?;
        let n_spatial = x_dims.len() - 2;

        let n = x_dims[0];
        let c = x_dims[1];
        let spatial: Vec<usize> = x_dims[2..].to_vec();
        let out_spatial: Vec<usize> = grid_dims[1..grid_dims.len() - 1].to_vec();
        let spatial_prod: usize = spatial.iter().product();
        let out_prod: usize = out_spatial.iter().product();
        let dtype = self.uop().dtype();

        // Flatten X spatial: [N, C, prod(spatial)]
        let x_flat = self.try_reshape(&[n as isize, c as isize, spatial_prod as isize])?;

        // Flatten grid spatial: [N, prod(out_spatial), n_spatial]
        let grid_flat = grid.try_reshape(&[n as isize, out_prod as isize, n_spatial as isize])?;

        // Strides for flat index: stride[i] = product of spatial[i+1..]
        let strides = compute_strides(&spatial);

        // Extract, denormalize coordinates for each spatial dim.
        // Grid stores coords in reverse order: grid[...,0]=x→last spatial dim, etc.
        let mut coords: Vec<Tensor> = Vec::with_capacity(n_spatial);
        for (i, &dim_size) in spatial.iter().enumerate() {
            let grid_idx = n_spatial - 1 - i;
            let coord = slice_last_dim(&grid_flat, grid_idx, n, out_prod)?;
            let denorm = gs_denormalize(&coord, dim_size, align_corners, &dtype)?;
            coords.push(denorm);
        }

        // Apply padding mode to float coordinates before interpolation
        let coords = match padding_mode {
            GridSamplePaddingMode::Border => coords
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    let zero = Tensor::const_(0.0, dtype.clone());
                    let max_val = Tensor::const_((spatial[i] - 1) as f64, dtype.clone());
                    c.clamp().min(&zero).max(&max_val).call()
                })
                .collect::<Result<Vec<_>>>()?,
            GridSamplePaddingMode::Reflection => coords
                .iter()
                .enumerate()
                .map(|(i, c)| gs_reflect(c, spatial[i], align_corners, &dtype))
                .collect::<Result<Vec<_>>>()?,
            GridSamplePaddingMode::Zeros => coords,
        };

        let result = match mode {
            GridSampleMode::Nearest => {
                interpolate_nearest(&x_flat, &coords, &spatial, &strides, padding_mode, n, c, out_prod, &dtype)?
            }
            GridSampleMode::Linear => {
                interpolate_linear(&x_flat, &coords, &spatial, &strides, padding_mode, n, c, out_prod, &dtype)?
            }
            GridSampleMode::Cubic => {
                interpolate_cubic(&x_flat, &coords, &spatial, &strides, padding_mode, n, c, out_prod, &dtype)?
            }
        };

        // Reshape to [N, C, *out_spatial]
        let mut out_shape: Vec<isize> = vec![n as isize, c as isize];
        out_shape.extend(out_spatial.iter().map(|&d| d as isize));
        result.try_reshape(&out_shape)
    }
}

fn compute_strides(dims: &[usize]) -> Vec<usize> {
    let n = dims.len();
    let mut strides = vec![1usize; n];
    for i in (0..n.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    strides
}

/// Extract `t[:, :, idx]` from shape `[N, out_prod, n_spatial]` → `[N, out_prod]`.
fn slice_last_dim(t: &Tensor, idx: usize, n: usize, out_prod: usize) -> Result<Tensor> {
    t.try_shrink(&[(0, n as isize), (0, out_prod as isize), (idx as isize, (idx + 1) as isize)])?.try_squeeze(Some(-1))
}

/// Denormalize grid coordinate from [-1, 1] to pixel space.
fn gs_denormalize(coord: &Tensor, dim_size: usize, align_corners: bool, dtype: &DType) -> Result<Tensor> {
    if align_corners {
        // x = (n + 1) / 2 * (dim_size - 1)
        coord
            .try_add(&Tensor::const_(1.0, dtype.clone()))?
            .try_mul(&Tensor::const_(0.5 * (dim_size - 1) as f64, dtype.clone()))
    } else {
        // x = ((n + 1) * dim_size - 1) / 2
        coord
            .try_add(&Tensor::const_(1.0, dtype.clone()))?
            .try_mul(&Tensor::const_(dim_size as f64, dtype.clone()))?
            .try_sub(&Tensor::const_(1.0, dtype.clone()))?
            .try_mul(&Tensor::const_(0.5, dtype.clone()))
    }
}

/// Reflect coordinate into [lo, hi] range for reflection padding.
fn gs_reflect(coord: &Tensor, dim_size: usize, align_corners: bool, dtype: &DType) -> Result<Tensor> {
    let (lo, hi) = if align_corners { (0.0, (dim_size - 1) as f64) } else { (-0.5, dim_size as f64 - 0.5) };
    let rng = hi - lo;
    if rng == 0.0 {
        return Ok(Tensor::const_(lo, dtype.clone()));
    }
    let lo_t = Tensor::const_(lo, dtype.clone());
    let rng_t = Tensor::const_(rng, dtype.clone());
    let period_t = Tensor::const_(2.0 * rng, dtype.clone());

    // Shift to [0, 2*rng) via positive modulo
    let shifted = coord.try_sub(&lo_t)?;
    let t = shifted.try_sub(&shifted.try_div(&period_t)?.floor()?.try_mul(&period_t)?)?;

    // Reflect: if t > rng → 2*rng - t, else t
    let two_rng_t = Tensor::const_(2.0 * rng, dtype.clone());
    let reflected = two_rng_t.try_sub(&t)?;
    let cond = rng_t.try_lt(&t)?; // t > rng
    reflected.where_(&cond, &t)?.try_add(&lo_t)
}

/// Build flat index from per-dim integer indices and accumulate validity mask for zeros padding.
fn build_flat_index(
    indices: &[Tensor],
    spatial: &[usize],
    strides: &[usize],
    padding_mode: GridSamplePaddingMode,
) -> Result<(Tensor, Option<Tensor>)> {
    let n_spatial = indices.len();
    let mut flat_idx = Tensor::const_(ConstValue::Int(0), DType::Int32);
    let mut valid_mask: Option<Tensor> = None;

    for i in 0..n_spatial {
        let idx = &indices[i];

        if padding_mode == GridSamplePaddingMode::Zeros {
            let zero_i = Tensor::const_(ConstValue::Int(0), DType::Int32);
            let max_i = Tensor::const_(ConstValue::Int(spatial[i] as i64), DType::Int32);
            let v = idx.try_ge(&zero_i)?.bitwise_and(&idx.try_lt(&max_i)?)?;
            valid_mask = Some(match valid_mask {
                Some(m) => m.bitwise_and(&v)?,
                None => v,
            });
        }

        // Clamp for safe gather (even out-of-bounds values need a valid index for gather)
        let zero_i = Tensor::const_(ConstValue::Int(0), DType::Int32);
        let max_i = Tensor::const_(ConstValue::Int((spatial[i] - 1) as i64), DType::Int32);
        let safe_idx = idx.clamp().min(&zero_i).max(&max_i).call()?;

        let stride_t = Tensor::const_(ConstValue::Int(strides[i] as i64), DType::Int32);
        flat_idx = flat_idx.try_add(&safe_idx.try_mul(&stride_t)?)?;
    }

    Ok((flat_idx, valid_mask))
}

/// Gather from flat X and apply zeros mask if needed.
fn gather_and_mask(
    x_flat: &Tensor,
    flat_idx: &Tensor,
    valid_mask: Option<&Tensor>,
    n: usize,
    c: usize,
    out_prod: usize,
    dtype: &DType,
) -> Result<Tensor> {
    let expanded_idx = flat_idx.try_unsqueeze(1)?.try_expand(&[n as isize, c as isize, out_prod as isize])?;
    let mut gathered = x_flat.gather(2, &expanded_idx)?;
    if let Some(mask) = valid_mask {
        let mask = mask.try_unsqueeze(1)?.try_expand(&[n as isize, c as isize, out_prod as isize])?;
        gathered = gathered.try_mul(&mask.cast(dtype.clone())?)?;
    }
    Ok(gathered)
}

#[allow(clippy::too_many_arguments)]
fn interpolate_nearest(
    x_flat: &Tensor,
    coords: &[Tensor],
    spatial: &[usize],
    strides: &[usize],
    padding_mode: GridSamplePaddingMode,
    n: usize,
    c: usize,
    out_prod: usize,
    dtype: &DType,
) -> Result<Tensor> {
    // ONNX uses np.rint (round to nearest even); Tensor::round() implements this.
    let rounded: Vec<Tensor> = coords.iter().map(|c| c.round()?.cast(DType::Int32)).collect::<Result<_>>()?;
    let (flat_idx, valid_mask) = build_flat_index(&rounded, spatial, strides, padding_mode)?;
    gather_and_mask(x_flat, &flat_idx, valid_mask.as_ref(), n, c, out_prod, dtype)
}

#[allow(clippy::too_many_arguments)]
fn interpolate_linear(
    x_flat: &Tensor,
    coords: &[Tensor],
    spatial: &[usize],
    strides: &[usize],
    padding_mode: GridSamplePaddingMode,
    n: usize,
    c: usize,
    out_prod: usize,
    dtype: &DType,
) -> Result<Tensor> {
    let n_spatial = coords.len();
    let floors: Vec<Tensor> = coords.iter().map(|c| c.floor()).collect::<Result<_>>()?;
    let fracs: Vec<Tensor> = coords.iter().zip(&floors).map(|(c, f)| c.try_sub(f)).collect::<Result<_>>()?;

    // 2^n_spatial corners
    let n_combos = 1usize << n_spatial;
    let mut result = Tensor::const_(ConstValue::Float(0.0), dtype.clone());

    for combo in 0..n_combos {
        let mut weight = Tensor::const_(ConstValue::Float(1.0), dtype.clone());
        let mut corner_indices: Vec<Tensor> = Vec::with_capacity(n_spatial);

        for i in 0..n_spatial {
            let use_ceil = (combo >> i) & 1 == 1;
            let idx_f =
                if use_ceil { floors[i].try_add(&Tensor::const_(1.0, dtype.clone()))? } else { floors[i].clone() };
            let w = if use_ceil { fracs[i].clone() } else { Tensor::const_(1.0, dtype.clone()).try_sub(&fracs[i])? };
            weight = weight.try_mul(&w)?;
            corner_indices.push(idx_f.cast(DType::Int32)?);
        }

        let (flat_idx, valid_mask) = build_flat_index(&corner_indices, spatial, strides, padding_mode)?;
        let gathered = gather_and_mask(x_flat, &flat_idx, valid_mask.as_ref(), n, c, out_prod, dtype)?;

        let weight = weight.try_unsqueeze(1)?.try_expand(&[n as isize, c as isize, out_prod as isize])?;
        result = result.try_add(&gathered.try_mul(&weight)?)?;
    }

    Ok(result)
}

#[allow(clippy::too_many_arguments)]
fn interpolate_cubic(
    x_flat: &Tensor,
    coords: &[Tensor],
    spatial: &[usize],
    strides: &[usize],
    padding_mode: GridSamplePaddingMode,
    n: usize,
    c: usize,
    out_prod: usize,
    dtype: &DType,
) -> Result<Tensor> {
    let n_spatial = coords.len();
    let floors: Vec<Tensor> = coords.iter().map(|c| c.floor()).collect::<Result<_>>()?;
    let fracs: Vec<Tensor> = coords.iter().zip(&floors).map(|(c, f)| c.try_sub(f)).collect::<Result<_>>()?;

    // Cubic coefficients for each spatial dim (4 weights per dim)
    let coeffs: Vec<[Tensor; 4]> = fracs.iter().map(|s| gs_cubic_coeffs(s, -0.75, dtype)).collect::<Result<_>>()?;

    // 4^n_spatial combinations
    let n_combos = 4usize.pow(n_spatial as u32);
    let mut result = Tensor::const_(ConstValue::Float(0.0), dtype.clone());

    for combo in 0..n_combos {
        let mut weight = Tensor::const_(ConstValue::Float(1.0), dtype.clone());
        let mut corner_indices: Vec<Tensor> = Vec::with_capacity(n_spatial);

        for i in 0..n_spatial {
            let offset_idx = (combo / 4usize.pow(i as u32)) % 4;
            let offset = offset_idx as f64 - 1.0; // -1, 0, 1, 2

            let idx_f = floors[i].try_add(&Tensor::const_(offset, dtype.clone()))?;
            weight = weight.try_mul(&coeffs[i][offset_idx])?;
            corner_indices.push(idx_f.cast(DType::Int32)?);
        }

        let (flat_idx, valid_mask) = build_flat_index(&corner_indices, spatial, strides, padding_mode)?;
        let gathered = gather_and_mask(x_flat, &flat_idx, valid_mask.as_ref(), n, c, out_prod, dtype)?;

        let weight = weight.try_unsqueeze(1)?.try_expand(&[n as isize, c as isize, out_prod as isize])?;
        result = result.try_add(&gathered.try_mul(&weight)?)?;
    }

    Ok(result)
}

/// Cubic interpolation coefficients (Keys convolution, alpha = -0.75).
/// Returns weights for offsets [-1, 0, 1, 2] relative to floor(x).
fn gs_cubic_coeffs(s: &Tensor, a: f64, dtype: &DType) -> Result<[Tensor; 4]> {
    let one = Tensor::const_(1.0, dtype.clone());
    let two = Tensor::const_(2.0, dtype.clone());

    // c0: |x| = s+1 (far neighbor)
    // c0 = ((a*(s+1) - 5a)*(s+1) + 8a)*(s+1) - 4a
    let sp1 = s.try_add(&one)?;
    let c0 = sp1
        .try_mul(&Tensor::const_(a, dtype.clone()))?
        .try_sub(&Tensor::const_(5.0 * a, dtype.clone()))?
        .try_mul(&sp1)?
        .try_add(&Tensor::const_(8.0 * a, dtype.clone()))?
        .try_mul(&sp1)?
        .try_sub(&Tensor::const_(4.0 * a, dtype.clone()))?;

    // c1: |x| = s (center-left)
    // c1 = ((a+2)*s - (a+3))*s*s + 1
    let c1 = s
        .try_mul(&Tensor::const_(a + 2.0, dtype.clone()))?
        .try_sub(&Tensor::const_(a + 3.0, dtype.clone()))?
        .try_mul(s)?
        .try_mul(s)?
        .try_add(&one)?;

    // c2: |x| = 1-s (center-right)
    let sm1 = one.try_sub(s)?;
    let c2 = sm1
        .try_mul(&Tensor::const_(a + 2.0, dtype.clone()))?
        .try_sub(&Tensor::const_(a + 3.0, dtype.clone()))?
        .try_mul(&sm1)?
        .try_mul(&sm1)?
        .try_add(&Tensor::const_(1.0, dtype.clone()))?;

    // c3: |x| = 2-s (far neighbor)
    let sm2 = two.try_sub(s)?;
    let c3 = sm2
        .try_mul(&Tensor::const_(a, dtype.clone()))?
        .try_sub(&Tensor::const_(5.0 * a, dtype.clone()))?
        .try_mul(&sm2)?
        .try_add(&Tensor::const_(8.0 * a, dtype.clone()))?
        .try_mul(&sm2)?
        .try_sub(&Tensor::const_(4.0 * a, dtype.clone()))?;

    Ok([c0, c1, c2, c3])
}
