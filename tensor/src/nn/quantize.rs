use bon::bon;
use morok_dtype::DType;

use crate::Tensor;

type Result<T> = crate::Result<T>;

#[bon]
impl Tensor {
    /// Clamp to the representable range of `dtype`, then cast.
    pub fn clamp_cast(&self, dtype: DType) -> Result<Self> {
        let min = Tensor::const_(dtype.min_value(), self.uop().dtype());
        let max = Tensor::const_(dtype.max_value(), self.uop().dtype());
        self.clamp().min(&min).max(&max).call()?.cast(dtype)
    }

    /// Quantized convolution: zero-point–adjust inputs, convolve in int32,
    /// rescale and requantize to the output dtype.
    #[builder]
    pub fn qlinear_conv(
        &self,
        x_scale: &Tensor,
        x_zero_point: &Tensor,
        weight: &Tensor,
        w_scale: &Tensor,
        w_zero_point: &Tensor,
        y_scale: &Tensor,
        y_zero_point: &Tensor,
        bias: Option<&Tensor>,
        #[builder(default)] auto_pad: super::AutoPad,
        #[builder(default = 1)] group: usize,
        kernel_shape: Option<&[usize]>,
        pads: Option<&[i64]>,
        strides: Option<&[i64]>,
        dilations: Option<&[i64]>,
    ) -> Result<Tensor> {
        let adj_x = self.cast(DType::Int32)?.try_sub(&x_zero_point.cast(DType::Int32)?)?;
        let w_i32 = weight.cast(DType::Int32)?;
        let w_zp = reshape_per_channel(&w_zero_point.cast(DType::Int32)?, w_i32.ndim()?)?;
        let adj_w = w_i32.try_sub(&w_zp)?;
        let conv_out = adj_x
            .conv()
            .weight(&adj_w)
            .maybe_bias(bias)
            .auto_pad(auto_pad)
            .group(group)
            .maybe_kernel_shape(kernel_shape)
            .maybe_pads(pads)
            .maybe_strides(strides)
            .maybe_dilations(dilations)
            .call()?;
        requantize(&conv_out, &[x_scale, w_scale], y_scale, y_zero_point)
    }

    /// Integer convolution: zero-point–adjust inputs and convolve in int32.
    /// No rescaling — returns raw int32 result.
    #[builder]
    pub fn conv_integer(
        &self,
        weight: &Tensor,
        x_zero_point: Option<&Tensor>,
        w_zero_point: Option<&Tensor>,
        bias: Option<&Tensor>,
        #[builder(default)] auto_pad: super::AutoPad,
        #[builder(default = 1)] group: usize,
        kernel_shape: Option<&[usize]>,
        pads: Option<&[i64]>,
        strides: Option<&[i64]>,
        dilations: Option<&[i64]>,
    ) -> Result<Tensor> {
        let adj_x = if let Some(zp) = x_zero_point {
            self.cast(DType::Int32)?.try_sub(&zp.cast(DType::Int32)?)?
        } else {
            self.cast(DType::Int32)?
        };
        let w_i32 = weight.cast(DType::Int32)?;
        let adj_w = if let Some(zp) = w_zero_point {
            let w_zp = reshape_per_channel(&zp.cast(DType::Int32)?, w_i32.ndim()?)?;
            w_i32.try_sub(&w_zp)?
        } else {
            w_i32
        };
        adj_x
            .conv()
            .weight(&adj_w)
            .maybe_bias(bias)
            .auto_pad(auto_pad)
            .group(group)
            .maybe_kernel_shape(kernel_shape)
            .maybe_pads(pads)
            .maybe_strides(strides)
            .maybe_dilations(dilations)
            .call()
    }

    /// Quantized matrix multiplication: zero-point–adjust inputs, matmul in int32,
    /// rescale and requantize to the output dtype.
    #[builder]
    pub fn qlinear_matmul(
        &self,
        a_scale: &Tensor,
        a_zero_point: &Tensor,
        b: &Tensor,
        b_scale: &Tensor,
        b_zero_point: &Tensor,
        y_scale: &Tensor,
        y_zero_point: &Tensor,
    ) -> Result<Tensor> {
        let adj_a = self.cast(DType::Int32)?.try_sub(&a_zero_point.cast(DType::Int32)?)?;
        let adj_b = b.cast(DType::Int32)?.try_sub(&b_zero_point.cast(DType::Int32)?)?;
        let out = adj_a.matmul(&adj_b)?;
        requantize(&out, &[a_scale, b_scale], y_scale, y_zero_point)
    }
}

/// Reshape a per-channel zero point `(C,)` to broadcast against a weight
/// tensor `(C, ...)` by appending singleton dimensions.
fn reshape_per_channel(zp: &Tensor, target_ndim: usize) -> Result<Tensor> {
    let zp_ndim = zp.ndim()?;
    if zp_ndim == 0 || zp_ndim == target_ndim {
        return Ok(zp.clone());
    }
    let mut shape: Vec<isize> = vec![-1];
    shape.extend(std::iter::repeat_n(1, target_ndim - 1));
    zp.try_reshape(&shape)
}

/// Rescale an integer result and requantize to the output zero-point's dtype.
///
/// No clamping: overflow means broken calibration — let it surface as garbage
/// rather than silently saturating to boundary values.
/// Round → Int32 → target dtype (int-to-int trunc wraps naturally).
fn requantize(int_result: &Tensor, scales: &[&Tensor], out_scale: &Tensor, out_zero_point: &Tensor) -> Result<Tensor> {
    let out_dtype = out_zero_point.uop().dtype();
    let scale_dtype = out_scale.uop().dtype();
    // Compute combined scale with explicit rounding to the scale's native
    // dtype between operations. LLVM promotes _Float16 to float for
    // arithmetic on x86 and may skip the intermediate fptrunc, keeping
    // float32 precision. Roundtripping through float64→scale_dtype after
    // each step forces correct intermediate rounding (matching numpy).
    let mut combined = scales[0].cast(DType::Float64)?;
    for s in &scales[1..] {
        combined = combined.try_mul(&s.cast(DType::Float64)?)?.cast(scale_dtype.clone())?.cast(DType::Float64)?;
    }
    combined = combined.try_div(&out_scale.cast(DType::Float64)?)?.cast(scale_dtype.clone())?;
    // Promote both operands to f64 for the final multiply (int32 * f16 → f64 in numpy)
    let rescaled = int_result
        .cast(DType::Float64)?
        .try_mul(&combined.cast(DType::Float64)?)?
        .try_add(&out_zero_point.cast(DType::Float64)?)?
        .round()?;
    // Float → Int32 (safe range) → target dtype (int trunc wraps)
    rescaled.cast(DType::Int32)?.cast(out_dtype)
}
