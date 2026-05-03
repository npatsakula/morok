//! Neural network operations: convolution, pooling, normalization.

mod conv;
mod grid_sample;
mod linear;
mod norm;
pub mod pad;
mod pool;
mod quantize;
mod resize;
mod rnn;

pub use linear::Linear;
pub use rnn::{GruOutput, LstmOutput, RnnOutput};

/// A neural network layer.
pub trait Layer {
    fn forward(&self, x: &Tensor) -> Result<Tensor>;
}

/// ReLU activation layer: `max(0, x)`.
pub struct Relu;

impl Layer for Relu {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.relu()
    }
}

pub use pad::{auto_pad_split, flat_pads_to_pairs, resolve_pool_pads};

use bon::bon;
use morok_dtype::DType;
use morok_ir::SInt;
use snafu::ResultExt;

use crate::Tensor;
use crate::error::{DivisibilitySnafu, NdimExactSnafu, NdimMinimumSnafu, ParamRangeSnafu, UOpSnafu};
use crate::reduce::AxisSpec;

type Result<T> = crate::Result<T>;

// =========================================================================
// Type-safe enums for string parameters
// =========================================================================

use strum::{Display, EnumString};

/// Auto-padding mode for convolution and pooling.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, EnumString, Display)]
pub enum AutoPad {
    #[default]
    #[strum(serialize = "NOTSET", serialize = "")]
    NotSet,
    #[strum(serialize = "VALID")]
    Valid,
    #[strum(serialize = "SAME_UPPER")]
    SameUpper,
    #[strum(serialize = "SAME_LOWER")]
    SameLower,
}

/// Reduction mode for loss functions.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, EnumString, Display)]
pub enum Reduction {
    #[strum(serialize = "none")]
    None,
    #[default]
    #[strum(serialize = "mean")]
    Mean,
    #[strum(serialize = "sum")]
    Sum,
}

/// Resize interpolation mode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, EnumString, Display)]
pub enum ResizeMode {
    #[default]
    #[strum(serialize = "nearest")]
    Nearest,
    #[strum(serialize = "linear")]
    Linear,
    #[strum(serialize = "cubic")]
    Cubic,
}

/// Coordinate transformation mode for resize.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, EnumString, Display)]
pub enum CoordinateTransformMode {
    #[default]
    #[strum(serialize = "half_pixel")]
    HalfPixel,
    #[strum(serialize = "align_corners")]
    AlignCorners,
    #[strum(serialize = "asymmetric")]
    Asymmetric,
    #[strum(serialize = "pytorch_half_pixel")]
    PytorchHalfPixel,
    #[strum(serialize = "half_pixel_symmetric")]
    HalfPixelSymmetric,
    #[strum(serialize = "tf_crop_and_resize")]
    TfCropAndResize,
}

/// Nearest-neighbor rounding mode for resize.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, EnumString, Display)]
pub enum NearestMode {
    #[default]
    #[strum(serialize = "round_prefer_floor")]
    RoundPreferFloor,
    #[strum(serialize = "round_prefer_ceil")]
    RoundPreferCeil,
    #[strum(serialize = "floor")]
    Floor,
    #[strum(serialize = "ceil")]
    Ceil,
}

/// Depth-to-space rearrangement mode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, EnumString, Display)]
pub enum DepthToSpaceMode {
    /// DCR: depth-column-row (default, ONNX standard).
    #[default]
    #[strum(serialize = "DCR")]
    Dcr,
    /// CRD: column-row-depth (PyTorch pixel_shuffle order).
    #[strum(serialize = "CRD")]
    Crd,
}

/// Padding fill mode.
///
/// Determines how values outside the original tensor are filled when padding.
/// ONNX uses "edge"/"reflect"/"wrap"; Tinygrad uses "replicate"/"reflect"/"circular".
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, EnumString, Display)]
pub enum PadMode {
    /// Fill with a constant value (default: 0).
    #[default]
    #[strum(serialize = "constant")]
    Constant,
    /// Replicate boundary values. `[1,2,3]` pad(2,2) → `[1,1,1,2,3,3,3]`.
    #[strum(serialize = "edge", serialize = "replicate")]
    Replicate,
    /// Mirror without repeating boundary. `[1,2,3]` pad(2,2) → `[3,2,1,2,3,2,1]`.
    #[strum(serialize = "reflect")]
    Reflect,
    /// Wrap around (circular). `[1,2,3]` pad(2,2) → `[2,3,1,2,3,1,2]`.
    #[strum(serialize = "wrap", serialize = "circular")]
    Circular,
}

/// GridSample interpolation mode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, EnumString, Display)]
pub enum GridSampleMode {
    #[default]
    #[strum(serialize = "linear", serialize = "bilinear")]
    Linear,
    #[strum(serialize = "nearest")]
    Nearest,
    #[strum(serialize = "cubic", serialize = "bicubic")]
    Cubic,
}

/// GridSample padding mode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, EnumString, Display)]
pub enum GridSamplePaddingMode {
    #[default]
    #[strum(serialize = "zeros")]
    Zeros,
    #[strum(serialize = "border")]
    Border,
    #[strum(serialize = "reflection")]
    Reflection,
}

/// Aspect ratio policy for resize.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, EnumString, Display)]
pub enum AspectRatioPolicy {
    #[default]
    #[strum(serialize = "stretch")]
    Stretch,
    #[strum(serialize = "not_larger")]
    NotLarger,
    #[strum(serialize = "not_smaller")]
    NotSmaller,
}

// =========================================================================
// Higher-level building blocks (ONNX-style wrappers)
// =========================================================================

#[bon]
impl Tensor {
    /// Negative log-likelihood loss.
    ///
    /// `self` is `[N, C, ...]` log-probabilities, `target` is `[N, ...]` class indices
    /// (dtype `i64`). Gathers the log-prob at the target class and negates it.
    ///
    /// Supports optional per-class `weight`, `ignore_index` to mask out a class,
    /// and `reduction` (default `Mean`).
    ///
    /// # Examples
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// # use ndarray::array;
    /// let logprobs = Tensor::from_ndarray(&array![[-0.5f32, -1.0, -2.0]]);
    /// let target = Tensor::from_slice([0i64]);
    /// let mut loss = logprobs.nll_loss().target(&target).call().unwrap();
    /// loss.realize().unwrap();
    /// let val = loss.as_vec::<f32>().unwrap();
    /// // -(-0.5) = 0.5
    /// assert!((val[0] - 0.5).abs() < 1e-5);
    /// ```
    ///
    /// With sum reduction:
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// # use morok_tensor::nn::Reduction;
    /// # use ndarray::array;
    /// let logprobs = Tensor::from_ndarray(&array![[-0.5f32, -1.0], [-2.0, -0.3]]);
    /// let target = Tensor::from_slice([0i64, 1]);
    /// let mut loss = logprobs.nll_loss().target(&target).reduction(Reduction::Sum).call().unwrap();
    /// loss.realize().unwrap();
    /// let val = loss.as_vec::<f32>().unwrap();
    /// // sum of 0.5 + 0.3 = 0.8
    /// assert!((val[0] - 0.8).abs() < 1e-5);
    /// ```
    #[builder]
    pub fn nll_loss(
        &self,
        target: &Tensor,
        weight: Option<&Tensor>,
        ignore_index: Option<i64>,
        #[builder(default)] reduction: Reduction,
    ) -> Result<Tensor> {
        let ndim = self.ndim()?;
        snafu::ensure!(ndim >= 2, NdimMinimumSnafu { op: "nll_loss", min: 2_usize, actual: ndim });
        // Gather log-probs at target class, negate
        let nll = self.gather(1, &target.try_unsqueeze(1)?)?.try_squeeze(Some(1))?.try_neg()?;

        // Per-sample weight: weight[target] or ones
        let sample_weight = match weight {
            Some(w) => {
                let flat = target.try_reshape([-1])?;
                let sel = w.gather(0, &flat)?;
                let target_shape = morok_ir::shape::to_vec_isize(&target.shape()?).context(UOpSnafu)?;
                sel.try_reshape(&target_shape)?
            }
            None => {
                let shape = morok_ir::shape::to_vec_usize(&target.shape()?).context(UOpSnafu)?;
                Tensor::full(&shape, 1.0, self.uop().dtype())?
            }
        };

        // Mask out ignore_index
        let masked_weight = match ignore_index {
            Some(idx) => {
                let mask = target.try_ne(&Tensor::const_(idx as f64, target.uop().dtype()))?;
                sample_weight.try_mul(&mask.cast(sample_weight.uop().dtype())?)?
            }
            None => sample_weight,
        };

        let weighted_loss = nll.try_mul(&masked_weight)?;
        match reduction {
            Reduction::Mean => weighted_loss.sum(AxisSpec::All)?.try_div(&masked_weight.sum(AxisSpec::All)?),
            Reduction::Sum => weighted_loss.sum(AxisSpec::All),
            Reduction::None => Ok(weighted_loss),
        }
    }

    /// Dropout: randomly zeros elements during training, passes through in inference.
    ///
    /// Returns `(output, mask)` where mask is a boolean tensor (`true` = kept).
    /// In inference mode (`training=false`, the default), the output is identical
    /// to the input and the mask is all-true.
    ///
    /// **Note:** Training mode is not yet implemented (requires RNG); currently
    /// returns identity regardless of `training`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// # use ndarray::array;
    /// let x = Tensor::from_ndarray(&array![1.0f32, 2.0, 3.0]);
    /// let (mut out, mut mask) = x.dropout().p(0.5).call().unwrap();
    /// out.realize().unwrap();
    /// mask.realize().unwrap();
    /// // Default is inference mode: output == input
    /// assert_eq!(out.as_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
    /// assert_eq!(mask.as_vec::<bool>().unwrap(), vec![true, true, true]);
    /// ```
    #[builder]
    pub fn dropout(&self, p: f64, #[builder(default = false)] training: bool) -> Result<(Tensor, Tensor)> {
        snafu::ensure!(
            (0.0..=1.0).contains(&p),
            ParamRangeSnafu { op: "dropout", param: "p", value: p.to_string(), constraint: "0.0 <= p <= 1.0" }
        );
        let _ = p;
        let shape = morok_ir::shape::to_vec_usize(&self.shape()?).context(UOpSnafu)?;
        if !training {
            let mask = Tensor::full(&shape, true, DType::Bool)?;
            return Ok((self.clone(), mask));
        }
        // Training mode deferred (needs RNG: rand_like / Threefry)
        let mask = Tensor::full(&shape, true, DType::Bool)?;
        Ok((self.clone(), mask))
    }
    /// Convolution with ONNX-style parameters.
    ///
    /// Wraps the lower-level [`conv2d`](Tensor::conv2d) after resolving ONNX padding conventions
    /// (`auto_pad`, flat `pads`). Input shape is `[N, C, H, W, ...]` and weight
    /// shape is `[out_channels, in_channels/group, kH, kW, ...]`.
    ///
    /// # Examples
    ///
    /// Basic convolution with no padding:
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// # use ndarray::Array4;
    /// let x = Tensor::from_ndarray(&Array4::from_elem((1, 1, 5, 5), 1.0f32));
    /// let w = Tensor::from_ndarray(&Array4::from_elem((1, 1, 3, 3), 1.0f32));
    /// let mut y = x.conv().weight(&w).call().unwrap();
    /// y.realize().unwrap();
    /// let shape: Vec<_> = y.shape().unwrap().iter().map(|d| d.as_const().unwrap()).collect();
    /// assert_eq!(shape, [1, 1, 3, 3]);
    /// // Each output element sums a 3x3 window of ones = 9.0
    /// assert_eq!(y.as_vec::<f32>().unwrap(), vec![9.0; 9]);
    /// ```
    ///
    /// With explicit padding and strides:
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// # use ndarray::Array4;
    /// let x = Tensor::from_ndarray(&Array4::from_elem((1, 1, 5, 5), 1.0f32));
    /// let w = Tensor::from_ndarray(&Array4::from_elem((1, 1, 3, 3), 1.0f32));
    /// let mut y = x.conv().weight(&w).pads(&[1, 1, 1, 1]).strides(&[2, 2]).call().unwrap();
    /// y.realize().unwrap();
    /// let shape: Vec<_> = y.shape().unwrap().iter().map(|d| d.as_const().unwrap()).collect();
    /// assert_eq!(shape, [1, 1, 3, 3]);
    /// assert_eq!(y.as_vec::<f32>().unwrap(), vec![4.0, 6.0, 4.0, 6.0, 9.0, 6.0, 4.0, 6.0, 4.0]);
    /// ```
    #[builder]
    pub fn conv(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        #[builder(default)] auto_pad: AutoPad,
        #[builder(default = 1)] group: usize,
        kernel_shape: Option<&[usize]>,
        pads: Option<&[i64]>,
        strides: Option<&[i64]>,
        dilations: Option<&[i64]>,
    ) -> Result<Tensor> {
        let w_shape = weight.shape()?;
        let kernel: Vec<usize> = kernel_shape
            .map(|ks| ks.to_vec())
            .unwrap_or_else(|| w_shape[2..].iter().map(|s| s.as_const().unwrap()).collect());
        let n = kernel.len();
        let strides_u: Vec<usize> =
            strides.map(|s| s.iter().map(|&v| v as usize).collect()).unwrap_or_else(|| vec![1; n]);
        let dilations_u: Vec<usize> =
            dilations.map(|d| d.iter().map(|&v| v as usize).collect()).unwrap_or_else(|| vec![1; n]);
        let x_shape = self.shape()?;
        let input_spatial: Vec<SInt> = x_shape[2..].to_vec();
        let empty_pads: Vec<i64> = vec![];
        let padding =
            resolve_pool_pads(&input_spatial, pads.unwrap_or(&empty_pads), &kernel, &dilations_u, &strides_u, auto_pad);
        self.conv2d()
            .weight(weight)
            .maybe_bias(bias)
            .groups(group)
            .stride(&strides_u)
            .dilation(&dilations_u)
            .padding(&padding)
            .call()
    }

    /// Transposed convolution with ONNX-style parameters.
    ///
    /// Wraps [`conv_transpose2d`](Tensor::conv_transpose2d) after resolving ONNX padding conventions.
    /// Supports `output_shape` and `output_padding` for precise output size control.
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
    /// let mut y = x.conv_transpose().weight(&w).call().unwrap();
    /// y.realize().unwrap();
    /// let vals = y.as_vec::<f32>().unwrap();
    /// assert_eq!(vals.len(), 16); // 4x4 output
    /// assert_eq!(vals[5], 4.0); // center sees full overlap
    /// ```
    ///
    /// With stride (larger upsampling factor):
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// # use ndarray::Array4;
    /// let x = Tensor::from_ndarray(&Array4::from_elem((1, 1, 2, 2), 1.0f32));
    /// let w = Tensor::from_ndarray(&Array4::from_elem((1, 1, 3, 3), 1.0f32));
    /// let mut y = x.conv_transpose().weight(&w).strides(&[2, 2]).call().unwrap();
    /// y.realize().unwrap();
    /// let vals = y.as_vec::<f32>().unwrap();
    /// assert_eq!(vals.len(), 25); // 5x5 output
    /// ```
    #[builder]
    pub fn conv_transpose(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        #[builder(default)] auto_pad: AutoPad,
        #[builder(default = 1)] group: usize,
        kernel_shape: Option<&[usize]>,
        pads: Option<&[i64]>,
        output_shape: Option<&[i64]>,
        output_padding: Option<&[usize]>,
        strides: Option<&[i64]>,
        dilations: Option<&[i64]>,
    ) -> Result<Tensor> {
        let w_shape = weight.shape()?;
        let kernel: Vec<usize> = kernel_shape
            .map(|ks| ks.to_vec())
            .unwrap_or_else(|| w_shape[2..].iter().map(|s| s.as_const().unwrap()).collect());
        let n = kernel.len();
        let x_shape = self.shape()?;
        let input_spatial: Vec<SInt> = x_shape[2..].to_vec();
        let strides_u: Vec<usize> =
            strides.map(|s| s.iter().map(|&v| v as usize).collect()).unwrap_or_else(|| vec![1; n]);
        let dilations_u: Vec<usize> =
            dilations.map(|d| d.iter().map(|&v| v as usize).collect()).unwrap_or_else(|| vec![1; n]);
        let output_padding_u: Vec<usize> = output_padding.map(|op| op.to_vec()).unwrap_or_else(|| vec![0; n]);

        // 3-path padding resolution (matches Tinygrad's ConvTranspose)
        let mut pads_resolved: Option<Vec<isize>> = None;

        // ConvTranspose padding resolution requires concrete spatial dims.
        let input_spatial_c: Vec<usize> = input_spatial
            .iter()
            .map(|s| s.as_const().expect("conv_transpose requires concrete spatial dims"))
            .collect();

        // Path 1: output_shape provided → derive total pads, apply auto_pad
        if let Some(os) = output_shape {
            let total_pads: Vec<isize> = (0..n)
                .map(|i| {
                    (strides_u[i] * (input_spatial_c[i] - 1)
                        + output_padding_u[i]
                        + (kernel[i] - 1) * dilations_u[i]
                        + 1) as isize
                        - os[i] as isize
                })
                .collect();
            pads_resolved = Some(auto_pad_split(&total_pads, auto_pad));
        }

        // Path 2: no explicit pads → derive from default output_shape
        if pads_resolved.is_none() && pads.is_none_or(|p| p.is_empty()) {
            let default_out: Vec<usize> = (0..n).map(|i| input_spatial_c[i] * strides_u[i]).collect();
            let total_pads: Vec<isize> = (0..n)
                .map(|i| {
                    (strides_u[i] * (input_spatial_c[i] - 1)
                        + output_padding_u[i]
                        + (kernel[i] - 1) * dilations_u[i]
                        + 1) as isize
                        - default_out[i] as isize
                })
                .collect();
            pads_resolved =
                Some(if auto_pad != AutoPad::NotSet { auto_pad_split(&total_pads, auto_pad) } else { vec![0; n * 2] });
        }

        // Path 3: explicit pads provided
        let padding: Vec<(isize, isize)> = if let Some(flat) = pads_resolved {
            let half = flat.len() / 2;
            (0..half).map(|i| (flat[i], flat[i + half])).collect()
        } else {
            flat_pads_to_pairs(pads.unwrap())
        };

        self.conv_transpose2d()
            .weight(weight)
            .maybe_bias(bias)
            .groups(group)
            .stride(&strides_u)
            .dilation(&dilations_u)
            .padding(&padding)
            .output_padding(&output_padding_u)
            .call()
    }

    /// Average pooling with ONNX-style parameters.
    ///
    /// Wraps [`avg_pool2d`](Tensor::avg_pool2d) after resolving ONNX padding and stride conventions.
    /// Stride defaults to 1 (unlike [`avg_pool2d`](Tensor::avg_pool2d) which defaults to `kernel_size`).
    /// Input shape is `[N, C, H, W]`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// # use ndarray::Array4;
    /// let x = Tensor::from_ndarray(&Array4::from_elem((1, 1, 4, 4), 1.0f32));
    /// let mut y = x.avg_pool().kernel_shape(&[2, 2]).call().unwrap();
    /// y.realize().unwrap();
    /// let shape: Vec<_> = y.shape().unwrap().iter().map(|d| d.as_const().unwrap()).collect();
    /// assert_eq!(shape, [1, 1, 3, 3]);
    /// // Average of all-ones windows is 1.0
    /// assert!(y.as_vec::<f32>().unwrap().iter().all(|&v| (v - 1.0).abs() < 1e-6));
    /// ```
    ///
    /// With strides:
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// # use ndarray::Array4;
    /// let x = Tensor::from_ndarray(&Array4::from_elem((1, 1, 4, 4), 1.0f32));
    /// let mut y = x.avg_pool().kernel_shape(&[2, 2]).strides(&[2, 2]).call().unwrap();
    /// y.realize().unwrap();
    /// let shape: Vec<_> = y.shape().unwrap().iter().map(|d| d.as_const().unwrap()).collect();
    /// assert_eq!(shape, [1, 1, 2, 2]);
    /// assert_eq!(y.as_vec::<f32>().unwrap(), vec![1.0; 4]);
    /// ```
    #[builder]
    pub fn avg_pool(
        &self,
        kernel_shape: &[usize],
        #[builder(default)] auto_pad: AutoPad,
        #[builder(default = false)] ceil_mode: bool,
        #[builder(default = false)] count_include_pad: bool,
        pads: Option<&[i64]>,
        strides: Option<&[i64]>,
        dilations: Option<&[i64]>,
    ) -> Result<Tensor> {
        let n = kernel_shape.len();
        let strides_u: Vec<usize> =
            strides.map(|s| s.iter().map(|&v| v as usize).collect()).unwrap_or_else(|| vec![1; n]);
        let dilations_u: Vec<usize> =
            dilations.map(|d| d.iter().map(|&v| v as usize).collect()).unwrap_or_else(|| vec![1; n]);
        let x_shape = self.shape()?;
        let input_spatial: Vec<SInt> = x_shape[2..].to_vec();
        let empty_pads: Vec<i64> = vec![];
        let padding = resolve_pool_pads(
            &input_spatial,
            pads.unwrap_or(&empty_pads),
            kernel_shape,
            &dilations_u,
            &strides_u,
            auto_pad,
        );
        self.avg_pool2d()
            .kernel_size(kernel_shape)
            .stride(&strides_u)
            .dilation(&dilations_u)
            .padding(&padding)
            .ceil_mode(ceil_mode)
            .count_include_pad(count_include_pad)
            .call()
    }

    /// Lp norm pooling with ONNX-style parameters.
    ///
    /// Computes `(sum(|x|^p))^(1/p)` over each pooling window. Defaults to
    /// `p=2` (L2 pooling). Input shape is `[N, C, H, W]`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// # use ndarray::Array4;
    /// let x = Tensor::from_ndarray(&Array4::from_elem((1, 1, 4, 4), 1.0f32));
    /// let mut y = x.lp_pool().kernel_shape(&[2, 2]).call().unwrap();
    /// y.realize().unwrap();
    /// let shape: Vec<_> = y.shape().unwrap().iter().map(|d| d.as_const().unwrap()).collect();
    /// assert_eq!(shape, [1, 1, 3, 3]);
    /// // L2 pool of 2x2 window of ones = sqrt(4) = 2.0
    /// assert!((y.as_vec::<f32>().unwrap()[0] - 2.0).abs() < 1e-5);
    /// ```
    #[builder]
    pub fn lp_pool(
        &self,
        kernel_shape: &[usize],
        #[builder(default = 2)] p: usize,
        #[builder(default)] auto_pad: AutoPad,
        #[builder(default = false)] ceil_mode: bool,
        pads: Option<&[i64]>,
        strides: Option<&[i64]>,
        dilations: Option<&[i64]>,
    ) -> Result<Tensor> {
        snafu::ensure!(p >= 1, ParamRangeSnafu { op: "lp_pool", param: "p", value: p.to_string(), constraint: ">= 1" });
        let n_spatial = kernel_shape.len();
        let strides_u: Vec<usize> =
            strides.map(|s| s.iter().map(|&v| v as usize).collect()).unwrap_or_else(|| vec![1; n_spatial]);
        let dilations_u: Vec<usize> =
            dilations.map(|d| d.iter().map(|&v| v as usize).collect()).unwrap_or_else(|| vec![1; n_spatial]);
        let x_shape = self.shape()?;
        let input_spatial: Vec<SInt> = x_shape[2..].to_vec();
        let empty_pads: Vec<i64> = vec![];
        let padding = resolve_pool_pads(
            &input_spatial,
            pads.unwrap_or(&empty_pads),
            kernel_shape,
            &dilations_u,
            &strides_u,
            auto_pad,
        );

        let p_f = p as f64;
        let dtype = self.uop().dtype();
        let p_tensor = Tensor::const_(p_f, dtype.clone());
        let inv_p = Tensor::const_(1.0 / p_f, dtype);
        let x_abs_p = self.try_abs()?.try_pow(&p_tensor)?;

        // Pad, pool (create windows), then sum over kernel axes.
        // This computes sum(|x|^p) directly — correct for all padding/ceil modes
        // because padded zeros contribute 0 to the sum.
        let reg_pads = padding;
        let ceil_pads = if ceil_mode {
            pad::apply_ceil_mode(&reg_pads, &input_spatial, kernel_shape, &strides_u, &dilations_u)
        } else {
            reg_pads.clone()
        };
        let pads_to_use = if ceil_mode { &ceil_pads } else { &reg_pads };
        let mut padded = x_abs_p;
        if pads_to_use.iter().any(|&(b, e)| b != 0 || e != 0) {
            let n_batch = x_shape.len() - n_spatial;
            let mut full_pad: Vec<(isize, isize)> = vec![(0, 0); n_batch];
            full_pad.extend_from_slice(pads_to_use);
            padded = padded.try_pad(&full_pad)?;
        }
        let pooled = padded.pool(kernel_shape, &strides_u, &dilations_u)?;
        let reduce_axes: Vec<isize> = (0..n_spatial).map(|j| -(1 + j as isize)).collect();
        let sum_p = pooled.sum(crate::reduce::AxisSpec::Multiple(reduce_axes))?;
        sum_p.try_pow(&inv_p)
    }

    /// Rearrange depth data into spatial blocks (inverse of [`space_to_depth`](Tensor::space_to_depth)).
    ///
    /// Equivalent to PyTorch's `F.pixel_shuffle`. Reshapes a `[N, C, H, W]`
    /// tensor to `[N, C/(b*b), H*b, W*b]` where `b` is the blocksize.
    ///
    /// # Examples
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// # use ndarray::Array4;
    /// let x = Tensor::from_ndarray(&Array4::from_elem((1, 4, 1, 1), 1.0f32));
    /// let mut y = x.depth_to_space().blocksize(2).call().unwrap();
    /// y.realize().unwrap();
    /// let shape: Vec<_> = y.shape().unwrap().iter().map(|d| d.as_const().unwrap()).collect();
    /// assert_eq!(shape, [1, 1, 2, 2]);
    /// assert_eq!(y.as_vec::<f32>().unwrap(), vec![1.0; 4]);
    /// ```
    ///
    /// Using CRD mode (PyTorch pixel_shuffle order):
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// # use morok_tensor::nn::DepthToSpaceMode;
    /// # use ndarray::Array4;
    /// let x = Tensor::from_ndarray(&Array4::from_elem((1, 4, 1, 1), 1.0f32));
    /// let mut y = x.depth_to_space().blocksize(2).mode(DepthToSpaceMode::Crd).call().unwrap();
    /// y.realize().unwrap();
    /// assert_eq!(y.as_vec::<f32>().unwrap(), vec![1.0; 4]);
    /// ```
    #[builder]
    pub fn depth_to_space(&self, blocksize: usize, #[builder(default)] mode: DepthToSpaceMode) -> Result<Tensor> {
        let ndim = self.ndim()?;
        snafu::ensure!(ndim == 4, NdimExactSnafu { op: "depth_to_space", expected: 4_usize, actual: ndim });
        snafu::ensure!(
            blocksize > 0,
            ParamRangeSnafu {
                op: "depth_to_space",
                param: "blocksize",
                value: blocksize.to_string(),
                constraint: "> 0"
            }
        );
        let shape = self.shape()?;
        let (b, c, h, w) = (
            shape[0].as_const().unwrap(),
            shape[1].as_const().unwrap(),
            shape[2].as_const().unwrap(),
            shape[3].as_const().unwrap(),
        );
        let bs_sq = blocksize * blocksize;
        snafu::ensure!(
            c.is_multiple_of(bs_sq),
            DivisibilitySnafu {
                op: "depth_to_space",
                lhs_name: "channels",
                lhs: c,
                rhs_name: "blocksize^2",
                rhs: bs_sq
            }
        );
        let c_out = c / bs_sq;
        let result = if mode == DepthToSpaceMode::Crd {
            self.try_reshape([
                b as isize,
                c_out as isize,
                blocksize as isize,
                blocksize as isize,
                h as isize,
                w as isize,
            ])?
            .try_permute(&[0, 1, 4, 2, 5, 3])?
        } else {
            // DCR (default)
            self.try_reshape([
                b as isize,
                blocksize as isize,
                blocksize as isize,
                c_out as isize,
                h as isize,
                w as isize,
            ])?
            .try_permute(&[0, 3, 4, 1, 5, 2])?
        };
        result.try_reshape([b as isize, c_out as isize, (h * blocksize) as isize, (w * blocksize) as isize])
    }

    /// Rearrange spatial data into depth (inverse of [`depth_to_space`](Tensor::depth_to_space)).
    ///
    /// Reshapes a `[N, C, H, W]` tensor to `[N, C*b*b, H/b, W/b]` where `b`
    /// is the blocksize. Both `H` and `W` must be divisible by `blocksize`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// # use ndarray::Array4;
    /// let x = Tensor::from_ndarray(&Array4::from_elem((1, 1, 4, 4), 1.0f32));
    /// let mut y = x.space_to_depth(2).unwrap();
    /// y.realize().unwrap();
    /// let shape: Vec<_> = y.shape().unwrap().iter().map(|d| d.as_const().unwrap()).collect();
    /// assert_eq!(shape, [1, 4, 2, 2]);
    /// assert_eq!(y.as_vec::<f32>().unwrap(), vec![1.0; 16]);
    /// ```
    pub fn space_to_depth(&self, blocksize: usize) -> Result<Tensor> {
        let ndim = self.ndim()?;
        snafu::ensure!(ndim == 4, NdimExactSnafu { op: "space_to_depth", expected: 4_usize, actual: ndim });
        snafu::ensure!(
            blocksize > 0,
            ParamRangeSnafu {
                op: "space_to_depth",
                param: "blocksize",
                value: blocksize.to_string(),
                constraint: "> 0"
            }
        );
        let shape = self.shape()?;
        let (b, c, h, w) = (
            shape[0].as_const().unwrap(),
            shape[1].as_const().unwrap(),
            shape[2].as_const().unwrap(),
            shape[3].as_const().unwrap(),
        );
        snafu::ensure!(
            h.is_multiple_of(blocksize),
            DivisibilitySnafu {
                op: "space_to_depth",
                lhs_name: "height",
                lhs: h,
                rhs_name: "blocksize",
                rhs: blocksize
            }
        );
        snafu::ensure!(
            w.is_multiple_of(blocksize),
            DivisibilitySnafu {
                op: "space_to_depth",
                lhs_name: "width",
                lhs: w,
                rhs_name: "blocksize",
                rhs: blocksize
            }
        );
        self.try_reshape([
            b as isize,
            c as isize,
            (h / blocksize) as isize,
            blocksize as isize,
            (w / blocksize) as isize,
            blocksize as isize,
        ])?
        .try_permute(&[0, 3, 5, 1, 2, 4])?
        .try_reshape([
            b as isize,
            (c * blocksize * blocksize) as isize,
            (h / blocksize) as isize,
            (w / blocksize) as isize,
        ])
    }

    /// Max pooling with ONNX-style parameters.
    ///
    /// Always returns `(values, indices)` where indices are flattened positions
    /// (dtype `i64`). Wraps [`max_pool2d_with_indices`](Tensor::max_pool2d_with_indices) after resolving ONNX
    /// padding conventions.
    ///
    /// # Examples
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// # use ndarray::Array4;
    /// let x = Tensor::from_ndarray(&Array4::from_elem((1, 1, 4, 4), 1.0f32));
    /// let (vals, indices) = x.max_pool().kernel_shape(&[2, 2]).call().unwrap();
    /// let shape: Vec<_> = vals.shape().unwrap().iter().map(|d| d.as_const().unwrap()).collect();
    /// assert_eq!(shape, [1, 1, 3, 3]);
    /// ```
    ///
    /// With strides:
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// # use ndarray::Array4;
    /// let x = Tensor::from_ndarray(&Array4::from_elem((1, 1, 4, 4), 1.0f32));
    /// let (vals, _) = x.max_pool().kernel_shape(&[2, 2]).strides(&[2, 2]).call().unwrap();
    /// let shape: Vec<_> = vals.shape().unwrap().iter().map(|d| d.as_const().unwrap()).collect();
    /// assert_eq!(shape, [1, 1, 2, 2]);
    /// ```
    #[builder]
    pub fn max_pool(
        &self,
        kernel_shape: &[usize],
        #[builder(default)] auto_pad: AutoPad,
        #[builder(default = false)] ceil_mode: bool,
        #[builder(default = 0)] storage_order: usize,
        pads: Option<&[i64]>,
        strides: Option<&[i64]>,
        dilations: Option<&[i64]>,
    ) -> Result<(Tensor, Tensor)> {
        let n = kernel_shape.len();
        let strides_u: Vec<usize> =
            strides.map(|s| s.iter().map(|&v| v as usize).collect()).unwrap_or_else(|| vec![1; n]);
        let dilations_u: Vec<usize> =
            dilations.map(|d| d.iter().map(|&v| v as usize).collect()).unwrap_or_else(|| vec![1; n]);
        let x_shape = self.shape()?;
        let input_spatial: Vec<SInt> = x_shape[2..].to_vec();
        let empty_pads: Vec<i64> = vec![];
        let padding = resolve_pool_pads(
            &input_spatial,
            pads.unwrap_or(&empty_pads),
            kernel_shape,
            &dilations_u,
            &strides_u,
            auto_pad,
        );
        let (values, indices) = self
            .max_pool2d_with_indices()
            .kernel_size(kernel_shape)
            .stride(&strides_u)
            .dilation(&dilations_u)
            .padding(&padding)
            .ceil_mode(ceil_mode)
            .call()?;
        let indices = if storage_order == 1 {
            indices.try_transpose(-2, -1)?.cast(DType::Int64)?
        } else {
            indices.cast(DType::Int64)?
        };
        Ok((values, indices))
    }

    /// Local Response Normalization (LRN).
    ///
    /// Normalizes each element by dividing by a scaled sum of squares over a
    /// local neighborhood of `size` channels:
    /// `y = x / (bias + alpha * avg_pool(x^2, size))^beta`.
    ///
    /// Input must be 4-D `[N, C, H, W]`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// # use ndarray::Array4;
    /// let x = Tensor::from_ndarray(&Array4::from_elem((1, 3, 2, 2), 1.0f32));
    /// let y = x.lrn().size(3).call().unwrap();
    /// let shape: Vec<_> = y.shape().unwrap().iter().map(|d| d.as_const().unwrap()).collect();
    /// assert_eq!(shape, [1, 3, 2, 2]);
    /// ```
    ///
    /// Custom alpha, beta, and bias:
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// # use ndarray::Array4;
    /// let x = Tensor::from_ndarray(&Array4::from_elem((1, 3, 2, 2), 1.0f32));
    /// let y = x.lrn().size(3).alpha(0.001).beta(0.5).bias(2.0).call().unwrap();
    /// let shape: Vec<_> = y.shape().unwrap().iter().map(|d| d.as_const().unwrap()).collect();
    /// assert_eq!(shape, [1, 3, 2, 2]);
    /// ```
    #[builder]
    pub fn lrn(
        &self,
        size: usize,
        #[builder(default = 0.0001)] alpha: f64,
        #[builder(default = 0.75)] beta: f64,
        #[builder(default = 1.0)] bias: f64,
    ) -> Result<Tensor> {
        let ndim = self.ndim()?;
        snafu::ensure!(ndim == 4, NdimExactSnafu { op: "lrn", expected: 4_usize, actual: ndim });
        snafu::ensure!(
            size > 0,
            ParamRangeSnafu { op: "lrn", param: "size", value: size.to_string(), constraint: "> 0" }
        );
        let shape = self.shape()?;
        let (b, c, h, w) = (
            shape[0].as_const().unwrap(),
            shape[1].as_const().unwrap(),
            shape[2].as_const().unwrap(),
            shape[3].as_const().unwrap(),
        );
        let x_sq = self.square()?;
        let x_sq = x_sq.try_reshape([b as isize, 1, c as isize, (h * w) as isize])?;
        let pad_before = ((size - 1) / 2) as isize;
        let pad_after = (size / 2) as isize;
        let x_sq = x_sq.try_pad(&[(0, 0), (0, 0), (pad_before, pad_after), (0, 0)])?;
        let pooled = x_sq.avg_pool2d().kernel_size(&[size, 1]).stride(&[1, 1]).call()?;
        let pooled = pooled.try_reshape([b as isize, c as isize, h as isize, w as isize])?;
        let dtype = self.uop().dtype();
        let scale = pooled
            .try_mul(&Tensor::const_(alpha, dtype.clone()))?
            .try_add(&Tensor::const_(bias, dtype.clone()))?
            .try_pow(&Tensor::const_(beta, dtype))?;
        self.try_div(&scale)
    }
}

impl Tensor {
    /// Apply a sequence of layers to this tensor.
    pub fn sequential(&self, layers: &[&dyn Layer]) -> Result<Tensor> {
        let mut x = self.clone();
        for layer in layers {
            x = layer.forward(&x)?;
        }
        Ok(x)
    }
}
