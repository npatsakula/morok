//! Neural network operations: convolution, pooling, normalization.

mod conv;
mod grid_sample;
mod norm;
pub mod pad;
mod pool;
mod resize;
mod rnn;

pub use rnn::RnnOutput;

pub use pad::{auto_pad_split, flat_pads_to_pairs, resolve_pool_pads};

use bon::bon;
use morok_dtype::DType;
use snafu::ResultExt;

use crate::Tensor;
use crate::error::UOpSnafu;
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

impl Tensor {
    /// Negative log-likelihood loss.
    ///
    /// `self` is `[N, C, ...]` log-probs, `target` is `[N, ...]` class indices.
    /// Matches Tinygrad's `nll_loss` (tensor.py:3391-3413).
    pub fn nll_loss(
        &self,
        target: &Tensor,
        weight: Option<&Tensor>,
        ignore_index: Option<i64>,
        reduction: Reduction,
    ) -> Result<Tensor> {
        // Gather log-probs at target class, negate
        let nll = self.gather(1, &target.try_unsqueeze(1)?)?.try_squeeze(Some(1))?.try_neg()?;

        // Per-sample weight: weight[target] or ones
        let sample_weight = match weight {
            Some(w) => {
                let flat = target.try_reshape(&[-1])?;
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

    /// Dropout: zeros random elements during training, passes through in inference.
    ///
    /// Returns `(output, mask)` where mask is a boolean tensor (true = kept).
    /// Training mode is deferred until RNG infrastructure is available.
    pub fn dropout(&self, _p: f64, training: bool) -> Result<(Tensor, Tensor)> {
        let shape = morok_ir::shape::to_vec_usize(&self.shape()?).context(UOpSnafu)?;
        if !training {
            let mask = Tensor::full(&shape, true, DType::Bool)?;
            return Ok((self.clone(), mask));
        }
        // Training mode deferred (needs RNG: rand_like / Threefry)
        let mask = Tensor::full(&shape, true, DType::Bool)?;
        Ok((self.clone(), mask))
    }
}

// =========================================================================
// Higher-level building blocks (ONNX-style wrappers)
// =========================================================================

#[bon]
impl Tensor {
    /// Convolution with ONNX-style parameters. Wraps `conv2d`.
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
        let input_spatial: Vec<usize> = x_shape[2..].iter().map(|s| s.as_const().unwrap()).collect();
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

    /// Transposed convolution with ONNX-style parameters. Wraps `conv_transpose2d`.
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
        let input_spatial: Vec<usize> = x_shape[2..].iter().map(|s| s.as_const().unwrap()).collect();
        let strides_u: Vec<usize> =
            strides.map(|s| s.iter().map(|&v| v as usize).collect()).unwrap_or_else(|| vec![1; n]);
        let dilations_u: Vec<usize> =
            dilations.map(|d| d.iter().map(|&v| v as usize).collect()).unwrap_or_else(|| vec![1; n]);
        let output_padding_u: Vec<usize> = output_padding.map(|op| op.to_vec()).unwrap_or_else(|| vec![0; n]);

        // 3-path padding resolution (matches Tinygrad's ConvTranspose)
        let mut pads_resolved: Option<Vec<isize>> = None;

        // Path 1: output_shape provided → derive total pads, apply auto_pad
        if let Some(os) = output_shape {
            let total_pads: Vec<isize> = (0..n)
                .map(|i| {
                    (strides_u[i] * (input_spatial[i] - 1) + output_padding_u[i] + (kernel[i] - 1) * dilations_u[i] + 1)
                        as isize
                        - os[i] as isize
                })
                .collect();
            pads_resolved = Some(auto_pad_split(&total_pads, auto_pad));
        }

        // Path 2: no explicit pads → derive from default output_shape
        if pads_resolved.is_none() && pads.is_none_or(|p| p.is_empty()) {
            let default_out: Vec<usize> = (0..n).map(|i| input_spatial[i] * strides_u[i]).collect();
            let total_pads: Vec<isize> = (0..n)
                .map(|i| {
                    (strides_u[i] * (input_spatial[i] - 1) + output_padding_u[i] + (kernel[i] - 1) * dilations_u[i] + 1)
                        as isize
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

    /// Average pooling with ONNX-style parameters. Wraps `avg_pool2d`.
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
        let input_spatial: Vec<usize> = x_shape[2..].iter().map(|s| s.as_const().unwrap()).collect();
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

    /// Max pooling with ONNX-style parameters. Always returns `(values, indices)`. Wraps `max_pool2d_with_indices`.
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
        let input_spatial: Vec<usize> = x_shape[2..].iter().map(|s| s.as_const().unwrap()).collect();
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
}
