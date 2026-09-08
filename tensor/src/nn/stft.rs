//! Short-time Fourier transform and complex helpers.
//!
//! [`Tensor::stft`] and [`Tensor::istft`] are graph ops: the DFT basis, the
//! analysis window and the overlap-add normalization are all built from
//! `arange`/`cos`/`sin`, so a transform is a single convolution against a
//! constant kernel rather than a host-side FFT. That keeps an audio front-end
//! (mel filterbank, VAD, speech enhancement) inside one JIT graph.
//!
//! ## Conventions
//!
//! Everything follows `torch.stft` / `torch.istft`:
//!
//! - Input `[B, L]` (or `[L]`), output `[B, F, T, 2]` (or `[F, T, 2]`) with the
//!   trailing axis holding `(real, imag)` — the layout GTCRN's mask branch and
//!   Silero's magnitude branch both consume.
//! - `F = n_fft / 2 + 1` when `onesided` (the default), else `n_fft`.
//! - `T = (L' - n_fft) / hop + 1`, where `L'` is `L + n_fft` under
//!   `center` (reflect padding of `n_fft / 2` on both sides) and `L` otherwise.
//! - `window` is periodic (`torch.hann_window(periodic=true)`); a
//!   `win_length` below `n_fft` is zero-padded symmetrically, as torch does.
//! - `normalized` scales the forward transform by `1 / sqrt(n_fft)`.
//!
//! The complex helpers ([`magnitude`](Tensor::magnitude),
//! [`power`](Tensor::power), [`complex_mul`](Tensor::complex_mul), …) operate
//! on that same trailing-2 layout, so a spectrogram never has to be split into
//! two tensors by hand.

use std::f64::consts::TAU;

use bon::bon;
use snafu::ensure;
use svod_dtype::DType;
use svod_ir::SInt;

use crate::Tensor;
use crate::error::{FloatDTypeRequiredSnafu, NdimExactSnafu, NdimMinimumSnafu, ParamRangeSnafu, ShapeMismatchSnafu};

type Result<T> = crate::Result<T>;

/// Floor on the window-square overlap-add divisor in [`Tensor::istft`].
///
/// `istft` divides by `Σ_t w²(n - t·hop)`. Torch requires that sum to be
/// non-zero everywhere (the NOLA condition); where it is not — `hop` larger
/// than the window support, or a window with interior zeros — the frame is
/// unrecoverable. Rather than fail at graph-build time on a value that is only
/// known after realization, the divisor is clamped to this floor, which drives
/// those samples to zero instead of to infinity.
const NOLA_EPS: f64 = 1e-11;

/// Analysis window for [`Tensor::stft`] / [`Tensor::istft`].
///
/// The named windows are the periodic (`torch.*_window(periodic=true)`) forms
/// when built through [`Tensor::window`] with `periodic = true`; `Custom`
/// carries an explicit `[n]` tensor, which is used as is (a GTCRN-style
/// `hann_window(n).sqrt()`, say).
#[derive(Clone, Default)]
pub enum Window {
    /// All ones — no tapering.
    Rectangular,
    /// `0.5 - 0.5·cos(2πk/d)`.
    #[default]
    Hann,
    /// `0.54 - 0.46·cos(2πk/d)`.
    Hamming,
    /// Explicit `[win_length]` window.
    Custom(Tensor),
}

impl std::fmt::Debug for Window {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rectangular => f.write_str("Rectangular"),
            Self::Hann => f.write_str("Hann"),
            Self::Hamming => f.write_str("Hamming"),
            Self::Custom(_) => f.write_str("Custom(..)"),
        }
    }
}

impl Window {
    /// `(a0, a1)` of the cosine-sum form `a0 - a1·cos(2πk/d)`.
    const fn cosine_sum(&self) -> Option<(f64, f64)> {
        match self {
            Self::Hann => Some((0.5, 0.5)),
            Self::Hamming => Some((0.54, 0.46)),
            _ => None,
        }
    }
}

impl Tensor {
    /// Build a `[n]` window in the graph.
    ///
    /// `periodic` picks the DFT-even form (denominator `n`, what an STFT wants)
    /// over the symmetric one (denominator `n - 1`, what a filter design
    /// wants), matching `torch.hann_window`'s `periodic` flag.
    /// [`Window::Custom`] ignores `periodic` and only checks the length.
    ///
    /// # Examples
    ///
    /// ```
    /// # use svod_tensor::Tensor;
    /// # use svod_tensor::nn::Window;
    /// # use svod_dtype::DType;
    /// let w = Tensor::window(&Window::Hann, 4, true, DType::Float32).unwrap();
    /// assert_eq!(w.to_vec::<f32>().unwrap(), vec![0.0, 0.5, 1.0, 0.5]);
    /// ```
    #[track_caller]
    pub fn window(kind: &Window, n: usize, periodic: bool, dtype: DType) -> Result<Tensor> {
        origin_call!("window");
        ensure!(n > 0, ParamRangeSnafu { op: "window", param: "n", value: n.to_string(), constraint: "> 0" });
        match kind {
            Window::Custom(w) => {
                let ndim = w.ndim()?;
                ensure!(ndim == 1, NdimExactSnafu { op: "window", expected: 1_usize, actual: ndim });
                let len = w.dim_const(0)?;
                ensure!(
                    len == n,
                    ShapeMismatchSnafu { context: "window", expected: format!("[{n}]"), actual: format!("[{len}]") }
                );
                Ok(w.cast(dtype))
            }
            Window::Rectangular => Ok(Tensor::ones(&[n], dtype)),
            cosine => {
                let (a0, a1) = cosine.cosine_sum().expect("Rectangular and Custom handled above");
                let denom = if periodic || n == 1 { n } else { n - 1 };
                let k = Tensor::arange_f64(0.0, n as f64, 1.0, DType::Float32)?;
                let phase = k.try_mul(TAU / denom as f64)?;
                Ok(phase.cos()?.try_mul(-a1)?.try_add(a0)?.cast(dtype))
            }
        }
    }
}

// =========================================================================
// Kernel construction
// =========================================================================

/// `(cos, sin)` of `2πkn/n_fft` as `[n_bins, n_fft]` Float32 matrices.
///
/// `k·n` is reduced modulo `n_fft` in integers first, so the angle stays in
/// `[0, 2π)` and f32 `cos`/`sin` keep full precision even for a large `n_fft`
/// (the naive angle reaches `2π·n_fft/2` and loses ~4 decimal digits at 512).
fn dft_basis(n_fft: usize, n_bins: usize) -> Result<(Tensor, Tensor)> {
    let k = Tensor::arange(0, Some(n_bins as i64), None)?.try_reshape([n_bins as isize, 1])?;
    let n = Tensor::arange(0, Some(n_fft as i64), None)?.try_reshape([1, n_fft as isize])?;
    let phase = k.try_mul(&n)?.try_mod(n_fft as i64)?.cast(DType::Float32).try_mul(TAU / n_fft as f64)?;
    Ok((phase.cos()?, phase.sin()?))
}

/// `[2F, 1, n_fft]` analysis kernel: rows `0..F` are `w[n]·cos(2πkn/N)` and
/// rows `F..2F` are `-w[n]·sin(2πkn/N)`, so one `conv1d` with `stride = hop`
/// emits the framed real and imaginary parts stacked on the channel axis.
fn analysis_kernel(n_fft: usize, n_bins: usize, win: &Tensor, dtype: DType) -> Result<Tensor> {
    let (cos, sin) = dft_basis(n_fft, n_bins)?;
    let re = cos.try_mul(win)?;
    let im = sin.neg().try_mul(win)?;
    Ok(Tensor::cat(&[&re, &im], 0)?.try_reshape([2 * n_bins as isize, 1, n_fft as isize])?.cast(dtype))
}

/// `[2F, 1, n_fft]` synthesis kernel: the inverse DFT of a `[2F]` bin vector,
/// already multiplied by the synthesis window, so one `conv_transpose` with
/// `stride = hop` performs the frame IDFT *and* the overlap-add.
///
/// Under `onesided`, bins `1..F-1` stand in for their conjugate partners and
/// carry a factor of two; DC and (for even `n_fft`) Nyquist carry one, and
/// their imaginary parts are dropped exactly as `irfft` — and therefore
/// `torch.istft` — does.
fn synthesis_kernel(n_fft: usize, n_bins: usize, onesided: bool, win: &Tensor, dtype: DType) -> Result<Tensor> {
    let (cos, sin) = dft_basis(n_fft, n_bins)?;
    let inv = 1.0 / n_fft as f64;
    let (re, im) = if onesided {
        // `edge` is 1.0 on the bins without a conjugate partner, 0.0 elsewhere;
        // `2 - edge` and `2·(1 - edge)` are then the real/imaginary weights.
        let k = Tensor::arange(0, Some(n_bins as i64), None)?;
        let nyquist = if n_fft.is_multiple_of(2) { (n_fft / 2) as i64 } else { -1 };
        let edge = k.try_eq(0_i64)?.try_bitor(&k.try_eq(nyquist)?)?.cast(DType::Float32);
        let edge = edge.try_reshape([n_bins as isize, 1])?;
        (edge.try_mul(-1.0)?.try_add(2.0)?, edge.try_add(-1.0)?.try_mul(-2.0)?)
    } else {
        (Tensor::ones(&[1, 1], DType::Float32), Tensor::ones(&[1, 1], DType::Float32))
    };
    let re = cos.try_mul(&re)?.try_mul(inv)?.try_mul(win)?;
    let im = sin.try_mul(&im)?.try_mul(-inv)?.try_mul(win)?;
    Ok(Tensor::cat(&[&re, &im], 0)?.try_reshape([2 * n_bins as isize, 1, n_fft as isize])?.cast(dtype))
}

/// The window zero-padded to `n_fft`, centered as `torch.stft` centers it.
fn framed_window(kind: &Window, n_fft: usize, win_length: usize, dtype: DType) -> Result<Tensor> {
    let w = Tensor::window(kind, win_length, true, dtype)?;
    if win_length == n_fft {
        return Ok(w);
    }
    let left = ((n_fft - win_length) / 2) as isize;
    w.try_pad(&[(left, (n_fft - win_length) as isize - left)])
}

/// `(hop, win_length)` with the torch defaults applied and validated.
fn resolve(op: &'static str, n_fft: usize, hop: Option<usize>, win_length: Option<usize>) -> Result<(usize, usize)> {
    ensure!(n_fft > 0, ParamRangeSnafu { op, param: "n_fft", value: n_fft.to_string(), constraint: "> 0" });
    let hop = hop.unwrap_or((n_fft / 4).max(1));
    ensure!(hop > 0, ParamRangeSnafu { op, param: "hop", value: hop.to_string(), constraint: "> 0" });
    let win_length = win_length.unwrap_or(n_fft);
    ensure!(
        win_length > 0 && win_length <= n_fft,
        ParamRangeSnafu { op, param: "win_length", value: win_length.to_string(), constraint: "in 1..=n_fft" }
    );
    Ok((hop, win_length))
}

/// Reflect-pad the trailing axis only, leaving every leading axis — a symbolic
/// batch included — untouched. [`Tensor::pad_with`] needs every dimension to be
/// concrete, which an unbound `Variable` batch is not.
fn reflect_pad_last(x: &Tensor, pad: usize) -> Result<Tensor> {
    if pad == 0 {
        return Ok(x.clone());
    }
    let ndim = x.ndim()?;
    let len = x.dim_const(-1)?;
    ensure!(
        pad < len,
        ParamRangeSnafu { op: "stft", param: "n_fft / 2", value: pad.to_string(), constraint: "< signal length" }
    );
    let axis = ndim as isize - 1;
    let wing = |begin: usize, end: usize| -> Result<Tensor> {
        let mut ranges: Vec<Option<(SInt, SInt)>> = vec![None; ndim];
        ranges[ndim - 1] = Some((SInt::from(begin), SInt::from(end)));
        x.try_shrink(ranges)?.flip(&[axis])
    };
    let (left, right) = (wing(1, 1 + pad)?, wing(len - 1 - pad, len - 1)?);
    Tensor::cat(&[&left, x, &right], axis)
}

// =========================================================================
// stft / istft
// =========================================================================

#[bon]
impl Tensor {
    /// Short-time Fourier transform of a `[B, L]` (or `[L]`) real signal,
    /// returning `[B, F, T, 2]` (or `[F, T, 2]`) with `(real, imag)` trailing.
    ///
    /// Mirrors `torch.stft(..., return_complex=false)`. Implemented as a single
    /// `conv1d` against a `[2F, 1, n_fft]` windowed DFT kernel, so the whole
    /// transform stays in the graph and the batch axis may be symbolic.
    ///
    /// Defaults: `hop = n_fft / 4`, `win_length = n_fft`, periodic Hann window,
    /// `center`, `onesided`, no normalization.
    ///
    /// # Examples
    ///
    /// ```
    /// # use svod_tensor::Tensor;
    /// # use svod_tensor::nn::Window;
    /// let x = Tensor::from_slice(vec![0.25f32; 64]);
    /// let spec = x.stft().n_fft(16).hop(4).window(Window::Hann).call().unwrap();
    /// assert_eq!(spec.dims().unwrap(), vec![9, 17, 2]);
    /// ```
    #[builder]
    #[track_caller]
    pub fn stft(
        &self,
        n_fft: usize,
        hop: Option<usize>,
        win_length: Option<usize>,
        #[builder(default)] window: Window,
        #[builder(default = true)] center: bool,
        #[builder(default = true)] onesided: bool,
        #[builder(default = false)] normalized: bool,
    ) -> Result<Tensor> {
        origin_call!("stft");
        let (hop, win_length) = resolve("stft", n_fft, hop, win_length)?;
        let ndim = self.ndim()?;
        ensure!(ndim == 1 || ndim == 2, NdimExactSnafu { op: "stft", expected: 2_usize, actual: ndim });
        let dtype = self.dtype();
        ensure!(dtype.is_float(), FloatDTypeRequiredSnafu { op: "stft", arg: "input", dtype: dtype.clone() });

        let x = if ndim == 1 { self.try_unsqueeze(0)? } else { self.clone() };
        let x = if center { reflect_pad_last(&x, n_fft / 2)? } else { x };
        let len = x.dim_const(-1)?;
        ensure!(
            len >= n_fft,
            ParamRangeSnafu { op: "stft", param: "padded length", value: len.to_string(), constraint: ">= n_fft" }
        );

        let n_bins = if onesided { n_fft / 2 + 1 } else { n_fft };
        let win = framed_window(&window, n_fft, win_length, dtype.clone())?;
        let kernel = analysis_kernel(n_fft, n_bins, &win, dtype)?;

        // [B, 1, L] -> [B, 2F, T] -> [B, 2, F, T] -> [B, F, T, 2].
        let y = x.try_unsqueeze(1)?.conv1d().weight(&kernel).stride(hop).call()?;
        let y = if normalized { y.try_mul(1.0 / (n_fft as f64).sqrt())? } else { y };
        let out = y.unflatten(1, &[2, n_bins as isize])?.try_permute(&[0, 2, 3, 1])?;
        if ndim == 1 { out.try_squeeze(Some(0)) } else { Ok(out) }
    }

    /// Inverse short-time Fourier transform of a `[B, F, T, 2]` (or `[F, T, 2]`)
    /// spectrogram, returning `[B, L]` (or `[L]`).
    ///
    /// Mirrors `torch.istft`: each frame is inverse-transformed, multiplied by
    /// the synthesis window, overlap-added, and divided by the overlap-add of
    /// the squared window. Both steps are `conv_transpose` against constant
    /// kernels, so this is two convolutions rather than a host-side loop.
    ///
    /// The parameters must match the [`stft`](Tensor::stft) that produced the
    /// input. Without `length`, the result is `(T - 1) · hop` samples under
    /// `center` (the analysis padding is trimmed) and `(T - 1) · hop + n_fft`
    /// otherwise; with `length` it is trimmed or zero-padded to exactly that.
    ///
    /// # NOLA
    ///
    /// Reconstruction is exact only where the window satisfies NOLA — the
    /// overlap-add of `window²` is non-zero at every sample. `hann` at 75%
    /// overlap does; a `hop` wider than `win_length` does not. Samples where
    /// the divisor falls below `1e-11` are returned as zero rather than as
    /// infinity, so a NOLA violation shows up as silence, not as NaN.
    #[builder]
    #[track_caller]
    pub fn istft(
        &self,
        n_fft: usize,
        hop: Option<usize>,
        win_length: Option<usize>,
        #[builder(default)] window: Window,
        #[builder(default = true)] center: bool,
        #[builder(default = true)] onesided: bool,
        #[builder(default = false)] normalized: bool,
        length: Option<usize>,
    ) -> Result<Tensor> {
        origin_call!("istft");
        let (hop, win_length) = resolve("istft", n_fft, hop, win_length)?;
        let ndim = self.ndim()?;
        ensure!(ndim == 3 || ndim == 4, NdimExactSnafu { op: "istft", expected: 4_usize, actual: ndim });
        let dtype = self.dtype();
        ensure!(dtype.is_float(), FloatDTypeRequiredSnafu { op: "istft", arg: "input", dtype: dtype.clone() });

        let x = if ndim == 3 { self.try_unsqueeze(0)? } else { self.clone() };
        let shape = x.shape()?;
        let n_bins = x.dim_const(1)?;
        let expected = if onesided { n_fft / 2 + 1 } else { n_fft };
        let frames = x.dim_const(2)?;
        let pair = x.dim_const(3)?;
        ensure!(
            n_bins == expected && pair == 2,
            ShapeMismatchSnafu {
                context: "istft",
                expected: format!("[.., {expected}, T, 2]"),
                actual: format!("[.., {n_bins}, {frames}, {pair}]"),
            }
        );
        ensure!(
            frames > 0,
            ParamRangeSnafu { op: "istft", param: "frames", value: frames.to_string(), constraint: "> 0" }
        );

        let win = framed_window(&window, n_fft, win_length, dtype.clone())?;
        let kernel = synthesis_kernel(n_fft, n_bins, onesided, &win, dtype.clone())?;

        // [B, F, T, 2] -> [B, 2, F, T] -> [B, 2F, T]; the batch stays symbolic.
        let z = x.try_permute(&[0, 3, 1, 2])?.try_reshape([
            shape[0].clone(),
            SInt::from(2 * n_bins),
            SInt::from(frames),
        ])?;
        let z = if normalized { z.try_mul((n_fft as f64).sqrt())? } else { z };
        let ola = z.conv_transpose2d().weight(&kernel).stride(&[hop]).call()?;

        // Same overlap-add applied to window², giving the per-sample divisor.
        let wsq = win.square().try_reshape([1, 1, n_fft as isize])?;
        let ones = Tensor::ones(&[1, 1, frames], dtype);
        let norm = ones.conv_transpose2d().weight(&wsq).stride(&[hop]).call()?;
        let y = ola.try_div(&norm.maximum(NOLA_EPS)?)?.try_squeeze(Some(1))?;

        let out_len = (frames - 1) * hop + n_fft;
        let start = if center { n_fft / 2 } else { 0 };
        let keep = length.unwrap_or(out_len - 2 * start);
        let take = keep.min(out_len - start);
        let y = y.narrow(-1, start, take)?;
        let y = if take < keep { y.try_pad(&[(0, 0), (0, (keep - take) as isize)])? } else { y };
        if ndim == 3 { y.try_squeeze(Some(0)) } else { Ok(y) }
    }
}

// =========================================================================
// Complex helpers on the trailing-2 layout
// =========================================================================

impl Tensor {
    /// `(real, imag)` of a `[..., 2]` complex tensor, each `[...]`.
    fn complex_parts(&self, op: &'static str) -> Result<(Tensor, Tensor)> {
        let ndim = self.ndim()?;
        ensure!(ndim >= 1, NdimMinimumSnafu { op, min: 1_usize, actual: ndim });
        let last = self.dim_const(-1)?;
        ensure!(
            last == 2,
            ShapeMismatchSnafu { context: op, expected: "[.., 2]".to_string(), actual: format!("[.., {last}]") }
        );
        Ok((
            self.narrow(-1, 0_usize, 1_usize)?.try_squeeze(Some(-1))?,
            self.narrow(-1, 1_usize, 1_usize)?.try_squeeze(Some(-1))?,
        ))
    }

    /// Real part of a `[..., 2]` complex tensor.
    #[track_caller]
    pub fn complex_real(&self) -> Result<Tensor> {
        origin_call!("complex_real");
        Ok(self.complex_parts("complex_real")?.0)
    }

    /// Imaginary part of a `[..., 2]` complex tensor.
    #[track_caller]
    pub fn complex_imag(&self) -> Result<Tensor> {
        origin_call!("complex_imag");
        Ok(self.complex_parts("complex_imag")?.1)
    }

    /// `re² + im²` of a `[..., 2]` complex tensor — the power spectrum, without
    /// the square root a magnitude would pay for.
    #[track_caller]
    pub fn power(&self) -> Result<Tensor> {
        origin_call!("power");
        let (re, im) = self.complex_parts("power")?;
        re.square().try_add(im.square())
    }

    /// `sqrt(re² + im² + eps)` of a `[..., 2]` complex tensor.
    ///
    /// `eps` guards the gradient at the origin; pass `0.0` for the plain
    /// modulus (or call [`complex_abs`](Tensor::complex_abs)).
    ///
    /// # Examples
    ///
    /// ```
    /// # use svod_tensor::Tensor;
    /// let z = Tensor::from_slice([3.0f32, 4.0]).try_reshape([1, 2]).unwrap();
    /// assert_eq!(z.magnitude(0.0).unwrap().to_vec::<f32>().unwrap(), vec![5.0]);
    /// ```
    #[track_caller]
    pub fn magnitude(&self, eps: f64) -> Result<Tensor> {
        origin_call!("magnitude");
        let p = self.power()?;
        if eps == 0.0 { p.try_sqrt() } else { p.try_add(eps)?.try_sqrt() }
    }

    /// Modulus of a `[..., 2]` complex tensor — [`magnitude`](Tensor::magnitude)
    /// with no epsilon.
    #[track_caller]
    pub fn complex_abs(&self) -> Result<Tensor> {
        origin_call!("complex_abs");
        self.magnitude(0.0)
    }

    /// Elementwise complex product of two `[..., 2]` tensors:
    /// `(a + bi)(c + di) = (ac - bd) + (ad + bc)i`.
    ///
    /// This is the complex ratio mask GTCRN applies to its input spectrogram.
    #[track_caller]
    pub fn complex_mul(&self, other: &Tensor) -> Result<Tensor> {
        origin_call!("complex_mul");
        let (a, b) = self.complex_parts("complex_mul")?;
        let (c, d) = other.complex_parts("complex_mul")?;
        let re = a.try_mul(&c)?.try_sub(&b.try_mul(&d)?)?;
        let im = a.try_mul(&d)?.try_add(&b.try_mul(&c)?)?;
        Tensor::stack(&[&re, &im], -1)
    }

    /// `[..., 2]` complex tensor from magnitude and phase:
    /// `mag·(cos φ + i·sin φ)`.
    #[track_caller]
    pub fn complex_from_polar(mag: &Tensor, phase: &Tensor) -> Result<Tensor> {
        origin_call!("complex_from_polar");
        let re = mag.try_mul(&phase.cos()?)?;
        let im = mag.try_mul(&phase.sin()?)?;
        Tensor::stack(&[&re, &im], -1)
    }
}
