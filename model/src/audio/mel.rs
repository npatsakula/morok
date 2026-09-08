//! Audio preprocessing: log-mel spectrograms on the host (`realfft`, eager
//! `Vec<f32>`) and in the graph ([`Tensor::mel_spectrogram`]).
//!
//! Both paths share one filterbank ([`Tensor::mel_filterbank`]) and one f32
//! window, so they agree to FFT rounding. The graph path reads *host-framed*
//! windows — each reflect-padded (`center`) and zero-extended to a fixed
//! length, with its valid frame count alongside — which is what lets VAD
//! windows of different lengths share one fixed-shape JIT: the reflection
//! needs a window's own tail, which zero extension would hide.

use std::f32::consts::PI;
use std::sync::Arc;

use ndarray::ArrayViewMutD;
use realfft::{RealFftPlanner, RealToComplex};
use svod_dtype::default_device::default_device;
use svod_dtype::{DType, DeviceSpec};
use svod_macros::jit_wrapper;
use svod_tensor::Tensor;
use svod_tensor::nn::{self, MelLog, MelNorm, Window};

type Result<T> = svod_tensor::error::Result<T>;

/// Whether transcribers build the mel front-end in the graph: the default off
/// the CPU (where `realfft` wins), overridable with `SVOD_GRAPH_MEL=0/1`.
pub fn use_graph_mel() -> bool {
    match std::env::var("SVOD_GRAPH_MEL").as_deref() {
        Ok("0") => false,
        Ok("1") => true,
        _ => !matches!(default_device(), DeviceSpec::Cpu),
    }
}

/// Mel filterbank scale.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[derive(Default)]
pub enum MelScale {
    /// HTK mel scale: `2595·log10(1+f/700)`, peak height 1 (unnormalized).
    /// Matches torchaudio's `melscale_fbanks(slk_norm=None)`.
    #[default]
    Htk,
    /// librosa Slaney scale: linear below 1 kHz, log above; area-normalized
    /// triangles. Matches `librosa.filters.mel(norm='slaney')` and
    /// Whisper's pre-computed `mel_filters.npz`.
    Slaney,
}

impl MelScale {
    fn filterbank(self) -> (nn::MelScale, Option<MelNorm>) {
        match self {
            Self::Htk => (nn::MelScale::Htk, None),
            Self::Slaney => (nn::MelScale::Slaney, Some(MelNorm::Slaney)),
        }
    }
}

/// Configuration for mel spectrogram extraction.
#[derive(Clone, Debug)]
pub struct MelConfig {
    pub sample_rate: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub win_length: usize,
    pub n_mels: usize,
    pub center: bool,
    pub mel_scale: MelScale,
}

/// Log-mel spectrogram extractor: host (`forward_into`, `forward_power`) and
/// graph (`forward_tensor`, `forward_power_tensor`) paths over one config.
#[derive(Clone)]
pub struct MelSpectrogram {
    config: MelConfig,
    r2c: Arc<dyn RealToComplex<f32>>,
    /// Sparse filterbank rows: `(first_bin, weights)` per mel — each triangular
    /// filter covers only a handful of contiguous FFT bins, so the dense
    /// `[n_mels, n_bins]` matvec wastes ~40x. Built dense, sparsified once;
    /// ascending-bin accumulation keeps the matvec bit-identical to dense.
    mel_fb: Vec<(usize, Vec<f32>)>,
    window: Vec<f32>,
}

impl MelSpectrogram {
    /// `torch.log(x.clamp(1e-9, 1e9))` — the compression of
    /// [`forward_into`](Self::forward_into) and [`forward_tensor`](Self::forward_tensor).
    pub const LOG: MelLog = MelLog::Ln { min: 1e-9, max: 1e9 };

    pub fn new(config: &MelConfig) -> Result<Self> {
        let (scale, norm) = config.mel_scale.filterbank();
        let n_bins = config.n_fft / 2 + 1;
        let f_max = config.sample_rate as f64 / 2.0;
        let dense = Tensor::mel_filterbank(
            config.sample_rate,
            config.n_fft,
            config.n_mels,
            0.0,
            f_max,
            scale,
            norm,
            DType::Float32,
        )?
        .to_vec::<f32>()?;
        let mel_fb = dense
            .chunks(n_bins)
            .map(|row| {
                let first = row.iter().position(|&w| w != 0.0).unwrap_or(0);
                let last = row.iter().rposition(|&w| w != 0.0).map_or(first, |l| l + 1);
                (first, row[first..last].to_vec())
            })
            .collect();
        let r2c = RealFftPlanner::<f32>::new().plan_fft_forward(config.n_fft);
        let window = hann_window(config.n_fft, config.win_length);
        Ok(Self { config: config.clone(), r2c, mel_fb, window })
    }

    pub fn config(&self) -> &MelConfig {
        &self.config
    }

    pub fn n_mels(&self) -> usize {
        self.mel_fb.len()
    }

    pub fn num_frames(&self, waveform_len: usize) -> usize {
        let signal_len = self.framed_len(waveform_len);
        let (n_fft, hop) = (self.config.n_fft, self.config.hop_length);
        if signal_len >= n_fft { (signal_len - n_fft) / hop + 1 } else { 0 }
    }

    /// Samples the graph path reads per window: the window plus its reflect
    /// padding under `center`.
    pub fn framed_len(&self, waveform_len: usize) -> usize {
        if self.config.center { waveform_len + self.config.n_fft } else { waveform_len }
    }

    /// Host framing for the graph path: `waveform` reflect-padded (under
    /// `center`) into the head of `out`, the rest zeroed. A row of
    /// `(num_frames - 1) · hop + n_fft` samples already covers every frame,
    /// so padding past the end of a shorter `out` is dropped.
    pub fn frame_into(&self, waveform: &[f32], out: &mut [f32]) {
        let n = self.framed_len(waveform.len()).min(out.len());
        if self.config.center {
            out[..n].copy_from_slice(&reflect_pad(waveform, self.config.n_fft / 2)[..n]);
        } else {
            out[..n].copy_from_slice(&waveform[..n]);
        }
        out[n..].fill(0.0);
    }

    /// Graph mel power of host-framed windows: `framed` is `[B, L']` (or
    /// `[L']`) filled by [`frame_into`](Self::frame_into); the result is
    /// `[B, n_mels, (L' - n_fft) / hop + 1]`, the leading
    /// [`num_frames`](Self::num_frames) columns of each row being the frames
    /// the host path computes.
    pub fn forward_power_tensor(&self, framed: &Tensor) -> Result<Tensor> {
        let (scale, norm) = self.config.mel_scale.filterbank();
        framed
            .mel_spectrogram()
            .sample_rate(self.config.sample_rate)
            .n_fft(self.config.n_fft)
            .hop(self.config.hop_length)
            .window(Window::Custom(Tensor::from_slice(&self.window)))
            .center(false)
            .n_mels(self.config.n_mels)
            .mel_scale(scale)
            .maybe_norm(norm)
            .call()
    }

    /// Graph twin of [`forward_into`](Self::forward_into) over a `[B, L']`
    /// framed batch: log-mel `[B, n_mels, T]` with the columns past each row's
    /// `frames` (`[B]` valid frame counts) zeroed, as the JIT packers leave them.
    pub fn forward_tensor(&self, framed: &Tensor, frames: &Tensor) -> Result<Tensor> {
        let mel = self.forward_power_tensor(framed)?.mel_log(Self::LOG)?;
        let valid = Tensor::sequence_mask(frames, mel.dim_const(-1)?)?.cast(mel.dtype()).try_unsqueeze(1)?;
        mel.try_mul(&valid)
    }

    pub fn forward_into(&self, waveform: &[f32], out: &mut ArrayViewMutD<'_, f32>) {
        let n_fft = self.config.n_fft;
        let signal_owned = self.config.center.then(|| reflect_pad(waveform, n_fft / 2));
        let signal: &[f32] = signal_owned.as_deref().unwrap_or(waveform);

        let n_frames = self.num_frames(waveform.len());
        let n_bins = n_fft / 2 + 1;
        let n_mels = self.mel_fb.len();

        debug_assert!(
            {
                let shape = out.shape();
                shape.len() >= 2
                    && shape[shape.len() - 2] == n_mels
                    && shape[shape.len() - 1] == n_frames
                    && shape[..shape.len() - 2].iter().all(|&d| d == 1)
            },
            "forward_into: expected output trailing dims [.., {n_mels}, {n_frames}] with leading 1s, got {:?}",
            out.shape(),
        );

        let out_slice = out.as_slice_mut().expect("output must be contiguous");

        out_slice[..n_mels * n_frames].fill(0.0);

        let mut indata = self.r2c.make_input_vec();
        let mut outdata = self.r2c.make_output_vec();
        let mut power = vec![0.0f32; n_bins];

        for frame_idx in 0..n_frames {
            let start = frame_idx * self.config.hop_length;
            for i in 0..n_fft {
                indata[i] = signal[start + i] * self.window[i];
            }
            self.r2c.process(&mut indata, &mut outdata).expect("FFT failed");

            for (i, c) in outdata.iter().enumerate() {
                power[i] = c.re * c.re + c.im * c.im;
            }

            for (mel_idx, (first, weights)) in self.mel_fb.iter().enumerate() {
                let mut sum = 0.0f32;
                for (w, &p) in weights.iter().zip(&power[*first..]) {
                    sum += w * p;
                }
                out_slice[mel_idx * n_frames + frame_idx] = sum.clamp(1e-9, 1e9).ln();
            }
        }
    }

    /// Compute the raw mel power spectrogram (no log compression) into a flat
    /// `Vec<f32>` of length `n_mels * n_frames`. Used by Whisper which applies
    /// its own `log10` + clamp + normalize.
    pub fn forward_power(&self, waveform: &[f32]) -> Vec<f32> {
        let n_fft = self.config.n_fft;
        let signal_owned = self.config.center.then(|| reflect_pad(waveform, n_fft / 2));
        let signal: &[f32] = signal_owned.as_deref().unwrap_or(waveform);

        let n_frames_raw = self.num_frames(waveform.len());
        // Match torch.stft(...)[..., :-1]: drop the last frame.
        // torch.stft with center=True produces ceil(L/hop) frames but Whisper
        // drops the trailing one for exact N_FRAMES alignment.
        let n_frames = n_frames_raw.saturating_sub(1);
        let n_bins = n_fft / 2 + 1;
        let n_mels = self.mel_fb.len();

        let mut result = vec![0.0f32; n_mels * n_frames];

        let mut indata = self.r2c.make_input_vec();
        let mut outdata = self.r2c.make_output_vec();
        let mut power = vec![0.0f32; n_bins];

        for frame_idx in 0..n_frames_raw {
            let start = frame_idx * self.config.hop_length;
            for i in 0..n_fft {
                indata[i] = signal[start + i] * self.window[i];
            }
            self.r2c.process(&mut indata, &mut outdata).expect("FFT failed");

            for (i, c) in outdata.iter().enumerate() {
                power[i] = c.re * c.re + c.im * c.im;
            }

            // Skip the last frame (matching torch.stft[..., :-1])
            if frame_idx >= n_frames {
                continue;
            }

            for (mel_idx, (first, weights)) in self.mel_fb.iter().enumerate() {
                let mut sum = 0.0f32;
                for (w, &p) in weights.iter().zip(&power[*first..]) {
                    sum += w * p;
                }
                result[mel_idx * n_frames + frame_idx] = sum;
            }
        }

        result
    }
}

/// Periodic Hann window, matching `torch.hann_window(periodic=True)`, which is
/// torchaudio's default in `MelSpectrogram`. `realfft` handles only the FFT;
/// it does not provide STFT window builders.
pub(crate) fn hann_window(n_fft: usize, win_length: usize) -> Vec<f32> {
    let mut window = vec![0.0f32; n_fft];
    for (i, w) in window.iter_mut().enumerate().take(win_length) {
        *w = 0.5 * (1.0 - (2.0 * PI * i as f32 / win_length as f32).cos());
    }
    window
}

// Graph front-end JIT: host-framed windows + valid frame counts -> log-mel
// `[B, n_mels, T]`, whose device output feeds an encoder JIT's mel input.
jit_wrapper! {
    MelJit(MelSpectrogram) {
        framed: Tensor,
        frames: Tensor,

        build(framed, frames) {
            model.forward_tensor(framed, frames)
        }
    }
}

/// Reflect-pad a signal by `pad` samples on each side, mirroring PyTorch's
/// `Reflect1d`: the boundary element is not duplicated, and `pad` must be
/// strictly less than the signal length (single-bounce reflection only).
pub(crate) fn reflect_pad(signal: &[f32], pad: usize) -> Vec<f32> {
    let len = signal.len();
    assert!(
        pad < len,
        "reflect_pad requires pad ({pad}) < signal length ({len}); multi-bounce reflection is not supported",
    );

    let mut padded = Vec::with_capacity(len + 2 * pad);
    for i in (1..=pad).rev() {
        padded.push(signal[i]);
    }
    padded.extend_from_slice(signal);
    for i in 1..=pad {
        padded.push(signal[len - 1 - i]);
    }
    padded
}
