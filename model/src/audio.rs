//! Audio preprocessing: mel spectrogram, STFT, mel filterbanks.
//!
//! Runs eagerly on CPU using `realfft` — not through Morok's lazy tensor pipeline.

use std::f32::consts::PI;

use std::sync::Arc;

use ndarray::{Array2, ArrayViewMutD};
use realfft::{RealFftPlanner, RealToComplex};

/// Configuration for mel spectrogram extraction.
pub struct MelConfig {
    pub sample_rate: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub win_length: usize,
    pub n_mels: usize,
    pub center: bool,
}

/// CPU-based log-mel spectrogram extractor.
pub struct MelSpectrogram {
    r2c: Arc<dyn RealToComplex<f32>>,
    mel_fb: Array2<f32>,
    window: Vec<f32>,
    n_fft: usize,
    hop_length: usize,
    center: bool,
}

impl MelSpectrogram {
    pub fn new(config: &MelConfig) -> Self {
        let n_fft = config.n_fft;

        let mut planner = RealFftPlanner::<f32>::new();
        let r2c = planner.plan_fft_forward(n_fft);

        let mut window = vec![0.0f32; n_fft];
        for (i, w) in window.iter_mut().enumerate().take(config.win_length) {
            *w = 0.5 * (1.0 - (2.0 * PI * i as f32 / (config.win_length as f32 - 1.0)).cos());
        }

        let mel_fb = build_mel_filterbank(config.n_mels, n_fft, config.sample_rate as f32);

        Self { r2c, mel_fb, window, n_fft, hop_length: config.hop_length, center: config.center }
    }

    pub fn n_mels(&self) -> usize {
        self.mel_fb.nrows()
    }

    pub fn num_frames(&self, waveform_len: usize) -> usize {
        let signal_len = if self.center { waveform_len + self.n_fft } else { waveform_len };
        if signal_len >= self.n_fft { (signal_len - self.n_fft) / self.hop_length + 1 } else { 0 }
    }

    pub fn forward_into(&self, waveform: &[f32], out: &mut ArrayViewMutD<'_, f32>) {
        let n_fft = self.n_fft;
        let signal: &[f32];
        let signal_owned: Vec<f32>;

        if self.center {
            let pad = n_fft / 2;
            signal_owned = reflect_pad(waveform, pad);
            signal = &signal_owned;
        } else {
            signal = waveform;
        }

        let n_frames = if signal.len() >= n_fft { (signal.len() - n_fft) / self.hop_length + 1 } else { 0 };
        let n_bins = n_fft / 2 + 1;
        let n_mels = self.mel_fb.nrows();

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
            let start = frame_idx * self.hop_length;
            for i in 0..n_fft {
                indata[i] = signal[start + i] * self.window[i];
            }
            self.r2c.process(&mut indata, &mut outdata).expect("FFT failed");

            for (i, c) in outdata.iter().enumerate() {
                power[i] = c.re * c.re + c.im * c.im;
            }

            for mel_idx in 0..n_mels {
                let mut sum = 0.0f32;
                for (bin, &p) in power.iter().enumerate() {
                    sum += self.mel_fb[[mel_idx, bin]] * p;
                }
                out_slice[mel_idx * n_frames + frame_idx] = sum.clamp(1e-9, 1e9).ln();
            }
        }
    }
}

/// Build HTK mel filterbank matrix of shape `[n_mels, n_fft/2+1]`.
fn build_mel_filterbank(n_mels: usize, n_fft: usize, sample_rate: f32) -> Array2<f32> {
    let n_bins = n_fft / 2 + 1;
    let f_max = sample_rate / 2.0;

    let hz_to_mel = |f: f32| 2595.0 * (1.0 + f / 700.0).log10();
    let mel_to_hz = |m: f32| 700.0 * (10.0f32.powf(m / 2595.0) - 1.0);

    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(f_max);

    let mel_points: Vec<f32> =
        (0..n_mels + 2).map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32).collect();
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
    let bin_points: Vec<f32> = hz_points.iter().map(|&f| f * n_fft as f32 / sample_rate).collect();

    let mut fb = Array2::zeros((n_mels, n_bins));
    for i in 0..n_mels {
        let left = bin_points[i];
        let center = bin_points[i + 1];
        let right = bin_points[i + 2];

        for j in 0..n_bins {
            let freq = j as f32;
            if freq >= left && freq <= center && center > left {
                fb[[i, j]] = (freq - left) / (center - left);
            } else if freq > center && freq <= right && right > center {
                fb[[i, j]] = (right - freq) / (right - center);
            }
        }
    }
    fb
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
