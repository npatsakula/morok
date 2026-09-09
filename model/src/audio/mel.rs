//! Audio preprocessing: log-mel spectrograms in the graph
//! ([`Tensor::mel_spectrogram`]), the host only staging samples.
//!
//! A JIT has a fixed input capacity, and `center` reflect-pads a signal with
//! its own tail — which a row zero-extended to that capacity would hide, so
//! its last frames would read zeros where the reference reads the reflection.
//! Rows of one known length (Whisper's 30 s `pad_or_trim` windows) reflect in
//! the graph; rows of differing lengths (VAD windows sharing one GigaAM JIT)
//! are reflect-padded on the host by [`MelSpectrogram::frame_into`] and
//! transformed with `center = false`, so every row's frames match a
//! transform of that row alone.

use svod_macros::jit_wrapper;
use svod_tensor::Tensor;
use svod_tensor::nn::{self, MelLog, MelNorm, Window};

type Result<T> = svod_tensor::error::Result<T>;

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

/// Log-mel spectrogram extractor over one config: periodic Hann window
/// (`torch.hann_window(periodic=True)`, torchaudio's `MelSpectrogram`
/// default), power 2, `f_max = sample_rate / 2`.
#[derive(Clone)]
pub struct MelSpectrogram {
    config: MelConfig,
}

impl MelSpectrogram {
    /// `torch.log(x.clamp(1e-9, 1e9))` — the compression of
    /// [`forward_tensor`](Self::forward_tensor).
    pub const LOG: MelLog = MelLog::Ln { min: 1e-9, max: 1e9 };

    /// The config is validated when a graph is built from it.
    pub fn new(config: &MelConfig) -> Self {
        Self { config: config.clone() }
    }

    pub fn n_mels(&self) -> usize {
        self.config.n_mels
    }

    pub fn num_frames(&self, waveform_len: usize) -> usize {
        let signal_len = self.framed_len(waveform_len);
        let (n_fft, hop) = (self.config.n_fft, self.config.hop_length);
        if signal_len >= n_fft { (signal_len - n_fft) / hop + 1 } else { 0 }
    }

    /// Samples a host-framed row holds per window: the window plus its
    /// `n_fft / 2` reflect padding on each side under `center`.
    pub fn framed_len(&self, waveform_len: usize) -> usize {
        if self.config.center { waveform_len + 2 * (self.config.n_fft / 2) } else { waveform_len }
    }

    /// Host framing for rows of differing lengths: `waveform` reflect-padded
    /// (under `center`) into the head of `out`, the rest zeroed. A row of
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

    fn mel_power(&self, x: &Tensor, center: bool) -> Result<Tensor> {
        let (scale, norm) = self.config.mel_scale.filterbank();
        x.mel_spectrogram()
            .sample_rate(self.config.sample_rate)
            .n_fft(self.config.n_fft)
            .hop(self.config.hop_length)
            .win_length(self.config.win_length)
            .window(Window::Hann)
            .center(center)
            .n_mels(self.config.n_mels)
            .mel_scale(scale)
            .maybe_norm(norm)
            .call()
    }

    /// Mel power of raw `[B, L]` (or `[L]`) samples, `center` reflect-padding
    /// in the graph: `[B, n_mels, num_frames(L)]`.
    pub fn forward_power_tensor(&self, samples: &Tensor) -> Result<Tensor> {
        self.mel_power(samples, self.config.center)
    }

    /// Log-mel of a `[B, L']` batch of host-framed rows
    /// ([`frame_into`](Self::frame_into)): `[B, n_mels, (L' - n_fft) / hop + 1]`
    /// with the columns past each row's `frames` (`[B]` valid frame counts)
    /// zeroed, as an encoder's mel input expects them.
    pub fn forward_tensor(&self, framed: &Tensor, frames: &Tensor) -> Result<Tensor> {
        let mel = self.mel_power(framed, false)?.mel_log(Self::LOG)?;
        let valid = Tensor::sequence_mask(frames, mel.dim_const(-1)?)?.cast(mel.dtype()).try_unsqueeze(1)?;
        mel.try_mul(&valid)
    }
}

// Front-end JIT: host-framed rows + valid frame counts -> log-mel
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
