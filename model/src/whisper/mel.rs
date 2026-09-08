//! Whisper mel spectrogram: the Slaney filterbank of Whisper's
//! `assets/mel_filters.npz`, the trailing STFT frame dropped, then
//! `log10` + clamp + normalize — on the host and in the graph.

use svod_macros::jit_wrapper;
use svod_tensor::Tensor;
use svod_tensor::nn::MelLog;

use crate::audio::{MelConfig, MelScale, MelSpectrogram};

use super::config::{HOP_LENGTH, N_FFT, N_FRAMES, N_SAMPLES, SAMPLE_RATE};

/// Whisper-specific mel spectrogram extractor.
#[derive(Clone)]
pub struct WhisperMel {
    inner: MelSpectrogram,
}

impl WhisperMel {
    pub fn new(n_mels: usize) -> svod_tensor::error::Result<Self> {
        let inner = MelSpectrogram::new(&MelConfig {
            sample_rate: SAMPLE_RATE,
            n_fft: N_FFT,
            hop_length: HOP_LENGTH,
            win_length: N_FFT,
            n_mels,
            center: true,
            mel_scale: MelScale::Slaney,
        })?;
        Ok(Self { inner })
    }

    pub fn n_mels(&self) -> usize {
        self.inner.n_mels()
    }

    pub fn num_frames(&self, waveform_len: usize) -> usize {
        self.inner.num_frames(waveform_len)
    }

    /// Samples per framed row the graph path reads: 30 s plus reflect padding.
    pub fn framed_len(&self) -> usize {
        self.inner.framed_len(N_SAMPLES)
    }

    /// `whisper.audio.pad_or_trim`: zero-pad or cut to 30 seconds.
    fn pad_or_trim(waveform: &[f32]) -> Vec<f32> {
        let mut audio = vec![0.0f32; N_SAMPLES];
        let copy_len = waveform.len().min(N_SAMPLES);
        audio[..copy_len].copy_from_slice(&waveform[..copy_len]);
        audio
    }

    /// Host framing for the graph path: pad-or-trim, then reflect-pad into
    /// `out` (`[framed_len]`).
    pub fn frame_into(&self, waveform: &[f32], out: &mut [f32]) {
        self.inner.frame_into(&Self::pad_or_trim(waveform), out);
    }

    /// Graph twin of [`compute`](Self::compute) over `[B, framed_len]` rows:
    /// `[B, n_mels, N_FRAMES]`. The trailing frame goes before the log so the
    /// per-signal maximum sees exactly the frames Whisper keeps.
    pub fn forward_tensor(&self, framed: &Tensor) -> svod_tensor::error::Result<Tensor> {
        self.inner.forward_power_tensor(framed)?.narrow(-1, 0_usize, N_FRAMES)?.mel_log(MelLog::Whisper)
    }

    /// Compute log-mel spectrogram matching `whisper.audio.log_mel_spectrogram`.
    /// Returns `[n_mels, n_frames]` row-major.
    ///
    /// Audio is pad-or-trimmed to 30 seconds (N_SAMPLES) before STFT, matching
    /// `whisper.audio.pad_or_trim` + `log_mel_spectrogram`.
    pub fn compute(&self, waveform: &[f32]) -> Vec<f32> {
        let power = self.inner.forward_power(&Self::pad_or_trim(waveform));
        if power.is_empty() {
            return power;
        }

        // log10(clamp(x, 1e-10))
        let mut log_spec: Vec<f32> = power.iter().map(|&p| p.clamp(1e-10, f32::MAX).log10()).collect();

        // max(x, x.max() - 8.0)
        let max_val = log_spec.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let clamp_floor = max_val - 8.0;
        for v in log_spec.iter_mut() {
            *v = v.max(clamp_floor);
        }

        // (x + 4.0) / 4.0
        for v in log_spec.iter_mut() {
            *v = (*v + 4.0) / 4.0;
        }

        log_spec
    }
}

// Graph front-end JIT: `[B, framed_len]` host-framed 30 s windows -> log-mel
// `[B, n_mels, N_FRAMES]`, copied on-device into the encoder JIT's mel input.
jit_wrapper! {
    WhisperMelJit(WhisperMel) {
        framed: Tensor,

        build(framed) {
            model.forward_tensor(framed)
        }
    }
}
