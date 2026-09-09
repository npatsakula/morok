//! Kaldi-compatible log-mel fbank for FireRedVAD, matching
//! `kaldi-native-fbank` with the upstream options (16 kHz, 25 ms / 10 ms,
//! `snip_edges`, `dither = 0`, 80 bins): per-frame DC removal, per-frame
//! pre-emphasis 0.97, Povey window, power spectrum over a 512-point FFT,
//! triangular mel bins on `1127·ln(1 + f/700)` between 20 Hz and Nyquist,
//! natural log floored at `f32::EPSILON` — in the graph, on
//! [`Tensor::stft_padded`].
//!
//! Everything before the power spectrum is linear in the frame, so Kaldi's
//! per-frame chain folds onto a signal-level STFT:
//!
//! - the Povey window's leading weight is zero, so per-frame pre-emphasis,
//!   whose first sample is special-cased, equals signal-level
//!   `x[n] - 0.97·x[n - 1]` under the window;
//! - removing a frame's mean `m` before pre-emphasis subtracts the constant
//!   `(1 - 0.97)·m` from every emphasized sample, and
//!   `DFT(w·(y - c)) = DFT(w·y) - c·DFT(w)`: the means come from a box
//!   convolution at the frame stride, `0.03·DFT(w)` is a host constant.
//!
//! `snip_edges` is `center = false` with Kaldi's frame count, which is the
//! convolution's: `1 + (L - 400) / 160`.

use svod_dtype::DType;
use svod_macros::jit_wrapper;
use svod_tensor::nn::Window;
use svod_tensor::{PrepareConfig, Tensor};

use super::{FRAME_LENGTH, FRAME_SHIFT, N_MELS};
use crate::jit::InputSpec;

const N_FFT: usize = 512; // next power of two >= FRAME_LENGTH
const LOW_FREQ: f64 = 20.0;
const SAMPLE_RATE: f64 = 16_000.0;
/// Kaldi reads 16-bit PCM without normalization; Svod waveforms are `[-1, 1]`.
const INT16_SCALE: f64 = 32_768.0;
const PREEMPH: f64 = 0.97;

/// The fbank graph: `[B, L]` samples in `[-1, 1]` to `[B, T, N_MELS]`
/// pre-CMVN log-mel rows, `T = 1 + (L - FRAME_LENGTH) / FRAME_SHIFT`.
#[derive(Clone)]
pub struct KaldiFbank {
    /// Kaldi's symmetric Povey window, zero-padded to `N_FFT` — the analysis
    /// window in full, so the frames start where Kaldi's do rather than
    /// centered in the FFT as torch places a short window.
    window: Vec<f32>,
}

impl KaldiFbank {
    pub fn new() -> svod_tensor::error::Result<Self> {
        let mut window = Tensor::window(&Window::Povey, FRAME_LENGTH, false, DType::Float32)?.to_vec::<f32>()?;
        window.resize(N_FFT, 0.0);
        Ok(Self { window })
    }

    /// `snip_edges` frame count: only complete 25 ms windows produce a frame.
    pub fn num_frames(n_samples: usize) -> usize {
        if n_samples < FRAME_LENGTH { 0 } else { 1 + (n_samples - FRAME_LENGTH) / FRAME_SHIFT }
    }

    /// Samples `frames` frames read: the window is zero past `FRAME_LENGTH`,
    /// but the transform frames `N_FFT` samples.
    pub fn samples(frames: usize) -> usize {
        (frames - 1) * FRAME_SHIFT + N_FFT
    }

    /// `[B, L]` -> `[B, T, N_MELS]`, `L` covering [`samples`](Self::samples)`(T)`.
    pub fn forward_tensor(&self, samples: &Tensor) -> svod_tensor::error::Result<Tensor> {
        let len = samples.dim_const(-1)?;
        let previous = samples.try_pad(&[(0, 0), (1, 0)])?.narrow(-1, 0_usize, len)?;
        let emphasized = samples.try_sub(&previous.try_mul(PREEMPH)?)?.try_mul(INT16_SCALE)?.contiguous();
        let (spec, _, frames) = emphasized
            .stft_padded()
            .n_fft(N_FFT)
            .hop(FRAME_SHIFT)
            .window(Window::Custom(Tensor::from_slice(self.window.clone())))
            .center(false)
            .call()?;
        let (bins, padded_frames) = (spec.dim_const(1)?, spec.dim_const(2)?);

        // Frame means at the frame stride, padded to the STFT's frame count.
        let box_kernel = Tensor::from_slice(vec![(INT16_SCALE / FRAME_LENGTH as f64) as f32; FRAME_LENGTH])
            .try_reshape([1, 1, FRAME_LENGTH as isize])?;
        let mean = samples.try_unsqueeze(1)?.conv1d().weight(&box_kernel).stride(FRAME_SHIFT).call()?;
        let mean = mean
            .narrow(-1, 0_usize, frames)?
            .try_pad(&[(0, 0), (0, 0), (0, (padded_frames - frames) as isize)])?
            .try_unsqueeze(-1)?;
        let dc = Tensor::from_slice(self.window_dft(bins)).try_reshape([bins as isize, 1, 2])?;
        let power = spec.try_sub(&mean.try_mul(&dc)?)?.power()?;

        let banks = Tensor::from_slice(kaldi_mel_banks(bins)).try_reshape([bins as isize, N_MELS as isize])?;
        let mel = power.try_transpose(-1, -2)?.matmul(&banks)?.maximum(f32::EPSILON as f64)?.try_log()?;
        mel.narrow(1, 0_usize, frames)
    }

    /// `(1 - PREEMPH)·DFT(window)` as `[bins, 2]` `(re, im)` rows, zero past
    /// the one-sided bins — the analysis kernel's own sign convention.
    fn window_dft(&self, bins: usize) -> Vec<f32> {
        let mut table = vec![0f32; bins * 2];
        for k in 0..N_FFT / 2 + 1 {
            let (re, im) = (0..N_FFT).fold((0.0, 0.0), |(re, im), n| {
                let angle = std::f64::consts::TAU * ((k * n) % N_FFT) as f64 / N_FFT as f64;
                let w = f64::from(self.window[n]);
                (re + w * angle.cos(), im - w * angle.sin())
            });
            table[2 * k] = ((1.0 - PREEMPH) * re) as f32;
            table[2 * k + 1] = ((1.0 - PREEMPH) * im) as f32;
        }
        table
    }
}

/// Kaldi `MelBanks`: triangles on the `1127·ln(1 + f/700)` mel axis over the
/// first `N_FFT/2` FFT bins (Nyquist bin excluded, as in Kaldi), 20 Hz to
/// Nyquist, as a `[bins, N_MELS]` table the power spectrum contracts against
/// (zero rows past the one-sided bins).
fn kaldi_mel_banks(bins: usize) -> Vec<f32> {
    let mel = |f: f64| 1127.0 * (1.0 + f / 700.0).ln();
    let fft_bin_width = SAMPLE_RATE / N_FFT as f64;
    let (mel_low, mel_high) = (mel(LOW_FREQ), mel(SAMPLE_RATE / 2.0));
    let delta = (mel_high - mel_low) / (N_MELS + 1) as f64;

    let mut table = vec![0f32; bins * N_MELS];
    for m in 0..N_MELS {
        let (left, center, right) =
            (mel_low + m as f64 * delta, mel_low + (m + 1) as f64 * delta, mel_low + (m + 2) as f64 * delta);
        for i in 0..N_FFT / 2 {
            let mel_f = mel(i as f64 * fft_bin_width);
            if mel_f > left && mel_f < right {
                let w = if mel_f <= center { (mel_f - left) / delta } else { (right - mel_f) / delta };
                table[i * N_MELS + m] = w as f32;
            }
        }
    }
    table
}

jit_wrapper! {
    KaldiFbankJit(KaldiFbank) {
        samples: Tensor,

        build(samples) {
            model.forward_tensor(samples)
        }
    }
}

/// Host driver over a fixed-shape [`KaldiFbankJit`]: `batch` windows of
/// `capacity` frames per execute, frames being independent under
/// `snip_edges`. The output buffer is shaped for a device copy into a model
/// JIT's `[batch, capacity, N_MELS]` input, so features never round-trip
/// through the host on that path.
pub struct FireRedFbank {
    jit: KaldiFbankJit,
    /// `[batch, samples(capacity)]` host staging for the device-local input:
    /// one `copyin` per execute.
    staging: Vec<f32>,
    batch: usize,
    capacity: usize,
}

impl FireRedFbank {
    /// Prepare the JIT for `batch` windows of `capacity` frames per execute.
    pub fn new(batch: usize, capacity: usize) -> crate::jit::Result<Self> {
        assert!(batch >= 1 && capacity >= 1, "fbank batch and capacity must be >= 1");
        let mut jit = KaldiFbankJit::new(KaldiFbank::new()?);
        let row = KaldiFbank::samples(capacity);
        jit.prepare_with_config(InputSpec::f32(&[batch, row]).device_local(), &PrepareConfig::device_local())?;
        Ok(Self { jit, staging: vec![0.0; batch * row], batch, capacity })
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn num_frames(&self, n_samples: usize) -> usize {
        KaldiFbank::num_frames(n_samples)
    }

    /// One execute: row `i` holds the `capacity` frames of `waveform` from
    /// frame `first_frames[i]` on (frames outside the waveform read zeros),
    /// rows past `first_frames.len()` are silence. Returns the
    /// `[batch, capacity, N_MELS]` output buffer.
    pub fn forward_windows(
        &mut self,
        waveform: &[f32],
        first_frames: &[isize],
    ) -> crate::jit::Result<&svod_device::Buffer> {
        assert!(first_frames.len() <= self.batch, "more windows than the fbank batch");
        let row_len = KaldiFbank::samples(self.capacity);
        self.staging.fill(0.0);
        let len = waveform.len() as isize;
        for (row, &first) in self.staging.chunks_mut(row_len).zip(first_frames) {
            let start = first * FRAME_SHIFT as isize;
            let (lo, hi) = (start.clamp(0, len), (start + row_len as isize).clamp(0, len));
            if lo < hi {
                row[(lo - start) as usize..][..(hi - lo) as usize].copy_from_slice(&waveform[lo as usize..hi as usize]);
            }
        }
        self.jit.samples_mut()?.copyin(bytemuck::cast_slice(&self.staging))?;
        self.jit.execute()?;
        self.jit.output()
    }

    /// Extract `[num_frames * N_MELS]` row-major log-mel features from a
    /// `[-1, 1]`-scale waveform (pre-CMVN — normalization happens inside the
    /// model graph) — the host-side form, `batch · capacity` frames per
    /// execute.
    pub fn forward(&mut self, waveform: &[f32]) -> crate::jit::Result<Vec<f32>> {
        let mut feat = vec![0.0f32; KaldiFbank::num_frames(waveform.len()) * N_MELS];
        let block = self.batch * self.capacity;
        for (k, rows) in feat.chunks_mut(block * N_MELS).enumerate() {
            let windows = (rows.len() / N_MELS).div_ceil(self.capacity);
            let firsts: Vec<isize> = (0..windows).map(|i| ((k * self.batch + i) * self.capacity) as isize).collect();
            self.forward_windows(waveform, &firsts)?.copyout_prefix(bytemuck::cast_slice_mut(rows))?;
        }
        Ok(feat)
    }
}
