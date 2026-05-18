//! Silero V5 voice-activity detection.
//!
//! The forward pass mirrors the upstream Silero architecture: STFT via a
//! convolutional filterbank, four 1D conv blocks, an LSTM cell carrying
//! `(h, c)` between chunks, and a sigmoid head that produces a per-chunk
//! speech probability.
//!
//! [`VadInference::probs`] exposes the raw per-chunk probability array (one
//! entry per [`NUM_SAMPLES`] samples). [`VadInference::segment`] feeds those
//! into [`svod_arch::vad::chunks_from_probs`] to produce sample ranges
//! suitable for long-form ASR — see the `svod-arch::vad` module for
//! tunable knobs (min/max chunk duration, alignment, padding, etc.).

extern crate self as svod_model;

mod splitter;

pub use splitter::{SileroFixedWindowSplitter, SileroVadSplitter, SileroVadSplitterError};

#[cfg(test)]
pub(crate) use splitter::fixed_windows_from_probs;

use std::path::Path;

use snafu::{ResultExt, Snafu};
use svod_dtype::DType;
use svod_macros::jit_wrapper;
use svod_tensor::Tensor;
use svod_tensor::nn::{Conv1d, LSTMCell, Layer, PadMode};

use crate::init::fan_in_uniform;
use crate::state;

#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum Error {
    #[snafu(display("{source}"))]
    Tensor {
        #[snafu(source(from(svod_tensor::error::Error, Box::new)))]
        source: Box<svod_tensor::error::Error>,
    },
    #[snafu(display("{source}"))]
    State {
        #[snafu(source(from(crate::state::Error, Box::new)))]
        source: Box<crate::state::Error>,
    },
    #[snafu(display("hub error: {source}"))]
    Hub { source: hf_hub::api::sync::ApiError },
}

pub type Result<T> = std::result::Result<T, Error>;

/// Number of input samples covered by one VAD probability entry. Exposed so
/// callers can build [`svod_arch::vad::ChunkerOpts`] with the right
/// `samples_per_prob`.
pub const NUM_SAMPLES: usize = 512;
pub(crate) const CONTEXT_SIZE: usize = 64;
const STFT_PAD: usize = 64;
const CUTOFF: usize = 128 + 1;
pub(crate) const HIDDEN: usize = 128;
const CHUNK_LEN: usize = CONTEXT_SIZE + NUM_SAMPLES;

pub struct SileroVad {
    stft_conv: Conv1d,
    conv1: Conv1d,
    conv2: Conv1d,
    conv3: Conv1d,
    conv4: Conv1d,
    lstm: LSTMCell,
    final_conv: Conv1d,
}

impl SileroVad {
    pub fn from_hub() -> Result<Self> {
        let api = hf_hub::api::sync::Api::new().context(HubSnafu)?;
        let repo =
            api.repo(hf_hub::Repo::with_revision("vpermilp/silero-vad".into(), hf_hub::RepoType::Model, "main".into()));
        let path = repo.get("silero_vad_16k.safetensors").context(HubSnafu)?;
        Self::from_safetensors(&path)
    }

    pub fn from_safetensors(path: &Path) -> Result<Self> {
        let sd = state::load_safetensors(path).context(StateSnafu)?;
        Ok(Self {
            stft_conv: Conv1d::new(get(&sd, "stft_conv.weight")?, None).with_stride(128),
            conv1: Conv1d::new(get(&sd, "conv1.weight")?, Some(get(&sd, "conv1.bias")?)).with_padding((1, 1)),
            conv2: Conv1d::new(get(&sd, "conv2.weight")?, Some(get(&sd, "conv2.bias")?))
                .with_stride(2)
                .with_padding((1, 1)),
            conv3: Conv1d::new(get(&sd, "conv3.weight")?, Some(get(&sd, "conv3.bias")?))
                .with_stride(2)
                .with_padding((1, 1)),
            conv4: Conv1d::new(get(&sd, "conv4.weight")?, Some(get(&sd, "conv4.bias")?)).with_padding((1, 1)),
            lstm: LSTMCell::new(
                get(&sd, "lstm_cell.weight_ih")?,
                get(&sd, "lstm_cell.weight_hh")?,
                get(&sd, "lstm_cell.bias_ih")?,
                get(&sd, "lstm_cell.bias_hh")?,
            ),
            final_conv: Conv1d::new(get(&sd, "final_conv.weight")?, Some(get(&sd, "final_conv.bias")?)),
        })
    }

    /// Build with random weights matching the Silero V5 16 kHz layout. Strides
    /// and paddings mirror [`Self::from_safetensors`]; the lazy
    /// `fan_in_uniform` graphs keep the forward path from collapsing under
    /// const-folding so the JIT pipeline can be exercised without a checkpoint.
    pub fn with_random_weights() -> Self {
        let dt = DType::Float32;
        let mk_conv = |shape: [usize; 3], has_bias: bool, configure: fn(Conv1d) -> Conv1d| -> Conv1d {
            let fan_in = shape[1] * shape[2];
            let weight = fan_in_uniform(&shape, fan_in, dt.clone());
            let bias = has_bias.then(|| fan_in_uniform(&[shape[0]], fan_in, dt.clone()));
            configure(Conv1d::new(weight, bias))
        };

        Self {
            stft_conv: mk_conv([258, 1, 256], false, |c| c.with_stride(128)),
            conv1: mk_conv([128, 129, 3], true, |c| c.with_padding((1, 1))),
            conv2: mk_conv([64, 128, 3], true, |c| c.with_stride(2).with_padding((1, 1))),
            conv3: mk_conv([64, 64, 3], true, |c| c.with_stride(2).with_padding((1, 1))),
            conv4: mk_conv([128, 64, 3], true, |c| c.with_padding((1, 1))),
            lstm: LSTMCell::new(
                fan_in_uniform(&[4 * HIDDEN, HIDDEN], HIDDEN, dt.clone()),
                fan_in_uniform(&[4 * HIDDEN, HIDDEN], HIDDEN, dt.clone()),
                fan_in_uniform(&[4 * HIDDEN], HIDDEN, dt.clone()),
                fan_in_uniform(&[4 * HIDDEN], HIDDEN, dt.clone()),
            ),
            final_conv: mk_conv([1, 128, 1], true, |c| c),
        }
    }

    pub fn forward_chunk(&self, chunk: &Tensor, state_h: &Tensor, state_c: &Tensor) -> Result<Tensor> {
        let x = chunk
            .pad_with()
            .padding(&[(0, 0), (0, STFT_PAD as isize)])
            .mode(PadMode::Reflect)
            .call()
            .context(TensorSnafu)?
            .try_unsqueeze(1)
            .context(TensorSnafu)?;

        let x = self.stft_conv.forward(&x).context(TensorSnafu)?;

        let real = x.try_shrink([(0, 1), (0, CUTOFF), (0, 4)]).context(TensorSnafu)?;
        let imag = x.try_shrink([(0, 1), (CUTOFF, 258), (0, 4)]).context(TensorSnafu)?;
        let x = real
            .square()
            .context(TensorSnafu)?
            .try_add(&imag.square().context(TensorSnafu)?)
            .context(TensorSnafu)?
            .try_sqrt()
            .context(TensorSnafu)?;

        let x = self.conv1.forward(&x).context(TensorSnafu)?.relu().context(TensorSnafu)?;
        let x = self.conv2.forward(&x).context(TensorSnafu)?.relu().context(TensorSnafu)?;
        let x = self.conv3.forward(&x).context(TensorSnafu)?.relu().context(TensorSnafu)?;
        let x = self
            .conv4
            .forward(&x)
            .context(TensorSnafu)?
            .relu()
            .context(TensorSnafu)?
            .try_squeeze(Some(-1))
            .context(TensorSnafu)?;

        let (new_h, new_c) = self.lstm.step(&x, state_h, state_c).context(TensorSnafu)?;

        let prob = new_h.try_unsqueeze(-1).context(TensorSnafu)?.relu().context(TensorSnafu)?;
        let prob = self
            .final_conv
            .forward(&prob)
            .context(TensorSnafu)?
            .sigmoid()
            .context(TensorSnafu)?
            .try_squeeze(Some(-1))
            .context(TensorSnafu)?
            .mean_with()
            .axes(-1isize)
            .keepdim(true)
            .call()
            .context(TensorSnafu)?;

        Tensor::cat(&[&prob, &new_h, &new_c], 1).context(TensorSnafu)
    }
}

fn get(sd: &state::StateDict, key: &str) -> Result<Tensor> {
    sd.get(key)
        .cloned()
        .ok_or_else(|| Error::State { source: Box::new(state::Error::MissingKey { key: key.to_string() }) })
}

jit_wrapper! {
    SileroVadJit(SileroVad) {
        chunk: Tensor,
        state_h: Tensor,
        state_c: Tensor,

        build(chunk, state_h, state_c) {
            model.forward_chunk(chunk, state_h, state_c)
        }
    }
}

impl crate::jit::RecurrentJit for SileroVadJit {
    fn pack_state(&mut self, s: &crate::jit::LstmState) -> crate::jit::Result<()> {
        {
            let buf = self.state_h_mut()?;
            let mut view = buf.as_array_mut::<f32>().context(crate::jit::DeviceSnafu)?;
            view.as_slice_mut().expect("contiguous state_h").copy_from_slice(&s.h);
        }
        {
            let buf = self.state_c_mut()?;
            let mut view = buf.as_array_mut::<f32>().context(crate::jit::DeviceSnafu)?;
            view.as_slice_mut().expect("contiguous state_c").copy_from_slice(&s.c);
        }
        Ok(())
    }

    fn execute_step(&mut self) -> crate::jit::Result<()> {
        self.execute()
    }

    fn output_buffer(&self) -> crate::jit::Result<&svod_device::Buffer> {
        self.output()
    }
}

pub struct VadInference {
    inner: crate::jit::JitRecurrent<SileroVadJit>,
}

impl VadInference {
    pub fn new(vad: SileroVad) -> crate::jit::Result<Self> {
        use crate::jit::InputSpec;
        let mut jit = SileroVadJit::new(vad);
        jit.prepare(InputSpec::f32(&[1, CHUNK_LEN]), InputSpec::f32(&[1, HIDDEN]), InputSpec::f32(&[1, HIDDEN]))?;
        Ok(Self { inner: crate::jit::JitRecurrent::new(jit, crate::jit::LstmState::zeros(HIDDEN), 1)? })
    }

    pub fn process_chunk(&mut self, chunk: &[f32]) -> crate::jit::Result<f32> {
        let head = self.inner.step(|jit| {
            let buf = jit.chunk_mut()?;
            let mut view = buf.as_array_mut::<f32>().context(crate::jit::DeviceSnafu)?;
            view.as_slice_mut().expect("contiguous chunk")[..chunk.len()].copy_from_slice(chunk);
            Ok(())
        })?;
        Ok(head[0])
    }

    /// Run Silero V5 across the waveform and collect one speech probability
    /// per [`NUM_SAMPLES`]-sample window. The output length is
    /// `ceil(waveform.len() / NUM_SAMPLES)`; trailing entries cover the
    /// zero-padding past the waveform end.
    pub fn probs(&mut self, waveform: &[f32]) -> crate::jit::Result<Vec<f32>> {
        self.inner.reset();
        let total = waveform.len();
        if total == 0 {
            return Ok(Vec::new());
        }

        let pad_len = (NUM_SAMPLES - total % NUM_SAMPLES) % NUM_SAMPLES;
        let padded_len = CONTEXT_SIZE + total + pad_len;
        let mut padded = vec![0.0f32; padded_len];
        padded[CONTEXT_SIZE..CONTEXT_SIZE + total].copy_from_slice(waveform);

        let n_chunks = (total + pad_len) / NUM_SAMPLES;
        let mut probs: Vec<f32> = Vec::with_capacity(n_chunks);
        for i in 0..n_chunks {
            let start = i * NUM_SAMPLES;
            let chunk = &padded[start..start + CHUNK_LEN];
            probs.push(self.process_chunk(chunk)?);
        }
        Ok(probs)
    }

    /// Convenience wrapper around [`Self::probs`] +
    /// [`svod_arch::vad::chunks_from_probs`] with default chunker knobs and
    /// the given `threshold`. Errors from the JIT or chunker are swallowed —
    /// callers that need fault-visibility should drive `probs()` and
    /// `chunks_from_probs` directly.
    pub fn segment(&mut self, waveform: &[f32], threshold: f32) -> Vec<(usize, usize)> {
        let Ok(probs) = self.probs(waveform) else { return Vec::new() };
        let opts = svod_arch::vad::ChunkerOpts {
            threshold,
            samples_per_prob: NUM_SAMPLES,
            ..svod_arch::vad::ChunkerOpts::default()
        };
        svod_arch::vad::chunks_from_probs(&probs, &opts)
            .unwrap_or_default()
            .into_iter()
            .map(|c| (c.start_sample, c.end_sample))
            .collect()
    }
}
