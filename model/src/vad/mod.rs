//! Silero V5 voice-activity detection.
//!
//! The forward pass mirrors the upstream Silero architecture: STFT via a
//! convolutional filterbank, four 1D conv blocks, an LSTM cell carrying
//! `(h, c)` between chunks, and a sigmoid head that produces a per-chunk
//! speech probability.
//!
//! The chunk-counter segmentation in [`VadInference::segment`] is a deliberate
//! simplification — it thresholds probabilities and merges nearby segments,
//! and is not equivalent to PyAnnote-style frame-classification VAD. Outputs
//! should not be expected to match a PyAnnote pipeline numerically.

mod error;

pub use error::{Error, Result};

extern crate self as morok_model;

use std::path::Path;

use morok_dtype::DType;
use morok_macros::jit_wrapper;
use morok_tensor::Tensor;
use morok_tensor::nn::PadMode;
use snafu::ResultExt;

use crate::state;

use error::{HubSnafu, StateSnafu, TensorSnafu};

const NUM_SAMPLES: usize = 512;
const CONTEXT_SIZE: usize = 64;
const STFT_PAD: usize = 64;
const CUTOFF: usize = 128 + 1;
const HIDDEN: usize = 128;
const CHUNK_LEN: usize = CONTEXT_SIZE + NUM_SAMPLES;

pub struct SileroVad {
    stft_conv_weight: Tensor,
    conv1_weight: Tensor,
    conv1_bias: Tensor,
    conv2_weight: Tensor,
    conv2_bias: Tensor,
    conv3_weight: Tensor,
    conv3_bias: Tensor,
    conv4_weight: Tensor,
    conv4_bias: Tensor,
    lstm_weight_ih: Tensor,
    lstm_weight_hh: Tensor,
    lstm_bias_ih: Tensor,
    lstm_bias_hh: Tensor,
    final_conv_weight: Tensor,
    final_conv_bias: Tensor,
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
            stft_conv_weight: get(&sd, "stft_conv.weight")?,
            conv1_weight: get(&sd, "conv1.weight")?,
            conv1_bias: get(&sd, "conv1.bias")?,
            conv2_weight: get(&sd, "conv2.weight")?,
            conv2_bias: get(&sd, "conv2.bias")?,
            conv3_weight: get(&sd, "conv3.weight")?,
            conv3_bias: get(&sd, "conv3.bias")?,
            conv4_weight: get(&sd, "conv4.weight")?,
            conv4_bias: get(&sd, "conv4.bias")?,
            lstm_weight_ih: get(&sd, "lstm_cell.weight_ih")?,
            lstm_weight_hh: get(&sd, "lstm_cell.weight_hh")?,
            lstm_bias_ih: get(&sd, "lstm_cell.bias_ih")?,
            lstm_bias_hh: get(&sd, "lstm_cell.bias_hh")?,
            final_conv_weight: get(&sd, "final_conv.weight")?,
            final_conv_bias: get(&sd, "final_conv.bias")?,
        })
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

        let x = x.conv2d().weight(&self.stft_conv_weight).stride(&[128]).call().context(TensorSnafu)?;

        let real = x.try_shrink([(0, 1), (0, CUTOFF), (0, 4)]).context(TensorSnafu)?;
        let imag = x.try_shrink([(0, 1), (CUTOFF, 258), (0, 4)]).context(TensorSnafu)?;
        let x = real
            .square()
            .context(TensorSnafu)?
            .try_add(&imag.square().context(TensorSnafu)?)
            .context(TensorSnafu)?
            .try_sqrt()
            .context(TensorSnafu)?;

        let x = conv1d(&x, &self.conv1_weight, &self.conv1_bias, 1, &[(1, 1)])?.relu().context(TensorSnafu)?;
        let x = conv1d(&x, &self.conv2_weight, &self.conv2_bias, 2, &[(1, 1)])?.relu().context(TensorSnafu)?;
        let x = conv1d(&x, &self.conv3_weight, &self.conv3_bias, 2, &[(1, 1)])?.relu().context(TensorSnafu)?;
        let x = conv1d(&x, &self.conv4_weight, &self.conv4_bias, 1, &[(1, 1)])?
            .relu()
            .context(TensorSnafu)?
            .try_squeeze(Some(-1))
            .context(TensorSnafu)?;

        let (new_h, new_c) = lstm_cell(
            &x,
            state_h,
            state_c,
            &self.lstm_weight_ih,
            &self.lstm_weight_hh,
            &self.lstm_bias_ih,
            &self.lstm_bias_hh,
        )?;

        let prob = new_h.try_unsqueeze(-1).context(TensorSnafu)?.relu().context(TensorSnafu)?;
        let prob = prob
            .conv2d()
            .weight(&self.final_conv_weight)
            .bias(&self.final_conv_bias)
            .call()
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

fn conv1d(x: &Tensor, weight: &Tensor, bias: &Tensor, stride: usize, padding: &[(isize, isize)]) -> Result<Tensor> {
    x.conv2d().weight(weight).bias(bias).stride(&[stride]).padding(padding).call().context(TensorSnafu)
}

fn lstm_cell(
    x: &Tensor,
    h: &Tensor,
    c: &Tensor,
    w_ih: &Tensor,
    w_hh: &Tensor,
    b_ih: &Tensor,
    b_hh: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let gates_x = x.linear().weight(w_ih).bias(b_ih).call().context(TensorSnafu)?;
    let gates_h = h.linear().weight(w_hh).bias(b_hh).call().context(TensorSnafu)?;
    let gates = gates_x.try_add(&gates_h).context(TensorSnafu)?;

    let parts = gates.split(&[HIDDEN, HIDDEN, HIDDEN, HIDDEN], 1).context(TensorSnafu)?;
    let i = parts[0].sigmoid().context(TensorSnafu)?;
    let f = parts[1].sigmoid().context(TensorSnafu)?;
    let g = parts[2].tanh().context(TensorSnafu)?;
    let o = parts[3].sigmoid().context(TensorSnafu)?;

    let new_c =
        f.try_mul(c).context(TensorSnafu)?.try_add(&i.try_mul(&g).context(TensorSnafu)?).context(TensorSnafu)?;
    let new_h = o.try_mul(&new_c.tanh().context(TensorSnafu)?).context(TensorSnafu)?;

    Ok((new_h, new_c))
}

fn get(sd: &state::StateDict, key: &str) -> Result<Tensor> {
    sd.get(key)
        .cloned()
        .ok_or_else(|| error::Error::State { source: Box::new(state::Error::MissingKey { key: key.to_string() }) })
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

pub struct VadInference {
    jit: SileroVadJit,
    h: Tensor,
    c: Tensor,
}

impl VadInference {
    pub fn new(vad: SileroVad) -> crate::jit::Result<Self> {
        let h = make_zero_state();
        let c = make_zero_state();
        let mut chunk_placeholder = Tensor::full(&[1, CHUNK_LEN], 0.0f32, DType::Float32)
            .map_err(|e| crate::jit::JitError::Tensor { source: Box::new(e) })?;
        chunk_placeholder.realize().map_err(|e| crate::jit::JitError::Tensor { source: Box::new(e) })?;

        let mut jit = SileroVadJit::new(vad);
        jit.prepare(&chunk_placeholder, &h, &c)?;

        Ok(Self { jit, h, c })
    }

    pub fn process_chunk(&mut self, chunk: &[f32]) -> crate::jit::Result<f32> {
        {
            let buf = self.jit.chunk_mut()?;
            let mut view =
                buf.as_array_mut::<f32>().map_err(|e| crate::jit::JitError::Build { source: Box::new(e) })?;
            let slice = view.as_slice_mut().expect("contiguous");
            slice[..chunk.len()].copy_from_slice(chunk);
        }

        write_state_into(&self.h, self.jit.state_h_mut()?)?;
        write_state_into(&self.c, self.jit.state_c_mut()?)?;

        self.jit.execute()?;

        let output = self.jit.output()?;
        let logits = output.as_array::<f32>().expect("output read");
        let logits_slice = logits.as_slice().expect("contiguous");
        let prob = logits_slice[0];

        let h_data = &logits_slice[1..1 + HIDDEN];
        let c_data = &logits_slice[1 + HIDDEN..1 + 2 * HIDDEN];

        {
            let mut view =
                self.h.array_view_mut::<f32>().map_err(|e| crate::jit::JitError::Build { source: Box::new(e) })?;
            view.as_slice_mut().expect("contiguous").copy_from_slice(h_data);
        }
        {
            let mut view =
                self.c.array_view_mut::<f32>().map_err(|e| crate::jit::JitError::Build { source: Box::new(e) })?;
            view.as_slice_mut().expect("contiguous").copy_from_slice(c_data);
        }

        Ok(prob)
    }

    pub fn segment(&mut self, waveform: &[f32], threshold: f32) -> Vec<(usize, usize)> {
        self.reset();

        let total = waveform.len();
        if total == 0 {
            return vec![];
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
            if let Ok(p) = self.process_chunk(chunk) {
                probs.push(p);
            }
        }

        threshold_segments(&probs, threshold, NUM_SAMPLES)
    }

    fn reset(&mut self) {
        if let Ok(mut view) = self.h.array_view_mut::<f32>()
            && let Some(s) = view.as_slice_mut()
        {
            s.fill(0.0);
        }
        if let Ok(mut view) = self.c.array_view_mut::<f32>()
            && let Some(s) = view.as_slice_mut()
        {
            s.fill(0.0);
        }
    }
}

fn make_zero_state() -> Tensor {
    let mut t = Tensor::full(&[1, HIDDEN], 0.0f32, DType::Float32).expect("state alloc");
    t.realize().expect("state realize");
    t
}

fn write_state_into(src: &Tensor, dst: &mut morok_device::Buffer) -> crate::jit::Result<()> {
    let src_buf = src.buffer().ok_or(crate::jit::JitError::NotPrepared)?;
    let mut data = vec![0u8; src_buf.size()];
    src_buf.copyout(&mut data).map_err(|e| crate::jit::JitError::Build { source: Box::new(e) })?;
    dst.copyin(&data).map_err(|e| crate::jit::JitError::Build { source: Box::new(e) })?;
    Ok(())
}

fn threshold_segments(probs: &[f32], threshold: f32, chunk_samples: usize) -> Vec<(usize, usize)> {
    let min_speech_chunks = 8usize;
    let min_silence_chunks = 4usize;

    let mut raw: Vec<(usize, usize)> = Vec::new();
    let mut speech_start: Option<usize> = None;
    let mut silence_count = 0usize;

    for (i, &p) in probs.iter().enumerate() {
        if p >= threshold {
            if speech_start.is_none() {
                speech_start = Some(i);
            }
            silence_count = 0;
        } else if let Some(start) = speech_start {
            silence_count += 1;
            if silence_count >= min_silence_chunks {
                let end = i - min_silence_chunks + 1;
                if end - start >= min_speech_chunks {
                    raw.push((start * chunk_samples, end * chunk_samples));
                }
                speech_start = None;
                silence_count = 0;
            }
        }
    }

    if let Some(start) = speech_start {
        let end = probs.len();
        if end - start >= min_speech_chunks {
            raw.push((start * chunk_samples, end * chunk_samples));
        }
    }

    let merge_gap_chunks = 8usize;
    let merge_gap_samples = merge_gap_chunks * chunk_samples;
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for seg in raw {
        if let Some(last) = merged.last_mut()
            && seg.0 - last.1 <= merge_gap_samples
        {
            last.1 = seg.1;
            continue;
        }
        merged.push(seg);
    }

    merged
}
