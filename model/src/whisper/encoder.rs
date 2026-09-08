//! Audio encoder: Conv1d frontend + sinusoidal positional embeddings + transformer blocks.

use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::{Conv1d, Layer, LayerNorm, Linear, Module};

use crate::init::{Bias, conv1d, layer_norm, linear};
use crate::state::{scoped, scoped_index};

use super::attention::{MultiHeadAttention, padded_fa_sequence_len};
use super::blocks::{linear_forward, sinusoids};
use super::config::ModelDimensions;
use super::error::Result;

/// Encoder transformer block: self-attention + MLP, pre-norm.
#[derive(Clone, Module)]
pub struct EncoderBlock {
    pub attn: MultiHeadAttention,
    pub attn_ln: LayerNorm,
    #[module(key = "mlp.0")]
    pub mlp0: Linear,
    #[module(key = "mlp.2")]
    pub mlp2: Linear,
    pub mlp_ln: LayerNorm,
    pub n_state: usize,
}

impl EncoderBlock {
    pub fn empty(n_state: usize, n_head: usize) -> Self {
        Self::empty_dtype(n_state, n_head, DType::Float32)
    }

    pub fn empty_dtype(n_state: usize, n_head: usize, dtype: DType) -> Self {
        let mlp = n_state * 4;
        Self {
            attn: MultiHeadAttention::empty_dtype(n_state, n_head, dtype.clone()),
            attn_ln: layer_norm(n_state, dtype.clone()),
            mlp0: linear(n_state, mlp, Bias::FanIn, dtype.clone()),
            mlp2: linear(mlp, n_state, Bias::FanIn, dtype.clone()),
            mlp_ln: layer_norm(n_state, dtype),
            n_state,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_with_key_lens(x, None)
    }

    fn forward_with_key_lens(&self, x: &Tensor, key_lens: Option<&Tensor>) -> Result<Tensor> {
        // Self-attention (pre-norm)
        let h = scoped("attn_ln", || self.attn_ln.forward(x))?;
        let attn_out = scoped("attn", || self.attn.forward_with_key_lens(&h, None, None, key_lens))?;
        let x = x.try_add(&attn_out)?;

        // MLP (pre-norm)
        let h = scoped("mlp_ln", || self.mlp_ln.forward(&x))?;
        let h = linear_forward(&self.mlp0, &h)?.gelu_exact()?;
        let h = linear_forward(&self.mlp2, &h)?;
        Ok(x.try_add(&h)?)
    }
}

/// Whisper audio encoder: Conv1d × 2 + sinusoidal pos-emb + N × EncoderBlock + LayerNorm.
#[derive(Clone, Module)]
pub struct AudioEncoder {
    pub conv1: Conv1d,
    pub conv2: Conv1d,
    pub positional_embedding: Tensor,
    pub blocks: Vec<EncoderBlock>,
    pub ln_post: LayerNorm,
    pub n_state: usize,
    pub n_head: usize,
}

impl AudioEncoder {
    pub fn empty(dims: &ModelDimensions) -> Self {
        let n_state = dims.n_audio_state;
        let dtype = dims.dtype.clone();
        Self {
            conv1: conv1d(dims.n_mels, n_state, 3, Bias::FanIn, dtype.clone()).with_padding((1, 1)),
            conv2: conv1d(n_state, n_state, 3, Bias::FanIn, dtype.clone()).with_stride(2).with_padding((1, 1)),
            positional_embedding: sinusoids(dims.n_audio_ctx, n_state, 10_000.0).expect("sinusoidal embedding"),
            blocks: (0..dims.n_audio_layer)
                .map(|_| EncoderBlock::empty_dtype(n_state, dims.n_audio_head, dtype.clone()))
                .collect(),
            ln_post: layer_norm(n_state, dtype),
            n_state,
            n_head: dims.n_audio_head,
        }
    }

    /// Forward: mel `[B, n_mels, T]` → encoder features `[B, T/2, D]`.
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        // Cast input to the compute dtype (weights are dims.dtype; the host
        // feeds fp32 mel). Matches `model.py:48` weight.to(x.dtype) from the
        // other direction — we cast x to the weight dtype so the graph is uniform.
        let dtype = self.conv1.weight.dtype();
        let mel = mel.cast(dtype.clone());
        let x = scoped("conv1", || self.conv1.forward(&mel))?.gelu_exact()?;
        let x = scoped("conv2", || self.conv2.forward(&x))?.gelu_exact()?;

        // [B, D, T/2] → [B, T/2, D]
        let x = x.try_permute(&[0, 2, 1])?;

        // Add positional embedding [n_audio_ctx, D]
        let x = x.try_add(&self.positional_embedding)?.cast(dtype);

        let (batch, sequence) = (x.dim_const(0)?, x.dim_const(1)?);
        let padded_sequence = encoder_padded_sequence_len(&x.device(), sequence);
        let (mut x, key_lens) = match padded_sequence {
            Some(padded) => {
                let x = x.try_pad(&[(0, 0), (0, (padded - sequence) as isize), (0, 0)])?;
                let lens =
                    Tensor::full(&[batch], svod_ir::ConstValue::Int(sequence as i64), DType::Int32).to(x.device());
                (x, Some(lens))
            }
            None => (x, None),
        };

        // Transformer blocks
        for (index, block) in self.blocks.iter().enumerate() {
            x = scoped_index("blocks", index, || block.forward_with_key_lens(&x, key_lens.as_ref()))?;
        }
        if padded_sequence.is_some() {
            x = x.narrow(1, 0usize, sequence)?;
        }

        // Final LayerNorm + cast to fp32. The encoder output is consumed by the
        // host (copyout_prefix into Vec<f32>) and fed to the prefill/step JITs
        // which cast it back to the compute dtype. Keeping the output fp32 means
        // the host read path works regardless of compute dtype.
        Ok(scoped("ln_post", || self.ln_post.forward(&x))?.cast(DType::Float32))
    }
}

pub(crate) fn encoder_padded_sequence_len(device: &svod_dtype::DeviceSpec, sequence: usize) -> Option<usize> {
    svod_tk::flash_attention_supported(device)
        .then(|| padded_fa_sequence_len(false, sequence, sequence, sequence))
        .flatten()
}
