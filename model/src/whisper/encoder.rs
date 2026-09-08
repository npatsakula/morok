//! Audio encoder: Conv1d frontend + sinusoidal positional embeddings + transformer blocks.

use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::init::fan_in_uniform;
use crate::state::{self, HasStateDict, StateDict, get_tensor, prefixed, scoped, scoped_index};

use super::attention::{MultiHeadAttention, padded_fa_sequence_len};
use super::blocks::{Conv1dWeights, LayerNormWeights, linear_with_bias, sinusoids};
use super::config::ModelDimensions;
use super::error::Result;

/// Encoder transformer block: self-attention + MLP, pre-norm.
#[derive(Clone)]
pub struct EncoderBlock {
    pub attn: MultiHeadAttention,
    pub attn_ln: LayerNormWeights,
    pub mlp0_w: Tensor,
    pub mlp0_b: Tensor,
    pub mlp1_w: Tensor,
    pub mlp1_b: Tensor,
    pub mlp_ln: LayerNormWeights,
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
            attn_ln: LayerNormWeights::empty_dtype(n_state, dtype.clone()),
            mlp0_w: fan_in_uniform(&[mlp, n_state], n_state, dtype.clone()),
            mlp0_b: fan_in_uniform(&[mlp], n_state, dtype.clone()),
            mlp1_w: fan_in_uniform(&[n_state, mlp], mlp, dtype.clone()),
            mlp1_b: fan_in_uniform(&[n_state], mlp, dtype.clone()),
            mlp_ln: LayerNormWeights::empty_dtype(n_state, dtype),
            n_state,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_with_key_lens(x, None)
    }

    fn forward_with_key_lens(&self, x: &Tensor, key_lens: Option<&Tensor>) -> Result<Tensor> {
        // Self-attention (pre-norm)
        let h = scoped("attn_ln", || self.attn_ln.apply(x))?;
        let attn_out = scoped("attn", || self.attn.forward_with_key_lens(&h, None, None, key_lens))?;
        let x = x.try_add(&attn_out)?;

        // MLP (pre-norm)
        let h = scoped("mlp_ln", || self.mlp_ln.apply(&x))?;
        let h = linear_with_bias(&h, &self.mlp0_w, &self.mlp0_b)?;
        let h = h.gelu_exact()?;
        let h = linear_with_bias(&h, &self.mlp1_w, &self.mlp1_b)?;
        let x = x.try_add(&h)?;
        Ok(x)
    }
}

impl HasStateDict for EncoderBlock {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.extend(self.attn.state_dict(&prefixed(prefix, "attn")));
        sd.extend(self.attn_ln.state_dict(&prefixed(prefix, "attn_ln")));
        sd.insert(prefixed(prefix, "mlp.0.weight"), self.mlp0_w.clone());
        sd.insert(prefixed(prefix, "mlp.0.bias"), self.mlp0_b.clone());
        sd.insert(prefixed(prefix, "mlp.2.weight"), self.mlp1_w.clone());
        sd.insert(prefixed(prefix, "mlp.2.bias"), self.mlp1_b.clone());
        sd.extend(self.mlp_ln.state_dict(&prefixed(prefix, "mlp_ln")));
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.attn.load_state_dict(sd, &prefixed(prefix, "attn"))?;
        self.attn_ln.load_state_dict(sd, &prefixed(prefix, "attn_ln"))?;
        self.mlp0_w = get_tensor(sd, &prefixed(prefix, "mlp.0.weight"))?;
        self.mlp0_b = get_tensor(sd, &prefixed(prefix, "mlp.0.bias"))?;
        self.mlp1_w = get_tensor(sd, &prefixed(prefix, "mlp.2.weight"))?;
        self.mlp1_b = get_tensor(sd, &prefixed(prefix, "mlp.2.bias"))?;
        self.mlp_ln.load_state_dict(sd, &prefixed(prefix, "mlp_ln"))?;
        Ok(())
    }
}

/// Whisper audio encoder: Conv1d × 2 + sinusoidal pos-emb + N × EncoderBlock + LayerNorm.
#[derive(Clone)]
pub struct AudioEncoder {
    pub conv1: Conv1dWeights,
    pub conv2: Conv1dWeights,
    pub positional_embedding: Tensor,
    pub blocks: Vec<EncoderBlock>,
    pub ln_post: LayerNormWeights,
    pub n_state: usize,
    pub n_head: usize,
}

impl AudioEncoder {
    pub fn empty(dims: &ModelDimensions) -> Self {
        let n_state = dims.n_audio_state;
        let dtype = dims.dtype.clone();
        Self {
            conv1: Conv1dWeights::empty_dtype(dims.n_mels, n_state, 3, 1, 1, true, dtype.clone()),
            conv2: Conv1dWeights::empty_dtype(n_state, n_state, 3, 2, 1, true, dtype.clone()),
            positional_embedding: sinusoids(dims.n_audio_ctx, n_state, 10_000.0).expect("sinusoidal embedding"),
            blocks: (0..dims.n_audio_layer)
                .map(|_| EncoderBlock::empty_dtype(n_state, dims.n_audio_head, dtype.clone()))
                .collect(),
            ln_post: LayerNormWeights::empty_dtype(n_state, dtype),
            n_state,
            n_head: dims.n_audio_head,
        }
    }

    /// Forward: mel `[B, n_mels, T]` → encoder features `[B, T/2, D]`.
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        // Cast input to the compute dtype (weights are dims.dtype; the host
        // feeds fp32 mel). Matches `model.py:48` weight.to(x.dtype) from the
        // other direction — we cast x to the weight dtype so the graph is uniform.
        let dtype = self.conv1.weight.dtype().clone();
        let mel = mel.cast(dtype.clone());
        let x = scoped("conv1", || self.conv1.forward(&mel))?;
        let x = x.gelu_exact()?;
        let x = scoped("conv2", || self.conv2.forward(&x))?;
        let x = x.gelu_exact()?;

        // [B, D, T/2] → [B, T/2, D]
        let x = x.try_permute(&[0, 2, 1])?;

        // Add positional embedding [n_audio_ctx, D]
        let x = x.try_add(&self.positional_embedding)?.cast(dtype);

        let (batch, sequence, state) = (x.dim_const(0)?, x.dim_const(1)?, x.dim_const(2)?);
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
            x = x.try_shrink([(0, batch), (0, sequence), (0, state)])?;
        }

        // Final LayerNorm + cast to fp32. The encoder output is consumed by the
        // host (copyout_prefix into Vec<f32>) and fed to the prefill/step JITs
        // which cast it back to the compute dtype. Keeping the output fp32 means
        // the host read path works regardless of compute dtype.
        Ok(scoped("ln_post", || self.ln_post.apply(&x))?.cast(DType::Float32))
    }
}

pub(crate) fn encoder_padded_sequence_len(device: &svod_dtype::DeviceSpec, sequence: usize) -> Option<usize> {
    svod_tk::flash_attention_supported(device)
        .then(|| padded_fa_sequence_len(false, sequence, sequence, sequence))
        .flatten()
}

impl HasStateDict for AudioEncoder {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.extend(self.conv1.state_dict(&prefixed(prefix, "conv1")));
        sd.extend(self.conv2.state_dict(&prefixed(prefix, "conv2")));
        sd.insert(prefixed(prefix, "positional_embedding"), self.positional_embedding.clone());
        for (i, block) in self.blocks.iter().enumerate() {
            sd.extend(block.state_dict(&prefixed(prefix, &format!("blocks.{i}"))));
        }
        sd.extend(self.ln_post.state_dict(&prefixed(prefix, "ln_post")));
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.conv1.load_state_dict(sd, &prefixed(prefix, "conv1"))?;
        self.conv2.load_state_dict(sd, &prefixed(prefix, "conv2"))?;
        self.positional_embedding = get_tensor(sd, &prefixed(prefix, "positional_embedding"))?;
        for (i, block) in self.blocks.iter_mut().enumerate() {
            block.load_state_dict(sd, &prefixed(prefix, &format!("blocks.{i}")))?;
        }
        self.ln_post.load_state_dict(sd, &prefixed(prefix, "ln_post"))?;
        Ok(())
    }
}
