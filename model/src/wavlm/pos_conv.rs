//! Convolutional positional embedding: a single grouped Conv1d (k=128,
//! groups=16, padding=64) whose output is GELU'd and added to the input.
//!
//! Mirrors `ConvolutionalPositionalEmbedding` from
//! `submodules/DiariZen/diarizen/models/module/wav2vec2/components.py:317-380`.
//! Notable wrinkles:
//!
//! - PyTorch's weight-norm parametrization stores two tensors instead of one
//!   `weight`. Modern keys: `conv.parametrizations.weight.original0` (the
//!   magnitude `g`) and `conv.parametrizations.weight.original1` (the
//!   direction `v`). Legacy keys: `conv.weight_g` / `conv.weight_v`. With
//!   `dim=2` (kernel axis), `g.shape = (1, 1, k)` and `v.shape = (out, in/g, k)`.
//!   The effective weight is `g * v / ||v||_{dims!=2}`, where `||·||_{...}` is
//!   the L2 norm over all dims except the kernel dim. We reconstruct this
//!   eagerly at load time so forward sees a plain `Conv1d` — which is why the
//!   [`Module`] impl is hand-written rather than derived.
//! - Even kernels (`k=128`) make the conv's output one frame longer than the
//!   input; trim one trailing frame (`num_remove = 1`) to keep the length.
//! - The pos-conv output is *added* to the input (residual-style), not
//!   concatenated.

use svod_dtype::DType;
use svod_ir::SInt;
use svod_tensor::Tensor;
use svod_tensor::nn::{Conv1d, Layer, Module, StateDict, get_tensor, prefixed};

use crate::init::{fan_in_uniform, zeros};

use super::error::Result;

#[derive(Clone)]
pub struct ConvolutionalPositionalEmbedding {
    pub conv: Conv1d,
    /// Trailing frames to drop so the output length matches the input.
    pub num_remove: usize,
}

impl ConvolutionalPositionalEmbedding {
    pub fn empty(embed_dim: usize, kernel_size: usize, groups: usize) -> Self {
        assert!(groups > 0 && embed_dim.is_multiple_of(groups), "groups must divide embed_dim");
        let in_per_group = embed_dim / groups;
        let fan_in = in_per_group * kernel_size;
        let weight = fan_in_uniform(&[embed_dim, in_per_group, kernel_size], fan_in, DType::Float32);
        let bias = zeros(&[embed_dim], DType::Float32);
        let padding = (kernel_size / 2) as isize;
        Self {
            conv: Conv1d::new(weight, Some(bias)).with_groups(groups).with_padding((padding, padding)),
            num_remove: usize::from(kernel_size.is_multiple_of(2)),
        }
    }

    /// Forward on `(B, T, C)` (channels-last). Internally transposes to NCT
    /// for the conv, then transposes back.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.conv.forward(&x.try_permute(&[0, 2, 1])?)?;
        // Trim the trailing `num_remove` frames so output length matches input.
        let keep = y.dim(2)? - SInt::from(self.num_remove);
        let y = if self.num_remove > 0 { y.narrow(2, SInt::Const(0), keep)? } else { y };
        // Exact (erf-based) GELU matches PyTorch's `nn.functional.gelu`.
        Ok(y.gelu_exact()?.try_permute(&[0, 2, 1])?)
    }
}

/// Reconstruct `weight = g * v / ||v||_dim01` where the norm is L2 over the
/// (out, in/groups) axes (i.e., over all dims except dim=2, the kernel dim).
/// `g` is expected with shape `(1, 1, k)`, `v` with shape `(out, in/g, k)`.
fn weight_norm_reconstruct(g: &Tensor, v: &Tensor) -> svod_tensor::error::Result<Tensor> {
    let norm = v.try_mul(v)?.sum_with().axes(vec![0, 1]).keepdim(true).call()?.try_sqrt()?;
    g.try_mul(&v.try_div(&norm)?)
}

impl Module for ConvolutionalPositionalEmbedding {
    /// Emits a single `conv.weight` (the reconstructed effective weight) and
    /// `conv.bias`. The original `(g, v)` pair is *not* round-tripped — it is
    /// only needed on the load path from torch checkpoints.
    fn write_state(&self, prefix: &str, out: &mut StateDict) {
        self.conv.write_state(&prefixed(prefix, "conv"), out);
    }

    /// Loads either the modern `parametrizations.weight.original{0,1}` form,
    /// the legacy `weight_g` / `weight_v` form, or a flat `weight`.
    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> svod_tensor::error::Result<()> {
        self.conv.bias = Some(get_tensor(sd, &prefixed(prefix, "conv.bias"))?);

        let pair = [
            ("conv.parametrizations.weight.original0", "conv.parametrizations.weight.original1"),
            ("conv.weight_g", "conv.weight_v"),
        ]
        .into_iter()
        .map(|(g, v)| (prefixed(prefix, g), prefixed(prefix, v)))
        .find(|(g, v)| sd.contains_key(g) && sd.contains_key(v));

        self.conv.weight = match pair {
            Some((g, v)) => weight_norm_reconstruct(&get_tensor(sd, &g)?, &get_tensor(sd, &v)?)?,
            None => get_tensor(sd, &prefixed(prefix, "conv.weight"))?,
        };
        Ok(())
    }
}
