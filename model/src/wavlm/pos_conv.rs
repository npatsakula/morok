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
//!   eagerly at load time so forward sees a plain `Conv1d`.
//! - Even kernels (`k=128`) make the conv's output one frame longer than the
//!   input; trim one trailing frame (`num_remove = 1`) to keep the length.
//! - The pos-conv output is *added* to the input (residual-style), not
//!   concatenated.

use svod_dtype::DType;
use svod_ir::SInt;
use svod_tensor::Tensor;

use crate::init::{fan_in_uniform, zeros};
use crate::state::{self, HasStateDict, StateDict, get_tensor, prefixed};

use super::error::Result;

#[derive(Clone)]
pub struct ConvolutionalPositionalEmbedding {
    pub embed_dim: usize,
    pub kernel_size: usize,
    pub groups: usize,
    pub padding: usize,
    pub num_remove: usize,
    /// Effective conv weight `(out, in/groups, k)`, reconstructed from the
    /// weight-norm `(g, v)` pair on load.
    pub weight: Tensor,
    pub bias: Tensor,
}

impl ConvolutionalPositionalEmbedding {
    pub fn empty(embed_dim: usize, kernel_size: usize, groups: usize) -> Self {
        assert!(groups > 0 && embed_dim.is_multiple_of(groups), "groups must divide embed_dim");
        let in_per_group = embed_dim / groups;
        let fan_in = in_per_group * kernel_size;
        let weight = fan_in_uniform(&[embed_dim, in_per_group, kernel_size], fan_in, DType::Float32);
        let bias = zeros(&[embed_dim], DType::Float32);
        let padding = kernel_size / 2;
        let num_remove = if kernel_size.is_multiple_of(2) { 1 } else { 0 };
        Self { embed_dim, kernel_size, groups, padding, num_remove, weight, bias }
    }

    /// Forward on `(B, T, C)` (channels-last). Internally transposes to NCT
    /// for the conv, then transposes back.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // (B, T, C) → (B, C, T)
        let xt = x.try_permute(&[0, 2, 1])?;

        let p = self.padding as isize;
        let mut y = xt
            .conv2d()
            .weight(&self.weight)
            .bias(&self.bias)
            .groups(self.groups)
            .stride(&[1])
            .padding(&[(p, p)])
            .call()?;

        // Trim the trailing `num_remove` frames so output length matches input.
        if self.num_remove > 0 {
            let t_full = y.shape()?[2].clone();
            let keep = t_full - SInt::from(self.num_remove);
            y = y.try_shrink([None, None, Some((SInt::Const(0), keep))])?;
        }

        // Exact (erf-based) GELU matches PyTorch's `nn.functional.gelu`.
        let y = y.gelu_exact()?;
        // (B, C, T) → (B, T, C)
        Ok(y.try_permute(&[0, 2, 1])?)
    }
}

/// Reconstruct `weight = g * v / ||v||_dim01` where the norm is L2 over the
/// (out, in/groups) axes (i.e., over all dims except dim=2, the kernel dim).
/// `g` is expected with shape `(1, 1, k)`, `v` with shape `(out, in/g, k)`.
fn weight_norm_reconstruct(g: &Tensor, v: &Tensor) -> std::result::Result<Tensor, state::Error> {
    // ||v||_{dim=0,1}: sum of squares over dims 0 and 1, then sqrt, keepdim.
    let v_sq = v.try_mul(v)?;
    let v_norm_sq = v_sq.sum_with().axes(svod_tensor::reduce::AxisSpec::Multiple(vec![0, 1])).keepdim(true).call()?;
    let v_norm = v_norm_sq.try_sqrt()?;
    let v_dir = v.try_div(&v_norm)?;
    Ok(g.try_mul(&v_dir)?)
}

impl HasStateDict for ConvolutionalPositionalEmbedding {
    /// On save, we emit a single `conv.weight` (the reconstructed effective
    /// weight) and `conv.bias`. We do NOT round-trip the original `(g, v)`
    /// pair — that's only needed on the *load* path from torch checkpoints.
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.insert(prefixed(prefix, "conv.weight"), self.weight.clone());
        sd.insert(prefixed(prefix, "conv.bias"), self.bias.clone());
        sd
    }

    /// Loads either the modern `parametrizations.weight.original{0,1}` form,
    /// the legacy `weight_g` / `weight_v` form, or a flat `weight`.
    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.bias = get_tensor(sd, &prefixed(prefix, "conv.bias"))?;

        let modern_g = prefixed(prefix, "conv.parametrizations.weight.original0");
        let modern_v = prefixed(prefix, "conv.parametrizations.weight.original1");
        let legacy_g = prefixed(prefix, "conv.weight_g");
        let legacy_v = prefixed(prefix, "conv.weight_v");
        let flat = prefixed(prefix, "conv.weight");

        if sd.contains_key(&modern_g) && sd.contains_key(&modern_v) {
            let g = get_tensor(sd, &modern_g)?;
            let v = get_tensor(sd, &modern_v)?;
            self.weight = weight_norm_reconstruct(&g, &v)?;
        } else if sd.contains_key(&legacy_g) && sd.contains_key(&legacy_v) {
            let g = get_tensor(sd, &legacy_g)?;
            let v = get_tensor(sd, &legacy_v)?;
            self.weight = weight_norm_reconstruct(&g, &v)?;
        } else {
            self.weight = get_tensor(sd, &flat)?;
        }
        Ok(())
    }
}
