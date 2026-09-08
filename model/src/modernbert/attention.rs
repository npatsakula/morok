//! ModernBERT multi-head attention with fused QKV and RoPE.
//!
//! Direct port of `FlexBertUnpadRopeAttention` (padded path): one fused
//! `Wqkv: Linear(D, 3D)` (no bias), RoPE applied to Q and K, scaled
//! dot-product attention, then `Wo: Linear(D, D)` (no bias).
//!
//! The fused QKV output along dim -1 is `[Q(H*hd) | K(H*hd) | V(H*hd)]`; we
//! split it into three and reshape each to `(B, L, H, hd)` → permute
//! `(B, H, L, hd)` for RoPE + SDPA. Sliding-window local layers pass a `window`;
//! global layers pass `None`.

use svod_dtype::DType;
use svod_ir::SInt;
use svod_tensor::Tensor;

use crate::init::fan_in_uniform;
use crate::state::{self, HasStateDict, StateDict, get_tensor, prefixed};

use super::error::Result;

use super::rotary::RotaryTable;

#[derive(Clone)]
pub struct ModernBertAttention {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    /// `None` for global layers; `Some((left, right))` for local layers.
    pub window: Option<(usize, usize)>,
    pub qkv_weight: Tensor,
    pub out_weight: Tensor,
}

impl ModernBertAttention {
    pub fn empty(
        hidden_size: usize,
        num_heads: usize,
        head_dim: usize,
        window: Option<(usize, usize)>,
        dtype: DType,
    ) -> Self {
        let qkv_weight = fan_in_uniform(&[3 * hidden_size, hidden_size], hidden_size, dtype.clone());
        let out_weight = fan_in_uniform(&[hidden_size, hidden_size], hidden_size, dtype);
        Self { hidden_size, num_heads, head_dim, window, qkv_weight, out_weight }
    }

    /// Forward. `x`: `(B, L, D)`. Returns `(B, L, D)`.
    /// `rotary`: the per-layer cos/sin table. `padding_mask`: optional bool
    /// `(B, 1, 1, L)` where `true` masks out (padding) positions in the KEY
    /// axis.
    pub fn forward(&self, x: &Tensor, rotary: &RotaryTable, padding_mask: Option<&Tensor>) -> Result<Tensor> {
        let b = x.dim(0)?;
        let l = x.dim_const(1)?;
        let h = self.num_heads as isize;
        let hd = self.head_dim as isize;
        let bsint: SInt = b;

        // Fused QKV: (B, L, 3D) → chunk into q, k, v each (B, L, D).
        let qkv = x.linear().weight(&self.qkv_weight).call()?;
        let mut parts = qkv.chunk(3, -1)?;
        let v = parts.pop().expect("chunk(3) yields 3 parts");
        let k = parts.pop().expect("chunk(3) yields 3 parts");
        let q = parts.pop().expect("chunk(3) yields 3 parts");

        // (B, L, D) → (B, L, H, hd) → (B, H, L, hd)
        let to_heads = |t: Tensor| -> Result<Tensor> {
            Ok(t.view([bsint.clone(), l.into(), h.into(), hd.into()])?.try_permute(&[0, 2, 1, 3])?)
        };
        let q = rotary.apply(&to_heads(q)?)?;
        let k = rotary.apply(&to_heads(k)?)?;
        let v = to_heads(v)?;

        // Scaled dot-product attention. Window restricts keys for local
        // layers; the bon builder is type-stated, so chain unconditionally
        // using `maybe_window` (None for global layers).
        let attn = q
            .scaled_dot_product_attention()
            .key(&k)
            .value(&v)
            .maybe_attn_mask(padding_mask)
            .maybe_window(self.window)
            .call()?;

        // (B, H, L, hd) → (B, L, H*hd) = (B, L, D) → output projection.
        let attn = attn.try_permute(&[0, 2, 1, 3])?.view([bsint, l.into(), (self.num_heads * self.head_dim).into()])?;
        Ok(attn.linear().weight(&self.out_weight).call()?)
    }
}

impl HasStateDict for ModernBertAttention {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.insert(prefixed(prefix, "Wqkv.weight"), self.qkv_weight.clone());
        sd.insert(prefixed(prefix, "Wo.weight"), self.out_weight.clone());
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.qkv_weight = get_tensor(sd, &prefixed(prefix, "Wqkv.weight"))?;
        self.out_weight = get_tensor(sd, &prefixed(prefix, "Wo.weight"))?;
        Ok(())
    }
}
