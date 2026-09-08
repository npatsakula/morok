//! XLM-RoBERTa self-attention with separate biased Q/K/V/O projections.
//!
//! Standard BERT-style multi-head attention: separate `Linear(D, D, bias)` for
//! Q, K, V; reshape to `(B, H, L, hd)`; scaled dot-product attention (global,
//! no window); output `Linear(D, D, bias)`. No RoPE, no relative positions.
//!
//! State-dict keys match the published `BAAI/bge-m3` `pytorch_model.bin`:
//! `attention.self.{query,key,value}.{weight,bias}`,
//! `attention.output.dense.{weight,bias}`.

use svod_dtype::DType;
use svod_ir::SInt;
use svod_tensor::Tensor;

use crate::init::{fan_in_uniform, zeros};
use crate::state::{self, HasStateDict, StateDict, get_tensor, prefixed};

use super::error::Result;

#[derive(Clone)]
pub struct XlmRobertaAttention {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub query_weight: Tensor,
    pub query_bias: Tensor,
    pub key_weight: Tensor,
    pub key_bias: Tensor,
    pub value_weight: Tensor,
    pub value_bias: Tensor,
    pub out_weight: Tensor,
    pub out_bias: Tensor,
}

impl XlmRobertaAttention {
    pub fn empty(hidden_size: usize, num_heads: usize, head_dim: usize, dtype: DType) -> Self {
        Self {
            hidden_size,
            num_heads,
            head_dim,
            query_weight: fan_in_uniform(&[hidden_size, hidden_size], hidden_size, dtype.clone()),
            query_bias: zeros(&[hidden_size], dtype.clone()),
            key_weight: fan_in_uniform(&[hidden_size, hidden_size], hidden_size, dtype.clone()),
            key_bias: zeros(&[hidden_size], dtype.clone()),
            value_weight: fan_in_uniform(&[hidden_size, hidden_size], hidden_size, dtype.clone()),
            value_bias: zeros(&[hidden_size], dtype.clone()),
            out_weight: fan_in_uniform(&[hidden_size, hidden_size], hidden_size, dtype.clone()),
            out_bias: zeros(&[hidden_size], dtype),
        }
    }

    /// Forward. `x`: `(B, L, D)`. Returns `(B, L, D)`.
    /// `padding_mask`: optional bool `(B, 1, 1, L)` where `true` masks out
    /// (padding) positions in the KEY axis.
    pub fn forward(&self, x: &Tensor, padding_mask: Option<&Tensor>) -> Result<Tensor> {
        let b = x.dim(0)?;
        let l = x.dim_const(1)?;
        let h = self.num_heads as isize;
        let hd = self.head_dim as isize;
        let bsint: SInt = b;

        let q = x.linear().weight(&self.query_weight).bias(&self.query_bias).call()?;
        let k = x.linear().weight(&self.key_weight).bias(&self.key_bias).call()?;
        let v = x.linear().weight(&self.value_weight).bias(&self.value_bias).call()?;

        let to_heads = |t: Tensor| -> Result<Tensor> {
            Ok(t.view([bsint.clone(), l.into(), h.into(), hd.into()])?.try_permute(&[0, 2, 1, 3])?)
        };
        let q = to_heads(q)?;
        let k = to_heads(k)?;
        let v = to_heads(v)?;

        let attn = q.scaled_dot_product_attention().key(&k).value(&v).maybe_attn_mask(padding_mask).call()?;

        let attn = attn.try_permute(&[0, 2, 1, 3])?.view([bsint, l.into(), (self.num_heads * self.head_dim).into()])?;

        Ok(attn.linear().weight(&self.out_weight).bias(&self.out_bias).call()?)
    }
}

impl HasStateDict for XlmRobertaAttention {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.insert(prefixed(prefix, "self.query.weight"), self.query_weight.clone());
        sd.insert(prefixed(prefix, "self.query.bias"), self.query_bias.clone());
        sd.insert(prefixed(prefix, "self.key.weight"), self.key_weight.clone());
        sd.insert(prefixed(prefix, "self.key.bias"), self.key_bias.clone());
        sd.insert(prefixed(prefix, "self.value.weight"), self.value_weight.clone());
        sd.insert(prefixed(prefix, "self.value.bias"), self.value_bias.clone());
        sd.insert(prefixed(prefix, "output.dense.weight"), self.out_weight.clone());
        sd.insert(prefixed(prefix, "output.dense.bias"), self.out_bias.clone());
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.query_weight = get_tensor(sd, &prefixed(prefix, "self.query.weight"))?;
        self.query_bias = get_tensor(sd, &prefixed(prefix, "self.query.bias"))?;
        self.key_weight = get_tensor(sd, &prefixed(prefix, "self.key.weight"))?;
        self.key_bias = get_tensor(sd, &prefixed(prefix, "self.key.bias"))?;
        self.value_weight = get_tensor(sd, &prefixed(prefix, "self.value.weight"))?;
        self.value_bias = get_tensor(sd, &prefixed(prefix, "self.value.bias"))?;
        self.out_weight = get_tensor(sd, &prefixed(prefix, "output.dense.weight"))?;
        self.out_bias = get_tensor(sd, &prefixed(prefix, "output.dense.bias"))?;
        Ok(())
    }
}
