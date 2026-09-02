//! Multi-head attention: self-attention (encoder/decoder) and cross-attention (decoder).
//!
//! Matches `whisper.model.MultiHeadAttention`. Key projection has no bias;
//! query, value, and output projections have bias.
//!
//! Attention scaling: Whisper pre-scales Q and K by `d_head^{-0.25}`, which
//! equals `d_head^{-0.5}` on the scores — identical to SDPA's default
//! `1/sqrt(d_head)`.  So we use the SDPA default scale.

use snafu::ResultExt;
use svod_dtype::DType;
use svod_ir::ConstValue;
use svod_tensor::Tensor;

use crate::state::{self, HasStateDict, StateDict, prefixed};

use super::blocks::LinearWeights;
use super::error::{Result, TensorSnafu, tk_launch_error};

const MIN_PADDED_FA_SEQUENCE: usize = 1024;
const MAX_PADDED_FA_OVERHEAD_DIVISOR: usize = 16;

/// Returns the padded sequence length only when padding is both necessary and
/// sufficiently small for encoder-style self-attention.
pub(crate) fn padded_fa_sequence_len(causal: bool, q_len: usize, k_len: usize, v_len: usize) -> Option<usize> {
    let multiple = svod_tk::FLASH_ATTENTION_SEQUENCE_MULTIPLE;
    if causal || q_len != k_len || q_len != v_len || q_len < MIN_PADDED_FA_SEQUENCE || q_len.is_multiple_of(multiple) {
        return None;
    }
    let padded = q_len.next_multiple_of(multiple);
    (padded - q_len <= q_len / MAX_PADDED_FA_OVERHEAD_DIVISOR).then_some(padded)
}

#[derive(Clone)]
pub struct MultiHeadAttention {
    pub query: LinearWeights,
    pub key: LinearWeights,
    pub value: LinearWeights,
    pub out: LinearWeights,
    pub n_head: usize,
}

impl MultiHeadAttention {
    pub fn empty(n_state: usize, n_head: usize) -> Self {
        Self::empty_dtype(n_state, n_head, DType::Float32)
    }

    pub fn empty_dtype(n_state: usize, n_head: usize, dtype: DType) -> Self {
        Self {
            query: LinearWeights::empty_dtype(n_state, n_state, true, dtype.clone()),
            key: LinearWeights::empty_dtype(n_state, n_state, false, dtype.clone()),
            value: LinearWeights::empty_dtype(n_state, n_state, true, dtype.clone()),
            out: LinearWeights::empty_dtype(n_state, n_state, true, dtype),
            n_head,
        }
    }

    /// Forward pass. `xa = None` for self-attention, `Some(enc)` for cross-attention.
    /// `mask` is the causal mask for decoder self-attention (additive float mask).
    pub fn forward(&self, x: &Tensor, xa: Option<&Tensor>, mask: Option<&Tensor>) -> Result<Tensor> {
        self.forward_with_key_lens(x, xa, mask, None)
    }

    pub(crate) fn forward_with_key_lens(
        &self,
        x: &Tensor,
        xa: Option<&Tensor>,
        mask: Option<&Tensor>,
        key_lens: Option<&Tensor>,
    ) -> Result<Tensor> {
        let q = self.query.forward(x)?;
        let kv_input = xa.unwrap_or(x);
        let k = self.key.forward(kv_input)?;
        let v = self.value.forward(kv_input)?;

        let out = self.fa_attention(&q, &k, &v, mask.is_some(), key_lens)?;
        self.out.forward(&out)
    }

    pub fn forward_return_kv(
        &self,
        x: &Tensor,
        xa: Option<&Tensor>,
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let q = self.query.forward(x)?;
        let kv_input = xa.unwrap_or(x);
        let k = self.key.forward(kv_input)?;
        let v = self.value.forward(kv_input)?;

        let out = self.fa_attention(&q, &k, &v, mask.is_some(), None)?;
        let out = self.out.forward(&out)?;
        Ok((out, k, v))
    }

    pub fn split_heads(&self, t: &Tensor) -> Result<Tensor> {
        // t: [B, S, D] -> [B, S, H, Dh] -> [B, H, S, Dh]
        let shape = t.shape().context(TensorSnafu)?;
        let b = shape[0].clone();
        let s = shape[1].clone();
        let d = shape[2].as_const().ok_or_else(|| super::error::Error::Tensor {
            source: Box::new(svod_tensor::error::Error::SymbolicShapeUnsupported { operation: "split_heads d".into() }),
        })?;
        let dh = d / self.n_head;
        t.try_reshape(&[b, s, svod_ir::SInt::Const(self.n_head), svod_ir::SInt::Const(dh)])
            .context(TensorSnafu)?
            .try_permute(&[0, 2, 1, 3])
            .context(TensorSnafu)
    }

    pub fn merge_heads(&self, t: &Tensor) -> Result<Tensor> {
        // t: [B, H, S, Dh] -> [B, S, H, Dh] -> [B, S, D]
        let shape = t.shape().context(TensorSnafu)?;
        let b = shape[0].clone();
        let s = shape[2].clone();
        let h = shape[1].as_const().ok_or_else(|| super::error::Error::Tensor {
            source: Box::new(svod_tensor::error::Error::SymbolicShapeUnsupported { operation: "merge_heads h".into() }),
        })?;
        let dh = shape[3].as_const().ok_or_else(|| super::error::Error::Tensor {
            source: Box::new(svod_tensor::error::Error::SymbolicShapeUnsupported {
                operation: "merge_heads dh".into(),
            }),
        })?;
        let d = h * dh;
        t.try_permute(&[0, 2, 1, 3])
            .context(TensorSnafu)?
            .try_reshape(&[b, s, svod_ir::SInt::Const(d)])
            .context(TensorSnafu)
    }

    /// Flash-attention path: Q/K/V in [B, S, D] → split to [B, S, H, Dh] for FA,
    /// fall back to SDPA if FA doesn't apply. `causal` controls the mask.
    fn fa_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        causal: bool,
        key_lens: Option<&Tensor>,
    ) -> Result<Tensor> {
        let shape = q.shape().context(TensorSnafu)?;
        let b = shape[0].clone();
        let s = shape[1].clone();
        let d = shape[2].as_const().ok_or_else(|| super::error::Error::Tensor {
            source: Box::new(svod_tensor::error::Error::SymbolicShapeUnsupported {
                operation: "fa_attention d".into(),
            }),
        })?;

        // Split each to [B, S, H, Dh] — FA's layout (seq-major, no permute)
        let split = |t: &Tensor| -> Result<Tensor> {
            let sh = t.shape().context(TensorSnafu)?;
            let tb = sh[0].clone();
            let ts = sh[1].clone();
            let td = sh[2].as_const().ok_or_else(|| super::error::Error::Tensor {
                source: Box::new(svod_tensor::error::Error::SymbolicShapeUnsupported {
                    operation: "fa_attention d".into(),
                }),
            })?;
            t.try_reshape(&[tb, ts, svod_ir::SInt::Const(self.n_head), svod_ir::SInt::Const(td / self.n_head)])
                .context(TensorSnafu)
        };
        let q_fa = split(q)?;
        let k_fa = split(k)?;
        let v_fa = split(v)?;

        // Cast to bf16 for FA kernel
        let dt = q_fa.uop().dtype();
        let need_cast = dt != svod_dtype::DType::BFloat16 && dt != svod_dtype::DType::Float16;
        let (q_f, k_f, v_f) = if need_cast {
            let to = svod_dtype::DType::BFloat16;
            (
                q_fa.cast(to).context(TensorSnafu)?,
                k_fa.cast(svod_dtype::DType::BFloat16).context(TensorSnafu)?,
                v_fa.cast(svod_dtype::DType::BFloat16).context(TensorSnafu)?,
            )
        } else {
            (q_fa.clone(), k_fa.clone(), v_fa.clone())
        };

        let direct = if (d / self.n_head).is_multiple_of(16) {
            svod_tk::flash_attention_with(&q_f, &k_f, &v_f, svod_tk::FaOpts { causal, key_lens })
                .map_err(tk_launch_error)
                .context(TensorSnafu)?
        } else {
            None
        };
        match direct {
            Some(out) => {
                let out = if need_cast { out.cast(dt).context(TensorSnafu)? } else { out };
                out.try_reshape(&[b, s, svod_ir::SInt::Const(d)]).context(TensorSnafu)
            }
            None => {
                // SDPA fallback (needs [B, H, S, Dh])
                let perm = |t: &Tensor| t.try_permute(&[0, 2, 1, 3]).context(TensorSnafu);
                let mask = match key_lens {
                    Some(lens) => {
                        let n = shape[1].as_const().ok_or_else(|| super::error::Error::Tensor {
                            source: Box::new(svod_tensor::error::Error::SymbolicShapeUnsupported {
                                operation: "key-length attention mask".into(),
                            }),
                        })?;
                        let range = Tensor::arange(n as i64, None, None)
                            .context(TensorSnafu)?
                            .try_reshape([1usize, 1, 1, n])
                            .context(TensorSnafu)?;
                        let lens = lens
                            .try_reshape([b.clone(), 1usize.into(), 1usize.into(), 1usize.into()])
                            .context(TensorSnafu)?;
                        Some(range.try_ge(&lens).context(TensorSnafu)?)
                    }
                    None => None,
                };
                let out = perm(&q_fa)?
                    .scaled_dot_product_attention()
                    .key(&perm(&k_fa)?)
                    .value(&perm(&v_fa)?)
                    .is_causal(causal)
                    .maybe_attn_mask(mask.as_ref())
                    .call()
                    .context(TensorSnafu)?;
                self.merge_heads(&out)
            }
        }
    }
}

impl HasStateDict for MultiHeadAttention {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.extend(self.query.state_dict(&prefixed(prefix, "query")));
        sd.extend(self.key.state_dict(&prefixed(prefix, "key")));
        sd.extend(self.value.state_dict(&prefixed(prefix, "value")));
        sd.extend(self.out.state_dict(&prefixed(prefix, "out")));
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.query.load_state_dict(sd, &prefixed(prefix, "query"))?;
        self.key.load_state_dict(sd, &prefixed(prefix, "key"))?;
        self.value.load_state_dict(sd, &prefixed(prefix, "value"))?;
        self.out.load_state_dict(sd, &prefixed(prefix, "out"))?;
        Ok(())
    }
}

/// Build the causal mask for the decoder: `[1, 1, L, L]` upper-triangular -inf.
pub fn causal_mask(seq_len: usize, dtype: DType) -> Result<Tensor> {
    // [L, 1] vs [L] → [L, L] bool: True where col > row (upper triangle)
    let q_idx =
        Tensor::arange(0, Some(seq_len as i64), None).context(TensorSnafu)?.try_unsqueeze(-1).context(TensorSnafu)?; // [L, 1]
    let k_idx = Tensor::arange(0, Some(seq_len as i64), None).context(TensorSnafu)?; // [L]
    let upper = q_idx.try_lt(&k_idx).context(TensorSnafu)?; // [L, L] bool

    let neg_inf = Tensor::const_(ConstValue::Float(f32::NEG_INFINITY as f64), dtype.clone());
    let zero = Tensor::const_(ConstValue::Float(0.0), dtype);
    let float_mask = neg_inf.where_(&upper, &zero).context(TensorSnafu)?;
    float_mask.try_unsqueeze(0).context(TensorSnafu)?.try_unsqueeze(0).context(TensorSnafu)
}
