//! Masked mean pooling for sentence embeddings — the sentence-transformers
//! convention (`sum(token_emb * mask) / sum(mask)`), not ModernBERT's native
//! unmasked `FlexBertPoolingHead` (which averages padding zeros in).
//!
//! Built from `sum_with` + `try_div` (the same primitives `wespeaker::tstp`
//! uses), so the L (sequence) axis may be symbolic — `sum_with` carries a
//! symbolic reduced-axis size through; only `mean()` needs the count, which we
//! supply explicitly as `sum(mask)`. The batch dim `b` is symbolic as usual.

use svod_tensor::{Tensor, s};

use super::error::Result;

/// Numerical epsilon on the denominator so an all-padded row divides by ~0
/// instead of erroring in `try_div` (which rejects a constant-zero divisor).
const EPS: f64 = 1e-9;

/// Masked mean pooling over the sequence axis.
///
/// `hidden_states`: `(B, L, D)`. `mask`: `(B, L)` bool where `true` = real
/// token (the same convention the encoder's `padding_mask` uses). Returns
/// `(B, D)` — the mean of the real-token embeddings per row.
///
/// Matches sentence-transformers `sum(token_emb * input_mask_expanded) /
/// clamp(sum(input_mask_expanded), min=1e-9)`. Padded positions contribute
/// zero to both numerator and denominator. `sum_with` carries a symbolic L
/// axis through, so this works on the JIT batch-rebound pipeline.
pub fn masked_mean(hidden_states: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let dtype = hidden_states.dtype();

    // mask (B, L) bool → (B, L, 1) float (1.0 = real, 0.0 = pad) for broadcast
    // over the hidden dim D. cast(bool) yields 1.0/0.0. Both num and den share
    // this 3D rank so the divide doesn't re-expand the L axis.
    let m = mask.cast(dtype.clone())?.try_unsqueeze(-1)?;

    // numerator = sum(hidden * mask, axis=L)                  → (B, 1, D)
    let xw = hidden_states.try_mul(&m)?;
    let num = xw.sum_with().axes(1).keepdim(true).call()?;

    // denominator = sum(mask, axis=L) + eps                   → (B, 1, 1)
    let eps = Tensor::const_(EPS, dtype);
    let den = m.try_add(&eps)?.sum_with().axes(1).keepdim(true).call()?;

    // pooled = num / den, then drop the now-size-1 L axis       → (B, D)
    let pooled = num.try_div(&den)?;
    let p_shape = pooled.shape()?;
    let b_dim = p_shape[0].clone();
    let d_dim = p_shape[p_shape.len() - 1].clone();
    Ok(pooled.try_reshape([b_dim, d_dim])?)
}

/// CLS pooling: take the first token's embedding. `hidden_states`: `(B, L, D)`
/// → `(B, D)`. `L` must be concrete (integer indexing), which the ModernBERT
/// JIT path guarantees (seq_len is baked at `prepare`).
pub fn cls(hidden_states: &Tensor) -> Result<Tensor> {
    Ok(hidden_states.getitem(s![.., 0, ..])?)
}
