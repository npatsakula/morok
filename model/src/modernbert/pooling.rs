//! Masked mean pooling for sentence embeddings — the sentence-transformers
//! convention (`sum(token_emb * mask) / sum(mask)`), not ModernBERT's native
//! unmasked `FlexBertPoolingHead` (which averages padding zeros in).
//!
//! Built from `sum_with` + `try_div` (the same primitives `wespeaker::tstp`
//! uses), so the L (sequence) axis may be symbolic — `sum_with` carries a
//! symbolic reduced-axis size through; only `mean()` needs the count, which we
//! supply explicitly as `sum(mask)`. The batch dim `b` is symbolic as usual.
//!
//! [`pool_embed`] layers L2-normalization on top of [`masked_mean`] for finished
//! embeddings (the sentence-transformers default for ModernBERT).

use snafu::ResultExt;
use svod_tensor::{Tensor, s};

use super::error::{Result, TensorSnafu};

/// Numerical epsilon on the denominator so an all-padded row divides by ~0
/// instead of erroring in `try_div` (which rejects a constant-zero divisor).
const EPS: f64 = 1e-9;

/// Numerical epsilon guarding the L2 norm in [`pool_embed`] — keeps an all-pad
/// row (whose mean-pool is the zero vector) finite instead of dividing by zero.
const NORM_EPS: f64 = 1e-12;

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
    let dtype = hidden_states.uop().dtype();

    // mask (B, L) bool → (B, L, 1) float (1.0 = real, 0.0 = pad) for broadcast
    // over the hidden dim D. cast(bool) yields 1.0/0.0. Both num and den share
    // this 3D rank so the divide doesn't re-expand the L axis.
    let m = mask.cast(dtype.clone()).context(TensorSnafu)?.try_unsqueeze(-1).context(TensorSnafu)?;

    // numerator = sum(hidden * mask, axis=L)                  → (B, 1, D)
    let xw = hidden_states.try_mul(&m).context(TensorSnafu)?;
    let num = xw.sum_with().axes(1).keepdim(true).call().context(TensorSnafu)?;

    // denominator = sum(mask, axis=L) + eps                   → (B, 1, 1)
    let eps = Tensor::const_(EPS, dtype);
    let den = m.try_add(&eps).context(TensorSnafu)?.sum_with().axes(1).keepdim(true).call().context(TensorSnafu)?;

    // pooled = num / den, then drop the now-size-1 L axis       → (B, D)
    let pooled = num.try_div(&den).context(TensorSnafu)?;
    let p_shape = pooled.shape().context(TensorSnafu)?;
    let b_dim = p_shape[0].clone();
    let d_dim = p_shape[p_shape.len() - 1].clone();
    pooled.try_reshape([b_dim, d_dim]).context(TensorSnafu)
}

/// CLS pooling: take the first token's embedding. `hidden_states`: `(B, L, D)`
/// → `(B, D)`. `L` must be concrete (integer indexing), which the ModernBERT
/// JIT path guarantees (seq_len is baked at `prepare`).
pub fn cls(hidden_states: &Tensor) -> Result<Tensor> {
    hidden_states.getitem(s![.., 0, ..]).context(TensorSnafu)
}

/// Masked mean-pool + L2-normalize → finished `(B, D)` embeddings.
///
/// `hidden`: `(B, L, D)` (B symbolic/rebindable, L and D concrete at prepare);
/// `attention_mask`: bool `(B, L)` where `true` = real token. Returns `(B, D)`,
/// mean-pooled over real tokens then L2-normalized per row. Pooling delegates
/// to [`masked_mean`]; the L2-normalize is an embedder-specific finishing step.
pub fn pool_embed(hidden: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
    let pooled = masked_mean(hidden, attention_mask)?;

    // L2-normalize per row: norm = sqrt(sum(pooled^2, axis=D) + eps), shape
    // (B, 1), broadcasts against pooled (B, D). EPS guards an all-pad zero row.
    let dtype = pooled.uop().dtype();
    let eps = Tensor::const_(NORM_EPS, dtype);
    let sq = pooled.square().context(TensorSnafu)?;
    let sq_sum = sq.sum_with().axes(-1isize).keepdim(true).call().context(TensorSnafu)?;
    let norm = sq_sum.try_add(&eps).context(TensorSnafu)?.try_sqrt().context(TensorSnafu)?;
    pooled.try_div(&norm).context(TensorSnafu)
}
