//! Scoring functions for BGE-M3 multi-modal retrieval.
//!
//! All functions operate on **pre-computed** embeddings, so they can be used
//! with stored/indexed vectors without re-running the model.

use svod_tensor::Tensor;

use crate::xlm_roberta::error::Result;

/// Dense retrieval score: `q @ p^T`. `q`: `(Bq, D)`, `p`: `(Bp, D)` → `(Bq, Bp)`.
pub fn dense_score(q: &Tensor, p: &Tensor) -> Result<Tensor> {
    Ok(q.matmul(&p.try_transpose(-1, -2)?)?)
}

/// Sparse (lexical) retrieval score: `q @ p^T`. Same operation as dense but on
/// `(B, vocab_size)` sparse vectors.
pub fn sparse_score(q: &Tensor, p: &Tensor) -> Result<Tensor> {
    Ok(q.matmul(&p.try_transpose(-1, -2)?)?)
}

/// ColBERT MaxSim score for a single query-passage pair.
///
/// `q`: `(Lq, Dc)` query token vectors, `p`: `(Lp, Dc)` passage token vectors.
/// Returns a scalar: `sum_i(max_j(q_i · p_j)) / Lq`.
pub fn colbert_score(q: &Tensor, p: &Tensor) -> Result<Tensor> {
    let token_scores = Tensor::einsum("in,jn->ij", &[q, p])?;
    let max_per_q = token_scores.max(-1)?;
    let sum = max_per_q.sum(())?;
    let lq_val = Tensor::const_(q.dim_const(0)? as f64, sum.dtype());
    Ok(sum.try_div(&lq_val)?)
}

/// Hybrid score: weighted combination of dense, sparse, and colbert scores.
///
/// `weights = [dense_w, sparse_w, colbert_w]`. Combined as
/// `(dense*w0 + sparse*w1 + colbert*w2) / (w0 + w1 + w2)`.
pub fn hybrid_score(dense: &Tensor, sparse: &Tensor, colbert: &Tensor, weights: &[f32; 3]) -> Result<Tensor> {
    let dtype = dense.dtype();
    let w_sum = weights.iter().map(|w| *w as f64).sum::<f64>();
    let w_sum_t = Tensor::const_(w_sum, dtype.clone());

    let w0 = Tensor::const_(weights[0] as f64, dtype.clone());
    let w1 = Tensor::const_(weights[1] as f64, dtype.clone());
    let w2 = Tensor::const_(weights[2] as f64, dtype.clone());

    let combined = dense.try_mul(&w0)?.try_add(&sparse.try_mul(&w1)?)?.try_add(&colbert.try_mul(&w2)?)?;

    Ok(combined.try_div(&w_sum_t)?)
}
