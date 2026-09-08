//! CLS pooling: take the first token's embedding.

use svod_tensor::{Tensor, s};

use super::error::Result;

/// CLS pooling: take the first token's embedding. `hidden_states`: `(B, L, D)`
/// → `(B, D)`.
pub fn cls(hidden_states: &Tensor) -> Result<Tensor> {
    Ok(hidden_states.getitem(s![.., 0, ..])?)
}
