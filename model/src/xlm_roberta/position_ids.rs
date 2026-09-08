//! XLM-RoBERTa position-ID computation.
//!
//! XLM-RoBERTa (and RoBERTa) compute position IDs from `input_ids` using the
//! fairseq `make_positions` convention:
//!
//! ```text
//! mask          = (input_ids != padding_idx).int()
//! incremental   = cumsum(mask, dim=-1) * mask
//! position_ids  = incremental + padding_idx
//! ```
//!
//! This means real tokens start at position `padding_idx + 1` (= 2 for
//! `pad_token_id = 1`), and padding tokens map to `padding_idx` (= 1), which
//! indexes the zeroed row of the position embedding table. Position rows 0 and
//! 1 are never used for real tokens.

use svod_dtype::DType;
use svod_tensor::Tensor;

use super::error::Result;

/// Compute XLM-RoBERTa position IDs from `input_ids`.
///
/// `input_ids`: `(B, L)` int. Returns `(B, L)` int32 position IDs where real
/// tokens are numbered starting from `padding_idx + 1` and padding positions
/// are set to `padding_idx`.
pub fn position_ids_from_input_ids(input_ids: &Tensor, padding_idx: usize) -> Result<Tensor> {
    let pad = Tensor::const_(padding_idx as i64, DType::Int32);
    let mask = input_ids.try_ne(&pad)?.cast(DType::Int32);
    let cumsum = mask.cumsum(-1)?;
    let incremental = cumsum.try_mul(&mask)?;
    Ok(incremental.try_add(&pad)?)
}
