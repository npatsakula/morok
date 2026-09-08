//! Powerset → multilabel decoding.
//!
//! Mirrors pyannote `Powerset.to_multilabel(soft=False)`: per frame, take the
//! argmax over the `K` powerset subsets, then map the winning subset to its
//! per-speaker membership. This is the segmentation model's own output
//! encoding (a pure tensor op), so it lives in the model layer; clustering and
//! RTTM are downstream concerns and stay out of Svod.

use svod_tensor::Tensor;

use super::config::powerset_table;
use super::error::Result;

/// Decode `(.., K)` powerset log-probs to `(.., max_per_chunk)` binary
/// multilabel speaker activations, where `K == powerset_class_count(
/// max_per_chunk, max_per_frame)` must equal `logits`' last dim.
///
/// `hardmax` selects the winning subset per frame as a one-hot row; the
/// `linear(weight = M)` then computes `onehot @ Mᵀ`, picking that subset's
/// speaker membership (`M[s][k] = 1` iff speaker `s ∈ subset k`).
pub fn powerset_to_multilabel(logits: &Tensor, max_per_chunk: usize, max_per_frame: usize) -> Result<Tensor> {
    let table = powerset_table(max_per_chunk, max_per_frame);
    let k = table.len();

    // Membership matrix M: (max_per_chunk, K), row-major.
    let mut membership = vec![0f32; max_per_chunk * k];
    for (subset_idx, subset) in table.iter().enumerate() {
        for &speaker in subset {
            membership[speaker * k + subset_idx] = 1.0;
        }
    }
    let mapping = Tensor::from_slice(&membership).try_reshape([max_per_chunk, k])?;

    Ok(logits.hardmax(-1)?.linear().weight(&mapping).call()?)
}
