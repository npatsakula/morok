//! State-dict pre-passes for PyTorch checkpoints.
//!
//! [`BatchNorm2d`](svod_tensor::nn::BatchNorm2d) reads PyTorch's own keys, so
//! nothing has to be renamed for it: the only entry a checkpoint carries that
//! no layer consumes is `num_batches_tracked`, which [`strip_metadata`] drops.

use crate::state::StateDict;

/// Drop `num_batches_tracked` — training metadata with no inference use.
pub fn strip_metadata(mut sd: StateDict) -> StateDict {
    sd.retain(|k, _| !k.ends_with("num_batches_tracked"));
    sd
}
