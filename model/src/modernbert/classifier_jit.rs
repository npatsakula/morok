//! JIT wrapper for the ModernBERT **classifier**: backbone forward (with the
//! padding mask) + fused classification head (pool → dense → GELU → norm →
//! classifier linear), compiled as ONE JIT plan. `input_ids` `(B, L)` int64 +
//! `attention_mask` `(B, L)` bool → raw logits `(B, num_labels)` (f32).
//!
//! Mirrors [`super::embedder_jit::ModernBertEmbedderJit`] (which fuses
//! backbone + pool + L2-norm). The mask is numerically load-bearing for the
//! same reason — masked attention keeps pad tokens out of real-token
//! representations. Fusing the head keeps the `(B, L, D)` activations
//! on-device; only the small `(B, num_labels)` logits are read back.

extern crate self as svod_model;

use svod_macros::jit_wrapper;

use super::classifier::ModernBertClassificationModel;
use super::head_jit::shrink_mask_for_b;

jit_wrapper! {
    ModernBertClassifierJit(ModernBertClassificationModel) {
        input_ids: Tensor,
        attention_mask: Tensor,

        vars {
            b: (1, model.backbone.config.max_batch_size),
        }

        build(input_ids, attention_mask, b) {
            let mask = shrink_mask_for_b(attention_mask, &b)?;
            model.forward_batch(input_ids, Some(&mask), &b)
        }
    }
}
