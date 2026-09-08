//! JIT wrappers for [`ModernBert`] and [`ModernBertForMaskedLm`]. Both bake the
//! `input_ids` / `attention_mask` shapes into the plan and expose `b` as the
//! rebindable batch variable. `ModernBertJit` returns the `(B, L, D)`
//! last-hidden-state; `ModernBertMlmJit` fuses the backbone + MLM head into one
//! plan (like `GigaAmCtcJit`), keeping activations on-device and reading back
//! only the `(B, L, V)` logits.

use svod_macros::jit_wrapper;

use super::head::ModernBertForMaskedLm;
use super::model::ModernBert;

jit_wrapper! {
    ModernBertJit(ModernBert) {
        input_ids: Tensor,
        attention_mask: Tensor,

        batch_var b: (1, model.config.max_batch_size),
        outputs { hidden }

        build(input_ids, attention_mask) {
            model.forward(input_ids, Some(attention_mask))
        }
    }
}

jit_wrapper! {
    ModernBertMlmJit(ModernBertForMaskedLm) {
        input_ids: Tensor,
        attention_mask: Tensor,

        batch_var b: (1, model.bert.config.max_batch_size),
        outputs { logits }

        build(input_ids, attention_mask) {
            model.forward(input_ids, Some(attention_mask))
        }
    }
}
