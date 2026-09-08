//! JIT wrapper for [`super::model::XlmRobertaModel`].

use svod_macros::jit_wrapper;

use super::model::XlmRobertaModel;

jit_wrapper! {
    XlmRobertaJit(XlmRobertaModel) {
        input_ids: Tensor,
        attention_mask: Tensor,

        batch_var b: (1, model.config.max_batch_size),
        outputs { hidden }

        build(input_ids, attention_mask) {
            model.forward(input_ids, Some(attention_mask))
        }
    }
}
