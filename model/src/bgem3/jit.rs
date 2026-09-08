//! JIT wrappers for BGE-M3 and BGE-reranker-v2-m3.

use svod_macros::jit_wrapper;

use super::embedder::BgeM3;
use super::reranker::BgeRerankerV2M3;

jit_wrapper! {
    BgeM3DenseJit(BgeM3) {
        input_ids: Tensor,
        attention_mask: Tensor,

        batch_var b: (1, model.model.config.max_batch_size),
        outputs { dense }

        build(input_ids, attention_mask) {
            model.encode_dense(input_ids, attention_mask)
        }
    }
}

jit_wrapper! {
    BgeM3ColbertJit(BgeM3) {
        input_ids: Tensor,
        attention_mask: Tensor,

        batch_var b: (1, model.model.config.max_batch_size),
        outputs { colbert }

        build(input_ids, attention_mask) {
            model.encode_colbert(input_ids, attention_mask)
        }
    }
}

jit_wrapper! {
    BgeRerankerJit(BgeRerankerV2M3) {
        input_ids: Tensor,
        attention_mask: Tensor,

        batch_var b: (1, model.model.config.max_batch_size),
        outputs { logits }

        build(input_ids, attention_mask) {
            model.forward(input_ids, Some(attention_mask))
        }
    }
}
