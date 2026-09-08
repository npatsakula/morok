//! JIT wrapper for [`Qwen3Embedding`]. Bakes the `input_ids` /
//! `attention_mask` shapes into the plan and exposes `b` as the rebindable
//! batch variable. The entire pipeline (backbone + last-token pooling +
//! L2 normalize) runs in one JIT plan.

use svod_macros::jit_wrapper;

use super::embedder::Qwen3Embedding;
use super::reranker::Qwen3Reranker;

jit_wrapper! {
    Qwen3EmbeddingJit(Qwen3Embedding) {
        input_ids: Tensor,
        attention_mask: Tensor,

        batch_var b: (1, model.model.config.max_batch_size),
        outputs { embeddings }

        build(input_ids, attention_mask) {
            model.encode(input_ids, attention_mask)
        }
    }
}

jit_wrapper! {
    Qwen3RerankerJit(Qwen3Reranker) {
        input_ids: Tensor,
        attention_mask: Tensor,

        batch_var b: (1, model.model.config.max_batch_size),
        outputs { scores }

        build(input_ids, attention_mask) {
            model.forward(input_ids, attention_mask)
        }
    }
}
