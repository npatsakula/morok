//! JIT wrapper for the DiariZen segmentation model. Output is `(B, T, K)`
//! log-probabilities over the 16 powerset speaker subsets.

use svod_macros::jit_wrapper;

use super::model::DiariZenSegmentationModel;

jit_wrapper! {
    DiariZenSegmentationJit(DiariZenSegmentationModel) {
        waveforms: Tensor,

        // `.max(1)` so a (nonsensical) `inference_batch_size == 0` can't make
        // the bound `(1, 0)` and panic in `Variable::new` before the driver's
        // own clamp runs.
        batch_var b: (1, model.config.inference_batch_size.max(1)),
        outputs { logits }

        build(waveforms) {
            model.forward(waveforms)
        }
    }
}
