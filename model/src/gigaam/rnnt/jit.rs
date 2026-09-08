//! `jit_wrapper!`-generated K-step block JIT for RN-T device-resident decode
//! (`super::block::forward_block`): all loop state device-local; the host
//! reads three tapes + one flag per block.

use svod_macros::jit_wrapper;

use crate::gigaam::model::GigaAm;

jit_wrapper! {
    RnntBlockJit(GigaAm) {
        inputs { enc: Tensor, valid: Tensor }
        // The five carried states recycle in the JIT's own input buffers:
        // `execute()` stores each block's final value where the next block's
        // step 0 reads it, so no host copy and no device->device recycle.
        state { time: Tensor, prev: Tensor, symbols: Tensor, h: Tensor, c: Tensor }
        outputs { tape, emit, frame, active_any }

        build(enc, valid, time, prev, symbols, h, c) {
            // WIND decode window. Byte-identical output for any W>=1 (a pure
            // perf knob, optimum is GPU-dependent); 4 is the validated default.
            let out: crate::gigaam::error::Result<_> =
                crate::gigaam::rnnt::block::forward_block::<4>(model, enc, time, prev, symbols, valid, h, c);
            out
        }
    }
}

jit_wrapper! {
    RnntEncProjJit(GigaAm) {
        enc: Tensor,

        build(enc) {
            // [B, T, E] -> [B, T, J] joint encoder projection, once per wave.
            let (rnnt_head, _) = model.head.expect_rnnt("RnntEncProjJit")?;
            let out: crate::gigaam::error::Result<_> = rnnt_head.joint.project_encoder(enc);
            out
        }
    }
}
