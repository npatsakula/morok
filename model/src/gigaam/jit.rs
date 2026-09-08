//! Shared encoder-only JIT for [`GigaAm`]. Output is cast to fp32 so
//! the head-side path (CTC log-probs computed by `CtcHeadJit`, RN-T frames
//! consumed by the predictor/joint step JITs) sees a uniform dtype regardless
//! of whether the encoder ran in fp16, bf16, or fp32.

use svod_macros::jit_wrapper;

use super::model::GigaAm;

jit_wrapper! {
    GigaAmEncoderJit(GigaAm) {
        mel: Tensor,
        lengths: Tensor,

        outputs { frames },

        build(mel, lengths) {
            let out = model.encoder.forward_batch(mel, lengths)?;
            // Permute [B, d_model, T_sub] → [B, T_sub, d_model] on-device: the
            // RN-T decoder consumes frame-major rows, and doing it here turns
            // the host-side strided transpose over the slow mapping into one
            // contiguous copyout.
            Ok::<_, super::error::Error>(out.cast(svod_dtype::DType::Float32).try_permute(&[0, 2, 1])?)
        }
    }
}
