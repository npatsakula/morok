//! `GigaAmCtcJit`: the full CTC inference graph — encoder + CTC projection
//! (Conv1d k=1 + log-softmax) — compiled as ONE JIT plan, mel features →
//! `[B, T_sub, vocab_size]` log-probs.
//!
//! Fusing the encoder and head into a single plan removes the cross-plan
//! handoff buffer. With two plans the encoder output is an *output* of one and
//! an *input* of the other, so it must round-trip through host-mapped VRAM (an
//! uncached PCIe read — ~2 s on a 10-minute clip). One plan keeps the encoder
//! activations on-device; only the small final log-probs are read back.
//!
//! The RN-T path still uses the standalone [`crate::gigaam::GigaAmEncoderJit`]
//! — it shares the encoder with per-step predictor/joint JITs, so the encoder
//! output genuinely *is* a reused boundary there.
//!
//! The `jit_wrapper!` macro expands to `svod_model::jit::*` paths, so this
//! file needs the `extern crate self as svod_model;` binding in scope.

extern crate self as svod_model;

use snafu::ResultExt;
use svod_macros::jit_wrapper;

use crate::gigaam::error::TensorSnafu;
use crate::gigaam::model::GigaAm;

jit_wrapper! {
    GigaAmCtcJit(GigaAm) {
        mel: Tensor,
        lengths: Tensor,

        build(mel, lengths) {
            let out = model.encoder.forward_batch(mel, lengths)?;
            // Match the standalone encoder JIT's fp32 cast so the head sees the
            // same dtype regardless of the encoder's compute dtype.
            let out = out.cast(svod_dtype::DType::Float32).context(TensorSnafu)?;
            let head = model.head.expect_ctc("GigaAmCtcJit")?;
            crate::state::scoped("head", || head.forward(&out))
        }
    }
}
