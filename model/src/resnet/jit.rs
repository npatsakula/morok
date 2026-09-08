//! JIT wrapper for [`ResNet`]. Compiles the full forward graph once at
//! `prepare()` time and replays it per call with the batch dimension bound to
//! the live value of `b` through
//! [`execute_bound`](ResNetJit::execute_bound).
//!
//! Shape contract:
//! - `prepare(InputSpec::f32(&[max_b, 3, H, W]))` bakes `H`/`W` into the plan.
//! - `b` is rebindable on every call; values are clamped to
//!   `[1, max_batch_size]` by the macro-generated setters.
//! - For a different image resolution, prepare a fresh wrapper or call
//!   `prepare` again on this one with a new `InputSpec`.
//! - The single output slot is named `logits`; in [`OutputMode::Features`]
//!   mode it carries the `[B, 512*exp, H/32, W/32]` feature map instead.
//!
//! [`OutputMode::Features`]: super::config::OutputMode::Features
//!
//! See `website/docs/architecture/jit-graphs.md` for the wrapper contract.

use svod_macros::jit_wrapper;

use super::model::ResNet;

jit_wrapper! {
    ResNetJit(ResNet) {
        inputs { images: Tensor }
        batch_var b: (1, model.config.max_batch_size),
        outputs { logits }
        build(images) { model.forward(images) }
    }
}
