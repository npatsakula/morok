//! Shared machinery for the ModernBERT encoder heads (embedder / classifier /
//! token classifier): one head error, the shared JIT execute skeleton, and the
//! mask-cast IR helper reused by the three `jit_wrapper!` build closures.
//!
//! The three heads are mechanically identical up to (a) the JIT wrapper type,
//! (b) the output-decode stride, and (c) the profile stage label. [`HeadError`]
//! collapses their three near-identical error enums (which differed only in the
//! capacity-message noun); [`execute_head`] folds their identical capacity-guard
//! → pack → execute → read-back prelude, leaving each head just its decode tail;
//! [`shrink_mask_for_b`] folds the byte-identical mask cast+shrink from the three
//! `_jit.rs` build closures.

use std::time::Duration;

use snafu::{ResultExt, Snafu};
use svod_arch::pipelines::text::{Encoding, RunProfile};
use svod_ir::SInt;
use svod_runtime::{KernelProfile, StageProfile};
use svod_tensor::{BoundVariable, Tensor};

use crate::jit;
use crate::modernbert::classifier_jit::ModernBertClassifierJit;
use crate::modernbert::embedder_jit::ModernBertEmbedderJit;
use crate::modernbert::error::TensorSnafu;
use crate::modernbert::packing::{pack_ids_buffer, pack_mask_buffer};
use crate::modernbert::token_classifier_jit::ModernBertTokenClassifierJit;

/// Head-level error for the ModernBERT encoder heads. Replaces the three
/// near-identical per-head errors (which differed only in the capacity-message
/// noun); `CapacityExceeded` carries the head's `stage` so the message stays
/// task-specific ("embed batch of …", "classify batch of …", "classify_tokens batch of …").
#[derive(Debug, Snafu)]
#[snafu(visibility(pub(crate)))]
pub enum HeadError {
    #[snafu(display("JIT op failed: {source}"))]
    Jit { source: jit::JitError },
    #[snafu(display("device op failed: {source}"))]
    Device { source: svod_device::error::Error },
    #[snafu(display("{stage} batch of {got} exceeds prepared max_batch {max}"))]
    CapacityExceeded { stage: &'static str, got: usize, max: usize },
    #[snafu(display("{stage} sequence length {got} exceeds prepared max_seq {max}"))]
    SequenceTooLong { stage: &'static str, got: usize, max: usize },
}

/// The JIT interface the three `jit_wrapper!`-generated head types share. The
/// macro emits inherent methods (no trait), so [`execute_head`] is bounded on
/// this trait — implemented once per wrapper as straight delegations. Method
/// names mirror the macro-generated DSL identifiers (`input_ids`,
/// `attention_mask`, `b`).
pub(crate) trait JitHeadPlan {
    fn input_ids_mut(&mut self) -> jit::Result<&mut svod_device::Buffer>;
    fn attention_mask_mut(&mut self) -> jit::Result<&mut svod_device::Buffer>;
    fn execute_with_vars(&mut self, vars: &[(&str, i64)]) -> jit::Result<()>;
    fn execute_with_vars_profiled(&mut self, vars: &[(&str, i64)]) -> jit::Result<Vec<KernelProfile>>;
    fn output(&self) -> jit::Result<&svod_device::Buffer>;
}

impl JitHeadPlan for ModernBertEmbedderJit {
    fn input_ids_mut(&mut self) -> jit::Result<&mut svod_device::Buffer> {
        ModernBertEmbedderJit::input_ids_mut(self)
    }
    fn attention_mask_mut(&mut self) -> jit::Result<&mut svod_device::Buffer> {
        ModernBertEmbedderJit::attention_mask_mut(self)
    }
    fn execute_with_vars(&mut self, vars: &[(&str, i64)]) -> jit::Result<()> {
        ModernBertEmbedderJit::execute_with_vars(self, vars)
    }
    fn execute_with_vars_profiled(&mut self, vars: &[(&str, i64)]) -> jit::Result<Vec<KernelProfile>> {
        ModernBertEmbedderJit::execute_with_vars_profiled(self, vars)
    }
    fn output(&self) -> jit::Result<&svod_device::Buffer> {
        ModernBertEmbedderJit::output(self)
    }
}

impl JitHeadPlan for ModernBertClassifierJit {
    fn input_ids_mut(&mut self) -> jit::Result<&mut svod_device::Buffer> {
        ModernBertClassifierJit::input_ids_mut(self)
    }
    fn attention_mask_mut(&mut self) -> jit::Result<&mut svod_device::Buffer> {
        ModernBertClassifierJit::attention_mask_mut(self)
    }
    fn execute_with_vars(&mut self, vars: &[(&str, i64)]) -> jit::Result<()> {
        ModernBertClassifierJit::execute_with_vars(self, vars)
    }
    fn execute_with_vars_profiled(&mut self, vars: &[(&str, i64)]) -> jit::Result<Vec<KernelProfile>> {
        ModernBertClassifierJit::execute_with_vars_profiled(self, vars)
    }
    fn output(&self) -> jit::Result<&svod_device::Buffer> {
        ModernBertClassifierJit::output(self)
    }
}

impl JitHeadPlan for ModernBertTokenClassifierJit {
    fn input_ids_mut(&mut self) -> jit::Result<&mut svod_device::Buffer> {
        ModernBertTokenClassifierJit::input_ids_mut(self)
    }
    fn attention_mask_mut(&mut self) -> jit::Result<&mut svod_device::Buffer> {
        ModernBertTokenClassifierJit::attention_mask_mut(self)
    }
    fn execute_with_vars(&mut self, vars: &[(&str, i64)]) -> jit::Result<()> {
        ModernBertTokenClassifierJit::execute_with_vars(self, vars)
    }
    fn execute_with_vars_profiled(&mut self, vars: &[(&str, i64)]) -> jit::Result<Vec<KernelProfile>> {
        ModernBertTokenClassifierJit::execute_with_vars_profiled(self, vars)
    }
    fn output(&self) -> jit::Result<&svod_device::Buffer> {
        ModernBertTokenClassifierJit::output(self)
    }
}

/// Shared execute skeleton for the encoder heads: capacity guard, input packing,
/// batch-rebind execute, and read-back of the flat `[max_batch, …]` f32 output
/// buffer. Returns the live batch size, a view over the output rows, and the
/// optional profile (one GPU stage labelled `stage`). Each head decodes its rows
/// from `flat` per its own output stride.
///
/// The returned slice borrows the wrapper's own output buffer (tied to the
/// `&mut J` borrow); the head reads it immediately and the borrow ends before the
/// next call.
pub(crate) fn execute_head<'a, J: JitHeadPlan>(
    jit: &'a mut J,
    batch: &[&Encoding],
    max_batch: usize,
    max_seq: usize,
    profile: bool,
    stage: &'static str,
) -> Result<(usize, &'a [f32], Option<RunProfile>), HeadError> {
    let b = batch.len();
    if b == 0 {
        return Ok((0, &[], profile.then(RunProfile::default)));
    }
    if b > max_batch {
        return Err(CapacityExceededSnafu { stage, got: b, max: max_batch }.build());
    }
    // Once chunks can arrive pre-built (the public chunk seam), a caller may
    // feed an Encoding longer than the JIT's prepared max_seq. Catch it here
    // rather than silently truncating inside pack_*.
    if let Some(got) = batch.iter().map(|e| e.input_ids.len()).find(|&n| n > max_seq) {
        return Err(SequenceTooLongSnafu { stage, got, max: max_seq }.build());
    }

    pack_ids_buffer(jit.input_ids_mut().context(JitSnafu)?, batch, max_seq).context(DeviceSnafu)?;
    pack_mask_buffer(jit.attention_mask_mut().context(JitSnafu)?, batch, max_seq).context(DeviceSnafu)?;

    let vars = &[("b", b as i64)];
    let prof = if profile {
        let kernels = jit.execute_with_vars_profiled(vars).context(JitSnafu)?;
        let mut p = RunProfile::default();
        // Fused backbone(+head): one GPU stage — kernels carry the timing; host
        // wall is negligible relative to the GPU work (like gigaam).
        p.push(StageProfile::gpu(stage, Duration::ZERO, kernels));
        Some(p)
    } else {
        jit.execute_with_vars(vars).context(JitSnafu)?;
        None
    };

    let out = jit.output().context(JitSnafu)?;
    let flat = out.as_slice::<f32>().context(DeviceSnafu)?;
    Ok((b, flat, prof))
}

/// Cast `attention_mask` to bool and shrink its batch dim to the live `b` — the
/// byte-identical prelude shared by all three head JIT build closures. The mask
/// arrives as i64 (`InputSpec` has no bool constructor); `b` is the symbolic
/// batch `BoundVariable`. Shrinking here keeps the mask aligned with the
/// symbolic-batch hidden state that `forward_batch` returns (forward_batch
/// shrinks its own copy; without shrinking here, pool_embed would see a concrete
/// max_batch mask against a symbolic-b hidden — a broadcast mismatch).
pub(crate) fn shrink_mask_for_b(
    attention_mask: &Tensor,
    b: &BoundVariable,
) -> Result<Tensor, crate::modernbert::error::Error> {
    let mask = attention_mask.cast(svod_dtype::DType::Bool).context(TensorSnafu)?;
    mask.try_shrink([Some((SInt::Const(0), b.as_sint())), None]).context(TensorSnafu)
}
