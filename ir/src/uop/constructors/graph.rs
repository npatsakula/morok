//! Graph organization: sink, group, dependencies, contiguous.
//!
//! This module contains graph organization and optimization operations:
//! - Graph structure: sink, group
//! - Dependencies: after
//! - Materialization: detach, contiguous, contiguous_backward
//! - Optimization hints: precast
//! - Custom code: custom, customi

use std::sync::Arc;

use morok_dtype::{AddrSpace, DType};
use smallvec::SmallVec;

use crate::op::Op;
use crate::types::{CallInfo, CustomFunctionKind, KernelInfo};
use crate::uop::UOp;
use crate::{Error, Result};

/// Bodies routed through `Op::Call` rather than auto-wrapped into
/// `Op::Function`. These are already-compiled / non-value-producing nodes —
/// their dtype is `Void` and they have no value to project via `gettuple`.
fn is_opaque_call_body(op: &Op) -> bool {
    matches!(
        op,
        Op::Sink { .. }
            | Op::Program { .. }
            | Op::Linear { .. }
            | Op::Copy { .. }
            | Op::BufferView { .. }
            | Op::CustomFunction { .. }
    )
}

impl UOp {
    fn placeholder_like_anchor(src: &Arc<Self>) -> Arc<Self> {
        match src.op() {
            Op::Multi { src: shard, .. } => Self::placeholder_like_anchor(shard),
            Op::MSelect { buffer, .. } => Self::placeholder_like_anchor(buffer),
            Op::MStack { buffers } if !buffers.is_empty() => Self::placeholder_like_anchor(&buffers[0]),
            _ => src.clone(),
        }
    }

    // =========================================================================
    // Graph Structure
    // =========================================================================

    /// Create a sink operation (graph termination).
    ///
    /// Sink marks outputs that must be evaluated. All sources are dependencies.
    pub fn sink(sources: Vec<Arc<Self>>) -> Arc<Self> {
        Self::new(Op::Sink { sources: SmallVec::from_vec(sources), info: None }, DType::Void)
    }

    /// Create a sink carrying a structural [`KernelInfo`] marker.
    ///
    /// The marker participates in hash consing — marked and unmarked sinks
    /// with otherwise identical sources are distinct UOps. Used to mark a
    /// SINK as a fully-formed kernel AST that downstream gates skip over.
    pub fn sink_with_info(sources: Vec<Arc<Self>>, info: KernelInfo) -> Arc<Self> {
        Self::new(Op::Sink { sources: SmallVec::from_vec(sources), info: Some(info) }, DType::Void)
    }

    /// Create a group operation (merging/organizing related ops).
    ///
    /// Group is a NOOP that helps organize related operations together.
    /// It passes through the first source while ensuring all sources are dependencies.
    pub fn group(sources: Vec<Arc<Self>>) -> Arc<Self> {
        let dtype = if sources.is_empty() { DType::Void } else { sources[0].dtype.clone() };
        Self::new(Op::Group { sources: SmallVec::from_vec(sources) }, dtype)
    }

    // =========================================================================
    // Dependencies
    // =========================================================================

    /// Ordering constraint: self depends on deps.
    ///
    /// # Arguments
    /// * `deps` - Dependencies that must complete before this value is used
    ///
    /// # Panics (debug only)
    /// Panics if self is a control flow node (Range, End)
    pub fn after(self: &Arc<Self>, deps: SmallVec<[Arc<Self>; 4]>) -> Arc<Self> {
        #[cfg(debug_assertions)]
        debug_assert!(
            !matches!(self.op(), Op::Range { .. } | Op::End { .. }),
            "AFTER passthrough must be data-producing node, got {:?} (id={})",
            self.op(),
            self.id
        );

        let dtype = self.dtype();
        Self::new(Op::After { passthrough: self.clone(), deps }, dtype)
    }

    // =========================================================================
    // Materialization
    // =========================================================================

    /// Detach from gradient flow / force materialization.
    pub fn detach(self: &Arc<Self>) -> Arc<Self> {
        let dtype = self.dtype();
        Self::new(Op::Detach { src: self.clone() }, dtype)
    }

    /// Ensure contiguous memory layout.
    ///
    /// Elides the CONTIGUOUS wrapper when the source is already contiguous:
    /// - Already a CONTIGUOUS node (no double wrapping)
    /// - Has buffer identity (BUFFER, or RESHAPE/MULTI chain to BUFFER)
    ///
    /// Based on Tinygrad's `UOp.contiguous()` (ops.py:463-466).
    pub fn contiguous(self: &Arc<Self>) -> Arc<Self> {
        if matches!(self.op(), Op::Contiguous { .. }) {
            return self.clone();
        }
        if self.has_buffer_identity() {
            return self.clone();
        }
        let dtype = self.dtype();
        Self::new(Op::Contiguous { src: self.clone(), opts: smallvec::SmallVec::new() }, dtype)
    }

    /// Ensure contiguous memory layout with optimization hints.
    ///
    /// The hints are extracted during rangeify and passed to the optimizer.
    /// Based on Tinygrad's CONTIGUOUS.arg which carries Opt tuples.
    pub fn contiguous_with_opts(
        self: &Arc<Self>,
        opts: smallvec::SmallVec<[crate::types::ContiguousHint; 4]>,
    ) -> Arc<Self> {
        let dtype = self.dtype();
        Self::new(Op::Contiguous { src: self.clone(), opts }, dtype)
    }

    /// Contiguous backward pass.
    pub fn contiguous_backward(self: &Arc<Self>) -> Arc<Self> {
        let dtype = self.dtype();
        Self::new(Op::ContiguousBackward { src: self.clone() }, dtype)
    }

    // =========================================================================
    // Optimization Hints
    // =========================================================================

    /// Optimizer hint to force materialization before type conversion.
    ///
    /// Inserted before BITCAST to ensure the source is rendered separately
    /// in codegen (prevents invalid cast fusion).
    pub fn precast(self: &Arc<Self>) -> Arc<Self> {
        let dtype = self.dtype();
        Self::new(Op::Precast { src: self.clone() }, dtype)
    }

    // =========================================================================
    // Custom Code
    // =========================================================================

    /// Inject custom code as a statement in the generated kernel.
    ///
    /// `deps` are UOps whose rendered names can be referenced in `code`.
    /// `dtype` specifies the result type (often Void for statements).
    ///
    /// This op is rendered by backend codegen as inline/source-template code.
    /// For typed runtime helpers (encdec/graph style), use `custom_function`.
    pub fn custom(deps: SmallVec<[Arc<Self>; 4]>, code: String, dtype: DType) -> Arc<Self> {
        Self::new(Op::Custom { deps, code }, dtype)
    }

    /// Create an explicit runtime custom-function operation.
    ///
    /// This mirrors Tinygrad's `CUSTOM_FUNCTION` op family: the body op encodes
    /// semantic runtime behavior (for example `EncDec`), while CALL provides the
    /// runtime buffer arguments.
    pub fn custom_function(kind: CustomFunctionKind, attrs: SmallVec<[Arc<Self>; 4]>) -> Arc<Self> {
        Self::new(Op::CustomFunction { kind, attrs }, DType::Void)
    }

    /// Inject custom code as an inline expression.
    ///
    /// Unlike `custom` (statement), this is substituted directly into expressions.
    /// `deps` provide values to reference; result has specified `dtype`.
    pub fn customi(deps: SmallVec<[Arc<Self>; 4]>, code: String, dtype: DType) -> Arc<Self> {
        Self::new(Op::CustomI { deps, code }, dtype)
    }

    /// Create a PARAM placeholder shaped like `src` for custom kernel building.
    ///
    /// Rejects symbolic input via `all_int`-style check and creates a global
    /// pointer PARAM with the same logical shape, matching tinygrad's
    /// `placeholder_like`. For multi-device wrappers (MULTI/MSELECT/MSTACK),
    /// placeholder shape is derived from the underlying shard (analog of
    /// tinygrad's `max_shard_shape`).
    pub fn placeholder_like(src: &Arc<Self>, slot: usize) -> Result<Arc<Self>> {
        let anchor = Self::placeholder_like_anchor(src);
        let shape = anchor
            .shape()?
            .cloned()
            .ok_or_else(|| Error::MissingShape { operation: "placeholder_like".to_string() })?;

        let concrete_shape: Vec<usize> = shape
            .iter()
            .map(|d| {
                d.as_const()
                    .ok_or_else(|| Error::SymbolicShapeUnsupported { operation: "placeholder_like".to_string() })
            })
            .collect::<Result<_>>()?;

        let size = concrete_shape.iter().product::<usize>().max(1);
        let dtype = match anchor.dtype() {
            DType::Ptr { .. } => anchor.dtype(),
            dt => dt.ptr(Some(size), AddrSpace::Global),
        };
        let placeholder = UOp::param(slot, size, dtype, None);
        if concrete_shape.len() <= 1 {
            return Ok(placeholder);
        }

        let reshaped = placeholder
            .try_reshape(&crate::shape::Shape::from_iter(concrete_shape.into_iter().map(crate::SInt::Const)))?;
        Ok(reshaped)
    }

    /// Build a custom kernel callable and return `AFTER(callable)` outputs for all inputs.
    ///
    /// Input sources are made contiguous (except existing AFTER nodes), placeholders
    /// are built from those sources, and the closure returns the kernel body UOp.
    ///
    /// Body dispatch mirrors tinygrad's `_OPAQUE_CALL_BODIES` set:
    /// - Opaque bodies (`Sink`, `Program`, `Linear`, `Copy`, `BufferView`,
    ///   `CustomFunction`) wrap into `Op::Call`. An info-less `Sink` is
    ///   auto-promoted to a kernel-marked `Sink` first.
    /// - Value-producing bodies wrap into `Op::Function`, which auto-wraps
    ///   non-`Tuple` bodies via `maketuple`. This keeps `custom_kernel`
    ///   spec-compliant when the closure returns e.g. an arithmetic UOp.
    pub fn custom_kernel<F>(srcs: Vec<Arc<Self>>, fxn: F, info: CallInfo) -> Result<Vec<Arc<Self>>>
    where
        F: FnOnce(Vec<Arc<Self>>) -> Arc<Self>,
    {
        let contig_srcs: Vec<Arc<Self>> =
            srcs.into_iter().map(|x| if matches!(x.op(), Op::After { .. }) { x } else { x.contiguous() }).collect();

        let placeholders: Vec<Arc<Self>> =
            contig_srcs.iter().enumerate().map(|(i, s)| UOp::placeholder_like(s, i)).collect::<Result<_>>()?;

        let mut body = fxn(placeholders);
        if let Op::Sink { sources, info: None } = body.op() {
            body = Self::sink_with_info(sources.to_vec(), KernelInfo::default());
        }
        let args = SmallVec::from_vec(contig_srcs.clone());
        let callable = if is_opaque_call_body(body.op()) { body.call(args, info) } else { body.function(args, info) };

        Ok(contig_srcs.into_iter().map(|s| s.after(SmallVec::from_vec(vec![callable.clone()]))).collect())
    }
}
