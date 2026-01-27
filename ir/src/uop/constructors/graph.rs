//! Graph organization: sink, group, assign, contiguous.
//!
//! This module contains graph organization and optimization operations:
//! - Graph structure: sink, group
//! - Assignment: assign
//! - Dependencies: after
//! - Materialization: detach, contiguous, contiguous_backward
//! - Optimization hints: precast
//! - Custom code: custom, customi

use std::sync::Arc;

use morok_dtype::DType;
use smallvec::SmallVec;

use crate::op::Op;
use crate::uop::UOp;

impl UOp {
    // =========================================================================
    // Graph Structure
    // =========================================================================

    /// Create a sink operation (graph termination).
    ///
    /// Sink marks outputs that must be evaluated. All sources are dependencies.
    pub fn sink(sources: Vec<Arc<Self>>) -> Arc<Self> {
        Self::new(Op::Sink { sources: SmallVec::from_vec(sources) }, DType::Void)
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
    // Assignment
    // =========================================================================

    /// In-place assignment.
    pub fn assign(target: Arc<Self>, value: Arc<Self>) -> Arc<Self> {
        let dtype = target.dtype();
        Self::new(Op::Assign { target, value }, dtype)
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
    pub fn contiguous(self: &Arc<Self>) -> Arc<Self> {
        let dtype = self.dtype();
        Self::new(Op::Contiguous { src: self.clone() }, dtype)
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
    pub fn custom(deps: SmallVec<[Arc<Self>; 4]>, code: String, dtype: DType) -> Arc<Self> {
        Self::new(Op::Custom { deps, code }, dtype)
    }

    /// Inject custom code as an inline expression.
    ///
    /// Unlike `custom` (statement), this is substituted directly into expressions.
    /// `deps` provide values to reference; result has specified `dtype`.
    pub fn customi(deps: SmallVec<[Arc<Self>; 4]>, code: String, dtype: DType) -> Arc<Self> {
        Self::new(Op::CustomI { deps, code }, dtype)
    }
}
