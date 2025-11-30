//! Graph organization: sink, group, assign, contiguous.
//!
//! This module contains graph organization and optimization operations:
//! - Graph structure: sink, group
//! - Assignment: assign
//! - Dependencies: after
//! - Materialization: detach, contiguous, contiguous_backward
//! - Optimization hints: precast
//! - Custom code: custom, customi

use std::rc::Rc;

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
    pub fn sink(sources: Vec<Rc<Self>>) -> Rc<Self> {
        Self::new(Op::Sink { sources: SmallVec::from_vec(sources) }, DType::Void)
    }

    /// Create a group operation (merging/organizing related ops).
    ///
    /// Group is a NOOP that helps organize related operations together.
    /// It passes through the first source while ensuring all sources are dependencies.
    pub fn group(sources: Vec<Rc<Self>>) -> Rc<Self> {
        let dtype = if sources.is_empty() { DType::Void } else { sources[0].dtype.clone() };
        Self::new(Op::Group { sources: SmallVec::from_vec(sources) }, dtype)
    }

    // =========================================================================
    // Assignment
    // =========================================================================

    /// In-place assignment.
    pub fn assign(target: Rc<Self>, value: Rc<Self>) -> Rc<Self> {
        let dtype = target.dtype();
        Self::new(Op::Assign { target, value }, dtype)
    }

    // =========================================================================
    // Dependencies
    // =========================================================================

    /// Ordering constraint: passthrough depends on deps.
    pub fn after(passthrough: Rc<Self>, deps: SmallVec<[Rc<Self>; 4]>) -> Rc<Self> {
        let dtype = passthrough.dtype();
        Self::new(Op::After { passthrough, deps }, dtype)
    }

    // =========================================================================
    // Materialization
    // =========================================================================

    /// Detach from gradient flow / force materialization.
    pub fn detach(src: Rc<Self>) -> Rc<Self> {
        let dtype = src.dtype();
        Self::new(Op::Detach { src }, dtype)
    }

    /// Ensure contiguous memory layout.
    pub fn contiguous(src: Rc<Self>) -> Rc<Self> {
        let dtype = src.dtype();
        Self::new(Op::Contiguous { src }, dtype)
    }

    /// Contiguous backward pass.
    pub fn contiguous_backward(src: Rc<Self>) -> Rc<Self> {
        let dtype = src.dtype();
        Self::new(Op::ContiguousBackward { src }, dtype)
    }

    // =========================================================================
    // Optimization Hints
    // =========================================================================

    /// Optimizer hint to force materialization before type conversion.
    ///
    /// Inserted before BITCAST to ensure the source is rendered separately
    /// in codegen (prevents invalid cast fusion).
    pub fn precast(src: Rc<Self>) -> Rc<Self> {
        let dtype = src.dtype();
        Self::new(Op::Precast { src }, dtype)
    }

    // =========================================================================
    // Custom Code
    // =========================================================================

    /// Inject custom code as a statement in the generated kernel.
    ///
    /// `deps` are UOps whose rendered names can be referenced in `code`.
    /// `dtype` specifies the result type (often Void for statements).
    pub fn custom(deps: SmallVec<[Rc<Self>; 4]>, code: String, dtype: DType) -> Rc<Self> {
        Self::new(Op::Custom { deps, code }, dtype)
    }

    /// Inject custom code as an inline expression.
    ///
    /// Unlike `custom` (statement), this is substituted directly into expressions.
    /// `deps` provide values to reference; result has specified `dtype`.
    pub fn customi(deps: SmallVec<[Rc<Self>; 4]>, code: String, dtype: DType) -> Rc<Self> {
        Self::new(Op::CustomI { deps, code }, dtype)
    }
}
