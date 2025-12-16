//! Control flow: range, if/end, barrier, symbolic variables.
//!
//! This module contains control flow operations:
//! - Loop constructs: range, range_const, range_axis
//! - Conditionals: if_, endif, end
//! - Synchronization: barrier
//! - Symbolic variables: var, define_var, bind
//! - Special: special (GPU dimension index)

use std::sync::Arc;

use morok_dtype::DType;
use smallvec::SmallVec;

use crate::op::Op;
use crate::types::{AxisId, AxisType, ConstValue};
use crate::uop::UOp;

impl UOp {
    // =========================================================================
    // Range Operations
    // =========================================================================

    /// Create a Range operation with specified axis type.
    pub fn range_axis(end: Arc<Self>, axis_id: AxisId, axis_type: AxisType) -> Arc<Self> {
        Self::new(Op::Range { end, axis_id, axis_type }, DType::Index)
    }

    /// Create a RANGE operation with Loop axis type (convenience for tests).
    ///
    /// Uses `AxisId::Renumbered` since tests typically work with renumbered kernels.
    pub fn range(end: Arc<Self>, axis_id: usize) -> Arc<Self> {
        Self::range_axis(end, AxisId::Renumbered(axis_id), AxisType::Loop)
    }

    /// Create a RANGE operation with constant end value (convenience for tests).
    ///
    /// Uses `AxisId::Renumbered` since tests typically work with renumbered kernels.
    /// Creates a `Loop` range (inside kernels).
    pub fn range_const(end_value: i64, axis_id: usize) -> Arc<Self> {
        let end = Self::const_(DType::Index, ConstValue::Int(end_value));
        Self::range_axis(end, AxisId::Renumbered(axis_id), AxisType::Loop)
    }

    /// Create an OUTER RANGE operation with constant end value (convenience for tests).
    ///
    /// Uses `AxisId::Renumbered` since tests typically work with renumbered kernels.
    /// Creates an `Outer` range (wraps entire kernels).
    pub fn range_outer_const(end_value: i64, axis_id: usize) -> Arc<Self> {
        let end = Self::const_(DType::Index, ConstValue::Int(end_value));
        Self::range_axis(end, AxisId::Renumbered(axis_id), AxisType::Outer)
    }

    // =========================================================================
    // Conditional Operations
    // =========================================================================

    /// Create a conditional block that executes body when condition is true.
    ///
    /// Body contains operations to execute; use `endif` to close the block.
    pub fn if_(condition: Arc<Self>, body: SmallVec<[Arc<Self>; 4]>) -> Arc<Self> {
        Self::new(Op::If { condition, body }, DType::Void)
    }

    /// End if block.
    pub fn endif(if_op: Arc<Self>) -> Arc<Self> {
        Self::new(Op::EndIf { if_op }, DType::Void)
    }

    /// End of range or reduce scope.
    ///
    /// Wraps a computation and closes the specified ranges.
    /// This marks the end of RANGE or REDUCE loops.
    ///
    /// # Arguments
    ///
    /// * `computation` - The computation being performed (e.g., STORE)
    /// * `ranges` - The RANGE or REDUCE operations being closed
    pub fn end(computation: Arc<Self>, ranges: SmallVec<[Arc<Self>; 4]>) -> Arc<Self> {
        Self::new(Op::End { computation, ranges }, DType::Void)
    }

    // =========================================================================
    // Synchronization
    // =========================================================================

    /// Insert a synchronization barrier.
    ///
    /// `src` passes through; `deps` are operations that must complete before
    /// any consumer of this barrier executes.
    pub fn barrier(src: Arc<Self>, deps: SmallVec<[Arc<Self>; 4]>) -> Arc<Self> {
        let dtype = src.dtype();
        Self::new(Op::Barrier { src, deps }, dtype)
    }

    // =========================================================================
    // Symbolic Variables
    // =========================================================================

    /// Create a DefineVar operation for range-bounded variables.
    ///
    /// Used in testing and symbolic analysis to define variables with known ranges.
    /// Range is [min_val, max_val] inclusive.
    pub fn var(name: impl Into<String>, dtype: DType, min_val: i64, max_val: i64) -> Arc<Self> {
        Self::new(Op::DefineVar { name: name.into(), min_val, max_val }, dtype)
    }

    /// Define a symbolic variable with known bounds for range analysis.
    ///
    /// Range is [min_val, max_val] inclusive.
    pub fn define_var(name: String, min_val: i64, max_val: i64) -> Arc<Self> {
        Self::new(Op::DefineVar { name, min_val, max_val }, DType::Index)
    }

    /// Bind concrete value to symbolic variable.
    pub fn bind(var: Arc<Self>, value: Arc<Self>) -> Arc<Self> {
        let dtype = var.dtype();
        Self::new(Op::Bind { var, value }, dtype)
    }

    // =========================================================================
    // Special Operations
    // =========================================================================

    /// Create a GPU-specific dimension variable (e.g., blockIdx.x, threadIdx.y).
    ///
    /// Unlike RANGE which is a loop, SPECIAL represents hardware-provided indices.
    /// The `name` identifies the dimension (rendered as-is in codegen).
    pub fn special(end: Arc<Self>, name: String) -> Arc<Self> {
        Self::new(Op::Special { end, name }, DType::Index)
    }
}
