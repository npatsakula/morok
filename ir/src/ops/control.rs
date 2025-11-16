//! Control flow operations (if, range, barrier, etc.).

use std::rc::Rc;

use smallvec::SmallVec;

use morok_dtype::DType;

use super::super::{AxisType, Op, UOp};

impl UOp {
    /// Conditional: if(condition) { body }.
    pub fn if_(condition: Rc<Self>, body: SmallVec<[Rc<Self>; 4]>) -> Rc<Self> {
        Self::new(Op::If { condition, body }, DType::Void)
    }

    /// End if block.
    pub fn endif(if_op: Rc<Self>) -> Rc<Self> {
        Self::new(Op::EndIf { if_op }, DType::Void)
    }

    /// Loop range with axis information.
    pub fn range(end: Rc<Self>, axis_id: usize, axis_type: AxisType) -> Rc<Self> {
        Self::new(Op::Range { end, axis_id, axis_type }, DType::Index)
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
    ///
    /// # Example
    ///
    /// ```ignore
    /// let range = UOp::range(10, 0, AxisType::Loop);
    /// let store = UOp::store(buffer, index, value);
    /// let end = UOp::end(store, smallvec![range]);
    /// ```
    pub fn end(computation: Rc<Self>, ranges: SmallVec<[Rc<Self>; 4]>) -> Rc<Self> {
        Self::new(Op::End { computation, ranges }, DType::Void)
    }

    /// Synchronization barrier.
    pub fn barrier(src: Rc<Self>, deps: SmallVec<[Rc<Self>; 4]>) -> Rc<Self> {
        let dtype = src.dtype();
        Self::new(Op::Barrier { src, deps }, dtype)
    }
}
