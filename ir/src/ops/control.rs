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
    pub fn end(range_or_reduce: Rc<Self>) -> Rc<Self> {
        let dtype = range_or_reduce.dtype();
        Self::new(Op::End { range_or_reduce }, dtype)
    }

    /// Synchronization barrier.
    pub fn barrier(src: Rc<Self>, deps: SmallVec<[Rc<Self>; 4]>) -> Rc<Self> {
        let dtype = src.dtype();
        Self::new(Op::Barrier { src, deps }, dtype)
    }
}
