//! Reduction and aggregation operations.

use std::rc::Rc;

use smallvec::SmallVec;

use super::super::{Op, ReduceOp, Result, UOp};

impl UOp {
    /// Reduce along specified axes using reduce_op.
    ///
    /// # Errors
    /// Returns error if any axis is >= number of dimensions.
    ///
    /// Note: Validation only occurs if source shape can be inferred.
    pub fn try_reduce_axis(src: Rc<Self>, reduce_op: ReduceOp, axes: Vec<usize>) -> Result<Rc<Self>> {
        // TODO: Validate axes using symbolic shape system
        // Shape validation will be done when shape inference is implemented
        // if let Some(src_shape) = src.shape() {
        //     Self::validate_reduce_axes(&axes, src_shape.len())?;
        // }
        let dtype = src.dtype();
        Ok(Self::new(Op::ReduceAxis { src, reduce_op, axes }, dtype))
    }

    /// Reduce across loop ranges.
    pub fn reduce(src: Rc<Self>, ranges: SmallVec<[Rc<Self>; 4]>, reduce_op: ReduceOp) -> Rc<Self> {
        let dtype = src.dtype();
        Self::new(Op::Reduce { src, ranges, reduce_op }, dtype)
    }

    /// All-reduce across multiple devices.
    pub fn allreduce(src: Rc<Self>, device: Rc<Self>, reduce_op: ReduceOp) -> Rc<Self> {
        let dtype = src.dtype();
        Self::new(Op::AllReduce { src, device, reduce_op }, dtype)
    }
}
