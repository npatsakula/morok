//! Reduction operations: reduce, allreduce.
//!
//! This module contains reduction and aggregation operations:
//! - try_reduce_axis: Reduce along specified axes
//! - reduce: Reduce across loop ranges
//! - allreduce: All-reduce across multiple devices

use std::rc::Rc;

use smallvec::SmallVec;

use crate::Result;
use crate::op::Op;
use crate::types::ReduceOp;
use crate::uop::UOp;

impl UOp {
    /// Reduce along specified axes using reduce_op.
    ///
    /// # Errors
    /// Returns error if any axis is >= number of dimensions.
    pub fn try_reduce_axis(self: &Rc<Self>, reduce_op: ReduceOp, axes: Vec<usize>) -> Result<Rc<Self>> {
        // Validate axes if source shape is known
        if let Some(src_shape) = self.shape()? {
            Self::validate_reduce_axes(&axes, src_shape.len())?;
        }
        let dtype = self.dtype();
        Ok(Self::new(Op::ReduceAxis { src: self.clone(), reduce_op, axes }, dtype))
    }

    /// Reduce across loop ranges using reduce_op.
    ///
    /// Unlike `try_reduce_axis` (operates on tensor axes), this reduces
    /// values accumulated across RANGE loop iterations.
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
