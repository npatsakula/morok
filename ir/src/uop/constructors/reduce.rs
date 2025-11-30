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
    /// Implements Tinygrad's early-return pattern: when all axes are reduced
    /// or when all reduction axes have dimension 1, returns self instead of
    /// creating a ReduceAxis operation.
    ///
    /// # Errors
    /// Returns error if any axis is >= number of dimensions.
    pub fn try_reduce_axis(self: &Rc<Self>, reduce_op: ReduceOp, axes: Vec<usize>) -> Result<Rc<Self>> {
        use crate::SInt;

        // Validate axes if source shape is known
        if let Some(src_shape) = self.shape()? {
            Self::validate_reduce_axes(&axes, src_shape.len())?;

            // Filter out axes where dimension is 1 (no-op reductions)
            let active_axes: Vec<usize> = axes
                .iter()
                .filter(|&&axis| {
                    src_shape
                        .get(axis)
                        .map(|dim| !matches!(dim, SInt::Const(1)))
                        .unwrap_or(false)
                })
                .copied()
                .collect();

            // Tinygrad pattern: if no active axes remain, return self
            // This prevents creating scalar ReduceAxis operations that would
            // propagate empty shapes through the pipeline
            if active_axes.is_empty() {
                return Ok(self.clone());
            }

            // Create ReduceAxis only for non-trivial reductions
            let dtype = self.dtype();
            return Ok(Self::new(Op::ReduceAxis { src: self.clone(), reduce_op, axes: active_axes }, dtype));
        }

        // If shape is unknown, create ReduceAxis with original axes
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
