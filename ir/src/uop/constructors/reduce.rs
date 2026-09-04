//! Reduction operations: reduce, allreduce.
//!
//! This module contains reduction and aggregation operations:
//! - try_reduce_axis: Reduce along specified axes
//! - reduce: Reduce across loop ranges
//! - allreduce: All-reduce across multiple devices

use std::sync::Arc;

use smallvec::SmallVec;

use crate::Result;
use crate::op::Op;
use crate::ops;
use crate::types::ReduceOp;
use crate::uop::UOp;

impl UOp {
    /// Reduce along specified axes using reduce_op.
    ///
    /// # Errors
    /// Returns error if any axis is >= number of dimensions.
    pub fn try_reduce_axis(self: &Arc<Self>, reduce_op: ReduceOp, mut axes: Vec<usize>) -> Result<Arc<Self>> {
        use crate::SInt;

        let src_shape = self.shape()?.ok_or(crate::Error::VoidTypeInOp)?;
        axes.sort_unstable();
        Self::validate_reduce_axes(&axes, src_shape.len())?;

        let reduce_axes: Vec<usize> =
            axes.iter().copied().filter(|&axis| !matches!(src_shape[axis], SInt::Const(1))).collect();
        let output_shape: SmallVec<[SInt; 4]> =
            src_shape.iter().enumerate().filter(|(axis, _)| !axes.contains(axis)).map(|(_, dim)| dim.clone()).collect();

        if reduce_axes.is_empty() {
            return self.try_reshape(&output_shape);
        }

        let permutation: Vec<usize> = reduce_axes
            .iter()
            .copied()
            .chain((0..src_shape.len()).filter(|axis| !reduce_axes.contains(axis)))
            .collect();
        let permuted = self.try_permute(permutation)?;
        let reduced = permuted.reduce_with_num_axes(SmallVec::new(), reduce_op, reduce_axes.len());

        if axes == reduce_axes { Ok(reduced) } else { reduced.try_reshape(&output_shape) }
    }

    /// Reduce across loop ranges using reduce_op.
    ///
    /// Unlike `try_reduce_axis` (operates on tensor axes), this reduces
    /// values accumulated across RANGE loop iterations.
    pub fn reduce(self: &Arc<Self>, ranges: SmallVec<[Arc<Self>; 4]>, reduce_op: ReduceOp) -> Arc<Self> {
        self.reduce_with_num_axes(ranges, reduce_op, 0)
    }

    /// Reduce leading shaped axes and loop ranges using `reduce_op`.
    pub fn reduce_with_num_axes(
        self: &Arc<Self>,
        ranges: SmallVec<[Arc<Self>; 4]>,
        reduce_op: ReduceOp,
        num_axes: usize,
    ) -> Arc<Self> {
        let dtype = self.dtype();
        Self::new(Op::Reduce(ops::Reduce { src: self.clone(), ranges, reduce_op, num_axes }), dtype)
    }

    /// All-reduce across multiple devices.
    pub fn allreduce(src: Arc<Self>, device: svod_dtype::DeviceSpec, reduce_op: ReduceOp) -> Arc<Self> {
        let dtype = src.dtype();
        Self::new(Op::AllReduce(ops::AllReduce { src, device, reduce_op }), dtype)
    }
}
