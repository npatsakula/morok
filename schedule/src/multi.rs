//! Exact in-kernel multi-device rewrites.
//!
//! `Op::Multi` is the single-axis subset of Tinygrad's `UNSHARD`. It does not
//! carry a shard range or a tuple-valued device, so rewrites that need either
//! are deliberately not represented here.

use std::collections::HashMap;
use std::sync::Arc;

use smallvec::{SmallVec, smallvec};
use svod_dtype::{DType, ScalarDType};
use svod_ir::{CallInfo, ConstValue, CustomFunctionKind, Op, ReduceOp, UOp, UOpKey};

use crate::TypedPatternMatcher;
use svod_ir::ops;

/// Hardware-independent subset supported before range assignment.
///
/// `None` is an ordinary unsharded layout and is valid by itself or as a
/// scalar ALU broadcast. `Axis` is Svod's single represented shard layout.
/// Two axes, nested layouts, and operations requiring shard ranges are rejected
/// by [`validate_supported_subset`] rather than leaking into rangeification.
///
/// Supported rewrites are MSELECT(MSTACK), same-axis ALU with scalar operands,
/// non-sharded-axis reductions, PERMUTE, non-shard-axis FLIP/PAD, and the
/// dtype/contiguity wrappers listed in [`multi_pm`]. Outer MULTI, MSTACK, and
/// independent graph outputs remain structural markers at this boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MultiLayout {
    None,
    Axis(usize),
}

fn multi_axis(uop: &Arc<UOp>) -> Option<(Arc<UOp>, usize)> {
    match uop.op() {
        Op::Multi(ops::Multi { src, axis }) => Some((src.clone(), *axis)),
        _ => None,
    }
}

fn rewrite_per_shard_alu(root: &Arc<UOp>) -> Option<Arc<UOp>> {
    if !matches!(root.op(), Op::Unary(..) | Op::Binary(..) | Op::Ternary(..)) {
        return None;
    }

    let axis = root.op().sources().iter().find_map(|src| multi_axis(src).map(|(_, axis)| axis))?;
    let mut local_sources = Vec::with_capacity(root.op().sources().len());
    for src in root.op().sources() {
        if let Some((local, src_axis)) = multi_axis(&src) {
            if src_axis != axis {
                return None;
            }
            local_sources.push(local);
        } else if src.shape().ok().flatten().is_some_and(|shape| shape.is_empty()) {
            local_sources.push(src.clone());
        } else {
            return None;
        }
    }
    Some(UOp::multi(root.with_sources(local_sources), axis).rtag(root.tag().clone()).rorigin(root.origin()))
}

fn passthrough_unary_wrapper(root: &Arc<UOp>, multi: &Arc<UOp>) -> Option<Arc<UOp>> {
    let (local, axis) = multi_axis(multi)?;
    if !matches!(
        root.op(),
        Op::Cast(..) | Op::BitCast(..) | Op::Contiguous(..) | Op::Detach(..) | Op::ContiguousBackward(..)
    ) {
        return None;
    }
    Some(UOp::multi(root.with_sources(vec![local]), axis).rtag(root.tag().clone()).rorigin(root.origin()))
}

fn reduce_multi(root: &Arc<UOp>, multi: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Reduce(ops::Reduce { ranges, reduce_op, num_axes, .. }) = root.op() else { return None };
    let (local, axis) = multi_axis(multi)?;
    if *num_axes == 0 || !ranges.is_empty() {
        return None;
    }
    if axis >= *num_axes {
        return Some(
            UOp::multi(local.reduce_with_num_axes(ranges.clone(), *reduce_op, *num_axes), axis - num_axes)
                .rtag(root.tag().clone())
                .rorigin(root.origin()),
        );
    }

    if !matches!(reduce_op, ReduceOp::Add | ReduceOp::Max) {
        return None;
    }
    let mstacks: Vec<_> = local.toposort().into_iter().filter(|node| matches!(node.op(), Op::MStack(..))).collect();
    let [mstack] = mstacks.as_slice() else { return None };
    let Op::MStack(ops::MStack { buffers }) = mstack.op() else { unreachable!() };
    if buffers.len() < 2 {
        return None;
    }
    let expected_dtype = buffers[0].dtype();
    let expected_shape = buffers[0].shape().ok()??;
    if buffers.iter().any(|shard| {
        shard.dtype() != expected_dtype
            || shard.shape().ok().flatten() != Some(expected_shape)
            || shard.device_spec().is_none()
    }) {
        return None;
    }
    let device = buffers.first()?.device_spec()?;
    let widen_dtype = buffers
        .iter()
        .map(|shard| match shard.op() {
            Op::Cast(ops::Cast { src, .. }) if [DType::Float16, DType::BFloat16].contains(&src.dtype()) => {
                Some(src.dtype())
            }
            _ => None,
        })
        .collect::<Option<Vec<_>>>()
        .and_then(|dtypes| dtypes.iter().all(|dtype| dtype == &dtypes[0]).then(|| dtypes[0].clone()));

    let mut local_reductions = SmallVec::with_capacity(buffers.len());
    for shard in buffers {
        let substitutions = HashMap::from([(UOpKey(mstack.clone()), shard.clone())]);
        let local_shard = local.substitute(&substitutions);
        let reduced = local_shard.reduce_with_num_axes(ranges.clone(), *reduce_op, *num_axes);
        local_reductions.push(match &widen_dtype {
            Some(dtype) => reduced.cast(dtype.clone()),
            None => reduced,
        });
    }
    let collective = UOp::allreduce(UOp::mstack(local_reductions), device, *reduce_op);
    let result = if widen_dtype.is_some() { collective.cast(root.dtype()) } else { collective };
    Some(result.rtag(root.tag().clone()).rorigin(root.origin()))
}

fn lower_host_allreduce(root: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::AllReduce(ops::AllReduce { src, device, reduce_op }) = root.op() else { return None };
    if !matches!(reduce_op, ReduceOp::Add | ReduceOp::Max) {
        return None;
    }
    let Op::MStack(ops::MStack { buffers }) = src.op() else { return None };
    if buffers.len() < 2 || buffers.iter().any(|buffer| buffer.device_spec().is_none()) {
        return None;
    }
    // DeviceSpec cannot represent Tinygrad's tuple-valued collective target.
    // This subset returns one result on the first shard's device.
    if buffers.first()?.device_spec().as_ref() != Some(device) {
        return None;
    }

    // Host staging reads every shard before publishing, so materialized shard
    // zero is a safe in-place destination. The output and first input must be
    // the same UOp or schedule canonicalization can select the pre-reduction
    // source allocation as the callback destination.
    let materialized: SmallVec<[Arc<UOp>; 4]> = buffers.iter().map(|buffer| buffer.contiguous()).collect();
    let output = materialized[0].clone();
    let mut args = SmallVec::with_capacity(buffers.len() + 1);
    args.push(output.clone());
    args.extend(materialized.iter().cloned());
    let mut formals = smallvec![UOp::placeholder_like(&output, 0, svod_ir::AddrSpace::Global).ok()?];
    for (slot, buffer) in materialized.iter().enumerate() {
        formals.push(UOp::placeholder_like(buffer, slot + 1, svod_ir::AddrSpace::Global).ok()?);
    }
    let body = UOp::custom_function(CustomFunctionKind::AllReduce { reduce_op: *reduce_op }, formals);
    // Host collectives never reach `split_store`, so this is their only chance to be
    // attributed; the whole staged reduction belongs to the ALLREDUCE's scope.
    let call = body.call(
        args,
        CallInfo {
            name: Some("host_allreduce".into()),
            precompile: true,
            origin: root.origin(),
            origins: root.origin().into_iter().collect(),
            ..CallInfo::default()
        },
    );
    Some(output.after(smallvec![call]).rtag(root.tag().clone()).rorigin(root.origin()))
}

fn host_allreduce_dtype_supported(dtype: &DType) -> bool {
    matches!(
        dtype,
        DType::Scalar(
            ScalarDType::Float16
                | ScalarDType::BFloat16
                | ScalarDType::Float32
                | ScalarDType::Float64
                | ScalarDType::Int8
                | ScalarDType::Int16
                | ScalarDType::Int32
                | ScalarDType::Int64
                | ScalarDType::UInt8
                | ScalarDType::UInt16
                | ScalarDType::UInt32
                | ScalarDType::UInt64
        )
    )
}

fn permute_multi(root: &Arc<UOp>, multi: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Permute(ops::Permute { axes, .. }) = root.op() else { return None };
    let (local, axis) = multi_axis(multi)?;
    let new_axis = axes.iter().position(|&candidate| candidate == axis)?;
    Some(UOp::multi(root.with_sources(vec![local]), new_axis).rtag(root.tag().clone()).rorigin(root.origin()))
}

fn flip_multi(root: &Arc<UOp>, multi: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Flip(ops::Flip { axes, .. }) = root.op() else { return None };
    let (local, axis) = multi_axis(multi)?;
    if axes.get(axis).copied().unwrap_or(true) {
        return None;
    }
    Some(UOp::multi(root.with_sources(vec![local]), axis).rtag(root.tag().clone()).rorigin(root.origin()))
}

fn const_at(uop: &Arc<UOp>, axis: usize) -> Option<ConstValue> {
    match uop.op() {
        Op::Stack(ops::Stack { sources }) => match sources.get(axis)?.op() {
            Op::Const(value) => Some(value.0),
            _ => None,
        },
        Op::VConst(ops::VConst { values }) => values.get(axis).copied(),
        _ if axis == 0 => match uop.op() {
            Op::Const(value) => Some(value.0),
            _ => None,
        },
        _ => None,
    }
}

fn pad_multi(root: &Arc<UOp>, multi: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Pad(ops::Pad { begin_pads, end_pads, .. }) = root.op() else { return None };
    let (local, axis) = multi_axis(multi)?;
    if !matches!(const_at(begin_pads, axis), Some(ConstValue::Int(0) | ConstValue::UInt(0)))
        || !matches!(const_at(end_pads, axis), Some(ConstValue::Int(0) | ConstValue::UInt(0)))
    {
        return None;
    }
    Some(
        UOp::multi(root.with_sources(vec![local, begin_pads.clone(), end_pads.clone()]), axis)
            .rtag(root.tag().clone())
            .rorigin(root.origin()),
    )
}

fn move_mselect_before_movement(root: &Arc<UOp>, buffer: &Arc<UOp>, device_index: usize) -> Option<Arc<UOp>> {
    if !buffer.op().is_movement() {
        return None;
    }
    let mut sources: Vec<_> = buffer.op().sources().iter().map(|src| (*src).clone()).collect();
    sources[0] = sources[0].mselect(device_index);
    Some(buffer.with_sources(sources).rtag(root.tag().clone()).rorigin(root.origin()))
}

/// Tinygrad `multi_pm` clauses that have an exact representation in Svod.
pub fn multi_pm() -> TypedPatternMatcher {
    crate::patterns! {
        selected @ MSelect { buffer: MStack { buffers }, device_index: _ }
            => {
                let Op::MSelect(ops::MSelect { device_index, .. }) = selected.op() else { unreachable!() };
                buffers.get(*device_index).cloned()
            },
        selected @ MSelect { buffer, device_index: _ }
            if buffer.op().is_movement()
            => {
                let Op::MSelect(ops::MSelect { device_index, .. }) = selected.op() else { unreachable!() };
                move_mselect_before_movement(selected, buffer, *device_index)
            },
        root @ Reduce { src: multi @ Multi { src: _ }, ranges: _, reduce_op: _, num_axes: _ }
            => reduce_multi(root, multi),
        root @ Permute { src: multi @ Multi { src: _ }, axes: _ }
            => permute_multi(root, multi),
        root @ Flip { src: multi @ Multi { src: _ }, axes: _ }
            => flip_multi(root, multi),
        root @ Pad { src: multi @ Multi { src: _ }, begin_pads: _, end_pads: _ }
            => pad_multi(root, multi),
        root @ Cast { src: multi @ Multi { src: _ }, dtype: _ }
            => passthrough_unary_wrapper(root, multi),
        root @ BitCast { src: multi @ Multi { src: _ }, dtype: _ }
            => passthrough_unary_wrapper(root, multi),
        root @ Contiguous { src: multi @ Multi { src: _ }, opts: _ }
            => passthrough_unary_wrapper(root, multi),
        root @ Detach { src: multi @ Multi { src: _ } }
            => passthrough_unary_wrapper(root, multi),
        root @ ContiguousBackward { src: multi @ Multi { src: _ } }
            => passthrough_unary_wrapper(root, multi),
        root if matches!(root.op(), Op::Unary(..) | Op::Binary(..) | Op::Ternary(..))
            => rewrite_per_shard_alu(root),
    }
}

/// Lower represented collectives into opaque host-runtime calls before kernel
/// formation. The call body is never sent through `spec_program`.
pub fn lower_allreduce_pm() -> TypedPatternMatcher {
    crate::patterns! {
        root @ AllReduce { src: _, device: _, reduce_op: _ } => lower_host_allreduce(root),
    }
}

fn operation_name(op: &Op) -> &'static str {
    match op {
        Op::Unary(..) => "unary ALU",
        Op::Binary(..) => "binary ALU",
        Op::Ternary(..) => "ternary ALU",
        Op::ReduceAxis(..) | Op::Reduce(..) => "reduction",
        Op::Reshape(..) => "RESHAPE",
        Op::Permute(..) => "PERMUTE",
        Op::Expand(..) => "EXPAND",
        Op::Pad(..) => "PAD",
        Op::Shrink(..) => "SHRINK",
        Op::Flip(..) => "FLIP",
        Op::MSelect(..) => "MSELECT",
        _ => "operation",
    }
}

fn source_layout(source: &Arc<UOp>) -> MultiLayout {
    match source.op() {
        Op::Multi(ops::Multi { axis, .. }) => MultiLayout::Axis(*axis),
        _ => MultiLayout::None,
    }
}

fn classify_supported_form(node: &Arc<UOp>) -> svod_ir::Result<()> {
    if let Op::Multi(ops::Multi { src, axis }) = node.op() {
        if src.toposort().iter().any(|inner| matches!(inner.op(), Op::Multi(..))) {
            return Err(svod_ir::Error::MultiNested { axis: *axis });
        }
        let shape = src.shape()?.ok_or(svod_ir::Error::MultiUnsupported {
            operation: "MULTI",
            reason: "source layout has no inferable shape",
        })?;
        if *axis >= shape.len() {
            return Err(svod_ir::Error::MultiUnsupported {
                operation: "MULTI",
                reason: "shard axis is outside the source shape",
            });
        }
        if let Op::MStack(ops::MStack { buffers }) = src.op()
            && let Some(first) = buffers.first()
        {
            let expected_dtype = first.dtype();
            let expected_shape = first.shape()?.cloned();
            if buffers.iter().any(|shard| {
                shard.dtype() != expected_dtype
                    || shard.shape().ok().flatten() != expected_shape.as_ref()
                    || shard.device_spec().is_none()
            }) {
                return Err(svod_ir::Error::MultiUnsupported {
                    operation: "MULTI",
                    reason: "explicit shards must have identical dtype and shape with concrete devices",
                });
            }
        }
        return Ok(());
    }

    if let Op::MSelect(..) = node.op() {
        return Err(svod_ir::Error::MultiUnsupported {
            operation: "MSELECT",
            reason: "selection did not resolve to an in-range MSTACK shard",
        });
    }

    if let Op::AllReduce(ops::AllReduce { src, device, reduce_op }) = node.op() {
        let Op::MStack(ops::MStack { buffers }) = src.op() else {
            return Err(svod_ir::Error::MultiUnsupported {
                operation: "ALLREDUCE",
                reason: "collective source must be an explicit MSTACK",
            });
        };
        if buffers.len() < 2 {
            return Err(svod_ir::Error::MultiUnsupported {
                operation: "ALLREDUCE",
                reason: "collective requires at least two explicit shards",
            });
        }
        if !host_allreduce_dtype_supported(&src.dtype()) {
            return Err(svod_ir::Error::MultiUnsupported {
                operation: "ALLREDUCE",
                reason: "host collective dtype is not supported",
            });
        }
        if !matches!(reduce_op, ReduceOp::Add | ReduceOp::Max) {
            return Err(svod_ir::Error::MultiUnsupported {
                operation: "ALLREDUCE",
                reason: "only SUM and MAX collectives are supported",
            });
        }
        let expected_shape = buffers[0].shape()?.cloned();
        if buffers.iter().any(|buffer| {
            buffer.dtype() != src.dtype()
                || buffer.device_spec().is_none()
                || buffer.shape().ok().flatten() != expected_shape.as_ref()
        }) {
            return Err(svod_ir::Error::MultiUnsupported {
                operation: "ALLREDUCE",
                reason: "explicit shards must have identical dtype and shape with concrete devices",
            });
        }
        if buffers.first().and_then(|buffer| buffer.device_spec()).as_ref() != Some(device) {
            return Err(svod_ir::Error::MultiUnsupported {
                operation: "ALLREDUCE",
                reason: "single-output collective target must match the first shard device",
            });
        }
        return Ok(());
    }

    let layouts: Vec<_> = node.op().sources().iter().map(source_layout).collect();
    let mut axes: Vec<_> = layouts
        .iter()
        .filter_map(|layout| match layout {
            MultiLayout::Axis(axis) => Some(*axis),
            MultiLayout::None => None,
        })
        .collect();
    if axes.is_empty() {
        return Ok(());
    }

    // Graph containers do not combine their independent source layouts.
    if matches!(node.op(), Op::Sink(..) | Op::Group(..) | Op::Tuple(..)) {
        return Ok(());
    }
    axes.sort_unstable();
    axes.dedup();

    let operation = operation_name(node.op());
    if axes.len() != 1 {
        return Err(svod_ir::Error::MultiAxisMismatch { operation, axes });
    }
    let axis = axes[0];

    match node.op() {
        Op::Unary(..) | Op::Binary(..) | Op::Ternary(..) => {
            for (source, layout) in node.op().sources().iter().zip(layouts) {
                if layout == MultiLayout::None && !source.shape()?.is_some_and(|shape| shape.is_empty()) {
                    return Err(svod_ir::Error::MultiLayoutMissing { operation, axis, source_id: source.id });
                }
            }
            Err(svod_ir::Error::MultiUnsupported {
                operation,
                reason: "supported per-shard ALU did not normalize before rangeification",
            })
        }
        Op::ReduceAxis(ops::ReduceAxis { axes, .. }) => {
            if axes.contains(&axis) {
                Err(svod_ir::Error::MultiReductionAcrossShardAxis { axis })
            } else {
                Err(svod_ir::Error::MultiUnsupported {
                    operation,
                    reason: "non-sharded-axis reduction did not normalize before rangeification",
                })
            }
        }
        Op::Reduce(ops::Reduce { num_axes, .. }) => {
            if axis < *num_axes {
                Err(svod_ir::Error::MultiReductionAcrossShardAxis { axis })
            } else {
                Err(svod_ir::Error::MultiUnsupported {
                    operation,
                    reason: "non-sharded-axis reduction did not normalize before rangeification",
                })
            }
        }
        Op::Reshape(..) => Err(svod_ir::Error::MultiMovementUnsupported {
            operation,
            axis,
            reason: "the shard boundary cannot be mapped without shard-count metadata",
        }),
        op if op.is_movement() => Err(svod_ir::Error::MultiMovementUnsupported {
            operation,
            axis,
            reason: "the movement crosses or cannot prove preservation of the shard boundary",
        }),
        _ => Err(svod_ir::Error::MultiUnsupported {
            operation,
            reason: "no hardware-independent per-shard rewrite is defined",
        }),
    }
}

/// Reject every unresolved form outside the exact hardware-independent subset.
pub fn validate_supported_subset(root: &Arc<UOp>) -> svod_ir::Result<()> {
    for node in root.toposort_call_aware(true) {
        classify_supported_form(&node)?;
    }
    Ok(())
}

/// Final scheduling boundary: collectives must have become opaque runtime calls.
pub fn validate_no_unresolved_allreduce(root: &Arc<UOp>) -> svod_ir::Result<()> {
    if root.toposort_call_aware(true).iter().any(|node| matches!(node.op(), Op::AllReduce(..))) {
        return Err(svod_ir::Error::MultiUnsupported {
            operation: "ALLREDUCE",
            reason: "collective did not lower to an executable host schedule item",
        });
    }
    Ok(())
}
