//! Shaped upcast/unroll expansion.
//!
//! This is the `expander2` model from Tinygrad 8c8b43de. Upcast and unroll
//! ranges become shaped coordinates. WMMA expansion metadata directly shapes
//! the hardware operands and reconstructs the output coordinates.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

use smallvec::{SmallVec, smallvec};
use svod_ir::{AxisId, AxisType, ConstValue, Op, SInt, UOp};

use crate::TypedPatternMatcher;
use svod_ir::ops;

pub type RangeMap = HashMap<AxisId, usize>;

/// Assign every UPCAST/UNROLL range a stable shaped-coordinate position.
pub fn build_range_map(sink: &Arc<UOp>) -> RangeMap {
    let mut ranges = HashMap::new();
    for node in sink.toposort() {
        if let Op::Range(ops::Range { axis_id, axis_type: AxisType::Upcast | AxisType::Unroll, .. }) = node.op() {
            let next = ranges.len();
            ranges.entry(axis_id.clone()).or_insert(next);
        }
    }
    ranges
}

fn static_extent(value: &ConstValue) -> Option<usize> {
    match value {
        ConstValue::Int(value) if *value >= 0 => Some(*value as usize),
        ConstValue::UInt(value) => Some(*value as usize),
        _ => None,
    }
}

fn expand_range(ctx: &RangeMap, range: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Range(ops::Range { end, axis_id, .. }) = range.op() else { return None };
    let position = *ctx.get(axis_id)?;
    let Op::Const(value) = end.op() else { return None };
    let extent = static_extent(&value.0)?;
    let values: SmallVec<[Arc<UOp>; 4]> =
        (0..extent).map(|value| UOp::const_(range.dtype(), ConstValue::Int(value as i64))).collect();
    let mut shape = smallvec![SInt::Const(1); ctx.len()];
    shape[position] = SInt::Const(extent);
    UOp::stack(values).try_reshape(&shape).ok()
}

/// Move the selected coordinate dimensions to the tail and flatten them.
fn contract_axis(ctx: &RangeMap, source: &Arc<UOp>, axes: &[(AxisId, usize)]) -> Option<Arc<UOp>> {
    if axes.is_empty() {
        return Some(source.clone());
    }
    let shape = source.shape().ok().flatten()?;
    let tail: Vec<usize> = axes.iter().map(|(axis, _)| ctx.get(axis).copied()).collect::<Option<_>>()?;
    if tail.iter().any(|&position| position >= shape.len()) {
        return None;
    }
    for ((_, extent), &position) in axes.iter().zip(&tail) {
        if shape[position].as_const() != Some(*extent) {
            return None;
        }
    }
    let head: Vec<usize> = (0..shape.len()).filter(|position| !tail.contains(position)).collect();
    let permutation: Vec<usize> = head.iter().chain(&tail).copied().collect();
    let permuted = source.try_permute(permutation).ok()?;
    let permuted_shape = permuted.shape().ok().flatten()?;
    let mut output_shape: svod_ir::shape::Shape = permuted_shape[..head.len()].into();
    output_shape.push(svod_ir::sint_prod(&permuted_shape[head.len()..]));
    permuted.try_reshape(&output_shape).ok()
}

/// Restore flattened coordinate dimensions to their target coordinate order.
fn unroll_axis(ctx: &RangeMap, source: &Arc<UOp>, axes: &[(AxisId, usize)]) -> Option<Arc<UOp>> {
    if axes.is_empty() {
        return Some(source.clone());
    }
    let shape = source.shape().ok().flatten()?;
    let (last, prefix) = shape.split_last()?;
    let extent = axes.iter().map(|(_, extent)| extent).product::<usize>();
    if last.as_const() != Some(extent) {
        return None;
    }
    let tail: Vec<usize> = axes.iter().map(|(axis, _)| ctx.get(axis).copied()).collect::<Option<_>>()?;
    let mut expanded_shape: svod_ir::shape::Shape = prefix.into();
    expanded_shape.extend(axes.iter().map(|(_, extent)| SInt::Const(*extent)));
    let expanded = source.try_reshape(&expanded_shape).ok()?;

    let head: Vec<usize> = (0..expanded_shape.len()).filter(|position| !tail.contains(position)).collect();
    let order: Vec<usize> = head.iter().chain(&tail).copied().collect();
    if order.len() != expanded_shape.len() || order.iter().any(|&position| position >= order.len()) {
        return None;
    }
    let mut inverse = vec![0; order.len()];
    for (position, &axis) in order.iter().enumerate() {
        inverse[axis] = position;
    }
    expanded.try_permute(inverse).ok()
}

fn expand_wmma(
    ctx: &RangeMap,
    a: &Arc<UOp>,
    b: &Arc<UOp>,
    c: &Arc<UOp>,
    metadata: &svod_ir::WmmaMetadata,
) -> Option<Arc<UOp>> {
    let axes = metadata.upcast_axes.as_ref()?;
    let mut expanded_metadata = metadata.clone();
    expanded_metadata.upcast_axes = None;
    let wmma =
        UOp::wmma(contract_axis(ctx, a, &axes.a)?, contract_axis(ctx, b, &axes.b)?, c.clone(), expanded_metadata);
    unroll_axis(ctx, &wmma, &axes.c)
}

/// Convert shaped non-range REDUCE inputs into leading horizontal axes.
fn expand_reduce(reduce: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Reduce(ops::Reduce { src, ranges, reduce_op, num_axes }) = reduce.op() else { return None };
    if *num_axes != 0 {
        return None;
    }
    let mut loop_ranges = SmallVec::new();
    let mut horizontal_axes = Vec::new();
    for range in ranges {
        if matches!(range.op(), Op::Range(..)) {
            loop_ranges.push(range.clone());
        } else {
            let shape = range.shape().ok().flatten()?;
            horizontal_axes
                .extend(shape.iter().enumerate().filter_map(|(axis, extent)| (extent.as_const()? > 1).then_some(axis)));
        }
    }
    horizontal_axes.sort_unstable();
    horizontal_axes.dedup();
    if horizontal_axes.is_empty() {
        return None;
    }

    let source_shape = match src.shape() {
        Ok(Some(shape)) => shape,
        Ok(None) => return None,
        Err(error) => {
            tracing::trace!(?error, source = src.tree(), "cannot infer shaped reduction source");
            return None;
        }
    };
    let permutation: Vec<usize> = horizontal_axes
        .iter()
        .copied()
        .chain((0..source_shape.len()).filter(|axis| !horizontal_axes.contains(axis)))
        .collect();
    let permuted = src.try_permute(permutation).ok()?;
    let reduced = UOp::new(
        Op::Reduce(ops::Reduce {
            src: permuted,
            ranges: loop_ranges,
            reduce_op: *reduce_op,
            num_axes: horizontal_axes.len(),
        }),
        reduce.dtype(),
    );
    let output_shape: svod_ir::shape::Shape = source_shape
        .iter()
        .enumerate()
        .map(|(axis, extent)| if horizontal_axes.contains(&axis) { SInt::Const(1) } else { extent.clone() })
        .collect();
    reduced.try_reshape(&output_shape).ok()
}

/// Tinygrad `expander2`: shaped ranges, horizontal reductions, and direct WMMA expansion.
pub fn expander2() -> &'static TypedPatternMatcher<RangeMap> {
    crate::cached_patterns! {
        @context RangeMap;
        reduce @ Reduce { .. } => expand_reduce(reduce),
        range @ Range { end: _, axis_id, axis_type }
            if matches!(axis_type, AxisType::Upcast | AxisType::Unroll) && ctx.contains_key(axis_id)
            => expand_range(ctx, range),
        Wmma { a, b, c, metadata } if metadata.upcast_axes.is_some() => expand_wmma(ctx, a, b, c, metadata),
    }
}

pub fn pre_expand(ast: &Arc<UOp>) -> Arc<UOp> {
    static PM: LazyLock<TypedPatternMatcher<RangeMap>> = LazyLock::new(|| {
        expander2().clone()
            + crate::rangeify::pm_flatten_range().clone().with_context::<RangeMap>()
            + crate::devectorize::mop_cleanup_patterns().with_context::<RangeMap>()
    });
    let mut range_map = build_range_map(ast);
    crate::rewrite::graph_rewrite(&*PM, ast.clone(), &mut range_map)
}

fn fix_group_for_reduce(reduce: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Reduce(ops::Reduce { src, reduce_op, ranges, num_axes }) = reduce.op() else { return None };
    let (grouped, other): (Vec<_>, Vec<_>) = ranges
        .iter()
        .partition(|range| matches!(range.op(), Op::Range(ops::Range { axis_type: AxisType::GroupReduce, .. })));
    if grouped.is_empty() {
        return None;
    }
    let locals: Vec<_> = reduce
        .toposort()
        .into_iter()
        .filter(|node| matches!(node.op(), Op::Range(ops::Range { axis_type: AxisType::Local, .. })))
        .collect();
    let partial = UOp::new(
        Op::Reduce(ops::Reduce {
            src: src.clone(),
            ranges: other.into_iter().cloned().collect(),
            reduce_op: *reduce_op,
            num_axes: *num_axes,
        }),
        reduce.dtype(),
    );
    let loops: Vec<_> = grouped
        .iter()
        .filter_map(|range| match range.op() {
            Op::Range(ops::Range { end, axis_id, deps, .. }) => Some(UOp::new(
                Op::Range(ops::Range {
                    end: end.clone(),
                    axis_id: axis_id.group_reduce_loop(),
                    axis_type: AxisType::Reduce,
                    deps: deps.clone(),
                }),
                range.dtype(),
            )),
            _ => None,
        })
        .collect();
    let buffer_ranges = locals.iter().cloned().chain(grouped.iter().map(|range| (*range).clone())).collect();
    let grouped_axis = match grouped[0].op() {
        Op::Range(ops::Range { axis_id, .. }) => axis_id.clone(),
        _ => unreachable!("grouped reductions contain RANGE sources"),
    };
    let buffer = UOp::stage(partial, buffer_ranges, svod_ir::BufferizeOpts::local_for_axis(grouped_axis));
    let indices: Vec<_> = locals.iter().cloned().chain(loops.iter().cloned()).collect();
    let indexed = UOp::index().buffer(buffer).indices(indices).call().ok()?;
    Some(indexed.reduce_with_num_axes(loops.into_iter().collect(), *reduce_op, 0))
}

/// Grouped reduction lowering runs with reduction removal, after expansion.
pub fn pm_group_for_reduce() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        reduce @ Reduce { .. } => fix_group_for_reduce(reduce),
    }
}
