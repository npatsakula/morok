use std::collections::HashSet;
use std::sync::{Arc, LazyLock};

use smallvec::{SmallVec, smallvec};
use svod_dtype::AddrSpace;
use svod_ir::ops;
use svod_ir::{AxisType, ConstValue, Op, TypedPatternMatcher, UOp, UOpKey};

fn access_buffer(uop: &Arc<UOp>) -> Option<Arc<UOp>> {
    match uop.op() {
        Op::Param(..) | Op::Buffer(..) | Op::MSelect(..) | Op::MStack(..) => Some(uop.clone()),
        Op::Index(ops::Index { buffer, .. }) | Op::After(ops::After { passthrough: buffer, .. }) => {
            access_buffer(buffer)
        }
        Op::Cast(ops::Cast { src, .. })
        | Op::Reshape(ops::Reshape { src, .. })
        | Op::Permute(ops::Permute { src, .. })
        | Op::Expand(ops::Expand { src, .. })
        | Op::Pad(ops::Pad { src, .. })
        | Op::Shrink(ops::Shrink { src, .. })
        | Op::Flip(ops::Flip { src, .. }) => access_buffer(src),
        _ => None,
    }
}

fn is_local_store(uop: &Arc<UOp>) -> bool {
    matches!(uop.op(), Op::Store(..)) && uop.addrspace() == Some(AddrSpace::Local)
}

fn barrier_from_sources(sources: &[Arc<UOp>]) -> Option<Arc<UOp>> {
    let (src, deps) = sources.split_first()?;
    Some(src.barrier(deps.iter().cloned().collect()))
}

fn add_raw_barrier(after: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::After(ops::After { passthrough, deps }) = after.op() else { return None };
    if after.addrspace() != Some(AddrSpace::Local) {
        return None;
    }

    // Match Tinygrad's single gated toposort over SINK(*after.src[1:]).
    let dependency_sink = UOp::sink(deps.iter().cloned().collect());
    let dependency_toposort = dependency_sink.toposort_filtered(|uop| !matches!(uop.op(), Op::Barrier(..)));
    if !dependency_toposort.iter().any(is_local_store) {
        return None;
    }

    Some(passthrough.after(smallvec![barrier_from_sources(deps)?]))
}

fn add_war_barrier(end: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::End(ops::End { computation, ranges }) = end.op() else { return None };
    if matches!(computation.op(), Op::Barrier(..)) {
        return None;
    }

    let loop_ranges: Vec<_> = ranges
        .iter()
        .filter(|range| {
            matches!(
                range.op(),
                Op::Range(ops::Range { axis_type: AxisType::Reduce | AxisType::Weak | AxisType::Loop, .. })
            ) && matches!(range.vmax(), ConstValue::Int(vmax) if *vmax > 0)
        })
        .cloned()
        .collect();
    if loop_ranges.is_empty() {
        return None;
    }

    // `toposort` covers the same node set as `backward_slice_with_self` but uses
    // the pre-sized FxHashSet visited set instead of a fresh SipHash pointer set.
    let backward_slice = computation.toposort();
    let loop_range_ids: HashSet<_> = loop_ranges.iter().map(|range| range.id).collect();
    let store_buffers: HashSet<_> = backward_slice
        .iter()
        .filter(|uop| is_local_store(uop) && uop.in_scope_ranges().iter().any(|id| loop_range_ids.contains(id)))
        .filter_map(|uop| match uop.op() {
            Op::Store(ops::Store { index, .. }) => access_buffer(index).map(UOpKey),
            _ => None,
        })
        .collect();

    let loads: SmallVec<[Arc<UOp>; 4]> = backward_slice
        .iter()
        .filter(|uop| match uop.op() {
            Op::Load(ops::Load { index, .. }) => {
                access_buffer(index).is_some_and(|buffer| store_buffers.contains(&UOpKey(buffer)))
            }
            _ => false,
        })
        .cloned()
        .collect();
    if loads.is_empty() {
        return None;
    }

    Some(computation.barrier(loads).end(ranges.clone()))
}

pub(crate) fn pm_implicit_barriers() -> &'static TypedPatternMatcher {
    static PM: LazyLock<TypedPatternMatcher> = LazyLock::new(|| {
        crate::patterns! {
            after @ After { passthrough: _, deps: _ } => add_raw_barrier(after),
            end @ End { computation: _, ranges: _ } => add_war_barrier(end),
        }
    });
    &PM
}
