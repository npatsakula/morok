//! Consolidated pattern matchers for rangeify transformations.
//!
//! This module contains all pattern matchers used during scheduling/rangeify:
//! - Early cleanup rewrites (DETACH, CONTIGUOUS_BACKWARD removal)
//! - Movement op → BUFFERIZE conversion
//! - Buffer folding and removal
//! - Kernel splitting patterns (BUFFER → DEFINE_GLOBAL, AFTER handling)
//! - Codegen preparation (NOOP removal, INDEX linearization)
//! - Buffer limit enforcement
//!
//! Consolidated from: patterns.rs, codegen_patterns.rs, movement_patterns.rs,
//! split_patterns.rs, buffer_limits.rs

use std::collections::HashSet;
use std::sync::Arc;

use morok_device::DeviceSpec;
use morok_dtype::{AddrSpace, DType};
use morok_ir::{AxisId, AxisType, BufferizeOpts, ConstValue, Op, UOp, UOpKey};
use smallvec::SmallVec;

use crate::pattern::UPat;
use crate::pattern::matcher::{PatternMatcher, RewriteFn, RewriteResult};

// Re-export from indexing for backwards compatibility
pub use super::indexing::{is_dead_axis, ranges_equal};

// Forward declarations for types from other modules
use super::indexing::IndexingContext;
use super::kernel::KernelContext;
use super::kernel::{PcontigConfig, SplitReduceOpConfig, split_reduceop};
use super::transforms::{reduce_unparented, should_remove_movement_op, transform_sources_with_bufferize};

// ============================================================================
// HELPER FUNCTIONS (private)
// ============================================================================

/// Check if a shape UOp represents a scalar (empty or Vectorize with 0 elements).
fn is_scalar_shape(shape: &Arc<UOp>) -> bool {
    match shape.op() {
        Op::Vectorize { elements } => elements.is_empty(),
        _ => false,
    }
}

/// Check if an op is cheap to inline (no buffering needed).
pub fn is_cheap_to_inline(op: &Op) -> bool {
    matches!(
        op,
        // Nullary - always cheap
        Op::Const(_)
            | Op::Unique(_)
            | Op::Device(_)
            | Op::Noop
            | Op::DefineVar { .. }
            | Op::DefineReg { .. }
            | Op::VConst { .. }
            // Simple operations - cheap to recompute
            | Op::Unary(..)
            | Op::Binary(..)
            | Op::Ternary(..)
            | Op::Cast { .. }
            | Op::BitCast { .. }
            // Vector operations - cheap
            | Op::Gep { .. }
            | Op::Vectorize { .. }
            // Index/pointer operations - cheap
            | Op::PointerIndex { .. }
    )
}

/// Check if an op must always run (shouldn't be buffered).
pub fn is_always_run_op(op: &Op) -> bool {
    matches!(op, Op::Contiguous { .. } | Op::Copy { .. } | Op::Assign { .. })
}

/// Check if an INDEX operation has multiple indices.
fn has_multiple_indices(idx: &Arc<UOp>) -> bool {
    if let Op::Index { indices, .. } = idx.op() { indices.len() > 1 } else { false }
}

/// Check if an INDEX has another INDEX as its buffer.
fn is_cascaded_index(idx: &Arc<UOp>) -> bool {
    if let Op::Index { buffer, .. } = idx.op() { matches!(buffer.op(), Op::Index { .. }) } else { false }
}

/// Check if operation is elementwise (Binary or Ternary).
pub fn is_elementwise(uop: &Arc<UOp>) -> bool {
    matches!(uop.op(), Op::Binary(..) | Op::Ternary(..))
}

// ============================================================================
// EARLY CLEANUP PATTERNS
// ============================================================================

/// Pattern matcher for early cleanup rewrites during scheduling.
///
/// This handles schedule-specific cleanup:
/// - DETACH removal (gradient computation marker no longer needed)
/// - CONTIGUOUS_BACKWARD removal (gradient computation marker no longer needed)
/// - RESHAPE to scalar (empty shape) removal
/// - RESHAPE on REDUCE removal (REDUCE output doesn't need reshaping)
pub fn early_rewrites() -> PatternMatcher {
    crate::patterns! {
        Detach(x) ~> x,
        ContiguousBackward(x) ~> x,
        x => {
            if let Op::Reshape { src, new_shape } = x.op() {
                // RESHAPE to scalar - always remove
                if is_scalar_shape(new_shape) {
                    return Some(src.clone());
                }
                // RESHAPE on REDUCE - the reduce output is already "indexed" by its ranges
                // The reshape just changes the view of the buffer, which can be done in STORE indexing
                if matches!(src.op(), Op::Reduce { .. }) {
                    return Some(src.clone());
                }
            }
            None
        }
    }
}

// ============================================================================
// RANGEIFY TRANSFORMATION PATTERNS
// ============================================================================

/// Pattern matcher for removing movement ops after rangeify transformation.
pub fn movement_op_removal() -> PatternMatcher<IndexingContext> {
    crate::patterns! {
        @context IndexingContext;
        x if x.op().is_movement() => remove_movement_op(x, ctx),
    }
}

/// Create patterns for applying rangeify transformation with IndexingContext.
pub fn apply_rangeify_patterns() -> PatternMatcher<IndexingContext> {
    crate::patterns! {
        @context IndexingContext;
        // ReduceAxis conversion MUST come first - before bufferize wraps it
        x if matches!(x.op(), Op::ReduceAxis { .. }) => convert_reduceaxis_with_context(x, ctx),
        x => apply_bufferize_transform(x, ctx),
        x if x.op().is_movement() => remove_movement_op(x, ctx),
    }
}

/// Apply BUFFERIZE transformation to op sources.
fn apply_bufferize_transform(x: &Arc<UOp>, ctx: &mut IndexingContext) -> Option<Arc<UOp>> {
    if let Some(new_sources) = transform_sources_with_bufferize(x, ctx) {
        return Some(x.with_sources(new_sources));
    }
    None
}

/// Remove movement ops after transformation has been applied.
fn remove_movement_op(x: &Arc<UOp>, ctx: &mut IndexingContext) -> Option<Arc<UOp>> {
    if should_remove_movement_op(x, ctx) {
        return x.op().sources().first().map(Arc::clone);
    }
    None
}

/// Convert ReduceAxis → REDUCE using IndexingContext.
fn convert_reduceaxis_with_context(x: &Arc<UOp>, ctx: &mut IndexingContext) -> Option<Arc<UOp>> {
    let Op::ReduceAxis { src, reduce_op, axes } = x.op() else {
        return None;
    };

    let (input_ranges, output_ranges) = ctx.get_ranges(x)?;
    let reduce_ranges: SmallVec<[Arc<UOp>; 4]> = input_ranges
        .iter()
        .enumerate()
        .filter_map(|(i, range)| {
            if axes.contains(&i) {
                if let Op::Range { axis_type: AxisType::Reduce, .. } = range.op() {
                    Some(Arc::clone(range))
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect();

    if reduce_ranges.is_empty() {
        return Some(Arc::clone(src));
    }

    let ret = UOp::reduce(Arc::clone(src), reduce_ranges, *reduce_op);

    // KEY: Transfer range_map to new UOp (like Tinygrad line 94)
    // When graph_rewrite creates a new UOp, the old range_map entry becomes invalid.
    // We must copy the ranges to the new UOp's identity.
    ctx.set_ranges(&ret, input_ranges.clone(), output_ranges.clone());

    // Also transfer ending_ranges to the new UOp
    let ending = ctx.get_ending_ranges(x);
    ctx.set_ending_ranges(&ret, ending);

    // Transfer realize_map to new UOp - critical for nested reductions!
    // If the original ReduceAxis was marked for realization, the new REDUCE must be too.
    if let Some(axes) = ctx.get_realize_axes(x).cloned() {
        ctx.mark_realize(&ret, axes);
    }

    Some(ret)
}

// ============================================================================
// BUFFER FOLDING PATTERNS
// ============================================================================

/// Pattern matcher for buffer folding and constant propagation.
pub fn buffer_folding() -> PatternMatcher {
    crate::patterns! {
        Bufferize { compute: c, .. } if matches!(c.op(), Op::Const(_)) ~> c,
        Index { buffer: c, .. } if matches!(c.op(), Op::Const(_)) ~> c,
        Copy { src: c, .. } if matches!(c.op(), Op::Const(_)) ~> c,
        Index { buffer: Bufferize { compute, ranges, .. }, indices, gate: None }
            if ranges_equal(&ranges, &indices) ~> compute,
    }
}

/// Pattern matcher for dead axis removal.
pub fn dead_axis_removal() -> PatternMatcher {
    crate::patterns! {
        buf if matches!(buf.op(), Op::Bufferize { .. }) => filter_dead_bufferize(buf),
        idx if matches!(idx.op(), Op::Index { .. }) => adjust_index_for_dead_axes(idx),
    }
}

/// Filter dead axes from BUFFERIZE operations.
fn filter_dead_bufferize(buf: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Bufferize { compute, ranges, opts } = buf.op() else {
        return None;
    };

    let live_ranges: Vec<_> = ranges.iter().filter(|r| !is_dead_axis(r)).map(Arc::clone).collect();

    if live_ranges.len() < ranges.len() {
        if live_ranges.is_empty() {
            return Some(Arc::clone(compute));
        }
        return Some(UOp::bufferize(Arc::clone(compute), live_ranges, opts.clone()));
    }
    None
}

/// Adjust INDEX indices when BUFFERIZE has dead axes removed.
fn adjust_index_for_dead_axes(idx: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Index { buffer, indices, gate: _ } = idx.op() else {
        return None;
    };
    let Op::Bufferize { ranges: buf_ranges, .. } = buffer.op() else {
        return None;
    };

    let mut new_indices = Vec::new();
    let mut idx_iter = indices.iter();

    for range in buf_ranges.iter() {
        if !is_dead_axis(range) {
            if let Some(idx_val) = idx_iter.next() {
                new_indices.push(Arc::clone(idx_val));
            }
        } else {
            idx_iter.next();
        }
    }

    if new_indices.len() < indices.len() {
        if new_indices.is_empty() {
            return None;
        }
        return UOp::index(Arc::clone(buffer), new_indices).ok();
    }
    None
}

// ============================================================================
// BUFFER REMOVAL PATTERNS
// ============================================================================

/// Pattern matcher for cost-based buffer removal.
pub fn buffer_removal() -> PatternMatcher {
    crate::patterns! {
        Bufferize { compute, .. } if is_cheap_to_inline(compute.op()) ~> compute,
        Bufferize { compute, .. } if is_always_run_op(compute.op()) ~> compute,
        Bufferize { compute: Bufferize { compute: inner, .. }, ranges, opts }
            => Some(UOp::bufferize(Arc::clone(inner), ranges.to_vec(), opts.clone())),
    }
}

/// Pattern matcher for cost-based buffer removal with partial contiguous support.
pub fn buffer_removal_with_pcontig() -> PatternMatcher<PcontigConfig> {
    crate::patterns! {
        @context PcontigConfig;
        idx if matches!(idx.op(), Op::Index { .. }) => apply_pcontig_removal(idx, ctx),
        buf if ctx.level > 0 && matches!(buf.op(), Op::Bufferize { .. }) => remove_cheap_bufferize(buf),
        buf if ctx.level > 0 && matches!(buf.op(), Op::Bufferize { .. }) => remove_always_run_bufferize(buf),
        Bufferize { compute: Bufferize { compute: inner, .. }, ranges, opts }
            => Some(UOp::bufferize(Arc::clone(inner), ranges.to_vec(), opts.clone())),
    }
}

fn remove_cheap_bufferize(buf: &Arc<UOp>) -> Option<Arc<UOp>> {
    if let Op::Bufferize { compute, .. } = buf.op()
        && is_cheap_to_inline(compute.op())
    {
        return Some(compute.clone());
    }
    None
}

fn remove_always_run_bufferize(buf: &Arc<UOp>) -> Option<Arc<UOp>> {
    if let Op::Bufferize { compute, .. } = buf.op()
        && is_always_run_op(compute.op())
    {
        return Some(compute.clone());
    }
    None
}

/// Apply partial contiguous buffer removal.
fn apply_pcontig_removal(idx: &Arc<UOp>, config: &mut PcontigConfig) -> Option<Arc<UOp>> {
    use super::kernel::{
        apply_partial_contiguous, calculate_buffer_size, calculate_out_in_ratio, collect_accessed_buffers,
        collect_indexes, collect_local_indexes, collect_reduces, extract_exclude_ranges, has_buffer_in_reduce,
        partition_ranges,
    };

    let Op::Index { buffer, indices: idx_ranges, gate: None } = idx.op() else {
        return None;
    };
    let Op::Bufferize { compute: src, ranges: buf_ranges, .. } = buffer.op() else {
        return None;
    };

    if config.level == 0 {
        return None;
    }

    if is_always_run_op(src.op()) {
        return None;
    }

    let accessed_buffers = collect_accessed_buffers(src);
    if accessed_buffers.len() > config.max_buffers_threshold {
        return None;
    }

    if let Some(output_size) = calculate_buffer_size(buffer)
        && let Some(ratio) = calculate_out_in_ratio(output_size, &accessed_buffers)
        && ratio < config.out_in_ratio_threshold
    {
        return None;
    }

    let reduces = collect_reduces(src);
    let buf_in_reduce = has_buffer_in_reduce(&reduces);

    if !buf_in_reduce {
        use std::collections::HashMap;
        #[allow(clippy::mutable_key_type)]
        let subs_map: HashMap<UOpKey, Arc<UOp>> =
            buf_ranges.iter().zip(idx_ranges.iter()).map(|(k, v)| (UOpKey(Arc::clone(k)), Arc::clone(v))).collect();
        return Some(src.substitute(&subs_map));
    }

    let indexes = collect_indexes(src);
    let local_indexes = collect_local_indexes(&indexes);
    #[allow(clippy::mutable_key_type)]
    let exclude_ranges = extract_exclude_ranges(&local_indexes);

    let (materialize, substitute) = partition_ranges(buf_ranges, idx_ranges, &exclude_ranges);

    if materialize.is_empty() {
        use std::collections::HashMap;
        #[allow(clippy::mutable_key_type)]
        let subs_map: HashMap<UOpKey, Arc<UOp>> = substitute.into_iter().map(|(k, v)| (UOpKey(k), v)).collect();
        return Some(src.substitute(&subs_map));
    }

    apply_partial_contiguous(src, materialize, substitute)
}

// ============================================================================
// REDUCTION SIMPLIFY PATTERNS
// ============================================================================

/// Pattern matcher for reduction simplifications.
pub fn reduction_simplify_patterns() -> PatternMatcher<SplitReduceOpConfig> {
    crate::patterns! {
        @context SplitReduceOpConfig;
        x => reduce_unparented(x),
        x => super::transforms::reduce_collapse(x),
        reduce if matches!(reduce.op(), Op::ReduceAxis { .. }) => split_reduceop(reduce, ctx),
    }
}

// ============================================================================
// MOVEMENT OP PATTERNS (pm_mops equivalent)
// ============================================================================

/// Create pattern matcher for pushing movement ops through INDEX operations.
pub fn movement_op_patterns() -> PatternMatcher {
    crate::patterns! {
        Index { buffer: mop, indices, gate } if mop.op().is_movement() => {
            transform_movement_through_index(mop, &indices, &gate)
        }
        Index { buffer: inner_idx, indices, gate: None }
            if matches!(inner_idx.op(), morok_ir::Op::Index { .. }) => {
            flatten_nested_index(inner_idx, &indices)
        }
    }
}

/// Transform a movement op through INDEX by applying the movement to indices.
fn transform_movement_through_index(
    mop: &Arc<UOp>,
    indices: &SmallVec<[Arc<UOp>; 4]>,
    gate: &Option<Arc<UOp>>,
) -> Option<Arc<UOp>> {
    use super::indexing::apply_movement_op;

    let src = &mop.op().sources()[0];
    let src_shape = src.shape().ok()??;
    let transformed = apply_movement_op(mop.op(), src_shape, indices.as_slice());

    match gate {
        Some(g) => UOp::index_gated(src.clone(), transformed, g.clone()),
        None => UOp::index(src.clone(), transformed),
    }
    .ok()
}

/// Flatten nested INDEX operations.
fn flatten_nested_index(inner_idx: &Arc<UOp>, outer_indices: &SmallVec<[Arc<UOp>; 4]>) -> Option<Arc<UOp>> {
    if let morok_ir::Op::Index { indices: inner_indices, gate: None, .. } = inner_idx.op()
        && inner_indices.len() == 1
        && outer_indices.len() == 1
        && inner_indices[0].id == outer_indices[0].id
    {
        return Some(inner_idx.clone());
    }
    None
}

// ============================================================================
// CODEGEN PREPARATION PATTERNS
// ============================================================================

/// Replace NOOP operations with actual zero values.
pub fn remove_noop(noop: &Arc<UOp>) -> Option<Arc<UOp>> {
    if !matches!(noop.op(), Op::Noop) {
        return None;
    }

    let dtype = noop.dtype();
    let base = dtype.base();

    if base == morok_dtype::ScalarDType::Void {
        return None;
    }

    if dtype.is_vector() {
        let count = dtype.count();
        let zero_value = ConstValue::zero(base);
        let zeros: SmallVec<[Arc<UOp>; 4]> = (0..count).map(|_| UOp::const_(DType::Scalar(base), zero_value)).collect();
        return Some(UOp::vectorize(zeros));
    }

    let zero_value = ConstValue::zero(base);
    Some(UOp::const_(dtype, zero_value))
}

/// Remove CONTIGUOUS markers.
pub fn get_contiguous(contiguous: &Arc<UOp>) -> Option<Arc<UOp>> {
    if !matches!(contiguous.op(), Op::Contiguous { .. }) {
        return None;
    }
    Some(contiguous.op().sources()[0].clone())
}

/// Fix AFTER operations wrapping EXPAND (broadcast).
pub fn fix_after_broadcast(after: &Arc<UOp>) -> Option<Arc<UOp>> {
    let (passthrough, deps) = match after.op() {
        Op::After { passthrough, deps } => (passthrough, deps),
        _ => return None,
    };

    let expand_src = match passthrough.op() {
        Op::Expand { src, .. } => src,
        _ => return None,
    };

    #[allow(clippy::mutable_key_type)]
    let consumer_map = expand_src.get_consumer_map();
    let has_range_parents = consumer_map
        .get(&UOpKey(expand_src.clone()))
        .map(|consumers| consumers.iter().any(|c| matches!(c.op(), Op::Range { .. })))
        .unwrap_or(false);

    if has_range_parents {
        panic!("can't have a local AFTER");
    }

    Some(UOp::after(expand_src.clone(), deps.clone()))
}

/// Linearize multi-index INDEX to single-index INDEX.
pub fn linearize_index(idx: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Index { buffer, indices, gate } = idx.op() else {
        return None;
    };

    if indices.len() <= 1 {
        return None;
    }

    let dims: Vec<i64> = indices
        .iter()
        .map(|idx_uop| {
            if let Op::Range { end, .. } = idx_uop.op()
                && let Op::Const(cv) = end.op()
                && let ConstValue::Int(size) = cv.0
            {
                return size;
            }
            1
        })
        .collect();

    let mut strides = vec![1i64; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }

    let mut linear = UOp::index_const(0);
    for (idx_uop, &stride) in indices.iter().zip(strides.iter()) {
        let stride_uop = UOp::index_const(stride);
        let term = idx_uop.try_mul(&stride_uop).ok()?;
        linear = linear.try_add(&term).ok()?;
    }

    match gate {
        Some(g) => UOp::index_gated(buffer.clone(), vec![linear], g.clone()).ok(),
        None => UOp::index(buffer.clone(), vec![linear]).ok(),
    }
}

/// Flatten cascaded INDEX operations (INDEX of INDEX).
pub fn flatten_cascaded_index(idx: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Index { buffer: inner_idx, indices: outer_indices, gate: outer_gate } = idx.op() else {
        return None;
    };

    let Op::Index { buffer: real_buffer, indices: inner_indices, gate: inner_gate } = inner_idx.op() else {
        return None;
    };

    if outer_indices.len() != 1 || inner_indices.len() != 1 {
        return None;
    }

    if outer_gate.is_some() || inner_gate.is_some() {
        return None;
    }

    let inner_offset = &inner_indices[0];
    UOp::index(real_buffer.clone(), vec![inner_offset.clone()]).ok()
}

/// Create patterns for codegen preparation.
pub fn rangeify_codegen_patterns() -> PatternMatcher<()> {
    crate::patterns! {
        noop if matches!(noop.op(), Op::Noop) => remove_noop(noop),
        cont if matches!(cont.op(), Op::Contiguous { .. }) => get_contiguous(cont),
        after if matches!(after.op(), Op::After { .. }) => fix_after_broadcast(after),
        idx if has_multiple_indices(idx) => linearize_index(idx),
        idx if is_cascaded_index(idx) => flatten_cascaded_index(idx),
    }
}

// ============================================================================
// KERNEL SPLITTING PATTERNS
// ============================================================================

/// Replace BUFFER with DEFINE_GLOBAL or DEFINE_LOCAL.
pub fn debuf(buf: &Arc<UOp>, ctx: &mut KernelContext) -> Option<Arc<UOp>> {
    let (ptr_dtype, addrspace) = match buf.op() {
        Op::Buffer { size, .. } => {
            let base_dtype = buf.dtype();
            let ptr_dtype = base_dtype.ptr(Some(*size), AddrSpace::Global);
            (ptr_dtype, AddrSpace::Global)
        }
        _ => return None,
    };

    let replacement = if addrspace == AddrSpace::Global {
        let global_id = ctx.next_global();
        UOp::define_global(global_id, ptr_dtype)
    } else {
        let local_id = ctx.next_local();
        UOp::define_local(local_id, ptr_dtype)
    };

    ctx.map_buffer(buf.clone(), replacement.clone());
    Some(replacement)
}

/// Handle AFTER: extract buffer and track dependency.
pub fn handle_after(after: &Arc<UOp>, ctx: &mut KernelContext) -> Option<Arc<UOp>> {
    let passthrough = match after.op() {
        Op::After { passthrough, .. } => passthrough,
        _ => return None,
    };

    let buf = match passthrough.op() {
        Op::MStack { buffers } if !buffers.is_empty() => buffers[0].clone(),
        Op::MSelect { buffer, .. } => buffer.clone(),
        _ => passthrough.clone(),
    };

    if matches!(buf.dtype(), DType::Ptr { addrspace: AddrSpace::Local, .. }) {
        return Some(buf);
    }

    ctx.map_buffer(buf.clone(), after.clone());
    Some(buf)
}

/// Remove BIND: extract var and track it.
pub fn unbind_kernel(bind: &Arc<UOp>, ctx: &mut KernelContext) -> Option<Arc<UOp>> {
    let (var, _value) = match bind.op() {
        Op::Bind { var, value } => (var, value),
        _ => return None,
    };

    ctx.add_var(var.clone());
    Some(var.clone())
}

/// Renumber RANGE axis_id starting from 0 for kernel deduplication.
pub fn renumber_range(range: &Arc<UOp>, ctx: &mut KernelContext) -> Option<Arc<UOp>> {
    let (end, old_axis_id, axis_type) = match range.op() {
        Op::Range { end, axis_id, axis_type } => (end, *axis_id, *axis_type),
        _ => return None,
    };

    match old_axis_id {
        AxisId::Unrenumbered(_) => {}
        AxisId::Renumbered(_) => return None,
    }

    if axis_type == AxisType::Outer {
        let var_name = format!("range_{}", old_axis_id.value());
        let vmax = match end.vmax() {
            morok_ir::ConstValue::Int(v) => *v - 1,
            morok_ir::ConstValue::UInt(v) => (*v - 1) as i64,
            _ => {
                let new_axis_id = AxisId::Renumbered(ctx.next_range());
                let new_range = UOp::range_axis(end.clone(), new_axis_id, axis_type);
                return Some(new_range);
            }
        };

        let var = UOp::define_var(var_name, vmax);
        let new_axis_id = AxisId::Renumbered(ctx.next_range());
        let renumbered_range = UOp::range_axis(end.clone(), new_axis_id, axis_type);
        return Some(UOp::bind(var, renumbered_range));
    }

    let new_axis_id = AxisId::Renumbered(ctx.next_range());
    let new_range = UOp::range_axis(end.clone(), new_axis_id, axis_type);
    Some(new_range)
}

/// Remove spurious sources from CONST and DEFINE_VAR.
pub fn cleanup_const(op: &Arc<UOp>, _ctx: &mut KernelContext) -> Option<Arc<UOp>> {
    let should_clean = matches!(op.op(), Op::Const(_) | Op::DefineVar { .. });

    if !should_clean {
        return None;
    }

    let sources = op.op().sources();
    if sources.is_empty() {
        return None;
    }

    let cleaned = match op.op() {
        Op::Const(val) => UOp::const_(op.dtype(), val.0),
        Op::DefineVar { name, max_val } => UOp::var(name.clone(), op.dtype(), *max_val),
        _ => unreachable!(),
    };

    Some(cleaned)
}

/// Replace RANGE(end=0) with CONST(0).
pub fn remove_zero_range(range: &Arc<UOp>, _ctx: &mut KernelContext) -> Option<Arc<UOp>> {
    let end = match range.op() {
        Op::Range { end, .. } => end,
        _ => return None,
    };

    let is_zero = match end.op() {
        Op::Const(val) => match val.0 {
            ConstValue::Int(i) => i == 0,
            ConstValue::UInt(u) => u == 0,
            _ => false,
        },
        _ => false,
    };

    if !is_zero {
        return None;
    }

    Some(UOp::index_const(0))
}

/// Create patterns for to_define_global transformation.
pub fn to_define_global_patterns() -> PatternMatcher<KernelContext> {
    crate::patterns! {
        @context KernelContext;
        buf if matches!(buf.op(), Op::Buffer { .. }) => debuf(buf, ctx),
        b if matches!(b.op(), Op::Bind { .. }) => unbind_kernel(b, ctx),
        after if matches!(after.op(), Op::After { .. }) => handle_after(after, ctx),
        c if matches!(c.op(), Op::Const(_) | Op::DefineVar { .. }) => cleanup_const(c, ctx),
        r if matches!(r.op(), Op::Range { .. }) => remove_zero_range(r, ctx),
        r if matches!(r.op(), Op::Range { .. }) => renumber_range(r, ctx),
    }
}

// ============================================================================
// BUFFER LIMIT PATTERNS
// ============================================================================

/// Extract device specification from a UOp graph (first device found).
#[allow(clippy::mutable_key_type)]
pub fn extract_device_from_graph(root: &Arc<UOp>) -> Option<DeviceSpec> {
    let mut visited = HashSet::new();

    fn visit(uop: &Arc<UOp>, visited: &mut HashSet<UOpKey>) -> Option<DeviceSpec> {
        let key = UOpKey(Arc::clone(uop));
        if !visited.insert(key) {
            return None;
        }

        match uop.op() {
            Op::Device(spec) => return Some(spec.clone()),
            Op::Buffer { device, .. } => {
                if let Op::Device(spec) = device.op() {
                    return Some(spec.clone());
                }
            }
            Op::Bufferize { opts, .. } => {
                if let Some(device_spec) = &opts.device {
                    return Some(device_spec.clone());
                }
            }
            _ => {}
        }

        for child in uop.op().sources() {
            if let Some(device) = visit(&child, visited) {
                return Some(device);
            }
        }

        None
    }

    visit(root, &mut visited)
}

/// Create pattern matcher for buffer limit enforcement.
pub fn buffer_limit_patterns(max_buffers: usize) -> PatternMatcher {
    use super::kernel::collect_accessed_buffers;
    use crate::pattern::{BindingStore, BindingStoreExt, VarIntern};

    let mut patterns = vec![];

    let limit = max_buffers;
    patterns.push((
        UPat::var("op"),
        Box::new(move |bindings: &BindingStore, intern: &VarIntern, _ctx: &mut ()| {
            let Some(op) = intern.get_index("op").and_then(|i| bindings.get_by_index(i)) else {
                return RewriteResult::NoMatch;
            };

            let sources = match op.op() {
                Op::Binary(_, left, right) => vec![left.clone(), right.clone()],
                Op::Ternary(_, cond, true_val, false_val) => {
                    vec![cond.clone(), true_val.clone(), false_val.clone()]
                }
                _ => return RewriteResult::NoMatch,
            };

            let mut all_buffers = Vec::new();
            for src in &sources {
                all_buffers.extend(collect_accessed_buffers(src));
            }

            #[allow(clippy::mutable_key_type)]
            let mut seen = HashSet::new();
            all_buffers.retain(|b| seen.insert(UOpKey(Arc::clone(b))));

            if all_buffers.len() > limit.saturating_sub(1) {
                let mut new_sources = Vec::new();
                let mut any_changed = false;

                for src in &sources {
                    let new_src = if is_elementwise(src) { force_bufferize(src) } else { Arc::clone(src) };

                    if !Arc::ptr_eq(&new_src, src) {
                        any_changed = true;
                    }
                    new_sources.push(new_src);
                }

                if any_changed {
                    let rewritten = match op.op() {
                        Op::Binary(bin_op, _, _) => {
                            let lhs = &new_sources[0];
                            let rhs = &new_sources[1];
                            match bin_op {
                                morok_ir::BinaryOp::Add => lhs.try_add(rhs),
                                morok_ir::BinaryOp::Sub => lhs.try_sub(rhs),
                                morok_ir::BinaryOp::Mul => lhs.try_mul(rhs),
                                morok_ir::BinaryOp::Idiv => lhs.try_div(rhs),
                                morok_ir::BinaryOp::Fdiv => lhs.try_div(rhs),
                                morok_ir::BinaryOp::Mod => lhs.try_mod(rhs),
                                morok_ir::BinaryOp::And => lhs.try_and_op(rhs),
                                morok_ir::BinaryOp::Or => lhs.try_or_op(rhs),
                                morok_ir::BinaryOp::Xor => lhs.try_xor_op(rhs),
                                morok_ir::BinaryOp::Lt => lhs.try_cmplt(rhs),
                                morok_ir::BinaryOp::Le => lhs.try_cmple(rhs),
                                morok_ir::BinaryOp::Eq => lhs.try_cmpeq(rhs),
                                morok_ir::BinaryOp::Ne => lhs.try_cmpne(rhs),
                                morok_ir::BinaryOp::Gt => lhs.try_cmpgt(rhs),
                                morok_ir::BinaryOp::Ge => lhs.try_cmpge(rhs),
                                morok_ir::BinaryOp::Max => lhs.try_max(rhs),
                                morok_ir::BinaryOp::Pow => lhs.try_pow(rhs),
                                morok_ir::BinaryOp::Shl | morok_ir::BinaryOp::Shr | morok_ir::BinaryOp::Threefry => {
                                    Ok(UOp::new(Op::Binary(*bin_op, lhs.clone(), rhs.clone()), op.dtype()))
                                }
                            }
                            .expect("Binary op reconstruction should succeed")
                        }
                        Op::Ternary(tern_op, _, _, _) => match tern_op {
                            morok_ir::TernaryOp::Where => {
                                UOp::try_where(new_sources[0].clone(), new_sources[1].clone(), new_sources[2].clone())
                                    .unwrap()
                            }
                            morok_ir::TernaryOp::MulAcc => {
                                UOp::try_mulacc(new_sources[0].clone(), new_sources[1].clone(), new_sources[2].clone())
                                    .expect("MulAcc reconstruction should succeed")
                            }
                        },
                        _ => unreachable!(),
                    };
                    return RewriteResult::Rewritten(rewritten);
                }
            }

            RewriteResult::NoMatch
        }) as RewriteFn<()>,
    ));

    PatternMatcher::new(patterns)
}

/// Force bufferization of a computation to GLOBAL memory.
fn force_bufferize(src: &Arc<UOp>) -> Arc<UOp> {
    let ranges = collect_ranges(src);

    if ranges.is_empty() {
        return Arc::clone(src);
    }

    let opts = BufferizeOpts { device: None, addrspace: AddrSpace::Global };
    let bufferized = UOp::bufferize(Arc::clone(src), ranges.clone(), opts);

    UOp::index(bufferized, ranges).unwrap_or_else(|_| Arc::clone(src))
}

/// Collect all RANGE operations from a computation tree.
#[allow(clippy::mutable_key_type)]
fn collect_ranges(src: &Arc<UOp>) -> Vec<Arc<UOp>> {
    let mut ranges = Vec::new();
    let mut visited = HashSet::new();

    fn visit(uop: &Arc<UOp>, ranges: &mut Vec<Arc<UOp>>, visited: &mut HashSet<UOpKey>) {
        let key = UOpKey(Arc::clone(uop));
        if !visited.insert(key) {
            return;
        }

        if matches!(uop.op(), Op::Range { .. }) {
            ranges.push(Arc::clone(uop));
        }

        for child in uop.op().sources() {
            visit(&child, ranges, visited);
        }
    }

    visit(src, &mut ranges, &mut visited);

    let mut seen = HashSet::new();
    ranges.retain(|r| seen.insert(UOpKey(Arc::clone(r))));

    ranges
}

// ============================================================================
// DEPRECATED PATTERNS (for backwards compatibility)
// ============================================================================

/// Pattern matcher for ReduceAxis → REDUCE conversion (deprecated).
#[allow(dead_code)]
pub fn reduceaxis_to_reduce_patterns() -> PatternMatcher<IndexingContext> {
    crate::patterns! {
        @context IndexingContext;
        x => convert_reduceaxis_with_context(x, ctx),
    }
}
