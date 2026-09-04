//! Optimization operation implementations.
//!
//! Implements: UPCAST (SIMD), LOCAL (shared memory), GROUP (two-stage reduction),
//! UNROLL (loop unrolling), SWAP (axis reordering), NOLOCALS (disable local mem).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use smallvec::SmallVec;
use svod_ir::uop::cached_property::CachedProperty;
use svod_ir::uop::properties::VminVmaxProperty;
use svod_ir::{AxisType, ConstValue, Op, UOp, UOpKey};

use crate::optimizer::{Opt, OptArgExt, OptOps, Scheduler, error::*, tc};
use svod_ir::ops;

// ============================================================================
// DISPATCHER
// ============================================================================

/// Apply an optimization to the scheduler.
pub fn apply_opt(scheduler: &mut Scheduler, opt: &Opt, append_opt: bool) -> Result<(), OptError> {
    let real_axis = scheduler.real_axis(opt.op, opt.axis)?;
    let rng = if real_axis >= 0 { Some(scheduler.rngs()[real_axis as usize].clone()) } else { None };

    match opt.op {
        OptOps::TC => {
            let (tc_select, tc_opt, use_tensor_cores) = opt.arg.tc()?;
            let _axes = tc::apply_with_axis_choice(scheduler, tc_select, tc_opt, use_tensor_cores, opt.axis)?;
        }
        OptOps::UPCAST => {
            let r = rng.ok_or_else(|| MissingAxisParameterSnafu.build())?;
            let amount = resolve_full_axis(&r, opt.arg.int()?, "UPCAST")?;
            apply_upcast(scheduler, r, amount)?;
        }
        OptOps::LOCAL => {
            let r = rng.ok_or_else(|| MissingAxisParameterSnafu.build())?;
            let amount = resolve_full_axis(&r, opt.arg.int()?, "LOCAL")?;
            apply_local(scheduler, r, amount)?;
        }
        OptOps::UNROLL => {
            apply_unroll(scheduler, opt.axis.ok_or_else(|| MissingAxisParameterSnafu.build())?, opt.arg.int()?)?;
        }
        OptOps::NOLOCALS => {
            apply_nolocals(scheduler)?;
        }
        OptOps::SWAP => {
            apply_swap(scheduler, opt.axis.ok_or_else(|| MissingAxisParameterSnafu.build())?, opt.arg.swap()?)?;
        }
        OptOps::GROUP => {
            let r = rng.ok_or_else(|| MissingAxisParameterSnafu.build())?;
            let amount = resolve_full_axis(&r, opt.arg.int()?, "GROUP")?;
            apply_group(scheduler, r, amount, false)?;
        }
        OptOps::GROUPTOP => {
            let r = rng.ok_or_else(|| MissingAxisParameterSnafu.build())?;
            let amount = resolve_full_axis(&r, opt.arg.int()?, "GROUPTOP")?;
            apply_group(scheduler, r, amount, true)?;
        }
        OptOps::THREAD => {
            let r = rng.ok_or_else(|| MissingAxisParameterSnafu.build())?;
            let amount = resolve_full_axis(&r, opt.arg.int()?, "THREAD")?;
            apply_thread(scheduler, r, amount)?;
        }
        OptOps::PADTO => {
            apply_padto(scheduler, rng.ok_or_else(|| MissingAxisParameterSnafu.build())?, opt.arg.int()?)?;
        }
    }

    if append_opt {
        scheduler.applied_opts.push(opt.clone());
    }
    Ok(())
}

/// Resolve `amount=0` to the full size of `rng`'s axis via `vmax+1`.
///
/// `arg=0` means "use the full axis size"; resolved through `VminVmaxProperty`
/// so both constant- and symbolic-end Ranges work. Beam search emits this
/// for `Opt::{upcast,local,group,thread}(_, 0)` variants.
fn resolve_full_axis(rng: &Arc<UOp>, amount: usize, op_name: &'static str) -> Result<usize, OptError> {
    if amount != 0 {
        return Ok(amount);
    }
    if !matches!(rng.op(), Op::Range(..)) {
        return ExpectedRangeOperationSnafu.fail();
    }
    let (_, vmax) = VminVmaxProperty::get(rng);
    let vmax_i64 = match vmax {
        ConstValue::Int(v) => v,
        _ => return ValidationFailedSnafu { op: op_name, reason: "axis vmax has non-Int ConstValue" }.fail(),
    };
    vmax_i64
        .checked_add(1)
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| ValidationFailedSnafu { op: op_name, reason: "axis vmax+1 out of range" }.build())
}

// ============================================================================
// UPCAST - Vectorization (SIMD)
// ============================================================================

/// Split dimension into smaller range + UPCAST for vector operations.
///
/// UPCAST is for output dimension vectorization (GLOBAL/LOCAL/WEAK).
/// For reduce axis unrolling, use UNROLL instead.
fn apply_upcast(scheduler: &mut Scheduler, rng: Arc<UOp>, amount: usize) -> Result<(), OptError> {
    let axis_type = match rng.op() {
        Op::Range(ops::Range { axis_type, .. }) => *axis_type,
        _ => return ExpectedRangeOperationSnafu.fail(),
    };

    // UPCAST applies to GLOBAL/LOCAL/WEAK axes only — REDUCE/GROUP_REDUCE
    // should use UNROLL.
    if !matches!(axis_type, AxisType::Global | AxisType::Local | AxisType::Weak) {
        return ValidationFailedSnafu { op: "UPCAST", reason: "can only upcast Global/Local/Weak axes" }.fail();
    }

    if amount > scheduler.ren.upcast_max {
        return DeviceLimitExceededSnafu { limit_type: "upcast", value: amount, max: scheduler.ren.upcast_max }.fail();
    }

    scheduler.shift_to(rng, amount, AxisType::Upcast, false, None)?;
    Ok(())
}

// ============================================================================
// LOCAL - Shared memory (GPU workgroup)
// ============================================================================

/// Split dimension into smaller range + LOCAL for GPU workgroup threads.
fn apply_local(scheduler: &mut Scheduler, rng: Arc<UOp>, amount: usize) -> Result<(), OptError> {
    if !scheduler.ren.has_local {
        return UnsupportedFeatureSnafu { feature: "local memory" }.fail();
    }
    if scheduler.dont_use_locals {
        return ValidationFailedSnafu { op: "LOCAL", reason: "NOLOCALS was applied" }.fail();
    }

    let axis_type = match rng.op() {
        Op::Range(ops::Range { axis_type, .. }) => *axis_type,
        _ => return ExpectedRangeOperationSnafu.fail(),
    };

    if !matches!(axis_type, AxisType::Global | AxisType::Weak) {
        return ValidationFailedSnafu { op: "LOCAL", reason: "can only localize Global/Weak axes" }.fail();
    }

    scheduler.shift_to(rng, amount, AxisType::Local, false, None)?;
    Ok(())
}

// ============================================================================
// GROUP/GROUPTOP - Two-stage reduction
// ============================================================================

/// Split reduction into smaller range + GROUP_REDUCE using shared memory.
fn apply_group(scheduler: &mut Scheduler, rng: Arc<UOp>, amount: usize, top: bool) -> Result<(), OptError> {
    if scheduler.applied_opts.iter().any(|opt| opt.op == OptOps::TC) {
        return ValidationFailedSnafu { op: "GROUP", reason: "no grouping with tensor cores" }.fail();
    }
    if !scheduler.ren.has_local {
        return UnsupportedFeatureSnafu { feature: "local memory" }.fail();
    }
    if !scheduler.ren.has_shared {
        return UnsupportedFeatureSnafu { feature: "shared memory" }.fail();
    }

    let axis_type = match rng.op() {
        Op::Range(ops::Range { axis_type, .. }) => *axis_type,
        _ => return ExpectedRangeOperationSnafu.fail(),
    };

    if axis_type != AxisType::Reduce {
        return ValidationFailedSnafu { op: "GROUP", reason: "can only group REDUCE axes" }.fail();
    }

    // Calculate shared memory usage
    let upcast_local_sz: usize = scheduler
        .rngs()
        .iter()
        .filter_map(|r| {
            if let Op::Range(ops::Range { axis_type, end, .. }) = r.op()
                && matches!(axis_type, AxisType::Upcast | AxisType::Warp | AxisType::Local | AxisType::GroupReduce)
                && let Op::Const(cv) = end.op()
                && let ConstValue::Int(sz) = cv.0
            {
                return Some(sz as usize);
            }
            None
        })
        .product();

    let reduce_uop = find_reduce_using_range(scheduler, &rng)?;
    let smem_sz = amount * upcast_local_sz * reduce_uop.dtype().bytes();

    if smem_sz > scheduler.ren.shared_max {
        return DeviceLimitExceededSnafu { limit_type: "shared memory", value: smem_sz, max: scheduler.ren.shared_max }
            .fail();
    }

    // Check not inside nested reduction
    let reduce_ptr = Arc::as_ptr(&reduce_uop);
    for node in reduce_uop.backward_slice() {
        if let Op::Reduce(..) = node.op()
            && Arc::as_ptr(&node) != reduce_ptr
        {
            return ValidationFailedSnafu { op: "GROUP", reason: "cannot apply GROUP inside another reduction" }.fail();
        }
    }

    scheduler.shift_to(rng, amount, AxisType::GroupReduce, top, None)?;
    Ok(())
}

fn find_reduce_using_range(scheduler: &Scheduler, rng: &Arc<UOp>) -> Result<Arc<UOp>, OptError> {
    for reduce in scheduler.reduceops() {
        if let Op::Reduce(ops::Reduce { ranges, .. }) = reduce.op()
            && ranges.iter().any(|r| Arc::ptr_eq(r, rng))
        {
            return Ok(reduce.clone());
        }
    }
    ValidationFailedSnafu { op: "GROUP", reason: "could not find REDUCE using this range" }.fail()
}

// ============================================================================
// UNROLL - Loop unrolling
// ============================================================================

/// Split reduction into smaller range + UNROLL for compile-time expansion.
/// When `amount == 0`, the entire axis is unrolled (full unroll). Resolution
/// is shared with UPCAST/LOCAL/GROUP/GROUPTOP/THREAD via [`resolve_full_axis`].
fn apply_unroll(scheduler: &mut Scheduler, axis: usize, amount: usize) -> Result<(), OptError> {
    let unrollable = scheduler.unrollable_dims();
    let real_axis =
        *unrollable.get(axis).ok_or_else(|| AxisOutOfBoundsSnafu { axis, max: unrollable.len() }.build())?;
    let rng = scheduler.rngs()[real_axis].clone();

    let amount = resolve_full_axis(&rng, amount, "UNROLL")?;

    const MAX_UNROLL: usize = 32;
    if amount > MAX_UNROLL {
        return DeviceLimitExceededSnafu { limit_type: "unroll", value: amount, max: MAX_UNROLL }.fail();
    }

    scheduler.shift_to(rng, amount, AxisType::Unroll, false, None)?;
    Ok(())
}

// ============================================================================
// SWAP - Axis reordering
// ============================================================================

/// Swap two GLOBAL ranges for memory-access optimization.
///
/// Tinygrad's `OptOps.SWAP` (`codegen/opt/postrange.py`) tags the two replacement
/// ranges (`tag=1`), runs the fixed-point `substitute`, then strips the tags with
/// `graph_rewrite(remove_all_tags)`. Svod instead applies the swap directly;
/// parent hash-consing keys include ordered child IDs, so tagged and untagged
/// child variants remain distinct during reconstruction.
///
/// Instead we apply the swap as a single-pass [`substitute_walk`]. The map is the
/// simultaneous `{rng -> range(end1, axis_id2), altrng -> range(end2, axis_id1)}`;
/// for equal-extent axes (square matmul, M==N) hash-consing makes the replacements
/// collapse to `{rng -> altrng, altrng -> rng}`. A re-traversing fixed-point
/// `substitute` would re-apply the map to its own output and hit the engine's
/// cycle guard; the single-pass walk applies each mapping exactly once, yielding
/// the correct simultaneous swap without tags.
///
/// [`substitute_walk`]: UOp::substitute_walk
fn apply_swap(scheduler: &mut Scheduler, axis: usize, other_axis: usize) -> Result<(), OptError> {
    let rngs = scheduler.rngs();
    let rng = rngs.get(axis).ok_or_else(|| AxisOutOfBoundsSnafu { axis, max: rngs.len() }.build())?.clone();
    let altrng =
        rngs.get(other_axis).ok_or_else(|| AxisOutOfBoundsSnafu { axis: other_axis, max: rngs.len() }.build())?.clone();

    let (end1, axis_id1, axis_type1) = match rng.op() {
        Op::Range(ops::Range { end, axis_id, axis_type, .. }) => (end.clone(), axis_id.clone(), *axis_type),
        _ => return ExpectedRangeOperationSnafu.fail(),
    };
    let (end2, axis_id2, axis_type2) = match altrng.op() {
        Op::Range(ops::Range { end, axis_id, axis_type, .. }) => (end.clone(), axis_id.clone(), *axis_type),
        _ => return ExpectedRangeOperationSnafu.fail(),
    };

    if axis_type1 != AxisType::Global || axis_type2 != AxisType::Global {
        return ValidationFailedSnafu { op: "SWAP", reason: "swap only for globals" }.fail();
    }

    let new_rng = UOp::range_axis(end1, axis_id2, axis_type1);
    let new_altrng = UOp::range_axis(end2, axis_id1, axis_type2);

    let mut subst_map = HashMap::new();
    subst_map.insert(UOpKey(rng), new_rng);
    subst_map.insert(UOpKey(altrng), new_altrng);

    let swapped = scheduler.ast().substitute_walk(&subst_map);
    scheduler.set_ast(swapped);

    Ok(())
}

// ============================================================================
// NOLOCALS - Disable local memory
// ============================================================================

/// Set flag to prevent future LOCAL/WARP/GROUP_REDUCE optimizations.
fn apply_nolocals(scheduler: &mut Scheduler) -> Result<(), OptError> {
    for rng in scheduler.rngs() {
        if let Op::Range(ops::Range { axis_type, .. }) = rng.op()
            && matches!(axis_type, AxisType::Local | AxisType::Warp | AxisType::GroupReduce)
        {
            return ValidationFailedSnafu {
                op: "NOLOCALS",
                reason: "cannot apply after LOCAL/WARP/GROUP_REDUCE exist",
            }
            .fail();
        }
    }
    scheduler.dont_use_locals = true;
    Ok(())
}

// ============================================================================
// THREAD - CPU parallel dispatch
// ============================================================================

// ============================================================================
// PADTO - Tensor core alignment padding
// ============================================================================

/// Pad dimension to alignment for tensor core compatibility.
///
/// PADTO rounds up a loop dimension to enable tensor core alignment.
///
/// # Constraints
///
/// - Only pad constant-sized axes
/// - Cannot pad UPCAST/UNROLL/THREAD axes (already vectorized/expanded)
/// - Padding must add strictly less than 4x work
///
/// There is deliberately **no** reduce-op guard. Tinygrad used to require
/// `reduce_op == ADD` with no `GroupOp.UnsafePad` op in the reduce's backward
/// slice; tinygrad 5f1e2d390 ("PADTO pads Invalids", #16562) deleted both that
/// check and `GroupOp.UnsafePad`, because the padded lanes now index through
/// `WHERE(valid, INDEX, Invalid)` instead of reading real memory. `postrange.py`
/// at the pinned reference checks only the three constraints above, and its
/// `test_padto_max` / `test_padto_sum` assert MAX reduces and `exp`/`lt` above a
/// reduce pad correctly. This port matches that shape.
///
/// # Algorithm
///
/// 1. Round up range size to alignment
/// 2. Create validity condition: idx < old_size
/// 3. Add WHERE-Invalid validity to all INDEX ops using this range
fn apply_padto(scheduler: &mut Scheduler, rng: Arc<UOp>, alignment: usize) -> Result<(), OptError> {
    let (end, axis_id, axis_type) = match rng.op() {
        Op::Range(ops::Range { end, axis_id, axis_type, .. }) => (end.clone(), axis_id.clone(), *axis_type),
        _ => return ExpectedRangeOperationSnafu.fail(),
    };

    // Constraint 1: only pad constant-sized axes
    let old_sz = match end.op() {
        Op::Const(cv) => match cv.0 {
            ConstValue::Int(v) if v > 0 => v as usize,
            _ => return ValidationFailedSnafu { op: "PADTO", reason: "range end must be positive integer" }.fail(),
        },
        _ => return ValidationFailedSnafu { op: "PADTO", reason: "can only pad constant-sized axes" }.fail(),
    };

    // Constraint 2: cannot pad UPCAST/UNROLL/THREAD axes
    if matches!(axis_type, AxisType::Upcast | AxisType::Unroll | AxisType::Thread) {
        return ValidationFailedSnafu { op: "PADTO", reason: "cannot pad vectorized/unrolled/thread axes" }.fail();
    }

    // Calculate new padded size
    let new_sz = old_sz.div_ceil(alignment) * alignment;

    // Match Tinygrad: padding must add strictly less than 4x work.
    if old_sz <= new_sz / 4 {
        return ValidationFailedSnafu { op: "PADTO", reason: "padding would add more than 4x work" }.fail();
    }

    // Create new padded range
    let new_end = UOp::index_const(new_sz as i64);
    let new_rng = UOp::range_axis(new_end, axis_id, axis_type);

    // Create validity condition: new_rng < old_size
    let old_sz_const = UOp::index_const(old_sz as i64);
    let valid = new_rng
        .try_cmplt(&old_sz_const)
        .map_err(|_| ValidationFailedSnafu { op: "PADTO", reason: "failed to create validity condition" }.build())?;

    // Build substitution map
    let mut subst_map = HashMap::new();
    subst_map.insert(UOpKey(rng.clone()), new_rng.clone());

    let store_targets: HashSet<UOpKey> = scheduler
        .ast()
        .backward_slice()
        .into_iter()
        .filter_map(|node| match node.op() {
            Op::Store(ops::Store { index, .. }) => Some(UOpKey(index.clone())),
            _ => None,
        })
        .collect();

    // Update INDEX operations that use this range, keeping validity as
    // WHERE(cond, index, Invalid).
    // The replacement INDEX must use the new padded range in its indices
    // (not the original range), since substitute replaces the INDEX node
    // directly without recursing into its children.
    let range_subst: HashMap<UOpKey, Arc<UOp>> = [(UOpKey(rng.clone()), new_rng.clone())].into_iter().collect();

    for buf_op in scheduler.bufs() {
        if buf_uses_range(buf_op, &rng)
            && let Op::Index(ops::Index { buffer, indices }) = buf_op.op()
        {
            if indices.len() != 1 {
                return ValidationFailedSnafu {
                    op: "PADTO",
                    reason: "multi-index INDEX is unsupported; Tinygrad PADTO requires one index source",
                }
                .fail();
            }

            // Substitute old range → new range in index expressions
            let new_indices: SmallVec<[Arc<UOp>; 4]> = indices.iter().map(|idx| idx.substitute(&range_subst)).collect();

            let first_idx = new_indices
                .first()
                .cloned()
                .ok_or_else(|| ValidationFailedSnafu { op: "PADTO", reason: "INDEX has no index source" }.build())?;
            let combined = valid
                .try_and_op(&first_idx.get_valid())
                .map_err(|_| ValidationFailedSnafu { op: "PADTO", reason: "failed to combine validity" }.build())?;
            let mut valid_indices = new_indices;
            valid_indices[0] = first_idx.get_idx().valid(combined);
            let new_index =
                UOp::index().buffer(buffer.clone()).indices(valid_indices).call().map_err(|_| {
                    ValidationFailedSnafu { op: "PADTO", reason: "failed to create valid INDEX" }.build()
                })?;
            let replacement = if store_targets.contains(&UOpKey(buf_op.clone())) {
                new_index
            } else {
                new_index.valid(valid.clone())
            };
            subst_map.insert(UOpKey(buf_op.clone()), replacement);
        }
    }

    // Apply substitutions
    let new_ast = scheduler.ast().substitute(&subst_map);
    scheduler.set_ast(new_ast);

    Ok(())
}

/// Check if a buffer INDEX operation uses a specific range.
fn buf_uses_range(buf_op: &Arc<UOp>, rng: &Arc<UOp>) -> bool {
    if let Op::Index(ops::Index { indices, .. }) = buf_op.op() {
        for idx in indices {
            for node in idx.get_idx().toposort() {
                if Arc::ptr_eq(&node, rng) {
                    return true;
                }
            }
        }
    }
    false
}

// ============================================================================
// THREAD - CPU parallel dispatch
// ============================================================================

/// Split dimension into smaller range + THREAD for CPU parallel dispatch.
///
/// THREAD works like GPU's GLOBAL but for CPU: instead of GPU thread blocks,
/// we use OS threads (via rayon). The work partition is baked into index
/// expressions at optimization time - runtime just provides core_id.
///
/// # Safety
///
/// Buffer safety is guaranteed by shift_to() transformation:
/// - Each core_id maps to disjoint output indices
/// - Index formula: `output[core_id * chunk_size + local_idx]`
/// - Same buffer pointers can be safely passed to all threads
fn apply_thread(scheduler: &mut Scheduler, rng: Arc<UOp>, amount: usize) -> Result<(), OptError> {
    // Validate renderer supports threads
    if !scheduler.ren.has_threads {
        return UnsupportedFeatureSnafu { feature: "CPU threads" }.fail();
    }

    // Reject if already threaded. The previous silent `Ok(())` made beam
    // expansions of a THREADed parent generate duplicate schedulers, which
    // then got dedup'd — truncating the beam fan-out and preventing
    // multi-step composition.
    let thread_axes = scheduler.axes_of(&[AxisType::Thread]);
    if !thread_axes.is_empty() {
        return ValidationFailedSnafu { op: "THREAD", reason: "already threaded" }.fail();
    }

    // Validate thread count within limits
    if let Some(global_max) = &scheduler.ren.global_max
        && let Some(&max_threads) = global_max.first()
        && amount > max_threads
    {
        return DeviceLimitExceededSnafu { limit_type: "thread count", value: amount, max: max_threads }.fail();
    }

    // Validate axis type (must be parallelizable)
    let axis_type = match rng.op() {
        Op::Range(ops::Range { axis_type, .. }) => *axis_type,
        _ => return ExpectedRangeOperationSnafu.fail(),
    };

    // THREAD only applies to globalizable ranges (LOOP). GLOBAL kept for the
    // GPU dispatch model.
    if !matches!(axis_type, AxisType::Global | AxisType::Weak) {
        return ValidationFailedSnafu { op: "THREAD", reason: "can only thread Global/Weak axes" }.fail();
    }

    // Only ranges globalizable across all outputs can be threaded safely.
    if !scheduler.globalizable_rngs().iter().any(|candidate| Arc::ptr_eq(candidate, &rng)) {
        return ValidationFailedSnafu { op: "THREAD", reason: "can't apply range to this dim" }.fail();
    }

    // Outer-most position (top=true) so the thread dim becomes core_id.
    let _ = scheduler.shift_to(rng, amount, AxisType::Thread, true, None)?;
    Ok(())
}
