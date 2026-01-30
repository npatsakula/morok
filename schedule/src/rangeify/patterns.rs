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
use morok_dtype::{AddrSpace, DType, ScalarDType};
use morok_ir::{AxisId, AxisType, BinaryOp, BufferizeOpts, ConstValue, Op, ReduceOp, UOp, UOpKey, UnaryOp};
use smallvec::SmallVec;
use tracing::trace;

use crate::TypedPatternMatcher;
use crate::rangeify::transforms::{cast_to_dtype, get_range_size, partition_reduce_ranges};

// Re-export from indexing for backwards compatibility
pub use super::indexing::{is_dead_axis, ranges_equal};

// Forward declarations for types from other modules
use super::indexing::IndexingContext;
use super::kernel::{KernelContext, LocalAddBufferContext};
use super::kernel::{PcontigConfig, SplitReduceOpConfig, split_reduceop};
use super::transforms::transform_sources_with_bufferize;

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
///
/// Note: Unary ops are included here but may need buffering in reduce context.
/// Use `unary_in_reduce_context()` to check before inlining Unary ops.
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

/// Check if a Unary op is within a reduce context (has REDUCE ranges in scope).
///
/// If a Unary op has reduce ranges in scope, it means it's being computed inside
/// a reduction loop and must NOT be inlined (would recompute N times per element).
fn unary_in_reduce_context(compute: &Arc<UOp>) -> bool {
    if !matches!(compute.op(), Op::Unary(..)) {
        return false;
    }
    compute
        .in_scope_ranges()
        .iter()
        .any(|key| if let Op::Range { axis_type, .. } = key.0.op() { *axis_type == AxisType::Reduce } else { false })
}

/// Block inlining of BUFFERIZE when it wraps a Unary op in reduce context.
/// Returns None to keep the BUFFERIZE as-is (no transformation).
fn block_reduce_unary_inline(_buf: &Arc<UOp>) -> Option<Arc<UOp>> {
    None
}

/// Check if an op must always run (shouldn't be buffered).
pub fn is_always_run_op(op: &Op) -> bool {
    matches!(op, Op::Contiguous { .. } | Op::Copy { .. } | Op::Assign { .. })
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
pub fn early_rewrites() -> TypedPatternMatcher {
    crate::patterns! {
        Detach(x) ~> |x| x.clone(),
        ContiguousBackward(x) ~> |x| x.clone(),
        Reshape { src, new_shape } => |src, new_shape| {
            // RESHAPE to scalar - always remove
            if is_scalar_shape(new_shape) {
                return Some(src.clone());
            }

            // RESHAPE on REDUCE - the reduce output is already "indexed" by its ranges
            // The reshape just changes the view of the buffer, which can be done in STORE indexing
            if matches!(src.op(), Op::Reduce { .. }) {
                return Some(src.clone());
            }

            None
        }
    }
}

// ============================================================================
// RANGEIFY TRANSFORMATION PATTERNS
// ============================================================================

/// Create patterns for applying rangeify transformation with IndexingContext.
///
/// Pattern order (like Tinygrad's pm_apply_rangeify):
/// 1. ReduceAxis → REDUCE conversion
/// 2. ALL ops get source bufferization (including movement ops)
/// 3. Movement ops get removed (simple - just return source)
pub fn apply_rangeify_patterns() -> TypedPatternMatcher<IndexingContext> {
    crate::patterns! {
        @context IndexingContext;
        // ReduceAxis conversion MUST come first - before bufferize wraps it
        x @ ReduceAxis { src: _ } => |x, ctx| convert_reduceaxis_with_context(x, ctx),
        // ALL ops (including movement) get source bufferization
        // This matches Tinygrad's approach where bufferization runs on GroupOp.All
        x => |x, ctx| apply_bufferize_transform(x, ctx),
        // Movement ops get removed AFTER bufferization - simple logic
        x if x.op().is_movement() => |x, ctx| remove_movement_op(x, ctx),
    }
}

/// Apply BUFFERIZE transformation to op sources.
fn apply_bufferize_transform(x: &Arc<UOp>, ctx: &mut IndexingContext) -> Option<Arc<UOp>> {
    if let Some(new_sources) = transform_sources_with_bufferize(x, ctx) {
        return Some(x.with_sources(new_sources));
    }
    None
}

/// Remove movement ops after source bufferization.
///
/// Simplified logic (like Tinygrad's remove_movement_op_after_rangeify):
/// - Buffer-like source → add INDEX with movement op's ranges
/// - Source is INDEX → return source (already processed by bufferize phase)
/// - Movement op in range_map → return source
///
/// Realization/bufferization is handled by apply_bufferize_transform which
/// runs BEFORE this pattern (like Tinygrad's create_bufferize_and_index_based_on_ranges).
fn remove_movement_op(x: &Arc<UOp>, ctx: &mut IndexingContext) -> Option<Arc<UOp>> {
    let src = x.op().sources().first()?.clone();

    // Case 1: Buffer-like source → add INDEX with movement op's input ranges
    // Buffer sources aren't in realize_map, so we add INDEX directly
    if matches!(
        src.op(),
        Op::Buffer { .. } | Op::BufferView { .. } | Op::MStack { .. } | Op::MSelect { .. } | Op::After { .. }
    ) {
        let (input_ranges, _) = ctx.get_ranges(x)?;
        return UOp::index().buffer(src).indices(input_ranges.clone()).call().ok();
    }

    // Case 2: Source is INDEX → already processed by bufferize phase
    if matches!(src.op(), Op::Index { .. }) {
        return Some(src);
    }

    // Case 3: Movement op in range_map → return source
    // The source was transformed by apply_bufferize_transform
    if ctx.get_ranges(x).is_some() {
        return Some(src);
    }

    None
}

/// Convert ReduceAxis → REDUCE using IndexingContext.
///
/// Simplified logic (like Tinygrad's convert_reduce_axis_to_reduce_with_ranges):
/// - Filter input ranges by axis index (no AxisType validation needed)
/// - Create REDUCE if we have ranges, return source otherwise
/// - Transfer range_map + realize_map to new identity
fn convert_reduceaxis_with_context(x: &Arc<UOp>, ctx: &mut IndexingContext) -> Option<Arc<UOp>> {
    let Op::ReduceAxis { src, reduce_op, axes } = x.op() else {
        return None;
    };

    let (input_ranges, output_ranges) = ctx.get_ranges(x)?;

    // Filter ranges by axis index (like Tinygrad - no AxisType check needed)
    let reduce_ranges: SmallVec<[Arc<UOp>; 4]> =
        input_ranges.iter().enumerate().filter(|(i, _)| axes.contains(i)).map(|(_, r)| Arc::clone(r)).collect();

    // Determine target: REDUCE if we have ranges, source otherwise
    let target = if reduce_ranges.is_empty() { Arc::clone(src) } else { src.reduce(reduce_ranges, *reduce_op) };

    // Transfer context to new identity (range_map + realize_map only)
    ctx.set_ranges(&target, input_ranges.clone(), output_ranges.clone());
    if let Some(realize_axes) = ctx.get_realize_axes(x).cloned() {
        ctx.mark_realize(&target, realize_axes);
    }

    Some(target)
}

// ============================================================================
// BUFFER FOLDING PATTERNS
// ============================================================================

/// Pattern matcher for buffer folding and constant propagation.
#[tracing::instrument]
pub fn buffer_folding() -> TypedPatternMatcher {
    crate::patterns! {
        Bufferize { compute: c @ Const(_), .. } ~> |c| c.clone(),
        Index { buffer: c @ Const(_), .. } ~> |c| c.clone(),
        Copy { src: c @ Const(_), .. } ~> |c| c.clone(),
        Index { buffer: Bufferize { compute, ranges, .. }, indices, gate: None }
            if ranges_equal(ranges, indices) ~> |compute| compute.clone(),
    }
}

/// Pattern matcher for dead axis removal.
///
/// Based on Tinygrad's cleanup_dead_axes (rangeify.py:123-142).
/// When dead axes are removed from BUFFERIZE, preserves shape via RESHAPE + EXPAND.
pub fn dead_axis_removal() -> TypedPatternMatcher {
    crate::patterns! {
        // Filter dead axes from BUFFERIZE with shape preservation
        bufferize @ Bufferize { compute, ranges, opts } => |bufferize, compute, ranges, opts| {
            cleanup_dead_axes_bufferize(bufferize, compute, ranges, opts)
        },
        // Adjust INDEX indices when BUFFERIZE has dead axes - use nested struct pattern
        Index { buffer: buffer @ Bufferize { ranges: buf_ranges, .. }, indices, gate: _ }
            => |buffer, buf_ranges, indices| {
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
                return UOp::index().buffer(buffer.clone()).indices(new_indices).call().ok();
            }
            None
        },
    }
}

/// Clean up dead axes from BUFFERIZE with shape preservation.
///
/// Based on Tinygrad's cleanup_dead_axes (rangeify.py:123-142).
/// When removing dead axes (ranges with size 1 or ranges not used by compute):
/// 1. Create new BUFFERIZE with only live ranges
/// 2. RESHAPE to insert size-1 dims for dead axes
/// 3. EXPAND to restore original shape
///
/// This preserves shape semantics for downstream operations.
fn cleanup_dead_axes_bufferize(
    bufferize: &Arc<UOp>,
    compute: &Arc<UOp>,
    ranges: &SmallVec<[Arc<UOp>; 4]>,
    opts: &BufferizeOpts,
) -> Option<Arc<UOp>> {
    use morok_ir::SInt;
    use morok_ir::shape::Shape;

    // Don't optimize ALWAYS_RUN_OPS (CONTIGUOUS, COPY, ASSIGN)
    if matches!(compute.op(), Op::Contiguous { .. } | Op::Copy { .. } | Op::Assign { .. }) {
        return None;
    }

    // Get original BUFFERIZE shape (now available after Fix 1)
    let original_shape = bufferize.shape().ok().flatten()?;

    // Get compute's ranges to check if a range is used
    let compute_ranges = compute.ranges();

    let mut new_ranges = Vec::new();
    let mut reshape_dims: Shape = SmallVec::new();
    let mut had_dead = false;

    for (i, range) in ranges.iter().enumerate() {
        // Skip symbolic ranges (non-const end) - Tinygrad TODO
        if let Op::Range { end, .. } = range.op()
            && !matches!(end.op(), Op::Const(_))
        {
            return None;
        }

        // A range is dead if:
        // 1. It's a CONST (already dead)
        // 2. OR it's a RANGE with size 1
        // 3. OR it's a RANGE not in compute's ranges
        let is_const = matches!(range.op(), Op::Const(_));
        let is_size_one = is_dead_axis(range);
        let is_unused = matches!(range.op(), Op::Range { .. }) && !compute_ranges.iter().any(|r| Arc::ptr_eq(r, range));

        if is_const || is_size_one || is_unused {
            reshape_dims.push(SInt::Const(1)); // Dead axis → size 1
            had_dead = true;
        } else {
            // Live axis: keep range and original dimension
            new_ranges.push(Arc::clone(range));
            if let Some(dim) = original_shape.get(i) {
                reshape_dims.push(dim.clone());
            } else {
                return None; // Shape mismatch
            }
        }
    }

    if !had_dead {
        return None;
    }

    // If all ranges are dead, just return compute
    if new_ranges.is_empty() {
        return Some(compute.clone());
    }

    // Create BUFFERIZE with fewer ranges
    let reduced = UOp::bufferize(compute.clone(), new_ranges, opts.clone());

    // RESHAPE to insert size-1 dims for dead axes
    let reshaped = reduced.try_reshape(&reshape_dims).ok()?;

    // EXPAND to restore original shape
    reshaped.try_expand(original_shape).ok()
}

// ============================================================================
// BUFFER REMOVAL PATTERNS
// ============================================================================

/// Pattern matcher for cost-based buffer removal.
///
/// Unary ops in reduce context are NOT inlined to avoid recomputing N times
/// inside the reduction loop (e.g., argmax(-x) needs to compute -x once, not per element).
pub fn buffer_removal() -> TypedPatternMatcher {
    crate::patterns! {
        // Unary in REDUCE context must NOT be inlined - check FIRST
        buf @ Bufferize { compute } if unary_in_reduce_context(compute) => |buf| block_reduce_unary_inline(buf),
        // Other cheap ops (including non-reduce Unary) can inline
        Bufferize { compute, .. } if is_cheap_to_inline(compute.op()) ~> |compute| compute.clone(),
        Bufferize { compute, .. } if is_always_run_op(compute.op()) ~> |compute| compute.clone(),
        Bufferize { compute: Bufferize { compute: inner, .. }, ranges, opts }
            => |inner, ranges, opts| Some(UOp::bufferize(Arc::clone(inner), ranges.to_vec(), opts.clone())),
    }
}

/// Pattern matcher for cost-based buffer removal with partial contiguous support.
pub fn buffer_removal_with_pcontig() -> TypedPatternMatcher<PcontigConfig> {
    crate::patterns! {
        @context PcontigConfig;
        // Partial contiguous removal - use nested struct + gate: None
        Index {
            buffer: buffer @ Bufferize { compute: src, ranges: buf_ranges, .. },
            indices: idx_ranges,
            gate: None
        } => |buffer, src, buf_ranges, idx_ranges, ctx| {
            apply_pcontig_removal_inner(buffer, src, buf_ranges, idx_ranges, ctx)
        },

        // Remove cheap bufferize - inline with compute extraction
        Bufferize { compute, .. }
            if ctx.level > 0 && is_cheap_to_inline(compute.op()) && !unary_in_reduce_context(compute)
            => |compute| {
                tracing::debug!(
                    compute_id = compute.id,
                    compute_op = ?std::mem::discriminant(compute.op()),
                    "removing cheap bufferize"
                );
                Some(compute.clone())
            },

        // Remove always-run bufferize - inline with compute extraction
        Bufferize { compute, .. } if ctx.level > 0 && is_always_run_op(compute.op()) ~> |compute| compute.clone(),

        // Flatten nested Bufferize
        Bufferize { compute: Bufferize { compute: inner, .. }, ranges, opts }
            => |inner, ranges, opts| Some(UOp::bufferize(Arc::clone(inner), ranges.to_vec(), opts.clone())),
    }
}

/// Apply partial contiguous buffer removal (inline, like Tinygrad's remove_bufferize).
///
/// This function decides whether to remove a BUFFERIZE+INDEX pair by analyzing:
/// 1. Buffer count threshold (too many buffers → keep materialized)
/// 2. Output/input ratio (wasteful expansion → keep materialized)
/// 3. Buffer in reduce scope (needs partial contiguous handling)
#[allow(clippy::mutable_key_type)]
fn apply_pcontig_removal_inner(
    buffer: &Arc<UOp>,
    src: &Arc<UOp>,
    buf_ranges: &SmallVec<[Arc<UOp>; 4]>,
    idx_ranges: &SmallVec<[Arc<UOp>; 4]>,
    config: &mut PcontigConfig,
) -> Option<Arc<UOp>> {
    use morok_ir::{AddrSpace, AxisType, BufferizeOpts, ConstValue};
    use std::collections::{HashMap, HashSet};

    if config.level == 0 || is_always_run_op(src.op()) {
        return None;
    }

    // Single-pass collection (like Tinygrad's gate approach)
    let mut accessed_buffers = Vec::new();
    let mut indexes = Vec::new();
    let mut reduces = Vec::new();
    let mut visited = HashSet::new();

    fn collect(
        uop: &Arc<UOp>,
        buffers: &mut Vec<Arc<UOp>>,
        indexes: &mut Vec<Arc<UOp>>,
        reduces: &mut Vec<Arc<UOp>>,
        visited: &mut HashSet<UOpKey>,
    ) -> bool {
        let key = UOpKey(Arc::clone(uop));
        if !visited.insert(key) {
            return true;
        }

        match uop.op() {
            Op::Bufferize { opts, .. } if opts.addrspace == AddrSpace::Global => {
                buffers.push(Arc::clone(uop));
                return false; // Stop traversing into GLOBAL bufferize
            }
            Op::Buffer { .. } | Op::MStack { .. } | Op::MSelect { .. } => {
                buffers.push(Arc::clone(uop));
            }
            Op::Index { .. } => indexes.push(Arc::clone(uop)),
            Op::Reduce { .. } => reduces.push(Arc::clone(uop)),
            _ => {}
        }

        for child in uop.op().sources() {
            collect(&child, buffers, indexes, reduces, visited);
        }
        true
    }

    collect(src, &mut accessed_buffers, &mut indexes, &mut reduces, &mut visited);

    // Deduplicate buffers
    let mut seen = HashSet::new();
    accessed_buffers.retain(|b| seen.insert(UOpKey(Arc::clone(b))));

    // Buffer count threshold
    if accessed_buffers.len() > config.max_buffers_threshold {
        return None;
    }

    // Output/input ratio check (applied to ALL cases, not just buffer_in_reduce)
    let output_size = match buffer.op() {
        Op::Bufferize { ranges, .. } => {
            let mut product = 1usize;
            for range in ranges {
                if let Op::Range { end, .. } = range.op()
                    && let Op::Const(cv) = end.op()
                    && let ConstValue::Int(n) = cv.0
                    && n > 0
                {
                    product = product.checked_mul(n as usize)?;
                } else {
                    return None;
                }
            }
            let element_size = buffer.dtype().base().bytes();
            product.checked_mul(element_size)?
        }
        Op::Buffer { size, .. } => *size,
        _ => return None,
    };

    let input_size: usize = accessed_buffers
        .iter()
        .filter_map(|buf| match buf.op() {
            Op::Bufferize { ranges, .. } => {
                let mut product = 1usize;
                for range in ranges {
                    if let Op::Range { end, .. } = range.op()
                        && let Op::Const(cv) = end.op()
                        && let ConstValue::Int(n) = cv.0
                        && n > 0
                    {
                        product = product.checked_mul(n as usize)?;
                    }
                }
                let elem_size = buf.dtype().base().bytes();
                product.checked_mul(elem_size)
            }
            Op::Buffer { size, .. } => Some(*size),
            Op::MStack { .. } | Op::MSelect { .. } => Some(1),
            _ => None,
        })
        .sum();

    let ratio = (output_size + 1) as f64 / (input_size + 1) as f64;
    if ratio < config.out_in_ratio_threshold {
        return None;
    }

    // Check if buffer accessed in reduce scope
    let buffer_in_reduce = if reduces.is_empty() {
        false
    } else {
        let reduce_sources: Vec<Arc<UOp>> = reduces
            .iter()
            .filter_map(|r| if let Op::Reduce { src, .. } = r.op() { Some(Arc::clone(src)) } else { None })
            .collect();

        if reduce_sources.is_empty() {
            false
        } else {
            let sink = UOp::sink(reduce_sources);
            let mut found = false;
            for node in sink.toposort() {
                if matches!(node.op(), Op::Buffer { .. } | Op::Bufferize { .. }) {
                    found = true;
                    break;
                }
            }
            found
        }
    };

    // Simple substitution path (no buffer in reduce)
    if !buffer_in_reduce {
        let subs_map: HashMap<UOpKey, Arc<UOp>> =
            buf_ranges.iter().zip(idx_ranges.iter()).map(|(k, v)| (UOpKey(Arc::clone(k)), Arc::clone(v))).collect();
        return Some(src.substitute(&subs_map));
    }

    // Partial contiguous path: filter local indexes and extract exclude ranges
    let local_indexes: Vec<_> = indexes
        .iter()
        .filter(|idx| {
            matches!(idx.op(), Op::Index { buffer, .. }
                if matches!(buffer.op(), Op::Bufferize { opts, .. }
                    if opts.addrspace == AddrSpace::Local))
        })
        .collect();

    let mut exclude_ranges = HashSet::new();
    for idx in &local_indexes {
        if let Op::Index { indices, .. } = idx.op() {
            for range in indices {
                for r in range.in_scope_ranges() {
                    exclude_ranges.insert(r.clone());
                }
            }
        }
    }

    // Partition ranges into materialize vs substitute
    let mut materialize = Vec::new();
    let mut substitute = Vec::new();

    for (buf_rng, idx_rng) in buf_ranges.iter().zip(idx_ranges.iter()) {
        if matches!(buf_rng.op(), Op::Const(_)) {
            continue;
        }

        let buf_key = UOpKey(Arc::clone(buf_rng));
        let should_materialize = exclude_ranges.contains(&buf_key)
            || idx_rng.in_scope_ranges().iter().any(|r| {
                if let Op::Range { axis_type, .. } = r.0.op() { matches!(axis_type, AxisType::Reduce) } else { false }
            });

        if should_materialize {
            materialize.push((Arc::clone(buf_rng), Arc::clone(idx_rng)));
        } else {
            substitute.push((Arc::clone(buf_rng), Arc::clone(idx_rng)));
        }
    }

    if substitute.is_empty() {
        return None;
    }

    // Apply substitution
    let subs_map: HashMap<UOpKey, Arc<UOp>> = substitute.into_iter().map(|(k, v)| (UOpKey(k), v)).collect();
    let substituted = src.substitute(&subs_map);

    if materialize.is_empty() {
        return Some(substituted);
    }

    // Create partial bufferize + index
    let (mat_buf_rngs, mat_idx_rngs): (Vec<_>, Vec<_>) = materialize.into_iter().unzip();
    let opts = BufferizeOpts::local();
    let bufferized = UOp::bufferize(substituted, mat_buf_rngs, opts);

    UOp::index().buffer(bufferized).indices(mat_idx_rngs).call().ok()
}

// ============================================================================
// REDUCTION SIMPLIFY PATTERNS
// ============================================================================

/// Pattern matcher for splitting large ReduceAxis operations.
/// Must run BEFORE ReduceAxis → REDUCE conversion (Step 2.5).
pub fn split_reduceop_patterns() -> TypedPatternMatcher<SplitReduceOpConfig> {
    crate::patterns! {
        @context SplitReduceOpConfig;
        reduce @ ReduceAxis { src: _ } => |reduce, ctx| split_reduceop(reduce, ctx),
    }
}

/// Pattern matcher for reduction simplifications (reduce_unparented, reduce_collapse).
/// Must run AFTER ReduceAxis → REDUCE conversion (Step 7) because these match Op::Reduce.
pub fn reduction_simplify_patterns() -> TypedPatternMatcher {
    crate::patterns! {
        reduce @ Reduce { src, ranges, reduce_op: Add | Mul | Max | Min } => |reduce, src, ranges, reduce_op| {
            #[allow(clippy::mutable_key_type)]
            let src_ranges = src.in_scope_ranges();
            let (parented, unparented) = partition_reduce_ranges(ranges, src_ranges);

            if unparented.is_empty() {
                return None;
            }

            let mut result = if !parented.is_empty() || reduce.dtype() != src.dtype() {
                src.reduce(parented, *reduce_op)
            } else {
                Arc::clone(src)
            };

            match reduce_op {
                ReduceOp::Add => {
                    for range in &unparented {
                        let size = get_range_size(range)?;
                        let size_casted = cast_to_dtype(&size, &result.dtype())?;
                        result = result.try_mul(&size_casted).ok()?;
                    }
                }
                ReduceOp::Mul => {
                    for range in &unparented {
                        let size = get_range_size(range)?;
                        let size_casted = cast_to_dtype(&size, &result.dtype())?;
                        result = result.try_pow(&size_casted).ok()?;
                    }
                }
                ReduceOp::Max | ReduceOp::Min => {}
            }

            Some(result)
        },
        Reduce { src, ranges } => || super::transforms::reduce_collapse(src, ranges),
    }
}

// ============================================================================
// MOVEMENT OP PATTERNS (pm_mops equivalent)
// ============================================================================

/// Create pattern matcher for pushing movement ops through INDEX operations.
pub fn movement_op_patterns() -> TypedPatternMatcher {
    crate::patterns! {
        Index { buffer: mop, indices, gate } if mop.op().is_movement() => |mop, indices, gate| {
            transform_movement_through_index(mop, indices, gate)
        },
        // Flatten nested INDEX when both have gate: None and matching single indices
        Index {
            buffer: inner @ Index { indices: inner_indices, gate: None },
            indices: outer_indices,
            gate: None
        } if inner_indices.len() == 1 && outer_indices.len() == 1
             && inner_indices[0].id == outer_indices[0].id
            ~> |inner| inner.clone(),
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
        Some(g) => UOp::index().buffer(src.clone()).indices(transformed).gate(g.clone()).call(),
        None => UOp::index().buffer(src.clone()).indices(transformed).call(),
    }
    .ok()
}

// ============================================================================
// CODEGEN PREPARATION PATTERNS
// ============================================================================

/// Create zero UOp for a given dtype (scalar or vector).
fn dtype_zero(dtype: DType) -> Arc<UOp> {
    let base = dtype.base();
    let zero = ConstValue::zero(base);
    if dtype.is_vector() {
        UOp::vectorize((0..dtype.count()).map(|_| UOp::const_(DType::Scalar(base), zero)).collect())
    } else {
        UOp::const_(dtype, zero)
    }
}

/// Create patterns for codegen preparation.
///
/// Note: Multi-index INDEX ops are preserved through the pipeline and
/// linearized at codegen time (not here). This aligns with Tinygrad's
/// architecture and prevents Binary(Range*stride) expressions in the IR.
pub fn rangeify_codegen_patterns() -> TypedPatternMatcher<()> {
    crate::patterns! {
        // NOOP → zero constant (scalar or vector)
        noop @ Noop() if noop.dtype().base() != morok_dtype::ScalarDType::Void => |noop| {
            Some(dtype_zero(noop.dtype()))
        },
        // CONTIGUOUS is a no-op at this stage
        Contiguous { src } ~> |src| src.clone(),
        // AFTER(EXPAND): strip EXPAND only if src is a valid passthrough
        After { passthrough: Expand { src, .. }, deps } => |src, deps| {
            // Don't unwrap if src is control flow (Range, End)
            // These are not valid AFTER passthroughs in Tinygrad
            if matches!(src.op(), Op::Range { .. } | Op::End { .. }) {
                return None;  // Keep original - can't strip EXPAND from control flow
            }

            #[allow(clippy::mutable_key_type)]
            let has_range = src.get_consumer_map()
                .get(&UOpKey(src.clone()))
                .is_some_and(|c| c.iter().any(|c| matches!(c.op(), Op::Range { .. })));
            assert!(!has_range, "can't have a local AFTER");
            Some(src.after(deps.clone()))
        },
    }
}

// ============================================================================
// KERNEL SPLITTING PATTERNS
// ============================================================================

/// Extract base dtype from a Ptr type, or return the type as-is.
fn extract_base_dtype(dtype: DType) -> DType {
    match dtype {
        DType::Ptr { base, .. } => (*base).clone(),
        other => other,
    }
}

/// Extract buffer from AFTER passthrough (handles MStack/MSelect).
fn extract_buffer_from_after(passthrough: &Arc<UOp>) -> Arc<UOp> {
    match passthrough.op() {
        Op::MStack { buffers } if !buffers.is_empty() => buffers[0].clone(),
        Op::MSelect { buffer, .. } => buffer.clone(),
        _ => passthrough.clone(),
    }
}

/// Find output DefineGlobal from KERNEL AST.
fn find_kernel_output(ast: &Arc<UOp>) -> Option<Arc<UOp>> {
    for node in ast.toposort() {
        // Use store_buffer() helper to get buffer from STORE via its INDEX child
        if let Some(buffer) = node.store_buffer() {
            let output_buf = match buffer.op() {
                Op::Index { buffer: inner_buf, .. } => inner_buf.clone(),
                _ => buffer.clone(),
            };
            if matches!(output_buf.op(), Op::DefineGlobal(_)) {
                return Some(output_buf);
            }
        }
    }
    None
}

/// Create patterns for to_define_global transformation.
pub fn to_define_global_patterns() -> TypedPatternMatcher<KernelContext> {
    crate::patterns! {
        @context KernelContext;
        // Buffer → DefineGlobal
        buf @ Buffer { size, unique: _ } => |buf, size, ctx| {
            let ptr_dtype = extract_base_dtype(buf.dtype()).ptr(Some(*size), AddrSpace::Global);
            let replacement = UOp::define_global(ctx.next_global(), ptr_dtype);
            ctx.define_to_buffer_id.insert(replacement.id, buf.id);
            ctx.map_buffer(buf.clone(), replacement.clone());
            Some(replacement)
        },
        // Remove BIND: extract var and track it
        Bind { var, value: _ } => |var, ctx| {
            ctx.add_var(var.clone());
            Some(var.clone())
        },
        // Handle AFTER: extract buffer and track dependency
        after @ After { passthrough } => |after, passthrough, ctx| {
            let buf = extract_buffer_from_after(passthrough);
            if matches!(buf.dtype(), DType::Ptr { addrspace: AddrSpace::Local, .. }) {
                return Some(buf);
            }
            ctx.map_buffer(buf.clone(), after.clone());
            Some(buf)
        },
        // Remove spurious sources from CONST and DEFINE_VAR
        c @ Const(_) | c @ DefineVar { name: _ } => |c, _ctx| {
            let sources = c.op().sources();
            if sources.is_empty() { return None; }
            Some(match c.op() {
                Op::Const(val) => UOp::const_(c.dtype(), val.0),
                Op::DefineVar { name, min_val, max_val } => UOp::var(name.clone(), c.dtype(), *min_val, *max_val),
                _ => return None,
            })
        },
        // Replace RANGE(end=0) with CONST(0)
        Range { end } if matches!(end.op(), Op::Const(v) if v.0.is_zero()) => |_r, _ctx| {
            Some(UOp::index_const(0))
        },
        // Renumber RANGE axis_id (Unrenumbered → Renumbered)
        Range { end, axis_id, axis_type } if matches!(axis_id, AxisId::Unrenumbered(_)) => |_r, end, axis_type, ctx| {
            Some(UOp::range_axis(end.clone(), AxisId::Renumbered(ctx.next_range()), *axis_type))
        },
        // Replace KERNEL references with their output buffer
        Kernel { ast } => |_k, ast, _ctx| find_kernel_output(ast),
    }
}

/// Create patterns for to_define_global transformation using LocalAddBufferContext.
///
/// Based on Tinygrad's to_define_global (rangeify.py:419-434).
/// This version uses the per-kernel LocalAddBufferContext instead of global KernelContext.
pub fn local_to_define_global_patterns() -> TypedPatternMatcher<LocalAddBufferContext> {
    crate::patterns! {
        @context LocalAddBufferContext;
        // Buffer → DefineGlobal (like Tinygrad's debuf, rangeify.py:385-389)
        // Creates DEFINE_GLOBAL and maps buf → buf (tracking original BUFFER)
        buf @ Buffer { size, unique: _ } => |buf, size, ctx| {
            let ptr_dtype = extract_base_dtype(buf.dtype()).ptr(Some(*size), AddrSpace::Global);
            let replacement = UOp::define_global(ctx.next_dg(), ptr_dtype);
            // Map buffer to itself (like Tinygrad: if buf not in ctx.map: ctx.map[buf] = buf)
            if !ctx.has_buffer(buf) {
                ctx.map_buffer(buf.clone(), buf.clone());
            }
            Some(replacement)
        },
        // Remove BIND: extract var and track it (like Tinygrad's unbind_kernel)
        Bind { var, value: _ } => |var, ctx| {
            ctx.add_var(var.clone());
            Some(var.clone())
        },
        // Handle AFTER: extract buffer and track dependency (like Tinygrad's handle_after, rangeify.py:395-402)
        // Maps buf → after (so kernel sources will include AFTER wrappers)
        after @ After { passthrough } => |after, passthrough, ctx| {
            // Skip local address space AFTERs (like Tinygrad)
            if matches!(passthrough.dtype(), DType::Ptr { addrspace: AddrSpace::Local, .. }) {
                return None;
            }
            // Use buf_uop() to get underlying buffer (like Tinygrad's after.as_buf())
            let buf = after.buf_uop();
            // HACK: Handle MSTACK/MSELECT like Tinygrad
            let buf = match buf.op() {
                Op::MStack { buffers } if !buffers.is_empty() => buffers[0].clone(),
                Op::MSelect { buffer, .. } => buffer.clone(),
                _ => buf,
            };
            // Skip if buffer already mapped
            if ctx.has_buffer(&buf) {
                return None;
            }
            // Map buf → after (kernel sources will be AFTERs)
            ctx.map_buffer(buf.clone(), after.clone());
            // Return the buffer to replace AFTER in the AST
            Some(buf)
        },
        // Remove spurious sources from CONST and DEFINE_VAR
        c @ Const(_) | c @ DefineVar { name: _ } => |c, _ctx| {
            let sources = c.op().sources();
            if sources.is_empty() { return None; }
            Some(match c.op() {
                Op::Const(val) => UOp::const_(c.dtype(), val.0),
                Op::DefineVar { name, min_val, max_val } => UOp::var(name.clone(), c.dtype(), *min_val, *max_val),
                _ => return None,
            })
        },
        // Replace RANGE(end=0) with CONST(0)
        Range { end } if matches!(end.op(), Op::Const(v) if v.0.is_zero()) => |_r, _ctx| {
            Some(UOp::index_const(0))
        },
        // Renumber RANGE axis_id (like Tinygrad's renumber_range)
        Range { end, axis_id, axis_type } if matches!(axis_id, AxisId::Unrenumbered(_)) => |_r, end, axis_type, ctx| {
            Some(UOp::range_axis(end.clone(), AxisId::Renumbered(ctx.next_range()), *axis_type))
        },
    }
}

/// Create pattern matcher for split_kernels.
///
/// Based on Tinygrad's split_kernels (rangeify.py:509-511).
/// This matches STORE and END operations and calls split_store on them.
pub fn split_kernels_pattern() -> TypedPatternMatcher<Vec<Arc<UOp>>> {
    use super::kernel::split_store;
    crate::patterns! {
        @context Vec<Arc<UOp>>;
        x @ Store { index: _, value: _, .. } => |x, ctx| split_store(ctx, x),
        x @ End { computation: _ } => |x, ctx| split_store(ctx, x),
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
///
/// Uses `binary [*]` and `ternary [*]` to match all binary/ternary ops,
/// checks if they access too many buffers, and forces bufferization of
/// elementwise sources to GLOBAL memory if so.
#[allow(unused_variables)] // `op` is used by macro expansion
pub fn buffer_limit_patterns(max_buffers: usize) -> TypedPatternMatcher<()> {
    crate::patterns! {
        for op in binary [*] {
            tree@op(a, b) => |tree, a, b| {
                check_buffer_limit(tree, &[a.clone(), b.clone()], max_buffers)
            },
        }

        for op in ternary [*] {
            tree@op(a, b, c) => |tree, a, b, c| {
                check_buffer_limit(tree, &[a.clone(), b.clone(), c.clone()], max_buffers)
            },
        }
    }
}

/// Check buffer limit and force bufferization if exceeded.
fn check_buffer_limit(tree: &Arc<UOp>, sources: &[Arc<UOp>], max_buffers: usize) -> Option<Arc<UOp>> {
    let all_buffers = collect_accessed_buffers(sources);

    if all_buffers.len() > max_buffers.saturating_sub(1) {
        let mut any_changed = false;
        let new_sources: Vec<_> = sources
            .iter()
            .map(|src| {
                if is_elementwise(src) {
                    let new = force_bufferize(src);
                    if !Arc::ptr_eq(&new, src) {
                        any_changed = true;
                    }
                    new
                } else {
                    src.clone()
                }
            })
            .collect();

        if any_changed {
            return Some(tree.with_sources(new_sources));
        }
    }
    None
}

/// Collect all accessed buffers from sources.
fn collect_accessed_buffers(sources: &[Arc<UOp>]) -> Vec<Arc<UOp>> {
    let mut all_buffers = Vec::new();
    #[allow(clippy::mutable_key_type)]
    let mut visited = HashSet::new();

    #[allow(clippy::mutable_key_type)]
    fn collect_recursive(uop: &Arc<UOp>, buffers: &mut Vec<Arc<UOp>>, visited: &mut HashSet<UOpKey>) {
        let key = UOpKey(Arc::clone(uop));
        if !visited.insert(key) {
            return;
        }
        match uop.op() {
            Op::Bufferize { opts, .. } if opts.addrspace == AddrSpace::Global => {
                buffers.push(Arc::clone(uop));
                return; // Stop at GLOBAL bufferize
            }
            Op::Buffer { .. } | Op::MStack { .. } | Op::MSelect { .. } => {
                buffers.push(Arc::clone(uop));
            }
            _ => {}
        }
        for child in uop.op().sources() {
            collect_recursive(&child, buffers, visited);
        }
    }

    for src in sources {
        collect_recursive(src, &mut all_buffers, &mut visited);
    }

    // Deduplicate
    #[allow(clippy::mutable_key_type)]
    let mut seen = HashSet::new();
    all_buffers.retain(|b| seen.insert(UOpKey(Arc::clone(b))));
    all_buffers
}

/// Force bufferization of a computation to GLOBAL memory.
fn force_bufferize(src: &Arc<UOp>) -> Arc<UOp> {
    let ranges = src.ranges().clone();
    if ranges.is_empty() {
        return Arc::clone(src);
    }
    let opts = BufferizeOpts { device: None, addrspace: AddrSpace::Global };
    let bufferized = UOp::bufferize(Arc::clone(src), ranges.clone(), opts);
    UOp::index().buffer(bufferized).indices(ranges).call().unwrap_or_else(|_| Arc::clone(src))
}

// ============================================================================
// PM_ADD_LOADS - Wrap INDEX with LOAD for arithmetic ops
// ============================================================================

/// Pattern matcher to wrap INDEX sources with LOAD for arithmetic operations.
///
/// Based on Tinygrad's pm_add_loads (devectorizer.py:320-326).
/// Simplified approach: transform INDEX → LOAD(INDEX) at source, then cleanup STORE.
///
/// Bottom-up rewriting propagates the transformation to all consumers automatically:
/// 1. INDEX → LOAD(buffer, INDEX) - wrap every INDEX with LOAD
/// 2. STORE cleanup - remove LOAD from index position (STORE needs raw INDEX)
pub fn pm_add_loads() -> TypedPatternMatcher<()> {
    crate::patterns! {
        // Pattern 1: INDEX with non-Ptr dtype → LOAD(buffer, INDEX with Ptr dtype)
        // Guard prevents re-matching: after transformation, INDEX has Ptr dtype.
        // Based on Tinygrad's pm_add_loads (devectorizer.py:320-323).
        idx @ Index { buffer, indices } if !matches!(idx.dtype(), DType::Ptr { .. }) => |idx, buffer, indices| {
            // Handle vectorized indices - if any index has vector dtype, LOAD produces vector
            let vec_count = indices.iter().find_map(|i| match i.dtype() {
                DType::Vector { count, .. } => Some(count),
                _ => None,
            });

            let element_dtype = match buffer.dtype() {
                DType::Ptr { base, .. } => (*base).clone(),
                other => other.clone(),
            };

            // Compute result dtype based on vectorized indices
            // If element_dtype is already vectorized, use it as-is to avoid double-vectorization
            let result_dtype = match vec_count {
                Some(count) if element_dtype.vcount() == 1 => element_dtype.vec(count),
                _ => element_dtype,
            };

            // Create INDEX with Ptr dtype (buffer's dtype) - won't match pattern again.
            // This matches Tinygrad's idx.replace(dtype=idx.src[0].dtype) approach.
            let gate = match idx.op() {
                Op::Index { gate, .. } => gate.clone(),
                _ => None,
            };
            let ptr_index = UOp::new(
                Op::Index { buffer: buffer.clone(), indices: indices.clone(), gate },
                buffer.dtype().clone(),  // Use buffer dtype (Ptr), not original INDEX dtype
            );

            Some(UOp::load().buffer(buffer.clone()).index(ptr_index).dtype(result_dtype).call())
        },

        // Pattern 2: Cleanup STORE - remove LOAD from index position
        Store { index: Load { index: real_index, .. }, value, ranges } =>
            |real_index, value, ranges| {
                Some(real_index.store_with_ranges(value.clone(), ranges.clone()))
            },
    }
}

// ============================================================================
// BOOL DEVECTORIZATION
// ============================================================================

/// Check if dtype is vectorized bool (needs devectorization).
///
/// LLVM's `<N x i1>` vectors are broken - no formal ABI, inconsistent backend
/// support, and Clang explicitly prohibits bool as vector element type.
fn is_vectorized_bool(dtype: &DType) -> bool {
    dtype.base() == ScalarDType::Bool && dtype.vcount() > 1
}

/// Unified devectorize for any binary op producing vectorized output.
///
/// Handles scalar-vector operand mix (from comparisons like `vec < scalar`).
/// Converts: OP(<N x T>, <N x T>) → VECTORIZE(OP(gep(a,0), gep(b,0)), ...)
fn devectorize_binary(op: &BinaryOp, result: &Arc<UOp>, a: &Arc<UOp>, b: &Arc<UOp>) -> Option<Arc<UOp>> {
    let out_vcount = result.dtype().vcount();
    if out_vcount <= 1 {
        return None;
    }

    let a_vcount = a.dtype().vcount();
    let b_vcount = b.dtype().vcount();
    let scalar_dtype = result.dtype().scalar_dtype();

    let scalar_ops: SmallVec<[Arc<UOp>; 4]> = (0..out_vcount)
        .map(|i| {
            // Handle scalar-vector mix: only GEP if operand is vectorized
            let a_elem = if a_vcount > 1 { a.gep(vec![i]) } else { a.clone() };
            let b_elem = if b_vcount > 1 { b.gep(vec![i]) } else { b.clone() };
            UOp::new(Op::Binary(*op, a_elem, b_elem), scalar_dtype.clone())
        })
        .collect();

    Some(UOp::vectorize(scalar_ops))
}

/// Unified devectorize for any unary op producing vectorized output.
fn devectorize_unary(op: &UnaryOp, result: &Arc<UOp>, src: &Arc<UOp>) -> Option<Arc<UOp>> {
    let out_vcount = result.dtype().vcount();
    if out_vcount <= 1 {
        return None;
    }

    let scalar_dtype = result.dtype().scalar_dtype();

    let scalar_ops: SmallVec<[Arc<UOp>; 4]> = (0..out_vcount)
        .map(|i| {
            let elem = src.gep(vec![i]);
            UOp::new(Op::Unary(*op, elem), scalar_dtype.clone())
        })
        .collect();

    Some(UOp::vectorize(scalar_ops))
}

/// Generic devectorize for any op (INDEX, CAST, BITCAST, etc).
/// Mirrors Tinygrad's no_vectorized_alu (devectorizer.py:219-223).
fn devectorize_generic(uop: &Arc<UOp>) -> Option<Arc<UOp>> {
    let vcount = uop.dtype().vcount();
    if vcount <= 1 {
        return None;
    }

    let scalar_dtype = uop.dtype().scalar_dtype();
    let sources = uop.op().sources();

    let elements: SmallVec<[Arc<UOp>; 4]> = (0..vcount)
        .map(|i| {
            let new_sources: Vec<Arc<UOp>> =
                sources.iter().map(|s| if s.dtype().vcount() > 1 { s.gep(vec![i]) } else { s.clone() }).collect();
            uop.replace().dtype(scalar_dtype.clone()).src(new_sources).call()
        })
        .collect();

    Some(UOp::vectorize(elements))
}

/// Pattern matcher for bool devectorization.
///
/// LLVM's `<N x i1>` vectors are broken (no formal ABI, segfaults in codegen).
/// This pass converts vectorized bool operations into scalar ops wrapped in
/// VECTORIZE, following Tinygrad's approach (cstyle.py:62-68, devectorizer.py:no_vectorized_alu).
///
/// Transforms:
/// - Any binary op producing vectorized bool → scalar ops + VECTORIZE
/// - Any unary op producing vectorized bool → scalar ops + VECTORIZE
/// - WHERE with vectorized condition → scalar WHERE + VECTORIZE
/// - INDEX with vectorized bool dtype → scalar INDEX + VECTORIZE
/// - CAST to/from vectorized bool → scalar CAST + VECTORIZE
pub fn pm_bool_devectorize() -> TypedPatternMatcher<()> {
    crate::patterns! {
        // Any binary op that produces vectorized bool output
        // Covers: And, Or, Xor, Max on bool vectors AND Lt, Le, Eq, Ne, Gt, Ge comparisons
        for op in binary [*] {
            result @ op(a, b) if is_vectorized_bool(&result.dtype()) => |result, a, b| {
                devectorize_binary(&op, result, a, b)
            },
        },

        // Any unary op that produces vectorized bool (only Not applies)
        for op in unary [*] {
            result @ op(src) if is_vectorized_bool(&result.dtype()) => |result, src| {
                devectorize_unary(&op, result, src)
            },
        },

        // WHERE with vectorized condition (output may not be bool, but condition is)
        // Based on Tinygrad's no_vectorized_alu (devectorizer.py:219-223)
        Where(cond, t, f) if cond.dtype().vcount() > 1 => |cond, t, f| {
            devectorize_where(cond, t, f)
        },

        // INDEX with vectorized bool dtype (cstyle.py:64)
        idx @ Index { buffer: _, .. } if is_vectorized_bool(&idx.dtype()) => devectorize_generic(idx),

        // CAST producing vectorized bool (cstyle.py:64)
        c @ Cast { src: _, .. } if is_vectorized_bool(&c.dtype()) => devectorize_generic(c),

        // CAST from vectorized bool (cstyle.py:66)
        c @ Cast { src, .. } if is_vectorized_bool(&src.dtype()) => devectorize_generic(c),

        // BITCAST with vectorized bool (cstyle.py:64)
        bc @ BitCast { src: _, .. } if is_vectorized_bool(&bc.dtype()) => devectorize_generic(bc),
    }
}

/// Devectorize WHERE operation by extracting elements with GEP and rebuilding with VECTORIZE.
///
/// Transforms: WHERE(<N x i1>, <N x T>, <N x T>) → VECTORIZE(WHERE(i1, T, T), ...)
fn devectorize_where(cond: &Arc<UOp>, t: &Arc<UOp>, f: &Arc<UOp>) -> Option<Arc<UOp>> {
    let vcount = cond.dtype().vcount();
    if vcount <= 1 {
        return None;
    }

    let t_vcount = t.dtype().vcount();
    let f_vcount = f.dtype().vcount();

    let scalar_wheres: SmallVec<[Arc<UOp>; 4]> = (0..vcount)
        .map(|i| {
            let cond_elem = cond.gep(vec![i]);
            let t_elem = if t_vcount > 1 { t.gep(vec![i]) } else { t.clone() };
            let f_elem = if f_vcount > 1 { f.gep(vec![i]) } else { f.clone() };
            UOp::try_where(cond_elem, t_elem, f_elem).expect("WHERE construction should succeed")
        })
        .collect();

    Some(UOp::vectorize(scalar_wheres))
}

// ============================================================================
// HORIZONTAL REDUCE (Tinygrad's devectorizer.py approach)
// ============================================================================

/// Apply binary reduce operation between two values (scalar or vector).
///
/// Handles all ReduceOp types:
/// - Add, Mul, Max: Direct binary ops
/// - Min: Implemented as Where(a < b, a, b) since BinaryOp::Min doesn't exist
///
/// Works with both scalar and vector dtypes - `Op::Binary` preserves input dtype,
/// performing element-wise operations for vectors.
fn apply_reduce_binary(reduce_op: ReduceOp, a: Arc<UOp>, b: Arc<UOp>, dtype: &DType) -> Arc<UOp> {
    // Tinygrad alignment: verify operand dtypes match for reduction chaining
    debug_assert!(
        a.dtype() == b.dtype(),
        "apply_reduce_binary: dtype mismatch between operands: a={:?}, b={:?}",
        a.dtype(),
        b.dtype()
    );

    match reduce_op {
        ReduceOp::Add => UOp::new(Op::Binary(BinaryOp::Add, a, b), dtype.clone()),
        ReduceOp::Mul => UOp::new(Op::Binary(BinaryOp::Mul, a, b), dtype.clone()),
        ReduceOp::Max => UOp::new(Op::Binary(BinaryOp::Max, a, b), dtype.clone()),
        ReduceOp::Min => {
            // Min(a, b) = Where(a < b, a, b)
            // Comparison dtype must match operand vector width:
            // - Scalar operands -> DType::Bool
            // - Vector operands -> DType::Bool.vec(vcount)
            let cond_dtype = DType::Bool.vec(dtype.vcount());
            let cond = UOp::new(Op::Binary(BinaryOp::Lt, a.clone(), b.clone()), cond_dtype);
            UOp::try_where(cond, a, b).unwrap()
        }
    }
}

/// Perform horizontal reduction from `src` dtype to `out_dtype`.
///
/// Uses stride pattern GEPs when reducing from larger to smaller vector.
/// Based on Tinygrad's `horizontal_reduce` (devectorizer.py:283-289).
///
/// Example: `<16 x float>` → `<4 x float>`
///   horizontal_amount = 16 / 4 = 4
///   Creates 4 GEPs with stride pattern:
///     gep([0, 4, 8, 12])  → `<4 x float>` (each output's 1st partial sum)
///     gep([1, 5, 9, 13])  → `<4 x float>` (each output's 2nd partial sum)
///     gep([2, 6, 10, 14]) → `<4 x float>` (each output's 3rd partial sum)
///     gep([3, 7, 11, 15]) → `<4 x float>` (each output's 4th partial sum)
///   Returns list of 4 `<4 x float>` vectors to be chained with ALU ops.
///
/// # Arguments
/// * `src` - The vectorized source (e.g., `<16 x float>`)
/// * `out_dtype` - Target output dtype (e.g., `<4 x float>` or scalar)
/// * `reduce_op` - The reduction operation (Add, Mul, Max, Min)
///
/// # Returns
/// A list of GEP operations, each producing `out_dtype` elements.
fn horizontal_reduce(src: &Arc<UOp>, out_dtype: &DType, reduce_op: ReduceOp) -> Vec<Arc<UOp>> {
    let src_count = src.dtype().vcount();
    let out_count = out_dtype.vcount();
    let horizontal_amount = src_count / out_count;

    // Edge case: uneven division - fall back to full scalar reduction
    // (Can happen with non-power-of-2 upcast amounts like 3)
    if !src_count.is_multiple_of(out_count) || horizontal_amount == 0 {
        let scalar_dtype = src.dtype().scalar_dtype();
        let elements: Vec<Arc<UOp>> = (0..src_count).map(|i| src.gep(vec![i])).collect();
        return vec![
            elements
                .into_iter()
                .reduce(|acc, elem| apply_reduce_binary(reduce_op, acc, elem, &scalar_dtype))
                .expect("src_count >= 2 guaranteed by guard"),
        ];
    }

    // Create stride pattern GEPs: range(i, src_count, horizontal_amount) for each i
    // This matches Tinygrad's: [inp.gep(tuple(range(i, inp.dtype.count, horizontal_amount))) for i in range(horizontal_amount)]
    (0..horizontal_amount)
        .map(|i| {
            let indices: Vec<usize> = (i..src_count).step_by(horizontal_amount).collect();
            src.gep(indices)
        })
        .collect()
}

/// Transform REDUCE with vectorized source to horizontal reduce + matching-dtype REDUCE.
///
/// When REDUCE has a vectorized input with different vcount than output dtype:
/// 1. First do horizontal reduction on the vector (extract stride-pattern elements, chain with ALU)
/// 2. Then REDUCE the result with the ranges (if any)
///
/// Based on Tinygrad's `reduce_to_acc` (devectorizer.py:296-316).
fn transform_vectorized_reduce(reduce: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Reduce { src, ranges, reduce_op } = reduce.op() else {
        return None;
    };

    let src_vcount = src.dtype().vcount();
    let out_vcount = reduce.dtype().vcount();

    // Only transform when source has more elements than output (need horizontal reduction).
    // Other cases (K-vectorized, bool reduce) are handled by pm_reduce_devectorize's branching.
    if src_vcount <= out_vcount {
        return None;
    }

    // Preserve the REDUCE's declared output dtype
    let out_dtype = reduce.dtype();

    trace!(
        src_vcount,
        out_vcount,
        reduce_op = ?reduce_op,
        out_dtype = ?out_dtype,
        "horizontal reducing vectorized REDUCE source"
    );

    // Get stride pattern GEPs
    let gep_list = horizontal_reduce(src, &out_dtype, *reduce_op);

    // Chain all GEPs: ((gep0 + gep1) + gep2) + gep3
    let chained = gep_list
        .into_iter()
        .reduce(|acc, elem| apply_reduce_binary(*reduce_op, acc, elem, &out_dtype))
        .expect("horizontal_reduce always returns non-empty list");

    // Wrap in REDUCE if ranges exist
    if ranges.is_empty() {
        Some(chained)
    } else {
        Some(UOp::new(Op::Reduce { src: chained, ranges: ranges.clone(), reduce_op: *reduce_op }, out_dtype))
    }
}

/// Unified guard: check if REDUCE needs any form of devectorization.
///
/// Combines three mutually exclusive cases:
/// 1. K-vectorized: CONTRACT source with vector output (SLP optimization)
/// 2. Bool reduce: matching vcounts, bool dtype, no CONTRACT (LLVM `<N x i1>` workaround)
/// 3. Horizontal: source has more elements than output (needs stride-pattern GEPs)
fn needs_reduce_devectorize(reduce: &Arc<UOp>) -> bool {
    let Op::Reduce { src, .. } = reduce.op() else {
        return false;
    };

    let src_vcount = src.dtype().vcount();
    let out_vcount = reduce.dtype().vcount();
    let is_bool = reduce.dtype().base() == ScalarDType::Bool;
    let has_contract = matches!(src.op(), Op::Contract { .. });

    // K-vectorized: CONTRACT source with vector output
    // Bool reduce: matching vcounts, bool dtype, no CONTRACT
    // Horizontal: source has more elements than output (but NOT vector→scalar, codegen handles that)
    has_contract && out_vcount > 1
        || out_vcount > 1 && is_bool && src_vcount == out_vcount
        || src_vcount > out_vcount && out_vcount > 1
}

/// Inline helper: check if REDUCE is K-vectorized (CONTRACT source).
#[inline]
fn is_k_vectorized(reduce: &Arc<UOp>, src: &Arc<UOp>) -> bool {
    reduce.dtype().vcount() > 1 && matches!(src.op(), Op::Contract { .. })
}

/// Inline helper: check if REDUCE is vectorized bool with matching vcounts.
#[inline]
fn is_bool_reduce(reduce: &Arc<UOp>, src: &Arc<UOp>) -> bool {
    let out_vcount = reduce.dtype().vcount();
    out_vcount > 1
        && reduce.dtype().base() == ScalarDType::Bool
        && src.dtype().vcount() == out_vcount
        && !matches!(src.op(), Op::Contract { .. })
}

/// Unified pattern matcher for REDUCE devectorization.
///
/// Combines three mutually exclusive REDUCE devectorization transforms:
///
/// 1. **K-vectorized** (CONTRACT source with vector output):
///    Transforms to N scalar REDUCEs + tree_reduce for SLP optimization.
///    Example: REDUCE(CONTRACT(src<4>), Add) → tree_reduce([REDUCE(gep(0)), ...])
///
/// 2. **Bool reduce** (matching vcounts, bool dtype, no CONTRACT):
///    Transforms to N scalar REDUCEs + VECTORIZE to avoid LLVM `<N x i1>` bugs.
///    Example: REDUCE(<2 x bool> src) → VECTORIZE(REDUCE(gep(0)), REDUCE(gep(1)))
///
/// 3. **Horizontal reduce** (src_vcount > out_vcount):
///    Transforms using stride-pattern GEPs + ALU chain.
///    Example: REDUCE(<16 x f32> → <4 x f32>) → chain(gep[0,4,8,12], gep[1,5,9,13], ...)
///
/// Based on Tinygrad's pm_reduce (devectorizer.py:283-316).
pub fn pm_reduce_devectorize() -> TypedPatternMatcher<()> {
    crate::patterns! {
        reduce @ Reduce { src } if needs_reduce_devectorize(reduce) => |reduce, src| {
            // Branch to appropriate transform based on case
            if is_k_vectorized(reduce, src) {
                devectorize_to_scalar_accumulators(reduce)
            } else if is_bool_reduce(reduce, src) {
                devectorize_bool_reduce(reduce)
            } else {
                transform_vectorized_reduce(reduce)
            }
        },
    }
}

// ============================================================================
// REDUCE DEVECTORIZATION TRANSFORMS (used by pm_reduce_devectorize)
// ============================================================================

/// Devectorize REDUCE with vectorized bool output to N scalar REDUCEs.
///
/// This avoids LLVM's `<N x i1>` accumulator issues by creating independent
/// scalar bool accumulators for each vector lane, then combining with VECTORIZE.
///
/// Transforms:
/// ```text
/// REDUCE(Min, <2 x bool> src, ranges)
/// → VECTORIZE(
///     REDUCE(Min, GEP(src, 0), ranges),  // lane 0: scalar bool accumulator
///     REDUCE(Min, GEP(src, 1), ranges)   // lane 1: scalar bool accumulator
///   )
/// ```
///
/// Based on Tinygrad's `no_vectorized_alu` pattern which devectorizes ALU ops.
fn devectorize_bool_reduce(reduce: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Reduce { src, ranges, reduce_op } = reduce.op() else {
        return None;
    };

    let vcount = reduce.dtype().vcount();
    if vcount <= 1 {
        return None;
    }

    let scalar_dtype = reduce.dtype().scalar_dtype();

    trace!(
        vcount,
        reduce_op = ?reduce_op,
        src_dtype = ?src.dtype(),
        "devectorizing bool REDUCE to avoid <N x i1> accumulators"
    );

    // Create N scalar REDUCEs, one for each vector lane
    let scalar_reduces: SmallVec<[Arc<UOp>; 4]> = (0..vcount)
        .map(|i| {
            let src_elem = src.gep(vec![i]);
            UOp::new(Op::Reduce { src: src_elem, ranges: ranges.clone(), reduce_op: *reduce_op }, scalar_dtype.clone())
        })
        .collect();

    // Wrap in VECTORIZE to produce the expected vector output
    Some(UOp::vectorize(scalar_reduces))
}

/// Devectorize K-vectorized REDUCE to N scalar REDUCEs for SLP optimization.
///
/// This transforms vectorized accumulators (created by K-vectorization)
/// into independent scalar accumulators that LLVM's SLP vectorizer can group.
///
/// Input:
///   REDUCE(CONTRACT(vec_src<N>, upcast_axes), ranges, Add, Vector<N>)
///
/// Output:
///   tree_reduce([
///     REDUCE(GEP(vec_src, 0), ranges, Add, scalar),
///     REDUCE(GEP(vec_src, 1), ranges, Add, scalar),
///     ...
///     REDUCE(GEP(vec_src, N-1), ranges, Add, scalar),
///   ], Add)
fn devectorize_to_scalar_accumulators(reduce: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Reduce { src, ranges, reduce_op } = reduce.op() else {
        return None;
    };

    let vec_count = reduce.dtype().vcount();
    if vec_count <= 1 {
        return None;
    }

    // Unwrap CONTRACT to get vectorized source
    let vec_src = if let Op::Contract { src: inner, .. } = src.op() { inner.clone() } else { src.clone() };

    let scalar_dtype = reduce.dtype().scalar_dtype();

    trace!(
        vec_count,
        reduce_op = ?reduce_op,
        src_dtype = ?vec_src.dtype(),
        "devectorizing K-vectorized REDUCE to scalar accumulators"
    );

    // Create N scalar REDUCEs, each extracting one lane from vectorized source
    let scalar_reduces: Vec<Arc<UOp>> = (0..vec_count)
        .map(|i| {
            let src_elem = vec_src.gep(vec![i]);
            UOp::new(Op::Reduce { src: src_elem, ranges: ranges.clone(), reduce_op: *reduce_op }, scalar_dtype.clone())
        })
        .collect();

    // Tree reduction: combine scalar REDUCEs with binary ops
    // ((r0 + r1) + (r2 + r3)) for balanced tree
    Some(tree_reduce(&scalar_reduces, *reduce_op, &scalar_dtype))
}

/// Perform tree reduction on elements using binary reduce operations.
///
/// Creates balanced tree: ((e0 op e1) op (e2 op e3)) instead of linear chain.
/// This is more efficient for parallel execution.
fn tree_reduce(elements: &[Arc<UOp>], reduce_op: ReduceOp, dtype: &DType) -> Arc<UOp> {
    if elements.len() == 1 {
        return elements[0].clone();
    }

    // Pairwise combine for balanced tree
    let mut level: Vec<Arc<UOp>> = elements.to_vec();
    while level.len() > 1 {
        let mut next_level = Vec::with_capacity(level.len().div_ceil(2));
        for chunk in level.chunks(2) {
            if chunk.len() == 2 {
                next_level.push(apply_reduce_binary(reduce_op, chunk[0].clone(), chunk[1].clone(), dtype));
            } else {
                next_level.push(chunk[0].clone());
            }
        }
        level = next_level;
    }
    level.remove(0)
}

// ============================================================================
// FMA DECOMPOSITION (a*b+c → MulAcc)
// ============================================================================

/// FMA pattern detection: a*b+c → MulAcc(a,b,c)
///
/// Based on Tinygrad's decompositions.py:362 (Python syntax):
/// ```text
/// if Ops.MULACC in ops: pat += [
///     (UPat.var('a')*UPat.var('b')+UPat.var('c'),
///      lambda a,b,c: a.alu(Ops.MULACC, b, c))
/// ]
/// ```
///
/// Applied late (post-optimization) so earlier passes can still work with Add(Mul) structure.
/// Only matches float types where FMA provides benefit (maps to llvm.fma intrinsic).
pub fn pm_fma_decomposition() -> TypedPatternMatcher<()> {
    crate::patterns! {
        // (a*b)+c or c+(a*b) → MulAcc(a,b,c) using commutative matching
        Add[Mul(a, b), c] if a.dtype().is_float() => |a, b, c| {
            UOp::try_mulacc(a.clone(), b.clone(), c.clone()).ok()
        },
    }
}

// ============================================================================
