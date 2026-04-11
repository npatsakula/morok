//! Consolidated pattern matchers for rangeify transformations.
//!
//! This module contains all pattern matchers used during scheduling/rangeify:
//! - Early cleanup rewrites (DETACH, CONTIGUOUS_BACKWARD removal)
//! - Movement op → BUFFERIZE conversion
//! - Buffer folding and removal
//! - Kernel splitting patterns (BUFFER → PARAM, AFTER handling)
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

/// Check if a UOp has zero total size (any shape dimension is 0).
/// Tinygrad equivalent: `x.size == 0` where `size = prod(shape)`.
fn has_zero_size(uop: &Arc<UOp>) -> bool {
    match uop.shape() {
        Ok(Some(shape)) => shape.iter().any(|d| d.as_const() == Some(0)),
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
/// - Zero-size tensor folding (Tinygrad: rangeify.py:116-120)
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
        },

        // Reduce of zero-sized input → identity element (Tinygrad: rangeify.py:116-117)
        reduce @ ReduceAxis { src: x } if has_zero_size(x) && !has_zero_size(reduce) => {
            let Op::ReduceAxis { reduce_op, .. } = reduce.op() else { return None };
            Some(crate::symbolic::dce::reduce_identity(*reduce_op, reduce.dtype()))
        },

        // Any non-SINK op with zero size → const 0 (Tinygrad: rangeify.py:119-120)
        x if !matches!(x.op(), Op::Sink { .. }) && has_zero_size(x) => {
            Some(x.const_like(0))
        }
    }
}

// ============================================================================
// REPLACE_CONTIGUOUS PATTERNS
// ============================================================================

/// Context for replace_contiguous: maps base UOps to their contiguous copies.
pub type ReplaceContiguousCtx = std::collections::HashMap<UOpKey, Arc<UOp>>;

/// Push CONTIGUOUS inward through movement ops to the base buffer.
///
/// Based on Tinygrad's `replace_contiguous` (rangeify.py:536-539):
/// - `CONTIGUOUS(PERMUTE(RESHAPE(...(BASE))))` → adjust contiguous through the chain, then `ctx[BASE] = adjusted_contig`
/// - For ALU ops: replace any source that appears in `ctx`
///
/// Only handles PERMUTE and RESHAPE movement ops (Tinygrad: found_contiguous returns None for others).
pub fn replace_contiguous() -> TypedPatternMatcher<ReplaceContiguousCtx> {
    crate::patterns! {
        @context ReplaceContiguousCtx;

        // Match CONTIGUOUS wrapping a movement op chain
        contig @ Contiguous { src, .. } if src.op().is_movement() => |contig, src, ctx| {
            found_contiguous(ctx, contig, src)
        },

        // ALU ops: replace sources that are in ctx (base → contiguous)
        // Tinygrad: UPat(GroupOp.ALU) — only Unary/Binary/Ternary ops.
        x if matches!(x.op(), Op::Unary(..) | Op::Binary(..) | Op::Ternary(..)) => |x, ctx| {
            if ctx.is_empty() { return None; }
            let sources = x.op().sources();

            let mut new_sources = Vec::with_capacity(sources.len());
            let mut any_changed = false;
            for src in sources.iter() {
                let key = UOpKey(src.clone());
                if let Some(replacement) = ctx.get(&key) {
                    new_sources.push(replacement.clone());
                    any_changed = true;
                } else {
                    new_sources.push(src.clone());
                }
            }
            if any_changed { Some(x.with_sources(new_sources)) } else { None }
        },
    }
}

/// Walk from a movement op chain toward its base, adjusting the contiguous marker.
///
/// Based on Tinygrad's `found_contiguous` (rangeify.py:528-535).
#[allow(clippy::mutable_key_type)] // UOpKey Hash/Eq is stable despite interior OnceLock
fn found_contiguous(ctx: &mut ReplaceContiguousCtx, contig: &Arc<UOp>, src: &Arc<UOp>) -> Option<Arc<UOp>> {
    let base = src.base();
    let mut adjusted_contig = contig.clone();
    let mut x = src.clone();

    while !Arc::ptr_eq(&x, &base) {
        match x.op() {
            Op::Permute { src: inner, axes } => {
                // Apply inverse permutation: contig = contig.permute(argsort(axes))
                let inv = argsort(axes);
                adjusted_contig = adjusted_contig.try_permute(inv).ok()?;
                x = inner.clone();
            }
            Op::Reshape { src: inner, .. } => {
                // Reshape contig to pre-reshape shape (inner source's shape)
                let inner_shape = inner.shape().ok()??;
                adjusted_contig = adjusted_contig.try_reshape(inner_shape).ok()?;
                x = inner.clone();
            }
            _ => return None, // Unsupported movement op — bail
        }
    }

    // Map the base to the adjusted contiguous
    ctx.insert(UOpKey(base.clone()), adjusted_contig);
    None // Don't transform the CONTIGUOUS node itself; ALU pattern does the replacement
}

/// Compute inverse permutation (argsort).
fn argsort(perm: &[usize]) -> Vec<usize> {
    let mut inv = vec![0; perm.len()];
    for (i, &p) in perm.iter().enumerate() {
        inv[p] = i;
    }
    inv
}

// ============================================================================
// RANGEIFY TRANSFORMATION PATTERNS
// ============================================================================

/// Create patterns for applying rangeify transformation with IndexingContext.
///
/// Pattern order (like Tinygrad's pm_apply_rangeify):
/// 1. ReduceAxis → REDUCE conversion
/// 2. PAD → WHERE conversion (convert_pad_to_where_to_keep_behavior_local)
/// 3. ALL ops get source bufferization (including movement ops)
/// 4. Movement ops get removed (simple - just return source)
pub fn apply_rangeify_patterns() -> TypedPatternMatcher<IndexingContext> {
    crate::patterns! {
        @context IndexingContext;
        // ReduceAxis conversion MUST come first - before bufferize wraps it
        x @ ReduceAxis { src: _ } => |x, ctx| convert_reduceaxis_with_context(x, ctx),
        // PAD → WHERE conversion BEFORE bufferization
        // Tinygrad: convert_pad_to_where_to_keep_behavior_local
        x @ Pad { src: _, begin_pads: _, end_pads: _ } => |x, ctx| convert_pad_to_where(x, ctx),
        // ALL ops (including movement) get source bufferization
        // This matches Tinygrad's approach where bufferization runs on GroupOp.All
        x => |x, ctx| apply_bufferize_transform(x, ctx),
        // Movement ops get removed AFTER bufferization - simple logic
        x if x.op().is_movement() => |x, ctx| remove_movement_op(x, ctx),
    }
}

/// Apply BUFFERIZE transformation to op sources.
///
/// When sources change, the new node gets a different Arc identity. We must
/// transfer range_map + realize_map so downstream patterns (e.g. `remove_movement_op`)
/// can find the new node's context — same as `convert_reduceaxis_with_context`.
fn apply_bufferize_transform(x: &Arc<UOp>, ctx: &mut IndexingContext) -> Option<Arc<UOp>> {
    if let Some(new_sources) = transform_sources_with_bufferize(x, ctx) {
        let new_node = x.with_sources(new_sources);
        // Transfer context to new identity
        if let Some((in_rngs, out_rngs)) = ctx.get_ranges(x) {
            ctx.set_ranges(&new_node, in_rngs.clone(), out_rngs.clone());
        }
        if let Some(realize_axes) = ctx.get_realize_axes(x).cloned() {
            ctx.mark_realize(&new_node, realize_axes);
        }
        return Some(new_node);
    }
    None
}

/// Convert PAD → WHERE(combined_valid, source, 0).
///
/// Tinygrad's `convert_pad_to_where_to_keep_behavior_local` (indexing.py:77-82):
/// Extracts validity conditions from PAD's input ranges (WHERE-Invalid patterns)
/// and wraps the PAD's data source in a WHERE that produces 0 for padded regions.
fn convert_pad_to_where(x: &Arc<UOp>, ctx: &mut IndexingContext) -> Option<Arc<UOp>> {
    let (input_ranges, output_ranges) = ctx.get_ranges(x)?;
    let input_ranges = input_ranges.clone();
    let output_ranges = output_ranges.clone();

    // Extract validity from each input range and AND them together.
    // Ranges from PAD are WHERE(valid, adjusted_idx, Invalid) → get_valid() extracts valid.
    // Non-pad ranges return constant true.
    let mut combined_valid: Option<Arc<UOp>> = None;
    for r in &input_ranges {
        let valid = r.get_valid();
        combined_valid = Some(match combined_valid {
            None => valid,
            Some(acc) => acc.try_and_op(&valid).unwrap_or(valid),
        });
    }

    let combined_valid = combined_valid?;

    // If all ranges are always-valid (no padding), no transformation needed
    if matches!(combined_valid.op(), Op::Const(cv) if cv.0 == ConstValue::Bool(true)) {
        return None;
    }

    // PAD source is the first source
    let pad_src = x.op().sources().first()?.clone();

    // Create WHERE(combined_valid, pad_source, const_0).
    // The scalar zero is compatible with any-shaped pad_src via implicit broadcast.
    let zero = UOp::const_(x.dtype(), ConstValue::zero(x.dtype().scalar().unwrap()));
    let ret = UOp::try_where(combined_valid, pad_src, zero).ok()?;

    // Transfer range_map as-is (Tinygrad keeps WHERE-Invalid in ranges).
    // pm_lower_index_dtype later converts INDEX(buf, WHERE(cond, idx, Invalid))
    // into gated INDEX(buf, idx, gate=cond).
    ctx.set_ranges(&ret, input_ranges, output_ranges);

    Some(ret)
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
    // Preserve tag from ReduceAxis (Tinygrad indexing.py:87: tag=x.tag)
    let target = if let Some(t) = x.tag() { target.with_tag(t.clone()) } else { target };

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
        Index { buffer: buf @ Bufferize { compute, ranges, .. }, indices, gate: None }
            if ranges_equal(ranges, indices) && !matches!(compute.op(), Op::BufferView { .. })
            => |compute, buf, ranges| {
                // Tinygrad (rangeify.py:257-258): merge tags, shrink to bufferize shape
                let mut merged = SmallVec::<[usize; 2]>::new();
                if let Some(t) = compute.tag() { merged.extend(t.iter().copied()); }
                if let Some(t) = buf.tag() { merged.extend(t.iter().copied()); }
                let tag = if merged.is_empty() { None } else { Some(merged) };
                let result = compute.rtag(tag);
                // Tinygrad (rangeify.py:258): .shrink(tuple((0, s) for s in b2.shape)) if b2.shape
                // try_shrink has noop detection (returns self when result shape == shrink shape)
                if !ranges.is_empty()
                    && let (Ok(Some(buf_shape)), Ok(Some(_))) = (buf.shape(), result.shape()) {
                        let shrink_ranges: Vec<_> = buf_shape.iter()
                            .map(|s| (morok_ir::SInt::Const(0), s.clone()))
                            .collect();
                        if let Ok(shrunk) = result.try_shrink(&shrink_ranges) {
                            return Some(shrunk);
                        }
                    }
                Some(result)
            },
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

    // NOTE: Even when ALL ranges are dead (scalar output), we MUST keep the BUFFERIZE.
    // Tinygrad keeps the BUFFERIZE with empty ranges, wraps with RESHAPE+EXPAND.
    // The BUFFERIZE will be converted to STORE by pm_add_buffers_local_patterns later.
    // Removing it here would cause NoKernelsFound since no STORE gets created.

    // Create BUFFERIZE with fewer (or zero) ranges
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
        Bufferize { compute: Bufferize { compute: inner, .. }, ranges, opts }
            => |inner, ranges, opts| Some(UOp::bufferize(Arc::clone(inner), ranges.to_vec(), opts.clone())),
    }
}

/// Pattern matcher for cost-based buffer removal with partial contiguous support.
pub fn buffer_removal_with_pcontig() -> TypedPatternMatcher<PcontigConfig> {
    crate::patterns! {
        @context PcontigConfig;
        // Partial contiguous removal - ungated INDEX with WHERE-Invalid in indices.
        // WHERE-Invalid stays in INDEX indices (not extracted to gate). Extract cond,
        // inline with clean idx, wrap result with WHERE(cond, inlined, 0).
        // Must be before the general ungated pattern to match first.
        Index {
            buffer: buffer @ Bufferize { compute: src, ranges: buf_ranges, .. },
            indices: idx_ranges,
            gate: None
        } if idx_ranges.len() == 1
          && matches!(idx_ranges[0].op(), Op::Ternary(morok_ir::TernaryOp::Where, _, _, f) if matches!(f.op(), Op::Invalid))
        => |buffer, src, buf_ranges, idx_ranges, ctx| {
            let Op::Ternary(morok_ir::TernaryOp::Where, cond, clean_idx, _) = idx_ranges[0].op() else {
                unreachable!()
            };
            let clean_indices: SmallVec<[Arc<UOp>; 4]> = smallvec::smallvec![clean_idx.clone()];
            let inlined = apply_pcontig_removal_inner(buffer, src, buf_ranges, &clean_indices, ctx)?;
            let zero = UOp::const_(inlined.dtype(), ConstValue::zero(inlined.dtype().scalar()?));
            UOp::try_where(cond.clone(), inlined, zero).ok()
        },

        // Partial contiguous removal - ungated INDEX (no WHERE-Invalid)
        Index {
            buffer: buffer @ Bufferize { compute: src, ranges: buf_ranges, .. },
            indices: idx_ranges,
            gate: None
        } => |buffer, src, buf_ranges, idx_ranges, ctx| {
            apply_pcontig_removal_inner(buffer, src, buf_ranges, idx_ranges, ctx)
        },

        // Partial contiguous removal - gated INDEX (from pm_move_where_on_load)
        // Inline the buffer, then wrap with WHERE(gate, inlined, 0) so out-of-bounds
        // values become zero.
        Index {
            buffer: buffer @ Bufferize { compute: src, ranges: buf_ranges, .. },
            indices: idx_ranges,
            gate: Some(gate)
        } => |buffer, src, buf_ranges, idx_ranges, gate, ctx| {
            let inlined = apply_pcontig_removal_inner(buffer, src, buf_ranges, idx_ranges, ctx)?;
            let zero = UOp::const_(inlined.dtype(), ConstValue::zero(inlined.dtype().scalar()?));
            UOp::try_where(gate.clone(), inlined, zero).ok()
        },

        // Constant buffer folding — BUFFERIZE(CONST) → CONST (like Tinygrad's pm_const_buffer_folding).
        // Only matches bare constants, not arbitrary cheap ops.
        Bufferize { compute: compute @ Const(_), .. }
            if ctx.level > 0
            => |compute| Some(compute.clone()),

        // Flatten nested Bufferize
        Bufferize { compute: Bufferize { compute: inner, .. }, ranges, opts }
            => |inner, ranges, opts| Some(UOp::bufferize(Arc::clone(inner), ranges.to_vec(), opts.clone())),
    }
}

/// Remove a BUFFERIZE+INDEX pair by inlining the computation (like Tinygrad's `remove_bufferize`).
///
/// Decision flow (matching Tinygrad):
/// 1. Always-run ops (Contiguous/Copy/Assign) → keep
/// 2. Buffer count > threshold (bypassed at level > 2) → keep
/// 3. No buffer in reduce scope → simple range substitution (always inline)
/// 4. Buffer in reduce at level ≤ 2 → keep (Tinygrad's `PCONTIG ≤ 2` default)
/// 5. Buffer in reduce at level > 2 → ratio check + partial contiguous
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

    // Non-removable BUFFERIZEs are multi-consumer realize boundaries.
    // Inlining them duplicates computation into each consumer's kernel.
    if let Op::Bufferize { opts, .. } = buffer.op()
        && !opts.removable
    {
        tracing::debug!(src_id = src.id, src_op = src.op().as_ref(), "buffer_removal: KEPT (non-removable)");
        return None;
    }

    // Single-pass collection of buffers, indexes, reduces in the src subtree.
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
            // Tinygrad (rangeify.py:208-209): BUFFERIZE(Global) and MSTACK count + stop traversal
            Op::Bufferize { opts, .. } if opts.addrspace == AddrSpace::Global => {
                buffers.push(Arc::clone(uop));
                return false; // Stop traversing into GLOBAL bufferize
            }
            Op::MStack { .. } => {
                buffers.push(Arc::clone(uop));
                return false; // Stop traversing into MSTACK (Tinygrad returns False)
            }
            // Tinygrad (rangeify.py:211): PARAM counts but continues traversal
            Op::Param { .. } => {
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

    // Buffer count threshold — bypassed at level > 2 (like Tinygrad's `not (PCONTIG > 2)`)
    if accessed_buffers.len() > config.max_buffers_threshold && config.level <= 2 {
        tracing::debug!(
            src_id = src.id,
            src_op = src.op().as_ref(),
            buf_count = accessed_buffers.len(),
            threshold = config.max_buffers_threshold,
            buf_ops = ?accessed_buffers.iter().map(|b| (b.id, b.op().as_ref())).collect::<Vec<_>>(),
            "buffer_removal: KEPT (buffer count exceeds threshold)"
        );
        return None;
    }

    // Check if any reduce body accesses a buffer
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
            // Tinygrad (rangeify.py:228): checks for PARAM or BUFFERIZE in reduce body.
            // Buffer is normalized to Param before buffer removal runs.
            sink.any_in_subtree(|n| matches!(n.op(), Op::Param { .. } | Op::Bufferize { .. }))
        }
    };

    // No buffer in reduce: range substitution with gating (always inline).
    // Tinygrad (rangeify.py:253): `src.substitute({k:v for k,v in zip(...) if k.op is not Ops.CONST}, extra_pm=pm_gate_substitute)`
    // Filter: skip CONST ranges (they're broadcast dims, not real ranges).
    // Gating: only substitute in subexpressions that contain the ranges.
    if !buffer_in_reduce {
        tracing::debug!(
            src_id = src.id,
            src_op = src.op().as_ref(),
            buf_count = accessed_buffers.len(),
            "buffer_removal: REMOVED (no buffer_in_reduce)"
        );
        let subs_map: HashMap<UOpKey, Arc<UOp>> = buf_ranges
            .iter()
            .zip(idx_ranges.iter())
            .filter(|(k, _)| !matches!(k.op(), Op::Const(_)))
            .map(|(k, v)| (UOpKey(Arc::clone(k)), Arc::clone(v)))
            .collect();
        return Some(src.substitute_gated(&subs_map));
    }

    // Buffer in reduce: at level ≤ 2, always keep (Tinygrad rangeify.py:247-248).
    if config.level <= 2 {
        tracing::debug!(
            src_id = src.id,
            src_op = src.op().as_ref(),
            reduce_count = reduces.len(),
            buf_count = accessed_buffers.len(),
            "buffer_removal: KEPT (buffer_in_reduce at level<=2)"
        );
        return None;
    }

    // Output/input ratio check — only computed here (level > 2 with buffer_in_reduce)
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
            // After fix A1, accessed_buffers only contains Bufferize(Global), MStack, Param
            Op::Param { size, .. } => Some(*size),
            Op::MStack { .. } => Some(1),
            _ => None,
        })
        .sum();

    let ratio = (output_size + 1) as f64 / (input_size + 1) as f64;
    if ratio < config.out_in_ratio_threshold {
        return None;
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
    let substituted = src.substitute_gated(&subs_map);

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

/// Pattern matcher for reduce_unparented: remove ranges not referenced by body.
///
/// Factored out so it can be shared between `pm_reduce_simplify` and the inner
/// `reduce_collapse` pattern matchers without duplication.
fn pm_reduce_unparented() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        reduce @ Reduce { src, ranges, reduce_op: Add | Mul | Max | Min } => |reduce, src, ranges, reduce_op| {
            // Tinygrad simplify.py:72: assert all(x.op is Ops.RANGE for x in red.src[1:])
            assert!(
                ranges.iter().all(|r| matches!(r.op(), Op::Range { .. })),
                "reduce_unparented: all reduce ranges must be RANGE ops, got: {:?}",
                ranges.iter().map(|r| r.op().as_ref().to_string()).collect::<Vec<_>>()
            );
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
    }
}

/// Check if a UOp's DAG references any of the given reduce ranges.
///
/// Uses `in_scope_ranges` (cached property) to check if any of the given ranges
/// appear in the UOp's dependency graph.
#[allow(clippy::mutable_key_type)]
fn references_any_reduce_range(uop: &Arc<UOp>, ranges: &SmallVec<[Arc<UOp>; 4]>) -> bool {
    let in_scope = uop.in_scope_ranges();
    ranges.iter().any(|r| in_scope.contains(&UOpKey(r.clone())))
}

/// Split a UOp by multiplication into leaf factors.
///
/// E.g. `(a * b) * c` → `[a, b, c]`
fn split_mul_factors(uop: &Arc<UOp>) -> SmallVec<[Arc<UOp>; 4]> {
    match uop.op() {
        Op::Binary(BinaryOp::Mul, a, b) => {
            let mut factors = split_mul_factors(a);
            factors.extend(split_mul_factors(b));
            factors
        }
        _ => smallvec::smallvec![uop.clone()],
    }
}

/// Factor multiplicative terms that don't depend on reduce ranges outside the REDUCE.
///
/// For ADD reduce: `REDUCE(x * c, ADD, ranges)` → `REDUCE(x, ADD, ranges) * c`
/// For MAX reduce: same, but only if the outside factor's vmin >= 0.
fn reduce_mul_chain(src: &Arc<UOp>, ranges: &SmallVec<[Arc<UOp>; 4]>, reduce_op: ReduceOp) -> Option<Arc<UOp>> {
    let factors = split_mul_factors(src);
    if factors.len() < 2 {
        return None;
    }

    let mut inside: SmallVec<[Arc<UOp>; 4]> = SmallVec::new();
    let mut outside: SmallVec<[Arc<UOp>; 4]> = SmallVec::new();

    for factor in &factors {
        if references_any_reduce_range(factor, ranges) {
            inside.push(factor.clone());
        } else {
            // For MAX reduce, only factor out non-negative values
            if reduce_op == ReduceOp::Max {
                let is_non_negative = match factor.vmin() {
                    ConstValue::Int(v) => *v >= 0,
                    ConstValue::UInt(_) => true,
                    ConstValue::Float(v) => *v >= 0.0,
                    ConstValue::Bool(_) => true,
                };
                if !is_non_negative {
                    inside.push(factor.clone());
                    continue;
                }
            }
            outside.push(factor.clone());
        }
    }

    if outside.is_empty() {
        return None;
    }

    // Reconstruct inside product (if all factors are outside, reduce over const 1)
    let inner = inside.into_iter().reduce(|a, b| a.mul(&b)).unwrap_or_else(|| src.const_like(1i64));
    let reduced = inner.reduce(ranges.clone(), reduce_op);

    // Multiply by outside product
    let mut result = reduced;
    for factor in &outside {
        result = result.mul(factor);
    }

    Some(result)
}

/// Pattern matcher for reduction simplifications (mega-pass path).
///
/// Matches Tinygrad's `pm_reduce_simplify`:
/// - `pm_reduce_unparented`: remove ranges not referenced by body
/// - `REDUCE(ADD) → reduce_collapse(src, ranges)`: delegate to procedural wrapper
/// - `reduce_mul_chain`: factor range-independent multipliers outside REDUCE
///
/// No duplication of distributive or bound patterns — those live inside
/// `reduce_collapse_inner_patterns()` and run via `reduce_collapse`.
pub fn pm_reduce_simplify() -> &'static TypedPatternMatcher {
    static CACHED: std::sync::LazyLock<TypedPatternMatcher> = std::sync::LazyLock::new(|| {
        pm_reduce_unparented()
            + crate::patterns! {
                Reduce { src, ranges, reduce_op } if *reduce_op == ReduceOp::Add
                    => |src, ranges| super::transforms::reduce_collapse(src, ranges),

                Reduce { src, ranges, reduce_op }
                    if matches!(reduce_op, ReduceOp::Add | ReduceOp::Max)
                    && matches!(src.op(), Op::Binary(BinaryOp::Mul, _, _))
                    => |src, ranges, reduce_op| reduce_mul_chain(src, ranges, *reduce_op),
            }
    });
    &CACHED
}

// ============================================================================
// MOVEMENT OP PATTERNS (pm_mops equivalent)
// ============================================================================

/// Create pattern matcher for pushing movement ops through INDEX operations.
///
/// Based on Tinygrad's `pm_mops` (rangeify.py:18-25).
pub fn movement_op_patterns() -> TypedPatternMatcher {
    crate::patterns! {
        // pm_mops rule 1: Movement op on INDEX → apply_movement_op to indices (Tinygrad rangeify.py:25-26)
        Index { buffer: mop, indices, gate } if mop.op().is_movement() => |mop, indices, gate| {
            transform_movement_through_index(mop, indices, gate)
        },
        // pm_mops rule 2: Movement ops through AFTER (Tinygrad rangeify.py:28-29)
        // AFTER(MOVEMENT(x, ...), deps) → MOVEMENT(AFTER(x, deps), ...)
        After { passthrough: mop, deps } if mop.op().is_movement()
            => |mop, deps| {
                super::transforms::push_movement_through_after(mop, deps)
            },
        // pm_mops rule 3: Movement ops through END (Tinygrad rangeify.py:30)
        // END(MOVEMENT(x, ...), ranges) → END(x, ranges)
        End { computation: mop, ranges } if mop.op().is_movement()
            => |mop, ranges| {
                let src = &mop.op().sources()[0];
                Some(src.end(ranges.clone()))
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

/// Pattern matcher for INDEX concatenation.
///
/// Based on Tinygrad's `pm_syntactic_sugar` (codegen/__init__.py:22-26).
///
/// Matches INDEX on ptr INDEX and concatenates their indices:
/// ```text
/// INDEX(INDEX(buffer, indices1..., gate1?), indices2..., gate2?)
/// → INDEX(buffer, indices1... + indices2..., combined_gate?)
/// ```
///
/// This only applies when:
/// - Inner INDEX has PtrDType (is a pointer)
/// - Outer INDEX doesn't have PtrDType (is an element access)
pub fn pm_syntactic_sugar() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // INDEX on ptr INDEX concats them
        outer @ Index { buffer: inner @ Index { buffer: base_buffer, indices: inner_indices, gate: inner_gate }, indices: outer_indices, gate: outer_gate }
            if matches!(inner.dtype(), DType::Ptr { .. }) && !matches!(outer.dtype(), DType::Ptr { .. })
            => |outer, inner, base_buffer, inner_indices, outer_indices, inner_gate, outer_gate| {
                concat_index_indices(base_buffer, inner_indices, outer_indices, inner_gate, outer_gate, outer.dtype())
            },
    }
}

/// Concatenate indices from nested INDEX operations.
fn concat_index_indices(
    base_buffer: &Arc<UOp>,
    inner_indices: &SmallVec<[Arc<UOp>; 4]>,
    outer_indices: &SmallVec<[Arc<UOp>; 4]>,
    inner_gate: &Option<Arc<UOp>>,
    outer_gate: &Option<Arc<UOp>>,
    result_dtype: DType,
) -> Option<Arc<UOp>> {
    // Concatenate: inner indices + outer indices
    let mut combined: SmallVec<[Arc<UOp>; 4]> = inner_indices.clone();
    combined.extend(outer_indices.iter().cloned());

    // Combine gates: if both exist, AND them; if one exists, use that; if neither, None
    let combined_gate = match (inner_gate, outer_gate) {
        (Some(g1), Some(g2)) => Some(g1.and_(g2)),
        (Some(g), None) | (None, Some(g)) => Some(g.clone()),
        (None, None) => None,
    };

    // Build new INDEX with combined indices
    match combined_gate {
        Some(g) => UOp::index().buffer(base_buffer.clone()).indices(combined).dtype(result_dtype).gate(g).call().ok(),
        None => UOp::index().buffer(base_buffer.clone()).indices(combined).dtype(result_dtype).call().ok(),
    }
}

/// Transform a movement op through INDEX by applying the movement to indices.
pub(crate) fn transform_movement_through_index(
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
///
/// # CONTIGUOUS Hint Extraction
///
/// Based on Tinygrad's `get_contiguous` (rangeify.py:436-438):
/// ```python
/// def get_contiguous(ctx:LocalAddBufferContext, x:UOp):
///   if isinstance(x.arg, tuple) and all(isinstance(y, Opt) for y in x.arg): ctx.opts = x.arg
///   return x.src[0]
/// ```
///
/// Extracts optimization hints from CONTIGUOUS.opts into ctx.opts for later use.
pub fn rangeify_codegen_patterns() -> TypedPatternMatcher<LocalAddBufferContext> {
    crate::patterns! {
        @context LocalAddBufferContext;
        // NOOP → zero constant (scalar or vector)
        noop @ Noop() if noop.dtype().base() != morok_dtype::ScalarDType::Void => |noop, _ctx| {
            Some(dtype_zero(noop.dtype()))
        },
        // CONTIGUOUS: extract hints and return source (Tinygrad: get_contiguous)
        Contiguous { src, opts } => |src, opts, ctx| {
            if !opts.is_empty() {
                ctx.opts.extend(opts.iter().cloned());
            }
            Some(src.clone())
        },
    }
}

/// Context-free rangeify_codegen for post-expander pipeline (Stage 10).
///
/// Simplified version of `rangeify_codegen_patterns()` that doesn't require
/// `LocalAddBufferContext`. At this point CONTIGUOUS opts have already been
/// extracted during kernel splitting, so we just strip CONTIGUOUS and NOOP.
///
/// Based on Tinygrad's codegen/__init__.py:79 where `rangeify_codegen` runs
/// with `ctx=itertools.count(0)` (no LocalAddBufferContext).
pub fn rangeify_codegen_simple() -> TypedPatternMatcher {
    crate::patterns! {
        // NOOP → zero constant (scalar or vector)
        noop @ Noop() if noop.dtype().base() != morok_dtype::ScalarDType::Void => |noop| {
            Some(dtype_zero(noop.dtype()))
        },
        // CONTIGUOUS → source (strip wrapper, no opts to extract at this stage)
        Contiguous { src, .. } => |src| {
            Some(src.clone())
        },
    }
}

/// rangeify_codegen that accepts KernelContext for combining with pm_add_buffers_local.
///
/// This is the same as `rangeify_codegen_simple` but accepts `KernelContext` to allow
/// combining with `pm_add_buffers_local_patterns` in a single pass (like Tinygrad does).
pub fn rangeify_codegen_with_kernel_ctx() -> TypedPatternMatcher<super::kernel::KernelContext> {
    crate::patterns! {
        @context super::kernel::KernelContext;
        // NOOP → zero constant (scalar or vector)
        noop @ Noop() if noop.dtype().base() != morok_dtype::ScalarDType::Void => |noop, _ctx| {
            Some(dtype_zero(noop.dtype()))
        },
        // CONTIGUOUS → source (strip wrapper, no opts to extract at this stage)
        Contiguous { src, .. } => |src, _ctx| {
            Some(src.clone())
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

/// Find output PARAM from KERNEL AST.
fn find_kernel_output(ast: &Arc<UOp>) -> Option<Arc<UOp>> {
    for node in ast.toposort() {
        // Use store_buffer() helper to get buffer from STORE via its INDEX child
        if let Some(buffer) = node.store_buffer() {
            let output_buf = match buffer.op() {
                Op::Index { buffer: inner_buf, .. } => inner_buf.clone(),
                _ => buffer.clone(),
            };
            if matches!(output_buf.op(), Op::Param { device: None, .. }) {
                return Some(output_buf);
            }
        }
    }
    None
}

/// Create patterns for to_param transformation (normalizes buffers to codegen PARAMs).
pub fn to_param_patterns() -> TypedPatternMatcher<KernelContext> {
    crate::patterns! {
        @context KernelContext;
        // Buffer → codegen PARAM
        buf @ Buffer { size, unique: _ } => |buf, size, ctx| {
            let ptr_dtype = extract_base_dtype(buf.dtype()).ptr(Some(*size), AddrSpace::Global);
            let replacement = UOp::param(ctx.next_global(), *size, ptr_dtype, None);
            ctx.map_buffer(buf.clone(), replacement.clone());
            Some(replacement)
        },
        // Remove BIND: extract var and track it with its bound value
        Bind { var, value } => |var, value, ctx| {
            let bound_val = match value.op() {
                Op::Const(cv) => cv.0.try_int(),
                _ => None,
            };
            ctx.add_var(var.clone(), bound_val);
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

/// Create patterns for to_param transformation using LocalAddBufferContext.
///
/// Based on Tinygrad's to_define_global/debuf (rangeify.py:419-423).
/// Creates per-kernel codegen PARAMs with sequential slots and PtrDType.
pub fn local_to_param_patterns() -> TypedPatternMatcher<LocalAddBufferContext> {
    crate::patterns! {
        @context LocalAddBufferContext;
        // Buffer → codegen PARAM (like Tinygrad's debuf, rangeify.py:419-423)
        buf @ Buffer { size, unique: _ } => |buf, size, ctx| {
            let ptr_dtype = extract_base_dtype(buf.dtype()).ptr(Some(*size), AddrSpace::Global);
            let replacement = UOp::param(ctx.next_param_slot(), *size, ptr_dtype, None);
            if !ctx.has_buffer(buf) {
                ctx.map_buffer(buf.clone(), buf.clone());
            }
            Some(replacement)
        },
        // Pre-kernel Param (device: Some) → codegen PARAM (device: None)
        // Guard: skip codegen PARAMs (device: None) to prevent infinite loop
        buf @ Param { slot: _, size } if matches!(buf.op(), Op::Param { device: Some(_), .. }) => |buf, size, ctx| {
            let ptr_dtype = extract_base_dtype(buf.dtype()).ptr(Some(*size), AddrSpace::Global);
            let replacement = UOp::param(ctx.next_param_slot(), *size, ptr_dtype, None);
            if !ctx.has_buffer(buf) {
                ctx.map_buffer(buf.clone(), buf.clone());
            }
            Some(replacement)
        },
        // Remove BIND: extract var and track it with bound value (like Tinygrad's unbind_kernel)
        Bind { var, value } => |var, value, ctx| {
            let bound_val = match value.op() {
                Op::Const(cv) => cv.0.try_int(),
                _ => None,
            };
            ctx.add_var(var.clone(), bound_val);
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
            // Duplicate buffer mapping — Tinygrad asserts here.
            if ctx.has_buffer(&buf) {
                debug_assert!(false, "handle_after: duplicate buffer mapping for buf id={}", buf.id);
                tracing::warn!(buf_id = buf.id, "handle_after: duplicate buffer mapping, skipping");
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
    let opts = BufferizeOpts { device: None, addrspace: AddrSpace::Global, removable: true };
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
pub fn pm_add_loads() -> &'static TypedPatternMatcher<()> {
    crate::cached_patterns! {
        // Pattern 1: INDEX with non-Ptr dtype → LOAD(buffer, INDEX with Ptr dtype)
        // Guard prevents re-matching: after transformation, INDEX has Ptr dtype.
        // Also skip Image dtype - image access handled separately in codegen.
        // Based on Tinygrad's pm_add_loads (devectorizer.py:320-323):
        //   idx.replace(dtype=idx.src[0].dtype).load(dtype=idx.dtype.base)
        // idx.dtype.base for non-PtrDType returns self — preserving vectorization.
        idx @ Index { buffer, indices } if !matches!(idx.dtype(), DType::Ptr { .. } | DType::Image { .. }) => |idx, buffer, indices| {
            // Use INDEX's own dtype as LOAD result type. The expander has already set
            // the correct vectorization (e.g. Float32.vec(3) for UPCAST=3).
            let result_dtype = idx.dtype().clone();

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

            // CAST and BITCAST need special handling: Op::Cast/BitCast has its own dtype field
            // that must be updated to scalar, not just the UOp's result dtype.
            // The generic replace chain doesn't update Op::Cast::dtype.
            match uop.op() {
                Op::Cast { .. } => new_sources[0].cast(scalar_dtype.clone()),
                Op::BitCast { .. } => new_sources[0].bitcast(scalar_dtype.clone()),
                _ => uop.replace().dtype(scalar_dtype.clone()).src(new_sources).call(),
            }
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
pub fn pm_bool_devectorize() -> &'static TypedPatternMatcher<()> {
    crate::cached_patterns! {
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
pub fn pm_reduce_devectorize() -> &'static TypedPatternMatcher<()> {
    crate::cached_patterns! {
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
pub fn pm_fma_decomposition() -> &'static TypedPatternMatcher<()> {
    crate::cached_patterns! {
        // (a*b)+c or c+(a*b) → MulAcc(a,b,c) using commutative matching
        // Dtype equality guard is an early-out; try_mulacc also validates matching dtypes.
        Add[Mul(a, b), c] if a.dtype().is_float() && a.dtype() == b.dtype() && a.dtype() == c.dtype() => |a, b, c| {
            UOp::try_mulacc(a.clone(), b.clone(), c.clone()).ok()
        },
    }
}

// ============================================================================
// PM_LOAD_COLLAPSE - Collapse REDUCE with conditional loads
// ============================================================================
// Based on Tinygrad's pm_load_collapse (simplify.py)

/// Check if UOp has no RANGE in backward slice (loop-invariant).
fn no_range(u: &Arc<UOp>) -> bool {
    !u.any_in_subtree(|x| matches!(x.op(), Op::Range { .. }))
}

/// Check if UOp has no INDEX (load) in backward slice.
///
/// Used for index overflow protection pattern - we want to ensure
/// we don't do math on a loaded index since that can cause overflow.
fn no_load(u: &Arc<UOp>) -> bool {
    !u.any_in_subtree(|x| matches!(x.op(), Op::Index { .. }))
}

/// Check if a UOp represents a zero constant.
fn is_const_zero(u: &Arc<UOp>) -> bool {
    if let Op::Const(cv) = u.op() { cv.0.is_zero() } else { false }
}

/// Compute minimum of two UOps: min(a, b) = -max(-a, -b)
///
/// Tinygrad encodes min this way (not as WHERE) so that the MAX bounds
/// elimination rule `max(x, y) → x when x.vmin >= y.vmax` can simplify
/// boundary cases like `min(x, N)` when `x.vmax == N`.
fn uop_min(a: &Arc<UOp>, b: &Arc<UOp>) -> Option<Arc<UOp>> {
    let neg_a = a.neg();
    let neg_b = b.neg();
    let max_neg = neg_a.try_max(&neg_b).ok()?;
    Some(max_neg.neg())
}

/// Try to collapse a REDUCE with conditional/gated patterns.
///
/// Core gated collapse logic shared by NE (Pattern 3) and EQ (Pattern 3b).
///
/// Substitutes range with `idx.cast(r.dtype).valid(in_bounds)` in the expression,
/// producing `where(in_bounds, expr[r:=valid_idx], 0)`.
fn gated_collapse_core(idx: &Arc<UOp>, range: &Arc<UOp>, end: &Arc<UOp>, expr: &Arc<UOp>) -> Option<Arc<UOp>> {
    let idx_casted = idx.cast(range.dtype());
    let zero = UOp::index_const(0);
    let in_bounds = idx_casted.try_cmpge(&zero).ok()?.try_and_op(&idx_casted.try_cmplt(end).ok()?).ok()?;
    let valid_idx = idx_casted.valid(in_bounds.clone());
    #[allow(clippy::mutable_key_type)]
    let subs: std::collections::HashMap<UOpKey, Arc<UOp>> = [(UOpKey(range.clone()), valid_idx)].into_iter().collect();
    let substituted = expr.substitute(&subs);
    let zero_like = UOp::const_(expr.dtype(), ConstValue::zero(expr.dtype().base()));
    UOp::try_where(in_bounds, substituted, zero_like).ok()
}

/// This implements reduction collapse patterns from Tinygrad (simplify.py:93-108):
/// 1. Sum of `where(r < cut, 0, val)` → `clamp(end-cut, 0, end) * val`
/// 2. Sum of `where(r < cut, val, 0)` → `clamp(cut, 0, end) * val`
/// 3. Sum of `where(idx != r, 0, expr)` → `where(in_bounds, expr[r:=idx], 0)`
/// 4. Sum of `where((r >= lower) & (r < upper), val, 0)` → two-sided bounds
fn try_reduce_collapse(
    _reduce: &Arc<UOp>,
    src: &Arc<UOp>,
    ranges: &SmallVec<[Arc<UOp>; 4]>,
    reduce_op: ReduceOp,
) -> Option<Arc<UOp>> {
    // Only handle Add for now (like Tinygrad)
    if reduce_op != ReduceOp::Add {
        return None;
    }

    // Must have exactly one range
    if ranges.len() != 1 {
        return None;
    }

    let range = &ranges[0];
    let Op::Range { end, .. } = range.op() else {
        return None;
    };

    // Pattern: WHERE(cond, true_val, false_val)
    let Op::Ternary(morok_ir::TernaryOp::Where, cond, true_val, false_val) = src.op() else {
        return None;
    };

    // Pattern 1: where(r < cut, 0, val) → (end - cut).max(0).min(end) * val
    if let Op::Binary(BinaryOp::Lt, lt_lhs, cut) = cond.op()
        && Arc::ptr_eq(lt_lhs, range)
        && is_const_zero(true_val)
        && no_range(false_val)
    {
        // count = (end - cut).max(0).min(end)  -- symbolic UOp arithmetic
        let zero = UOp::index_const(0);
        let diff = end.try_sub(cut).ok()?;
        let non_negative = diff.try_max(&zero).ok()?;
        let count = uop_min(&non_negative, end)?;
        let count_casted = count.cast(false_val.dtype());
        return count_casted.try_mul(false_val).ok();
    }

    // Pattern 2: where(r < cut, val, 0) → cut.max(0).min(end) * val
    if let Op::Binary(BinaryOp::Lt, lt_lhs, cut) = cond.op()
        && Arc::ptr_eq(lt_lhs, range)
        && is_const_zero(false_val)
        && no_range(true_val)
    {
        // count = cut.max(0).min(end)  -- symbolic UOp arithmetic
        let zero = UOp::index_const(0);
        let clamped = cut.try_max(&zero).ok()?;
        let count = uop_min(&clamped, end)?;
        let count_casted = count.cast(true_val.dtype());
        return count_casted.try_mul(true_val).ok();
    }

    // Pattern 2b: where(r >= lower, val, 0) → (end - lower).max(0).min(end) * val
    // Handles Ge directly (bound from below), equivalent to Pattern 1 with inverted condition.
    if let Some(lower) = extract_ge_lower_bound(cond, range)
        && is_const_zero(false_val)
        && no_range(true_val)
        && no_range(&lower)
    {
        let zero = UOp::index_const(0);
        let diff = end.try_sub(&lower).ok()?;
        let non_negative = diff.try_max(&zero).ok()?;
        let count = uop_min(&non_negative, end)?;
        let count_casted = count.cast(true_val.dtype());
        return count_casted.try_mul(true_val).ok();
    }

    // Pattern 2c: where(r >= lower, 0, val) → lower.max(0).min(end) * val
    // Inverted Ge: value when condition is FALSE (r < lower).
    if let Some(lower) = extract_ge_lower_bound(cond, range)
        && is_const_zero(true_val)
        && no_range(false_val)
        && no_range(&lower)
    {
        let zero = UOp::index_const(0);
        let clamped = lower.try_max(&zero).ok()?;
        let count = uop_min(&clamped, end)?;
        let count_casted = count.cast(false_val.dtype());
        return count_casted.try_mul(false_val).ok();
    }

    // Pattern 3: where(idx != r, 0, expr) — NE gated collapse
    // Pattern 3b: where(idx == r, expr, 0) — EQ gated collapse (Morok-specific)
    //
    // Both collapse to: where(in_bounds, expr[r:=idx.valid(v)], 0)
    // NE: idx != r with zero in true_val, expression in false_val
    // EQ: idx == r with expression in true_val, zero in false_val
    // Also handles .or_casted(): unwraps CAST around the range operand.
    {
        let (idx, cmp_range, expr) = match cond.op() {
            // NE: where(idx != range_side, 0, expr) — Tinygrad pattern
            Op::Binary(BinaryOp::Ne, idx, ne_range) if is_const_zero(true_val) && no_range(idx) => {
                Some((idx, ne_range, false_val))
            }
            // EQ: where(idx == range_side, expr, 0) — Morok-specific
            Op::Binary(BinaryOp::Eq, lhs, rhs) if is_const_zero(false_val) => {
                if no_range(lhs) {
                    Some((lhs, rhs, true_val))
                } else if no_range(rhs) {
                    Some((rhs, lhs, true_val))
                } else {
                    None
                }
            }
            _ => None,
        }?;

        let actual_range = if let Op::Cast { src, .. } = cmp_range.op() { src } else { cmp_range };
        if Arc::ptr_eq(actual_range, range) {
            return gated_collapse_core(idx, range, end, expr);
        }
    }

    // Pattern 4: Two-sided bounds
    // where((r >= lower) & (r < upper), val, 0) → (upper.min(end) - lower.max(0)).max(0).min(end) * val
    // Handles two AST representations:
    //   A: ((r < lower).logical_not() & (r < upper)) - NOT(LT) form
    //   B: ((r >= lower) & (r < upper)) - direct GE form
    if let Op::Binary(BinaryOp::And, lhs_cond, rhs_cond) = cond.op()
        && is_const_zero(false_val)
        && no_range(true_val)
    {
        // Try to extract (r >= lower) from lhs_cond - either NOT(LT) or GE form
        let lower_bound = extract_ge_lower_bound(lhs_cond, range).or_else(|| extract_ge_lower_bound(rhs_cond, range));

        // Try to extract (r < upper) from rhs_cond or lhs_cond
        let upper_bound = extract_lt_upper_bound(rhs_cond, range).or_else(|| extract_lt_upper_bound(lhs_cond, range));

        if let (Some(lower), Some(upper)) = (lower_bound, upper_bound)
            && no_range(&lower)
            && no_range(&upper)
        {
            // (upper.min(end) - lower.max(0)).max(0).min(end) * val
            let zero = UOp::index_const(0);
            let clamped_upper = uop_min(&upper, end)?;
            let clamped_lower = lower.try_max(&zero).ok()?;
            let diff = clamped_upper.try_sub(&clamped_lower).ok()?;
            let non_negative = diff.try_max(&zero).ok()?;
            let count = uop_min(&non_negative, end)?;
            let count_casted = count.cast(true_val.dtype());
            return count_casted.try_mul(true_val).ok();
        }
    }

    None
}

/// Extract lower bound from (r >= lower) condition.
/// Handles both NOT(r < lower) and (r >= lower) forms.
fn extract_ge_lower_bound(cond: &Arc<UOp>, range: &Arc<UOp>) -> Option<Arc<UOp>> {
    // Form A: NOT(r < lower)
    if let Op::Unary(UnaryOp::Not, lt_cond) = cond.op()
        && let Op::Binary(BinaryOp::Lt, lt_lhs, lower) = lt_cond.op()
        && Arc::ptr_eq(lt_lhs, range)
    {
        return Some(lower.clone());
    }
    // Form B: r >= lower (represented as NOT(r < lower) or Ge(r, lower))
    if let Op::Binary(BinaryOp::Ge, ge_lhs, lower) = cond.op()
        && Arc::ptr_eq(ge_lhs, range)
    {
        return Some(lower.clone());
    }
    None
}

/// Extract upper bound from (r < upper) condition.
fn extract_lt_upper_bound(cond: &Arc<UOp>, range: &Arc<UOp>) -> Option<Arc<UOp>> {
    if let Op::Binary(BinaryOp::Lt, lt_lhs, upper) = cond.op()
        && Arc::ptr_eq(lt_lhs, range)
    {
        return Some(upper.clone());
    }
    None
}

/// Try to collapse a REDUCE when DEFINE_VAR can be factored out.
///
/// Pattern: (DEFINE_VAR & y).where(c, 0).reduce(ADD) → y.where(c, 0).reduce(ADD) * DEFINE_VAR.cast(c.dtype)
fn try_define_var_factor(src: &Arc<UOp>, ranges: &SmallVec<[Arc<UOp>; 4]>) -> Option<Arc<UOp>> {
    // Pattern: WHERE((DEFINE_VAR & y), c, 0)
    let Op::Ternary(morok_ir::TernaryOp::Where, cond, true_val, false_val) = src.op() else {
        return None;
    };
    if !is_const_zero(false_val) {
        return None;
    }

    // Match AND(DEFINE_VAR, y) or AND(y, DEFINE_VAR)
    let Op::Binary(BinaryOp::And, and_lhs, and_rhs) = cond.op() else {
        return None;
    };

    let (define_var, other) = if matches!(and_lhs.op(), Op::DefineVar { .. }) {
        (and_lhs.clone(), and_rhs.clone())
    } else if matches!(and_rhs.op(), Op::DefineVar { .. }) {
        (and_rhs.clone(), and_lhs.clone())
    } else {
        return None;
    };

    // Build: other.where(c, 0).reduce(ADD) * DEFINE_VAR.cast(c.dtype)
    let inner_where = UOp::try_where(other, true_val.clone(), false_val.clone()).ok()?;
    let inner_reduce = inner_where.reduce(ranges.clone(), ReduceOp::Add);
    let casted_var = define_var.cast(true_val.dtype());
    inner_reduce.try_mul(&casted_var).ok()
}

/// Arithmetic lifting for comparisons.
///
/// Lifts operations out of Lt comparisons when they don't depend on ranges:
/// - (x + y) < c → x < (c - y) when y, c are range-free
/// - (x * y) < c → x < ceil(c/y) when y > 0, y, c range-free
///
/// Also handles `.or_casted()` variants where lhs is wrapped in a CAST:
/// - Cast(x + y) < c → x < (c.cast(inner_dtype) - y)
/// - Cast(x * y) < c → x < ceil(c.cast(inner_dtype)/y)
fn try_lift_arithmetic_from_lt(cond: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Binary(BinaryOp::Lt, lhs, rhs) = cond.op() else {
        return None;
    };

    // Both rhs must be range-free for lifting
    if !no_range(rhs) {
        return None;
    }

    // Unwrap optional CAST to get the inner expression (or_casted pattern).
    // When CAST is present, we need to cast the rhs constant to the inner dtype.
    let (inner_lhs, effective_rhs) = if let Op::Cast { src, .. } = lhs.op() {
        let inner_dtype = src.dtype();
        let casted_rhs = rhs.cast(inner_dtype);
        (src.as_ref(), casted_rhs)
    } else {
        (lhs.as_ref(), rhs.clone())
    };

    // Pattern: (x + y) < c → x < (c - y)
    if let Op::Binary(BinaryOp::Add, x, y) = inner_lhs.op()
        && no_range(y)
    {
        let new_rhs = effective_rhs.try_sub(y).ok()?;
        return x.try_cmplt(&new_rhs).ok();
    }

    // Pattern: (x * y) < c → x < ceil(c/y) when y > 0
    if let Op::Binary(BinaryOp::Mul, x, y) = inner_lhs.op()
        && no_range(y)
    {
        // Check y > 0 via vmin
        if let ConstValue::Int(ymin) = y.vmin()
            && *ymin > 0
        {
            // ceil(c/y) = (c + y - 1) / y
            let one = UOp::index_const(1);
            let c_plus_y = effective_rhs.try_add(y).ok()?;
            let c_plus_y_minus_1 = c_plus_y.try_sub(&one).ok()?;
            let new_rhs = c_plus_y_minus_1.try_div(y).ok()?;
            return x.try_cmplt(&new_rhs).ok();
        }
    }

    None
}

/// Arithmetic lifting for EQ comparisons (Morok-specific).
///
/// Isolates range-containing operands from arithmetic in EQ conditions:
/// - (x + y) == c → x == (c - y) or y == (c - x)
/// - (x - y) == c → x == (c + y) or y == (x - c)
/// - Cast(x ± y) == c → same with c cast to inner dtype
fn try_lift_arithmetic_from_eq(cond: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Binary(BinaryOp::Eq, raw_lhs, raw_rhs) = cond.op() else { return None };

    // Normalize: range-containing side on lhs, range-free on rhs.
    // The pattern `Eq[_, c] if no_range(c)` matches commutatively, but
    // cond.op() returns operands in storage order which may differ.
    let (lhs, rhs) = if no_range(raw_rhs) {
        (raw_lhs, raw_rhs)
    } else if no_range(raw_lhs) {
        (raw_rhs, raw_lhs)
    } else {
        return None;
    };

    // Unwrap optional CAST, adjusting rhs to inner dtype
    let (inner_lhs, effective_rhs) = if let Op::Cast { src, .. } = lhs.op() {
        (src.as_ref(), rhs.cast(src.dtype()))
    } else {
        (lhs.as_ref(), rhs.clone())
    };

    match inner_lhs.op() {
        Op::Binary(BinaryOp::Add, x, y) if no_range(y) => x.try_cmpeq(&effective_rhs.try_sub(y).ok()?).ok(),
        Op::Binary(BinaryOp::Add, x, y) if no_range(x) => y.try_cmpeq(&effective_rhs.try_sub(x).ok()?).ok(),
        Op::Binary(BinaryOp::Sub, x, y) if no_range(y) => x.try_cmpeq(&effective_rhs.try_add(y).ok()?).ok(),
        Op::Binary(BinaryOp::Sub, x, y) if no_range(x) => y.try_cmpeq(&x.try_sub(&effective_rhs).ok()?).ok(),
        _ => None,
    }
}

/// Arithmetic lifting for Ge comparisons.
///
/// Lifts operations out of Ge comparisons when they don't depend on ranges:
/// - (x + y) >= c → x >= (c - y) when y, c are range-free
fn try_lift_arithmetic_from_ge(cond: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Binary(BinaryOp::Ge, lhs, rhs) = cond.op() else {
        return None;
    };

    if !no_range(rhs) {
        return None;
    }

    // (x + y) >= c → x >= (c - y) when y is range-free
    if let Op::Binary(BinaryOp::Add, x, y) = lhs.op() {
        if no_range(y) {
            return x.try_cmpge(&rhs.try_sub(y).ok()?).ok();
        }
        if no_range(x) {
            return y.try_cmpge(&rhs.try_sub(x).ok()?).ok();
        }
    }

    None
}

/// Pattern matcher for load collapse optimizations.
///
/// Based on Tinygrad's pm_load_collapse (simplify.py).
/// Collapses REDUCE operations with gated/conditional loads.
///
/// Key optimizations:
/// 1. Bounded sum reduction: `sum(1 for i in range(n) if i >= k)` → `n - k`
/// 2. Two-sided bounds: `sum(1 for i in range(n) if lower <= i < upper)` → clamped count
/// 3. Gated load collapse: `sum(where(idx == r, val, 0))` → direct indexed load
/// 4. Arithmetic lifting: push comparisons through arithmetic operations
/// 5. DEFINE_VAR factoring: `(dv & y).where(c, 0).reduce(ADD)` → `y.where(c,0).reduce(ADD) * dv`
/// 6. MUL casted bool: `x * gate:bool.cast()` → `gate.where(x, 0)`
/// 7. NE lifting: `(x + y) != c` → `x != (c - y)`
/// 8. Index overflow protection: `(x:index + y) < c` → `x < (c - y)` when x has loads
pub fn pm_load_collapse() -> &'static TypedPatternMatcher<()> {
    crate::cached_patterns! {
        // Match REDUCE(ADD) with a single range → full reduce_load_collapse algorithm.
        //
        // Tinygrad: pm_load_collapse matches only single-range REDUCE(ADD)
        // and goes straight to reduce_load_collapse (the full algorithm with
        // gated toposort + DEFINE_VAR substitution). All arithmetic lifting,
        // NE lifting, .or_casted() patterns live inside the inner matcher
        // (build_reduce_load_collapse_matcher), not at this level.
        _reduce @ Reduce { src, ranges, reduce_op }
            if ranges.len() == 1 && *reduce_op == ReduceOp::Add
            => |src, ranges| {
                super::transforms::reduce_load_collapse(src, ranges)
            },

        // Index overflow undo rule: (x:index + y) < c → x < (c - y)
        // Only when x has loads but y, c don't — prevents overflow on loaded indices.
        // This undoes the arithmetic lifting that pm_reduce_load_collapse may have
        // applied when the lifted form risks integer overflow on loaded values.
        Lt(Add(x, y), c)
            if x.dtype() == DType::Index && !no_load(x) && no_load(y) && no_load(c)
            => |x, y, c| {
                let new_c = c.try_sub(y).ok()?;
                x.try_cmplt(&new_c).ok()
            },
    }
}

// ============================================================================
// PM_REDUCE_COLLAPSE - Inner patterns for reduce_collapse loop
// ============================================================================
// Based on Tinygrad's pm_reduce_collapse (simplify.py:87-111)
// Used inside reduce_collapse's per-range iteration to algebraically
// eliminate the synthetic REDUCE node.

/// Inner pattern matcher used inside `reduce_collapse` per-range iteration.
///
/// Combines reduce-specific algebraic patterns with full symbolic simplification.
/// Does NOT include a recursive `reduce_collapse` call (would infinite-loop).
pub fn build_reduce_collapse_matcher() -> &'static TypedPatternMatcher<()> {
    static CACHED: std::sync::LazyLock<TypedPatternMatcher<()>> =
        // Tinygrad simplify.py:112: pm_reduce_collapse uses full `symbolic`
        std::sync::LazyLock::new(|| reduce_collapse_inner_patterns() + crate::symbolic::symbolic());
    &CACHED
}

/// Extended pattern matcher for `reduce_load_collapse`.
///
/// Combines the basic `reduce_collapse` patterns with NE lifting.
/// NE lifting is only needed in the per-kernel `reduce_load_collapse` path,
/// matching Tinygrad where NE lifting is only in `pm_reduce_load_collapse`.
///
/// Does NOT include the REDUCE→reduce_load_collapse call to avoid infinite recursion.
pub fn build_reduce_load_collapse_matcher() -> &'static TypedPatternMatcher<()> {
    static CACHED: std::sync::LazyLock<TypedPatternMatcher<()>> =
        std::sync::LazyLock::new(|| build_reduce_collapse_matcher() + ne_lifting_patterns());
    &CACHED
}

/// NE lifting patterns for the extended `reduce_load_collapse` path.
///
/// Matches Tinygrad's `pm_reduce_load_collapse` additions over `pm_reduce_collapse`:
/// NE lifting is only needed here (not in the basic mega-pass matcher).
///
/// Note: index overflow undo lives in `pm_load_collapse()` (the outer matcher),
/// not here — inside reduce_collapse, external inputs are DEFINE_VARs with no loads,
/// so `!no_load(x)` never matches.
fn ne_lifting_patterns() -> TypedPatternMatcher<()> {
    crate::patterns! {
        // NE lifting: (x + y) != c → x != (c - y) when no_range(y, c)
        Ne(Add(x, y), c) if no_range(y) && no_range(c) => |x, y, c| {
            let new_c = c.try_sub(y).ok()?;
            x.try_cmpne(&new_c).ok()
        },

        // .or_casted() NE: Cast(x + y) != c → x != (c.cast(inner_dtype) - y)
        Ne(Cast { src: inner, .. }, c) if no_range(c) => |inner, c| {
            let Op::Binary(BinaryOp::Add, x, y) = inner.op() else { return None };
            if !no_range(y) { return None; }
            let casted_c = c.cast(inner.dtype());
            let new_c = casted_c.try_sub(y).ok()?;
            x.try_cmpne(&new_c).ok()
        },
    }
}

/// Reduce-specific algebraic patterns for use inside `reduce_collapse`.
///
/// Based on Tinygrad's pm_reduce_collapse (simplify.py:87-111):
/// 1. reduce_unparented: remove ranges not used by src
/// 2. Lt lifting: (x+y) < c → x < (c-y), (x*y) < c → x < ceil(c/y), including .or_casted()
/// 3. Ge lifting: (x+y) >= c → x >= (c-y)
/// 4. Distributive: (x+y).reduce(ADD) → x.reduce(ADD) + y.reduce(ADD)
/// 5. Bound-from-below/above/two-sided/gated collapse on REDUCE(ADD)
/// 6. DEFINE_VAR factoring: (dv & y).where(c,0).reduce(ADD)
/// 7. MUL casted bool: x * gate:bool.cast() → gate.where(x, 0)
/// 8. EQ lifting: (x+y)==c → x==(c-y), Morok-specific for gather's EQ pattern
fn reduce_collapse_inner_patterns() -> TypedPatternMatcher<()> {
    // Start with reduce_unparented (shared with pm_reduce_simplify)
    pm_reduce_unparented().with_context()
    // Lt/Ge arithmetic lifting: push range-free operands to rhs of comparison.
    // Handles direct and .or_casted() (CAST-wrapped) Add/Mul forms.
    + crate::patterns! {
        cond @ Lt(_, rhs) if no_range(rhs) => |cond| try_lift_arithmetic_from_lt(cond),
        cond @ Ge(_, rhs) if no_range(rhs) => |cond| try_lift_arithmetic_from_ge(cond),

        // Distributive: (x+y).reduce(ADD) → x.reduce(ADD) + y.reduce(ADD)
        Reduce { src, ranges, reduce_op } if *reduce_op == ReduceOp::Add => |src, ranges| {
            let Op::Binary(BinaryOp::Add, x, y) = src.op() else { return None };
            let x_reduced = x.reduce(ranges.clone(), ReduceOp::Add);
            let y_reduced = y.reduce(ranges.clone(), ReduceOp::Add);
            x_reduced.try_add(&y_reduced).ok()
        },

        // Bound patterns + DEFINE_VAR factoring on REDUCE(ADD) with single range
        // These match the synthetic REDUCE created by reduce_collapse's per-range iteration.
        // Patterns: bound-from-below, bound-from-above, two-sided, gated NE/EQ collapse.
        reduce @ Reduce { src, ranges, reduce_op }
            if !ranges.is_empty() && *reduce_op == ReduceOp::Add
            => |reduce, src, ranges| {
                try_reduce_collapse(reduce, src, ranges, ReduceOp::Add)
                    .or_else(|| try_define_var_factor(src, ranges))
            },

        // MUL casted bool: x * gate:bool.cast() → gate.where(x, 0)
        Mul[x, Cast { src: gate, .. }] if gate.dtype() == DType::Bool => |x, gate| {
            let zero = UOp::const_(x.dtype(), ConstValue::zero(x.dtype().base()));
            UOp::try_where(gate.clone(), x.clone(), zero).ok()
        },

        // EQ lifting: isolate range-containing operands from arithmetic in EQ conditions.
        // Morok-specific — needed because Morok uses EQ directly from gather,
        // unlike Tinygrad which decomposes eq() as ne().not() and only needs NE lifting.
        // Must be in basic matcher since mega-pass buffer_folding inlines arange constants,
        // enabling EQ lifting to work. Per-kernel pm_load_collapse runs before buffer_folding
        // so it can't see through LOADs.
        cond @ Eq[_, c] if no_range(c) => |cond| try_lift_arithmetic_from_eq(cond),
    }
}

// ============================================================================
// LATE DECOMPOSITION PATTERNS (get_late_rewrite_patterns)
// ============================================================================
// Based on Tinygrad's decompositions.py:321-367

/// MOD → AND optimization for power-of-two modulus.
///
/// x % 2^n → x & (2^n - 1)
///
/// This is a common optimization that converts expensive modulo operations
/// into cheap bitwise AND when the divisor is a power of two.
/// Only applies to integer types.
pub fn pm_mod_to_and() -> &'static TypedPatternMatcher<()> {
    use morok_ir::types::ConstValue;
    crate::cached_patterns! {
        // x % c where c is power of two → x & (c - 1)
        Mod(x, _c @const(c_val)) => |x, c_val| {
            // Only apply to integer types
            if !x.dtype().is_int() { return None; }

            let n = match c_val {
                ConstValue::Int(v) if v > 0 && (v as u64).is_power_of_two() => v,
                ConstValue::UInt(v) if v > 0 && v.is_power_of_two() => v as i64,
                _ => return None,
            };
            // x % n → x & (n - 1)
            let mask = UOp::const_(x.dtype(), ConstValue::Int(n - 1));
            x.try_and_op(&mask).ok()
        },
    }
}

/// Multiply → Shift optimization for power-of-two multiplier.
///
/// x * 2^n → x << n
///
/// Converts multiplication by power-of-two into left shift.
/// Only applies to integer types.
pub fn pm_mul_to_shl() -> &'static TypedPatternMatcher<()> {
    use morok_ir::types::ConstValue;
    crate::cached_patterns! {
        // x * c where c is power of two → x << log2(c)
        // Note: Only applies to integer types, but we check inside the closure
        Mul[x, _c @const(c_val)] => |x, c_val| {
            // Only apply to integer types
            if !x.dtype().is_int() { return None; }

            let (n, shift) = match c_val {
                ConstValue::Int(v) if v > 0 && (v as u64).is_power_of_two() => (v as u64, (v as u64).trailing_zeros()),
                ConstValue::UInt(v) if v > 0 && v.is_power_of_two() => (v, v.trailing_zeros()),
                _ => return None,
            };
            if n == 1 { return Some(x.clone()); } // x * 1 → x (handled elsewhere but be safe)
            let shift_amount = UOp::const_(x.dtype(), ConstValue::Int(shift as i64));
            x.try_shl_op(&shift_amount).ok()
        },
    }
}

/// Negation decompositions (Tinygrad decompositions.py:458-460):
/// - x * -1 → NEG(x)
/// - x + NEG(y) → SUB(x, y)
pub fn pm_neg_from_mul() -> &'static TypedPatternMatcher<()> {
    crate::cached_patterns! {
        // x * -1 → NEG(x)
        // Uses raw UOp::new to avoid infinite loop since .neg() now produces MUL(x, -1).
        Mul[x, _c @const(c_val)] if c_val.is_neg_one() => |x| {
            let dtype = x.dtype();
            Some(UOp::new(Op::Unary(UnaryOp::Neg, x.clone()), dtype))
        },
        // x + NEG(y) → SUB(x, y) (Tinygrad decompositions.py:460)
        Add[x, Neg(y)] ~> x.sub(y),
    }
}

/// Threefry2x32 PRNG decomposition (Tinygrad decompositions.py:302-316).
///
/// No real hardware supports THREEFRY natively. Decomposes to uint32 arithmetic:
/// split 64-bit x/key into halves, apply 5 rounds of add-rotate-xor mixing,
/// recombine to uint64.
pub fn pm_threefry_decomp() -> &'static TypedPatternMatcher<()> {
    crate::cached_patterns! {
        Threefry(x, key) if x.dtype() == DType::UInt64 => |x, key| {
            Some(threefry2x32(x, key))
        },
    }
}

/// Threefry2x32 mixing algorithm (Random123 library).
fn threefry2x32(x: &Arc<UOp>, key: &Arc<UOp>) -> Arc<UOp> {
    let u32_dt = DType::Scalar(morok_dtype::ScalarDType::UInt32);
    let u64_dt = DType::Scalar(morok_dtype::ScalarDType::UInt64);
    let mask32 = UOp::const_(u64_dt.clone(), ConstValue::UInt(0xFFFFFFFF));
    let pow32 = UOp::const_(u64_dt.clone(), ConstValue::UInt(1u64 << 32));

    // Split x and key from uint64 to two uint32
    let x0 = x.and_(&mask32).cast(u32_dt.clone());
    let x1 = x.idiv(&pow32).and_(&mask32).cast(u32_dt.clone());
    let key0 = key.and_(&mask32).cast(u32_dt.clone());
    let key1 = key.idiv(&pow32).and_(&mask32).cast(u32_dt.clone());

    // Key schedule: ks = [key1, key0 ^ key1 ^ 0x1BD11BDA, key0]
    let skein_const = UOp::const_(u32_dt.clone(), ConstValue::UInt(0x1BD11BDA));
    let ks = [key1.clone(), key0.xor(&key1).xor(&skein_const), key0.clone()];

    let rotations: [[u32; 4]; 2] = [[13, 15, 26, 6], [17, 29, 16, 24]];

    // Initialize: xr = [x0 + ks[2], x1 + ks[0]]
    let mut xr0 = x0.add(&ks[2]);
    let mut xr1 = x1.add(&ks[0]);

    // 5 rounds of mixing
    for i in 0..5u32 {
        for &r in &rotations[i as usize % 2] {
            let new_x0 = xr0.add(&xr1);
            // Rotation: (xr1 * 2^r) + (xr1 // 2^(32-r))  (barrel rotate via mul+div)
            let rot_left = xr1.mul(&UOp::const_(u32_dt.clone(), ConstValue::UInt(1u64 << r)));
            let rot_right = xr1.idiv(&UOp::const_(u32_dt.clone(), ConstValue::UInt(1u64 << (32 - r))));
            let rotated = rot_left.add(&rot_right);
            xr1 = new_x0.xor(&rotated);
            xr0 = new_x0;
        }
        // Key injection
        xr0 = xr0.add(&ks[i as usize % 3]);
        let round_const = UOp::const_(u32_dt.clone(), ConstValue::UInt((i + 1) as u64));
        xr1 = xr1.add(&ks[(i as usize + 1) % 3]).add(&round_const);
    }

    // Recombine: xr1.cast(u64) * 2^32 | xr0.cast(u64)
    xr1.cast(u64_dt.clone()).mul(&pow32).or_(&xr0.cast(u64_dt))
}

/// DeMorgan's law: NOT(x) & NOT(y) → NOT(x | y) (Tinygrad decompositions.py:446-447).
///
/// Reduces one NOT instruction by factoring out negation.
pub fn pm_demorgan() -> &'static TypedPatternMatcher<()> {
    crate::cached_patterns! {
        And[Not(x), Not(y)] if x.dtype().is_bool() ~> x.or_(y).not(),
    }
}

/// SHL+ADD → MULACC fusion (Tinygrad decompositions.py:475).
///
/// After `pm_mul_to_shl` converts `x * 2^n → x << n`, expressions like
/// `(x << n) + c` can fuse to MULACC(x, 2^n, c) for backends with FMA.
pub fn pm_shl_add_to_mulacc() -> &'static TypedPatternMatcher<()> {
    crate::cached_patterns! {
        Add[Shl(x, _n @const(nv)), c] => |x, nv, c| {
            let ConstValue::Int(v) = nv else { return None };
            if !(0..64).contains(&v) { return None; }
            let multiplier = UOp::const_(x.dtype(), ConstValue::Int(1i64 << v));
            UOp::try_mulacc(x.clone(), multiplier, c.clone()).ok()
        },
    }
}

/// Divide → Shift optimization for power-of-two divisor.
///
/// Tinygrad: decompositions.py:340-344
///
/// For unsigned integers: x // 2^n → x >> n
/// For signed integers: (x + (x<0).where(n-1, 0)) >> n
///   (handles rounding towards zero for negative dividends)
///
/// Shifts are typically 2-5x faster than divisions on modern CPUs and GPUs.
pub fn pm_div_to_shr() -> &'static TypedPatternMatcher<()> {
    use morok_ir::types::ConstValue;
    use morok_ir::uop::cached_property::CachedProperty;
    use morok_ir::uop::properties::VminVmaxProperty;

    crate::cached_patterns! {
        // x // c where c is power of two → x >> log2(c)
        Idiv(x, _c @const(c_val)) => |x, c_val| {
            // Only apply to integer types
            if !x.dtype().is_int() { return None; }

            let n = match c_val {
                ConstValue::Int(v) if v > 0 && (v as u64).is_power_of_two() => v,
                ConstValue::UInt(v) if v > 0 && v.is_power_of_two() => v as i64,
                _ => return None,
            };

            // Skip trivial case: x // 1 → x (handled elsewhere)
            if n == 1 { return None; }

            let shift = (n as u64).trailing_zeros() as i64;
            let shift_const = UOp::const_(x.dtype(), ConstValue::Int(shift));

            // Check if x is always non-negative via vmin/vmax analysis
            let (vmin, _) = VminVmaxProperty::get(x);
            let is_non_negative = match vmin {
                ConstValue::Int(v) => *v >= 0,
                ConstValue::UInt(_) => true, // unsigned always non-negative
                _ => false,
            };

            if is_non_negative || x.dtype().is_unsigned() {
                // Unsigned case: x // 2^n → x >> n
                x.try_shr_op(&shift_const).ok()
            } else {
                // Signed case with potentially negative dividend:
                // (x + (x < 0).where(n - 1, 0)) >> n
                // This bias corrects for rounding towards zero
                let zero = UOp::const_(x.dtype(), ConstValue::Int(0));
                let bias = UOp::const_(x.dtype(), ConstValue::Int(n - 1));
                let x_neg = x.try_cmplt(&zero).ok()?;
                let adjustment = UOp::try_where(x_neg, bias, zero).ok()?;
                let adjusted = x.try_add(&adjustment).ok()?;
                adjusted.try_shr_op(&shift_const).ok()
            }
        },
    }
}

/// MAX decomposition: MAX(a, b) → (a < b).where(b, a)
///
/// For backends that don't have native MAX support, decompose into
/// comparison and conditional select.
pub fn pm_max_decomposition() -> &'static TypedPatternMatcher<()> {
    crate::cached_patterns! {
        // MAX(a, b) → (a < b).where(b, a)
        Max(a, b) => |a, b| {
            let cond = a.try_cmplt(b).ok()?;
            UOp::try_where(cond, b.clone(), a.clone()).ok()
        },
    }
}

/// SQRT decomposition: SQRT(x) → POW(x, 0.5)
///
/// For backends that don't have native SQRT support, decompose into
/// power operation with exponent 0.5.
pub fn pm_sqrt_decomposition() -> &'static TypedPatternMatcher<()> {
    crate::cached_patterns! {
        // SQRT(x) → POW(x, 0.5)
        Sqrt(x) if x.dtype().is_float() => |x| {
            let half = UOp::const_(x.dtype(), morok_ir::types::ConstValue::Float(0.5));
            x.try_pow(&half).ok()
        },
    }
}

/// ERF decomposition using Abramowitz & Stegun 7.1.26 polynomial approximation.
///
/// Tinygrad decomposes erf at the tensor level (elementwise.py:783-785) so it never
/// reaches the renderer. We decompose it here because Morok keeps Erf as a UOp.
/// `@llvm.erf` is a libcall intrinsic (not a native hardware op like sqrt/fabs),
/// so it requires libm linkage which the LLVM JIT doesn't provide.
///
/// erf(x) = sign(x) * (1 - t * P(t) * exp(-x²))
/// where t = 1 / (1 + 0.3275911 * |x|)
///       P(t) = polyN(t, [1.061405429, -1.453152027, 1.421413741, -0.284496736, 0.254829592])
pub fn pm_erf_decomposition() -> &'static TypedPatternMatcher<()> {
    crate::cached_patterns! {
        Erf(x) if x.dtype().is_float() => |x| {
            let dt = x.dtype();
            let f = |v: f64| UOp::const_(dt.clone(), ConstValue::Float(v));

            let abs_x = x.abs();
            let t = f(1.0).try_div(&f(1.0).try_add(&f(0.3275911).try_mul(&abs_x).ok()?).ok()?).ok()?;

            // Horner's method: ((((a4*t + a3)*t + a2)*t + a1)*t + a0)
            let poly = f(1.061405429);
            let poly = poly.try_mul(&t).ok()?.try_add(&f(-1.453152027)).ok()?;
            let poly = poly.try_mul(&t).ok()?.try_add(&f(1.421413741)).ok()?;
            let poly = poly.try_mul(&t).ok()?.try_add(&f(-0.284496736)).ok()?;
            let poly = poly.try_mul(&t).ok()?.try_add(&f(0.254829592)).ok()?;

            // exp(-x²)
            let exp_val = x.square().neg().try_exp().ok()?;

            // sign(x) * (1 - t * poly * exp(-x²))
            let inner = f(1.0).try_sub(&t.try_mul(&poly).ok()?.try_mul(&exp_val).ok()?).ok()?;
            x.sign().try_mul(&inner).ok()
        },
    }
}

/// FDIV → MUL reciprocal optimization for floating-point division by constant.
///
/// Tinygrad: decompositions.py:364-366
///
/// x / c → x * (1/c) for float constants
///
/// Multiplication is typically 2-3x faster than division on modern CPUs and GPUs.
/// Guards against divide by zero (leaves as FDIV to preserve IEEE 754 semantics).
pub fn pm_fdiv_to_mul() -> &'static TypedPatternMatcher<()> {
    use morok_ir::types::ConstValue;
    crate::cached_patterns! {
        // x / c → x * (1/c) for float constants
        Fdiv(x, _c @const(c_val)) => |x, c_val| {
            // Only apply to float types
            if !x.dtype().is_float() { return None; }

            let f = match c_val {
                ConstValue::Float(v) => v,
                _ => return None,
            };

            // Guard against divide by zero - leave as FDIV to preserve IEEE 754 semantics
            if f == 0.0 { return None; }

            // Also guard against denormalized reciprocals that could cause precision loss
            let recip = 1.0 / f;
            if !recip.is_finite() { return None; }

            let recip_const = UOp::const_(x.dtype(), ConstValue::Float(recip));
            x.try_mul(&recip_const).ok()
        },
    }
}

/// Comparison negation patterns for integers.
///
/// Tinygrad: decompositions.py:354-361
///
/// These patterns simplify negated comparisons into equivalent direct comparisons:
/// - !(x < c) → (c-1) < x  (for integers)
/// - !(c < x) → x < (c+1)  (for integers)
/// - (c1 < x) & (x < c2) → x == (c1+1)  (when c2 == c1+2, range compression)
pub fn pm_comparison_negations() -> &'static TypedPatternMatcher<()> {
    use morok_ir::types::ConstValue;

    crate::cached_patterns! {
        // !(x < c) → (c-1) < x for integers
        // When x >= c, that's equivalent to (c-1) < x
        Not(Lt(x, _c @const(c_val))) if x.dtype().is_int() => |x, c_val| {
            let v = match c_val {
                ConstValue::Int(v) => v,
                ConstValue::UInt(v) => i64::try_from(v).ok()?,
                _ => return None,
            };
            // Guard against underflow
            let c_minus_1 = v.checked_sub(1)?;
            let c_minus_1_const = UOp::const_(x.dtype(), ConstValue::Int(c_minus_1));
            c_minus_1_const.try_cmplt(x).ok()
        },

        // !(c < x) → x < (c+1) for integers
        // When x <= c, that's equivalent to x < (c+1)
        Not(Lt(_c @const(c_val), x)) if x.dtype().is_int() => |x, c_val| {
            let v = match c_val {
                ConstValue::Int(v) => v,
                ConstValue::UInt(v) => i64::try_from(v).ok()?,
                _ => return None,
            };
            // Guard against overflow
            let c_plus_1 = v.checked_add(1)?;
            let c_plus_1_const = UOp::const_(x.dtype(), ConstValue::Int(c_plus_1));
            x.try_cmplt(&c_plus_1_const).ok()
        },

        // Range compression: (c1 < x) & (x < c2) → x == (c1+1) when c2 == c1+2
        // When x is in the open interval (c1, c2) and c2 - c1 == 2, x must be c1+1
        And[Lt(_c1 @const(c1_val), x), Lt(x2, _c2 @const(c2_val))]
            if x.dtype().is_int() && Arc::ptr_eq(x, x2)
            => |x, c1_val, c2_val| {
                let v1 = match c1_val {
                    ConstValue::Int(v) => v,
                    ConstValue::UInt(v) => i64::try_from(v).ok()?,
                    _ => return None,
                };
                let v2 = match c2_val {
                    ConstValue::Int(v) => v,
                    ConstValue::UInt(v) => i64::try_from(v).ok()?,
                    _ => return None,
                };
                // Only apply if c2 == c1 + 2 (single value in range)
                if v2 != v1.checked_add(2)? { return None; }
                let target = UOp::const_(x.dtype(), ConstValue::Int(v1 + 1));
                x.try_cmpeq(&target).ok()
            },

        // x*-1 < c → -c < x for integers
        // Tinygrad: decompositions.py:354-361
        // When comparing negated value with constant, flip the comparison
        Lt(Mul(x, _neg1 @const(neg_val)), _c @const(c_val)) if x.dtype().is_int() => |x, neg_val, c_val| {
            // Check that we're multiplying by -1
            if !matches!(neg_val, ConstValue::Int(-1)) { return None; }

            let c = match c_val {
                ConstValue::Int(v) => v,
                ConstValue::UInt(v) => i64::try_from(v).ok()?,
                _ => return None,
            };

            // x*-1 < c → -c < x
            let neg_c = c.checked_neg()?;
            let neg_c_const = UOp::const_(x.dtype(), ConstValue::Int(neg_c));
            neg_c_const.try_cmplt(x).ok()
        },

        // x*-1 < y*c → y*(-c) < x for integers
        // When comparing negated x with scaled y, flip and negate scale
        Lt(Mul(x, _neg1 @const(neg_val)), Mul(y, _c @const(c_val))) if x.dtype().is_int() => |x, neg_val, y, c_val| {
            // Check that we're multiplying x by -1
            if !matches!(neg_val, ConstValue::Int(-1)) { return None; }

            let c = match c_val {
                ConstValue::Int(v) => v,
                ConstValue::UInt(v) => i64::try_from(v).ok()?,
                _ => return None,
            };

            // x*-1 < y*c → y*(-c) < x
            let neg_c = c.checked_neg()?;
            let neg_c_const = UOp::const_(y.dtype(), ConstValue::Int(neg_c));
            let y_neg_c = y.try_mul(&neg_c_const).ok()?;
            y_neg_c.try_cmplt(x).ok()
        },
    }
}

// ============================================================================
