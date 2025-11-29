//! Pattern matchers for rangeify transformations.
//!
//! This module contains pattern matchers used during scheduling/rangeify:
//! - Movement op → BUFFERIZE conversion
//! - Early cleanup rewrites (schedule-specific)
//! - Buffer folding and removal
//! - Kernel splitting patterns
//!
//! Note: Algebraic simplifications (x+0, x*1, x*0, etc.) have been moved
//! to the symbolic module (schedule/src/symbolic/patterns.rs).

use std::rc::Rc;

use morok_ir::{Op, UOp};

use crate::pattern::matcher::PatternMatcher;

use super::buffer_cost::PcontigConfig;
use super::helpers::{is_always_run_op, is_cheap_to_inline, is_dead_axis, ranges_equal};
use super::indexing::IndexingContext;
use super::reduce_simplify::reduce_unparented;
use super::split_reduceop::{SplitReduceOpConfig, split_reduceop};
use super::transform::{should_remove_movement_op, transform_sources_with_bufferize};

/// Pattern matcher for removing movement ops after rangeify transformation.
///
/// Movement ops are removed when:
/// 1. They have ranges assigned (been processed), OR
/// 2. Their source is already INDEX (transformation applied)
///
/// This will be populated with actual patterns in Step 2.3.
/// The logic uses `should_remove_movement_op()` from the transform module.
pub fn movement_op_removal() -> PatternMatcher {
    // Placeholder - patterns will be added in Step 2.3 with context support
    PatternMatcher::new(vec![])
}

/// Pattern matcher for early cleanup rewrites during scheduling.
///
/// This handles schedule-specific cleanup:
/// - DETACH removal (gradient computation marker no longer needed)
/// - CONTIGUOUS_BACKWARD removal (gradient computation marker no longer needed)
///
/// Note: Algebraic simplifications (x+0, x*1, x*0, etc.) have been moved
/// to the symbolic module where they belong.
pub fn early_rewrites() -> PatternMatcher {
    // Using the patterns! proc-macro DSL for simple rewrites:
    // - DETACH(x) → x: DETACH marks gradient boundaries, not needed for scheduling
    // - CONTIGUOUS_BACKWARD(x) → x: backward pass marker, not needed for scheduling
    crate::patterns! {
        Detach(x) ~> x,
        ContiguousBackward(x) ~> x,
    }
}

/// Create patterns for applying rangeify transformation with IndexingContext.
///
/// This function creates a PatternMatcher whose patterns have access to the IndexingContext
/// via the `@context` DSL. The patterns implement the core rangeify algorithm:
///
/// 1. **Generic BUFFERIZE insertion**: Adds BUFFERIZE + INDEX to op sources where needed
/// 2. **Movement op removal**: Removes movement ops after transformation
///
/// # Returns
///
/// A PatternMatcher<IndexingContext> with all rangeify transformation patterns
pub fn apply_rangeify_patterns() -> PatternMatcher<IndexingContext> {
    crate::patterns! {
        @context IndexingContext;
        // Pattern 1: Generic BUFFERIZE insertion
        x => apply_bufferize_transform(x, ctx),
        // Pattern 2: Remove movement ops after transformation
        x if x.op().is_movement() => remove_movement_op(x, ctx),
    }
}

/// Apply BUFFERIZE transformation to op sources.
fn apply_bufferize_transform(x: &Rc<UOp>, ctx: &mut IndexingContext) -> Option<Rc<UOp>> {
    // Try to transform sources
    let new_sources = transform_sources_with_bufferize(x, ctx)?;
    Some(x.with_sources(new_sources))
}

/// Remove movement ops after transformation has been applied.
fn remove_movement_op(x: &Rc<UOp>, ctx: &mut IndexingContext) -> Option<Rc<UOp>> {
    // Check if it should be removed
    if should_remove_movement_op(x, ctx) {
        // Return the source (first operand)
        return x.op().sources().first().map(Rc::clone);
    }
    None
}

/// Pattern matcher for buffer folding and constant propagation.
///
/// This implements simple structural buffer folding patterns:
/// - Remove noop bufferize operations (INDEX with same ranges)
/// - Fold constants through buffer operations
///
/// More complex patterns (dead axis removal, cost-based removal) are in separate functions.
pub fn buffer_folding() -> PatternMatcher {
    crate::patterns! {
        // BUFFERIZE(CONST) → CONST: Constants don't need buffering
        Bufferize { compute: c, .. } if matches!(c.op(), Op::Const(_)) ~> c,
        // INDEX(CONST) → CONST: Indexing into a constant is still that constant
        Index { buffer: c, .. } if matches!(c.op(), Op::Const(_)) ~> c,
        // COPY(CONST) → CONST: Copying a constant is still that constant
        Copy { src: c, .. } if matches!(c.op(), Op::Const(_)) ~> c,
        // INDEX(BUFFERIZE) noop removal: INDEX(BUFFERIZE(x, ranges), ranges) → x
        Index { buffer: Bufferize { compute, ranges, .. }, indices, gate: None }
            if ranges_equal(&ranges, &indices) ~> compute,
    }
}

/// Pattern matcher for dead axis removal.
///
/// This removes dimensions from BUFFERIZE operations that have size 1 (dead axes).
/// Dead axes arise when:
/// - Loop ranges become constant 1 after optimization
/// - Dimensions are broadcast to size 1
///
/// Removing dead axes simplifies the buffer structure and indexing operations.
pub fn dead_axis_removal() -> PatternMatcher {
    crate::patterns! {
        // BUFFERIZE with dead axes → BUFFERIZE with only live axes
        buf if matches!(buf.op(), Op::Bufferize { .. }) => filter_dead_bufferize(buf),
        // INDEX into BUFFERIZE with dead axes → INDEX with adjusted indices
        idx if matches!(idx.op(), Op::Index { .. }) => adjust_index_for_dead_axes(idx),
    }
}

/// Filter dead axes from BUFFERIZE operations.
///
/// BUFFERIZE(compute, [r1, r2, dead, r4, ...], opts) → BUFFERIZE(compute, [r1, r2, r4, ...], opts)
fn filter_dead_bufferize(buf: &Rc<UOp>) -> Option<Rc<UOp>> {
    let Op::Bufferize { compute, ranges, opts } = buf.op() else {
        return None;
    };

    // Filter out dead axes
    let live_ranges: Vec<_> = ranges.iter().filter(|r| !is_dead_axis(r)).map(Rc::clone).collect();

    // Only rewrite if we removed some dead axes
    if live_ranges.len() < ranges.len() {
        // If all axes are dead, return the compute directly
        if live_ranges.is_empty() {
            return Some(Rc::clone(compute));
        }

        // Create new BUFFERIZE with only live axes
        return Some(UOp::bufferize(Rc::clone(compute), live_ranges, opts.clone()));
    }

    None
}

/// Adjust INDEX indices when BUFFERIZE has dead axes removed.
fn adjust_index_for_dead_axes(idx: &Rc<UOp>) -> Option<Rc<UOp>> {
    let Op::Index { buffer, indices, gate: _ } = idx.op() else {
        return None;
    };
    let Op::Bufferize { ranges: buf_ranges, .. } = buffer.op() else {
        return None;
    };

    // Build new indices by filtering out positions where buffer has dead axes
    let mut new_indices = Vec::new();
    let mut idx_iter = indices.iter();

    for range in buf_ranges.iter() {
        if !is_dead_axis(range) {
            // Keep this index for live axis
            if let Some(idx_val) = idx_iter.next() {
                new_indices.push(Rc::clone(idx_val));
            }
        } else {
            // Skip index for dead axis
            idx_iter.next();
        }
    }

    // Only rewrite if we removed some indices
    if new_indices.len() < indices.len() {
        // If no indices remain, this is invalid - keep original
        if new_indices.is_empty() {
            return None;
        }

        return UOp::index(Rc::clone(buffer), new_indices).ok();
    }

    None
}

/// Pattern matcher for cost-based buffer removal.
///
/// This removes BUFFERIZE operations that don't provide performance benefits
/// based on cost heuristics:
/// 1. Cheap compute that's inlined rather than buffered
/// 2. Operations that must always run (CONTIGUOUS, COPY, ASSIGN)
/// 3. Simple buffers that don't reduce recomputation
///
/// The goal is to remove unnecessary buffering overhead while preserving
/// buffers that genuinely improve performance through memoization.
pub fn buffer_removal() -> PatternMatcher {
    crate::patterns! {
        // Cheap to inline: remove buffering for simple ops
        Bufferize { compute, .. } if is_cheap_to_inline(compute.op()) ~> compute,
        // Always-run ops: CONTIGUOUS, COPY, ASSIGN shouldn't be buffered
        Bufferize { compute, .. } if is_always_run_op(compute.op()) ~> compute,
        // Nested BUFFERIZE: flatten redundant buffering
        Bufferize { compute: Bufferize { compute: inner, .. }, ranges, opts }
            => Some(UOp::bufferize(Rc::clone(inner), ranges.to_vec(), opts.clone())),
    }
}

/// Pattern matcher for cost-based buffer removal with partial contiguous support.
///
/// This extends buffer_removal() with cost-based heuristics to decide when to:
/// 1. Fully remove buffers (inline compute)
/// 2. Apply partial contiguous (materialize subset of dimensions)
/// 3. Keep buffers as-is (too complex to optimize)
///
/// Decision tree based on three heuristics:
/// - `accessed_buffers > max_buffers_threshold` → keep (complex multi-input)
/// - `out_in_ratio < out_in_ratio_threshold` → keep (efficient buffer)
/// - `buffer_in_reduce` → partial contiguous candidate
///
/// # Returns
///
/// A PatternMatcher<PcontigConfig> with buffer removal patterns including partial contiguous logic
pub fn buffer_removal_with_pcontig() -> PatternMatcher<PcontigConfig> {
    crate::patterns! {
        @context PcontigConfig;
        // Cheap to inline: remove buffering for simple ops
        Bufferize { compute, .. } if is_cheap_to_inline(compute.op()) ~> compute,
        // Always-run ops: CONTIGUOUS, COPY, ASSIGN shouldn't be buffered
        Bufferize { compute, .. } if is_always_run_op(compute.op()) ~> compute,
        // Nested BUFFERIZE: flatten redundant buffering
        Bufferize { compute: Bufferize { compute: inner, .. }, ranges, opts }
            => Some(UOp::bufferize(Rc::clone(inner), ranges.to_vec(), opts.clone())),
        // Complex INDEX(BUFFERIZE) with partial contiguous
        idx if matches!(idx.op(), Op::Index { .. }) => apply_pcontig_removal(idx, ctx),
    }
}

/// Apply partial contiguous buffer removal.
fn apply_pcontig_removal(idx: &Rc<UOp>, config: &mut PcontigConfig) -> Option<Rc<UOp>> {
    use super::buffer_cost::{
        apply_partial_contiguous, calculate_buffer_size, calculate_out_in_ratio, collect_accessed_buffers,
        collect_indexes, collect_local_indexes, collect_reduces, extract_exclude_ranges, has_buffer_in_reduce,
        partition_ranges,
    };

    // Match INDEX(BUFFERIZE(...), ...)
    let Op::Index { buffer, indices: idx_ranges, gate: None } = idx.op() else {
        return None;
    };
    let Op::Bufferize { compute: src, ranges: buf_ranges, .. } = buffer.op() else {
        return None;
    };

    // Early exit: Check if disabled
    if config.level == 0 {
        return None;
    }

    // Early exit: Always-run ops should not be transformed
    if is_always_run_op(src.op()) {
        return None;
    }

    // === Heuristic 1: accessed_buffers check ===
    // If too many buffers are accessed, keep the buffer (complex multi-input)
    let accessed_buffers = collect_accessed_buffers(src);
    if accessed_buffers.len() > config.max_buffers_threshold {
        return None;
    }

    // === Heuristic 2: out_in_ratio check ===
    // If output/input ratio is low, keep the buffer (efficient buffer)
    if let Some(output_size) = calculate_buffer_size(buffer)
        && let Some(ratio) = calculate_out_in_ratio(output_size, &accessed_buffers)
        && ratio < config.out_in_ratio_threshold
    {
        return None;
    }

    // === Heuristic 3: buffer_in_reduce check ===
    // Determines whether to do full removal or partial contiguous
    let reduces = collect_reduces(src);
    let buf_in_reduce = has_buffer_in_reduce(&reduces);

    if !buf_in_reduce {
        // No reduce usage → full removal via substitution
        use morok_ir::UOpKey;
        use std::collections::HashMap;

        // Build substitution map: buffer ranges → index ranges
        #[allow(clippy::mutable_key_type)]
        let subs_map: HashMap<UOpKey, Rc<UOp>> =
            buf_ranges.iter().zip(idx_ranges.iter()).map(|(k, v)| (UOpKey(Rc::clone(k)), Rc::clone(v))).collect();

        return Some(src.substitute(&subs_map));
    }

    // Buffer is used in reduce → partial contiguous candidate
    // Collect all INDEX operations and determine which ranges to materialize
    let indexes = collect_indexes(src);
    let local_indexes = collect_local_indexes(&indexes);
    #[allow(clippy::mutable_key_type)]
    let exclude_ranges = extract_exclude_ranges(&local_indexes);

    // Partition ranges: materialize (LOCAL, REDUCE) vs substitute (inline)
    let (materialize, substitute) = partition_ranges(buf_ranges, idx_ranges, &exclude_ranges);

    // Check if partial contiguous would be beneficial
    if materialize.is_empty() {
        // No dimensions to materialize → full removal
        use morok_ir::UOpKey;
        use std::collections::HashMap;

        #[allow(clippy::mutable_key_type)]
        let subs_map: HashMap<UOpKey, Rc<UOp>> = substitute.into_iter().map(|(k, v)| (UOpKey(k), v)).collect();

        return Some(src.substitute(&subs_map));
    }

    // Apply partial contiguous transformation
    apply_partial_contiguous(src, materialize, substitute)
}

/// Pattern matcher for reduction simplifications.
///
/// This applies optimizations to REDUCE operations:
/// 1. **reduce_unparented**: Remove ranges that don't appear in source (2-10x speedup)
/// 2. **reduce_collapse**: Lift range-independent computations outside reductions (2-10x speedup)
/// 3. **split_reduceop**: Split large reductions into two-stage for better parallelism
///
/// These patterns should run after symbolic simplification to benefit from
/// simplified index expressions.
///
/// # Returns
///
/// A PatternMatcher<SplitReduceOpConfig> with all reduction optimization patterns
///
/// # Example
///
/// ```ignore
/// let mut config = SplitReduceOpConfig::default();
/// let matcher = reduction_simplify_patterns();
/// let optimized = graph_rewrite(&matcher, graph, &mut config);
/// ```
pub fn reduction_simplify_patterns() -> PatternMatcher<SplitReduceOpConfig> {
    crate::patterns! {
        @context SplitReduceOpConfig;
        // reduce_unparented - Remove unused reduction ranges
        // REDUCE(const, [range], ADD) → const * range.size
        // REDUCE(const, [range], MUL) → const ^ range.size
        // REDUCE(const, [range], MAX/MIN) → const
        x => reduce_unparented(x),
        // reduce_collapse - Lift range-independent computations outside reductions
        // Uses symbolic substitution to detect and eliminate range dependencies
        // REDUCE(x, [range], op) → x if symbolic simplification proves x is range-independent
        x => super::reduce_simplify::reduce_collapse(x),
        // split_reduceop - Split large reductions into two stages
        // REDUCE_AXIS(src, [axis], op) where prod(shape)/prod(output) >= 32768
        // → reshape → permute → reduce1 → contiguous → reduce2 → reshape
        reduce if matches!(reduce.op(), Op::ReduceAxis { .. }) => apply_split_reduceop(reduce, ctx),
    }
}

/// Apply split_reduceop transformation.
fn apply_split_reduceop(reduce: &Rc<UOp>, config: &mut SplitReduceOpConfig) -> Option<Rc<UOp>> {
    split_reduceop(reduce, config)
}
