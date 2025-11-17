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

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use morok_ir::{Op, UOp};

use crate::pattern::UPat;
use crate::pattern::matcher::{PatternMatcher, RewriteFn, RewriteResult};

use super::helpers::{is_always_run_op, is_cheap_to_inline, is_dead_axis, ranges_equal};
use super::indexing::IndexingContext;
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
    let mut patterns = vec![];

    // Pattern 1: DETACH(x) → x
    // DETACH marks gradient boundaries during autodiff, but is not needed
    // for scheduling/execution.
    pattern!(patterns,
        UPat::detach(UPat::var("x")) => |x| {
            Some(Rc::clone(x))
        }
    );

    // Pattern 2: CONTIGUOUS_BACKWARD(x) → x
    // CONTIGUOUS_BACKWARD marks backward pass contiguous requirements,
    // but is not needed for scheduling/execution.
    pattern!(patterns,
        UPat::contiguous_backward(UPat::var("x")) => |x| {
            Some(Rc::clone(x))
        }
    );

    PatternMatcher::new(patterns)
}

/// Create patterns for applying rangeify transformation with IndexingContext.
///
/// This function creates a PatternMatcher whose patterns have access to the IndexingContext
/// via closure capture. The patterns implement the core rangeify algorithm:
///
/// 1. **Generic BUFFERIZE insertion**: Adds BUFFERIZE + INDEX to op sources where needed
/// 2. **Movement op removal**: Removes movement ops after transformation
/// 3. **REDUCE_AXIS → REDUCE**: Converts REDUCE_AXIS with ranges to REDUCE
/// 4. **PAD → WHERE**: Converts PAD operations to WHERE with validity checks
/// 5. **CONST/DEFINE_VAR cleanup**: Removes sources from constants
///
/// # Arguments
///
/// * `ctx` - Shared reference to IndexingContext for range and realize tracking
///
/// # Returns
///
/// A PatternMatcher with all rangeify transformation patterns
pub fn apply_rangeify_patterns(ctx: Rc<RefCell<IndexingContext>>) -> PatternMatcher {
    let mut patterns: Vec<(UPat, RewriteFn)> = vec![];

    // Pattern 1: Generic BUFFERIZE insertion
    // Matches any op (except already processed) and adds BUFFERIZE+INDEX to sources
    {
        let ctx_clone = Rc::clone(&ctx);
        patterns.push((
            UPat::var("x"),
            Box::new(move |bindings: &HashMap<String, Rc<UOp>>| {
                let Some(x) = bindings.get("x") else {
                    return RewriteResult::NoMatch;
                };

                // Try to transform sources (scope limits context borrow)
                let new_sources = {
                    let ctx_ref = ctx_clone.borrow();
                    transform_sources_with_bufferize(x, &ctx_ref)
                };

                if let Some(new_sources) = new_sources {
                    return RewriteResult::Rewritten(x.with_sources(new_sources));
                }

                RewriteResult::NoMatch
            }) as RewriteFn,
        ));
    }

    // Pattern 2: Remove movement ops after transformation
    // Matches RESHAPE, PERMUTE, EXPAND, PAD, SHRINK, FLIP and removes them if processed
    {
        let ctx_clone = Rc::clone(&ctx);
        patterns.push((
            UPat::var("x"),
            Box::new(move |bindings: &HashMap<String, Rc<UOp>>| {
                let Some(x) = bindings.get("x") else {
                    return RewriteResult::NoMatch;
                };

                // Check if this is a movement op
                if !x.op().is_movement() {
                    return RewriteResult::NoMatch;
                }

                // Check if it should be removed (scope limits context borrow)
                let should_remove = {
                    let ctx_ref = ctx_clone.borrow();
                    should_remove_movement_op(x, &ctx_ref)
                };

                if should_remove {
                    // Return the source (first operand)
                    if let Some(src) = x.op().sources().first() {
                        return RewriteResult::Rewritten(Rc::clone(src));
                    }
                }

                RewriteResult::NoMatch
            }) as RewriteFn,
        ));
    }

    PatternMatcher::new(patterns)
}

/// Pattern matcher for buffer folding and constant propagation.
///
/// This implements simple structural buffer folding patterns:
/// - Remove noop bufferize operations (INDEX with same ranges)
/// - Fold constants through buffer operations
///
/// More complex patterns (dead axis removal, cost-based removal) are in separate functions.
pub fn buffer_folding() -> PatternMatcher {
    let mut patterns = vec![];

    // Pattern 1: Remove noop BUFFERIZE
    // INDEX(BUFFERIZE(x, ranges1), ranges2) → x if ranges1 == ranges2
    pattern!(patterns,
        UPat::var("idx") => |idx: &Rc<UOp>| {
            if let Op::Index { buffer, indices, gate: None } = idx.op()
                && let Op::Bufferize { compute, ranges: buf_ranges, .. } = buffer.op()
                    && ranges_equal(buf_ranges, indices) {
                        return Some(Rc::clone(compute));
                    }
            None
        }
    );

    // Pattern 2: BUFFERIZE(CONST) → CONST
    // Constants don't need buffering
    pattern!(patterns,
        UPat::var("buf") => |buf: &Rc<UOp>| {
            if let Op::Bufferize { compute, .. } = buf.op()
                && matches!(compute.op(), Op::Const(_)) {
                    return Some(Rc::clone(compute));
                }
            None
        }
    );

    // Pattern 3: INDEX(CONST) → CONST
    // Indexing into a constant is still that constant
    pattern!(patterns,
        UPat::var("idx") => |idx: &Rc<UOp>| {
            if let Op::Index { buffer, .. } = idx.op()
                && matches!(buffer.op(), Op::Const(_)) {
                    return Some(Rc::clone(buffer));
                }
            None
        }
    );

    // Pattern 4: COPY(CONST, device) → CONST
    // Copying a constant is still that constant (device doesn't matter for constants)
    pattern!(patterns,
        UPat::var("copy") => |copy: &Rc<UOp>| {
            if let Op::Copy { src, .. } = copy.op()
                && matches!(src.op(), Op::Const(_)) {
                    return Some(Rc::clone(src));
                }
            None
        }
    );

    PatternMatcher::new(patterns)
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
    let mut patterns = vec![];

    // Pattern: BUFFERIZE with dead axes → BUFFERIZE with only live axes
    // BUFFERIZE(compute, [r1, r2, dead, r4, ...], opts) → BUFFERIZE(compute, [r1, r2, r4, ...], opts)
    pattern!(patterns,
        UPat::var("buf") => |buf: &Rc<UOp>| {
            if let Op::Bufferize { compute, ranges, opts } = buf.op() {
                // Filter out dead axes
                let live_ranges: Vec<_> = ranges
                    .iter()
                    .filter(|r| !is_dead_axis(r))
                    .map(Rc::clone)
                    .collect();

                // Only rewrite if we removed some dead axes
                if live_ranges.len() < ranges.len() {
                    // If all axes are dead, return the compute directly
                    if live_ranges.is_empty() {
                        return Some(Rc::clone(compute));
                    }

                    // Create new BUFFERIZE with only live axes
                    return Some(UOp::bufferize(
                        Rc::clone(compute),
                        live_ranges,
                        opts.clone(),
                    ));
                }
            }
            None
        }
    );

    // Pattern: INDEX into BUFFERIZE with dead axes → INDEX with adjusted indices
    // When BUFFERIZE has dead axes removed, INDEX operations need matching adjustments
    pattern!(patterns,
        UPat::var("idx") => |idx: &Rc<UOp>| {
            if let Op::Index { buffer, indices, gate: _ } = idx.op()
                && let Op::Bufferize { ranges: buf_ranges, .. } = buffer.op() {
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

                        return UOp::index(
                            Rc::clone(buffer),
                            new_indices,
                        ).ok();
                    }
                }
            None
        }
    );

    PatternMatcher::new(patterns)
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
    let mut patterns = vec![];

    // Pattern 1: Remove BUFFERIZE when compute is cheap to inline
    // BUFFERIZE(cheap_compute, ranges, opts) → cheap_compute
    // Only applies to simple operations (unary, binary, casts, etc.)
    pattern!(patterns,
        UPat::var("buf") => |buf: &Rc<UOp>| {
            if let Op::Bufferize { compute, .. } = buf.op() {
                // If compute is cheap and doesn't need buffering, inline it
                if is_cheap_to_inline(compute.op()) {
                    return Some(Rc::clone(compute));
                }
            }
            None
        }
    );

    // Pattern 2: Remove BUFFERIZE when compute must always run
    // Operations like CONTIGUOUS, COPY, ASSIGN have side effects
    // and shouldn't be wrapped in BUFFERIZE
    pattern!(patterns,
        UPat::var("buf") => |buf: &Rc<UOp>| {
            if let Op::Bufferize { compute, .. } = buf.op()
                && is_always_run_op(compute.op()) {
                    return Some(Rc::clone(compute));
                }
            None
        }
    );

    // Pattern 3: Remove nested BUFFERIZE (redundant buffering)
    // BUFFERIZE(BUFFERIZE(x, r1, o1), r2, o2) → BUFFERIZE(x, r2, o2)
    // Inner buffer is unnecessary if outer buffer exists
    pattern!(patterns,
        UPat::var("outer") => |outer: &Rc<UOp>| {
            if let Op::Bufferize { compute: outer_compute, ranges, opts } = outer.op()
                && let Op::Bufferize { compute: inner_compute, .. } = outer_compute.op() {
                    // Replace nested BUFFERIZE with single-level BUFFERIZE
                    return Some(UOp::bufferize(
                        Rc::clone(inner_compute),
                        ranges.to_vec(),
                        opts.clone(),
                    ));
                }
            None
        }
    );

    PatternMatcher::new(patterns)
}

/// Pattern matcher for kernel splitting.
///
/// This will split the graph into individual kernels at
/// STORE operation boundaries.
///
/// Phase 1 stub - returns empty matcher.
pub fn kernel_splitting() -> PatternMatcher {
    // TODO (Phase 2): Implement kernel boundary detection
    PatternMatcher::new(vec![])
}
