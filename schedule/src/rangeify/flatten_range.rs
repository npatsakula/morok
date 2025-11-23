//! Range flattening and canonicalization patterns.
//!
//! This module implements range flattening that unnests and canonicalizes RANGE
//! operations for kernel deduplication. By ensuring RANGEs appear in a consistent
//! order, we enable two functionally identical kernels to have identical structure.
//!
//! Based on Tinygrad's flatten_range (codegen/simplify.py:14-17).

use std::collections::HashMap;
use std::rc::Rc;

use morok_ir::{Op, UOp, UOpKey};

use crate::pattern::matcher::PatternMatcher;

/// Flatten nested RANGE operations into canonical form.
///
/// This function implements Tinygrad's modern flatten_range pattern using SINK + toposort.
/// It:
/// 1. Identifies operations with range sources (REDUCE, STORE, END)
/// 2. Uses SINK + toposort to gather all nested RANGEs (no consumer_map needed)
/// 3. Reconstructs the operation with flattened range structure
///
/// # Arguments
///
/// * `r` - The operation to flatten
///
/// # Returns
///
/// * `Some(flattened)` - The operation with canonicalized range ordering
/// * `None` - If no flattening was needed or operation doesn't support it
///
/// # Example
///
/// ```ignore
/// // Before: STORE(buffer, idx, value, RANGE(RANGE(end)))
/// // After:  STORE(buffer, idx, value, innermost_range, outer_range)
/// ```
///
/// Based on Tinygrad's flatten_range (codegen/simplify.py:7-12):
/// ```python
/// def flatten_range(r:UOp):
///   off = range_start[r.op]
///   rngs = r.src[off:]
///   if not len(rngs): return None
///   new_rngs = [x for x in UOp.sink(*rngs).toposort() if x.op is Ops.RANGE]
///   return r.replace(src=r.src[:off]+tuple(new_rngs))
/// ```
pub fn flatten_range_impl(r: &Rc<UOp>) -> Option<Rc<UOp>> {
    // Only process REDUCE, STORE, END operations (matches Tinygrad)
    // Each has range sources after a fixed offset
    let off = match r.op() {
        Op::Reduce { .. } => 1, // Skip first source (value)
        Op::Store { .. } => 3,  // Skip buffer, index, value
        Op::End { .. } => 1,    // Skip first source (computation)
        _ => return None,
    };

    // Extract range sources (sources after offset)
    let range_sources: Vec<Rc<UOp>> =
        r.op().sources().iter().skip(off).filter(|src| matches!(src.op(), Op::Range { .. })).cloned().collect();

    if range_sources.is_empty() {
        return None;
    }

    // Use SINK + toposort to gather all nested ranges (Tinygrad's modern approach)
    // This replaces the old consumer_map + sparents approach
    let sink = UOp::sink(range_sources);
    let new_ranges: Vec<Rc<UOp>> =
        sink.toposort().into_iter().filter(|uop| matches!(uop.op(), Op::Range { .. })).collect();

    // Reconstruct with flattened ranges
    let mut new_sources: Vec<Rc<UOp>> = r.op().sources()[..off].to_vec();
    new_sources.extend(new_ranges);

    Some(r.with_sources(new_sources))
}

/// Create patterns for range flattening.
///
/// Pattern matcher for range flattening (placeholder).
///
/// **NOTE:** This returns an empty PatternMatcher because range flattening is
/// implemented as a direct transformation (see `flatten_ranges()`) rather than
/// a pattern-based rewrite.
///
/// **Rationale for direct transformation:**
/// - Consumer map must be computed once and shared across all nodes (efficient)
/// - Toposort ensures correct traversal order for replacements
/// - Direct substitution is faster than pattern matching on every node
/// - No composition needed - flatten_range is a standalone transform
///
/// **Alternative considered:** Pattern-based approach using closure capture (similar
/// to `apply_rangeify_patterns()`). This would work but adds unnecessary overhead:
/// - Pattern matching on every node (slower than toposort + filter)
/// - Rc-wrapping consumer_map for closure capture (extra allocation)
/// - No composition benefits since this is not combined with other patterns
///
/// See `flatten_ranges()` for the actual implementation.
///
/// # Returns
///
/// An empty PatternMatcher (not used)
pub fn flatten_range_patterns() -> PatternMatcher {
    let patterns = vec![];
    PatternMatcher::new(patterns)
}

/// Apply range flattening to a computation graph.
///
/// This is a direct transformation function (not pattern-based) that flattens
/// nested RANGE operations in REDUCE/STORE/END operations for canonical ordering.
///
/// **Implementation Strategy:**
/// Uses direct transformation with SINK + toposort (Tinygrad's modern approach).
/// This is simpler and more efficient than the old consumer_map + sparents approach:
/// 1. No consumer map needed - SINK + toposort gathers nested ranges automatically
/// 2. Toposort ensures correct traversal order for dependencies
/// 3. Direct substitution avoids pattern matching overhead
/// 4. No composition needed - this is a standalone transform
///
/// **Pattern-based approach not used:** While we could use closure capture to
/// share state with patterns (see `apply_rangeify_patterns()` for example),
/// that would add overhead without providing composition benefits.
///
/// # Arguments
///
/// * `root` - The root of the computation graph
///
/// # Returns
///
/// The flattened computation graph
///
/// # Example
///
/// ```ignore
/// let flattened = flatten_ranges(&computation);
/// ```
#[allow(clippy::mutable_key_type)]
pub fn flatten_ranges(root: &Rc<UOp>) -> Rc<UOp> {
    // No consumer map needed! (simplified via SINK + toposort)
    let mut replacements: HashMap<UOpKey, Rc<UOp>> = HashMap::new();

    for node in root.toposort() {
        // Try to flatten this node
        if let Some(flattened) = flatten_range_impl(&node) {
            replacements.insert(UOpKey(node.clone()), flattened);
        }
    }

    // Apply replacements
    root.substitute(&replacements)
}
