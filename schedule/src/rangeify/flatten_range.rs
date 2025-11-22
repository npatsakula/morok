//! Range flattening and canonicalization patterns.
//!
//! This module implements range flattening that unnests and canonicalizes RANGE
//! operations for kernel deduplication. By ensuring RANGEs appear in a consistent
//! order, we enable two functionally identical kernels to have identical structure.
//!
//! Based on Tinygrad's flatten_range (codegen/simplify.py:14-17).

use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use morok_ir::{Op, UOp, UOpKey};

use crate::pattern::matcher::PatternMatcher;

/// Extract all RANGEs from a chain recursively.
///
/// Walks down the chain of RANGE operations, collecting all RANGEs found.
/// Used to flatten nested RANGE structures.
///
/// # Arguments
///
/// * `offset` - The offset expression that may contain nested RANGEs
///
/// # Returns
///
/// A vector of all RANGE operations found in the chain
///
/// # Example
///
/// ```ignore
/// // RANGE(end=RANGE(end=RANGE(end=const, ...), ...), ...)
/// // Returns: [innermost_range, middle_range, outer_range]
/// ```
pub(crate) fn get_range_chain(offset: &Rc<UOp>) -> Vec<Rc<UOp>> {
    match offset.op() {
        Op::Range { end, .. } => {
            let mut chain = get_range_chain(end);
            chain.push(offset.clone());
            chain
        }
        _ => vec![],
    }
}

/// Get all RANGE operations that are parents/consumers of this operation.
///
/// Uses the consumer map to find all RANGEs that depend on (consume) the given UOp.
/// This is our equivalent of Tinygrad's `sparents.get(Ops.RANGE, ())`.
///
/// # Arguments
///
/// * `uop` - The operation to get range parents for
/// * `consumer_map` - Precomputed consumer map for the graph
///
/// # Returns
///
/// A vector of RANGE operations that consume this UOp
pub(crate) fn get_range_parents(uop: &Rc<UOp>, consumer_map: &HashMap<UOpKey, Vec<Rc<UOp>>>) -> Vec<Rc<UOp>> {
    let key = UOpKey(uop.clone());

    consumer_map
        .get(&key)
        .map(|consumers| {
            consumers.iter().filter(|consumer| matches!(consumer.op(), Op::Range { .. })).cloned().collect()
        })
        .unwrap_or_default()
}

/// Flatten nested RANGE operations into canonical form.
///
/// This function implements Tinygrad's flatten_range pattern. It:
/// 1. Identifies operations that can have nested ranges (Binary, Ternary, WMMA, RANGE)
/// 2. Extracts all nested RANGEs from sources and parent ranges
/// 3. Deduplicates and sorts ranges by ID for canonical ordering
/// 4. Reconstructs the operation with flattened range structure
///
/// # Arguments
///
/// * `r` - The operation to flatten
/// * `consumer_map` - Precomputed consumer map for parent tracking
///
/// # Returns
///
/// * `Some(flattened)` - The operation with canonicalized range ordering
/// * `None` - If no flattening was needed or operation doesn't support it
///
/// # Example
///
/// ```ignore
/// // Before: Binary with nested ranges in sources and parent RANGEs
/// //         Binary(RANGE(RANGE(end)), parent_ranges=[R1, R2])
/// // After:  Binary(end, canonical_ranges=[innermost, R1, R2, middle, outer])
/// ```
///
/// Based on Tinygrad's flatten_range (codegen/simplify.py:14-17):
/// ```python
/// def flatten_range(uop:UOp) -> UOp|None:
///   if uop.op not in {Ops.ALU, Ops.DEFINE_ACC, Ops.WMMA, Ops.RANGE}: return None
///   # ... flattening logic ...
/// ```
///
/// Note: We don't have DEFINE_ACC in our IR yet, so it's omitted.
pub fn flatten_range_impl(r: &Rc<UOp>, consumer_map: &HashMap<UOpKey, Vec<Rc<UOp>>>) -> Option<Rc<UOp>> {
    // Only process operations that can have nested ranges
    match r.op() {
        Op::Binary { .. } | Op::Ternary { .. } | Op::Wmma { .. } | Op::Range { .. } => {}
        _ => return None,
    }

    // Extract all RANGE operations from sources based on operation type
    let ranges: Vec<Rc<UOp>> = match r.op() {
        Op::Range { .. } => {
            // For RANGE, look at sources[1..]
            r.op().sources().iter().skip(1).filter(|src| matches!(src.op(), Op::Range { .. })).cloned().collect()
        }
        _ => {
            // For other ops (Binary, Ternary, WMMA), look at all sources' parent ranges
            r.op().sources().iter().flat_map(|src| get_range_parents(src, consumer_map)).collect()
        }
    };

    if ranges.is_empty() {
        return None;
    }

    // Get parent RANGEs of this operation
    let parent_ranges = get_range_parents(r, consumer_map);

    // Flatten by extracting chain from each range's offset (src[0])
    let mut all_ranges = Vec::new();

    // Add chains from direct ranges + parent ranges
    for rng in ranges.iter().chain(parent_ranges.iter()) {
        if let Op::Range { end, .. } = rng.op() {
            let chain = get_range_chain(end);
            all_ranges.extend(chain);
        }
    }

    // Add the direct ranges themselves
    all_ranges.extend(ranges.clone());

    // Deduplicate and sort by pointer address for canonical ordering
    let mut seen = HashSet::new();
    let mut unique_ranges = Vec::new();
    for range in all_ranges {
        let key = Rc::as_ptr(&range) as usize;
        if seen.insert(key) {
            unique_ranges.push(range);
        }
    }

    // Sort by pointer address to ensure deterministic ordering
    unique_ranges.sort_by_key(|r| Rc::as_ptr(r) as usize);

    // If no change in range count, skip reconstruction
    if unique_ranges.len() == ranges.len() && unique_ranges.iter().zip(ranges.iter()).all(|(a, b)| Rc::ptr_eq(a, b)) {
        return None;
    }

    // Reconstruct operation with flattened ranges
    let new_sources: Vec<Rc<UOp>> = match r.op() {
        Op::Range { .. } => {
            // RANGE: replace sources[1..] with flattened range offsets (src[0])
            let mut new = vec![r.op().sources()[0].clone()];
            new.extend(
                unique_ranges
                    .iter()
                    .filter_map(|rng| if let Op::Range { end, .. } = rng.op() { Some(end.clone()) } else { None }),
            );
            new
        }
        _ => {
            // Binary, Ternary, WMMA: keep original sources (ranges are in parent relationships)
            // Flattening affects parent tracking, not direct sources
            // TODO: Handle parent range update when we have parent caching similar to sparents
            return None;
        }
    };

    // Create new operation with flattened sources
    Some(r.with_sources(new_sources))
}

/// Create patterns for range flattening.
///
/// Note: This implementation currently requires a consumer map to be passed in,
/// but our pattern system doesn't support per-pattern context. For now, we return
/// an empty matcher as a placeholder.
///
/// TODO: Integrate with graph_rewrite to pass consumer_map through context, or
/// implement as a direct transformation function rather than a pattern.
///
/// # Arguments
///
/// * (none currently)
///
/// # Returns
///
/// An empty PatternMatcher (placeholder)
pub fn flatten_range_patterns() -> PatternMatcher {
    let patterns = vec![];

    // TODO: Once we have pattern context support, implement as:
    // pattern!(patterns,
    //     UPat::var("r") => |r, consumer_map| {
    //         flatten_range_impl(r, consumer_map)
    //     }
    // );

    PatternMatcher::new(patterns)
}

/// Apply range flattening to a computation graph.
///
/// This is a direct transformation function (not pattern-based) that flattens
/// all RANGE operations in the graph for canonical ordering.
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
pub fn flatten_ranges(root: &Rc<UOp>) -> Rc<UOp> {
    // Build consumer map once for the entire graph
    let consumer_map = root.get_consumer_map();

    // Traverse in topological order and flatten each node
    let mut replacements: HashMap<UOpKey, Rc<UOp>> = HashMap::new();

    for node in root.toposort() {
        // Try to flatten this node
        if let Some(flattened) = flatten_range_impl(&node, &consumer_map) {
            replacements.insert(UOpKey(node.clone()), flattened);
        }
    }

    // Apply replacements
    root.substitute(&replacements)
}
