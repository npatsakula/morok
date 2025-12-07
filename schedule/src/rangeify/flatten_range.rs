//! Range flattening: unnest and canonicalize RANGE order for kernel deduplication.

use std::collections::HashMap;
use std::sync::Arc;

use morok_ir::{Op, UOp, UOpKey};

use crate::pattern::matcher::PatternMatcher;

/// Flatten nested RANGE operations into canonical form.
///
/// For END operations with nested END computations, this extracts the innermost
/// non-END computation and collects all ranges from all nesting levels:
/// `END(END(x, [r1]), [r2])` → `END(x, [r1, r2])`
pub fn flatten_range_impl(r: &Arc<UOp>) -> Option<Arc<UOp>> {
    // Only process REDUCE, STORE, END operations (matches Tinygrad)
    // Each has range sources after a fixed offset
    let off = match r.op() {
        Op::Reduce { .. } => 1, // Skip first source (value)
        Op::Store { .. } => 3,  // Skip buffer, index, value
        Op::End { .. } => 1,    // Skip first source (computation)
        _ => return None,
    };

    // Collect ranges from this level
    let mut all_range_sources: Vec<Arc<UOp>> = r.op().sources().iter().skip(off).cloned().collect();

    // For END: walk down nested END computations and collect their ranges too
    // This handles END(END(x, [r1]), [r2]) → END(x, [r1, r2])
    let innermost_computation = if matches!(r.op(), Op::End { .. }) {
        let mut computation = Arc::clone(&r.op().sources()[0]);

        // Walk down nested ENDs collecting ranges
        while matches!(computation.op(), Op::End { .. }) {
            // Add ranges from this nested END
            all_range_sources.extend(computation.op().sources().iter().skip(1).cloned());
            // Move to inner computation
            computation = Arc::clone(&computation.op().sources()[0]);
        }

        Some(computation)
    } else {
        None
    };

    if all_range_sources.is_empty() {
        return None;
    }

    // Use SINK + toposort to gather all ranges and deduplicate
    let sink = UOp::sink(all_range_sources);
    let new_ranges: Vec<Arc<UOp>> =
        sink.toposort().into_iter().filter(|uop| matches!(uop.op(), Op::Range { .. })).collect();

    // If no ranges found after flattening, nothing to do
    if new_ranges.is_empty() {
        return None;
    }

    // Reconstruct with flattened ranges and innermost computation
    let mut new_sources: Vec<Arc<UOp>> =
        if let Some(inner_comp) = innermost_computation { vec![inner_comp] } else { r.op().sources()[..off].to_vec() };
    new_sources.extend(new_ranges);

    Some(r.with_sources(new_sources))
}

/// Empty PatternMatcher (flattening uses direct transformation instead).
pub fn flatten_range_patterns() -> PatternMatcher {
    let patterns = vec![];
    PatternMatcher::new(patterns)
}

/// Apply range flattening to a computation graph via direct transformation.
#[allow(clippy::mutable_key_type)]
pub fn flatten_ranges(root: &Arc<UOp>) -> Arc<UOp> {
    // No consumer map needed! (simplified via SINK + toposort)
    let mut replacements: HashMap<UOpKey, Arc<UOp>> = HashMap::new();

    for node in root.toposort() {
        // Try to flatten this node
        if let Some(flattened) = flatten_range_impl(&node) {
            replacements.insert(UOpKey(node.clone()), flattened);
        }
    }

    // Apply replacements
    root.substitute(&replacements)
}
