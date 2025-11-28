//! Range flattening: unnest and canonicalize RANGE order for kernel deduplication.

use std::collections::HashMap;
use std::rc::Rc;

use morok_ir::{Op, UOp, UOpKey};

use crate::pattern::matcher::PatternMatcher;

/// Flatten nested RANGE operations into canonical form using SINK + toposort.
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

/// Empty PatternMatcher (flattening uses direct transformation instead).
pub fn flatten_range_patterns() -> PatternMatcher {
    let patterns = vec![];
    PatternMatcher::new(patterns)
}

/// Apply range flattening to a computation graph via direct transformation.
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
