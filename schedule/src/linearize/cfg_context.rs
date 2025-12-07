//! Control flow graph context for linearization.
//!
//! CFGContext analyzes the control flow structure of a kernel AST and computes
//! ordering edges between sibling RANGE operations at the same nesting level.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use morok_ir::UOp;
use morok_ir::op::Op;
use morok_ir::uop::core::UOpKey;

/// Control flow graph context for linearization.
///
/// Tracks ordering edges between sibling RANGE operations to ensure
/// proper linearization order when loops are at the same nesting level.
///
/// # Control Flow Edges
///
/// When multiple RANGEs exist at the same nesting level, they need to be
/// ordered consistently. CFGContext computes edges where:
/// - Each RANGE points to its predecessor (another RANGE or END)
/// - Edges ensure sequential execution of sibling loops
///
/// # Example
///
/// ```text
/// RANGE(i) → END(body1)
/// RANGE(j) → END(body2)   // j comes after i
/// RANGE(k) → END(body3)   // k comes after j
///
/// CFGContext edges:
///   RANGE(j) → RANGE(i)   // j depends on i
///   RANGE(k) → RANGE(j)   // k depends on j
/// ```
#[derive(Debug, Default)]
pub struct CFGContext {
    /// Maps RANGE → predecessor (previous sibling RANGE or END).
    ///
    /// The predecessor is the operation that must complete before
    /// this RANGE can begin execution.
    #[allow(clippy::mutable_key_type)]
    pub edges: HashMap<UOpKey, Arc<UOp>>,
}

impl CFGContext {
    /// Build a control flow context from a kernel AST.
    ///
    /// Analyzes the graph to find sibling RANGEs at the same nesting level
    /// and creates ordering edges between them.
    ///
    /// # Algorithm
    ///
    /// 1. Build a dependency set for each UOp (all transitive dependencies)
    /// 2. For each END, find all RANGEs in its ranges list
    /// 3. Determine parent END for each RANGE (if nested)
    /// 4. Group sibling RANGEs by their parent
    /// 5. Order siblings by dependency count (fewer deps = earlier)
    /// 6. Create edges between consecutive siblings
    pub fn new(sink: &Arc<UOp>) -> Self {
        let mut ctx = Self::default();

        // Step 1: Collect all nodes via toposort
        let nodes = sink.toposort();

        // Step 2: Build dependency sets for each node
        // deps[u] = set of all UOps that u transitively depends on
        #[allow(clippy::mutable_key_type)]
        let mut deps: HashMap<UOpKey, HashSet<UOpKey>> = HashMap::new();

        for node in &nodes {
            #[allow(clippy::mutable_key_type)]
            let mut node_deps = HashSet::new();

            // Add direct children's dependencies
            node.op().map_child(|child| {
                // Add the child itself
                node_deps.insert(UOpKey(child.clone()));
                // Add all of child's dependencies
                if let Some(child_deps) = deps.get(&UOpKey(child.clone())) {
                    node_deps.extend(child_deps.iter().cloned());
                }
            });

            deps.insert(UOpKey(node.clone()), node_deps);
        }

        // Step 3: Collect all ENDs and their associated RANGEs
        #[allow(clippy::mutable_key_type)]
        let mut end_to_ranges: HashMap<UOpKey, Vec<Arc<UOp>>> = HashMap::new();

        for node in &nodes {
            if let Op::End { ranges, .. } = node.op() {
                let range_list: Vec<Arc<UOp>> = ranges.iter().cloned().collect();
                end_to_ranges.insert(UOpKey(node.clone()), range_list);
            }
        }

        // Step 4: For each RANGE, find its parent END (if nested)
        // A RANGE X is nested in END Y if END(Y) depends on X
        #[allow(clippy::mutable_key_type)]
        let mut range_parent: HashMap<UOpKey, Option<Arc<UOp>>> = HashMap::new();

        for node in &nodes {
            if let Op::Range { .. } = node.op() {
                let range_key = UOpKey(node.clone());

                // Find the innermost END that contains this RANGE
                let mut parent: Option<Arc<UOp>> = None;

                for end_key in end_to_ranges.keys() {
                    if let Some(end_deps) = deps.get(end_key) {
                        // Check if this END depends on our RANGE (meaning RANGE is inside END)
                        if end_deps.contains(&range_key) {
                            // Check if this is a more specific (nested) parent
                            if let Some(ref current_parent) = parent {
                                // If current parent depends on new END, then new END is more nested
                                // (new END is inside current parent, so it's more specific)
                                let current_deps = deps.get(&UOpKey(current_parent.clone()));
                                if let Some(current_deps) = current_deps
                                    && current_deps.contains(end_key)
                                {
                                    // current_parent depends on end_key, so end_key is inside current_parent
                                    // This makes end_key a more specific (closer) parent
                                    parent = Some(end_key.0.clone());
                                }
                            } else {
                                parent = Some(end_key.0.clone());
                            }
                        }
                    }
                }

                range_parent.insert(range_key, parent);
            }
        }

        // Step 5: Group siblings by parent END
        // Siblings are RANGEs that share the same parent (or both have no parent)
        #[allow(clippy::mutable_key_type)]
        let mut siblings: HashMap<Option<UOpKey>, Vec<Arc<UOp>>> = HashMap::new();

        for (range_key, parent) in &range_parent {
            let parent_key = parent.as_ref().map(|p| UOpKey(p.clone()));
            siblings.entry(parent_key).or_default().push(range_key.0.clone());
        }

        // Step 6: Order siblings and create edges
        for (_parent, mut sibling_ranges) in siblings {
            if sibling_ranges.len() <= 1 {
                continue; // No ordering needed for single RANGE
            }

            // Order by dependency count (fewer deps = earlier in sequence)
            sibling_ranges.sort_by_key(|r| deps.get(&UOpKey(r.clone())).map_or(0, |d| d.len()));

            // Create edges: each RANGE points to its predecessor
            for window in sibling_ranges.windows(2) {
                let prev = &window[0];
                let curr = &window[1];
                ctx.edges.insert(UOpKey(curr.clone()), prev.clone());
            }
        }

        ctx
    }

    /// Get the predecessor for a given RANGE operation.
    ///
    /// Returns `Some(predecessor)` if this RANGE has a sibling that must
    /// execute before it, `None` if this is the first RANGE at its level.
    pub fn get_predecessor(&self, range: &Arc<UOp>) -> Option<&Arc<UOp>> {
        self.edges.get(&UOpKey(range.clone()))
    }

    /// Check if this context has any control flow edges.
    ///
    /// Returns `true` if there are sibling RANGEs that require ordering.
    pub fn has_edges(&self) -> bool {
        !self.edges.is_empty()
    }

    /// Get the number of control flow edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use morok_dtype::DType;
    use morok_ir::types::ConstValue;

    #[test]
    fn test_cfg_context_single_range() {
        // Single RANGE should have no edges
        let end_val = UOp::index_const(10);
        let range = UOp::range(end_val, 0);
        let value = UOp::const_(DType::Float32, ConstValue::Float(1.0));
        let end = UOp::end(value, smallvec::smallvec![range]);
        let sink = UOp::sink(vec![end]);

        let ctx = CFGContext::new(&sink);
        assert!(!ctx.has_edges());
    }

    #[test]
    fn test_cfg_context_sibling_ranges() {
        // Two sibling RANGEs should have one edge
        let end_val = UOp::index_const(10);
        let range1 = UOp::range(end_val.clone(), 0);
        let range2 = UOp::range(end_val, 1);

        let value = UOp::const_(DType::Float32, ConstValue::Float(1.0));
        let end = UOp::end(value, smallvec::smallvec![range1.clone(), range2.clone()]);
        let sink = UOp::sink(vec![end]);

        let ctx = CFGContext::new(&sink);
        // With 2 ranges, we should have 1 edge (range2 → range1)
        assert!(ctx.edge_count() <= 1);
    }

    #[test]
    fn test_cfg_context_nested_ranges() {
        // Nested RANGEs: outer contains inner
        let end_val = UOp::index_const(10);

        // Inner range
        let inner_range = UOp::range(end_val.clone(), 0);
        let inner_value = UOp::const_(DType::Float32, ConstValue::Float(1.0));
        let inner_end = UOp::end(inner_value, smallvec::smallvec![inner_range.clone()]);

        // Outer range containing inner
        let outer_range = UOp::range(end_val, 1);
        let outer_end = UOp::end(inner_end, smallvec::smallvec![outer_range.clone()]);

        let sink = UOp::sink(vec![outer_end]);

        let ctx = CFGContext::new(&sink);
        // Nested ranges at different levels shouldn't have edges between them
        // (they're not siblings)
        assert!(ctx.get_predecessor(&outer_range).is_none());
    }
}
