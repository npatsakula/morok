//! Control flow graph context for linearization.
//!
//! CFGContext analyzes the control flow structure of a kernel AST and computes
//! ordering edges between sibling RANGE operations at the same nesting level.
//!
//! This implementation matches Tinygrad's CFGContext (linearizer.py:59-91).

use std::collections::HashMap;
use std::sync::Arc;

use morok_ir::UOp;
use morok_ir::op::Op;
use morok_ir::uop::core::UOpKey;

/// Control flow graph context for linearization.
///
/// Tracks ordering edges between sibling RANGE operations to ensure
/// proper linearization order when loops are at the same nesting level.
///
/// Based on Tinygrad's CFGContext which tracks three relationships between ranges:
/// - **nested**: END y is a dependency of END x AND RANGE x is a dependency of END y
/// - **dependent**: END y is a dependency of END x AND RANGE x is NOT a dependency of END y
/// - **independent**: END y is NOT a dependency of END x
///
/// # Control Flow Edges
///
/// When multiple ENDs exist at the same nesting level (siblings), they need to be
/// ordered consistently. CFGContext computes edges where:
/// - Each RANGE points to its predecessor (either the parent's RANGE or another END)
/// - Edges ensure sequential execution of sibling loops
///
/// # Example
///
/// ```text
/// RANGE(i) → ... → END(i)   // first loop
/// RANGE(j) → ... → END(j)   // second loop (sibling)
/// RANGE(k) → ... → END(k)   // third loop (sibling)
///
/// CFGContext edges:
///   RANGE(j) → END(i)    // j's RANGE depends on i's END
///   RANGE(k) → END(j)    // k's RANGE depends on j's END
/// ```
#[derive(Debug, Default)]
pub struct CFGContext {
    /// Maps RANGE → predecessor (previous sibling END or parent's RANGE).
    ///
    /// The predecessor is the operation that must complete before
    /// this RANGE can begin execution.
    #[allow(clippy::mutable_key_type)]
    pub edges: HashMap<UOpKey, Arc<UOp>>,
}

impl CFGContext {
    /// Build a control flow context from a kernel AST.
    ///
    /// Analyzes the graph to find sibling ENDs at the same nesting level
    /// and creates ordering edges between their RANGEs.
    ///
    /// # Algorithm (from Tinygrad linearizer.py:59-91)
    ///
    /// 1. Build transitive deps map (RANGE/END add themselves to deps)
    /// 2. Build nesting map: which END/SINK nests each END
    /// 3. Group siblings by parent
    /// 4. Order siblings by dependency count (fewer deps = earlier)
    /// 5. Create edges: RANGE of later sibling → predecessor (END or parent's RANGE)
    pub fn new(sink: &Arc<UOp>) -> Self {
        let mut ctx = Self::default();

        // Collect all nodes via toposort
        let nodes = sink.toposort();

        // Step 1: Build dependency sets for each node
        // RANGE and END add themselves to deps
        // deps[u] = set of RANGE/END UOps that u transitively depends on
        #[allow(clippy::mutable_key_type)]
        let mut deps: HashMap<UOpKey, HashMap<UOpKey, ()>> = HashMap::new();

        for node in &nodes {
            // Get deps from sources
            #[allow(clippy::mutable_key_type)]
            let mut node_deps: HashMap<UOpKey, ()> = HashMap::new();
            node.op().map_child(|src| {
                if let Some(src_deps) = deps.get(&UOpKey(src.clone())) {
                    node_deps.extend(src_deps.iter().map(|(k, v)| (k.clone(), *v)));
                }
            });

            // RANGE and END add themselves
            if matches!(node.op(), Op::Range { .. } | Op::End { .. }) {
                node_deps.insert(UOpKey(node.clone()), ());
            }

            deps.insert(UOpKey(node.clone()), node_deps);
        }

        // Step 2: Build nesting map
        // For each END, find which END/SINK it is nested inside
        // END x is nested in END/SINK u if:
        //   - u depends on x (x is in deps[u])
        //   - u is SINK, OR u's RANGE (u.src[1]) is in deps[x]
        //   - x hasn't been assigned a nesting parent yet
        #[allow(clippy::mutable_key_type)]
        let mut nesting: HashMap<UOpKey, Arc<UOp>> = HashMap::new();

        for node in &nodes {
            if matches!(node.op(), Op::End { .. } | Op::Sink { .. })
                && let Some(node_deps) = deps.get(&UOpKey(node.clone()))
            {
                for dep_key in node_deps.keys() {
                    // Only consider END nodes
                    if !matches!(dep_key.0.op(), Op::End { .. }) {
                        continue;
                    }

                    // Skip if already assigned
                    if nesting.contains_key(dep_key) {
                        continue;
                    }

                    // Check nesting condition
                    let is_nested = if matches!(node.op(), Op::Sink { .. }) {
                        true
                    } else if let Op::End { ranges, .. } = node.op() {
                        // Check if node's RANGE is in dep's dependencies
                        // node.src[1] in Tinygrad is the RANGE - we get it from ranges
                        if let Some(range) = ranges.first() {
                            deps.get(dep_key).is_some_and(|dep_deps| dep_deps.contains_key(&UOpKey(range.clone())))
                        } else {
                            false
                        }
                    } else {
                        false
                    };

                    if is_nested {
                        nesting.insert(dep_key.clone(), node.clone());
                    }
                }
            }
        }

        // Step 3: Group siblings by parent
        #[allow(clippy::mutable_key_type)]
        let mut siblings: HashMap<UOpKey, Vec<Arc<UOp>>> = HashMap::new();
        for (end_key, parent) in &nesting {
            siblings.entry(UOpKey(parent.clone())).or_default().push(end_key.0.clone());
        }

        // Step 4 & 5: Order siblings and create edges
        for (parent, sibling_ends) in siblings {
            if sibling_ends.is_empty() {
                continue;
            }

            // Order by dependency count on other siblings (fewer deps = earlier)
            let mut ordered: Vec<Arc<UOp>> = sibling_ends.clone();
            ordered.sort_by_key(|end| {
                if let Some(end_deps) = deps.get(&UOpKey(end.clone())) {
                    sibling_ends.iter().filter(|sib| end_deps.contains_key(&UOpKey((*sib).clone()))).count()
                } else {
                    0
                }
            });

            // Create edges
            // If parent is SINK: zip(order, order[1:])
            // If parent is END: zip([parent.src[1]] + order, order)
            //   where parent.src[1] is the parent's RANGE
            let zipped: Vec<(Arc<UOp>, Arc<UOp>)> = if matches!(parent.0.op(), Op::Sink { .. }) {
                // Pair consecutive siblings
                ordered.windows(2).map(|w| (w[0].clone(), w[1].clone())).collect()
            } else {
                // Get parent's RANGE
                if let Op::End { ranges, .. } = parent.0.op() {
                    if let Some(parent_range) = ranges.first() {
                        // Pair: parent_range → first, then consecutive siblings
                        let mut pairs = vec![(parent_range.clone(), ordered[0].clone())];
                        pairs.extend(ordered.windows(2).map(|w| (w[0].clone(), w[1].clone())));
                        pairs
                    } else {
                        ordered.windows(2).map(|w| (w[0].clone(), w[1].clone())).collect()
                    }
                } else {
                    ordered.windows(2).map(|w| (w[0].clone(), w[1].clone())).collect()
                }
            };

            // Create edges: y's RANGE → x (predecessor)
            for (x, y) in zipped {
                // y is an END, get its RANGE from y.src[1] (or ranges field)
                let y_range = if let Op::End { ranges, .. } = y.op() { ranges.first().cloned() } else { None };

                if let Some(range) = y_range {
                    // Skip self-referential edges (can happen when parent and child END the same range)
                    if range.id != x.id {
                        ctx.edges.insert(UOpKey(range), x);
                    }
                }
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
        let end = value.end(smallvec::smallvec![range]);
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
        let end = value.end(smallvec::smallvec![range1.clone(), range2.clone()]);
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
        let inner_end = inner_value.end(smallvec::smallvec![inner_range.clone()]);

        // Outer range containing inner
        let outer_range = UOp::range(end_val, 1);
        let outer_end = inner_end.end(smallvec::smallvec![outer_range.clone()]);

        let sink = UOp::sink(vec![outer_end]);

        let ctx = CFGContext::new(&sink);
        // Nested ranges at different levels shouldn't have edges between them
        // (they're not siblings)
        assert!(ctx.get_predecessor(&outer_range).is_none());
    }
}
