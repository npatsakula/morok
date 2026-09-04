//! Per-pass memo of the selected nodes in each node's backward slice.

use std::sync::Arc;

use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use svod_ir::UOp;

type Nodes = Arc<[Arc<UOp>]>;

/// Selected nodes reachable from a node through gate-passing nodes, self included.
///
/// Equivalent to `toposort_filtered(gate)` followed by `filter(select)`, but merged
/// bottom-up from the children's entries: a pattern that would toposort every
/// matched node's slice pays O(N) once per pass instead of once per match.
pub struct SliceMemo {
    select: fn(&Arc<UOp>) -> bool,
    gate: fn(&Arc<UOp>) -> bool,
    memo: FxHashMap<u64, Nodes>,
    empty: Nodes,
}

impl SliceMemo {
    pub fn new(select: fn(&Arc<UOp>) -> bool, gate: fn(&Arc<UOp>) -> bool) -> Self {
        Self { select, gate, memo: FxHashMap::default(), empty: Vec::new().into() }
    }

    pub fn ungated(select: fn(&Arc<UOp>) -> bool) -> Self {
        Self::new(select, |_| true)
    }

    /// Sorted by id: the result is a set, its order carries no meaning.
    pub fn get(&mut self, root: &Arc<UOp>) -> Nodes {
        if let Some(hit) = self.memo.get(&root.id) {
            return hit.clone();
        }
        let mut stack = vec![(root.clone(), false)];
        while let Some((node, expanded)) = stack.pop() {
            if self.memo.contains_key(&node.id) {
                continue;
            }
            if !(self.gate)(&node) {
                self.memo.insert(node.id, self.empty.clone());
            } else if expanded {
                let nodes = self.merge(&node);
                self.memo.insert(node.id, nodes);
            } else {
                stack.push((node.clone(), true));
                node.op().map_child(|child| {
                    if !self.memo.contains_key(&child.id) {
                        stack.push((child.clone(), false));
                    }
                });
            }
        }
        self.memo[&root.id].clone()
    }

    fn merge(&self, node: &Arc<UOp>) -> Nodes {
        let mut parts: SmallVec<[&Nodes; 4]> = SmallVec::new();
        node.op().map_child(|child| {
            let nodes = &self.memo[&child.id];
            if !nodes.is_empty() && !parts.iter().any(|part| Arc::ptr_eq(part, nodes)) {
                parts.push(nodes);
            }
        });
        let selected = (self.select)(node);
        match parts.as_slice() {
            [] if !selected => self.empty.clone(),
            [only] if !selected => Arc::clone(only),
            _ => {
                let mut merged: Vec<Arc<UOp>> = parts.iter().flat_map(|part| part.iter().cloned()).collect();
                merged.extend(selected.then(|| node.clone()));
                merged.sort_unstable_by_key(|uop| uop.id);
                merged.dedup_by_key(|uop| uop.id);
                merged.into()
            }
        }
    }
}
