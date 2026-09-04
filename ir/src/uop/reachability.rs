//! Memoized gated backward-slice queries.
//!
//! One post-order DFS, memoized per node id, answers both "does any node in
//! this slice satisfy `select`?" ([`SliceMemo<bool>`], alias [`SubtreeMemo`])
//! and "which nodes do?" ([`SliceMemo<Nodes>`]). A gated node is opaque: it is
//! neither selected nor descended into, as in `UOp::toposort_filtered`.
//!
//! The memo table is shared across roots, so a pass that asks the same
//! question of many roots (every buffer, every index, every clause) costs O(N)
//! in total instead of O(N) per root. UOps are immutable and hash-consed, so a
//! node id identifies its slice for the process lifetime; the predicates must
//! be pure functions of the node.

use std::sync::{Arc, LazyLock};

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::uop::UOp;

pub type Pred = fn(&Arc<UOp>) -> bool;
pub type Nodes = Arc<[Arc<UOp>]>;
pub type SubtreeMemo<S = Pred, G = Pred> = SliceMemo<bool, S, G>;

/// A memo entry: what a node's gated slice folds down to.
pub trait Slice: Clone {
    /// Entry of a gated node.
    fn none() -> Self;
    /// Entry fixed by the node alone; `Some` skips its children.
    fn short(node: &Arc<UOp>, select: impl Fn(&Arc<UOp>) -> bool) -> Option<Self>;
    /// Entry of a node whose children are all memoized.
    fn merge(node: &Arc<UOp>, select: impl Fn(&Arc<UOp>) -> bool, memo: &FxHashMap<u64, Self>) -> Self;
}

impl Slice for bool {
    fn none() -> Self {
        false
    }

    fn short(node: &Arc<UOp>, select: impl Fn(&Arc<UOp>) -> bool) -> Option<Self> {
        select(node).then_some(true)
    }

    fn merge(node: &Arc<UOp>, _: impl Fn(&Arc<UOp>) -> bool, memo: &FxHashMap<u64, Self>) -> Self {
        let mut any = false;
        node.op().map_child(|child| any |= memo[&child.id]);
        any
    }
}

impl Slice for Nodes {
    fn none() -> Self {
        static EMPTY: LazyLock<Nodes> = LazyLock::new(|| Vec::new().into());
        EMPTY.clone()
    }

    fn short(_: &Arc<UOp>, _: impl Fn(&Arc<UOp>) -> bool) -> Option<Self> {
        None
    }

    /// Sorted by id: the result is a set, its order carries no meaning. A node
    /// with one non-empty child part shares that part's allocation.
    fn merge(node: &Arc<UOp>, select: impl Fn(&Arc<UOp>) -> bool, memo: &FxHashMap<u64, Self>) -> Self {
        let mut parts: SmallVec<[&Nodes; 4]> = SmallVec::new();
        node.op().map_child(|child| {
            let nodes = &memo[&child.id];
            if !nodes.is_empty() && !parts.iter().any(|part| Arc::ptr_eq(part, nodes)) {
                parts.push(nodes);
            }
        });
        let selected = select(node);
        match parts.as_slice() {
            [] if !selected => Self::none(),
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

/// Selected nodes reachable from a root through gate-passing nodes, root
/// included, memoized per node id and merged bottom-up from the children's
/// entries. Equivalent to `toposort_filtered(gate)` followed by
/// `filter(select)` (or `any(select)` for `V = bool`).
pub struct SliceMemo<V, S = Pred, G = Pred> {
    select: S,
    gate: G,
    memo: FxHashMap<u64, V>,
}

impl<V: Slice, S: Fn(&Arc<UOp>) -> bool> SliceMemo<V, S> {
    /// Ungated: every node of the backward slice is visible.
    pub fn new(select: S) -> Self {
        Self::gated(select, |_| true)
    }
}

impl<V: Slice, S: Fn(&Arc<UOp>) -> bool, G: Fn(&Arc<UOp>) -> bool> SliceMemo<V, S, G> {
    pub fn gated(select: S, gate: G) -> Self {
        Self { select, gate, memo: FxHashMap::default() }
    }

    pub fn get(&mut self, root: &Arc<UOp>) -> V {
        if let Some(hit) = self.memo.get(&root.id) {
            return hit.clone();
        }

        // Explicit stack: kernel graphs are deep enough to blow the real one.
        // `expanded` marks the post-order visit, when every child is memoized.
        let mut stack = vec![(root.clone(), false)];
        while let Some((node, expanded)) = stack.pop() {
            if self.memo.contains_key(&node.id) {
                continue;
            }
            let entry = if !(self.gate)(&node) {
                V::none()
            } else if expanded {
                V::merge(&node, &self.select, &self.memo)
            } else if let Some(entry) = V::short(&node, &self.select) {
                entry
            } else {
                stack.push((node.clone(), true));
                let memo = &self.memo;
                node.op().map_child(|child| {
                    if !memo.contains_key(&child.id) {
                        stack.push((child.clone(), false));
                    }
                });
                continue;
            };
            self.memo.insert(node.id, entry);
        }

        self.memo[&root.id].clone()
    }
}

impl<S: Fn(&Arc<UOp>) -> bool, G: Fn(&Arc<UOp>) -> bool> SliceMemo<bool, S, G> {
    /// True iff `root` or any node it reaches through gate-passing nodes is selected.
    pub fn contains(&mut self, root: &Arc<UOp>) -> bool {
        self.get(root)
    }
}

/// Memoized "is `target` reachable from this node", counting the node itself.
pub fn reaching(target: &Arc<UOp>) -> SubtreeMemo<impl Fn(&Arc<UOp>) -> bool + use<>> {
    let target = target.id;
    SliceMemo::new(move |node| node.id == target)
}

/// Memoized "which of `targets` are reachable from this node", counting the
/// node itself: one walk serves a query per target.
pub fn reaching_each(targets: &[Arc<UOp>]) -> SliceMemo<Nodes, impl Fn(&Arc<UOp>) -> bool + use<>> {
    let ids: Vec<u64> = targets.iter().map(|target| target.id).collect();
    SliceMemo::new(move |node| ids.contains(&node.id))
}

#[cfg(test)]
#[path = "../test/unit/uop/reachability.rs"]
mod tests;
