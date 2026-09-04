//! Memoized existential queries over backward slices.
//!
//! A one-shot `any_in_subtree` is O(N) per root; passes that ask the same
//! question of many roots (every buffer, every index, every clause) would pay
//! that per root. [`SubtreeMemo`] keeps one memo table across roots so the
//! whole pass costs O(N) in total, without the per-node slice sets that made
//! the answer O(N²) in memory.

use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::uop::UOp;

/// "Does any node in this subtree satisfy `hit`?", memoized per node id.
///
/// The predicate must be a pure function of the node: results are cached and
/// reused for every later root that reaches the same node. UOps are immutable
/// and hash-consed, so a node id identifies its subtree for the process
/// lifetime.
pub struct SubtreeMemo<F> {
    hit: F,
    memo: FxHashMap<u64, bool>,
}

impl<F: Fn(&Arc<UOp>) -> bool> SubtreeMemo<F> {
    pub fn new(hit: F) -> Self {
        Self { hit, memo: FxHashMap::default() }
    }

    /// True iff `root` or any node it depends on satisfies the predicate.
    pub fn contains(&mut self, root: &Arc<UOp>) -> bool {
        if let Some(&cached) = self.memo.get(&root.id) {
            return cached;
        }

        // Explicit stack: kernel graphs are deep enough to blow the real one.
        // `expanded` marks the post-order visit, when every child is memoized.
        let mut stack = vec![(root.clone(), false)];
        while let Some((node, expanded)) = stack.pop() {
            if self.memo.contains_key(&node.id) {
                continue;
            }
            if (self.hit)(&node) {
                self.memo.insert(node.id, true);
                continue;
            }
            if expanded {
                let mut found = false;
                node.op().map_child(|child| found |= self.memo[&child.id]);
                self.memo.insert(node.id, found);
            } else {
                stack.push((node.clone(), true));
                let memo = &self.memo;
                node.op().map_child(|child| {
                    if !memo.contains_key(&child.id) {
                        stack.push((child.clone(), false));
                    }
                });
            }
        }

        self.memo[&root.id]
    }
}

/// Memoized "is `target` reachable from this node", counting the node itself.
pub fn reaching(target: &Arc<UOp>) -> SubtreeMemo<impl Fn(&Arc<UOp>) -> bool + use<>> {
    let target = target.id;
    SubtreeMemo::new(move |node| node.id == target)
}

#[cfg(test)]
#[path = "../test/unit/uop/reachability.rs"]
mod tests;
