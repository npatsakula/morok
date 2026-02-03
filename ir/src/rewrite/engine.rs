//! Graph rewrite engine implementation.
//!
//! Implements the core graph rewriting algorithm with fixed-point iteration.
//!
//! # Algorithm (Tinygrad-aligned)
//!
//! The algorithm operates in 3 stages, matching Tinygrad's `unified_rewrite`:
//!
//! - **Stage 0 (PushChildren)**: Apply `bpm` patterns (if present), then push children
//!   - `bpm` patterns see **ORIGINAL** children
//!   - Used for bottom-up patterns that need to transform before descent
//!
//! - **Stage 1 (ApplyPatterns)**: Reconstruct with optimized children, then apply `pm` patterns
//!   - `pm` patterns see **OPTIMIZED** children
//!   - This is the default mode - patterns run after children are processed
//!
//! - **Stage 2 (Link)**: Link original node to final result
//!
//! # API
//!
//! - `graph_rewrite(pm, root, ctx)` - Default: patterns see optimized children (Stage 1)
//! - `graph_rewrite_bottom_up(bpm, root, ctx)` - Patterns see original children (Stage 0)
//!
//! # Pattern Context
//!
//! Context is passed at rewrite-time through `graph_rewrite()`, not captured in
//! closures. This provides compile-time type safety without `Rc<RefCell<>>`
//! boilerplate.
//!
//! ## Example
//!
//! ```ignore
//! use morok_ir::pattern::SimplifiedPatternMatcher;
//!
//! // Create context
//! let mut ctx = KernelContext::new();
//!
//! // Create matcher using the patterns! macro
//! let matcher = patterns! {
//!     Add(x, @zero) ~> |x| x.clone(),
//!     Mul(x, @one) ~> |x| x.clone(),
//! };
//!
//! // Pass context at rewrite time - patterns see OPTIMIZED children
//! let result = graph_rewrite(&matcher, root, &mut ctx);
//! ```
//!
//! Patterns that don't need context use `()` as the context type:
//!
//! ```ignore
//! let matcher = patterns! {
//!     Add(x, @zero) ~> |x| x.clone(),
//! };
//! let result = graph_rewrite(&matcher, root, &mut ());
//! ```

use crate::{UOp, UOpKey};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::pattern::{Matcher, RewriteResult};

/// Stage in the 3-stage rewrite algorithm (Tinygrad-aligned).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stage {
    /// Stage 0: Apply bpm patterns (if present), then push children for processing.
    /// bpm patterns see ORIGINAL children.
    PushChildren,
    /// Stage 1: After children processed, reconstruct and apply pm patterns.
    /// pm patterns see OPTIMIZED children.
    ApplyPatterns,
    /// Stage 2: Link original node to final result.
    Link,
}

/// Stack entry for the 3-stage rewrite algorithm.
///
/// The separation of `original` and `working` is crucial for result linking:
/// - `original`: The node that consumers reference (used as key in results cache)
/// - `working`: The node after pattern rewrites (may differ from original)
#[derive(Debug, Clone)]
struct StackEntry {
    /// The node that consumers reference - used as key in results cache
    original: Arc<UOp>,
    /// Current processing stage
    stage: Stage,
    /// The node we're actively working with (may differ after rewrites)
    working: Arc<UOp>,
    /// Retry count for ApplyPatterns stage (to detect infinite loops)
    retry_count: u32,
}

impl StackEntry {
    /// Create entry for a fresh node starting at PushChildren stage.
    fn new(node: Arc<UOp>) -> Self {
        Self { original: node.clone(), stage: Stage::PushChildren, working: node, retry_count: 0 }
    }

    /// Create entry for PushChildren stage where original == working.
    fn push_children(node: Arc<UOp>) -> Self {
        Self { original: node.clone(), stage: Stage::PushChildren, working: node, retry_count: 0 }
    }

    /// Create entry for ApplyPatterns stage with potentially different working node.
    fn apply_patterns(original: Arc<UOp>, working: Arc<UOp>) -> Self {
        Self { original, stage: Stage::ApplyPatterns, working, retry_count: 0 }
    }

    /// Create entry for ApplyPatterns stage with retry count (for re-pushed entries).
    fn apply_patterns_retry(original: Arc<UOp>, working: Arc<UOp>, retry_count: u32) -> Self {
        Self { original, stage: Stage::ApplyPatterns, working, retry_count }
    }

    /// Create entry for Link stage.
    fn link(original: Arc<UOp>, working: Arc<UOp>) -> Self {
        Self { original, stage: Stage::Link, working, retry_count: 0 }
    }
}

/// Efficient result lookup using path compression.
///
/// Provides O(α(n)) amortized lookup via path compression, where α is the
/// inverse Ackermann function (practically constant). This is much faster
/// than the O(chain_length) traversal of naive chain following.
#[derive(Default)]
struct ResultMap {
    /// Maps each node to its final result (with path compression on lookup)
    results: HashMap<UOpKey, Arc<UOp>>,
}

impl ResultMap {
    /// Get the final result for a node with path compression.
    ///
    /// Follows the chain of rewrites and compresses the path so future
    /// lookups are O(1). Returns the node itself if no result is cached.
    fn get(&mut self, node: &Arc<UOp>) -> Arc<UOp> {
        let key = UOpKey(node.clone());

        // Fast path: no result cached
        let Some(result) = self.results.get(&key).cloned() else {
            return node.clone();
        };

        // If result points to self, we're done
        if Arc::ptr_eq(&result, node) {
            return result;
        }

        // Follow chain with path compression
        let mut current = result;
        let mut path = vec![key.clone()];

        const MAX_DEPTH: usize = 100;
        for _ in 0..MAX_DEPTH {
            let current_key = UOpKey(current.clone());
            match self.results.get(&current_key) {
                Some(next) if !Arc::ptr_eq(next, &current) => {
                    path.push(current_key);
                    current = next.clone();
                }
                _ => break,
            }
        }

        // Path compression: all nodes in path now point directly to final result
        for k in path {
            self.results.insert(k, current.clone());
        }

        current
    }

    /// Link original node to its result.
    fn link(&mut self, original: Arc<UOp>, result: Arc<UOp>) {
        self.results.insert(UOpKey(original), result);
    }

    /// Check if node has a result (without path compression).
    fn contains(&self, node: &Arc<UOp>) -> bool {
        self.results.contains_key(&UOpKey(node.clone()))
    }

    /// Get result from raw HashMap (for early exit check).
    fn get_direct(&self, key: &UOpKey) -> Option<Arc<UOp>> {
        self.results.get(key).cloned()
    }
}

/// Internal rewrite engine that implements the 3-stage stack-based algorithm.
///
/// Generic over matcher types and context type for compile-time type-safe matching.
/// Supports separate `pm` (top-down) and `bpm` (bottom-up) matchers, matching Tinygrad's
/// `RewriteContext(pm, bpm, ctx)` structure.
#[allow(clippy::mutable_key_type)]
struct RewriteEngine<'a, PM, BPM, C>
where
    PM: Matcher<C>,
    BPM: Matcher<C>,
{
    /// Top-down pattern matcher: applied in Stage 1 (ApplyPatterns)
    /// Patterns see OPTIMIZED children
    pm: Option<&'a PM>,

    /// Bottom-up pattern matcher: applied in Stage 0 (PushChildren)
    /// Patterns see ORIGINAL children
    bpm: Option<&'a BPM>,

    /// Mutable reference to context passed through to patterns
    ctx: &'a mut C,

    /// Final results cache: maps original node → optimized result
    /// Uses path compression for O(α(n)) amortized lookups
    results: ResultMap,

    /// Nodes pending processing (prevents duplicate pushes in DAGs).
    /// A node is in this set if it's currently on the stack or being processed.
    pending: HashSet<UOpKey>,
}

impl<'a, PM, BPM, C> RewriteEngine<'a, PM, BPM, C>
where
    PM: Matcher<C>,
    BPM: Matcher<C>,
{
    fn new(pm: Option<&'a PM>, bpm: Option<&'a BPM>, ctx: &'a mut C) -> Self {
        Self { pm, bpm, ctx, results: ResultMap::default(), pending: HashSet::default() }
    }
}

impl<'a, PM, BPM, C> RewriteEngine<'a, PM, BPM, C>
where
    PM: Matcher<C>,
    BPM: Matcher<C>,
{
    /// Stage 0 (PushChildren): Apply bpm patterns (if present), then push children.
    ///
    /// This matches Tinygrad's unified_rewrite Stage 0:
    /// - If bpm is present, apply patterns in fixed-point (sees ORIGINAL children)
    /// - Push children for processing
    /// - Schedule Stage 1 (ApplyPatterns) for after children are done
    fn handle_push_children(&mut self, stack: &mut Vec<StackEntry>, original: Arc<UOp>, working: Arc<UOp>) {
        let mut node = working;

        // Apply bpm patterns if present (sees ORIGINAL children)
        // Tinygrad: "if bottom up, we rewrite this node early"
        if let Some(bpm) = &self.bpm {
            const MAX_ITERATIONS: usize = 1000;
            for i in 0..MAX_ITERATIONS {
                match bpm.rewrite(&node, self.ctx) {
                    RewriteResult::Rewritten(new_node) => {
                        node = new_node;
                    }
                    RewriteResult::Gate(_) => {
                        // Gate in bpm means "stop descent, result is ready"
                        // Tinygrad: "if the bpm matching raised a gate, we are done with this node"
                        self.link_result(original, node);
                        return;
                    }
                    RewriteResult::NoMatch => {
                        break;
                    }
                }
                if i == MAX_ITERATIONS - 1 {
                    panic!(
                        "BPM rewrite iteration limit ({}) exceeded: patterns may be creating an infinite loop. Node: {:?}",
                        MAX_ITERATIONS,
                        node.op()
                    );
                }
            }
        }

        // Schedule Stage 1 (ApplyPatterns) for after children are processed
        stack.push(StackEntry::apply_patterns(original, node.clone()));

        // Push children for processing (always, matching Tinygrad's unified_rewrite)
        let sources = node.op().sources();
        for child in sources.iter().rev() {
            let child_key = UOpKey(child.clone());
            if !self.pending.contains(&child_key) && !self.results.contains(child) {
                self.pending.insert(child_key);
                stack.push(StackEntry::push_children(child.clone()));
            }
        }
    }

    /// Stage 1 (ApplyPatterns): Reconstruct with optimized children, then apply pm patterns.
    ///
    /// This matches Tinygrad's unified_rewrite Stage 1:
    /// - Wait for all children to be processed
    /// - Reconstruct node if any children changed
    /// - Apply pm patterns (sees OPTIMIZED children)
    /// - If patterns produce new node, process it through Stage 0
    fn handle_apply_patterns(
        &mut self,
        stack: &mut Vec<StackEntry>,
        original: Arc<UOp>,
        working: Arc<UOp>,
        retry_count: u32,
    ) {
        let sources = working.op().sources();

        // For leaf nodes, apply pm patterns directly
        if sources.is_empty() {
            let final_node = self.apply_pm_patterns(&working);
            if Arc::ptr_eq(&final_node, &working) {
                // No change - link directly
                self.link_result(original, working);
            } else {
                // Pattern produced new node - process it
                let key = UOpKey(final_node.clone());
                if !self.results.contains(&final_node) && !self.pending.contains(&key) {
                    stack.push(StackEntry::link(original, final_node.clone()));
                    self.pending.insert(key);
                    stack.push(StackEntry::push_children(final_node));
                } else {
                    // Already processed - link to its result
                    let result = self.results.get(&final_node);
                    self.link_result(original, result);
                }
            }
            return;
        }

        // Check if all sources have been fully processed
        let mut needs_defer = false;
        for src in &sources {
            // Skip self-references (pattern created wrapper containing original)
            if Arc::ptr_eq(src, &original) {
                continue;
            }

            if !self.results.contains(src) {
                let src_key = UOpKey(src.clone());
                if self.pending.contains(&src_key) {
                    needs_defer = true;
                    break;
                }
            }
        }

        if needs_defer {
            const MAX_RETRIES: u32 = 10_000;
            if retry_count >= MAX_RETRIES {
                panic!("ApplyPatterns stuck waiting for sources after {} retries: {:?}", MAX_RETRIES, working.tree());
            }
            stack.insert(0, StackEntry::apply_patterns_retry(original, working, retry_count + 1));
            return;
        }

        // Collect optimized children
        let mut new_sources = Vec::with_capacity(sources.len());
        let mut any_changed = false;

        for src in &sources {
            let optimized = self.results.get(src);
            if !Arc::ptr_eq(&optimized, src) {
                any_changed = true;
            }
            new_sources.push(optimized);
        }

        // Reconstruct if children changed
        let node = if any_changed { working.with_sources(new_sources) } else { working };

        // Apply pm patterns (sees OPTIMIZED children!)
        // This is the key semantic difference from the old implementation
        let final_node = self.apply_pm_patterns(&node);

        // If pattern produced new node, process it through Stage 0
        if !Arc::ptr_eq(&final_node, &node) {
            let key = UOpKey(final_node.clone());
            if !self.results.contains(&final_node) && !self.pending.contains(&key) {
                // Schedule: Link after processing the new node
                stack.push(StackEntry::link(original, final_node.clone()));
                self.pending.insert(key);
                stack.push(StackEntry::push_children(final_node));
                return;
            }
            // New node already processed - link to its result
            let result = self.results.get(&final_node);
            self.link_result(original, result);
            return;
        }

        // If reconstruction created new node, check if already processed
        if any_changed {
            let recon_key = UOpKey(node.clone());
            if !self.results.contains(&node) && !self.pending.contains(&recon_key) {
                // Reconstructed node not seen - process it
                stack.push(StackEntry::link(original, node.clone()));
                self.pending.insert(recon_key);
                stack.push(StackEntry::push_children(node));
                return;
            }
            // Already processed - link to its result
            let result = self.results.get(&node);
            self.link_result(original, result);
            return;
        }

        // No changes - link original to working
        self.link_result(original, node);
    }

    /// Stage 2 (Link): Link original node to the result of working node.
    fn handle_link(&mut self, original: Arc<UOp>, working: Arc<UOp>) {
        let result = self.results.get(&working);
        self.link_result(original, result);
    }

    /// Apply pm patterns in fixed-point loop.
    fn apply_pm_patterns(&mut self, node: &Arc<UOp>) -> Arc<UOp> {
        let Some(pm) = &self.pm else {
            return node.clone();
        };

        const MAX_ITERATIONS: usize = 1000;
        let mut current = node.clone();

        for i in 0..MAX_ITERATIONS {
            match pm.rewrite(&current, self.ctx) {
                RewriteResult::Rewritten(new_node) => {
                    current = new_node;
                }
                RewriteResult::Gate(_) | RewriteResult::NoMatch => {
                    break;
                }
            }
            if i == MAX_ITERATIONS - 1 {
                panic!(
                    "PM rewrite iteration limit ({}) exceeded: patterns may be creating an infinite loop. Node: {:?}",
                    MAX_ITERATIONS,
                    current.op()
                );
            }
        }

        current
    }

    /// Link original node to its final result in the cache.
    fn link_result(&mut self, original: Arc<UOp>, result: Arc<UOp>) {
        // Remove from pending set - this node is now fully processed
        self.pending.remove(&UOpKey(original.clone()));

        // Record provenance if actually rewritten
        if !Arc::ptr_eq(&original, &result) {
            use crate::provenance::{PROVENANCE_TRACKER, PassName};
            PROVENANCE_TRACKER.with(|tracker| {
                tracker.borrow_mut().record_transform(result.id, original.id, PassName::RewritePattern);
            });
        }

        self.results.link(original, result);
    }

    /// Main rewrite method: Stack-based 3-stage traversal.
    ///
    /// Traverses the graph in a stack-based manner, processing each node
    /// through 3 stages (matching Tinygrad's unified_rewrite):
    /// 1. Stage 0 (PushChildren): Apply bpm patterns, push children
    /// 2. Stage 1 (ApplyPatterns): Reconstruct, apply pm patterns
    /// 3. Stage 2 (Link): Link original to final result
    fn rewrite(&mut self, root: Arc<UOp>) -> Arc<UOp> {
        let root_key = UOpKey(root.clone());

        // Early exit if already processed
        if let Some(result) = self.results.get_direct(&root_key) {
            return result;
        }

        self.pending.insert(root_key.clone());
        let mut stack: Vec<StackEntry> = vec![StackEntry::new(root.clone())];

        // Limit total iterations to catch infinite loops
        const MAX_TOTAL_ITERATIONS: usize = 100_000;
        let mut iterations = 0;

        while let Some(StackEntry { original, stage, working, retry_count }) = stack.pop() {
            iterations += 1;
            if iterations > MAX_TOTAL_ITERATIONS {
                panic!(
                    "Rewrite total iteration limit ({MAX_TOTAL_ITERATIONS}) exceeded: likely infinite loop. Stack size: {}, results cached: {}, original: {}, working: {}",
                    stack.len(),
                    self.results.results.len(),
                    original.tree(),
                    working.tree(),
                );
            }

            // Skip if original already has result
            if self.results.contains(&original) {
                continue;
            }

            match stage {
                Stage::PushChildren => self.handle_push_children(&mut stack, original, working),
                Stage::ApplyPatterns => self.handle_apply_patterns(&mut stack, original, working, retry_count),
                Stage::Link => self.handle_link(original, working),
            }
        }

        self.results.get_direct(&root_key).unwrap_or(root)
    }
}

/// Marker type for "no matcher" in generic contexts.
///
/// Used when only pm or only bpm is needed.
pub struct NoMatcher;

impl<C> Matcher<C> for NoMatcher {
    fn rewrite(&self, _node: &Arc<UOp>, _ctx: &mut C) -> RewriteResult {
        RewriteResult::NoMatch
    }
}

/// Apply graph rewriting to a UOp graph using the given pattern matcher.
///
/// This is the main entry point for graph rewriting. Patterns see **OPTIMIZED**
/// children (applied in Stage 1 after children are processed).
///
/// Matches Tinygrad's `graph_rewrite(sink, pm, ctx, bottom_up=False)`.
pub fn graph_rewrite<M: Matcher<C>, C>(matcher: &M, root: Arc<UOp>, ctx: &mut C) -> Arc<UOp> {
    RewriteEngine::new(Some(matcher), None::<&NoMatcher>, ctx).rewrite(root)
}

/// Apply graph rewriting with bottom-up pattern application.
///
/// Patterns see **ORIGINAL** children (applied in Stage 0 before children are processed).
/// Use this for patterns that need to transform nodes before their children are optimized.
///
/// Matches Tinygrad's `graph_rewrite(sink, pm, ctx, bottom_up=True)`.
pub fn graph_rewrite_bottom_up<M: Matcher<C>, C>(matcher: &M, root: Arc<UOp>, ctx: &mut C) -> Arc<UOp> {
    RewriteEngine::new(None::<&NoMatcher>, Some(matcher), ctx).rewrite(root)
}

/// Apply graph rewriting with both top-down and bottom-up patterns.
///
/// - `bpm` patterns see ORIGINAL children (Stage 0)
/// - `pm` patterns see OPTIMIZED children (Stage 1)
///
/// Matches Tinygrad's `graph_rewrite(sink, pm, ctx, bpm=bpm)`.
pub fn graph_rewrite_with_bpm<PM, BPM, C>(pm: &PM, bpm: &BPM, root: Arc<UOp>, ctx: &mut C) -> Arc<UOp>
where
    PM: Matcher<C>,
    BPM: Matcher<C>,
{
    RewriteEngine::new(Some(pm), Some(bpm), ctx).rewrite(root)
}

// Backward compatibility aliases
#[doc(hidden)]
#[deprecated(since = "0.2.0", note = "Use graph_rewrite instead")]
pub fn graph_rewrite_top_down<M: Matcher<C>, C>(matcher: &M, root: Arc<UOp>, ctx: &mut C) -> Arc<UOp> {
    graph_rewrite(matcher, root, ctx)
}

/// Result of graph rewriting including the transformation map.
///
/// The `becomes_map` maps original UOp nodes to their transformed versions.
/// This is useful for global substitution in systems like Tinygrad's
/// `_apply_map_to_tensors()`.
pub struct GraphRewriteOutput {
    /// The rewritten root node
    pub root: Arc<UOp>,
    /// Map from original nodes to their replacements
    pub becomes_map: HashMap<UOpKey, Arc<UOp>>,
}

/// Apply graph rewriting and return both the result and the transformation map.
///
/// Like `graph_rewrite`, but also returns a `becomes_map` that tracks which
/// original nodes were transformed to which new nodes. This is essential for
/// global graph coordination where multiple tensors share subgraphs.
#[allow(clippy::mutable_key_type)]
pub fn graph_rewrite_with_map<M, C>(matcher: &M, root: Arc<UOp>, ctx: &mut C) -> GraphRewriteOutput
where
    M: Matcher<C>,
{
    let mut engine = RewriteEngine::new(Some(matcher), None::<&NoMatcher>, ctx);
    let result_root = engine.rewrite(root.clone());
    // Extract becomes_map: only include entries where the result differs from original
    let becomes_map = engine.results.results.into_iter().filter(|(k, v)| !Arc::ptr_eq(&k.0, v)).collect();
    GraphRewriteOutput { root: result_root, becomes_map }
}

/// Apply bottom-up graph rewriting and return both the result and transformation map.
///
/// Like `graph_rewrite_bottom_up`, but also returns a `becomes_map`.
#[allow(clippy::mutable_key_type)]
pub fn graph_rewrite_bottom_up_with_map<M, C>(matcher: &M, root: Arc<UOp>, ctx: &mut C) -> GraphRewriteOutput
where
    M: Matcher<C>,
{
    let mut engine = RewriteEngine::new(None::<&NoMatcher>, Some(matcher), ctx);
    let result_root = engine.rewrite(root.clone());
    // Extract becomes_map: only include entries where the result differs from original
    let becomes_map = engine.results.results.into_iter().filter(|(k, v)| !Arc::ptr_eq(&k.0, v)).collect();
    GraphRewriteOutput { root: result_root, becomes_map }
}

// Backward compatibility aliases for _with_map functions
#[doc(hidden)]
#[deprecated(since = "0.2.0", note = "Use graph_rewrite_with_map instead")]
#[allow(clippy::mutable_key_type)]
pub fn graph_rewrite_top_down_with_map<M, C>(matcher: &M, root: Arc<UOp>, ctx: &mut C) -> GraphRewriteOutput
where
    M: Matcher<C>,
{
    graph_rewrite_with_map(matcher, root, ctx)
}
