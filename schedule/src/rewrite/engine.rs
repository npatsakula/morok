//! Graph rewrite engine implementation.
//!
//! Implements the core graph rewriting algorithm with fixed-point iteration.
//!
//! # Algorithm
//!
//! The algorithm operates in 2 stages:
//! - Stage 0 (Rewrite): Fixed-point pattern matching on the current node
//! - Stage 1 (Finalize): Reconstruct with children's results + link result
//!
//! By default, patterns see the **original** children (top-down style). If a
//! pattern needs optimized children first (bottom-up style), it should return
//! `RewriteResult::Gate` to trigger child processing before finalization.
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
//! // Create context
//! let mut ctx = KernelContext::new();
//!
//! // Patterns are simple functions - no closure capture needed
//! fn debuf(b: &BindingStore, i: &VarIntern, ctx: &mut KernelContext) -> RewriteResult {
//!     let id = ctx.next_global();  // Direct mutable access
//!     // ...
//! }
//!
//! let matcher: PatternMatcher<KernelContext> = PatternMatcher::new(vec![
//!     (pattern, Box::new(debuf)),
//! ]);
//!
//! // Pass context at rewrite time
//! let result = graph_rewrite(&matcher, root, &mut ctx);
//! ```
//!
//! Patterns that don't need context simply ignore the `_ctx` parameter:
//!
//! ```ignore
//! fn add_zero<C>(b: &BindingStore, _: &VarIntern, _ctx: &mut C) -> RewriteResult {
//!     // Don't use _ctx
//! }
//! ```

use morok_ir::{UOp, UOpKey};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use crate::pattern::{PatternMatcher, RewriteResult};

/// Stage in the 2-stage rewrite algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stage {
    /// Stage 0: Bottom-up fixed-point pattern matching
    Rewrite,
    /// Stage 1: Reconstruct with optimized children, then link result
    Finalize,
}

/// Stack entry for the 2-stage rewrite algorithm.
///
/// The separation of `original` and `working` is crucial for result linking:
/// - `original`: The node that consumers reference (used as key in results cache)
/// - `working`: The node after pattern rewrites (may differ from original)
#[derive(Debug, Clone)]
struct StackEntry {
    /// The node that consumers reference - used as key in results cache
    original: Rc<UOp>,
    /// Current processing stage
    stage: Stage,
    /// The node we're actively working with (may differ after rewrites)
    working: Rc<UOp>,
}

impl StackEntry {
    /// Create entry for a fresh node starting at Rewrite stage.
    fn new(node: Rc<UOp>) -> Self {
        Self { original: node.clone(), stage: Stage::Rewrite, working: node }
    }

    /// Create entry for Finalize stage with potentially different working node.
    fn finalize(original: Rc<UOp>, working: Rc<UOp>) -> Self {
        Self { original, stage: Stage::Finalize, working }
    }

    /// Create entry for Rewrite stage where original == working.
    fn rewrite(node: Rc<UOp>) -> Self {
        Self { original: node.clone(), stage: Stage::Rewrite, working: node }
    }
}

/// Efficient result lookup using path compression.
///
/// Provides O(α(n)) amortized lookup via path compression, where α is the
/// inverse Ackermann function (practically constant). This is much faster
/// than the O(chain_length) traversal of naive chain following.
struct ResultMap {
    /// Maps each node to its final result (with path compression on lookup)
    results: HashMap<UOpKey, Rc<UOp>>,
}

impl ResultMap {
    fn new() -> Self {
        Self { results: HashMap::new() }
    }

    /// Get the final result for a node with path compression.
    ///
    /// Follows the chain of rewrites and compresses the path so future
    /// lookups are O(1). Returns the node itself if no result is cached.
    fn get(&mut self, node: &Rc<UOp>) -> Rc<UOp> {
        let key = UOpKey(node.clone());

        // Fast path: no result cached
        let Some(result) = self.results.get(&key).cloned() else {
            return node.clone();
        };

        // If result points to self, we're done
        if Rc::ptr_eq(&result, node) {
            return result;
        }

        // Follow chain with path compression
        let mut current = result;
        let mut path = vec![key.clone()];

        const MAX_DEPTH: usize = 100;
        for _ in 0..MAX_DEPTH {
            let current_key = UOpKey(current.clone());
            match self.results.get(&current_key) {
                Some(next) if !Rc::ptr_eq(next, &current) => {
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
    fn link(&mut self, original: Rc<UOp>, result: Rc<UOp>) {
        self.results.insert(UOpKey(original), result);
    }

    /// Check if node has a result (without path compression).
    fn contains(&self, node: &Rc<UOp>) -> bool {
        self.results.contains_key(&UOpKey(node.clone()))
    }

    /// Get result from raw HashMap (for early exit check).
    fn get_direct(&self, key: &UOpKey) -> Option<Rc<UOp>> {
        self.results.get(key).cloned()
    }
}

/// Internal rewrite engine that implements the 2-stage stack-based algorithm.
///
/// Generic over context type `C` for compile-time type-safe context passing.
struct RewriteEngine<'a, C> {
    /// Pattern matcher for applying rewrite rules
    matcher: &'a PatternMatcher<C>,

    /// Mutable reference to context passed through to patterns
    ctx: &'a mut C,

    /// Final results cache: maps original node → optimized result
    /// Uses path compression for O(α(n)) amortized lookups
    results: ResultMap,

    /// Nodes pending processing (prevents duplicate pushes in DAGs).
    /// A node is in this set if it's currently on the stack or being processed.
    pending: HashSet<UOpKey>,

    /// Whether to always process children (bottom-up traversal).
    /// When true, all nodes have their children processed first.
    /// When false (default), only patterns returning Gate trigger child processing.
    bottom_up: bool,
}

impl<'a, C> RewriteEngine<'a, C> {
    fn new(matcher: &'a PatternMatcher<C>, ctx: &'a mut C) -> Self {
        Self { matcher, ctx, results: ResultMap::new(), pending: HashSet::new(), bottom_up: false }
    }

    fn new_bottom_up(matcher: &'a PatternMatcher<C>, ctx: &'a mut C) -> Self {
        Self { matcher, ctx, results: ResultMap::new(), pending: HashSet::new(), bottom_up: true }
    }

    /// Stage 0: Bottom-up fixed-point pattern matching.
    ///
    /// Applies patterns to the node until no more rewrites are possible.
    /// Children are only pushed for processing when a Gate is returned,
    /// matching Tinygrad's behavior where patterns control child processing.
    fn handle_rewrite(&mut self, stack: &mut Vec<StackEntry>, original: Rc<UOp>, working: Rc<UOp>) {
        // Fixed-point pattern matching with iteration limit (panic if exceeded)
        const MAX_ITERATIONS: usize = 1000;
        let mut node = working;
        let mut needs_children = false;

        for i in 0..MAX_ITERATIONS {
            match self.matcher.rewrite(&node, self.ctx) {
                RewriteResult::Rewritten(new_node) => {
                    node = new_node;
                }
                RewriteResult::Gate(_) => {
                    needs_children = true;
                    break;
                }
                RewriteResult::NoMatch => {
                    break;
                }
            }
            if i == MAX_ITERATIONS - 1 {
                panic!(
                    "Rewrite iteration limit ({}) exceeded: patterns may be creating an infinite loop. Node: {:?}",
                    MAX_ITERATIONS,
                    node.op()
                );
            }
        }

        // Schedule: Finalize (after children are done if needed)
        stack.push(StackEntry::finalize(original, node.clone()));

        // Push children for processing when:
        // 1. Gate was explicitly returned by a pattern, OR
        // 2. Bottom-up traversal is enabled (always process children)
        //
        // The `bottom_up` flag is set on the engine, not on individual patterns.
        // When false (default), only patterns returning Gate trigger child processing.
        // When true, all nodes have their children processed first.
        //
        // Use bottom_up=true for patterns like to_define_global that need to transform
        // deep nodes (e.g., BUFFER inside INDEX inside ADD inside STORE).
        // Use bottom_up=false for patterns like buffer_removal that need parent context.
        if needs_children || self.bottom_up {
            // Push children for processing in REVERSE order
            // Stack is LIFO, so reverse order means they're processed in original order
            let sources = node.op().sources();
            for child in sources.iter().rev() {
                let child_key = UOpKey(child.clone());
                if !self.pending.contains(&child_key) && !self.results.contains(child) {
                    self.pending.insert(child_key);
                    stack.push(StackEntry::rewrite(child.clone()));
                }
            }
        }
    }

    /// Stage 1: Reconstruct with optimized children and link result.
    ///
    /// Collects optimized children from results cache, reconstructs the node
    /// if any children changed, and caches the final result.
    ///
    /// If reconstruction creates a new node, we push it back to Rewrite stage
    /// to ensure any new patterns are applied and new children are processed.
    fn handle_finalize(&mut self, stack: &mut Vec<StackEntry>, original: Rc<UOp>, working: Rc<UOp>) {
        let sources = working.op().sources();

        // For leaf nodes, just link and return
        if sources.is_empty() {
            self.link_result(original, working);
            return;
        }

        // Collect optimized children
        let mut new_sources = Vec::with_capacity(sources.len());
        let mut any_changed = false;

        for src in &sources {
            let optimized = self.results.get(src);
            if !Rc::ptr_eq(&optimized, src) {
                any_changed = true;
            }
            new_sources.push(optimized);
        }

        if !any_changed {
            // No children changed - link working to its final result (following chain)
            let final_result = self.results.get(&working);
            self.link_result(original, final_result);
            return;
        }

        // Reconstruct with optimized children
        let reconstructed = working.with_sources(new_sources);

        // Reconstructed node may need its own rewrite pass if it hasn't been seen.
        //
        // Stack operations (LIFO execution order):
        //   1. Push: (original, Finalize, reconstructed) - runs SECOND
        //   2. Push: (reconstructed, Rewrite, reconstructed) - runs FIRST
        //
        // Execution order:
        //   - First: Rewrite reconstructed node (may trigger more rewrites)
        //   - Second: Link original → reconstructed's final result
        let recon_key = UOpKey(reconstructed.clone());
        if !self.results.contains(&reconstructed) && !self.pending.contains(&recon_key) {
            // Step 1: Link original to reconstructed's result (runs second due to LIFO)
            stack.push(StackEntry::finalize(original, reconstructed.clone()));

            // Step 2: Process reconstructed node first (runs first due to LIFO)
            self.pending.insert(recon_key);
            stack.push(StackEntry::rewrite(reconstructed));
            return;
        }

        // Reconstructed node already processed - link to its result
        let final_result = self.results.get(&reconstructed);
        self.link_result(original, final_result);
    }

    /// Link original node to its final result in the cache.
    fn link_result(&mut self, original: Rc<UOp>, result: Rc<UOp>) {
        // Record provenance if actually rewritten
        if !Rc::ptr_eq(&original, &result) {
            use morok_ir::provenance::{PROVENANCE_TRACKER, PassName};
            PROVENANCE_TRACKER.with(|tracker| {
                tracker.borrow_mut().record_transform(result.id, original.id, PassName::RewritePattern);
            });
        }

        self.results.link(original, result);
    }

    /// Main rewrite method: Stack-based 2-stage traversal.
    ///
    /// Traverses the graph in a stack-based manner, processing each node
    /// through 2 stages:
    /// 1. Stage 0 (Rewrite): Bottom-up pattern matching with fixed-point iteration
    /// 2. Stage 1 (Finalize): Source reconstruction + link final replacement
    fn rewrite(&mut self, root: Rc<UOp>) -> Rc<UOp> {
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

        while let Some(StackEntry { original, stage, working }) = stack.pop() {
            iterations += 1;
            if iterations > MAX_TOTAL_ITERATIONS {
                panic!(
                    "Rewrite total iteration limit ({}) exceeded: likely infinite loop. Stack size: {}, results cached: {}",
                    MAX_TOTAL_ITERATIONS,
                    stack.len(),
                    self.results.results.len()
                );
            }

            // Skip if original already has result
            if self.results.contains(&original) {
                continue;
            }

            match stage {
                Stage::Rewrite => self.handle_rewrite(&mut stack, original, working),
                Stage::Finalize => self.handle_finalize(&mut stack, original, working),
            }
        }

        self.results.get_direct(&root_key).unwrap_or(root)
    }
}

// Note: UOpKey uses stable UOp.id for hashing/equality, avoiding pointer-based ABA issues.

/// Apply graph rewriting to a UOp graph using the given pattern matcher.
///
/// This is the main entry point for graph rewriting. It applies a 2-stage
/// stack-based algorithm with fixed-point iteration.
///
/// # Type Parameters
///
/// * `C` - Context type passed through to pattern rewrite functions.
///   Use `()` for patterns that don't need context.
///
/// # Algorithm
///
/// 1. **Stage 0** (Rewrite): Try to match patterns against the current node.
///    - Apply patterns in fixed-point loop until no more matches
///    - If `Gate` is returned, push children for processing first (bottom-up)
///    - Otherwise, patterns see original children (top-down)
///
/// 2. **Stage 1** (Finalize): Reconstruct with children's results
///    - If reconstruction creates a new node, it's pushed back for rewriting
///    - Enables multi-stage optimizations (e.g., `WHERE(Lt(x,y),t,f) → WHERE(false,t,f) → f`)
///    - Link original node to final result
///
/// # Example
///
/// ```ignore
/// use schedule::{PatternMatcher, graph_rewrite};
/// use morok_ir::UOp;
///
/// // Patterns without context
/// let matcher: PatternMatcher<()> = PatternMatcher::new(vec![/* patterns */]);
/// let optimized = graph_rewrite(&matcher, root_uop, &mut ());
///
/// // Patterns with context
/// let matcher: PatternMatcher<KernelContext> = PatternMatcher::new(vec![/* patterns */]);
/// let mut ctx = KernelContext::new();
/// let optimized = graph_rewrite(&matcher, root_uop, &mut ctx);
/// ```
pub fn graph_rewrite<C>(matcher: &PatternMatcher<C>, root: Rc<UOp>, ctx: &mut C) -> Rc<UOp> {
    let mut engine = RewriteEngine::new(matcher, ctx);
    engine.rewrite(root)
}

/// Apply graph rewriting with bottom-up traversal.
///
/// Like `graph_rewrite`, but always processes children before parents.
/// Use this for patterns that need to transform deep nodes in the graph.
///
/// # Example
///
/// ```ignore
/// // Use bottom-up for patterns that need to transform deep nodes
/// let matcher = to_define_global_patterns();
/// let optimized = graph_rewrite_bottom_up(&matcher, root, &mut ctx);
/// ```
pub fn graph_rewrite_bottom_up<C>(matcher: &PatternMatcher<C>, root: Rc<UOp>, ctx: &mut C) -> Rc<UOp> {
    let mut engine = RewriteEngine::new_bottom_up(matcher, ctx);
    engine.rewrite(root)
}
