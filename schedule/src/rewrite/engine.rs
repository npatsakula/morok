//! Graph rewrite engine implementation.
//!
//! Implements the core graph rewriting algorithm with fixed-point iteration.
//!
//! # Algorithm
//!
//! The algorithm operates in 2 stages:
//! - Stage 0: Initial visit + bottom-up fixed-point iteration
//! - Stage 1: Source reconstruction after children are rewritten
//!   - If reconstruction creates a new node, patterns are re-applied to it
//!   - This enables multi-stage optimizations (e.g., WHERE elimination after comparison folding)
//!     (includes Stage 2: linking rewritten results back to original nodes)
//!
//! # Pattern Context
//!
//! Patterns that need external state should use **closure capture** rather than
//! a context parameter. This is the idiomatic Rust approach and provides better
//! type safety.
//!
//! ## Example
//!
//! ```ignore
//! use std::rc::Rc;
//! use std::cell::RefCell;
//!
//! // Create context wrapped in Rc<RefCell<>>
//! let ctx = Rc::new(RefCell::new(MyContext::new()));
//!
//! // Patterns capture context via closure
//! let mut patterns = vec![];
//! let ctx_clone = Rc::clone(&ctx);
//! patterns.push((
//!     UPat::var("x"),
//!     Box::new(move |bindings| {
//!         // Access context inside pattern
//!         let ctx_ref = ctx_clone.borrow();
//!         // ... use context
//!     })
//! ));
//!
//! let matcher = PatternMatcher::new(patterns);
//! graph_rewrite(&matcher, root);
//! ```
//!
//! See `crate::rangeify::patterns::apply_rangeify_patterns` for a real-world example.

use morok_ir::UOp;
use std::collections::HashMap;
use std::rc::Rc;

use crate::pattern::{PatternMatcher, RewriteResult};

/// Stage in the 3-stage rewrite algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Stage {
    /// Stage 0: Initial visit + bottom-up fixed-point iteration
    BottomUp,
    /// Stage 1: Source reconstruction after children are rewritten
    SourceReconstruction,
}

/// Internal rewrite engine that implements the 3-stage stack-based algorithm.
struct RewriteEngine<'a> {
    /// Pattern matcher for applying rewrite rules
    matcher: &'a PatternMatcher,

    /// Tracking which nodes have been visited in each stage
    /// Key: UOp pointer (as usize), Value: stage reached
    visited: HashMap<usize, Stage>,

    /// Cache of pattern match results (Stage 0)
    /// Key: UOp pointer (as usize), Value: RewriteResult
    match_cache: HashMap<usize, RewriteResult>,

    /// Cache of final replacements (Stage 2)
    /// Key: Original UOp pointer (as usize), Value: Rewritten UOp
    replacement_cache: HashMap<usize, Rc<UOp>>,
}

impl<'a> RewriteEngine<'a> {
    fn new(matcher: &'a PatternMatcher) -> Self {
        Self { matcher, visited: HashMap::new(), match_cache: HashMap::new(), replacement_cache: HashMap::new() }
    }

    /// Get a stable pointer value for a UOp (for use as HashMap key).
    #[inline]
    fn uop_key(uop: &Rc<UOp>) -> usize {
        Rc::as_ptr(uop) as usize
    }

    /// Stage 0: Bottom-up fixed-point iteration.
    ///
    /// Tries to rewrite the node using patterns. If a pattern matches:
    /// - Rewritten(...): Keep rewriting until fixed point
    /// - Gate(...): Skip this node (children need processing first)
    /// - NoMatch: No rewrite applicable
    fn stage0_bottom_up(&mut self, uop: &Rc<UOp>) -> RewriteResult {
        let key = Self::uop_key(uop);

        // Check cache first
        if let Some(result) = self.match_cache.get(&key) {
            return result.clone();
        }

        // Try to rewrite, applying fixed-point iteration
        let mut current = uop.clone();
        loop {
            match self.matcher.rewrite(&current) {
                RewriteResult::NoMatch => {
                    // No more rewrites possible - cache and return
                    let result = if Rc::ptr_eq(&current, uop) {
                        RewriteResult::NoMatch
                    } else {
                        RewriteResult::Rewritten(current.clone())
                    };
                    self.match_cache.insert(key, result.clone());
                    return result;
                }
                RewriteResult::Gate(gate_uop) => {
                    // Bottom-up gate: children need processing first
                    let result = RewriteResult::Gate(gate_uop);
                    self.match_cache.insert(key, result.clone());
                    return result;
                }
                RewriteResult::Rewritten(new_uop) => {
                    // Pattern matched - continue fixed-point iteration
                    current = new_uop;
                }
            }
        }
    }

    /// Stage 1: Source reconstruction.
    ///
    /// Reconstructs the UOp with rewritten children (sources).
    /// Returns the reconstructed UOp, or the original if no children changed.
    fn stage1_reconstruct(&mut self, uop: &Rc<UOp>) -> Rc<UOp> {
        // Get all source UOps
        let sources = uop.op().sources();

        if sources.is_empty() {
            // Leaf node - no reconstruction needed
            return uop.clone();
        }

        // Collect rewritten sources
        let mut new_sources = Vec::with_capacity(sources.len());
        let mut any_changed = false;

        for src in sources {
            let rewritten = self.get_replacement(&src);
            if !Rc::ptr_eq(&rewritten, &src) {
                any_changed = true;
            }
            new_sources.push(rewritten);
        }

        if !any_changed {
            // No sources changed - return original
            return uop.clone();
        }

        // Reconstruct with new sources
        uop.with_sources(new_sources)
    }

    /// Stage 2: Link result.
    ///
    /// Caches the final replacement mapping from original to rewritten UOp.
    fn stage2_link(&mut self, original: &Rc<UOp>, rewritten: Rc<UOp>) {
        let key = Self::uop_key(original);

        // Record provenance if UOp was actually rewritten
        if !Rc::ptr_eq(original, &rewritten) {
            use morok_ir::provenance::{PROVENANCE_TRACKER, PassName};
            PROVENANCE_TRACKER.with(|tracker| {
                tracker.borrow_mut().record_transform(rewritten.id, original.id, PassName::RewritePattern);
            });
        }

        self.replacement_cache.insert(key, rewritten);
    }

    /// Get the replacement for a UOp, or the original if not replaced.
    fn get_replacement(&self, uop: &Rc<UOp>) -> Rc<UOp> {
        let key = Self::uop_key(uop);
        self.replacement_cache.get(&key).cloned().unwrap_or_else(|| uop.clone())
    }

    /// Main rewrite method: Stack-based 3-stage traversal.
    ///
    /// Traverses the graph in a stack-based manner, processing each node
    /// through 3 stages:
    /// 1. Stage 0: Bottom-up pattern matching with fixed-point iteration
    /// 2. Stage 1: Source reconstruction with rewritten children
    /// 3. Stage 2: Link final replacement in cache
    fn rewrite(&mut self, root: Rc<UOp>) -> Rc<UOp> {
        // Stack of (UOp, Stage) to process
        let mut stack: Vec<(Rc<UOp>, Stage)> = vec![(root.clone(), Stage::BottomUp)];

        while let Some((uop, stage)) = stack.pop() {
            let key = Self::uop_key(&uop);

            match stage {
                Stage::BottomUp => {
                    // Check if already visited at this stage
                    if let Some(&visited_stage) = self.visited.get(&key)
                        && visited_stage >= Stage::BottomUp
                    {
                        continue; // Already processed
                    }

                    // Try Stage 0: bottom-up pattern matching
                    match self.stage0_bottom_up(&uop) {
                        RewriteResult::Gate(_) => {
                            // Gate hit: children need processing first
                            // Push this node for Stage 1 (after children are done)
                            stack.push((uop.clone(), Stage::SourceReconstruction));

                            // Push children for Stage 0
                            for src in uop.op().sources() {
                                stack.push((src, Stage::BottomUp));
                            }
                        }
                        RewriteResult::Rewritten(new_uop) => {
                            // Pattern matched and rewrote - process the new UOp
                            self.visited.insert(key, Stage::BottomUp);
                            self.stage2_link(&uop, new_uop.clone());

                            // Continue with the rewritten version
                            stack.push((new_uop, Stage::SourceReconstruction));
                        }
                        RewriteResult::NoMatch => {
                            // No rewrite - proceed to reconstruction
                            self.visited.insert(key, Stage::BottomUp);
                            stack.push((uop, Stage::SourceReconstruction));
                        }
                    }
                }

                Stage::SourceReconstruction => {
                    // Check if already visited at this stage
                    if let Some(&visited_stage) = self.visited.get(&key)
                        && visited_stage >= Stage::SourceReconstruction
                    {
                        continue;
                    }

                    // Stage 1: Reconstruct with rewritten children
                    let mut reconstructed = self.stage1_reconstruct(&uop);

                    // NEW: If reconstruction created a new node, try patterns on it
                    // This enables multi-stage optimizations where patterns can match
                    // on reconstructed nodes (e.g., WHERE with constant condition)
                    if !Rc::ptr_eq(&reconstructed, &uop) {
                        // Apply fixed-point iteration on the reconstructed node
                        const MAX_RECONSTRUCTION_REWRITES: usize = 1000;
                        let mut current = reconstructed.clone();
                        let mut iterations = 0;
                        loop {
                            if iterations >= MAX_RECONSTRUCTION_REWRITES {
                                eprintln!(
                                    "Warning: reconstruction rewrite limit ({}) reached for op: {:?}",
                                    MAX_RECONSTRUCTION_REWRITES,
                                    current.op()
                                );
                                reconstructed = current;
                                break;
                            }
                            match self.matcher.rewrite(&current) {
                                RewriteResult::Rewritten(new_uop) => {
                                    // Pattern matched - continue fixed-point iteration
                                    current = new_uop;
                                    iterations += 1;
                                }
                                _ => {
                                    // No more rewrites possible
                                    reconstructed = current;
                                    break;
                                }
                            }
                        }
                    }

                    self.visited.insert(key, Stage::SourceReconstruction);

                    // Stage 2: Link result
                    self.stage2_link(&uop, reconstructed);
                }
            }
        }

        // Return final replacement for root
        self.get_replacement(&root)
    }
}

/// Apply graph rewriting to a UOp graph using the given pattern matcher.
///
/// This is the main entry point for graph rewriting. It applies a 3-stage
/// stack-based algorithm to rewrite the graph bottom-up with fixed-point
/// iteration.
///
/// # Algorithm
///
/// 1. **Stage 0** (Bottom-up): Try to match patterns against each node.
///    - If a pattern matches, continue rewriting until fixed point
///    - If a Gate is returned, skip and process children first
///
/// 2. **Stage 1** (Source reconstruction): Reconstruct nodes with rewritten children
///    - If reconstruction creates a new node, patterns are re-applied
///    - Enables multi-stage optimizations (e.g., WHERE(Lt(x, y), t, f) → WHERE(false, t, f) → f)
///
/// 3. **Stage 2** (Link): Cache final replacements
///
/// # Example
///
/// ```ignore
/// use schedule::{PatternMatcher, graph_rewrite};
/// use morok_ir::UOp;
///
/// // Create pattern matcher with optimization patterns
/// let matcher = PatternMatcher::new(vec![/* patterns */]);
///
/// // Apply rewrites to the graph
/// let optimized = graph_rewrite(&matcher, root_uop);
/// ```
pub fn graph_rewrite(matcher: &PatternMatcher, root: Rc<UOp>) -> Rc<UOp> {
    let mut engine = RewriteEngine::new(matcher);
    engine.rewrite(root)
}
