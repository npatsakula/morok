//! Graph rewrite engine implementation.
//!
//! # Algorithm
//!
//! Stack-based 3-stage DFS traversal with waitlist for dependency resolution:
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
//! let mut ctx = MyContext::new();
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

use crate::{Op, UOp, UOpKey};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::pattern::{Matcher, RewriteResult};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TraversalMode {
    Full,
    PreserveCallBodies,
}

fn traversal_children(node: &Arc<UOp>, mode: TraversalMode) -> (Vec<Arc<UOp>>, Vec<Arc<UOp>>) {
    if mode == TraversalMode::Full {
        return (node.op().sources().into_iter().collect(), Vec::new());
    }

    match node.op() {
        Op::Call { body, args, .. } | Op::Function { body, args, .. } => {
            (args.iter().cloned().collect(), vec![body.clone()])
        }
        // Program holds compiled artifacts (linear/source/binary) wrapped as
        // UOps; traversing through them during rewrite passes is expensive
        // and unnecessary — only the device producer is traversed.
        Op::Program { sink, device, linear, source, binary } => {
            let mut skipped = Vec::with_capacity(
                1 + usize::from(linear.is_some()) + usize::from(source.is_some()) + usize::from(binary.is_some()),
            );
            skipped.push(sink.clone());
            if let Some(linear) = linear {
                skipped.push(linear.clone());
            }
            if let Some(source) = source {
                skipped.push(source.clone());
            }
            if let Some(binary) = binary {
                skipped.push(binary.clone());
            }
            (vec![device.clone()], skipped)
        }
        _ => (node.op().sources().into_iter().collect(), Vec::new()),
    }
}

/// Maximum stack size before we consider the rewrite to be in an infinite loop.
const REWRITE_STACK_LIMIT: usize = 500_000;

/// Stack entry for the 3-stage rewrite algorithm.
///
/// - `n`: the original node (used as key in `replace` dict)
/// - `stage`: 0 (PushChildren), 1 (ApplyPatterns), or 2 (Link)
/// - `new_n`: the working copy (may differ from `n` after bpm rewrites or reconstruction)
#[derive(Clone)]
struct Entry {
    n: Arc<UOp>,
    stage: u8,
    new_n: Arc<UOp>,
}

/// Internal rewrite engine.
///
/// Generic over matcher types and context type for compile-time type-safe matching.
/// Supports separate `pm` (top-down) and `bpm` (bottom-up) matchers.
struct RewriteEngine<'a, PM, BPM, C>
where
    PM: Matcher<C>,
    BPM: Matcher<C>,
{
    /// Top-down pattern matcher: applied in Stage 1 (ApplyPatterns).
    /// Patterns see OPTIMIZED children.
    pm: Option<&'a PM>,

    /// Bottom-up pattern matcher: applied in Stage 0 (PushChildren).
    /// Patterns see ORIGINAL children.
    bpm: Option<&'a BPM>,

    /// Mutable reference to context passed through to patterns.
    ctx: &'a mut C,

    /// Traversal mode controlling CALL/FUNCTION/PROGRAM boundary handling.
    traversal_mode: TraversalMode,

    /// Results cache: maps original node → optimized result.
    replace: HashMap<UOpKey, Arc<UOp>>,

    /// BPM result cache: prevents re-running pattern matching on nodes already seen.
    bpm_cache: HashMap<UOpKey, Option<Arc<UOp>>>,
}

impl<'a, PM, BPM, C> RewriteEngine<'a, PM, BPM, C>
where
    PM: Matcher<C>,
    BPM: Matcher<C>,
{
    fn new(pm: Option<&'a PM>, bpm: Option<&'a BPM>, ctx: &'a mut C, traversal_mode: TraversalMode) -> Self {
        Self { pm, bpm, ctx, traversal_mode, replace: HashMap::new(), bpm_cache: HashMap::new() }
    }

    /// Single-shot top-down pattern application.
    /// No cache needed: pm_rewrite is called at most once per UOp due to the
    /// replace dict check in the main loop.
    #[inline]
    fn pm_rewrite(&mut self, x: &Arc<UOp>) -> Option<Arc<UOp>> {
        let pm = self.pm.as_ref()?;
        match pm.rewrite(x, self.ctx) {
            RewriteResult::Rewritten(new_node) => {
                debug_assert!(
                    !Arc::ptr_eq(&new_node, x),
                    "PM pattern returned Rewritten but produced the same node (id={}). \
                     This causes infinite loops. Return NoMatch instead.\nOp: {:?}",
                    x.id,
                    x.op().as_ref(),
                );
                Some(new_node)
            }
            RewriteResult::Gate(_) | RewriteResult::NoMatch => None,
        }
    }

    /// Cached bottom-up pattern application.
    /// Cache prevents re-running patterns on nodes already seen during fixed-point.
    /// Gate results are NOT cached — Gate is an exception that bypasses the cache.
    #[inline]
    fn cached_bpm_rewrite(&mut self, x: &Arc<UOp>) -> Result<Option<Arc<UOp>>, Arc<UOp>> {
        let key = UOpKey(x.clone());
        if let Some(cached) = self.bpm_cache.get(&key) {
            return match cached {
                Some(node) => Ok(Some(node.clone())),
                None => Ok(None),
            };
        }
        let bpm = self.bpm.as_ref().unwrap();
        match bpm.rewrite(x, self.ctx) {
            RewriteResult::Rewritten(new_node) => {
                debug_assert!(
                    !Arc::ptr_eq(&new_node, x),
                    "BPM pattern returned Rewritten but produced the same node (id={}). \
                     This causes infinite loops. Return NoMatch instead.\nOp: {:?}",
                    x.id,
                    x.op().as_ref(),
                );
                self.bpm_cache.insert(key, Some(new_node.clone()));
                Ok(Some(new_node))
            }
            RewriteResult::Gate(gate_node) => Err(gate_node),
            RewriteResult::NoMatch => {
                self.bpm_cache.insert(key, None);
                Ok(None)
            }
        }
    }

    /// Record a result in the replace map, with provenance tracking.
    #[inline]
    fn record_replace(&mut self, original: &Arc<UOp>, result: Arc<UOp>) {
        if !Arc::ptr_eq(original, &result) {
            use crate::provenance::{PROVENANCE_TRACKER, PassName};
            PROVENANCE_TRACKER.with(|tracker| {
                tracker.borrow_mut().record_transform(result.id, original.id, PassName::RewritePattern);
            });
        }
        self.replace.insert(UOpKey(original.clone()), result);
    }

    /// Main rewrite method — stack-based 3-stage traversal.
    #[allow(clippy::mutable_key_type)]
    fn rewrite(&mut self, root: Arc<UOp>) -> Arc<UOp> {
        let mut stack: Vec<Entry> = vec![Entry { n: root.clone(), stage: 0, new_n: root.clone() }];

        // All UOps either on the stack or in self.replace — don't have to be placed again.
        let mut on_stack: HashSet<UOpKey> = HashSet::new();
        on_stack.insert(UOpKey(root.clone()));

        // UOps waiting on a dependency to be in self.replace.
        let mut waitlist: HashMap<UOpKey, Vec<Entry>> = HashMap::new();

        while let Some(Entry { n, stage, new_n }) = stack.pop() {
            if stack.len() > REWRITE_STACK_LIMIT {
                panic!(
                    "infinite loop in graph_rewrite (stack too big: {}). results cached: {}",
                    stack.len(),
                    self.replace.len(),
                );
            }

            let n_key = UOpKey(n.clone());

            if self.replace.contains_key(&n_key) {
                continue;
            }

            if stage == 0 {
                // Stage 0: PushChildren
                let mut working = new_n;

                if self.bpm.is_some() {
                    // Apply bpm rewrite rules until a fixed point is reached.
                    let mut seen: HashSet<UOpKey> = HashSet::new();
                    let mut gated = false;
                    loop {
                        let working_key = UOpKey(working.clone());
                        if seen.contains(&working_key) {
                            panic!(
                                "infinite loop in fixed_point_rewrite: node {:?} (id={}) seen twice",
                                working.op().as_ref(),
                                working.id
                            );
                        }
                        seen.insert(working_key);
                        match self.cached_bpm_rewrite(&working) {
                            Ok(Some(rewritten)) => {
                                working = rewritten;
                            }
                            Ok(None) => break,
                            Err(gate_node) => {
                                // Gate: done with this node, don't descend into children
                                self.record_replace(&n, gate_node);
                                if let Some(entries) = waitlist.remove(&n_key) {
                                    stack.extend(entries);
                                }
                                gated = true;
                                break;
                            }
                        }
                    }
                    if gated {
                        continue;
                    }
                }

                stack.push(Entry { n: n.clone(), stage: 1, new_n: working.clone() });

                let (sources, skipped) = traversal_children(&working, self.traversal_mode);
                for skipped_child in skipped {
                    self.replace.entry(UOpKey(skipped_child.clone())).or_insert(skipped_child);
                }

                for child in sources.into_iter().rev() {
                    let child_key = UOpKey(child.clone());
                    if on_stack.contains(&child_key) {
                        continue;
                    }
                    stack.push(Entry { n: child.clone(), stage: 0, new_n: child });
                    on_stack.insert(child_key);
                }
            } else if stage == 1 {
                // Stage 1: ApplyPatterns
                let sources = new_n.op().sources();

                let mut tmp: Vec<Arc<UOp>> = Vec::with_capacity(sources.len());
                let mut waiting = false;

                for src in &sources {
                    let src_key = UOpKey(src.clone());
                    if let Some(rx) = self.replace.get(&src_key) {
                        tmp.push(rx.clone());
                    } else {
                        // Source not ready: register in waitlist
                        waitlist.entry(src_key).or_default().push(Entry {
                            n: n.clone(),
                            stage: 1,
                            new_n: new_n.clone(),
                        });
                        waiting = true;
                        break;
                    }
                }

                if waiting {
                    continue;
                }

                // All sources ready — reconstruct if any changed
                let sources_changed = tmp.iter().zip(sources.iter()).any(|(a, b)| !Arc::ptr_eq(a, b));

                // Hash consing may collapse reconstruction to same node even when
                // sources logically changed. Detect this and treat as unchanged.
                let node = if sources_changed {
                    let reconstructed = new_n.with_sources(tmp);
                    if Arc::ptr_eq(&reconstructed, &new_n) { new_n.clone() } else { reconstructed }
                } else {
                    new_n.clone()
                };

                if Arc::ptr_eq(&node, &new_n) {
                    // Sources effectively unchanged: try pm rewrite
                    if let Some(new_src_n) = self.pm_rewrite(&new_n) {
                        stack.push(Entry { n: n.clone(), stage: 2, new_n: new_src_n.clone() });
                        stack.push(Entry { n: new_src_n.clone(), stage: 0, new_n: new_src_n });
                    } else {
                        // No pm match — done with this node
                        self.record_replace(&n, new_n);
                        if let Some(entries) = waitlist.remove(&n_key) {
                            stack.extend(entries);
                        }
                    }
                } else {
                    // Reconstruction produced a new node — process it, then link back
                    stack.push(Entry { n: n.clone(), stage: 2, new_n: node.clone() });
                    stack.push(Entry { n: node.clone(), stage: 0, new_n: node });
                }
            } else {
                // Stage 2: Link
                let new_n_key = UOpKey(new_n.clone());

                if let Some(replaced_new_n) = self.replace.get(&new_n_key).cloned() {
                    self.record_replace(&n, replaced_new_n);
                    if let Some(entries) = waitlist.remove(&n_key) {
                        stack.extend(entries);
                    }
                } else {
                    // Not ready: register in waitlist
                    waitlist.entry(new_n_key).or_default().push(Entry { n, stage: 2, new_n });
                }
            }
        }

        self.replace.get(&UOpKey(root.clone())).cloned().unwrap_or(root)
    }

    /// MLIR-style walk pattern rewrite driver — single-pass, no re-traversal.
    ///
    /// Differs from [`Self::rewrite`] in two ways:
    /// 1. `bpm` is applied **once** per node — no fixed-point loop.
    /// 2. When a pattern returns a replacement, the replacement is **not**
    ///    traversed: its children are never visited and `pm` is not re-applied.
    ///
    /// Use this when a replacement contains the original key, e.g.
    /// `Buffer → After(Buffer, [Store(...)])` for view-assign, where
    /// re-traversal would loop or wrap the key multiple times.
    fn walk_rewrite(&mut self, root: Arc<UOp>) -> Arc<UOp> {
        let mut stack: Vec<(Arc<UOp>, bool)> = vec![(root.clone(), false)];

        while let Some((n, processed)) = stack.pop() {
            if stack.len() > REWRITE_STACK_LIMIT {
                panic!(
                    "infinite loop in walk_rewrite (stack too big: {}). results cached: {}",
                    stack.len(),
                    self.replace.len(),
                );
            }

            let n_key = UOpKey(n.clone());
            if self.replace.contains_key(&n_key) {
                continue;
            }

            if !processed {
                // Try bpm exactly once on the original node. On match, record the
                // replacement and skip descent — the replacement is treated as a
                // leaf even if it contains the original key.
                if self.bpm.is_some() {
                    match self.cached_bpm_rewrite(&n) {
                        Ok(Some(rewritten)) => {
                            self.record_replace(&n, rewritten);
                            continue;
                        }
                        Err(gated) => {
                            self.record_replace(&n, gated);
                            continue;
                        }
                        Ok(None) => {}
                    }
                }

                // No bpm match — process children, then come back to rebuild.
                stack.push((n.clone(), true));

                let (sources, skipped) = traversal_children(&n, self.traversal_mode);
                for skipped_child in skipped {
                    self.replace.entry(UOpKey(skipped_child.clone())).or_insert(skipped_child);
                }
                for child in sources.into_iter().rev() {
                    let child_key = UOpKey(child.clone());
                    if !self.replace.contains_key(&child_key) {
                        stack.push((child, false));
                    }
                }
            } else {
                // Rebuild with rewritten sources.
                let sources = n.op().sources();
                let new_src: Vec<Arc<UOp>> = sources
                    .iter()
                    .map(|s| self.replace.get(&UOpKey(s.clone())).cloned().unwrap_or_else(|| s.clone()))
                    .collect();
                let any_changed = new_src.iter().zip(sources.iter()).any(|(a, b)| !Arc::ptr_eq(a, b));
                let rebuilt = if any_changed { n.with_sources(new_src) } else { n.clone() };

                // Apply pm exactly once on the rebuilt node — replacement is used as-is.
                let final_n = if self.pm.is_some() { self.pm_rewrite(&rebuilt).unwrap_or(rebuilt) } else { rebuilt };

                self.record_replace(&n, final_n);
            }
        }

        self.replace.get(&UOpKey(root.clone())).cloned().unwrap_or(root)
    }
}

/// Marker type for "no matcher" in generic contexts.
pub struct NoMatcher;

impl<C> Matcher<C> for NoMatcher {
    fn rewrite(&self, _node: &Arc<UOp>, _ctx: &mut C) -> RewriteResult {
        RewriteResult::NoMatch
    }
}

/// Apply graph rewriting. Patterns see **OPTIMIZED** children (Stage 1).
pub fn graph_rewrite<M: Matcher<C>, C>(matcher: &M, root: Arc<UOp>, ctx: &mut C) -> Arc<UOp> {
    RewriteEngine::new(Some(matcher), None::<&NoMatcher>, ctx, TraversalMode::Full).rewrite(root)
}

/// Apply graph rewriting with bottom-up pattern application.
/// Patterns see **ORIGINAL** children (Stage 0).
pub fn graph_rewrite_bottom_up<M: Matcher<C>, C>(matcher: &M, root: Arc<UOp>, ctx: &mut C) -> Arc<UOp> {
    RewriteEngine::new(None::<&NoMatcher>, Some(matcher), ctx, TraversalMode::Full).rewrite(root)
}

/// MLIR-style walk pattern rewrite — single-pass, no re-traversal.
///
/// Use when a replacement may contain the original key (e.g.
/// `Buffer → After(Buffer, [Store(...)])` for view-assign): a re-traversing
/// driver would loop or wrap the key multiple times.
pub fn graph_rewrite_walk<M: Matcher<C>, C>(matcher: &M, root: Arc<UOp>, ctx: &mut C) -> Arc<UOp> {
    RewriteEngine::new(None::<&NoMatcher>, Some(matcher), ctx, TraversalMode::Full).walk_rewrite(root)
}

/// Apply graph rewriting with both top-down and bottom-up patterns.
/// - `bpm` patterns see ORIGINAL children (Stage 0)
/// - `pm` patterns see OPTIMIZED children (Stage 1)
pub fn graph_rewrite_with_bpm<PM, BPM, C>(pm: &PM, bpm: &BPM, root: Arc<UOp>, ctx: &mut C) -> Arc<UOp>
where
    PM: Matcher<C>,
    BPM: Matcher<C>,
{
    RewriteEngine::new(Some(pm), Some(bpm), ctx, TraversalMode::Full).rewrite(root)
}

/// Apply graph rewriting with both top-down and bottom-up patterns while preserving
/// CALL/FUNCTION/PROGRAM boundaries.
/// - `bpm` patterns see ORIGINAL children (Stage 0)
/// - `pm` patterns see OPTIMIZED children (Stage 1)
/// - Traversal skips CALL/FUNCTION bodies and PROGRAM internals
/// - CALL/FUNCTION args and PROGRAM device are still traversed
pub fn graph_rewrite_with_bpm_preserve_calls<PM, BPM, C>(pm: &PM, bpm: &BPM, root: Arc<UOp>, ctx: &mut C) -> Arc<UOp>
where
    PM: Matcher<C>,
    BPM: Matcher<C>,
{
    RewriteEngine::new(Some(pm), Some(bpm), ctx, TraversalMode::PreserveCallBodies).rewrite(root)
}

/// Rewrite graph while preserving CALL/FUNCTION/PROGRAM boundaries.
///
/// CALL/FUNCTION/PROGRAM nodes are still matchable/rewriteable, but traversal
/// does not descend into CALL/FUNCTION bodies or PROGRAM internals.
/// CALL/FUNCTION arguments and PROGRAM device are still traversed.
pub fn graph_rewrite_preserve_calls<PM, C>(pm: &PM, root: Arc<UOp>, ctx: &mut C) -> Arc<UOp>
where
    PM: Matcher<C>,
{
    RewriteEngine::new(Some(pm), None::<&NoMatcher>, ctx, TraversalMode::PreserveCallBodies).rewrite(root)
}

/// Bottom-up rewrite with CALL/FUNCTION/PROGRAM boundary preservation.
///
/// CALL/FUNCTION/PROGRAM nodes are still matchable/rewriteable, but traversal
/// does not descend into CALL/FUNCTION bodies or PROGRAM internals.
/// CALL/FUNCTION arguments and PROGRAM device are still traversed.
pub fn graph_rewrite_bottom_up_preserve_calls<BPM, C>(bpm: &BPM, root: Arc<UOp>, ctx: &mut C) -> Arc<UOp>
where
    BPM: Matcher<C>,
{
    RewriteEngine::new(None::<&NoMatcher>, Some(bpm), ctx, TraversalMode::PreserveCallBodies).rewrite(root)
}
