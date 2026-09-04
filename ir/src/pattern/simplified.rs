//! Pattern matcher: a chain of compiled blocks.
//!
//! # Architecture
//!
//! The `patterns!` macro compiles a whole block into one function that dispatches on
//! the root op kind with a `match` over constant keys and tests each rule's constant
//! early-reject mask inline (Tinygrad's `UPat.early_reject`, uop/ops.py:1390). A
//! matcher is a sequence of such segments; each carries the mask of root kinds it can
//! match at all, so a `rewrite` skips a segment with one bit test. Composition (`+`)
//! appends segments; a guard (`guarded`) or a context lift (`with_context`) is a
//! property of a segment, checked once per `rewrite` call.
//!
//! ```ignore
//! let matcher = patterns! {
//!     Add(x, @zero) => x,              // `match` arm Binary(Add)
//!     Mul(x, @one) => x,               // `match` arm Binary(Mul)
//!     x if is_const(x) => fold(x),     // wildcard: sequential step
//! };
//! ```

use std::sync::Arc;

use crate::UOp;
use crate::op::OpMask;
use crate::op::pattern_derived::OpKey;

use super::RewriteResult;

/// A compiled block: root node, the mask of its direct children's kinds, and context.
///
/// The children mask is handed in rather than read from the node so early rejects can
/// be switched off for equivalence checks ([`SimplifiedPatternMatcher::without_early_reject`]).
pub type BlockFn<C> = Arc<dyn Fn(&Arc<UOp>, OpMask, &mut C) -> RewriteResult + Send + Sync>;

type Guard = Arc<dyn Fn(&Arc<UOp>) -> bool + Send + Sync>;

/// Dispatch metadata of one rule: the root kinds it can match, and what it demands of
/// the root's direct children.
pub type RuleMeta = (OpMask, OpMask);

enum Body<C> {
    Typed(BlockFn<C>),
    Unit(BlockFn<()>),
}

impl<C> Clone for Body<C> {
    fn clone(&self) -> Self {
        match self {
            Self::Typed(block) => Self::Typed(Arc::clone(block)),
            Self::Unit(block) => Self::Unit(Arc::clone(block)),
        }
    }
}

struct Segment<C> {
    /// Root kinds any rule of the block can match.
    root: OpMask,
    guard: Option<Guard>,
    early_reject: bool,
    rules: Vec<RuleMeta>,
    body: Body<C>,
}

impl<C> Clone for Segment<C> {
    fn clone(&self) -> Self {
        Self {
            root: self.root,
            guard: self.guard.clone(),
            early_reject: self.early_reject,
            rules: self.rules.clone(),
            body: self.body.clone(),
        }
    }
}

/// Pattern matcher: compiled blocks tried in order, each skipped by a root-kind bit test.
///
/// `C` is the context type passed to every rule; `()` for stateless matching.
pub struct SimplifiedPatternMatcher<C = ()> {
    segments: Vec<Segment<C>>,
}

impl<C> SimplifiedPatternMatcher<C> {
    /// Create a new empty pattern matcher.
    pub fn new() -> Self {
        Self { segments: Vec::new() }
    }

    /// Append a compiled block; `rules` describes each of its rules in source order.
    pub fn add_block<F>(&mut self, rules: &[RuleMeta], block: F)
    where
        F: Fn(&Arc<UOp>, OpMask, &mut C) -> RewriteResult + Send + Sync + 'static,
    {
        self.segments.push(Segment {
            root: rules.iter().fold(OpMask::EMPTY, |acc, (root, _)| acc.union(*root)),
            guard: None,
            early_reject: true,
            rules: rules.to_vec(),
            body: Body::Typed(Arc::new(block)),
        });
    }

    /// Add a hand-written rule for specific OpKey(s); an empty `keys` registers a wildcard.
    pub fn add<F>(&mut self, keys: &[OpKey], closure: F)
    where
        F: Fn(&Arc<UOp>, &mut C) -> RewriteResult + Send + Sync + 'static,
    {
        self.add_rejecting(keys, &[], closure);
    }

    /// Add a hand-written rule that can only match when the root's direct children
    /// include every op kind in `early_reject`; the closure is skipped otherwise.
    ///
    /// `early_reject` must be a *necessary* condition for the closure to match, exactly
    /// as in Tinygrad's `UPat.early_reject` (uop/ops.py:1390).
    pub fn add_rejecting<F>(&mut self, keys: &[OpKey], early_reject: &[OpKey], closure: F)
    where
        F: Fn(&Arc<UOp>, &mut C) -> RewriteResult + Send + Sync + 'static,
    {
        let root = if keys.is_empty() { OpMask::ALL } else { keys.iter().copied().collect() };
        let reject: OpMask = early_reject.iter().copied().collect();
        self.add_block(&[(root, reject)], move |uop, src_ops, ctx| {
            if reject.is_subset_of(src_ops) { closure(uop, ctx) } else { RewriteResult::NoMatch }
        });
    }

    /// Add a hand-written wildcard rule (tried for every op, in source order).
    pub fn add_wildcard<F>(&mut self, closure: F)
    where
        F: Fn(&Arc<UOp>, &mut C) -> RewriteResult + Send + Sync + 'static,
    {
        self.add(&[], closure);
    }

    /// Return a matcher whose rewrites run only when `guard` accepts the root.
    ///
    /// The guard is checked once per segment per `rewrite` call, and composes with
    /// `+` without changing source-order priority.
    pub fn guarded<F>(&self, guard: F) -> Self
    where
        C: 'static,
        F: Fn(&Arc<UOp>) -> bool + Send + Sync + 'static,
    {
        let guard: Guard = Arc::new(guard);
        let segments = self
            .segments
            .iter()
            .map(|segment| {
                let guard = match &segment.guard {
                    None => Arc::clone(&guard),
                    Some(inner) => {
                        let (inner, outer) = (Arc::clone(inner), Arc::clone(&guard));
                        Arc::new(move |uop: &Arc<UOp>| inner(uop) && outer(uop)) as Guard
                    }
                };
                Segment { guard: Some(guard), ..segment.clone() }
            })
            .collect();
        Self { segments }
    }

    /// Copy of this matcher with every early reject switched off, so all rules are tried.
    ///
    /// Equivalence hook: rewriting with this must produce exactly the same graph as
    /// rewriting with `self`, since an early reject only skips rules that cannot match.
    pub fn without_early_reject(&self) -> Self {
        let segments = self
            .segments
            .iter()
            .map(|segment| Segment {
                early_reject: false,
                rules: segment.rules.iter().map(|(root, _)| (*root, OpMask::EMPTY)).collect(),
                ..segment.clone()
            })
            .collect();
        Self { segments }
    }

    fn rules(&self) -> impl Iterator<Item = &RuleMeta> {
        self.segments.iter().flat_map(|segment| &segment.rules)
    }

    /// Number of registered rules.
    pub fn len(&self) -> usize {
        self.rules().count()
    }

    /// Check if no rules are registered.
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// Number of wildcard rules (tried for every op).
    pub fn wildcard_count(&self) -> usize {
        self.rules().filter(|(root, _)| *root == OpMask::ALL).count()
    }

    /// Number of `OpKey`s some non-wildcard rule is keyed under.
    pub fn indexed_count(&self) -> usize {
        self.rules()
            .filter(|(root, _)| *root != OpMask::ALL)
            .fold(OpMask::EMPTY, |acc, (root, _)| acc.union(*root))
            .count()
    }

    /// Early-reject masks of the rules that can match `key`, in dispatch order.
    ///
    /// Diagnostic view of what each compiled rule demands of a node's direct children.
    pub fn early_rejects(&self, key: &OpKey) -> Vec<OpMask> {
        self.rules().filter(|(root, _)| root.has(key)).map(|(_, reject)| *reject).collect()
    }

    /// Attempt to rewrite a UOp using registered patterns.
    ///
    /// # Tracing
    ///
    /// Enable trace-level logging to see pattern matching activity:
    /// ```bash
    /// RUST_LOG=svod_ir::pattern=trace cargo run
    /// ```
    pub fn rewrite(&self, uop: &Arc<UOp>, ctx: &mut C) -> RewriteResult {
        let key = OpKey::from_op(uop.op());
        let src_ops = uop.src_ops();
        // Consecutive segments often share one guard (`value_sensitive` in symbolic);
        // remember the last verdict so it is evaluated once per distinct guard.
        let mut last_guard: Option<(*const (), bool)> = None;

        for segment in &self.segments {
            if !segment.root.has(&key) {
                continue;
            }
            if let Some(guard) = &segment.guard {
                let id = Arc::as_ptr(guard) as *const ();
                let allowed = match last_guard {
                    Some((last, allowed)) if last == id => allowed,
                    _ => guard(uop),
                };
                last_guard = Some((id, allowed));
                if !allowed {
                    continue;
                }
            }
            let src_ops = if segment.early_reject { src_ops } else { OpMask::ALL };
            let result = match &segment.body {
                Body::Typed(block) => block(uop, src_ops, ctx),
                Body::Unit(block) => block(uop, src_ops, &mut ()),
            };
            if !matches!(result, RewriteResult::NoMatch) {
                tracing::trace!(op_key = ?key, "pattern matched");
                return result;
            }
        }
        RewriteResult::NoMatch
    }
}

impl<C> Clone for SimplifiedPatternMatcher<C> {
    fn clone(&self) -> Self {
        Self { segments: self.segments.clone() }
    }
}

impl<C> Default for SimplifiedPatternMatcher<C> {
    fn default() -> Self {
        Self::new()
    }
}

impl SimplifiedPatternMatcher<()> {
    /// Lift a context-free matcher into any context type.
    ///
    /// `()` rules ignore the context, so they can run under any `D`; the segments are
    /// re-tagged rather than wrapped. This enables combining context-free matchers with
    /// context-dependent ones via `+`:
    ///
    /// ```ignore
    /// let mega = symbolic().with_context::<SomeCtx>()
    ///     + some_ctx_aware_pattern(); // TypedPatternMatcher<SomeCtx>
    /// ```
    pub fn with_context<D: 'static + Send + Sync>(&self) -> SimplifiedPatternMatcher<D> {
        let segments = self
            .segments
            .iter()
            .map(|segment| {
                let block = match &segment.body {
                    Body::Typed(block) | Body::Unit(block) => Arc::clone(block),
                };
                Segment {
                    root: segment.root,
                    guard: segment.guard.clone(),
                    early_reject: segment.early_reject,
                    rules: segment.rules.clone(),
                    body: Body::Unit(block),
                }
            })
            .collect();
        SimplifiedPatternMatcher { segments }
    }
}

impl<C> super::Matcher<C> for SimplifiedPatternMatcher<C> {
    fn rewrite(&self, uop: &Arc<UOp>, ctx: &mut C) -> RewriteResult {
        SimplifiedPatternMatcher::rewrite(self, uop, ctx)
    }
}

impl<C> std::ops::Add for SimplifiedPatternMatcher<C> {
    type Output = Self;

    /// Combine two matchers. Rules from `rhs` come after every rule of `self`.
    fn add(mut self, rhs: Self) -> Self::Output {
        self.segments.extend(rhs.segments);
        self
    }
}

impl<C> std::ops::Add for &SimplifiedPatternMatcher<C> {
    type Output = SimplifiedPatternMatcher<C>;

    fn add(self, rhs: Self) -> Self::Output {
        self.clone() + rhs.clone()
    }
}

impl<C> std::ops::Add<&SimplifiedPatternMatcher<C>> for SimplifiedPatternMatcher<C> {
    type Output = SimplifiedPatternMatcher<C>;

    fn add(self, rhs: &SimplifiedPatternMatcher<C>) -> Self::Output {
        self + rhs.clone()
    }
}

impl<C> std::ops::Add<SimplifiedPatternMatcher<C>> for &SimplifiedPatternMatcher<C> {
    type Output = SimplifiedPatternMatcher<C>;

    fn add(self, rhs: SimplifiedPatternMatcher<C>) -> Self::Output {
        self.clone() + rhs
    }
}

#[cfg(test)]
#[path = "../test/unit/pattern/simplified_internal.rs"]
mod tests;
