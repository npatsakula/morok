//! Pattern matcher with O(1) `OpKey` dispatch.
//!
//! # Architecture
//!
//! A matcher is a sequence of segments, each a table of buckets indexed by
//! [`OpKey::index`]. Wildcard rules are copied into every bucket when added, so a
//! bucket already holds its candidates in source order and dispatch is one array index.
//! Composition (`+`) appends segments; a guard (`guarded`) or a context lift
//! (`with_context`) is a property of a segment, checked once per `rewrite` call rather
//! than wrapped around every closure.
//!
//! The `patterns!` macro generates the closures; each carries the early-reject mask of
//! op kinds its fixed-position sources demand of the root's direct children
//! (Tinygrad's `UPat.early_reject`, uop/ops.py:1390).
//!
//! ```ignore
//! let matcher = patterns! {
//!     Add(x, @zero) ~> x,              // bucket Binary(Add)
//!     Mul(x, @one) ~> x,               // bucket Binary(Mul)
//!     x if is_const(x) => fold(x),     // wildcard: every bucket
//! };
//! ```

use std::sync::Arc;

use crate::UOp;
use crate::op::OpMask;
use crate::op::pattern_derived::{OP_KEY_COUNT, OpKey};

use super::RewriteResult;

/// Closure type for pattern matching + rewriting.
pub type PatternClosure<C> = Arc<dyn Fn(&Arc<UOp>, &mut C) -> RewriteResult + Send + Sync>;

type Guard = Arc<dyn Fn(&Arc<UOp>) -> bool + Send + Sync>;

/// One compiled pattern: its early-reject mask and its closure.
struct PatternEntry<C> {
    early_reject: OpMask,
    closure: PatternClosure<C>,
}

impl<C> Clone for PatternEntry<C> {
    fn clone(&self) -> Self {
        Self { early_reject: self.early_reject, closure: Arc::clone(&self.closure) }
    }
}

/// Buckets indexed by `OpKey::index()`, each holding its candidates in source order.
type Buckets<C> = Vec<Vec<PatternEntry<C>>>;

fn empty_buckets<C>() -> Buckets<C> {
    (0..OP_KEY_COUNT).map(|_| Vec::new()).collect()
}

/// Entries of a segment, either taking the matcher's context or lifted from `()`.
enum Entries<C> {
    Typed(Buckets<C>),
    Unit(Buckets<()>),
}

impl<C> Clone for Entries<C> {
    fn clone(&self) -> Self {
        match self {
            Self::Typed(buckets) => Self::Typed(buckets.clone()),
            Self::Unit(buckets) => Self::Unit(buckets.clone()),
        }
    }
}

struct Segment<C> {
    guard: Option<Guard>,
    entries: Entries<C>,
}

impl<C> Clone for Segment<C> {
    fn clone(&self) -> Self {
        Self { guard: self.guard.clone(), entries: self.entries.clone() }
    }
}

impl<C> Segment<C> {
    fn map_masks(&self, f: impl Fn(OpMask) -> OpMask) -> Self {
        fn map<C>(buckets: &Buckets<C>, f: &impl Fn(OpMask) -> OpMask) -> Buckets<C> {
            buckets
                .iter()
                .map(|bucket| {
                    bucket
                        .iter()
                        .map(|entry| PatternEntry {
                            early_reject: f(entry.early_reject),
                            closure: Arc::clone(&entry.closure),
                        })
                        .collect()
                })
                .collect()
        }
        let entries = match &self.entries {
            Entries::Typed(buckets) => Entries::Typed(map(buckets, &f)),
            Entries::Unit(buckets) => Entries::Unit(map(buckets, &f)),
        };
        Self { guard: self.guard.clone(), entries }
    }

    fn masks(&self, key: OpKey) -> Vec<OpMask> {
        match &self.entries {
            Entries::Typed(buckets) => buckets[key.index()].iter().map(|entry| entry.early_reject).collect(),
            Entries::Unit(buckets) => buckets[key.index()].iter().map(|entry| entry.early_reject).collect(),
        }
    }

    fn bucket_len(&self, index: usize) -> usize {
        match &self.entries {
            Entries::Typed(buckets) => buckets[index].len(),
            Entries::Unit(buckets) => buckets[index].len(),
        }
    }

    fn is_open(&self) -> bool {
        self.guard.is_none() && matches!(self.entries, Entries::Typed(_))
    }
}

/// Pattern matcher: rules bucketed by root `OpKey`, tried in source order.
///
/// `C` is the context type passed to every closure; `()` for stateless matching.
pub struct SimplifiedPatternMatcher<C = ()> {
    segments: Vec<Segment<C>>,
    len: usize,
    wildcards: usize,
}

impl<C> SimplifiedPatternMatcher<C> {
    /// Create a new empty pattern matcher.
    pub fn new() -> Self {
        Self { segments: Vec::new(), len: 0, wildcards: 0 }
    }

    /// The trailing unguarded, context-typed segment, opened if needed.
    fn open_buckets(&mut self) -> &mut Buckets<C> {
        if !self.segments.last().is_some_and(Segment::is_open) {
            self.segments.push(Segment { guard: None, entries: Entries::Typed(empty_buckets()) });
        }
        match &mut self.segments.last_mut().expect("segment just opened").entries {
            Entries::Typed(buckets) => buckets,
            Entries::Unit(_) => unreachable!("open segment is context-typed"),
        }
    }

    /// Add pattern for specific OpKey(s); an empty `keys` registers a wildcard.
    pub fn add<F>(&mut self, keys: &[OpKey], closure: F)
    where
        F: Fn(&Arc<UOp>, &mut C) -> RewriteResult + Send + Sync + 'static,
    {
        self.add_rejecting(keys, &[], closure);
    }

    /// Add a pattern that can only match when the root's direct children include every
    /// op kind in `early_reject`; the closure is skipped otherwise.
    ///
    /// `early_reject` must be a *necessary* condition for the closure to match, i.e. the
    /// union of the op kinds demanded by the pattern's fixed-position sources. Sources
    /// that accept several kinds — or any kind — contribute nothing, exactly as in
    /// Tinygrad's `UPat.early_reject` (uop/ops.py:1390).
    pub fn add_rejecting<F>(&mut self, keys: &[OpKey], early_reject: &[OpKey], closure: F)
    where
        F: Fn(&Arc<UOp>, &mut C) -> RewriteResult + Send + Sync + 'static,
    {
        let entry = PatternEntry { early_reject: early_reject.iter().copied().collect(), closure: Arc::new(closure) };
        self.len += 1;
        if keys.is_empty() {
            self.wildcards += 1;
            for bucket in self.open_buckets() {
                bucket.push(entry.clone());
            }
        } else {
            let buckets = self.open_buckets();
            for key in keys {
                buckets[key.index()].push(entry.clone());
            }
        }
    }

    /// Add wildcard pattern (tried for every op, in source order).
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
                Segment { guard: Some(guard), entries: segment.entries.clone() }
            })
            .collect();
        Self { segments, len: self.len, wildcards: self.wildcards }
    }

    /// Copy of this matcher with every early reject cleared, so all entries are dispatched.
    ///
    /// Equivalence hook: rewriting with this must produce exactly the same graph as
    /// rewriting with `self`, since an early reject only skips entries that cannot match.
    pub fn without_early_reject(&self) -> Self {
        let segments = self.segments.iter().map(|segment| segment.map_masks(|_| OpMask::EMPTY)).collect();
        Self { segments, len: self.len, wildcards: self.wildcards }
    }

    /// Number of registered patterns.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if no patterns are registered.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of wildcard patterns (tried for every op).
    pub fn wildcard_count(&self) -> usize {
        self.wildcards
    }

    /// Number of `OpKey`s that have at least one non-wildcard candidate.
    pub fn indexed_count(&self) -> usize {
        (0..OP_KEY_COUNT)
            .filter(|&index| self.segments.iter().any(|segment| segment.bucket_len(index) > self.wildcards))
            .count()
    }

    /// Early-reject masks of the candidates for `key`, in dispatch order.
    ///
    /// Diagnostic view of what each compiled pattern demands of a node's direct children.
    pub fn early_rejects(&self, key: &OpKey) -> Vec<OpMask> {
        self.segments.iter().flat_map(|segment| segment.masks(*key)).collect()
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
        let index = OpKey::from_op(uop.op()).index();
        let src_ops = uop.src_ops();
        // Consecutive segments often share one guard (`value_sensitive` in symbolic);
        // remember the last verdict so it is evaluated once per distinct guard.
        let mut last_guard: Option<(*const (), bool)> = None;

        for segment in &self.segments {
            if segment.bucket_len(index) == 0 {
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
            let result = match &segment.entries {
                Entries::Typed(buckets) => try_entries(&buckets[index], src_ops, uop, ctx),
                Entries::Unit(buckets) => try_entries(&buckets[index], src_ops, uop, &mut ()),
            };
            if !matches!(result, RewriteResult::NoMatch) {
                tracing::trace!(op_key = ?OpKey::from_op(uop.op()), "pattern matched");
                return result;
            }
        }
        RewriteResult::NoMatch
    }
}

/// First candidate whose early reject passes and whose closure rewrites.
fn try_entries<C>(entries: &[PatternEntry<C>], src_ops: OpMask, uop: &Arc<UOp>, ctx: &mut C) -> RewriteResult {
    for entry in entries.iter().filter(|entry| entry.early_reject.is_subset_of(src_ops)) {
        let result = (entry.closure)(uop, ctx);
        if !matches!(result, RewriteResult::NoMatch) {
            return result;
        }
    }
    RewriteResult::NoMatch
}

impl<C> Clone for SimplifiedPatternMatcher<C> {
    fn clone(&self) -> Self {
        Self { segments: self.segments.clone(), len: self.len, wildcards: self.wildcards }
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
    /// `()` patterns ignore the context, so they can run under any `D`; the segments are
    /// re-tagged rather than each closure re-wrapped. This enables combining
    /// context-free matchers with context-dependent ones via `+`:
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
                let buckets = match &segment.entries {
                    Entries::Typed(buckets) | Entries::Unit(buckets) => buckets.clone(),
                };
                Segment { guard: segment.guard.clone(), entries: Entries::Unit(buckets) }
            })
            .collect();
        SimplifiedPatternMatcher { segments, len: self.len, wildcards: self.wildcards }
    }
}

impl<C> super::Matcher<C> for SimplifiedPatternMatcher<C> {
    fn rewrite(&self, uop: &Arc<UOp>, ctx: &mut C) -> RewriteResult {
        SimplifiedPatternMatcher::rewrite(self, uop, ctx)
    }
}

impl<C> std::ops::Add for SimplifiedPatternMatcher<C> {
    type Output = Self;

    /// Combine two matchers. Patterns from `rhs` are appended.
    fn add(mut self, rhs: Self) -> Self::Output {
        let mut incoming = rhs.segments.into_iter().peekable();
        // Two adjacent unguarded segments of one kind fold into a single table, so a
        // long `+` chain stays short.
        if let (Some(last), Some(first)) = (self.segments.last_mut(), incoming.peek())
            && last.guard.is_none()
            && first.guard.is_none()
        {
            match (&mut last.entries, &first.entries) {
                (Entries::Typed(mine), Entries::Typed(_)) => {
                    let Some(Segment { entries: Entries::Typed(theirs), .. }) = incoming.next() else { unreachable!() };
                    mine.iter_mut().zip(theirs).for_each(|(bucket, more)| bucket.extend(more));
                }
                (Entries::Unit(mine), Entries::Unit(_)) => {
                    let Some(Segment { entries: Entries::Unit(theirs), .. }) = incoming.next() else { unreachable!() };
                    mine.iter_mut().zip(theirs).for_each(|(bucket, more)| bucket.extend(more));
                }
                _ => {}
            }
        }
        self.segments.extend(incoming);
        self.len += rhs.len;
        self.wildcards += rhs.wildcards;
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
