//! High-performance pattern matcher with OpKey-based O(1) dispatch.
//!
//! # Architecture
//!
//! `SimplifiedPatternMatcher` uses a two-tier dispatch strategy:
//!
//! 1. **Indexed patterns**: Stored in a `HashMap<OpKey, Vec<Closure>>` for O(1) lookup
//! 2. **Wildcard patterns**: Tried after indexed patterns for ops without specific patterns
//!
//! The `patterns!` macro generates closures that use native Rust `match` expressions,
//! avoiding runtime pattern interpretation overhead.
//!
//! # Performance
//!
//! - O(1) dispatch to relevant patterns via `OpKey`
//! - Only patterns matching the input's `OpKey` are tried
//! - Wildcard patterns act as fallback for unmatched ops
//! - 5-10x faster than linear pattern scanning
//!
//! # Usage with patterns! macro
//!
//! ```ignore
//! use morok_macros::patterns;
//!
//! let matcher = patterns! {
//!     Add(x, @zero) ~> x,              // Indexed under OpKey::Binary(BinaryOp::Add)
//!     Mul(x, @one) ~> x,               // Indexed under OpKey::Binary(BinaryOp::Mul)
//!     x if is_const(x) => fold(x),     // Wildcard - tried for all ops
//! };
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use crate::UOp;
use crate::op::pattern_derived::OpKey;

use super::RewriteResult;

/// Closure type for pattern matching + rewriting.
///
/// Takes a UOp and mutable context, returns a RewriteResult.
/// Uses `Arc` instead of `Box` to enable `Clone` on `SimplifiedPatternMatcher`,
/// which is needed for caching combined matchers via `LazyLock`.
pub type PatternClosure<C> = Arc<dyn Fn(&Arc<UOp>, &mut C) -> RewriteResult + Send + Sync>;

/// High-performance pattern matcher with O(1) OpKey-based dispatch.
///
/// # Design
///
/// Instead of a single list of patterns that must be linearly scanned,
/// patterns are indexed by their `OpKey` in a `HashMap`. When matching:
///
/// 1. Extract `OpKey` from the input UOp
/// 2. Look up patterns for that key (O(1) HashMap lookup)
/// 3. Try only those patterns (typically 1-3 per key)
/// 4. Fall back to wildcard patterns if no match
///
/// # Type Parameter
///
/// - `C`: Context type passed to all pattern closures. Use `()` for stateless matching.
///
/// # Example
///
/// Typically used via the `patterns!` macro:
///
/// ```ignore
/// use morok_macros::patterns;
///
/// let matcher = patterns! {
///     Add(x, @zero) ~> x,
///     Mul(x, @one) ~> x,
/// };
///
/// // Use with graph_rewrite
/// let result = graph_rewrite(&ast, &matcher, &mut ());
/// ```
///
/// Manual construction (rarely needed):
///
/// ```ignore
/// let mut matcher = SimplifiedPatternMatcher::<()>::new();
/// matcher.add(
///     &[OpKey::Binary(BinaryOp::Add)],
///     |uop, _ctx| {
///         let Op::Binary(BinaryOp::Add, left, right) = uop.op() else {
///             return RewriteResult::NoMatch;
///         };
///         if is_zero(right) { RewriteResult::Rewritten(left.clone()) }
///         else { RewriteResult::NoMatch }
///     }
/// );
/// ```
pub struct SimplifiedPatternMatcher<C = ()> {
    /// Patterns indexed by OpKey - tried first for O(1) dispatch
    indexed: HashMap<OpKey, Vec<PatternClosure<C>>>,
    /// Wildcard patterns - tried after indexed patterns
    wildcards: Vec<PatternClosure<C>>,
}

impl<C> SimplifiedPatternMatcher<C> {
    /// Create a new empty pattern matcher.
    pub fn new() -> Self {
        Self { indexed: HashMap::new(), wildcards: Vec::new() }
    }

    /// Add pattern for specific OpKey(s).
    ///
    /// If `keys` is empty, the pattern is treated as a wildcard and will be
    /// tried for every UOp after all indexed patterns have been tried.
    pub fn add<F>(&mut self, keys: &[OpKey], closure: F)
    where
        F: Fn(&Arc<UOp>, &mut C) -> RewriteResult + Send + Sync + 'static,
    {
        if keys.is_empty() {
            // No keys = wildcard pattern
            self.wildcards.push(Arc::new(closure));
        } else if keys.len() == 1 {
            // Single key - store directly
            self.indexed.entry(keys[0].clone()).or_default().push(Arc::new(closure));
        } else {
            // Multiple keys - share the closure via Arc clone
            let shared: PatternClosure<C> = Arc::new(closure);
            for key in keys {
                self.indexed.entry(key.clone()).or_default().push(Arc::clone(&shared));
            }
        }
    }

    /// Add wildcard pattern (matches any op).
    ///
    /// Wildcard patterns are tried after all indexed patterns have been tried.
    pub fn add_wildcard<F>(&mut self, closure: F)
    where
        F: Fn(&Arc<UOp>, &mut C) -> RewriteResult + Send + Sync + 'static,
    {
        self.wildcards.push(Arc::new(closure));
    }

    /// Number of registered patterns.
    pub fn len(&self) -> usize {
        self.indexed.values().map(|v| v.len()).sum::<usize>() + self.wildcards.len()
    }

    /// Check if no patterns are registered.
    pub fn is_empty(&self) -> bool {
        self.indexed.is_empty() && self.wildcards.is_empty()
    }

    /// Number of wildcard patterns (tried for every op).
    pub fn wildcard_count(&self) -> usize {
        self.wildcards.len()
    }

    /// Number of indexed buckets (unique OpKeys with patterns).
    pub fn indexed_count(&self) -> usize {
        self.indexed.len()
    }

    /// Attempt to rewrite a UOp using registered patterns.
    ///
    /// This is an inherent method that provides the same functionality as
    /// `Matcher::rewrite()` without requiring the trait to be in scope.
    ///
    /// # Tracing
    ///
    /// Enable debug-level tracing to see pattern matching activity:
    /// ```bash
    /// RUST_LOG=morok_ir::pattern=debug cargo run
    /// ```
    pub fn rewrite(&self, uop: &Arc<UOp>, ctx: &mut C) -> RewriteResult {
        let key = OpKey::from_op(uop.op());

        // Try patterns indexed by this OpKey
        if let Some(patterns) = self.indexed.get(&key) {
            let pattern_count = patterns.len();
            tracing::trace!(op_key = ?key, pattern_count, "trying indexed patterns");

            for (idx, closure) in patterns.iter().enumerate() {
                let result = closure(uop, ctx);
                if !matches!(result, RewriteResult::NoMatch) {
                    tracing::debug!(op_key = ?key, pattern_idx = idx, "pattern matched");
                    return result;
                }
            }
        }

        // Try wildcard patterns
        if !self.wildcards.is_empty() {
            tracing::trace!(wildcard_count = self.wildcards.len(), "trying wildcard patterns");

            for (idx, closure) in self.wildcards.iter().enumerate() {
                let result = closure(uop, ctx);
                if !matches!(result, RewriteResult::NoMatch) {
                    tracing::debug!(wildcard_idx = idx, "wildcard pattern matched");
                    return result;
                }
            }
        }

        RewriteResult::NoMatch
    }
}

impl<C> Clone for SimplifiedPatternMatcher<C> {
    fn clone(&self) -> Self {
        Self { indexed: self.indexed.clone(), wildcards: self.wildcards.clone() }
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
    /// Since `()` patterns ignore the context parameter, they can safely run
    /// under any `D`. Each closure is re-wrapped to discard `&mut D` and pass
    /// `&mut ()` to the original. This enables combining context-free matchers
    /// with context-dependent ones via `+`:
    ///
    /// ```ignore
    /// let mega = symbolic().with_context::<PcontigConfig>()
    ///     + buffer_removal_with_pcontig(); // TypedPatternMatcher<PcontigConfig>
    /// ```
    pub fn with_context<D: 'static + Send + Sync>(&self) -> SimplifiedPatternMatcher<D> {
        let mut result = SimplifiedPatternMatcher::<D>::new();
        for (key, closures) in &self.indexed {
            for closure in closures {
                let closure = Arc::clone(closure);
                result
                    .indexed
                    .entry(key.clone())
                    .or_default()
                    .push(Arc::new(move |uop: &Arc<UOp>, _ctx: &mut D| closure(uop, &mut ())));
            }
        }
        for closure in &self.wildcards {
            let closure = Arc::clone(closure);
            result.wildcards.push(Arc::new(move |uop: &Arc<UOp>, _ctx: &mut D| closure(uop, &mut ())));
        }
        result
    }
}

// Implement Matcher trait for graph_rewrite compatibility
impl<C> super::Matcher<C> for SimplifiedPatternMatcher<C> {
    fn rewrite(&self, uop: &Arc<UOp>, ctx: &mut C) -> RewriteResult {
        // Delegate to inherent method
        SimplifiedPatternMatcher::rewrite(self, uop, ctx)
    }
}

// Implement Add<Self> for composition (matcher1 + matcher2)
impl<C> std::ops::Add for SimplifiedPatternMatcher<C> {
    type Output = Self;

    /// Combine two matchers. Patterns from `rhs` are appended.
    fn add(mut self, rhs: Self) -> Self::Output {
        // Merge indexed patterns
        for (key, patterns) in rhs.indexed {
            self.indexed.entry(key).or_default().extend(patterns);
        }
        // Merge wildcards
        self.wildcards.extend(rhs.wildcards);
        self
    }
}

// Implement Add for references — clones both sides then combines.
// Enables `pm_a() + pm_b()` when both return `&'static TypedPatternMatcher`.
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
