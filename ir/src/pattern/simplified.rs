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
pub type PatternClosure<C> = Box<dyn Fn(&Arc<UOp>, &mut C) -> RewriteResult + Send + Sync>;

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
            self.wildcards.push(Box::new(closure));
        } else if keys.len() == 1 {
            // Single key - store directly
            self.indexed.entry(keys[0].clone()).or_default().push(Box::new(closure));
        } else {
            // Multiple keys - need to share the closure via Arc
            let shared = Arc::new(closure);
            for key in keys {
                let shared_clone = Arc::clone(&shared);
                self.indexed
                    .entry(key.clone())
                    .or_default()
                    .push(Box::new(move |uop: &Arc<UOp>, ctx: &mut C| shared_clone(uop, ctx)));
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
        self.wildcards.push(Box::new(closure));
    }

    /// Number of registered patterns.
    pub fn len(&self) -> usize {
        self.indexed.values().map(|v| v.len()).sum::<usize>() + self.wildcards.len()
    }

    /// Check if no patterns are registered.
    pub fn is_empty(&self) -> bool {
        self.indexed.is_empty() && self.wildcards.is_empty()
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

impl<C> Default for SimplifiedPatternMatcher<C> {
    fn default() -> Self {
        Self::new()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::BinaryOp;
    use crate::{ConstValue, Op, UOp};
    use morok_dtype::DType;

    fn const_int(v: i64) -> Arc<UOp> {
        UOp::const_(DType::Int32, ConstValue::Int(v))
    }

    fn binary(op: BinaryOp, lhs: Arc<UOp>, rhs: Arc<UOp>) -> Arc<UOp> {
        // Use UOp::new to create binary ops directly for tests
        UOp::new(Op::Binary(op, lhs, rhs), DType::Int32)
    }

    #[test]
    fn test_empty_matcher() {
        let matcher = SimplifiedPatternMatcher::<()>::new();
        assert!(matcher.is_empty());
        assert_eq!(matcher.len(), 0);
    }

    #[test]
    fn test_add_indexed_pattern() {
        let mut matcher = SimplifiedPatternMatcher::<()>::new();

        matcher.add(&[OpKey::Binary(BinaryOp::Add)], |_uop, _ctx| RewriteResult::NoMatch);

        assert_eq!(matcher.len(), 1);
        assert!(!matcher.is_empty());
    }

    #[test]
    fn test_add_wildcard_pattern() {
        let mut matcher = SimplifiedPatternMatcher::<()>::new();

        matcher.add_wildcard(|_uop, _ctx| RewriteResult::NoMatch);

        assert_eq!(matcher.len(), 1);
        assert_eq!(matcher.wildcards.len(), 1);
    }

    #[test]
    fn test_combine_matchers() {
        let mut m1 = SimplifiedPatternMatcher::<()>::new();
        m1.add(&[OpKey::Binary(BinaryOp::Add)], |_, _| RewriteResult::NoMatch);

        let mut m2 = SimplifiedPatternMatcher::<()>::new();
        m2.add(&[OpKey::Binary(BinaryOp::Mul)], |_, _| RewriteResult::NoMatch);

        let combined = m1 + m2;
        assert_eq!(combined.len(), 2);
    }

    #[test]
    fn test_rewrite_basic() {
        let mut matcher = SimplifiedPatternMatcher::<()>::new();

        // Pattern: Add(x, 0) -> x
        matcher.add(&[OpKey::Binary(BinaryOp::Add)], |uop, _ctx| {
            let Op::Binary(BinaryOp::Add, left, right) = uop.op() else {
                return RewriteResult::NoMatch;
            };
            // Check if right is zero
            if let Op::Const(cv) = right.op()
                && cv.0.is_zero()
            {
                return RewriteResult::Rewritten(left.clone());
            }
            // Check if left is zero (commutative)
            if let Op::Const(cv) = left.op()
                && cv.0.is_zero()
            {
                return RewriteResult::Rewritten(right.clone());
            }
            RewriteResult::NoMatch
        });

        // Test: 5 + 0 -> 5
        let five = const_int(5);
        let zero = const_int(0);
        let expr = binary(BinaryOp::Add, five.clone(), zero);

        let result = matcher.rewrite(&expr, &mut ());
        assert!(matches!(result, RewriteResult::Rewritten(ref r) if Arc::ptr_eq(r, &five)));

        // Test: 0 + 5 -> 5
        let expr2 = binary(BinaryOp::Add, const_int(0), five.clone());
        let result2 = matcher.rewrite(&expr2, &mut ());
        assert!(matches!(result2, RewriteResult::Rewritten(ref r) if Arc::ptr_eq(r, &five)));

        // Test: 3 + 4 -> NoMatch
        let expr3 = binary(BinaryOp::Add, const_int(3), const_int(4));
        let result3 = matcher.rewrite(&expr3, &mut ());
        assert!(matches!(result3, RewriteResult::NoMatch));
    }

    #[test]
    fn test_wildcard_after_indexed() {
        let mut matcher = SimplifiedPatternMatcher::<()>::new();

        // Indexed pattern that doesn't match
        matcher.add(&[OpKey::Binary(BinaryOp::Add)], |_uop, _ctx| RewriteResult::NoMatch);

        // Wildcard that matches everything
        matcher.add_wildcard(|uop, _ctx| RewriteResult::Rewritten(uop.clone()));

        let expr = binary(BinaryOp::Add, const_int(1), const_int(2));

        // Should fall through to wildcard
        let result = matcher.rewrite(&expr, &mut ());
        assert!(matches!(result, RewriteResult::Rewritten(_)));
    }
}
