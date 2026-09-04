//! Pattern matching infrastructure for UOp graphs.
//!
//! This module provides pattern matching using `SimplifiedPatternMatcher`, which
//! uses closures that do inline Rust pattern matching. The `patterns!` macro
//! generates closures with native `match` expressions for O(1) OpKey dispatch.

pub mod helpers;
pub mod simplified;

use crate::UOp;
use std::sync::Arc;

// =============================================================================
// RewriteResult - Result of pattern matching
// =============================================================================

/// Result of applying a pattern rewrite.
#[derive(Debug, Clone)]
pub enum RewriteResult {
    /// Pattern didn't match or rewrite function declined to rewrite
    NoMatch,
    /// Pattern matched and returned a replacement UOp
    Rewritten(Arc<UOp>),
    /// Pattern matched and indicates bottom-up gate (Tinygrad's BottomUpGate)
    /// This signals that children should be processed before proceeding
    Gate(Arc<UOp>),
}

/// What a rule's right-hand side may evaluate to.
///
/// `Arc<UOp>` always rewrites, `Option<Arc<UOp>>` declines with `None`, and a
/// `RewriteResult` passes through so a rule can also `Gate`.
pub trait IntoRewriteResult {
    fn into_rewrite_result(self) -> RewriteResult;
}

impl IntoRewriteResult for Arc<UOp> {
    #[inline]
    fn into_rewrite_result(self) -> RewriteResult {
        RewriteResult::Rewritten(self)
    }
}

impl IntoRewriteResult for Option<Arc<UOp>> {
    #[inline]
    fn into_rewrite_result(self) -> RewriteResult {
        self.map_or(RewriteResult::NoMatch, RewriteResult::Rewritten)
    }
}

impl IntoRewriteResult for RewriteResult {
    #[inline]
    fn into_rewrite_result(self) -> RewriteResult {
        self
    }
}

// =============================================================================
// Pattern Exports
// =============================================================================

pub use helpers::{const_matches, is_any_const, is_neg_one, is_nonzero, is_one, is_zero, try_const};
pub use simplified::{BlockFn, RuleMeta, SimplifiedPatternMatcher};

/// Type alias for backwards compatibility.
pub type TypedPatternMatcher<C = ()> = SimplifiedPatternMatcher<C>;

// =============================================================================
// Matcher Trait - Unified interface for pattern matchers
// =============================================================================

/// Trait for pattern matchers used by the rewrite engine.
///
/// This trait provides a unified interface for pattern matching,
/// allowing the rewrite engine to work with different matcher implementations.
pub trait Matcher<C> {
    /// Attempt to rewrite a UOp using registered patterns.
    fn rewrite(&self, uop: &Arc<UOp>, ctx: &mut C) -> RewriteResult;
}

impl<C, M: Matcher<C> + ?Sized> Matcher<C> for &M {
    fn rewrite(&self, uop: &Arc<UOp>, ctx: &mut C) -> RewriteResult {
        (**self).rewrite(uop, ctx)
    }
}
