//! UOp decomposition framework.
//!
//! This module provides conditional decomposition of complex operations into
//! simpler primitives that all backends can handle. Backends that don't support
//! certain transcendental operations can use the pattern-based decompositor
//! to transform them into equivalent primitive operations.
//!
//! # Architecture
//!
//! 1. **Backend provides decomposition patterns** via `Renderer::decompositor()`
//! 2. **Decomposition pass** uses `graph_rewrite_bottom_up` to apply patterns
//! 3. **Each pattern** transforms one op into a subtree of primitive ops
//!
//! # Example
//!
//! ```ignore
//! // In tensor realization, before rendering:
//! if let Some(decompositor) = renderer.decompositor() {
//!     let ast = decompose_with(&kernel.ast, &decompositor);
//! }
//! let rendered = renderer.render(&ast)?;
//! ```

pub mod helpers;
pub mod transcendentals;

use std::sync::Arc;

use crate::pattern::PatternMatcher;
use crate::rewrite::graph_rewrite_bottom_up;
use crate::uop::UOp;
use morok_schedule_macros::patterns;

use transcendentals::{xcos, xerf, xexp, xexp2, xlog, xlog2, xpow, xrsqrt, xsin, xsqrt, xtan};

/// All decomposition patterns for transcendental operations.
///
/// Returns a `PatternMatcher` that decomposes:
/// - Unary: Exp2, Log2, Exp, Log, Sin, Cos, Tan, Sqrt, Rsqrt, Erf
/// - Binary: Pow
///
/// Backends that don't support these operations natively can use this
/// matcher with `decompose_with()` to decompose them into primitives.
///
/// # Example
///
/// ```ignore
/// impl Renderer for CpuRenderer {
///     fn decompositor(&self) -> Option<PatternMatcher<()>> {
///         Some(all_decomposition_patterns())
///     }
/// }
/// ```
pub fn all_decomposition_patterns() -> PatternMatcher<()> {
    patterns! {
        // Transcendental unary ops
        Exp2(src) ~> xexp2(src),
        Log2(src) ~> xlog2(src),
        Exp(src)  ~> xexp(src),
        Log(src)  ~> xlog(src),
        Sin(src)  ~> xsin(src),
        Cos(src)  ~> xcos(src),
        Tan(src)  ~> xtan(src),
        Sqrt(src) ~> xsqrt(src),
        Rsqrt(src) ~> xrsqrt(src),
        Erf(src)  ~> xerf(src),

        // Binary pow: x^y = exp2(y * log2(x))
        Pow(base, exp) ~> xpow(base, exp),
    }
}

/// Apply decomposition to a UOp graph using the provided pattern matcher.
///
/// Uses `graph_rewrite_bottom_up` to traverse the graph and apply decomposition
/// patterns. This ensures children are processed before parents, which is
/// important for recursive decomposition (e.g., when a decomposition result
/// contains more operations that need decomposition).
///
/// # Arguments
///
/// * `root` - The root UOp of the graph to decompose
/// * `matcher` - The pattern matcher containing decomposition rules
///
/// # Returns
///
/// A new UOp graph with matched operations replaced by their decompositions.
///
/// # Example
///
/// ```ignore
/// let matcher = all_decomposition_patterns();
/// let decomposed = decompose_with(&kernel.ast, &matcher);
/// ```
pub fn decompose_with(root: &Arc<UOp>, matcher: &PatternMatcher<()>) -> Arc<UOp> {
    graph_rewrite_bottom_up(matcher, root.clone(), &mut ())
}
