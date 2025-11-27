//! Macros for concise pattern definitions.
//!
//! This module provides the `pattern!` macro for defining rewrite patterns
//! with complex closure-based logic.
//!
//! # When to Use `pattern!` vs `patterns!`
//!
//! **Use the `patterns!` proc-macro DSL** for:
//! - Simple declarative patterns with guards: `Add(x, @zero) ~> x`
//! - Struct field extraction: `Bufferize { compute: c, .. } ~> c`
//! - Alternative patterns: `(Add | Mul)(x, y) ~> x`
//! - Permutation patterns: `Add[x, @const] ~> x`
//! - Nested struct patterns: `Index { buffer: Bufferize { compute, .. }, .. }`
//!
//! **Use the `pattern!` macro** for complex patterns requiring:
//! - Iterative logic (filtering/transforming collections)
//! - UOp construction with computed field values
//! - External function calls that return transformed UOps
//! - Multiple sequential operations within the rewrite
//!
//! Example patterns that MUST use `pattern!`:
//! ```ignore
//! // Complex: transforms indices through movement operation
//! pattern!(patterns, idx_pattern => |idx, mop| {
//!     let transformed = apply_movement_op(mop.op(), src_shape, indices);
//!     UOp::index(src, transformed).ok()
//! });
//!
//! // Complex: filters dead axes from ranges
//! pattern!(patterns, UPat::var("buf") => |buf| {
//!     let live_ranges: Vec<_> = ranges.iter().filter(|r| !is_dead_axis(r)).collect();
//!     Some(UOp::bufferize(compute, live_ranges, opts))
//! });
//! ```

/// Define a rewrite pattern with automatic variable binding extraction.
///
/// # Syntax
///
/// ```ignore
/// pattern!(patterns,
///     PATTERN => |var1, var2, ...| {
///         // Rewrite logic that returns Option<Rc<UOp>>
///         // Use ? for early return on None
///         Some(rewritten_uop)
///     }
/// );
/// ```
///
/// # Example
///
/// ```ignore
/// use morok_ir::BinaryOp;
///
/// let mut patterns = vec![];
///
/// // Pattern: x + 0 → x
/// pattern!(patterns,
///     UPat::var("x") + UPat::cvar("c") => |x, c| {
///         let const_val = get_const_value(c)?;
///         if is_identity_value(&const_val, &BinaryOp::Add, true) {
///             Some(x.clone())
///         } else {
///             None
///         }
///     }
/// );
/// ```
///
/// # Expansion
///
/// The macro expands to use the optimized BindingStore for O(1) indexed access:
///
/// ```ignore
/// patterns.push((
///     UPat::var("x") + UPat::cvar("c"),
///     Box::new(|bindings: &BindingStore, intern: &VarIntern| {
///         let x = match intern.get_index("x").and_then(|i| bindings.get_by_index(i)) {
///             Some(v) => v,
///             None => return RewriteResult::NoMatch,
///         };
///         // ...
///     }) as RewriteFn,
/// ));
/// ```
///
/// # Variable Names
///
/// Variable names in the closure parameters must match the names used in the pattern:
/// - `UPat::var("x")` requires parameter named `x`
/// - `UPat::cvar("c")` requires parameter named `c`
#[macro_export]
macro_rules! pattern {
    // Pattern with type annotations: patterns, PATTERN => |var: Type, ...| { body }
    ($patterns:ident, $pattern:expr => |$($var:ident: $ty:ty),* $(,)?| $body:expr) => {
        $patterns.push((
            $pattern,
            Box::new(|bindings: &$crate::pattern::BindingStore, intern: &$crate::pattern::VarIntern, _ctx: &mut ()| {
                use $crate::pattern::BindingStoreExt;
                // Extract each variable from bindings using indexed lookup
                $(
                    // Strip leading underscore from variable name for lookup
                    // (e.g., "_x" -> "x" to match UPat::var("x"))
                    let var_name = stringify!($var).trim_start_matches('_');
                    let $var: $ty = match intern.get_index(var_name).and_then(|i| bindings.get_by_index(i)) {
                        Some(v) => v,
                        None => return $crate::pattern::matcher::RewriteResult::NoMatch,
                    };
                )*

                // Call user's rewrite closure
                let rewrite_result = (|$($var: $ty),*| -> Option<std::rc::Rc<$crate::UOp>> {
                    $body
                })($($var),*);

                // Convert Option to RewriteResult
                match rewrite_result {
                    Some(uop) => $crate::pattern::matcher::RewriteResult::Rewritten(uop),
                    None => $crate::pattern::matcher::RewriteResult::NoMatch,
                }
            }) as $crate::pattern::matcher::RewriteFn<()>,
        ));
    };

    // Main pattern: patterns, PATTERN => |vars| { body }
    ($patterns:ident, $pattern:expr => |$($var:ident),* $(,)?| $body:expr) => {
        $patterns.push((
            $pattern,
            Box::new(|bindings: &$crate::pattern::BindingStore, intern: &$crate::pattern::VarIntern, _ctx: &mut ()| {
                use $crate::pattern::BindingStoreExt;
                // Extract each variable from bindings using indexed lookup
                $(
                    // Strip leading underscore from variable name for lookup
                    // (e.g., "_x" -> "x" to match UPat::var("x"))
                    let var_name = stringify!($var).trim_start_matches('_');
                    let $var = match intern.get_index(var_name).and_then(|i| bindings.get_by_index(i)) {
                        Some(v) => v,
                        None => return $crate::pattern::matcher::RewriteResult::NoMatch,
                    };
                )*

                // Call user's rewrite closure
                let rewrite_result = (|$($var),*| -> Option<std::rc::Rc<$crate::UOp>> {
                    $body
                })($($var),*);

                // Convert Option to RewriteResult
                match rewrite_result {
                    Some(uop) => $crate::pattern::matcher::RewriteResult::Rewritten(uop),
                    None => $crate::pattern::matcher::RewriteResult::NoMatch,
                }
            }) as $crate::pattern::matcher::RewriteFn<()>,
        ));
    };
}

// NOTE: The `pattern_ctx_mut!` and `pattern_ctx!` macros have been deprecated.
// Use the `patterns!` proc-macro DSL with `@context` declaration instead:
//
// ```rust
// let matcher = patterns! {
//     @context KernelContext;
//     buf if matches!(buf.op(), Op::Buffer { .. }) => debuf(buf, ctx),
// };
// ```
//
// This provides compile-time type safety without Rc<RefCell<>> boilerplate.
