//! Macros for concise pattern definitions.
//!
//! This module provides the `pattern!` macro for defining rewrite patterns
//! with minimal boilerplate.

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
/// The macro expands to:
///
/// ```ignore
/// patterns.push((
///     UPat::var("x") + UPat::cvar("c"),
///     Box::new(|bindings: &HashMap<String, Rc<UOp>>| {
///         let x = match bindings.get("x") {
///             Some(v) => v,
///             None => return RewriteResult::NoMatch,
///         };
///         let c = match bindings.get("c") {
///             Some(v) => v,
///             None => return RewriteResult::NoMatch,
///         };
///
///         let rewrite_result = (|x, c| {
///             let const_val = get_const_value(c)?;
///             if is_identity_value(&const_val, &BinaryOp::Add, true) {
///                 Some(x.clone())
///             } else {
///                 None
///             }
///         })(x, c);
///
///         match rewrite_result {
///             Some(uop) => RewriteResult::Rewritten(uop),
///             None => RewriteResult::NoMatch,
///         }
///     }) as RewriteFn,
/// ));
/// ```
///
/// # Variable Names
///
/// Variable names in the closure parameters must match the names used in the pattern:
/// - `UPat::var("x")` requires parameter named `x`
/// - `UPat::cvar("c")` requires parameter named `c`
///
/// The macro automatically extracts these bindings from the pattern matcher's HashMap.
#[macro_export]
macro_rules! pattern {
    // Pattern with type annotations: patterns, PATTERN => |var: Type, ...| { body }
    ($patterns:ident, $pattern:expr => |$($var:ident: $ty:ty),* $(,)?| $body:expr) => {
        $patterns.push((
            $pattern,
            Box::new(|bindings: &std::collections::HashMap<String, std::rc::Rc<$crate::UOp>>| {
                // Extract each variable from bindings
                $(
                    let $var: $ty = match bindings.get(stringify!($var)) {
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
            }) as $crate::pattern::matcher::RewriteFn,
        ));
    };

    // Main pattern: patterns, PATTERN => |vars| { body }
    ($patterns:ident, $pattern:expr => |$($var:ident),* $(,)?| $body:expr) => {
        $patterns.push((
            $pattern,
            Box::new(|bindings: &std::collections::HashMap<String, std::rc::Rc<$crate::UOp>>| {
                // Extract each variable from bindings
                $(
                    let $var = match bindings.get(stringify!($var)) {
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
            }) as $crate::pattern::matcher::RewriteFn,
        ));
    };
}
