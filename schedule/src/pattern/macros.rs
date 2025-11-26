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
            Box::new(|bindings: &$crate::pattern::BindingStore, intern: &$crate::pattern::VarIntern| {
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
            }) as $crate::pattern::matcher::RewriteFn,
        ));
    };

    // Main pattern: patterns, PATTERN => |vars| { body }
    ($patterns:ident, $pattern:expr => |$($var:ident),* $(,)?| $body:expr) => {
        $patterns.push((
            $pattern,
            Box::new(|bindings: &$crate::pattern::BindingStore, intern: &$crate::pattern::VarIntern| {
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
            }) as $crate::pattern::matcher::RewriteFn,
        ));
    };
}

/// Define a rewrite pattern with mutable context access.
///
/// This variant of the `pattern!` macro supports patterns that need to access
/// and modify shared context wrapped in `Rc<RefCell<T>>`.
///
/// # Syntax
///
/// ```ignore
/// pattern_ctx_mut!(patterns, ctx_clone,
///     PATTERN => |var1, var2, ctx| {
///         // ctx is &mut T (automatically borrowed via borrow_mut())
///         // Call functions that need &mut T
///         helper_function(bindings, ctx)
///     }
/// );
/// ```
///
/// # Example
///
/// ```ignore
/// use std::cell::RefCell;
/// use std::rc::Rc;
///
/// let ctx = Rc::new(RefCell::new(KernelContext::new()));
/// let mut patterns = vec![];
///
/// let ctx_clone = Rc::clone(&ctx);
/// pattern_ctx_mut!(patterns, ctx_clone,
///     UPat::var("buf") => |buf, ctx| {
///         // ctx is &mut KernelContext
///         debuf(bindings, ctx)
///     }
/// );
/// ```
///
/// # How It Works
///
/// The macro:
/// 1. Creates a closure that captures `ctx_clone`
/// 2. Extracts pattern variable bindings using indexed lookup
/// 3. Calls `ctx_clone.borrow_mut()` to get mutable context
/// 4. Passes context to your rewrite function
/// 5. Returns the rewrite function's result
///
/// # Notes
///
/// - The last parameter in your closure is always the context
/// - The context is automatically borrowed via `borrow_mut()`
/// - The borrow is scoped to minimize holding time
/// - Your function should return `RewriteResult` (not `Option<Rc<UOp>>`)
#[macro_export]
macro_rules! pattern_ctx_mut {
    // Special case: single variable with context to avoid ambiguity
    ($patterns:ident, $ctx:expr, $pattern:expr => |$var:ident, $ctx_var:ident| $body:expr) => {
        {
            let ctx_clone = $ctx;
            $patterns.push((
                $pattern,
                Box::new(move |bindings: &$crate::pattern::BindingStore, intern: &$crate::pattern::VarIntern| {
                    use $crate::pattern::BindingStoreExt;
                    // Extract the variable from bindings
                    let var_name = stringify!($var).trim_start_matches('_');
                    let $var = match intern.get_index(var_name).and_then(|i| bindings.get_by_index(i)) {
                        Some(v) => v,
                        None => return $crate::pattern::matcher::RewriteResult::NoMatch,
                    };

                    // Borrow context mutably and call user's rewrite function
                    let mut $ctx_var = ctx_clone.borrow_mut();
                    (|$var: &std::rc::Rc<$crate::UOp>, $ctx_var: &mut _| $body)($var, &mut *$ctx_var)
                }) as $crate::pattern::matcher::RewriteFn,
            ));
        }
    };

    // General case: multiple variables with context
    ($patterns:ident, $ctx:expr, $pattern:expr => |$($var:ident),+ , $ctx_var:ident| $body:expr) => {
        {
            let ctx_clone = $ctx;
            $patterns.push((
                $pattern,
                Box::new(move |bindings: &$crate::pattern::BindingStore, intern: &$crate::pattern::VarIntern| {
                    use $crate::pattern::BindingStoreExt;
                    // Extract each variable from bindings
                    $(
                        let var_name = stringify!($var).trim_start_matches('_');
                        let $var = match intern.get_index(var_name).and_then(|i| bindings.get_by_index(i)) {
                            Some(v) => v,
                            None => return $crate::pattern::matcher::RewriteResult::NoMatch,
                        };
                    )*

                    // Borrow context mutably and call user's rewrite function
                    let mut $ctx_var = ctx_clone.borrow_mut();
                    (|$($var: &std::rc::Rc<$crate::UOp>),* , $ctx_var: &mut _| $body)($($var),* , &mut *$ctx_var)
                }) as $crate::pattern::matcher::RewriteFn,
            ));
        }
    };
}

/// Define a rewrite pattern with immutable context access.
///
/// This variant of the `pattern!` macro supports patterns that need to read
/// (but not modify) shared context wrapped in `Rc<RefCell<T>>`.
///
/// # Syntax
///
/// ```ignore
/// pattern_ctx!(patterns, ctx_clone,
///     PATTERN => |var1, var2, ctx| {
///         // ctx is &T (automatically borrowed via borrow())
///         // Call functions that need &T
///         let result = helper_function(var1, ctx);
///         Some(result)
///     }
/// );
/// ```
///
/// # Example
///
/// ```ignore
/// use std::cell::RefCell;
/// use std::rc::Rc;
///
/// let ctx = Rc::new(RefCell::new(IndexingContext::new()));
/// let mut patterns = vec![];
///
/// let ctx_clone = Rc::clone(&ctx);
/// pattern_ctx!(patterns, ctx_clone,
///     UPat::var("x") => |x, ctx| {
///         // ctx is &IndexingContext
///         let new_sources = transform_sources_with_bufferize(x, ctx);
///         new_sources.map(|sources| x.with_sources(sources))
///     }
/// );
/// ```
///
/// # How It Works
///
/// The macro:
/// 1. Creates a closure that captures `ctx_clone`
/// 2. Extracts pattern variable bindings using indexed lookup
/// 3. Calls `ctx_clone.borrow()` to get immutable context
/// 4. Passes context to your rewrite function
/// 5. Converts `Option<Rc<UOp>>` to `RewriteResult`
///
/// # Notes
///
/// - The last parameter in your closure is always the context
/// - The context is automatically borrowed via `borrow()`
/// - The borrow is scoped to minimize holding time
/// - Your function should return `Option<Rc<UOp>>` (not `RewriteResult`)
#[macro_export]
macro_rules! pattern_ctx {
    // Pattern with immutable context: patterns, ctx_clone, PATTERN => |vars, ctx| { body }
    ($patterns:ident, $ctx:expr, $pattern:expr => |$($var:ident),* , $ctx_var:ident| $body:expr) => {
        {
            let ctx_clone = $ctx;
            $patterns.push((
                $pattern,
                Box::new(move |bindings: &$crate::pattern::BindingStore, intern: &$crate::pattern::VarIntern| {
                    use $crate::pattern::BindingStoreExt;
                    // Extract each variable from bindings
                    $(
                        let var_name = stringify!($var).trim_start_matches('_');
                        let $var = match intern.get_index(var_name).and_then(|i| bindings.get_by_index(i)) {
                            Some(v) => v,
                            None => return $crate::pattern::matcher::RewriteResult::NoMatch,
                        };
                    )*

                    // Borrow context immutably and call user's rewrite function
                    let $ctx_var = ctx_clone.borrow();
                    let rewrite_result = (|$($var: &std::rc::Rc<$crate::UOp>),* , $ctx_var: &_| -> Option<std::rc::Rc<$crate::UOp>> {
                        $body
                    })($($var),* , &*$ctx_var);

                    // Convert Option to RewriteResult
                    match rewrite_result {
                        Some(uop) => $crate::pattern::matcher::RewriteResult::Rewritten(uop),
                        None => $crate::pattern::matcher::RewriteResult::NoMatch,
                    }
                }) as $crate::pattern::matcher::RewriteFn,
            ));
        }
    };
}
