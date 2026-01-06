//! Proc-macros for morok.
//!
//! This crate provides:
//! - `#[derive(PatternEnum)]` for generating pattern matching infrastructure from Op enum
//! - `patterns!` macro for declarative pattern rewrite rules

use proc_macro::TokenStream;
use syn::{DeriveInput, parse_macro_input};

mod pattern_enum;
mod patterns;

/// Derive macro for generating pattern matching infrastructure from an Op enum.
///
/// This macro analyzes your `Op` enum and generates:
/// - `OpKey` enum for O(1) pattern dispatch
/// - `OpKey::from_op()` method to extract the key from an `Op`
/// - `pattern_metadata` module with variant information
///
/// # Usage
///
/// ```ignore
/// #[derive(PatternEnum)]
/// #[pattern(grouped = [Unary, Binary, Ternary])]
/// pub enum Op {
///     Const(ConstValue),
///     Unary(UnaryOp, Arc<UOp>),
///     Binary(BinaryOp, Arc<UOp>, Arc<UOp>),
///     #[pattern(skip)]
///     Invalid,
/// }
/// ```
///
/// # Attributes
///
/// ## Enum-level
///
/// - `#[pattern(grouped = [Variant1, Variant2, ...])]` - Marks variants where the first
///   field is a sub-enum discriminant. For example, `Binary(BinaryOp, ...)` has `BinaryOp`
///   as a sub-discriminant, so `OpKey::Binary(BinaryOp::Add)` differs from `OpKey::Binary(BinaryOp::Mul)`.
///
/// ## Variant-level
///
/// - `#[pattern(skip)]` - Skip pattern generation for this variant (e.g., `Invalid`).
///
/// # Field Type Detection
///
/// The macro automatically classifies field types:
/// - `Arc<UOp>` → child operand (fixed arity)
/// - `SmallVec<[Arc<UOp>; N]>` or `Vec<Arc<UOp>>` → variadic children
/// - `Option<Arc<UOp>>` → optional child
/// - Other types → filter/metadata (e.g., `DType`, `DeviceSpec`)
///
/// # Generated Items
///
/// ```ignore
/// mod pattern_derived {
///     // Discriminant enum for O(1) dispatch
///     pub enum OpKey {
///         Const,
///         Unary(UnaryOp),
///         Binary(BinaryOp),
///         // ...
///     }
///
///     impl OpKey {
///         pub fn from_op(op: &Op) -> Self { ... }
///     }
///
///     pub mod pattern_metadata {
///         pub const BINARY_OPS: &[&str] = &["Add", "Mul", ...];
///         // ...
///     }
/// }
/// ```
#[proc_macro_derive(PatternEnum, attributes(pattern))]
pub fn derive_pattern_enum(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match pattern_enum::generate(&input) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// Proc-macro for declarative pattern rewrite rules.
///
/// Generates a [`SimplifiedPatternMatcher`] from a list of pattern rewrite rules.
/// Patterns are compiled to efficient Rust code with O(1) dispatch via `OpKey`.
///
/// # Syntax Overview
///
/// ```text
/// patterns! {
///     // Basic rule: pattern ~> rewrite (or => for fallible)
///     Add(x, @zero) ~> x,
///
///     // With guard clause
///     Mul(x, y) if is_power_of_two(y) => { ... },
///
///     // For-loop to apply same pattern to multiple ops
///     for op in binary [Add, Mul, Sub] {
///         op(x, @zero) ~> x,
///     }
/// }
/// ```
///
/// # Arrow Types
///
/// - `~>` **Infallible**: Closure returns `Arc<UOp>` directly
/// - `=>` **Fallible**: Closure returns `Option<Arc<UOp>>`
///
/// # Pattern Syntax
///
/// ## Operation Patterns
///
/// ```text
/// Add(x, y)           // Tuple-style: match by position
/// Cast { src, dtype } // Struct-style: match by field name
/// ```
///
/// ## Special Constants
///
/// - `@zero` - Matches constant zero (any numeric type)
/// - `@one` - Matches constant one (any numeric type)
/// - `@const(cv)` - Matches any constant, binds value to `cv: &ConstValue`
/// - `_c@const(cv)` - Underscore prefix: don't bind the UOp, only the value
///
/// ## Duplicate Variables (Auto ptr_eq)
///
/// Same variable name appearing multiple times generates `Arc::ptr_eq` checks:
///
/// ```text
/// Add(x, x) ~> ...    // Matches when both children are the same node
/// Where(x, x, x) ~> ...  // All three must be ptr_eq
/// ```
///
/// ## Commutative Matching
///
/// Square brackets enable commutative matching (tries both orderings):
///
/// ```text
/// Add[x, @zero] ~> x  // Matches Add(x, 0) or Add(0, x)
/// ```
///
/// ## Alternative Patterns
///
/// Match any of several patterns:
///
/// ```text
/// (Add | Sub)(x, @zero) ~> x  // Matches Add(x, 0) or Sub(x, 0)
/// ```
///
/// ## Binding Patterns
///
/// Bind a name to a subpattern:
///
/// ```text
/// result@Add(x, y) => { ... use result, x, y ... }
/// ```
///
/// # For-Loops
///
/// Apply the same pattern template to multiple operations:
///
/// ```text
/// for op in unary [Neg, Not, Sqrt] {
///     op(x) if is_const(x) => { fold_unary(op, x) }
/// }
///
/// for op in binary [Add, Mul, Sub] {
///     op(x, @zero) ~> x,
/// }
/// ```
///
/// # Context Types
///
/// Declare a context type to pass mutable state through patterns:
///
/// ```text
/// patterns! {
///     @context MyContext;
///
///     Add(x, y) => |ctx, x, y| {
///         ctx.record_match();
///         Some(x.clone())
///     }
/// }
/// ```
///
/// # Generated Code
///
/// This macro generates a `SimplifiedPatternMatcher` with:
/// - Compile-time validation of all operation names
/// - O(1) dispatch via OpKey hashmap
/// - Inline pattern matching (no runtime pattern interpretation)
/// - Automatic `Arc::ptr_eq` checks for duplicate variables
///
/// [`SimplifiedPatternMatcher`]: morok_ir::pattern::SimplifiedPatternMatcher
#[proc_macro]
pub fn patterns(input: TokenStream) -> TokenStream {
    let pattern_list = parse_macro_input!(input as patterns::PatternList);

    match patterns::generate_simplified_pattern_matcher(&pattern_list) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.to_compile_error().into(),
    }
}
