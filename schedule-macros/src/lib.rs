//! Pattern DSL proc-macro for morok-schedule.
//!
//! This crate provides the `patterns!` proc-macro for declarative pattern
//! definitions that compile down to efficient `PatternMatcher` instances.
//!
//! # Arrow Types
//!
//! - `~>` (infallible): RHS returns `Rc<UOp>`, pattern always succeeds if matched
//! - `=>` (fallible): RHS returns `Option<Rc<UOp>>`, pattern may return None
//!
//! # Constant Value Extraction
//!
//! Use `name@const(value)` to automatically extract the ConstValue:
//! - `c@const(cv)` - `c` binds to `&Rc<UOp>`, `cv` binds to `ConstValue`
//! - `_c@const(cv)` - use underscore when you don't need the UOp
//!
//! # Example
//!
//! ```ignore
//! use morok_schedule_macros::patterns;
//!
//! let matcher = patterns! {
//!     // Infallible patterns (~>) - RHS returns Rc<UOp>
//!     Add(x, @zero) ~> x,
//!     Mul(x, @one) ~> x,
//!     And(x, x2) if Rc::ptr_eq(x, x2) ~> Rc::clone(x),
//!
//!     // Automatic ConstValue extraction with c@const(cv)
//!     Cast { src: c@const(cv), dtype } => Some(UOp::const_(dtype.clone(), cv.cast(&dtype)?)),
//!
//!     // Use _c when you only need the value
//!     Neg(_c@const(cv)) if cv.is_zero() ~> UOp::const_(DType::Int32, ConstValue::Int(0)),
//!
//!     // Complex fallible logic with block
//!     Where(cond, t, f) => {
//!         match VminVmaxProperty::get(cond) {
//!             (ConstValue::Bool(true), _) => Some(Rc::clone(t)),
//!             (ConstValue::Bool(false), _) => Some(Rc::clone(f)),
//!             _ => None,
//!         }
//!     },
//! };
//! ```

use proc_macro::TokenStream;
use syn::parse_macro_input;

mod codegen;
mod parser;

#[cfg(test)]
mod test;

use codegen::generate_pattern_matcher;
use parser::PatternList;

/// Proc-macro for declarative pattern definitions.
///
/// Generates a `PatternMatcher` from a list of pattern rewrite rules.
///
/// # Syntax
///
/// ```text
/// pattern_list := pattern_item ("," pattern_item)* ","?
/// pattern_item := rule | for_block
/// rule := lhs guard? arrow rhs
/// guard := "if" expr
/// arrow := "~>" | "=>"
/// lhs := op_pattern | binding_pattern
/// op_pattern := IDENT "{" field_list "}" | IDENT "(" arg_list ")"
/// binding_pattern := IDENT "@" pattern | IDENT "@const(" IDENT ")" | "_" | IDENT
/// rhs := IDENT | expr | "{" block "}"
/// ```
///
/// # Arrow Types
///
/// - `~>` (infallible): RHS returns `Rc<UOp>`, pattern always succeeds if matched
/// - `=>` (fallible): RHS returns `Option<Rc<UOp>>`, pattern may return None
///
/// # Example
///
/// ```ignore
/// let matcher = patterns! {
///     // Infallible patterns
///     Add(x, @zero) ~> x,
///     And(x, x2) if Rc::ptr_eq(x, x2) ~> Rc::clone(x),
///
///     // Automatic ConstValue extraction with c@const(cv)
///     Cast { src: c@const(cv), dtype } => Some(UOp::const_(dtype, cv.cast(&dtype)?)),
/// };
/// ```
#[proc_macro]
pub fn patterns(input: TokenStream) -> TokenStream {
    let pattern_list = parse_macro_input!(input as PatternList);

    match generate_pattern_matcher(&pattern_list) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.to_compile_error().into(),
    }
}
