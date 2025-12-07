//! Pattern matching infrastructure for UOp graphs.
//!
//! This module provides the UPat pattern matching DSL and PatternMatcher
//! for applying rewrite rules to UOp graphs.

pub mod matcher;
pub mod upat;

#[macro_use]
pub mod macros;

pub use matcher::{FastRewrite, PatternMatcher, RewriteFn, RewriteResult};
pub use upat::{BindingStore, BindingStoreExt, UPat, VarIntern};
