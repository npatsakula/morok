//! Pattern matching infrastructure for UOp graphs.
//!
//! This module provides the UPat pattern matching DSL and PatternMatcher
//! for applying rewrite rules to UOp graphs.

pub mod matcher;
pub mod upat;

#[macro_use]
pub mod macros;

pub use matcher::{PatternMatcher, RewriteResult};
pub use upat::UPat;
