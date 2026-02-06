//! Pattern DSL proc-macro for morok.
//!
//! This module provides the `patterns!` proc-macro for declarative pattern
//! definitions that compile down to efficient `SimplifiedPatternMatcher` instances.

pub mod codegen;
pub mod parser;

#[cfg(test)]
mod test;

pub use codegen::generate_simplified_pattern_matcher;
pub use parser::PatternList;
