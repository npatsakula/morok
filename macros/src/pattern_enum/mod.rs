//! Derive macro for generating pattern matching infrastructure from Op enum.
//!
//! This module provides `#[derive(PatternEnum)]` which generates:
//! - `OpKey` enum for pattern indexing
//! - Metadata constants for the `patterns!` macro

mod analyze;
mod codegen;
mod parse;
mod typed;

pub use codegen::generate;
pub use typed::expand as expand_op_enum;
