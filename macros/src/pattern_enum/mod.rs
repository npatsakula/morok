//! Derive macro for generating pattern matching infrastructure from Op enum.
//!
//! This module provides `#[derive(PatternEnum)]` and `#[op_enum]`, which generate:
//! - `OpKey` enum for pattern indexing, with a dense `index` backing `OpMask`
//! - the `alu` module the `patterns!` macro uses to destructure grouped ops by kind

mod analyze;
mod codegen;
mod parse;
mod typed;

pub use codegen::generate;
pub use typed::expand as expand_op_enum;
