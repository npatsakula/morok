//! LLVM IR code generation.
//!
//! This module generates LLVM IR code from optimized UOp graphs for CPU execution.

pub mod helpers;
pub mod ops;
pub mod renderer;
pub mod types;

pub use renderer::{LlvmRenderer, render};
