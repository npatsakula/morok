//! LLVM IR code generation.
//!
//! This module generates LLVM IR code from optimized UOp graphs for CPU execution.

pub mod renderer;
pub mod ops;
pub mod types;
pub mod helpers;

pub use renderer::{LlvmRenderer, render};
