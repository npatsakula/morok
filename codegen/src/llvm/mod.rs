//! LLVM IR code generation.
//!
//! This module generates LLVM IR code from optimized UOp graphs for CPU execution.

mod builders;
pub mod error;
mod intrinsics;

pub mod helpers;
pub mod ops;
pub mod renderer;
pub mod types;

pub use error::{Error, Result};
pub use renderer::{LlvmRenderer, render};
