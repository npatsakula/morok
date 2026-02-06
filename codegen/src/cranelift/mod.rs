//! Cranelift IR code generation.
//!
//! This module generates Cranelift IR code from optimized UOp graphs for CPU execution.
//! Cranelift provides faster compilation than LLVM at the cost of less optimized output.

pub mod error;
mod helpers;
mod ops;
pub mod renderer;
pub mod types;

pub use error::{Error, Result};
pub use renderer::{CraneliftRenderer, parse_metadata};
