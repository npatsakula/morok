//! Code generation for morok tensor operations.
//!
//! This crate provides backend-agnostic code generation infrastructure
//! for converting optimized UOp graphs into executable code.
//!
//! # Architecture
//!
//! - **Traits**: Backend-agnostic interfaces (`Renderer`)
//! - **LLVM**: LLVM IR code generation for CPU execution
//! - **Future**: CUDA, Metal, OpenCL renderers
//!
//! # Usage
//!
//! ```ignore
//! use morok_codegen::llvm;
//!
//! let kernel = llvm::text::render(&optimized_uop_graph, Some("kernel"))?;
//! ```

pub mod c;
pub mod common;
pub mod cranelift;
pub mod error;
pub mod llvm;
pub mod traits;
pub mod types;

#[cfg(test)]
pub mod test;

pub use common::collect_buffers_and_vars;
pub use error::*;
pub use traits::*;
pub use types::*;
