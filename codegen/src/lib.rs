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
//! let kernel = llvm::render(&optimized_uop_graph, Some("kernel"))?;
//! ```

pub mod context;
pub mod error;
pub mod llvm;
pub mod traits;
pub mod types;

#[cfg(test)]
pub mod test;

pub use context::*;
pub use error::*;
pub use traits::*;
pub use types::*;
