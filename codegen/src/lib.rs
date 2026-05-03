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
//! use morok_codegen::{llvm, program_pipeline};
//!
//! let linear = morok_ir::UOp::linear(morok_schedule::linearize_with_cfg(optimized_uop_graph).into());
//! let kernel = llvm::text::render(&linear, Some("kernel"))?;
//! // Canonical staged flow: PROGRAM -> LINEAR -> SOURCE -> BINARY.
//! // See `program_pipeline` for the strict staged entrypoints.
//! ```
//!
//! # Pre-render invariants
//!
//! Direct callers of [`Renderer::render`] (and the per-backend `render` free
//! functions) must pass a LINEAR-stage UOp produced by
//! [`morok_schedule::linearize::line_rewrite_cleanups`]. The cleanup pass
//! lowers gated LOADs into IF/STORE/ENDIF and provides the `alt` value that
//! per-backend op handlers rely on; backends report `Error::InvalidGraph` if
//! these invariants are violated. The staged entrypoints in
//! [`program_pipeline`] run the cleanup pass automatically.

pub mod c;
pub mod common;
pub mod error;
pub mod llvm;
#[cfg(feature = "mlir")]
pub mod mlir;
pub mod program_pipeline;
pub mod traits;
pub mod types;

#[cfg(test)]
pub mod test;

pub use common::collect_buffers_and_vars;
pub use error::*;
pub use traits::*;
pub use types::*;
