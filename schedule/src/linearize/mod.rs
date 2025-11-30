//! Linearization module for converting UOp DAGs to linear instruction sequences.
//!
//! This module implements priority-aware topological sorting for control flow,
//! primarily for future GPU/NPU backends that require linear instruction streams.
//!
//! # Architecture
//!
//! ```text
//! Kernel AST (Rc<UOp>)
//!     ↓
//! CFGContext::new(sink)     → Compute control flow edges
//!     ↓
//! linearize(sink)           → Priority-aware toposort
//!     ↓
//! Vec<Rc<UOp>>              → Linear instruction sequence
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use morok_schedule::linearize::{linearize, CFGContext};
//!
//! // For GPU/NPU backends that need linear instruction streams:
//! let cfg = CFGContext::new(&kernel_ast);
//! let instructions = linearize(kernel_ast);
//! ```
//!
//! # Note
//!
//! LLVM backends don't require linearization (they use SSA/basic blocks directly),
//! but this module future-proofs the codebase for GPU/NPU execution.

mod cfg_context;
#[allow(clippy::module_inception)]
mod linearize;

pub use cfg_context::CFGContext;
pub use linearize::linearize;
