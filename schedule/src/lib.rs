//! Schedule module for Morok compiler.
//!
//! This module implements optimization passes for the IR,
//! including symbolic simplification and graph transformations.
//!
//! # Module Organization
//!
//! - [`symbolic`] - Symbolic simplification patterns
//! - [`rangeify`] - RANGEIFY transformation (movement ops → kernels)
//!   - Phases 1-4: Movement ops to BUFFERIZE with symbolic simplification
//!   - Phase 5: Kernel splitting at STORE boundaries
//! - [`linearize`] - Priority-aware topological sort for GPU/NPU backends
//! - [`optimizer`] - Kernel optimization layer (OptOps, Scheduler, heuristics)
//!
//! # Pattern Matching and Rewriting
//!
//! Pattern matching infrastructure has moved to `morok_ir::pattern` and `morok_ir::rewrite`.
//! This crate re-exports these modules for convenience.

pub mod linearize;
pub mod optimizer;
pub mod rangeify;
pub mod symbolic;

#[cfg(feature = "z3")]
pub mod z3;

#[cfg(test)]
pub mod test;

// Re-export pattern matching and rewriting from morok_ir
// This maintains backward compatibility while the infrastructure lives in morok_ir
pub use morok_ir::pattern;
pub use morok_ir::rewrite;

// Re-export main types
pub use linearize::{CFGContext, linearize};
pub use morok_ir::pattern::{PatternMatcher, UPat};
pub use morok_ir::rewrite::graph_rewrite;
pub use rangeify::{rangeify, run_kernel_split_pipeline};

// Re-export optimizer entry points
pub use optimizer::{
    BeamConfig, BeamResult, OptError, OptStrategy, Renderer as OptimizerRenderer, Scheduler, beam_search_cached,
    optimize_kernel, optimize_kernel_with_strategy, prepare_scheduler,
};

// Re-export UOp for macro usage
pub use morok_ir::UOp;

// Re-export the patterns! proc-macro
pub use morok_schedule_macros::patterns;
