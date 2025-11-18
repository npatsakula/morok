//! Schedule module for Morok compiler.
//!
//! This module implements pattern matching and graph rewriting for the IR,
//! including symbolic simplification and optimization passes.
//!
//! # Module Organization
//!
//! - [`pattern`] - UPat pattern matching and PatternMatcher
//! - [`rewrite`] - Graph rewrite engine with fixed-point iteration
//! - [`symbolic`] - Symbolic simplification patterns
//! - [`rangeify`] - RANGEIFY transformation (movement ops → kernels)
//!   - Phases 1-4: Movement ops to BUFFERIZE with symbolic simplification
//!   - Phase 5: Kernel splitting at STORE boundaries

#[macro_use]
pub mod pattern;
pub mod rangeify;
pub mod rewrite;
pub mod symbolic;

#[cfg(feature = "z3")]
pub mod z3;

#[cfg(test)]
pub mod test;

// Re-export main types
pub use pattern::{PatternMatcher, UPat};
pub use rangeify::{rangeify, run_kernel_split_pipeline};
pub use rewrite::graph_rewrite;

// Re-export UOp for macro usage
pub use morok_ir::UOp;
