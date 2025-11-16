//! Graph rewrite engine with fixed-point iteration.
//!
//! This module implements the core graph rewriting algorithm that applies
//! pattern-based transformations to UOp graphs.

pub mod engine;

pub use engine::graph_rewrite;
