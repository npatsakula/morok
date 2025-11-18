//! UOp (micro-operation) implementation.
//!
//! This module contains the UOp struct and all related functionality for creating
//! and manipulating operations in the IR.
//!
//! # Module Organization
//!
//! - [`core`] - UOp struct and fundamental operations
//! - [`hash_consing`] - Caching infrastructure for deduplication
//! - [`constructors`] - Helper methods for creating UOps
//! - [`helpers`] - Pattern matching and simplification helpers
//! - [`cached_property`] - Reusable pattern for cached graph properties
//! - [`properties`] - Standard cached properties (shape, ranges, etc.)
//! - [`eval`] - Constant evaluation for operations
//! - [`range_eval`] - Range analysis (vmin/vmax) for operations
//! - [`comparison_analysis`] - Unified comparison analysis for optimizations

pub mod cached_property;
pub mod comparison_analysis;
pub mod constructors;
pub mod core;
pub mod eval;
pub mod hash_consing;
pub mod helpers;
pub mod properties;
pub mod range_eval;

// Re-export the main types
pub use core::{IntoUOp, UOp, UOpKey};
