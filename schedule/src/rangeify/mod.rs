//! RANGEIFY transformation: convert movement ops to BUFFERIZE+INDEX.
//!
//! This module implements Tinygrad's RANGEIFY algorithm, which transforms
//! high-level tensor operations into executable kernels by:
//! 1. Converting movement operations into explicit index transformations
//! 2. Inserting buffer materialization points
//! 3. Applying aggressive fusion to minimize memory transfers
//! 4. Splitting the computation graph into individual kernels
//!
//! ## Algorithm Overview
//!
//! RANGEIFY operates in multiple phases:
//! - **Range Assignment** ✅ (Phase 1): Determine input/output ranges for each UOp
//! - **Early Rewrites** ✅ (Phase 2): Cleanup DETACH and CONTIGUOUS_BACKWARD
//! - **Core Transform** ✅ (Phase 2): Movement ops → BUFFERIZE with INDEX via pattern matching
//! - **Buffer Management** ✅ (Phase 3): Cost-based buffer folding and removal
//! - **Symbolic Simplification** ✅ (Phase 4): Optimize index expressions
//! - **Kernel Splitting** 🚧 (Phase 5): Split graph at STORE boundaries
//!
//! ## Current Status (Phase 4 Complete)
//!
//! Phases 1-4 implement the full rangeify transformation with:
//! - Range assignment algorithm (`indexing::run_rangeify`)
//! - Early cleanup patterns (DETACH, CONTIGUOUS_BACKWARD removal)
//! - Generic BUFFERIZE + INDEX insertion via pattern matching
//! - Movement op removal after transformation
//! - Buffer folding, dead axis removal, and cost-based buffer removal
//! - Comprehensive symbolic simplification of index expressions
//!
//! Future phases will add kernel splitting at STORE boundaries.

pub mod context;
pub mod helpers;
pub mod indexing;
pub mod patterns;
pub mod transform;

pub use context::RangeifyContext;
pub use indexing::{IndexingContext, run_rangeify};
pub use transform::rangeify;
