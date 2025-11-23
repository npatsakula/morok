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
//! ## Current Status (Phase 5 In Progress)
//!
//! Phases 1-4 implement the full rangeify transformation with:
//! - Range assignment algorithm (`indexing::run_rangeify`)
//! - Early cleanup patterns (DETACH, CONTIGUOUS_BACKWARD removal)
//! - Generic BUFFERIZE + INDEX insertion via pattern matching
//! - Movement op removal after transformation
//! - Buffer folding, dead axis removal, and cost-based buffer removal
//! - Comprehensive symbolic simplification of index expressions
//!
//! Phase 5 (Kernel Splitting) adds:
//! - BUFFERIZE → STORE conversion (`bufferize_to_store`)
//! - Pattern matchers for kernel transformation (`split_patterns`)
//! - Kernel splitting orchestration (`split_kernel`, `pipeline`)
//! - AxisType::Outer for marking kernel boundaries
//!
//! The pipeline integration (`run_kernel_split_pipeline`) orchestrates the complete
//! transformation from BUFFERIZE operations to executable KERNEL operations.

// Core rangeify transformation (Phases 1-4)
pub mod context;
pub mod helpers;
pub mod indexing;
pub mod patterns;
pub mod transform;

// Kernel splitting components (Phase 5)
pub mod buffer_cost;
pub mod buffer_limits;
pub mod bufferize_to_store;
pub mod codegen_patterns;
pub mod cycle_detection;
pub mod flatten_range;
pub mod kernel_context;
pub mod movement_patterns;
pub mod pipeline;
pub mod reduce_simplify;
pub mod split_kernel;
pub mod split_patterns;
pub mod split_reduceop;

// Public API exports
pub use buffer_cost::{
    PcontigConfig, apply_partial_contiguous, calculate_buffer_size, calculate_out_in_ratio, collect_accessed_buffers,
    collect_indexes, collect_local_indexes, collect_reduces, extract_exclude_ranges, has_buffer_in_reduce,
    partition_ranges,
};
pub use buffer_limits::{buffer_limit_patterns, extract_device_from_graph, is_elementwise};
pub use context::RangeifyContext;
pub use indexing::{IndexingContext, run_rangeify};
pub use kernel_context::KernelContext;
pub use pipeline::run_kernel_split_pipeline;
pub use transform::rangeify;
