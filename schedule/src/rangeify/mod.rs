//! RANGEIFY transformation: convert movement ops to BUFFERIZE+INDEX, then split into kernels.
//!
//! ## Module Structure (Consolidated from 19 → 5 files)
//!
//! - `indexing` - Range assignment and IndexingContext
//! - `patterns` - All PatternMatcher constructors
//! - `transforms` - rangeify() entry point and transformation functions
//! - `kernel` - KernelContext, kernel splitting, buffer cost analysis

// Consolidated modules
pub mod context;
pub mod indexing;
pub mod kernel;
pub mod patterns;
pub mod transforms;

// ============================================================================
// PUBLIC API
// ============================================================================

// Context types
pub use context::RangeifyContext;
pub use indexing::{IndexingContext, run_rangeify};
pub use kernel::{KernelContext, KernelDependency};

// Entry points
pub use kernel::run_kernel_split_pipeline;
pub use transforms::{RangeifyResult, rangeify, rangeify_with_map};

// Configuration
pub use kernel::{PcontigConfig, SplitReduceOpConfig};

// Pattern matchers
pub use patterns::{
    apply_rangeify_patterns, buffer_folding, buffer_limit_patterns, buffer_removal, buffer_removal_with_pcontig,
    dead_axis_removal, early_rewrites, movement_op_patterns, rangeify_codegen_patterns, reduction_simplify_patterns,
    to_define_global_patterns,
};

// Buffer cost analysis
pub use kernel::{
    apply_partial_contiguous, calculate_buffer_size, calculate_out_in_ratio, collect_accessed_buffers, collect_indexes,
    collect_local_indexes, collect_reduces, extract_exclude_ranges, has_buffer_in_reduce, partition_ranges,
};

// Transforms
pub use transforms::{
    OpAccessType, bufferize_to_store, find_bufs, flatten_range_impl, flatten_ranges, reduce_collapse, reduce_unparented,
};

// Utilities (re-exported from indexing)
pub use indexing::{apply_movement_op, is_dead_axis, ranges_equal};
pub use patterns::{extract_device_from_graph, is_elementwise};

// Testing exports
#[cfg(test)]
pub(crate) use indexing::merge_consumer_ranges;
