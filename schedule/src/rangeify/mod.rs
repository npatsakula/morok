//! RANGEIFY transformation: convert movement ops to BUFFERIZE+INDEX, then split into kernels.
//!
//! ## Module Structure (Consolidated from 19 → 5 files)
//!
//! - `indexing` - Range assignment and IndexingContext
//! - `patterns` - All TypedPatternMatcher constructors
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
pub use kernel::{KernelContext, LocalAddBufferContext};

// Entry points
pub use kernel::run_kernel_split_pipeline;
pub use transforms::{RangeifyResult, rangeify, rangeify_with_map};

// Configuration
pub use kernel::{PcontigConfig, SplitReduceOpConfig};

// Pattern matchers
pub use patterns::{
    apply_rangeify_patterns, buffer_folding, buffer_limit_patterns, buffer_removal, buffer_removal_with_pcontig,
    dead_axis_removal, early_rewrites, movement_op_patterns, pm_comparison_negations, pm_div_to_shr, pm_fdiv_to_mul,
    pm_fma_decomposition, pm_load_collapse, pm_max_decomposition, pm_mod_to_and, pm_mul_to_shl, pm_neg_from_mul,
    pm_sqrt_decomposition, pm_syntactic_sugar, rangeify_codegen_patterns, reduction_simplify_patterns,
    split_reduceop_patterns, to_define_global_patterns,
};

// Transforms
pub use transforms::{
    OpAccessType, SplitRangesContext, SplitStoreContext, bufferize_to_store, find_bufs, flatten_range_impl,
    flatten_ranges, pm_add_buffers_local_patterns, pm_add_buffers_patterns, pm_simplify_ranges, pm_split_ranges,
    pm_split_store, reduce_collapse, simplify_merge_adjacent,
};

// Utilities (re-exported from indexing)
pub use indexing::{apply_movement_op, is_dead_axis, ranges_equal};
pub use patterns::{extract_device_from_graph, is_elementwise};

// Testing exports
#[cfg(test)]
pub(crate) use indexing::merge_consumer_ranges;
