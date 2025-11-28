//! RANGEIFY transformation: convert movement ops to BUFFERIZE+INDEX, then split into kernels.

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
