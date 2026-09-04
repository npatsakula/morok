//! Schedule module for Svod compiler.
//!
//! This module implements optimization passes for the IR,
//! including symbolic simplification and graph transformations.
//!
//! # Module Organization
//!
//! - [`symbolic`] - Symbolic simplification patterns
//! - [`mod@rangeify`] - RANGEIFY transformation (movement ops → kernels)
//!   - Phases 1-4: Movement ops to STAGE with symbolic simplification
//!   - Phase 5: Kernel splitting at STORE boundaries
//! - [`mod@linearize`] - Priority-aware topological sort for GPU/NPU backends
//! - [`optimizer`] - Kernel optimization layer (OptOps, Scheduler, heuristics)
//! - [`expand`] - Pre-expansion pass for UNROLL/UPCAST range handling
//!
//! # Pattern Matching and Rewriting
//!
//! Pattern matching infrastructure has moved to `svod_ir::pattern` and `svod_ir::rewrite`.
//! This crate re-exports these modules for convenience.

pub mod devectorize;
pub mod expand;
pub mod gpudims;
pub mod late;
pub mod linearize;
pub mod multi;
pub mod optimizer;
pub mod passes;
pub mod rangeify;
pub mod spec;
pub mod symbolic;
#[cfg(feature = "testing")]
pub mod testing;

#[cfg(feature = "z3")]
pub mod z3;

#[cfg(test)]
pub mod test;

// Re-export pattern matching and rewriting from svod_ir
// This maintains backward compatibility while the infrastructure lives in svod_ir
pub use svod_ir::pattern;
pub use svod_ir::rewrite;

// Re-export main types
pub use linearize::{CFGContext, add_control_flow, linearize, linearize_with_cfg};
pub use rangeify::{KernelGraphError, RangeifyResult, rangeify, rangeify_with_map, try_get_kernel_graph};
pub use svod_ir::pattern::{Matcher, RewriteResult, TypedPatternMatcher};
pub use svod_ir::rewrite::graph_rewrite;

// Re-export expand pass
pub use expand::{build_range_map, expander2, pm_group_for_reduce, pre_expand};

// Re-export devectorize pass
pub use devectorize::devectorize;

// Re-export gpudims pass
pub use gpudims::{GpuDimsContext, pm_add_gpudims, pm_lower_device_ranges};

// Re-export optimizer entry points
pub use optimizer::{
    BeamConfig, BeamResult, CandidateMetrics, HeuristicsConfig, KernelNaming, OptError, OptStrategy, OptimizerConfig,
    Renderer as OptimizerRenderer, Scheduler, TcOptLevel, TcSelect, TcUsage, apply_post_optimization,
    apply_post_optimization_with_config, apply_post_optimization_with_renderer, beam_search_cached_with_behavior,
    compute_ops_estimate, finalize_kernel_name, hand_coded_optimizations, hash_post_codegen_ir, optimize_kernel,
    optimize_kernel_with_config, optimize_kernel_with_config_and_final_rewrite, optimize_kernel_with_naming,
    optimize_kernel_with_strategy, prepare_scheduler, thread_budget, unique_kernel_name,
};

// Re-export UOp for macro usage
pub use svod_ir::UOp;

// Re-export the patterns! proc-macros
pub use svod_macros::{cached_patterns, patterns};

/// Compute inverse permutation (argsort).
pub(crate) fn argsort(perm: &[usize]) -> Vec<usize> {
    let mut inv = vec![0; perm.len()];
    for (i, &p) in perm.iter().enumerate() {
        inv[p] = i;
    }
    inv
}
