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
//! - [`expand`] - Pre-expansion pass for UNROLL/UPCAST range handling
//!
//! # Pattern Matching and Rewriting
//!
//! Pattern matching infrastructure has moved to `morok_ir::pattern` and `morok_ir::rewrite`.
//! This crate re-exports these modules for convenience.

pub mod devectorize;
pub mod expand;
pub mod gpudims;
pub mod linearize;
pub mod optimizer;
pub mod passes;
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
pub use linearize::{CFGContext, linearize, linearize_with_cfg};
pub use morok_ir::pattern::{Matcher, RewriteResult, TypedPatternMatcher};
pub use morok_ir::rewrite::graph_rewrite_top_down;
pub use rangeify::{RangeifyResult, rangeify, rangeify_with_map, run_kernel_split_pipeline};

// Re-export expand pass
pub use expand::{expander, pm_group_for_reduce, pm_pre_expander, pre_expand};

// Re-export devectorize pass
pub use devectorize::devectorize;

// Re-export gpudims pass
pub use gpudims::pm_add_gpudims;

// Re-export backend-agnostic passes
pub use passes::pm_linearize_multi_index;

// Re-export optimizer entry points
pub use optimizer::{
    BeamConfig, BeamResult, HeuristicsConfig, OptError, OptStrategy, OptimizerConfig, Renderer as OptimizerRenderer,
    Scheduler, TcOptLevel, TcSelect, TcUsage, apply_post_optimization, apply_post_optimization_with_renderer,
    beam_search_cached, hand_coded_optimizations, optimize_kernel, optimize_kernel_with_config,
    optimize_kernel_with_strategy, prepare_scheduler,
};

// Re-export UOp for macro usage
pub use morok_ir::UOp;

// Re-export the patterns! proc-macro
pub use morok_macros::patterns;
