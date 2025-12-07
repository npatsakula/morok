//! Kernel optimization layer for morok-schedule.
//!
//! This module implements hardware-aware kernel optimization based on Tinygrad's approach.
//! It provides a `Scheduler` that applies optimization primitives (OptOps) to transform
//! kernel execution for better performance on specific backends.
//!
//! # Architecture
//!
//! The optimization process follows this flow:
//!
//! 1. **Initialization**: Create `Scheduler` from UOp AST + `Renderer` (backend capabilities)
//! 2. **Initial Transform**: Convert eligible LOOP axes to GLOBAL (parallelization)
//! 3. **Optimization**: Apply `Opt` operations via `apply_opt()`
//!    - UPCAST: Vectorization (SIMD)
//!    - LOCAL: GPU workgroup dimensions (shared memory)
//!    - UNROLL: Loop unrolling for reductions
//!    - GROUP: Two-stage reductions with synchronization
//!    - TC: Tensor core acceleration
//!    - PADTO, SWAP, THREAD, NOLOCALS: Layout and configuration
//! 4. **Finalization**: Extract optimized AST with `get_optimized_ast()`
//!
//! # Optimization Strategies
//!
//! - **Hand-coded heuristics** (`heuristics` module): Fast, reasonable performance
//! - **Beam search** (`beam` module, optional): Slow, ML-quality performance
//!
//! # Example
//!
//! ```ignore
//! use morok_schedule::optimizer::{Scheduler, Renderer, Opt, OptOps};
//!
//! // Create scheduler with CUDA backend
//! let renderer = Renderer::cuda();
//! let mut scheduler = Scheduler::new(kernel_ast, renderer);
//!
//! // Apply optimizations
//! scheduler.convert_loop_to_global();
//! scheduler.apply_opt(Opt::upcast(0, 4), true)?; // Vectorize axis 0 by 4
//! scheduler.apply_opt(Opt::local(1, 16), true)?; // Local memory for axis 1
//!
//! // Get optimized kernel
//! let optimized_ast = scheduler.get_optimized_ast(None);
//! ```

pub mod error;
pub mod heuristics;
pub mod kernel_info;
pub mod opts;
pub mod renderer;
pub mod scheduler;
pub mod strategy;
pub mod tc;
pub mod types;

// Re-exports
pub use heuristics::hand_coded_optimizations;
pub use kernel_info::KernelInfo;
pub use opts::apply_opt;
pub use renderer::{Renderer, TcOpt, TensorCore};
pub use scheduler::Scheduler;
#[cfg(test)]
pub use scheduler::clear_kernel_name_counts;
pub use strategy::OptStrategy;
pub use types::{AxisType, Opt, OptArg, OptOps};

use crate::rewrite::graph_rewrite;
use crate::symbolic::patterns::symbolic;
use std::sync::Arc;

/// Apply optimizations to a kernel AST.
///
/// This is the main entry point for optimization in the tensor pipeline.
///
/// # Pipeline
///
/// 1. **Symbolic simplification** - Constant folding, identities, DCE
/// 2. **Loop→Global conversion** - Enable GPU parallelization
/// 3. **Hand-coded heuristics** - Vectorization, unrolling, tiling
///
/// # Arguments
///
/// * `ast` - The kernel AST (inner AST from KERNEL op)
/// * `renderer` - Backend capabilities descriptor
///
/// # Returns
///
/// Optimized AST with transformations applied.
///
/// # Environment Variables
///
/// * `MOROK_NOOPT=1` - Disable all optimizations (for debugging)
/// * `MOROK_BEAM=N` - Use beam search with width N (future)
pub fn optimize_kernel(ast: Arc<morok_ir::UOp>, renderer: &Renderer) -> Arc<morok_ir::UOp> {
    optimize_kernel_with_strategy(ast, renderer, OptStrategy::from_env())
}

/// Apply optimizations with explicit strategy selection.
///
/// Use this when you need explicit control over the optimization strategy,
/// such as in tests or when comparing different approaches.
pub fn optimize_kernel_with_strategy(
    ast: Arc<morok_ir::UOp>,
    renderer: &Renderer,
    strategy: OptStrategy,
) -> Arc<morok_ir::UOp> {
    match strategy {
        OptStrategy::None => ast,
        OptStrategy::Heuristic => optimize_heuristic(ast, renderer),
        OptStrategy::Beam { .. } => {
            // Beam search not yet implemented - fall back to heuristics
            // See BEAM_SEARCH_PLAN.md for implementation roadmap
            optimize_heuristic(ast, renderer)
        }
    }
}

/// Apply heuristic-based optimizations.
fn optimize_heuristic(ast: Arc<morok_ir::UOp>, renderer: &Renderer) -> Arc<morok_ir::UOp> {
    // Step 1: Symbolic simplification
    let simplified = graph_rewrite(&symbolic(), ast, &mut ());

    // Step 2: Create scheduler with backend capabilities
    let mut scheduler = Scheduler::new(simplified, renderer.clone());

    // Step 3: Convert eligible LOOP axes to GLOBAL (for GPU parallelization)
    let _ = scheduler.convert_loop_to_global();

    // Step 4: Apply hand-coded heuristics
    heuristics::hand_coded_optimizations(&mut scheduler);

    // Step 5: Extract optimized AST
    scheduler.get_optimized_ast(None)
}
