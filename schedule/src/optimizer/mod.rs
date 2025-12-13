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

pub mod beam;
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
pub use beam::{
    BeamConfig, BeamResult, beam_search, beam_search_cached, beam_search_with_timeout, clear_cache, replay_opts,
};
pub use error::OptError;
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
///
/// Note: For beam search strategy, this falls back to heuristics because
/// beam search requires a `compile_and_time` function from the runtime.
/// Use `optimize_kernel_beam()` for actual beam search optimization.
pub fn optimize_kernel_with_strategy(
    ast: Arc<morok_ir::UOp>,
    renderer: &Renderer,
    strategy: OptStrategy,
) -> Arc<morok_ir::UOp> {
    match strategy {
        OptStrategy::None => ast,
        OptStrategy::Heuristic => optimize_heuristic(ast, renderer),
        OptStrategy::Beam { .. } => {
            // Beam search requires a compile_and_time function.
            // Use optimize_kernel_beam() for actual beam search.
            // Fall back to heuristics for the simple API.
            optimize_heuristic(ast, renderer)
        }
    }
}

/// Apply beam search optimization with custom timing function.
///
/// This is the primary entry point for beam search auto-tuning. It requires
/// a `compile_and_time` function that compiles a scheduler state and returns
/// its execution timing.
///
/// # Arguments
///
/// * `ast` - The kernel AST to optimize
/// * `renderer` - Backend capabilities descriptor
/// * `config` - Beam search configuration
/// * `compile_and_time` - Function to compile and time a scheduler
///
/// # Returns
///
/// Result containing `BeamResult` with optimized scheduler and metrics.
///
/// # Example
///
/// ```ignore
/// use morok_schedule::optimizer::{optimize_kernel_beam, BeamConfig, Renderer};
/// use morok_runtime::{BenchmarkConfig, benchmark_kernel};
///
/// let config = BeamConfig::from_env();
/// let renderer = Renderer::cpu();
///
/// let compile_and_time = |scheduler: &Scheduler| -> Option<Duration> {
///     let ast = scheduler.get_optimized_ast(None);
///     let kernel = compile_kernel(&ast)?;
///     let result = benchmark_kernel(&kernel, &buffers, &vars, &bench_config).ok()?;
///     Some(result.min)
/// };
///
/// let result = optimize_kernel_beam(ast, &renderer, &config, compile_and_time)?;
/// let optimized_ast = result.scheduler.get_optimized_ast(None);
/// ```
pub fn optimize_kernel_beam<F>(
    ast: Arc<morok_ir::UOp>,
    renderer: &Renderer,
    config: &BeamConfig,
    compile_and_time: F,
) -> Result<BeamResult, error::OptError>
where
    F: Fn(&Scheduler) -> Option<std::time::Duration> + Sync,
{
    use crate::rewrite::graph_rewrite;
    use crate::symbolic::patterns::symbolic;

    // Step 1: Symbolic simplification
    let simplified = graph_rewrite(&symbolic(), ast, &mut ());

    // Step 2: Create scheduler
    let mut scheduler = Scheduler::new(simplified, renderer.clone());

    // Step 3: Convert loops to global (for GPU parallelization)
    let _ = scheduler.convert_loop_to_global();

    // Step 4: Run beam search (with caching)
    beam::beam_search_cached(scheduler, config, compile_and_time)
}

/// Create a scheduler ready for optimization without applying any opts.
///
/// This is useful when you want to manually control the optimization process
/// or use beam search with custom logic.
///
/// # Arguments
///
/// * `ast` - The kernel AST
/// * `renderer` - Backend capabilities descriptor
///
/// # Returns
///
/// A `Scheduler` with loops converted to globals (if applicable).
pub fn prepare_scheduler(ast: Arc<morok_ir::UOp>, renderer: &Renderer) -> Scheduler {
    let simplified = graph_rewrite(&symbolic(), ast, &mut ());
    let mut scheduler = Scheduler::new(simplified, renderer.clone());
    let _ = scheduler.convert_loop_to_global();
    scheduler
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
