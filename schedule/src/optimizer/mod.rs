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
pub mod config;
pub mod error;
pub mod heuristics;
pub mod kernel_info;
pub mod opts;
pub mod renderer;
pub mod scheduler;
pub mod tc;
pub mod types;

// Re-exports
pub use beam::{BeamResult, beam_search, beam_search_cached, beam_search_with_timeout, clear_cache, replay_opts};
pub use config::{BeamConfig, HeuristicsConfig, OptStrategy, OptimizerConfig, TcOpt as TcOptLevel, TcSelect, TcUsage};
pub use error::OptError;
pub use heuristics::hand_coded_optimizations;
pub use kernel_info::KernelInfo;
pub use opts::apply_opt;
pub use renderer::{Renderer, TcOpt, TensorCore};
pub use scheduler::Scheduler;
#[cfg(test)]
pub use scheduler::clear_kernel_name_counts;
pub use types::{AxisType, Opt, OptArg, OptOps};

use crate::rewrite::graph_rewrite;
use crate::symbolic::patterns::symbolic;
use std::sync::Arc;

/// Apply optimizations to a kernel AST.
///
/// This is the main entry point for optimization in the tensor pipeline.
/// Uses environment variables for configuration (see `OptimizerConfig::from_env`).
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
    optimize_kernel_with_config(ast, renderer, &OptimizerConfig::from_env())
}

/// Apply post-optimization passes to kernel AST.
///
/// These passes run AFTER heuristic/beam optimization and BEFORE codegen:
/// - pm_add_loads: Extract LOAD ops from INDEX
/// - pre_expand: Convert Range(Unroll/Upcast) → UNROLL, expand operations
/// - pm_reduce_devectorize: Unified REDUCE devectorization (K-vec, bool, horizontal)
/// - pm_bool_devectorize: Convert <N x i1> to scalar ops
/// - pm_fma_decomposition: a*b+c → MulAcc for float types
/// - bool_storage_patterns: Convert bool LOAD/STORE to uint8
///
/// Called by both heuristic and beam search paths for consistent behavior.
#[tracing::instrument(skip_all, fields(ast.initial = ast.tree()))]
pub fn apply_post_optimization(ast: Arc<morok_ir::UOp>) -> Arc<morok_ir::UOp> {
    // Add explicit LOAD ops for INDEX sources consumed by arithmetic ops.
    // This separates INDEX (returns indices for STORE scatter) from LOAD (performs gather).
    // Based on Tinygrad's pm_add_loads (devectorizer.py:320-326).
    let pm_loads = crate::rangeify::patterns::pm_add_loads();
    // First pass: wrap existing INDEX ops before expansion.
    // Pattern has guard (!Ptr dtype) and transforms INDEX dtype to Ptr, so safe for reuse.
    let with_loads = graph_rewrite(&pm_loads, ast, &mut ());

    // Post-optimization: Fix UNROLL substitutions in REDUCE ops
    // This handles arithmetic expressions created by shift_to UNROLL
    let expanded = crate::expand::pre_expand(&with_loads);

    // Second pass of pm_add_loads: pre_expand creates new INDEX ops (via CAT expansion)
    // that need LOAD wrapping. These weren't present before pre_expand.
    // Safe for bottom-up: pattern guard (!Ptr) prevents re-matching transformed INDEX.
    let expanded = crate::rewrite::graph_rewrite_bottom_up(&pm_loads, expanded, &mut ());

    // Devectorize pass: group contiguous memory accesses
    // Uses direct vector index analysis (VConst/UNROLL patterns) for termination safety
    let devectorized_mem = crate::devectorize::devectorize(&expanded);

    // Bool devectorization: Convert <N x i1> ALU ops to scalar ops + VECTORIZE.
    // LLVM's bool vectors are broken (no formal ABI, segfaults in codegen).
    // Based on Tinygrad's no_vectorized_alu approach.
    // Must run BEFORE reduce devectorization so comparisons are already scalar.
    let pm_devec = crate::rangeify::patterns::pm_bool_devectorize();
    let devectorized = crate::rewrite::graph_rewrite_bottom_up(&pm_devec, devectorized_mem, &mut ());

    // Unified REDUCE devectorization: Handles all 3 mutually exclusive cases:
    // - K-vectorized: CONTRACT source → N scalar REDUCEs + tree_reduce (SLP optimization)
    // - Bool reduce: matching vcounts, bool dtype → N scalar REDUCEs + VECTORIZE (<N x i1> workaround)
    // - Horizontal: src_vcount > out_vcount → stride-pattern GEPs + ALU chain
    // Based on Tinygrad's pm_reduce (devectorizer.py:283-316).
    let pm_reduce = crate::rangeify::patterns::pm_reduce_devectorize();
    let reduce_devec = crate::rewrite::graph_rewrite_bottom_up(&pm_reduce, devectorized, &mut ());

    // Second pass of bool ALU devectorization: Handle WHERE ops created by horizontal reduce.
    // Min reduction creates WHERE(<N x i1>, ...) which needs devectorization.
    let reduce_devec = crate::rewrite::graph_rewrite_bottom_up(&pm_devec, reduce_devec, &mut ());

    // Second pass of REDUCE devectorization: Handle REDUCEs created by horizontal reduce.
    // When horizontal reduce transforms REDUCE(<4 x bool>) → REDUCE(<2 x bool>), we need
    // to devectorize the resulting REDUCE to avoid LLVM <N x i1> accumulator issues.
    let reduce_devec = crate::rewrite::graph_rewrite_bottom_up(&pm_reduce, reduce_devec, &mut ());

    // FMA decomposition: a*b+c → MulAcc(a,b,c) for float types.
    // Applied late so optimizations can still see Add(Mul) structure.
    // Must run AFTER horizontal reduce (which may create Add chains from GEPs).
    // Based on Tinygrad's decompositions.py:362.
    let pm_fma = crate::rangeify::patterns::pm_fma_decomposition();
    let with_fma = crate::rewrite::graph_rewrite_bottom_up(&pm_fma, reduce_devec, &mut ());

    // Bool storage: Convert bool LOAD/STORE to uint8 to avoid LLVM i1 garbage bits.
    // LLVM's i1 type when stored to memory can have garbage in upper 7 bits.
    // Must run LAST, after all other transformations that might create bool stores.
    // Based on Tinygrad's PTX/NIR bool→uint8 patterns.
    let pm_bool_storage = crate::devectorize::bool_storage_patterns();
    crate::rewrite::graph_rewrite_bottom_up(&pm_bool_storage, with_fma, &mut ())
}

/// Apply optimizations with explicit configuration.
///
/// Use this when you need explicit control over the optimization settings.
///
/// Note: For beam search strategy, this falls back to heuristics because
/// beam search requires a `compile_and_time` function from the runtime.
/// Use `optimize_kernel_beam()` for actual beam search optimization.
pub fn optimize_kernel_with_config(
    ast: Arc<morok_ir::UOp>,
    renderer: &Renderer,
    config: &OptimizerConfig,
) -> Arc<morok_ir::UOp> {
    let optimized = match config.strategy {
        OptStrategy::None => return ast, // No optimization = no pre_expand needed
        OptStrategy::Heuristic => optimize_heuristic(ast, renderer, &config.heuristics),
        OptStrategy::Beam { .. } => {
            // Beam search requires a compile_and_time function.
            // Use optimize_kernel_beam() for actual beam search.
            // Fall back to heuristics for the simple API.
            optimize_heuristic(ast, renderer, &config.heuristics)
        }
    };

    apply_post_optimization(optimized)
}

/// Apply optimizations with explicit strategy selection (legacy API).
///
/// Prefer `optimize_kernel_with_config` for new code.
pub fn optimize_kernel_with_strategy(
    ast: Arc<morok_ir::UOp>,
    renderer: &Renderer,
    strategy: OptStrategy,
) -> Arc<morok_ir::UOp> {
    let config = OptimizerConfig { strategy, ..Default::default() };
    optimize_kernel_with_config(ast, renderer, &config)
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
    let _ = scheduler.convert_loop_to_global(); // GPU: LOOP→GLOBAL
    // Note: Don't apply threading here - let beam search explore THREAD actions naturally.
    // Heuristics apply threading via hand_coded_optimizations() with config.thread_count.
    scheduler
}

/// Apply heuristic-based optimizations.
fn optimize_heuristic(ast: Arc<morok_ir::UOp>, renderer: &Renderer, config: &HeuristicsConfig) -> Arc<morok_ir::UOp> {
    // Step 1: Symbolic simplification
    let simplified = graph_rewrite(&symbolic(), ast, &mut ());

    // Step 2: Create scheduler with backend capabilities
    let mut scheduler = Scheduler::new(simplified, renderer.clone());

    // Step 3: Convert axes for parallelization/vectorization
    let _ = scheduler.convert_loop_to_global(); // GPU: LOOP→GLOBAL
    let _ = scheduler.convert_outer_to_loop(); // CPU: OUTER→LOOP (enables UPCAST)

    // Step 4: Apply hand-coded heuristics with config
    heuristics::hand_coded_optimizations(&mut scheduler, config);

    // Step 5: Extract optimized AST
    scheduler.get_optimized_ast(None)
}
