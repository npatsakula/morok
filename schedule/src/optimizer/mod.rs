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

use crate::devectorize::{bool_storage_patterns, pm_flatten_range, pm_reduce, pm_vectorize_normalize};
use crate::passes::pm_linearize_multi_index;
use crate::rangeify::patterns::{pm_add_loads, pm_bool_devectorize, pm_reduce_devectorize};
use crate::rewrite::{graph_rewrite_bottom_up, graph_rewrite_top_down};
use crate::symbolic::patterns::{gep_pushing_patterns, symbolic};
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
/// - no_vectorized_alu (optional): Convert vector ALU to scalar + VECTORIZE
/// - pm_reduce_devectorize: Unified REDUCE devectorization (K-vec, bool, horizontal)
/// - pm_bool_devectorize: Convert <N x i1> to scalar ops
/// - bool_storage_patterns: Convert bool LOAD/STORE to uint8
///
/// NOTE: We do NOT apply FMA decomposition (a*b+c → MulAcc). Following Tinygrad's
/// approach, we let LLVM's optimizer fuse MUL+ADD into FMA when beneficial.
///
/// # Arguments
///
/// * `ast` - The kernel AST to optimize
/// * `devectorize_alu` - If true, convert vector ALU ops to scalar + VECTORIZE.
///   Use for backends without native vector support. Default: false (preserve vectors).
///
/// Called by both heuristic and beam search paths for consistent behavior.
#[tracing::instrument(skip_all, fields(ast.initial = %ast.tree()))]
pub fn apply_post_optimization(ast: Arc<morok_ir::UOp>, devectorize_alu: bool) -> Arc<morok_ir::UOp> {
    // Multi-index linearization: INDEX(buf, [i,j,k]) → INDEX(buf, [linear])
    // Moves row-major linearization from codegen to schedule, eliminating
    // duplicated logic in LLVM and Cranelift backends.
    // Must run BEFORE pm_add_loads (which transforms INDEX dtype to Ptr).
    // Uses bottom-up traversal to ensure children are processed before parents.
    let linearized = graph_rewrite_bottom_up(&pm_linearize_multi_index(), ast, &mut ());
    tracing::debug!(ast.optimized = linearized.tree(), "after pm_linearize_multi_index");

    // UNROLL expansion: Expand UNROLL ops to vectorized operations (Tinygrad expander.py)
    // CRITICAL: Must run BEFORE pm_reduce so that REDUCE sees its actual vectorized dtype.
    // In Tinygrad, expander runs first, then pm_reduce sees the expanded REDUCE with vec2 dtype.
    // This allows reduce_to_acc to create accumulators with the correct vector dtype.
    let expanded = crate::expand::pre_expand(&linearized);
    tracing::debug!(ast.optimized = expanded.tree(), "after pre_expand");

    // pm_reduce: Convert REDUCE → DEFINE_REG + accumulator pattern (Tinygrad devectorizer.py:310-316)
    // Now that UNROLL is expanded, REDUCE has its final dtype (e.g., vec2 if upcasted).
    // This creates accumulators with matching vector dtype.
    // IMPORTANT: Thread axes are excluded from reduce_to_acc's input_ranges via is_parallel().
    //
    // Tinygrad alignment: Combine pm_reduce + gep_pushing in single pass
    // (Tinygrad: graph_rewrite(sink, pm_reduce+gep_pushing, ctx=ReduceContext()))
    // This ensures GEP simplification happens atomically with REDUCE transformation,
    // preventing dtype mismatches in horizontal reduction.
    let pm_reduce_combined = pm_reduce() + gep_pushing_patterns();
    let reduced = graph_rewrite_bottom_up(&pm_reduce_combined, expanded, &mut ());
    tracing::debug!(ast.optimized = reduced.tree(), "after pm_reduce");

    // Add explicit LOAD ops for INDEX sources consumed by arithmetic ops.
    // This separates INDEX (returns indices for STORE scatter) from LOAD (performs gather).
    // Based on Tinygrad's pm_add_loads (devectorizer.py:320-326).
    let with_loads = graph_rewrite_bottom_up(&pm_add_loads(), reduced, &mut ());

    // ALU Devectorization + VECTORIZE Normalization
    //
    // Two modes based on devectorize_alu flag:
    //
    // MODE 1 (devectorize_alu=true): Follow Tinygrad's DEVECTORIZE=1 pipeline
    // 1. no_vectorized_alu: Convert ALL vector ALU to VECTORIZE(scalar)
    // 2. gep_pushing: Simplify GEPs created by no_vectorized_alu
    // 3. pm_vectorize_normalize: Handle remaining multi-index GEPs
    // 4. gep_pushing: Cleanup
    //
    // MODE 2 (devectorize_alu=false, DEFAULT): Preserve vector operations
    // Skip no_vectorized_alu, only normalize VECTORIZE/GEP patterns.
    // Better for backends with sophisticated optimizers (LLVM SLP vectorizer).

    let processed = if devectorize_alu {
        // Step 1: Devectorize ALL vector ALU ops to VECTORIZE(scalar)
        let no_vec_alu = crate::devectorize::no_vectorized_alu();
        let devec = graph_rewrite_bottom_up(&no_vec_alu, with_loads, &mut ());

        // Step 2: Simplify GEPs created by no_vectorized_alu
        graph_rewrite_bottom_up(&gep_pushing_patterns(), devec, &mut ())
    } else {
        with_loads
    };

    // Step 3: Normalize remaining VECTORIZE/GEP patterns (multi-index GEP, single-source VECTORIZE)
    let normalized = graph_rewrite_bottom_up(&pm_vectorize_normalize(), processed, &mut ());
    tracing::debug!(ast.optimized = normalized.tree(), "after pm_vectorize_normalize");

    // Step 4: Cleanup GEPs from pm_vectorize_normalize
    let cleaned = graph_rewrite_bottom_up(&gep_pushing_patterns(), normalized, &mut ());
    tracing::debug!(ast.optimized = cleaned.tree(), "after pm_pushing_patterns");

    // Second pass of pm_add_loads: expansion may create new INDEX ops that need LOAD wrapping.
    // NOTE: Uses top-down graph_rewrite to match Tinygrad's approach.
    let with_loads2 = graph_rewrite_top_down(&pm_add_loads(), cleaned, &mut ());
    tracing::debug!(ast.optimized = with_loads2.tree(), "after pm_add_loads");

    // Devectorize pass: group contiguous memory accesses
    // Uses direct vector index analysis (VConst/UNROLL patterns) for termination safety
    let devectorized_mem = crate::devectorize::devectorize(&with_loads2);
    tracing::debug!(ast.optimized = devectorized_mem.tree(), "after devectorize");

    // Flatten range: Filter non-RANGE ops from END/REDUCE/STORE ranges.
    // The symbolic pass (included in devectorize) converts trivial RANGE(end=1) to CONST(0).
    // When these RANGEs are referenced in END.ranges, the rewrite substitutes CONST,
    // causing END to have `ranges: [CONST, CONST]` instead of `ranges: [RANGE, RANGE]`.
    // This pattern filters END/REDUCE/STORE ranges to keep only actual RANGE ops.
    // Based on Tinygrad's pm_flatten_range (simplify.py:7-16).
    let flattened = graph_rewrite_bottom_up(&pm_flatten_range(), devectorized_mem, &mut ());
    tracing::debug!(ast.optimized = flattened.tree(), "after pm_flatten_range");

    // Bool devectorization: Convert <N x i1> ALU ops to scalar ops + VECTORIZE.
    // LLVM's bool vectors are broken (no formal ABI, segfaults in codegen).
    // Based on Tinygrad's no_vectorized_alu approach.
    // Must run BEFORE reduce devectorization so comparisons are already scalar.
    let devectorized = graph_rewrite_bottom_up(&pm_bool_devectorize(), flattened, &mut ());
    tracing::debug!(ast.optimized = devectorized.tree(), "after pm_bool_devectorize");

    // Unified REDUCE devectorization: Handles all 3 mutually exclusive cases:
    // - K-vectorized: CONTRACT source → N scalar REDUCEs + tree_reduce (SLP optimization)
    // - Bool reduce: matching vcounts, bool dtype → N scalar REDUCEs + VECTORIZE (<N x i1> workaround)
    // - Horizontal: src_vcount > out_vcount → stride-pattern GEPs + ALU chain
    // Based on Tinygrad's pm_reduce (devectorizer.py:283-316).
    let reduce_devec = graph_rewrite_bottom_up(&pm_reduce_devectorize(), devectorized, &mut ());
    tracing::debug!(ast.optimized = reduce_devec.tree(), "after pm_reduce_devectorize");

    // NOTE: We intentionally do NOT create MulAcc (FMA) ops here.
    // Tinygrad only creates MULACC if the backend explicitly supports it (via code_for_op).
    // For LLVM/CPU backends, MULACC is not in code_for_op, so Tinygrad keeps a*b+c as
    // separate MUL+ADD operations and lets LLVM's optimizer fuse them into FMA when beneficial.
    // This avoids devectorization complexity and lets LLVM make optimal FMA decisions.
    // See: tinygrad/uop/decompositions.py:362 - "if Ops.MULACC in ops: ..."

    // Bool storage: Convert bool LOAD/STORE to uint8 to avoid LLVM i1 garbage bits.
    // LLVM's i1 type when stored to memory can have garbage in upper 7 bits.
    // Must run LAST, after all other transformations that might create bool stores.
    // Based on Tinygrad's PTX/NIR bool→uint8 patterns.
    let bs = graph_rewrite_bottom_up(&bool_storage_patterns(), reduce_devec, &mut ());
    tracing::debug!(ast.optimized = bs.tree(), "after bool_storage_pattern");
    bs
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
        OptStrategy::None => ast, // No heuristic optimization, but post-optimization still needed
        OptStrategy::Heuristic => optimize_heuristic(ast, renderer, &config.heuristics),
        OptStrategy::Beam { .. } => {
            // Beam search requires a compile_and_time function.
            // Use optimize_kernel_beam() for actual beam search.
            // Fall back to heuristics for the simple API.
            optimize_heuristic(ast, renderer, &config.heuristics)
        }
    };

    // apply_post_optimization contains correctness transforms (pm_add_loads wraps INDEX
    // with LOAD for arithmetic ops) and must run even when optimizations are disabled.
    apply_post_optimization(optimized, config.devectorize_alu)
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
    use crate::rewrite::graph_rewrite_top_down;
    use crate::symbolic::patterns::symbolic;

    // Step 1: Symbolic simplification
    let simplified = graph_rewrite_top_down(&symbolic(), ast, &mut ());

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
    let simplified = graph_rewrite_top_down(&symbolic(), ast, &mut ());
    let mut scheduler = Scheduler::new(simplified, renderer.clone());
    let _ = scheduler.convert_loop_to_global(); // GPU: LOOP→GLOBAL
    // Note: Don't apply threading here - let beam search explore THREAD actions naturally.
    // Heuristics apply threading via hand_coded_optimizations() with config.thread_count.
    scheduler
}

/// Apply heuristic-based optimizations.
fn optimize_heuristic(ast: Arc<morok_ir::UOp>, renderer: &Renderer, config: &HeuristicsConfig) -> Arc<morok_ir::UOp> {
    // Step 1: Symbolic simplification
    let simplified = graph_rewrite_top_down(&symbolic(), ast, &mut ());

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
