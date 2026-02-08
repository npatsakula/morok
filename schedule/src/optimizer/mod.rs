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

use crate::devectorize::{bool_storage_patterns, pm_reduce, pm_render, pm_wmma_accumulate};
use crate::gpudims::pm_add_gpudims;
use crate::passes::pm_linearize_multi_index;
use crate::rangeify::patterns::{
    pm_add_loads, pm_bool_devectorize, pm_comparison_negations, pm_div_to_shr, pm_fdiv_to_mul, pm_fma_decomposition,
    pm_mod_to_and, pm_mul_to_shl, pm_neg_from_mul, pm_reduce_devectorize,
};
use crate::rewrite::graph_rewrite;
use crate::symbolic::patterns::{gep_pushing_patterns, symbolic, symbolic_simple};
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
/// - pm_add_gpudims (GPU only): Convert GLOBAL/LOCAL RANGE to SPECIAL thread indices
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
/// For GPU pipelines, use `apply_post_optimization_with_renderer` to enable GPU dimension injection.
#[tracing::instrument(skip_all, fields(ast.initial = %ast.tree()))]
pub fn apply_post_optimization(ast: Arc<morok_ir::UOp>, devectorize_alu: bool) -> Arc<morok_ir::UOp> {
    apply_post_optimization_with_renderer(ast, devectorize_alu, None)
}

/// Apply post-optimization passes with renderer context.
///
/// Same as `apply_post_optimization` but accepts an optional renderer for GPU-specific passes.
/// When a renderer with GPU capabilities (has_local) is provided, `pm_add_gpudims` is applied
/// to convert GLOBAL/LOCAL RANGE operations to SPECIAL thread indices.
///
/// # Arguments
///
/// * `ast` - The kernel AST to optimize
/// * `devectorize_alu` - If true, convert vector ALU ops to scalar + VECTORIZE
/// * `renderer` - Optional renderer for GPU dimension injection
#[tracing::instrument(skip_all, fields(ast.initial = %ast.tree()))]
pub fn apply_post_optimization_with_renderer(
    ast: Arc<morok_ir::UOp>,
    devectorize_alu: bool,
    renderer: Option<&Renderer>,
) -> Arc<morok_ir::UOp> {
    // Multi-index linearization: INDEX(buf, [i,j,k]) → INDEX(buf, [linear])
    // Moves row-major linearization from codegen to schedule, eliminating
    // duplicated logic in LLVM backends.
    // Must run BEFORE pm_add_loads (which transforms INDEX dtype to Ptr).
    // Uses bottom-up traversal to ensure children are processed before parents.
    let linearized = graph_rewrite(&pm_linearize_multi_index(), ast, &mut ());
    tracing::debug!(ast.optimized = linearized.tree(), "after pm_linearize_multi_index");

    // =========================================================================
    // Stage 8: Post-opt symbolic + WHERE movement (Tinygrad: sym + pm_move_where_on_load)
    // This MUST run BEFORE expander to optimize conditionals before expansion.
    // =========================================================================
    let post_opt_symbolic = symbolic();
    let with_symbolic = graph_rewrite(&post_opt_symbolic, linearized, &mut ());
    tracing::debug!(ast.optimized = with_symbolic.tree(), "Stage 8: after post-opt symbolic");

    // =========================================================================
    // Stage 9: Expander (Tinygrad: sym + pm_pre_expander + pm_group_for_reduce + expander)
    // =========================================================================
    // UNROLL expansion: Expand UNROLL ops to vectorized operations (Tinygrad expander.py)
    // CRITICAL: Must run BEFORE pm_reduce so that REDUCE sees its actual vectorized dtype.
    // In Tinygrad, expander runs first, then pm_reduce sees the expanded REDUCE with vec2 dtype.
    // This allows reduce_to_acc to create accumulators with the correct vector dtype.
    let expanded = crate::expand::pre_expand(&with_symbolic);
    tracing::debug!(ast.optimized = expanded.tree(), "Stage 9: after pre_expand");

    // pm_reduce: Convert REDUCE → DEFINE_REG + accumulator pattern (Tinygrad devectorizer.py:310-316)
    // Now that UNROLL is expanded, REDUCE has its final dtype (e.g., vec2 if upcasted).
    // This creates accumulators with matching vector dtype.
    // IMPORTANT: Thread axes are excluded from reduce_to_acc's input_ranges via is_parallel().
    //
    // Tinygrad alignment: Combine pm_reduce + pm_wmma_accumulate + gep_pushing in single pass
    // (Tinygrad: graph_rewrite(sink, pm_reduce+gep_pushing, ctx=ReduceContext()))
    // This ensures GEP simplification happens atomically with REDUCE transformation,
    // preventing dtype mismatches in horizontal reduction.
    //
    // pm_wmma_accumulate fuses Add into WMMA's accumulator: WMMA(a,b,c) + add → WMMA(a,b,c+add)
    // Tensor cores have built-in accumulation, so this is more efficient.
    let pm_reduce_combined = pm_reduce() + pm_wmma_accumulate() + gep_pushing_patterns();
    let reduced = graph_rewrite(&pm_reduce_combined, expanded, &mut ());
    tracing::debug!(ast.optimized = reduced.tree(), "after pm_reduce");

    // pm_add_gpudims: Convert GLOBAL/LOCAL RANGE → SPECIAL thread indices (Tinygrad gpudims.py)
    // Only applies for GPU backends (has_local). Must run AFTER pm_reduce (which handles accumulator
    // placement based on range types) and BEFORE pm_add_loads.
    // Creates SPECIAL ops named "gidx0", "lidx0", etc. with dimension limiting for hardware constraints.
    let with_gpudims = if let Some(ren) = renderer {
        if ren.has_local {
            // GPU backend: inject thread indices
            graph_rewrite(&pm_add_gpudims(), reduced, &mut ren.clone())
        } else {
            reduced
        }
    } else {
        reduced
    };
    tracing::debug!(ast.optimized = with_gpudims.tree(), "after pm_add_gpudims");

    // Add explicit LOAD ops for INDEX sources consumed by arithmetic ops.
    // This separates INDEX (returns indices for STORE scatter) from LOAD (performs gather).
    // Based on Tinygrad's pm_add_loads (devectorizer.py:320-326).
    let with_loads = graph_rewrite(&pm_add_loads(), with_gpudims, &mut ());

    // ALU Devectorization + GEP Simplification
    //
    // Tinygrad runs no_vectorized_alu + gep_pushing in a SINGLE graph_rewrite call
    // (codegen/__init__.py:78-81: pm_devectorize = sym+devectorize+load_store_folding+...).
    // This is critical: gep_pushing resolves GEPs incrementally as no_vectorized_alu
    // creates them, preventing graph explosion. Running them separately causes
    // no_vectorized_alu to materialize ALL GEPs first, then gep_pushing processes
    // the massive graph in bulk, hitting the 100k iteration limit.

    let processed = if devectorize_alu {
        let combined = crate::devectorize::no_vectorized_alu() + gep_pushing_patterns();
        graph_rewrite(&combined, with_loads, &mut ())
    } else {
        with_loads
    };

    // Cleanup remaining GEPs (always runs, handles non-devectorize paths too)
    let cleaned = graph_rewrite(&gep_pushing_patterns(), processed, &mut ());
    tracing::debug!(ast.optimized = cleaned.tree(), "after pm_pushing_patterns");

    // Second pass of pm_add_loads: expansion may create new INDEX ops that need LOAD wrapping.
    // NOTE: Uses top-down graph_rewrite to match Tinygrad's approach.
    let with_loads2 = graph_rewrite(&pm_add_loads(), cleaned, &mut ());
    tracing::debug!(ast.optimized = with_loads2.tree(), "after pm_add_loads");

    // Devectorize pass: group contiguous memory accesses
    // Uses direct vector index analysis (VConst/UNROLL patterns) for termination safety
    let devectorized_mem = crate::devectorize::devectorize(&with_loads2);
    tracing::debug!(ast.optimized = devectorized_mem.tree(), "after devectorize");

    // Bool devectorization: Convert <N x i1> ALU ops to scalar ops + VECTORIZE.
    // LLVM's bool vectors are broken (no formal ABI, segfaults in codegen).
    // Based on Tinygrad's no_vectorized_alu approach.
    // Must run BEFORE reduce devectorization so comparisons are already scalar.
    let devectorized = graph_rewrite(&pm_bool_devectorize(), devectorized_mem, &mut ());
    tracing::debug!(ast.optimized = devectorized.tree(), "after pm_bool_devectorize");

    // Unified REDUCE devectorization: Handles all 3 mutually exclusive cases:
    // - K-vectorized: CONTRACT source → N scalar REDUCEs + tree_reduce (SLP optimization)
    // - Bool reduce: matching vcounts, bool dtype → N scalar REDUCEs + VECTORIZE (<N x i1> workaround)
    // - Horizontal: src_vcount > out_vcount → stride-pattern GEPs + ALU chain
    // Based on Tinygrad's pm_reduce (devectorizer.py:283-316).
    let reduce_devec = graph_rewrite(&pm_reduce_devectorize(), devectorized, &mut ());
    tracing::debug!(ast.optimized = reduce_devec.tree(), "after pm_reduce_devectorize");

    // =========================================================================
    // Index dtype lowering (Tinygrad: pm_lower_index_dtype)
    // =========================================================================
    // Convert abstract Index dtype to concrete i32/i64 based on value bounds.
    // Must run AFTER devectorize (which creates VECTORIZE nodes with Index dtype)
    // so that all Index operations are properly lowered before codegen.
    //
    // Tinygrad runs devectorize BEFORE pm_lower_index_dtype for this reason.
    let with_lowered_idx = graph_rewrite(&crate::symbolic::pm_lower_index_dtype(), reduce_devec, &mut ());
    tracing::debug!(ast.optimized = with_lowered_idx.tree(), "after pm_lower_index_dtype");

    // Post-Index Symbolic: Constant folding after index lowering.
    let with_lowered_idx = graph_rewrite(&symbolic_simple(), with_lowered_idx, &mut ());
    tracing::debug!(ast.optimized = with_lowered_idx.tree(), "after post-index symbolic");

    // =========================================================================
    // Stage 18: Late rewrite patterns (Tinygrad: pm_decomp = symbolic_simple + get_late_rewrite_patterns)
    // =========================================================================
    // Apply decomposition patterns based on renderer capabilities.
    // Tinygrad: graph_rewrite(sink, pm_decomp, ctx=ren.device, name="decompositions")
    //
    // Key patterns:
    // - MULACC (FMA): a*b+c → MulAcc(a,b,c) if backend supports it
    // - Transcendental decompositions (EXP2, LOG2, SIN) if backend lacks native support
    // - MAX → CMPLT + WHERE if backend lacks native MAX
    //
    // Currently we always apply FMA decomposition for float types since most
    // backends benefit from explicit FMA ops (LLVM generates fma intrinsics).
    let pm_decomp = symbolic_simple() + get_late_rewrite_patterns(renderer);
    let decomposed = graph_rewrite(&pm_decomp, with_lowered_idx, &mut ());
    tracing::debug!(ast.optimized = decomposed.tree(), "Stage 18: after pm_decomp");

    // =========================================================================
    // Stage 19: Final rewrite / Render preparation (Tinygrad: pm_render)
    // =========================================================================
    // Prepare AST for codegen by converting vector constants and normalizing patterns.
    // Key patterns:
    // - Vector CONST → VECTORIZE of scalar CONSTs
    // - VCONST → VECTORIZE of scalar CONSTs
    // - CAT → VECTORIZE (for uniform elements)
    // - Single-element VECTORIZE unwrapping
    //
    // This moves pm_render from codegen to schedule pipeline for proper stage alignment.
    let rendered = graph_rewrite(&pm_render(), decomposed, &mut ());
    tracing::debug!(ast.optimized = rendered.tree(), "Stage 19: after pm_render");

    // Bool storage: Convert bool LOAD/STORE to uint8 to avoid LLVM i1 garbage bits.
    // LLVM's i1 type when stored to memory can have garbage in upper 7 bits.
    // Must run LAST, after all other transformations that might create bool stores.
    // Based on Tinygrad's PTX/NIR bool→uint8 patterns.
    let bs = graph_rewrite(&bool_storage_patterns(), rendered, &mut ());
    tracing::debug!(ast.optimized = bs.tree(), "after bool_storage_pattern");
    bs
}

/// Get late rewrite patterns based on renderer capabilities.
///
/// Based on Tinygrad's `get_late_rewrite_patterns` (decompositions.py:320-367).
///
/// Returns patterns for:
/// - MULACC (FMA): `a*b+c → MulAcc(a,b,c)` for float types
/// - MOD → AND: `x % 2^n → x & (2^n-1)` for power-of-two modulus
/// - MUL → SHL: `x * 2^n → x << n` for power-of-two multiplier
/// - NEG from MUL: `x * -1 → NEG(x)`
/// - MAX decomposition: `MAX(a,b) → (a<b).where(b,a)` if backend lacks native MAX
/// - SQRT decomposition: `SQRT(x) → POW(x, 0.5)` if backend lacks native SQRT
///
/// # Arguments
///
/// * `renderer` - Optional renderer to check supported ops. If None, applies all patterns
///   except conditional decompositions (MAX, SQRT).
fn get_late_rewrite_patterns(renderer: Option<&Renderer>) -> crate::TypedPatternMatcher {
    // Start with FMA decomposition - most backends benefit from explicit FMA ops
    // (LLVM generates fma intrinsics, GPUs have native FMA support)
    let mut patterns = pm_fma_decomposition();

    // Always apply pure optimizations that generate cheaper instructions:
    // - MOD → AND for power-of-2 (bitwise cheaper than modulo)
    // - MUL → SHL for power-of-2 (shift cheaper than multiply)
    // - DIV → SHR for power-of-2 (shift cheaper than divide) - Tinygrad: decompositions.py:340-344
    // - FDIV → MUL for float constant divisor (multiply cheaper than divide) - Tinygrad: decompositions.py:364-366
    // - NEG from MUL (negation cheaper than multiply by -1)
    // - Comparison negations: !(x<c) → (c-1)<x, etc. - Tinygrad: decompositions.py:354-361
    // - Fast division: x//d → (x*M)>>S for non-power-of-2 constants - Tinygrad: decompositions.py fast_idiv
    patterns = patterns
        + pm_mod_to_and()
        + pm_mul_to_shl()
        + pm_div_to_shr()
        + pm_fdiv_to_mul()
        + pm_neg_from_mul()
        + pm_comparison_negations()
        + crate::symbolic::fast_division_patterns();

    // Conditional decompositions based on backend capabilities.
    // These only apply if the backend lacks native support for the operation.
    // For now, we assume all backends support MAX and SQRT natively (LLVM, CUDA, Metal all do).
    // When we add backends that lack support, we can check renderer capabilities.
    //
    // Future: Add renderer.supports_max() / renderer.supports_sqrt() checks:
    // if !renderer.map_or(true, |r| r.supports_max()) {
    //     patterns = patterns + pm_max_decomposition();
    // }
    // if !renderer.map_or(true, |r| r.supports_sqrt()) {
    //     patterns = patterns + pm_sqrt_decomposition();
    // }
    let _ = renderer; // Will be used for capability checks when needed

    patterns
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
    // Pass the renderer to enable GPU dimension injection for GPU backends.
    apply_post_optimization_with_renderer(optimized, config.devectorize_alu, Some(renderer))
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
