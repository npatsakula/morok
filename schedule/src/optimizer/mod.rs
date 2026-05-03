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

use crate::devectorize::{
    Fp8DecompCtx, bool_storage_patterns, pm_float_decomp, pm_float_decomp_store, pm_reduce, pm_render,
    pm_wmma_accumulate,
};
use crate::gpudims::pm_add_gpudims;
use crate::rangeify::patterns::{
    pm_add_loads, pm_comparison_negations, pm_demorgan, pm_div_to_shr, pm_erf_decomposition, pm_fdiv_to_mul,
    pm_fma_decomposition, pm_load_collapse, pm_mod_to_and, pm_mul_to_shl, pm_neg_from_mul, pm_shl_add_to_mulacc,
    pm_threefry_decomp, rangeify_codegen_with_kernel_ctx,
};
use crate::rangeify::pm_add_buffers_local_patterns;
use crate::rangeify::transforms::{pm_flatten_range, pm_simplify_ranges, pm_split_ranges};
use crate::rewrite::graph_rewrite;
use crate::symbolic::patterns::{gep_pushing_patterns, sym, symbolic, symbolic_simple};
use std::sync::{Arc, LazyLock};

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
/// * `ast` - The kernel AST (CALL body AST)
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
/// - devectorize: Combined pass (sym + devec + load_store_folding + correct_load_store + indexing)
/// - bool_storage_patterns: Convert bool LOAD/STORE to uint8
///
/// NOTE: We do NOT apply FMA decomposition (a*b+c → MulAcc). Following Tinygrad's
/// approach, we let LLVM's optimizer fuse MUL+ADD into FMA when beneficial.
///
/// # Arguments
///
/// * `ast` - The kernel AST to optimize
///
/// Called by both heuristic and beam search paths for consistent behavior.
/// For GPU pipelines, use `apply_post_optimization_with_renderer` to enable GPU dimension injection.
#[tracing::instrument(skip_all)]
pub fn apply_post_optimization(ast: Arc<morok_ir::UOp>) -> Arc<morok_ir::UOp> {
    apply_post_optimization_with_renderer(ast, None)
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
/// * `renderer` - Optional renderer for GPU dimension injection
#[tracing::instrument(skip_all)]
pub fn apply_post_optimization_with_renderer(
    ast: Arc<morok_ir::UOp>,
    renderer: Option<&Renderer>,
) -> Arc<morok_ir::UOp> {
    // Save metadata before graph_rewrite destroys it (e.g., KernelInfo with kernel name)
    let saved_metadata = ast.metadata_raw();

    tracing::debug!(ast.initial = ast.tree(), node_count = ast.node_count(), "kernel initial");

    // Multi-index INDEX is normalized to single-index during devectorize
    // via pm_linearize_multi_index. Backends consume only linearized INDEX.

    // =========================================================================
    // Stage 8: Post-opt symbolic + WHERE movement (Tinygrad: sym + pm_move_where_on_load)
    // This MUST run BEFORE expander to optimize conditionals before expansion.
    // =========================================================================
    let t_stage = std::time::Instant::now();
    // Tinygrad: sym + pm_move_where_on_load (pm_move_where_on_load only at this stage, not global)
    static POST_OPT_SYM: LazyLock<crate::TypedPatternMatcher> =
        LazyLock::new(|| sym().clone() + crate::symbolic::patterns::pm_move_where_on_load());
    let with_symbolic = graph_rewrite(&*POST_OPT_SYM, ast, &mut ());
    tracing::debug!(
        ast.optimized = with_symbolic.tree(),
        node_count = with_symbolic.node_count(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "Stage 8: after post-opt symbolic"
    );

    // =========================================================================
    // Stage 9: Expander (Tinygrad: sym + pm_pre_expander + pm_group_for_reduce + expander)
    // =========================================================================
    // UNROLL expansion: Expand UNROLL ops to vectorized operations (Tinygrad expander.py)
    // CRITICAL: Must run BEFORE pm_reduce so that REDUCE sees its actual vectorized dtype.
    // In Tinygrad, expander runs first, then pm_reduce sees the expanded REDUCE with vec2 dtype.
    // This allows reduce_to_acc to create accumulators with the correct vector dtype.
    let t_stage = std::time::Instant::now();
    let expanded = crate::expand::pre_expand(&with_symbolic);
    tracing::debug!(
        ast.optimized = expanded.tree(),
        node_count = expanded.node_count(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "Stage 9: after pre_expand"
    );

    // =========================================================================
    // Stage 10: Add local buffers (Tinygrad: pm_add_buffers_local + rangeify_codegen)
    // =========================================================================
    // Converts BUFFERIZE(Local) → DEFINE_LOCAL + STORE + LOAD for GROUP_REDUCE.
    // Also strips leftover CONTIGUOUS and NOOP nodes.
    // Must run AFTER expander (which creates BUFFERIZE_LOCAL) and BEFORE pm_reduce.
    //
    // CRITICAL: Combine pm_add_buffers_local + rangeify_codegen in a SINGLE pass
    // (like Tinygrad) to ensure CONTIGUOUS is stripped BEFORE bufferize_to_store
    // sees it. Otherwise CONTIGUOUS(BUFFER) becomes the STORE value directly,
    // which fails codegen because STORE expects a value, not a buffer pointer.
    // Helper closure: check for UNROLL(GROUP) in graph
    let check_unroll_group = |label: &str, root: &Arc<morok_ir::UOp>| {
        for node in root.toposort() {
            if let morok_ir::Op::Unroll { src, unroll_axes, .. } = node.op()
                && matches!(src.op(), morok_ir::Op::Group { .. })
            {
                tracing::error!(id = node.id, axes = ?unroll_axes, stage = label, "UNROLL(GROUP) found!");
            }
        }
    };

    let t_stage = std::time::Instant::now();
    let with_local_buffers = {
        let mut buf_ctx = crate::rangeify::RangeifyBufferContext::new();
        static PM_LOCAL_BUF: LazyLock<crate::TypedPatternMatcher<crate::rangeify::RangeifyBufferContext>> =
            LazyLock::new(|| pm_add_buffers_local_patterns() + rangeify_codegen_with_kernel_ctx());
        graph_rewrite(&*PM_LOCAL_BUF, expanded, &mut buf_ctx)
    };
    tracing::debug!(
        ast.optimized = with_local_buffers.tree(),
        node_count = with_local_buffers.node_count(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "Stage 10: after add local buffers"
    );
    if cfg!(debug_assertions) {
        check_unroll_group("after_add_local_buffers", &with_local_buffers);
    }

    let t_stage = std::time::Instant::now();
    static PM_REDUCE_COMBINED: LazyLock<crate::TypedPatternMatcher<crate::devectorize::ReduceContext>> =
        LazyLock::new(|| pm_reduce() + pm_wmma_accumulate().with_context() + gep_pushing_patterns().with_context());
    let mut reduce_ctx = crate::devectorize::ReduceContext::default();
    let reduced = graph_rewrite(&*PM_REDUCE_COMBINED, with_local_buffers, &mut reduce_ctx);
    tracing::debug!(
        ast.optimized = reduced.tree(),
        node_count = reduced.node_count(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "after pm_reduce"
    );
    if cfg!(debug_assertions) {
        check_unroll_group("after_pm_reduce", &reduced);
    }

    let t_stage = std::time::Instant::now();
    let with_gpudims = if let Some(ren) = renderer {
        if ren.has_local || ren.has_threads {
            graph_rewrite(&pm_add_gpudims(), reduced, &mut ren.clone())
        } else {
            reduced
        }
    } else {
        reduced
    };
    tracing::debug!(
        ast.optimized = with_gpudims.tree(),
        node_count = with_gpudims.node_count(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "after pm_add_gpudims"
    );
    if cfg!(debug_assertions) {
        check_unroll_group("after_pm_add_gpudims", &with_gpudims);
    }

    let t_stage = std::time::Instant::now();
    let with_loads = graph_rewrite(pm_add_loads(), with_gpudims, &mut ());
    tracing::debug!(
        ast.optimized = with_loads.tree(),
        node_count = with_loads.node_count(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "after pm_add_loads"
    );
    if cfg!(debug_assertions) {
        check_unroll_group("after_pm_add_loads", &with_loads);
        // Also check for any UNROLL or CONTRACT
        for node in with_loads.toposort() {
            if let morok_ir::Op::Unroll { src, unroll_axes, .. } = node.op() {
                tracing::error!(
                    id = node.id,
                    src_op = src.op().as_ref(),
                    axes = ?unroll_axes,
                    "BEFORE devectorize: found UNROLL!"
                );
            }
            if let morok_ir::Op::Contract { src, upcast_ranges, .. } = node.op() {
                tracing::error!(
                    id = node.id,
                    src_op = src.op().as_ref(),
                    axes = ?upcast_ranges,
                    "BEFORE devectorize: found CONTRACT!"
                );
            }
        }
    }

    // ALU devectorization happens inside devectorize() Phase 1, alongside expand_index
    // and full symbolic (including gep_pushing). This matches Tinygrad's structure where
    // no_vectorized_alu runs in the same pass as load_store_folding (step 14).
    // Previously, an isolated pass here combined no_vectorized_alu + gep_pushing without
    // load/store folding, causing graph explosion on wide VECTORIZE nodes (VECTORIZE(135)).
    // Tinygrad Stage 14: devectorize — single combined pass handles ALL devectorization
    // including bool ALU (via no_vectorized_alu). No separate pm_bool_devectorize or
    // pm_reduce_devectorize passes — matching Tinygrad's pipeline exactly.
    let t_stage = std::time::Instant::now();
    let devectorized = crate::devectorize::devectorize(&with_loads);
    tracing::debug!(
        ast.optimized = devectorized.tree(),
        node_count = devectorized.node_count(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "after devectorize"
    );
    check_unroll_group("after_devectorize", &devectorized);

    // Tinygrad Stage 15: pm_lower_index_dtype + load_store_indexing + gep_pushing
    let t_stage = std::time::Instant::now();
    static PM_LOWER_COMBINED: LazyLock<crate::TypedPatternMatcher> = LazyLock::new(|| {
        crate::symbolic::pm_lower_index_dtype()
            + crate::devectorize::load_store_indexing_patterns()
            + gep_pushing_patterns()
    });
    let with_lowered_idx = graph_rewrite(&*PM_LOWER_COMBINED, devectorized, &mut ());
    tracing::debug!(
        ast.optimized = with_lowered_idx.tree(),
        node_count = with_lowered_idx.node_count(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "after pm_lower_index_dtype"
    );
    check_unroll_group("after_pm_lower_index_dtype", &with_lowered_idx);

    // Tinygrad: symbolic (step 16) — full symbolic (includes gep_pushing, div_and_mod, etc.)
    let t_stage = std::time::Instant::now();
    static POST_INDEX_SYM: LazyLock<crate::TypedPatternMatcher> = LazyLock::new(|| symbolic().clone());
    let with_lowered_idx = graph_rewrite(&*POST_INDEX_SYM, with_lowered_idx, &mut ());
    tracing::debug!(
        ast.optimized = with_lowered_idx.tree(),
        node_count = with_lowered_idx.node_count(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "after post-index symbolic"
    );

    // =========================================================================
    // Stage 18-19: Decompositions + Render (Tinygrad: pm_decomp + pm_render in one pass)
    // =========================================================================
    let t_stage = std::time::Instant::now();
    static PM_FINAL: LazyLock<crate::TypedPatternMatcher> =
        LazyLock::new(|| symbolic_simple() + get_late_rewrite_patterns() + pm_render());
    let rendered = graph_rewrite(&*PM_FINAL, with_lowered_idx, &mut ());
    tracing::debug!(
        ast.optimized = rendered.tree(),
        node_count = rendered.node_count(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "Stage 18-19: after pm_decomp + pm_render"
    );
    assert_gated_loads_have_alt("after_pm_decomp_pm_render", &rendered);

    // Merge sibling ENDs that share the same reduce ranges.
    // pm_decomp+pm_render can create new sibling ENDs (e.g. by rewriting computations
    // inside an END differently per vector lane). merge_reduce_ends ran earlier in
    // pm_reduce but only caught ENDs that existed at that point.
    let t_merge = std::time::Instant::now();
    let rendered = crate::devectorize::merge_sibling_ends(&rendered);
    tracing::debug!(
        ast.optimized = rendered.tree(),
        node_count = rendered.node_count(),
        elapsed_ms = t_merge.elapsed().as_millis() as u64,
        "after merge_sibling_ends"
    );

    // FP8 float decomposition: promote FP8 computation to Float16 via bitwise conversion.
    // Uses graph_rewrite_with_bpm: STORE pattern in bpm (sees ORIGINAL children to detect
    // FP8 buffer ptrs), all other patterns in pm (sees OPTIMIZED children).
    // Run once per FP8 type. Tinygrad: codegen/__init__.py:97-99
    let t_stage = std::time::Instant::now();
    let fp8_pm = pm_float_decomp();
    let fp8_bpm = pm_float_decomp_store();
    let mut fp8_decomposed = rendered;
    for (fr, to) in [
        (morok_dtype::ScalarDType::FP8E5M2, morok_dtype::ScalarDType::Float16),
        (morok_dtype::ScalarDType::FP8E4M3, morok_dtype::ScalarDType::Float16),
    ] {
        let mut ctx = Fp8DecompCtx { from: fr, to };
        fp8_decomposed = morok_ir::rewrite::graph_rewrite_with_bpm(&fp8_pm, &fp8_bpm, fp8_decomposed, &mut ctx);
    }
    tracing::debug!(
        ast.optimized = fp8_decomposed.tree(),
        node_count = fp8_decomposed.node_count(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "after pm_float_decomp"
    );
    assert_gated_loads_have_alt("after_pm_float_decomp", &fp8_decomposed);

    let t_stage = std::time::Instant::now();
    let bs = graph_rewrite(bool_storage_patterns(), fp8_decomposed, &mut ());
    tracing::debug!(
        ast.optimized = bs.tree(),
        node_count = bs.node_count(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "after bool_storage_pattern"
    );
    assert_gated_loads_have_alt("after_bool_storage_pattern", &bs);

    // Re-attach metadata (e.g., KernelInfo) that was lost during graph rewrites
    match saved_metadata {
        Some(meta) => bs.with_metadata_raw(meta),
        None => bs,
    }
}

fn assert_gated_loads_have_alt(stage: &str, root: &Arc<morok_ir::UOp>) {
    for node in root.toposort() {
        let morok_ir::Op::Load { index, alt, .. } = node.op() else {
            continue;
        };
        if index_has_gate(index) && alt.is_none() {
            panic!("pipeline invariant violation ({stage}): gated LOAD {} has no alt value", node.id);
        }
    }
}

fn index_has_gate(index: &Arc<morok_ir::UOp>) -> bool {
    match index.op() {
        morok_ir::Op::Index { gate: Some(_), .. } => true,
        morok_ir::Op::Cast { src, .. } => index_has_gate(src),
        _ => false,
    }
}

/// Late rewrite patterns for algebraic decompositions.
///
/// Based on Tinygrad's `get_late_rewrite_patterns` (decompositions.py:438-480).
///
/// Returns patterns for:
/// - MULACC (FMA): `a*b+c → MulAcc(a,b,c)` for float types
/// - MOD → AND: `x % 2^n → x & (2^n-1)` for power-of-two modulus
/// - MUL → SHL: `x * 2^n → x << n` for power-of-two multiplier
/// - NEG from MUL: `x * -1 → NEG(x)`
/// - Fast integer division (magic number multiplication)
fn get_late_rewrite_patterns() -> &'static crate::TypedPatternMatcher {
    // All current backends support MAX and SQRT natively (LLVM, CUDA, Metal).
    // When we add backends that lack support, this should take a capability set
    // (like Tinygrad's `ops: tuple[Ops, ...]`) and conditionally include patterns.
    static CACHED: LazyLock<crate::TypedPatternMatcher> = LazyLock::new(|| {
        pm_fma_decomposition()
            + pm_erf_decomposition()
            + pm_mod_to_and()
            + pm_mul_to_shl()
            + pm_div_to_shr()
            + pm_fdiv_to_mul()
            + pm_neg_from_mul()
            + pm_demorgan()
            + pm_shl_add_to_mulacc()
            + pm_threefry_decomp()
            + pm_comparison_negations()
            + crate::symbolic::fast_division_patterns()
            + pm_mod_to_idiv()
    });
    &CACHED
}

/// MOD → IDIV decomposition (Tinygrad decompositions.py:457).
///
/// `x % d → x - d*(x//d)` for non-power-of-2 constant divisors.
/// Runs AFTER fast_division_patterns so the resulting IDIV gets decomposed
/// to magic-number multiplication. Without this, standalone MOD nodes
/// for non-power-of-2 divisors survive to codegen unlowered.
fn pm_mod_to_idiv() -> &'static crate::TypedPatternMatcher {
    crate::cached_patterns! {
        Mod(x, d @const(d_val))
            if x.dtype().is_int()
            && matches!(d_val.try_int(), Some(v) if v > 1 && !((v as u64).is_power_of_two()))
            => {
                // x % d → x - d * (x // d)
                let div = x.idiv(d);
                let mul = d.try_mul(&div).ok()?;
                x.try_sub(&mul).ok()
            },
    }
}

/// Apply per-kernel pre-optimization passes.
///
/// These stages run BEFORE heuristic/beam optimization, per-kernel
/// (Tinygrad: inside `full_rewrite_to_sink()`, codegen/__init__.py:28-51).
///
/// Stages:
/// 0. Movement ops + syntactic sugar (`pm_mops + pm_syntactic_sugar`, bottom-up)
/// 1. Load collapse (`pm_load_collapse`)
/// 2. Split ranges + flatten (`pm_split_ranges + pm_flatten_range`)
/// 3. Symbolic + flatten (`sym + pm_flatten_range`)
/// 4. Simplify ranges (`pm_simplify_ranges`)
///
/// Called by both heuristic and beam search paths.
#[tracing::instrument(skip_all)]
pub fn apply_pre_optimization(ast: Arc<morok_ir::UOp>) -> Arc<morok_ir::UOp> {
    tracing::debug!(ast.initial = ast.tree(), node_count = ast.node_count(), "kernel initial");

    use crate::rangeify::transforms::SplitRangesContext;

    let t_stage = std::time::Instant::now();
    use crate::rangeify::patterns::{movement_op_patterns, pm_syntactic_sugar};
    use crate::rewrite::graph_rewrite_bottom_up;
    static PM_EARLY_MOPS: LazyLock<crate::TypedPatternMatcher> =
        LazyLock::new(|| movement_op_patterns() + pm_syntactic_sugar());
    let mut sink = graph_rewrite_bottom_up(&*PM_EARLY_MOPS, ast, &mut ());
    tracing::debug!(
        ast.pre = sink.tree(),
        node_count = sink.node_count(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "pre-opt: movement ops + syntactic sugar complete"
    );

    let t_stage = std::time::Instant::now();
    sink = graph_rewrite(pm_load_collapse(), sink, &mut ());
    tracing::debug!(
        ast.pre = sink.tree(),
        node_count = sink.node_count(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "pre-opt: load collapse complete"
    );

    let t_stage = std::time::Instant::now();
    let mut split_ctx = SplitRangesContext::default();
    sink = graph_rewrite(&pm_split_ranges(), sink, &mut split_ctx);
    sink = graph_rewrite(pm_flatten_range(), sink, &mut ());
    tracing::debug!(
        ast.pre = sink.tree(),
        node_count = sink.node_count(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "pre-opt: split ranges complete"
    );

    let t_stage = std::time::Instant::now();
    // Tinygrad: sym + pm_flatten_range (pre-opt uses full sym tier)
    static PM_SYM_FLATTEN: LazyLock<crate::TypedPatternMatcher> = LazyLock::new(|| sym().clone() + pm_flatten_range());
    sink = graph_rewrite(&*PM_SYM_FLATTEN, sink, &mut ());
    tracing::debug!(
        ast.pre = sink.tree(),
        node_count = sink.node_count(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "pre-opt: symbolic + flatten complete"
    );

    let t_stage = std::time::Instant::now();
    static PM_SIMPLIFY_FLATTEN: LazyLock<crate::TypedPatternMatcher> =
        LazyLock::new(|| pm_flatten_range() + pm_simplify_ranges());
    sink = graph_rewrite(&*PM_SIMPLIFY_FLATTEN, sink, &mut ());
    tracing::debug!(
        ast.pre = sink.tree(),
        node_count = sink.node_count(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "pre-opt: simplify ranges complete"
    );

    sink
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
    // Pre-optimization: per-kernel stages (Tinygrad: full_rewrite_to_sink)
    let pre_optimized = apply_pre_optimization(ast);

    let optimized = match config.strategy {
        OptStrategy::None => pre_optimized, // No heuristic optimization, but post-optimization still needed
        OptStrategy::Heuristic => optimize_heuristic(pre_optimized, renderer, &config.heuristics),
        OptStrategy::Beam { .. } => {
            // Beam search requires a compile_and_time function.
            // Use optimize_kernel_beam() for actual beam search.
            // Fall back to heuristics for the simple API.
            optimize_heuristic(pre_optimized, renderer, &config.heuristics)
        }
    };

    // apply_post_optimization contains correctness transforms (pm_add_loads wraps INDEX
    // with LOAD for arithmetic ops) and must run even when optimizations are disabled.
    // Pass the renderer to enable GPU dimension injection for GPU backends.

    apply_post_optimization_with_renderer(optimized, Some(renderer))
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
    // Step 0: Per-kernel pre-optimization (Tinygrad: full_rewrite_to_sink)
    let pre_optimized = apply_pre_optimization(ast);

    // Step 1: Create scheduler (AST already simplified by apply_pre_optimization Stage 3)
    let mut scheduler = Scheduler::new(pre_optimized, renderer.clone());

    // Step 2: Convert loops to global (for GPU parallelization)
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
    let pre_optimized = apply_pre_optimization(ast);
    let mut scheduler = Scheduler::new(pre_optimized, renderer.clone());
    let _ = scheduler.convert_loop_to_global(); // GPU: LOOP→GLOBAL
    // Note: Don't apply threading here - let beam search explore THREAD actions naturally.
    // Heuristics apply threading via hand_coded_optimizations() with config.thread_count.
    scheduler
}

/// Apply heuristic-based optimizations.
fn optimize_heuristic(ast: Arc<morok_ir::UOp>, renderer: &Renderer, config: &HeuristicsConfig) -> Arc<morok_ir::UOp> {
    // Step 1: Create scheduler (AST already simplified by apply_pre_optimization Stage 3)
    let mut scheduler = Scheduler::new(ast, renderer.clone());

    // Step 3: Convert axes for parallelization/vectorization
    let _ = scheduler.convert_loop_to_global(); // GPU: LOOP→GLOBAL
    let _ = scheduler.convert_outer_to_loop(); // CPU: OUTER→LOOP (enables UPCAST)

    // Step 4: Apply hand-coded heuristics with config
    heuristics::hand_coded_optimizations(&mut scheduler, config);

    // Step 5: Extract optimized AST
    scheduler.get_optimized_ast(None)
}
