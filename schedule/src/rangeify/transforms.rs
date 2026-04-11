//! Consolidated transformation functions for rangeify.
//!
//! This module contains:
//! - Main `rangeify()` entry point
//! - Movement op → BUFFERIZE+INDEX transformation helpers
//! - BUFFERIZE → STORE conversion
//! - Reduction simplifications (reduce_unparented, reduce_collapse)
//! - Range flattening (flatten_range_impl)
//! - Cycle detection (find_bufs)
//!
//! Consolidated from: transform.rs, bufferize_to_store.rs, reduce_simplify.rs,
//! flatten_range.rs, cycle_detection.rs

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::context::RangeifyContext;
use super::indexing::IndexingContext;
use super::kernel::KernelContext;
use morok_ir::shape::Shape;
use morok_ir::{AddrSpace, AxisType, BufferizeOpts, ConstValue, DType, Op, UOp, UOpKey};
use smallvec::{SmallVec, smallvec};

// ============================================================================
// ADD_TAGS — Tinygrad rangeify.py:542-555
// ============================================================================

/// Context for the add_tags pass.
pub struct AddTagsCtx {
    /// Sequential list of tagged UOps (index = tag value).
    pub uop_list: Vec<Arc<UOp>>,
    /// UOps excluded from tagging (e.g., nodes inside KERNEL/CALL).
    excluded: HashSet<UOpKey>,
}

impl Default for AddTagsCtx {
    fn default() -> Self {
        Self::new()
    }
}

impl AddTagsCtx {
    pub fn new() -> Self {
        Self { uop_list: Vec::new(), excluded: HashSet::new() }
    }
}

/// Ops that should NOT be tagged (Tinygrad rangeify.py:552-553).
/// MStack/MSelect are handled separately with conditional logic.
/// Note: Tinygrad also excludes LUNIQUE — Morok uses counter-based local IDs instead.
fn should_skip_tag(op: &Op) -> bool {
    matches!(
        op,
        Op::Param { .. }
            | Op::Const(_)
            | Op::Device(_)
            | Op::Unique(_)
            | Op::DefineVar { .. }
            | Op::Bind { .. }
            | Op::End { .. }
            | Op::Range { .. }
    ) || op.is_movement()
}

/// Create the add_tags pattern matcher (Tinygrad rangeify.py:550-555).
///
/// Assigns sequential integer tags `[i]` to each taggable UOp. Tags track which
/// original tensor UOps map to which final kernel outputs through the pipeline.
pub fn add_tags_patterns() -> crate::TypedPatternMatcher<AddTagsCtx> {
    crate::patterns! {
        @context AddTagsCtx;
        // Wildcard: handles all ops, applies tag logic per Tinygrad rangeify.py:542-554
        x => {
            if x.tag().is_some() || ctx.excluded.contains(&UOpKey(x.clone())) { return None; }
            // Kernel/Call: exclude entire subgraph from tagging (Tinygrad line 544-546)
            if let Op::Kernel { ast, .. } = x.op() {
                for u in ast.toposort() {
                    ctx.excluded.insert(UOpKey(u));
                }
            }
            if should_skip_tag(x.op()) { return None; }
            // Index-typed scalars are not tagged (Tinygrad line 547)
            if x.dtype().base() == morok_dtype::ScalarDType::Index { return None; }
            // MStack/MSelect: only tag if NOT all sources are PARAM (Tinygrad line 554)
            if matches!(x.op(), Op::MStack { .. } | Op::MSelect { .. })
                && x.op().sources().iter().all(|s| matches!(s.op(), Op::Param { .. }))
            {
                return None;
            }
            ctx.uop_list.push(x.clone());
            Some(x.with_tag(smallvec![ctx.uop_list.len() - 1]))
        },
    }
}

// ============================================================================
// PUBLIC API
// ============================================================================

/// Main rangeify transformation entry point.
///
/// Converts movement operations (RESHAPE, PERMUTE, EXPAND, PAD, SHRINK, FLIP)
/// into BUFFERIZE + INDEX operations with explicit loop ranges.
pub fn rangeify(
    sink: Arc<UOp>,
    pcontig_config: Option<&super::kernel::PcontigConfig>,
) -> morok_ir::Result<(Arc<UOp>, RangeifyContext)> {
    let result = rangeify_with_map(sink, pcontig_config)?;
    Ok((result.sink, result.context))
}

/// Result of rangeify transformation.
pub struct RangeifyResult {
    /// The transformed sink node
    pub sink: Arc<UOp>,
    /// Context with range information
    pub context: RangeifyContext,
    /// Tagged UOps from add_tags pass (index = tag value).
    /// Used for tag-based becomes_map construction (Tinygrad rangeify.py:614-619).
    pub uop_list: Vec<Arc<UOp>>,
}

/// Main rangeify transformation entry point with becomes_map tracking.
///
/// Like `rangeify`, but also returns a `becomes_map` that tracks which
/// original nodes were transformed. This is essential for global graph
/// coordination when multiple tensors share subgraphs.
///
/// # Pipeline (Tinygrad-aligned)
///
/// The pipeline follows Tinygrad's structure from codegen/__init__.py:
///
/// **Stage 0**: Range assignment (run_rangeify)
/// **Stage 1**: pm_mops + pm_syntactic_sugar (BOTTOM_UP) - Early movement ops
/// **Stage 2**: pm_load_collapse - Collapse load tensor indexing
/// **Stage 3**: pm_split_ranges + pm_flatten_range - Range splitting
/// **Stage 4**: sym + pm_flatten_range - Initial symbolic (TOP_DOWN)
/// **Stage 5**: pm_simplify_ranges - Simplify/merge ranges
/// **Stage 6**: apply_opts - Post-range optimization (happens in optimizer)
#[allow(clippy::mutable_key_type)]
#[tracing::instrument(skip_all)]
pub fn rangeify_with_map(
    sink: Arc<UOp>,
    pcontig_config: Option<&super::kernel::PcontigConfig>,
) -> morok_ir::Result<RangeifyResult> {
    // add_tags: assign sequential integer tags to UOps (Tinygrad rangeify.py:575).
    // MUST run FIRST — tags track tensor identity through the entire pipeline.
    let t_stage = std::time::Instant::now();
    let mut tag_ctx = AddTagsCtx::new();
    let mut sink = crate::rewrite::graph_rewrite_bottom_up(&add_tags_patterns(), sink, &mut tag_ctx);
    let uop_list = tag_ctx.uop_list;
    tracing::debug!(
        tagged_count = uop_list.len(),
        node_count = sink.node_count(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "add_tags complete"
    );

    // Combined early pass (Tinygrad: earliest_rewrites + replace_contiguous, ctx={})
    // MUST run BEFORE range assignment so rangeify sees a cleaned graph.
    // Tinygrad (rangeify.py:577): bottom_up=True — patterns see ORIGINAL children.
    let t_stage = std::time::Instant::now();
    let early_combined = super::patterns::early_rewrites().with_context::<super::patterns::ReplaceContiguousCtx>()
        + super::patterns::replace_contiguous();
    let mut contig_ctx = super::patterns::ReplaceContiguousCtx::new();
    sink = crate::rewrite::graph_rewrite_bottom_up(&early_combined, sink, &mut contig_ctx);
    tracing::debug!(
        uop.tree = sink.tree(),
        node_count = sink.node_count(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "early rewrites + replace contiguous complete"
    );

    // Split large reductions BEFORE range assignment (Tinygrad: inside earliest_rewrites)
    let t_stage = std::time::Instant::now();
    let mut split_config = super::kernel::SplitReduceOpConfig::default();
    let split_matcher = super::patterns::split_reduceop_patterns();
    sink = crate::rewrite::graph_rewrite(&split_matcher, sink, &mut split_config);
    tracing::debug!(
        uop.tree = sink.tree(),
        node_count = sink.node_count(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "split reduceops complete"
    );

    // =========================================================================
    // Stage 0: Range assignment + apply rangeify patterns
    // Tinygrad: run_rangeify() includes pm_generate_realize_map, assign loop,
    // and pm_apply_rangeify (REDUCE_AXIS→REDUCE, PAD→WHERE, BUFFERIZE+INDEX)
    // =========================================================================
    let t_stage = std::time::Instant::now();
    let (rangeified, indexing_ctx) = super::indexing::run_rangeify(sink)?;
    sink = rangeified;
    tracing::debug!(
        uop.tree = sink.tree(),
        node_count = sink.node_count(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "Stage 0: range assignment + apply rangeify complete"
    );

    // =========================================================================
    // Mega-pass: symbolic + reduce_simplify + buffer_folding + buffer_removal
    // (Tinygrad rangeify.py:582: symbolic + pm_reduce_simplify + pm_const_buffer_folding + pm_remove_bufferize)
    //
    // One fixpoint pass combining all simplification + buffer removal.
    // Uses PcontigConfig as the shared context (buffer_removal needs it;
    // other patterns are lifted via with_context()).
    // =========================================================================
    {
        use super::kernel::PcontigConfig;
        let t_stage = std::time::Instant::now();
        use std::sync::LazyLock;
        static MEGA_PASS: LazyLock<crate::TypedPatternMatcher<PcontigConfig>> = LazyLock::new(|| {
            crate::symbolic::symbolic().with_context::<PcontigConfig>()
                + super::patterns::pm_reduce_simplify().with_context()
                + super::patterns::buffer_folding().with_context()
                + super::patterns::dead_axis_removal().with_context()
                // pm_mops: Tinygrad includes movement_op_patterns in pm_const_buffer_folding (rangeify.py:260)
                + super::patterns::movement_op_patterns().with_context()
                + super::patterns::buffer_removal_with_pcontig()
        });
        let mega_pass = &*MEGA_PASS;
        tracing::debug!(
            total_patterns = mega_pass.len(),
            wildcard_count = mega_pass.wildcard_count(),
            indexed_buckets = mega_pass.indexed_count(),
            "mega-pass pattern stats"
        );
        let mut pcontig = pcontig_config.cloned().unwrap_or_default();
        sink = crate::rewrite::graph_rewrite(mega_pass, sink, &mut pcontig);
        tracing::debug!(
            node_count = sink.node_count(),
            elapsed_ms = t_stage.elapsed().as_millis() as u64,
            "mega-pass complete"
        );
    }

    // Stages 2a-6 (load_collapse, split_ranges, symbolic+flatten, simplify_ranges,
    // split_store) now run per-kernel in optimizer::apply_pre_optimization().

    // SINK rebuild: filter sources to tagged valid output types (Tinygrad rangeify.py:585-589).
    // TODO: Full Tinygrad approach scans backward_slice for ALL tagged nodes and rebuilds
    // SINK with them. This requires Phase 7 (tag-based becomes_map) to be implemented first.
    // For now, filter existing SINK sources by tag + op type.
    if let Op::Sink { sources } = sink.op() {
        let filtered: Vec<Arc<UOp>> = sources
            .iter()
            .filter(|s| {
                let valid_op = matches!(
                    s.base().op(),
                    Op::Bufferize { .. } | Op::MStack { .. } | Op::Const(_) | Op::Param { .. } | Op::After { .. }
                );
                valid_op
            })
            .cloned()
            .collect();
        if !filtered.is_empty() && filtered.len() != sources.len() {
            tracing::debug!(
                original = sources.len(),
                filtered = filtered.len(),
                "SINK cleanup: removed invalid-type sources after mega-pass"
            );
            sink = UOp::sink(filtered);
        }
    }

    // Buffer limit enforcement
    if let Some(device) = super::patterns::extract_device_from_graph(&sink)
        && let Some(limit) = device.max_buffers()
    {
        let t_stage = std::time::Instant::now();
        let limit_matcher = super::patterns::buffer_limit_patterns(limit);
        sink = crate::rewrite::graph_rewrite(&limit_matcher, sink, &mut ());
        tracing::debug!(
            uop.tree = sink.tree(),
            elapsed_ms = t_stage.elapsed().as_millis() as u64,
            "Stage 7b: buffer limit enforcement complete"
        );
    }

    // =========================================================================
    // Stage 8: Post-range optimization happens in optimizer module (apply_opts)
    // =========================================================================

    // Build RangeifyContext for return
    let rangeify_ctx = RangeifyContext { range_counter: indexing_ctx.range_counter(), range_map: HashMap::new() };

    Ok(RangeifyResult { sink, context: rangeify_ctx, uop_list })
}

/// Pattern matcher for range flattening.
///
/// Based on Tinygrad's pm_flatten_range (simplify.py:14-17).
/// Extracts all RANGE operations from nested END/REDUCE/STORE structures.
pub fn pm_flatten_range() -> &'static crate::TypedPatternMatcher {
    crate::cached_patterns! {
        r @ End { computation: _, ranges } if !ranges.is_empty() => |r| flatten_range_impl(r),
        r @ Reduce { src: _, ranges, reduce_op: _ } if !ranges.is_empty() => |r| flatten_range_impl(r),
        r @ Store { index: _, value: _, ranges } if !ranges.is_empty() => |r| flatten_range_impl(r),
    }
}

// ============================================================================
// RANGE SPLITTING (pm_split_ranges equivalent)
// ============================================================================

/// Context for tracking ranges that should be split via modulo decomposition.
///
/// Based on Tinygrad's pm_split_ranges (simplify.py:60-64).
/// When we see `RANGE % const`, we mark the range for splitting at the SINK.
#[derive(Default)]
pub struct SplitRangesContext {
    /// Maps RANGE ids to their modulo constant for decomposition
    pub marked_ranges: HashMap<u64, i64>,
    /// RANGE ids that should NOT be substituted (e.g., used in ImageDType stores)
    protected_ranges: HashSet<u64>,
}

/// Pattern matcher for range splitting via modulo arithmetic.
///
/// Based on Tinygrad's pm_split_ranges (simplify.py:60-64).
/// This is a context-collecting pass that:
/// 1. Marks RANGE ops used in `RANGE % const` expressions
/// 2. Protects ranges used in ImageDType stores from substitution
/// 3. At SINK, substitutes marked (non-protected) ranges with divmod decomposition
///
/// Example transformation for `RANGE(12) % 4`:
/// - Original: `r = RANGE(12)`
/// - After: `r_div = RANGE(3) * 4`, `r_mod = RANGE(4)`, substitute `r → r_div + r_mod`
///
/// # ImageDType Protection
///
/// Range splitting must NOT apply to ImageDType stores because image addressing
/// uses special 2D coordinates that don't follow standard linear indexing.
/// Applying range splitting to image stores corrupts the addressing scheme.
pub fn pm_split_ranges() -> crate::TypedPatternMatcher<SplitRangesContext> {
    crate::patterns! {
        @context SplitRangesContext;

        // Mark RANGE % const: record the modulo constant for this range
        _modop @ Mod(r @ Range { end, axis_id: _, axis_type: _ }, c @ Const(_))
            if is_divisible_range_end(end, c) => |r, c| {
                mark_range_mod(ctx, r, c);
                None // Don't transform yet, just mark
            },

        // Protect ranges used in ImageDType stores from substitution
        _store @ Store { index: idx @ Index { buffer: buf, indices: _, gate: _ }, value: _, ranges: _ }
            if is_image_dtype(buf) => |idx| {
                protect_ranges_for_image(ctx, idx);
                None // Don't transform, just protect
            },

        // At SINK: perform the substitution
        sink @ Sink { sources: _ } if !ctx.marked_ranges.is_empty() => |sink| {
            do_split_ranges_substitute(ctx, sink)
        },
    }
}

/// Check if a buffer has ImageDType.
fn is_image_dtype(buf: &Arc<UOp>) -> bool {
    matches!(buf.dtype(), DType::Image { .. })
}

/// Protect all ranges reachable from an INDEX used in an ImageDType store.
fn protect_ranges_for_image(ctx: &mut SplitRangesContext, idx: &Arc<UOp>) {
    for node in idx.toposort() {
        if matches!(node.op(), Op::Range { .. }) {
            ctx.protected_ranges.insert(node.id);
            // Also remove from marked_ranges if already marked
            ctx.marked_ranges.remove(&node.id);
        }
    }
}

/// Extract i64 from a Const UOp.
fn const_uop_to_i64(c: &Arc<UOp>) -> Option<i64> {
    match c.op() {
        Op::Const(cv) => match cv.0 {
            ConstValue::Int(v) => Some(v),
            ConstValue::UInt(v) => Some(v as i64),
            _ => None,
        },
        _ => None,
    }
}

/// Check if a RANGE end is divisible by the modulo constant.
fn is_divisible_range_end(end: &Arc<UOp>, c: &Arc<UOp>) -> bool {
    let Some(end_val) = const_uop_to_i64(end) else {
        return false;
    };
    let Some(mod_val) = const_uop_to_i64(c) else {
        return false;
    };
    mod_val > 0 && end_val % mod_val == 0
}

/// Mark a range for modulo decomposition.
fn mark_range_mod(ctx: &mut SplitRangesContext, r: &Arc<UOp>, c: &Arc<UOp>) {
    // Don't mark if already marked or protected (e.g., used in ImageDType store)
    if ctx.marked_ranges.contains_key(&r.id) || ctx.protected_ranges.contains(&r.id) {
        return;
    }
    if let Some(mod_val) = const_uop_to_i64(c) {
        ctx.marked_ranges.insert(r.id, mod_val);
    }
}

/// Perform the substitution at SINK level.
///
/// For each marked RANGE with `end` divisible by `mod_val`:
/// - Create `r_outer = RANGE(end / mod_val)` with same axis type, shifted axis_id
/// - Create `r_inner = RANGE(mod_val)` with same axis type, shifted axis_id
/// - Substitute `r → r_outer * mod_val + r_inner`
fn do_split_ranges_substitute(ctx: &mut SplitRangesContext, sink: &Arc<UOp>) -> Option<Arc<UOp>> {
    use morok_ir::AxisId;
    use morok_ir::rewrite::graph_rewrite_bottom_up;

    if ctx.marked_ranges.is_empty() {
        return None;
    }

    // Build substitution map
    let mut subs: HashMap<u64, Arc<UOp>> = HashMap::new();

    // Collect all UOps to find the marked ranges and max axis_id
    let topo = sink.toposort();

    // Find max axis_id across ALL ranges to avoid collisions when creating split ranges
    let mut max_axis_id: usize = 0;
    for uop in &topo {
        if let Op::Range { axis_id, .. } = uop.op() {
            max_axis_id = max_axis_id.max(axis_id.value());
        }
    }
    let mut next_id = max_axis_id + 1;

    for uop in &topo {
        // Skip protected ranges (e.g., used in ImageDType stores)
        if ctx.protected_ranges.contains(&uop.id) {
            continue;
        }
        if let Some(&mod_val) = ctx.marked_ranges.get(&uop.id)
            && let Op::Range { end, axis_type, .. } = uop.op()
        {
            let Some(end_val) = const_uop_to_i64(end) else {
                continue;
            };

            // Create outer range with unique axis_id
            let outer_end = end_val / mod_val;
            let outer_range = UOp::range_axis(UOp::index_const(outer_end), AxisId::Renumbered(next_id), *axis_type);
            next_id += 1;

            // Create inner range with unique axis_id
            let inner_range = UOp::range_axis(UOp::index_const(mod_val), AxisId::Renumbered(next_id), *axis_type);
            next_id += 1;

            // Substitution: r → outer * mod_val + inner
            let mod_const = UOp::index_const(mod_val);
            let outer_scaled = outer_range.mul(&mod_const);
            let combined = outer_scaled.add(&inner_range);

            subs.insert(uop.id, combined);
        }
    }

    if subs.is_empty() {
        return None;
    }

    // Apply substitutions using the substitute pattern
    let substitute_pm = crate::patterns! {
        r @ Range { end: _, axis_id: _, axis_type: _ } if subs.contains_key(&r.id) => {
            subs.get(&r.id).cloned()
        },
    };

    let result = graph_rewrite_bottom_up(&substitute_pm, sink.clone(), &mut ());

    // Clear the context after substitution
    ctx.marked_ranges.clear();

    Some(result)
}

// ============================================================================
// TRANSFORM HELPERS (movement ops → BUFFERIZE + INDEX)
// ============================================================================

/// Transform a UOp's sources by adding BUFFERIZE + INDEX where needed.
pub fn transform_sources_with_bufferize(x: &Arc<UOp>, ctx: &mut IndexingContext) -> Option<Vec<Arc<UOp>>> {
    if matches!(x.op(), Op::Bufferize { .. } | Op::Index { .. } | Op::After { .. }) {
        return None;
    }

    let sources = x.op().sources();
    if sources.is_empty() {
        return None;
    }

    // Tinygrad: INDEX is only added when `x in ctx.range_map`.
    // For SINK (not in range_map), realized sources still get BUFFERIZE but no INDEX.
    let input_ranges = if let Some((ranges, _)) = ctx.get_ranges(x) { ranges.clone() } else { Vec::new() };

    let mut new_sources = Vec::with_capacity(sources.len());
    let mut any_changed = false;

    for src in sources.iter() {
        let new_src = transform_single_source(x, src, &input_ranges, ctx);
        if !Arc::ptr_eq(&new_src, src) {
            any_changed = true;
        }
        new_sources.push(new_src);
    }

    if any_changed { Some(new_sources) } else { None }
}

/// Flatten multi-range BUFFERIZE to single-range via RESHAPE to 1D.
///
/// Matches Tinygrad's `flatten_bufferize` (rangeify.py:381-389):
/// 1. Reshapes multi-dim ranges to a single flat index via apply_reshape_ranges
/// 2. Creates new BUFFERIZE with single computed range
/// 3. Wraps with RESHAPE back to original shape for downstream movement ops
/// 4. For symbolic range ends, adds SHRINK to symbolic shape
///
/// After this, `bufferize_to_store` only sees single-range BUFFERIZE.
fn flatten_bufferize(bufferize: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Bufferize { compute, ranges, opts } = bufferize.op() else { return None };
    if ranges.len() <= 1 {
        return None;
    }
    // Extract shape from ranges: RANGE(end) → SInt::from(end), CONST(0) → 1
    let shape: Vec<morok_ir::SInt> = ranges
        .iter()
        .map(|r| match r.op() {
            Op::Range { end, .. } => morok_ir::SInt::from(end.clone()),
            _ => morok_ir::SInt::from(1usize),
        })
        .collect();

    // Flatten: apply_reshape_ranges(in_shape=(prod,), out_shape=shape, rngs=ranges)
    let flat_shape = vec![morok_ir::sint_prod(&shape)];
    let ranges_vec: Vec<Arc<UOp>> = ranges.iter().cloned().collect();
    let flat_indices = super::indexing::apply_reshape_ranges(&flat_shape, &shape, &ranges_vec);
    assert_eq!(flat_indices.len(), 1, "flatten_bufferize: expected 1 flat index, got {}", flat_indices.len());
    // New BUFFERIZE with single range
    let flat_buf = UOp::bufferize(compute.clone(), vec![flat_indices[0].clone()], opts.clone());

    // RESHAPE back to original shape (Tinygrad: ret.forced_reshape(x.shape))
    let shape_smallvec: Shape = shape.iter().cloned().collect();
    let reshaped = flat_buf.try_reshape(&shape_smallvec).expect("flatten_bufferize: try_reshape failed");

    // For symbolic range ends, add SHRINK to symbolic shape
    // Tinygrad: if any(r.op is Ops.RANGE and r.src[0].op is not Ops.CONST for r in rngs)
    let has_symbolic =
        ranges.iter().any(|r| matches!(r.op(), Op::Range { end, .. } if !matches!(end.op(), Op::Const(_))));

    if has_symbolic {
        let sym_ranges: Vec<(morok_ir::SInt, morok_ir::SInt)> = ranges
            .iter()
            .map(|r| match r.op() {
                Op::Range { end, .. } => (morok_ir::SInt::from(0usize), morok_ir::SInt::from(end.clone())),
                _ => (morok_ir::SInt::from(0usize), morok_ir::SInt::from(1usize)),
            })
            .collect();
        Some(reshaped.try_shrink(&sym_ranges).expect("flatten_bufferize: try_shrink failed for symbolic ranges"))
    } else {
        Some(reshaped)
    }
}

/// Push movement op through AFTER: `AFTER(MOVEMENT(x), deps) → MOVEMENT(AFTER(x, deps))`.
///
/// Matches Tinygrad's pm_mops rule 2 (rangeify.py:28-29):
///   `UOp(r.op, r.dtype, (a.replace(src=(r.src[0],)+a.src[1:]),)+r.src[1:], r.arg)`
/// Directly reuses the original movement op's parameters (no roundtrip/validation).
pub(crate) fn push_movement_through_after(mop: &Arc<UOp>, deps: &SmallVec<[Arc<UOp>; 4]>) -> Option<Arc<UOp>> {
    let inner_src = &mop.op().sources()[0];
    let new_after = inner_src.after(deps.clone());
    // Re-create the movement op with new_after as source, reusing original parameters.
    // Tinygrad: UOp(r.op, r.dtype, (new_after,)+r.src[1:], r.arg)
    let new_op = match mop.op() {
        Op::Reshape { new_shape, .. } => Op::Reshape { src: new_after, new_shape: new_shape.clone() },
        Op::Permute { axes, .. } => Op::Permute { src: new_after, axes: axes.clone() },
        Op::Expand { new_shape, .. } => Op::Expand { src: new_after, new_shape: new_shape.clone() },
        Op::Pad { begin_pads, end_pads, .. } => {
            Op::Pad { src: new_after, begin_pads: begin_pads.clone(), end_pads: end_pads.clone() }
        }
        Op::Shrink { begins, ends, .. } => Op::Shrink { src: new_after, begins: begins.clone(), ends: ends.clone() },
        Op::Flip { axes, .. } => Op::Flip { src: new_after, axes: axes.clone() },
        _ => return None,
    };
    Some(UOp::new(new_op, mop.dtype()))
}

/// Transform a single source by adding BUFFERIZE + INDEX if needed.
///
/// Non-recursive: only handles immediate buffer-like and realized sources.
/// Movement ops and compute ops are left for the BPM rewrite engine to
/// process individually (matching Tinygrad's `create_bufferize_and_index_based_on_ranges`).
///
/// INDEX nodes are created with a single linear index (matching Tinygrad),
/// computed from the buffer's dimensional ranges and the consumer's index
/// expressions. This eliminates the need for a later linearization pass.
pub(crate) fn transform_single_source(
    consumer: &Arc<UOp>,
    src: &Arc<UOp>,
    input_ranges: &[Arc<UOp>],
    ctx: &mut IndexingContext,
) -> Arc<UOp> {
    // Case 1: Buffer-like op → add multi-index INDEX
    // Unlike Case 2 (BUFFERIZE), we can't linearize here because the buffer's
    // dimensional structure isn't directly available from ctx — the output_ranges
    // may contain PAD validity expressions, not clean RANGE ops.
    // Multi-index INDEX is preserved through the pipeline; codegen linearizes at render time.
    if matches!(
        src.op(),
        Op::Buffer { .. }
            | Op::Param { .. }
            | Op::BufferView { .. }
            | Op::MStack { .. }
            | Op::MSelect { .. }
            | Op::After { .. }
    ) {
        if !input_ranges.is_empty() {
            return UOp::index()
                .buffer(Arc::clone(src))
                .indices(input_ranges.to_vec())
                .call()
                .expect("Failed to create INDEX for buffer source");
        }
        return Arc::clone(src);
    }

    // Case 2: source needs realization → wrap in BUFFERIZE + INDEX
    let realize_axes_opt = ctx.get_realize_axes(src).cloned();
    if let Some(ref realize_axes) = realize_axes_opt {
        let (_, output_ranges) = ctx.get_ranges(src).expect("Realized op must have ranges");

        let closed_ranges: Vec<_> = output_ranges
            .iter()
            .enumerate()
            .filter(|(i, _)| realize_axes.contains(i))
            .map(|(_, r)| Arc::clone(r))
            .collect();

        // Tinygrad (indexing.py:67): removable = x.op is not Ops.COPY and s.op not in ALWAYS_CONTIGUOUS
        let is_copy_consumer = matches!(consumer.op(), Op::Copy { .. });
        let is_always_contiguous_src = super::indexing::is_always_contiguous(src);
        let removable = !is_copy_consumer && !is_always_contiguous_src;
        let addrspace = if output_ranges.len() == realize_axes.len() { AddrSpace::Global } else { AddrSpace::Local };
        tracing::debug!(
            src_id = src.id,
            src_op = src.op().as_ref(),
            consumer_id = consumer.id,
            consumer_op = consumer.op().as_ref(),
            realize_axes = ?realize_axes,
            output_ranges_len = output_ranges.len(),
            addrspace = ?addrspace,
            removable = removable,
            "BUFFERIZE decision"
        );
        // Propagate source device to BUFFERIZE opts (Tinygrad indexing.py:69: device=s.device)
        let device = src.device_spec();
        let opts = BufferizeOpts { device, addrspace, removable };

        // Tinygrad (indexing.py:71): tag=s.tag if GLOBAL, else None
        let buf_tag = if addrspace == AddrSpace::Global { src.tag().clone() } else { None };
        let bufferized = UOp::bufferize(Arc::clone(src), closed_ranges.clone(), opts);
        let bufferized = if let Some(t) = buf_tag { bufferized.with_tag(t) } else { bufferized };

        let index_ranges: Vec<_> = input_ranges
            .iter()
            .enumerate()
            .filter(|(i, _)| realize_axes.contains(i))
            .map(|(_, r)| Arc::clone(r))
            .collect();

        if !index_ranges.is_empty() {
            // Create multi-index INDEX; linearization happens in pm_add_buffers_patterns
            // via linearize_index_on_bufferize (BPM pattern).
            return UOp::index()
                .buffer(bufferized)
                .indices(index_ranges)
                .call()
                .expect("Failed to create INDEX after BUFFERIZE");
        } else {
            return bufferized;
        }
    }

    // Default: no transformation — BPM engine handles movement/compute ops individually
    Arc::clone(src)
}

// ============================================================================
// BUFFERIZE TO STORE CONVERSION
// ============================================================================

// ============================================================================
// HELPER FUNCTIONS FOR BUFFERIZE_TO_STORE
// ============================================================================

/// Apply movement ops chain in reverse order.
/// Walks from chain root to base using pattern matching.
/// Uses existing .base() method at ir/src/uop/core.rs:425-438.
fn apply_movement_ops_chain(result: &Arc<UOp>, chain: &Arc<UOp>) -> Option<Arc<UOp>> {
    let mut mops = Vec::new();
    let mut walk = chain.clone();

    // Walk chain collecting movement ops using pattern matching
    while walk.op().is_movement() {
        mops.push(walk.clone());
        // Extract src via pattern matching
        walk = match walk.op() {
            Op::Reshape { src, .. }
            | Op::Permute { src, .. }
            | Op::Expand { src, .. }
            | Op::Pad { src, .. }
            | Op::Shrink { src, .. }
            | Op::Flip { src, .. } => src.clone(),
            _ => break,
        };
    }

    // Apply in reverse order
    let mut current = result.clone();
    for mop in mops.into_iter().rev() {
        current = apply_single_movement_op(&current, mop.op())?;
    }

    Some(current)
}

/// Apply a single movement operation.
///
/// Note: This extracts shape from the movement op's stored source (which has the
/// target shape after the movement) rather than from UOp shape metadata.
fn apply_single_movement_op(uop: &Arc<UOp>, op: &Op) -> Option<Arc<UOp>> {
    match op {
        Op::Reshape { new_shape, .. } => {
            let shape = extract_shape_from_uop(new_shape)?;
            uop.try_reshape(&shape).ok()
        }
        Op::Permute { axes, .. } => uop.try_permute(axes.clone()).ok(),
        Op::Expand { new_shape, .. } => {
            let shape = extract_shape_from_uop(new_shape)?;
            uop.try_expand(&shape).ok()
        }
        Op::Pad { begin_pads, end_pads, .. } => {
            let begins = extract_shape_from_uop(begin_pads)?;
            let ends = extract_shape_from_uop(end_pads)?;
            let padding: Vec<_> = begins.into_iter().zip(ends).collect();
            uop.try_pad(&padding).ok()
        }
        Op::Shrink { begins, ends, .. } => {
            let begin_shape = extract_shape_from_uop(begins)?;
            let end_shape = extract_shape_from_uop(ends)?;
            let ranges: Vec<_> = begin_shape.into_iter().zip(end_shape).collect();
            uop.try_shrink(&ranges).ok()
        }
        Op::Flip { axes, .. } => uop.try_flip(axes.clone()).ok(),
        _ => None,
    }
}

/// Extract shape from a UOp (for movement op parameters).
/// Handles VECTORIZE, CONST, and VCONST patterns.
fn extract_shape_from_uop(shape_uop: &Arc<UOp>) -> Option<Shape> {
    use morok_ir::SInt;
    match shape_uop.op() {
        // VECTORIZE with Index-typed elements
        Op::Vectorize { elements } => Some(elements.iter().cloned().map(SInt::from).collect()),
        // Single CONST value (for 1D shapes)
        Op::Const(const_hash) => match const_hash.0 {
            ConstValue::Int(v) if v >= 0 => Some(smallvec![SInt::from(v as usize)]),
            ConstValue::UInt(v) => Some(smallvec![SInt::from(v as usize)]),
            _ => None,
        },
        // VConst for multiple concrete dimensions
        Op::VConst { values } => {
            let mut dims = smallvec![];
            for val in values {
                match val {
                    ConstValue::Int(v) if *v >= 0 => dims.push(SInt::from(*v as usize)),
                    ConstValue::UInt(v) => dims.push(SInt::from(*v as usize)),
                    _ => return None,
                }
            }
            Some(dims)
        }
        _ => None,
    }
}

/// Create a LOOP range from an OUTER range with the same axis_id.
fn create_loop_range_from_outer(outer: &Arc<UOp>, size: usize) -> Option<Arc<UOp>> {
    use morok_ir::AxisType;
    let Op::Range { axis_id, .. } = outer.op() else {
        return None;
    };
    Some(UOp::range_axis(UOp::index_const(size as i64), *axis_id, AxisType::Loop))
}

/// Convert ReduceOp to binary operation.
fn reduce_op_to_binary(op: morok_ir::ReduceOp, lhs: &Arc<UOp>, rhs: &Arc<UOp>) -> Option<Arc<UOp>> {
    use morok_ir::types::{BinaryOp, ReduceOp};
    let dtype = lhs.dtype();
    Some(match op {
        ReduceOp::Add => UOp::new(Op::Binary(BinaryOp::Add, lhs.clone(), rhs.clone()), dtype),
        ReduceOp::Mul => UOp::new(Op::Binary(BinaryOp::Mul, lhs.clone(), rhs.clone()), dtype),
        ReduceOp::Max => UOp::new(Op::Binary(BinaryOp::Max, lhs.clone(), rhs.clone()), dtype),
        ReduceOp::Min => {
            // Min uses WHERE(a < b, a, b) pattern
            let cond = UOp::new(Op::Binary(BinaryOp::Lt, lhs.clone(), rhs.clone()), morok_dtype::DType::Bool);
            UOp::try_where(cond, lhs.clone(), rhs.clone()).expect("reduce_op_to_binary: try_where failed for Min")
        }
    })
}

/// Calculate buffer size from RANGE operations.
/// Calculate buffer size from BUFFERIZE ranges.
/// Matches Tinygrad: `size = prod(x.shape)` where `x.shape = [int(r.vmax+1) for r in src[1:]]`.
/// Each range contributes `vmax+1` to the product (RANGE UOps have vmax = end-1, so vmax+1 = end).
/// For flattened BUFFERIZE (single computed expression), vmax+1 gives the total flat size.
fn calculate_size_from_ranges(ranges: &SmallVec<[Arc<UOp>; 4]>) -> usize {
    if ranges.is_empty() {
        return 1;
    }

    ranges
        .iter()
        .map(|r| {
            // Tinygrad: int(r.vmax+1) — works for both RANGE and computed expressions
            let vmax = r.vmax();
            match vmax {
                ConstValue::Int(v) if *v >= 0 => (*v + 1) as usize,
                ConstValue::UInt(v) => (*v + 1) as usize,
                other => panic!(
                    "Cannot allocate buffer: range vmax resolved to {:?}. \
                     Buffers require concrete sizes (Tinygrad: 'no symbolic sized buffers')",
                    other
                ),
            }
        })
        .product()
}

/// Sort ranges by (axis_id, axis_type) for correct row-major linearization.
///
/// Tinygrad reference (rangeify.py:303):
///   rngs = sorted(idx.ranges, key=lambda x: x.arg)
///
/// Tinygrad's RANGE.arg is (axis_id, axis_type), so sorting uses both.
/// This ensures that multi-dimensional ranges are linearized in the correct
/// order regardless of their insertion order in the graph.
fn sort_ranges_by_axis_id(ranges: &SmallVec<[Arc<UOp>; 4]>) -> SmallVec<[Arc<UOp>; 4]> {
    let mut sorted: Vec<_> = ranges.iter().cloned().collect();
    sorted.sort_by_key(|r| {
        if let Op::Range { axis_id, axis_type, .. } = r.op() {
            // Sort by (axis_id, axis_type) to match Tinygrad's sorting by x.arg
            (axis_id.value(), axis_type_ordinal(*axis_type))
        } else {
            (usize::MAX, u8::MAX)
        }
    });
    sorted.into()
}

/// Convert AxisType to ordinal for consistent sorting.
/// Order matches enum definition order in ir/src/types.rs.
fn axis_type_ordinal(at: AxisType) -> u8 {
    match at {
        AxisType::Outer => 0,
        AxisType::Global => 1,
        AxisType::Warp => 2,
        AxisType::Local => 3,
        AxisType::Loop => 4,
        AxisType::GroupReduce => 5,
        AxisType::Reduce => 6,
        AxisType::Upcast => 7,
        AxisType::Unroll => 8,
        AxisType::Thread => 9,
        AxisType::Placeholder => 10,
    }
}

/// Collect RANGE UOps from BUFFERIZE ranges, traversing flattened expressions.
///
/// After `flatten_bufferize`, `ranges[0]` may be a computed expression (Add/Mul of RANGEs)
/// rather than a direct RANGE UOp. This helper traverses all range entries:
/// - Direct RANGE UOps are collected immediately
/// - Non-CONST expressions are traversed via `.ranges()` to find embedded RANGE UOps
/// - CONST entries (collapsed singleton dims) are skipped
/// - Deduplicates by UOp id
fn collect_range_uops(ranges: &SmallVec<[Arc<UOp>; 4]>) -> SmallVec<[Arc<UOp>; 4]> {
    let mut collected = SmallVec::new();
    for r in ranges.iter() {
        if matches!(r.op(), Op::Range { .. }) {
            collected.push(r.clone());
        } else if !matches!(r.op(), Op::Const(_)) {
            for rng in r.ranges().iter() {
                if !collected.iter().any(|c: &Arc<UOp>| c.id == rng.id) {
                    collected.push(rng.clone());
                }
            }
        }
    }
    collected
}

/// Convert BUFFERIZE operation to STORE with buffer allocation and END wrapping.
///
/// # Arguments
///
/// * `bufferize_op` - The BUFFERIZE UOp to convert
/// * `ctx` - Kernel context for tracking buffers and generating IDs
/// * `allow_locals` - If false, treat local address space as global (Tinygrad: pm_add_buffers).
///   If true, create DEFINE_LOCAL for local address space (Tinygrad: pm_add_buffers_local).
pub fn bufferize_to_store(bufferize_op: &Arc<UOp>, ctx: &mut KernelContext, allow_locals: bool) -> Option<Arc<UOp>> {
    let (compute, ranges, opts) = match bufferize_op.op() {
        Op::Bufferize { compute, ranges, opts } => {
            tracing::debug!(
                bufferize_id = bufferize_op.id,
                compute_id = compute.id,
                ranges_len = ranges.len(),
                allow_locals = allow_locals,
                "bufferize_to_store: CONVERTING BUFFERIZE to STORE→AFTER"
            );
            (compute, ranges, opts)
        }
        _ => return None,
    };

    // Calculate size and base dtype upfront (needed for both buffer creation and INDEX dtype)
    let size = calculate_size_from_ranges(ranges);
    let base_dtype = match bufferize_op.dtype() {
        DType::Ptr { base, .. } => (*base).clone(),
        other => other,
    };

    // Calculate sdtype explicitly like Tinygrad (rangeify.py:306):
    //   sdtype = x.dtype.ptr(size=size, addrspace=x.arg.addrspace)
    // This is the pointer type used for STORE targets, ensuring consistent
    // size and addrspace across all INDEX operations in this function.
    let sdtype = base_dtype.clone().ptr(Some(size), opts.addrspace);

    // Get end_ranges for wrapping stores.
    // Tinygrad: `.end(*rngs)` where `rngs = sorted(idx.ranges, ...)`.
    let end_ranges: SmallVec<[Arc<UOp>; 4]> = sort_ranges_by_axis_id(&collect_range_uops(ranges));

    // =========================================================================
    // Case 1: ASSIGN → STORE (reuse existing buffer from ASSIGN target)
    // Tinygrad reference: rangeify.py:307-320
    // =========================================================================
    if let Op::Assign { target, value, movement_ops } = compute.op() {
        // Target must be an INDEX pointing to a buffer
        let Op::Index { buffer, indices, gate } = target.op() else {
            return None;
        };

        // Create store target with explicit sdtype (Tinygrad: assign_target.replace(dtype=sdtype))
        let store_target = UOp::index()
            .buffer(buffer.clone())
            .indices(indices.to_vec())
            .maybe_gate(gate.clone())
            .dtype(sdtype.clone())
            .call()
            .expect("bufferize_to_store: failed to create INDEX for ASSIGN target");

        // Create STORE and wrap with END
        let store = store_target.store_value(value.clone());
        let do_store = if end_ranges.is_empty() { store } else { store.end(end_ranges.clone()) };

        // Apply movement ops in reverse order
        let mut result = buffer.after(smallvec![do_store]);
        if let Some(mops_chain) = movement_ops {
            result = apply_movement_ops_chain(&result, mops_chain)?;
        }

        ctx.map_buffer(bufferize_op.clone(), result.clone());
        return Some(result);
    }

    // =========================================================================
    // Case 2: OUTER REDUCE with zero initialization
    // Tinygrad reference: rangeify.py:323-332
    // =========================================================================
    if let Op::Reduce { src: reduce_src, ranges: reduce_ranges, reduce_op } = compute.op() {
        // OUTER reduce case: exactly ONE range that is OUTER type
        // Tinygrad: len(x.src[0].src) == 2 means src + 1 range
        if reduce_ranges.len() == 1
            && let Op::Range { axis_type, .. } = reduce_ranges[0].op()
            && *axis_type == AxisType::Outer
        {
            // Must be global address space
            if opts.addrspace != AddrSpace::Global {
                return None;
            }

            let outer_range = reduce_ranges[0].clone();
            let device = opts.device.clone().unwrap_or(morok_ir::DeviceSpec::Cpu);

            // Create buffer
            let buf = UOp::new_buffer(device, size, base_dtype.clone());

            // Create zero-init range (same axis_id but AxisType::Loop)
            let zero_range = create_loop_range_from_outer(&outer_range, size)?;

            // Get identity value for reduce op
            use crate::symbolic::dce::reduce_identity;
            let identity = reduce_identity(*reduce_op, base_dtype.clone());

            // Zero-initialize: buf[zero_range] = identity
            let zero_idx = UOp::index()
                .buffer(buf.clone())
                .indices(vec![zero_range.clone()])
                .dtype(sdtype.clone())
                .call()
                .expect("bufferize_to_store: failed to create INDEX for OUTER REDUCE zero-init");
            let zero_store = zero_idx.store_value(identity).end(smallvec![zero_range.clone()]);
            let buf_zeroed = buf.after(smallvec![zero_store]);

            // Use BUFFERIZE's index directly (already flattened by flatten_bufferize).
            // Matches Tinygrad: `bufi = buf.index(idx, dtype=sdtype)` where idx = x.src[1]
            debug_assert!(
                ranges.len() <= 1 || ranges.iter().all(|r| matches!(r.op(), Op::Const(_))),
                "bufferize_to_store: unexpected multi-range in OUTER REDUCE after flatten_bufferize"
            );
            let idx = if ranges.len() == 1 && !matches!(ranges[0].op(), Op::Const(_)) {
                ranges[0].clone()
            } else if !end_ranges.is_empty() {
                sort_ranges_by_axis_id(&end_ranges)[0].clone()
            } else {
                UOp::index_const(0)
            };

            // Collect RANGE UOps from the index expression for END wrapping
            let sorted_end_ranges = sort_ranges_by_axis_id(&collect_range_uops(ranges));

            // Accumulation: buf[idx] = buf[idx] OP reduce_src (Tinygrad: bufi = buf.index(idx, dtype=sdtype))
            let buf_idx = UOp::index()
                .buffer(buf_zeroed.clone())
                .indices(vec![idx])
                .dtype(sdtype.clone())
                .call()
                .expect("bufferize_to_store: failed to create INDEX for OUTER REDUCE accumulation");
            let loaded = UOp::load().buffer(buf_zeroed.clone()).index(buf_idx.clone()).call();
            let accumulated = reduce_op_to_binary(*reduce_op, &loaded, reduce_src)?;

            // Wrap store with both collected end_ranges AND outer_range
            let do_store = buf_idx.store_value(accumulated).end(sorted_end_ranges).end(smallvec![outer_range]);

            let result = buf_zeroed.after(smallvec![do_store]);
            ctx.map_buffer(bufferize_op.clone(), result.clone());
            return Some(result);
        }
    }

    // Determine effective address space based on allow_locals parameter
    // Tinygrad has two matchers:
    // - pm_add_buffers (allow_locals=False): skips local buffers entirely (returns None)
    // - pm_add_buffers_local (allow_locals=True): creates DEFINE_LOCAL for local
    //
    // When allow_locals=false and the buffer is LOCAL, we return None to leave the
    // BUFFERIZE as-is. This matches Tinygrad's behavior where local buffers are only
    // converted during codegen (pm_add_buffers_local), NOT during kernel splitting.
    if !allow_locals && opts.addrspace == AddrSpace::Local {
        return None;
    }
    let effective_addrspace = opts.addrspace;

    let buffer = if let Some(existing_buffer) = ctx.get_buffer(bufferize_op) {
        existing_buffer.clone()
    } else if effective_addrspace == AddrSpace::Global {
        // Create BUFFER node (like Tinygrad's UOp.new_buffer)
        // The BUFFER → PARAM conversion happens later in split_store
        let device = opts.device.clone().unwrap_or(morok_ir::DeviceSpec::Cpu);
        UOp::new_buffer(device, size, base_dtype.clone())
    } else {
        // For local address space (only when allow_locals=true), create DEFINE_LOCAL directly (like Tinygrad)
        let local_ptr_dtype = base_dtype.clone().ptr(Some(size), opts.addrspace);
        let local_id = ctx.next_local();
        UOp::define_local(local_id, local_ptr_dtype)
    };

    // Use ptr=true to keep Ptr dtype for STORE targets (Tinygrad-aligned).
    // This ensures INDEX returns pointer type, which STORE codegen expects.
    // ptr=true is equivalent to setting dtype to buffer.dtype(), but is the
    // idiomatic way per Tinygrad's buf.index(idx, ptr=True).

    // Collect active RANGE UOps from the ranges.
    // Tinygrad: `rngs = sorted(idx.ranges, ...)` — traverses expression tree for RANGE UOps.
    let active_ranges: SmallVec<[Arc<UOp>; 4]> = collect_range_uops(ranges);

    // Sort active ranges by axis_id for correct row-major linearization (Tinygrad: rangeify.py:303)
    let sorted_ranges = sort_ranges_by_axis_id(&active_ranges);

    // Broadcast buffer for STORE-side INDEX only (Tinygrad: buf.broadcast(count).index(idx))
    // The AFTER return uses the unbroadcast buffer so consumers can broadcast it properly.
    let vcount = compute.dtype().vcount();
    let store_buffer = if vcount > 1 { buffer.broadcast(vcount) } else { buffer.clone() };

    let store_target = if !sorted_ranges.is_empty() {
        // After flatten_bufferize, ranges[0] may be the already-linearized flat index.
        // Use it directly. For non-flattened single-range, the RANGE is used directly.
        // Matches Tinygrad: buf.index(idx, dtype=sdtype)
        assert!(
            ranges.len() <= 1 || ranges.iter().all(|r| matches!(r.op(), Op::Const(_))),
            "bufferize_to_store: unexpected multi-range in general path after flatten_bufferize"
        );
        let idx = if ranges.len() == 1 && !matches!(ranges[0].op(), Op::Const(_)) {
            // Single range element (possibly flattened expression or RANGE)
            ranges[0].clone()
        } else {
            // Multiple RANGE UOps (shouldn't happen after flatten, but fallback)
            sorted_ranges[0].clone()
        };
        UOp::index()
            .buffer(store_buffer)
            .indices(vec![idx])
            .dtype(sdtype.clone())
            .call()
            .expect("Failed to create INDEX for BUFFERIZE-to-STORE conversion")
    } else {
        // Scalar store: create INDEX with buffer + index 0 and explicit sdtype
        UOp::index()
            .buffer(store_buffer)
            .indices(vec![UOp::index_const(0)])
            .dtype(sdtype.clone())
            .call()
            .expect("Failed to create INDEX for scalar STORE")
    };

    // Create STORE and wrap with END if there are output ranges.
    // This matches Tinygrad's architecture: .store().end(*rngs)
    //
    // The END wrapper is critical because:
    // 1. split_store looks for END { computation: STORE, ranges } pattern
    // 2. END.ranges define the iteration space for the OUTPUT (not internal computations)
    // 3. For scalar stores (e.g., REDUCE results), no END wrapping (ranges is empty)
    // 4. REDUCE's loop is handled by pm_reduce which creates its own END internally
    // NOTE: STORE takes (index, value) - buffer is accessed via index.buffer
    let store = store_target.store_value(compute.clone());

    // Determine END ranges: use only actual RANGE ops from BUFFERIZE (Tinygrad-aligned).
    //
    // Tinygrad's `rngs = sorted(idx.ranges, ...)` naturally excludes CONST(0) entries
    // because `.ranges` only collects RANGE UOps. END should only wrap with actual
    // iteration ranges, not collapsed singleton dimensions.
    let end_ranges: SmallVec<[Arc<UOp>; 4]> = sorted_ranges.clone();

    let mut do_store = if !end_ranges.is_empty() { store.end(end_ranges) } else { store };

    if opts.addrspace == AddrSpace::Local {
        do_store = do_store.barrier(SmallVec::new());
    }

    let result = buffer.after(SmallVec::from_elem(do_store, 1));
    ctx.map_buffer(bufferize_op.clone(), result.clone());

    Some(result)
}

// ============================================================================
// REDUCTION SIMPLIFICATIONS
// ============================================================================

/// Partition ranges into parented and unparented.
#[allow(clippy::mutable_key_type)]
pub(crate) fn partition_reduce_ranges(
    ranges: &SmallVec<[Arc<UOp>; 4]>,
    src_ranges: &HashSet<UOpKey>,
) -> (SmallVec<[Arc<UOp>; 4]>, Vec<Arc<UOp>>) {
    let mut parented = SmallVec::new();
    let mut unparented = Vec::new();

    for range in ranges {
        let key = UOpKey(Arc::clone(range));
        if src_ranges.contains(&key) {
            parented.push(Arc::clone(range));
        } else {
            unparented.push(Arc::clone(range));
        }
    }

    (parented, unparented)
}

pub(crate) fn get_range_size(range: &Arc<UOp>) -> Option<Arc<UOp>> {
    if let Op::Range { end, .. } = range.op() { Some(Arc::clone(end)) } else { None }
}

/// Collapse REDUCE(ADD) by algebraic simplification following Tinygrad's algorithm.
///
/// Core reduce collapse algorithm — parameterized by pattern matcher.
///
/// For each reduce range:
/// 1. Gated toposort to find nodes "in scope" of the range
/// 2. Replace external inputs (nodes NOT in scope) with synthetic DEFINE_VAR
/// 3. Wrap substituted body in a synthetic REDUCE
/// 4. Run algebraic patterns (bound-from-below/above, distributive, etc.)
/// 5. If REDUCE is eliminated (no_range), reverse-substitute back
///
/// Based on Tinygrad's `reduce_collapse` (simplify.py:121-134).
/// Parameterized by `pm` following Tinygrad's `def reduce_collapse(red, u, pm=pm_reduce_collapse)`.
#[allow(clippy::mutable_key_type)]
fn reduce_collapse_with(src: &Arc<UOp>, ranges: &[Arc<UOp>], pm: &crate::TypedPatternMatcher<()>) -> Option<Arc<UOp>> {
    use morok_ir::ReduceOp;

    if ranges.is_empty() {
        return None;
    }

    let mut u = Arc::clone(src);

    for range in ranges {
        // 1. Gated toposort: find nodes "in scope" of this range
        let range_key = UOpKey(range.clone());
        let in_scope: HashSet<UOpKey> =
            u.toposort_filtered(|node| node.in_scope_ranges().contains(&range_key)).into_iter().map(UOpKey).collect();

        // Bail if nested REDUCE or STORE in scope (can't collapse through these)
        if in_scope.iter().any(|k| matches!(k.0.op(), Op::Reduce { .. } | Op::Store { .. })) {
            return None;
        }

        // 2. Identify external inputs and substitute with DEFINE_VAR
        // (Tinygrad excludes: CONST, VCONST, PARAM, DEFINE_LOCAL, DEFINE_VAR)
        let mut replaces: HashMap<UOpKey, Arc<UOp>> = HashMap::new();
        for node in &in_scope {
            node.0.op().map_child(|child| {
                let key = UOpKey(child.clone());
                if in_scope.contains(&key) || replaces.contains_key(&key) {
                    return;
                }
                if matches!(
                    child.op(),
                    Op::Const(_)
                        | Op::VConst { .. }
                        | Op::DefineVar { .. }
                        | Op::Param { device: None, .. }
                        | Op::DefineLocal { .. }
                ) {
                    return;
                }
                let vmin = match child.vmin() {
                    ConstValue::Int(i) => *i,
                    ConstValue::UInt(u) => *u as i64,
                    ConstValue::Float(f) => *f as i64,
                    ConstValue::Bool(b) => *b as i64,
                };
                let vmax = match child.vmax() {
                    ConstValue::Int(i) => *i,
                    ConstValue::UInt(u) => *u as i64,
                    ConstValue::Float(f) => *f as i64,
                    ConstValue::Bool(b) => *b as i64,
                };
                let var = UOp::define_var(format!("in{}", replaces.len()), vmin, vmax).with_dtype(child.dtype());
                replaces.insert(key, var);
            });
        }

        // 3. Build synthetic REDUCE: substituted_body.reduce([range], ADD)
        let substituted = u.substitute(&replaces);
        let synthetic_reduce = substituted.reduce(smallvec![range.clone()], ReduceOp::Add);

        // 4. Apply algebraic patterns to try eliminating the range
        let result = crate::rewrite::graph_rewrite(pm, synthetic_reduce, &mut ());

        // 5. Check range eliminated (use plain toposort, NOT in_scope_ranges,
        //    since REDUCE "ends" ranges and would give a false positive)
        let has_range = result.toposort().iter().any(|x| matches!(x.op(), Op::Range { .. }));
        if has_range {
            return None;
        }

        // 6. Reverse substitute: DEFINE_VAR → original external inputs
        let reverse: HashMap<UOpKey, Arc<UOp>> = replaces.into_iter().map(|(k, v)| (UOpKey(v), k.0)).collect();
        u = result.substitute(&reverse);
    }

    Some(u)
}

/// Collapse REDUCE using `pm_reduce_collapse` patterns.
///
/// Tinygrad: `reduce_collapse(red, u)` (uses default `pm=pm_reduce_collapse`).
pub fn reduce_collapse(src: &Arc<UOp>, ranges: &[Arc<UOp>]) -> Option<Arc<UOp>> {
    reduce_collapse_with(src, ranges, super::patterns::build_reduce_collapse_matcher())
}

/// Collapse REDUCE using extended `pm_reduce_load_collapse` patterns.
///
/// Tinygrad: `reduce_load_collapse(red, u)` — calls `reduce_collapse` with
/// `pm=pm_reduce_load_collapse` which includes `.or_casted()` variants,
/// NE lifting, and the full `pm_load_collapse` non-REDUCE patterns.
pub fn reduce_load_collapse(src: &Arc<UOp>, ranges: &[Arc<UOp>]) -> Option<Arc<UOp>> {
    reduce_collapse_with(src, ranges, super::patterns::build_reduce_load_collapse_matcher())
}

pub(crate) fn cast_to_dtype(value: &Arc<UOp>, target_dtype: &morok_dtype::DType) -> Option<Arc<UOp>> {
    use morok_dtype::DType;

    let scalar_type = match target_dtype {
        DType::Scalar(s) => DType::Scalar(*s),
        DType::Vector { scalar, .. } => DType::Scalar(*scalar),
        _ => return None,
    };

    let casted = value.cast(scalar_type);

    if target_dtype.is_vector() {
        let count = target_dtype.count();
        let elements: SmallVec<[Arc<UOp>; 4]> = (0..count).map(|_| casted.clone()).collect();
        Some(UOp::vectorize(elements))
    } else {
        Some(casted)
    }
}

// ============================================================================
// RANGE SIMPLIFICATION
// ============================================================================

/// Simplify ranges by merging adjacent ranges to reduce divmod operations.
///
/// Based on Tinygrad's `pm_simplify_ranges` (simplify.py:20-37).
///
/// This optimization merges adjacent ranges when the merge reduces the number of
/// IDIV and MOD operations in the computation graph. The merged range is then
/// decomposed back using divmod to preserve correctness.
///
/// Key validation from Tinygrad:
/// - Both ranges must appear in the same REDUCE operations (consistent scoping)
/// - Both ranges must have the same axis type
///
/// Example:
/// - Original: Two ranges R1(16) and R2(8)
/// - Merge: Create R_merged(128), decompose as R1 = merged // 8 and R2 = merged % 8
/// - Accept: Only if this reduces or maintains the divmod count
pub fn simplify_merge_adjacent(u: &Arc<UOp>) -> Option<Arc<UOp>> {
    use crate::passes::linearize_index::count_divmod;

    // Get ended ranges for this operation
    let ended_ranges = match u.op() {
        Op::End { computation: _, ranges } => ranges.clone(),
        Op::Reduce { ranges, .. } => ranges.clone(),
        _ => return None,
    };

    if ended_ranges.len() < 2 {
        return None;
    }

    // Collect all REDUCE operations in the backward slice (Tinygrad simplify.py:21)
    let reduce_ranges: Vec<SmallVec<[Arc<UOp>; 4]>> = u
        .toposort()
        .iter()
        .filter_map(|dep| match dep.op() {
            Op::Reduce { ranges, .. } => Some(ranges.clone()),
            _ => None,
        })
        .collect();

    // Cumulative merging (Tinygrad simplify.py:37: `u = nidx` inside loop)
    // Try all pairs and accumulate successful merges into `current`.
    let mut current = Arc::clone(u);
    let mut changed = false;

    // Re-extract ranges from current for each iteration
    let pairs: Vec<(usize, usize)> = if matches!(u.op(), Op::End { .. }) {
        (0..ended_ranges.len() - 1).map(|i| (i, i + 1)).collect()
    } else {
        let mut perms = Vec::new();
        for i in 0..ended_ranges.len() {
            for j in 0..ended_ranges.len() {
                if i != j {
                    perms.push((i, j));
                }
            }
        }
        perms
    };

    for (i0, i1) in pairs {
        let r0 = &ended_ranges[i0];
        let r1 = &ended_ranges[i1];

        let (r0_axis_type, r0_end) = match r0.op() {
            Op::Range { end, axis_type, .. } => (axis_type, end),
            _ => continue,
        };
        let (r1_axis_type, r1_end) = match r1.op() {
            Op::Range { end, axis_type, .. } => (axis_type, end),
            _ => continue,
        };

        if r0_axis_type != r1_axis_type {
            continue;
        }

        // Check same REDUCE scope (Tinygrad simplify.py:25-27)
        let valid_reduce_scope = reduce_ranges.iter().all(|rngs| {
            let r0_in = rngs.iter().any(|rng| Arc::ptr_eq(rng, r0));
            let r1_in = rngs.iter().any(|rng| Arc::ptr_eq(rng, r1));
            r0_in == r1_in
        });
        if !valid_reduce_scope {
            continue;
        }

        if let Some(v) = const_uop_to_i64(r0_end)
            && v <= 0
        {
            continue;
        }
        if let Some(v) = const_uop_to_i64(r1_end)
            && v <= 0
        {
            continue;
        }
        if let (Some(s0), Some(s1)) = (const_uop_to_i64(r0_end), const_uop_to_i64(r1_end))
            && s0.checked_mul(s1).is_none()
        {
            continue;
        }

        let merged_size_uop = r0_end.mul(r1_end);
        let merged_range = r0.with_sources(vec![merged_size_uop]);

        let new_r0 = merged_range.idiv(r1_end);
        let new_r1 = merged_range.mod_(r1_end);

        #[allow(clippy::mutable_key_type)]
        let mut subs: HashMap<UOpKey, Arc<UOp>> = HashMap::new();
        subs.insert(UOpKey(r0.clone()), new_r0);
        subs.insert(UOpKey(r1.clone()), new_r1);

        // Apply substitution and simplify (Tinygrad simplify.py:30-31)
        let rewritten = current.substitute(&subs);
        static MERGE_SYM: std::sync::LazyLock<crate::TypedPatternMatcher> =
            std::sync::LazyLock::new(|| crate::symbolic::symbolic().clone() + pm_flatten_range().clone());
        let simplified = crate::rewrite::graph_rewrite(&*MERGE_SYM, rewritten, &mut ());

        // Accept if divmod count is reduced or equal (Tinygrad simplify.py:34-36)
        let original_divmod = count_divmod(&current);
        let new_divmod = count_divmod(&simplified);

        if new_divmod <= original_divmod {
            current = simplified;
            changed = true;
        }
    }

    if changed { Some(current) } else { None }
}

/// Pattern matcher for range simplification.
///
/// Tries to merge adjacent ranges to reduce divmod operations.
pub fn pm_simplify_ranges() -> &'static crate::TypedPatternMatcher {
    crate::cached_patterns! {
        // Match END ops with ranges
        u @ End { computation: _, ranges } if !ranges.is_empty() => |u| simplify_merge_adjacent(u),
        // Match REDUCE ops with ranges
        u @ Reduce { src: _, ranges, reduce_op: _ } if !ranges.is_empty() => |u| simplify_merge_adjacent(u),
    }
}

// ============================================================================
// RANGE FLATTENING
// ============================================================================

/// Flatten nested RANGE operations into canonical form.
pub fn flatten_range_impl(r: &Arc<UOp>) -> Option<Arc<UOp>> {
    let off = match r.op() {
        Op::Reduce { .. } => 1,
        Op::Store { .. } => 2, // (index, value, ranges...) - ranges start at index 2
        Op::End { .. } => 1,
        _ => return None,
    };

    let original_sources = r.op().sources();
    let original_ranges: Vec<&Arc<UOp>> = original_sources.iter().skip(off).collect();
    let mut all_range_sources: Vec<Arc<UOp>> = original_ranges.iter().map(|r| (*r).clone()).collect();

    let innermost_computation = if matches!(r.op(), Op::End { .. }) {
        let mut computation = Arc::clone(&original_sources[0]);

        while matches!(computation.op(), Op::End { .. }) {
            all_range_sources.extend(computation.op().sources().iter().skip(1).cloned());
            computation = Arc::clone(&computation.op().sources()[0]);
        }

        Some(computation)
    } else {
        None
    };

    if all_range_sources.is_empty() {
        return None;
    }

    let sink = UOp::sink(all_range_sources);
    let new_ranges: Vec<Arc<UOp>> =
        sink.toposort().into_iter().filter(|uop| matches!(uop.op(), Op::Range { .. })).collect();

    if new_ranges.is_empty() {
        return None;
    }

    // Check if anything actually changed
    if new_ranges.len() == original_ranges.len()
        && innermost_computation.as_ref().is_none_or(|c| Arc::ptr_eq(c, &original_sources[0]))
        && new_ranges.iter().zip(original_ranges.iter()).all(|(a, b)| Arc::ptr_eq(a, *b))
    {
        return None; // No change, avoid infinite loop
    }

    let mut new_sources: Vec<Arc<UOp>> =
        if let Some(inner_comp) = innermost_computation { vec![inner_comp] } else { original_sources[..off].to_vec() };
    new_sources.extend(new_ranges);

    Some(r.with_sources(new_sources))
}

/// Apply range flattening to a computation graph.
#[allow(clippy::mutable_key_type)]
pub fn flatten_ranges(root: &Arc<UOp>) -> Arc<UOp> {
    let mut replacements: HashMap<UOpKey, Arc<UOp>> = HashMap::new();

    for node in root.toposort() {
        if let Some(flattened) = flatten_range_impl(&node) {
            replacements.insert(UOpKey(node.clone()), flattened);
        }
    }

    root.substitute(&replacements)
}

// ============================================================================
// CYCLE DETECTION
// ============================================================================

/// Buffer access types for cycle detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpAccessType {
    Load,
    Store,
}

/// Unwrap buffer-like ops to get the underlying buffer.
pub fn as_buf(uop: &Arc<UOp>) -> Arc<UOp> {
    match uop.op() {
        Op::MSelect { buffer, .. } => buffer.clone(),
        Op::MStack { buffers } if !buffers.is_empty() => buffers[0].clone(),
        Op::After { passthrough, .. } => passthrough.clone(),
        _ => uop.clone(),
    }
}

/// Detect conflicting buffer accesses. Panics if same buffer has both LOAD and STORE.
#[allow(clippy::mutable_key_type)]
pub fn find_bufs(store: &Arc<UOp>) -> HashMap<UOpKey, OpAccessType> {
    let mut ret: HashMap<UOpKey, OpAccessType> = HashMap::new();

    let nodes = store.toposort_filtered(|uop| !matches!(uop.op(), Op::After { .. }));

    for node in nodes {
        if let Op::Load { buffer, .. } = node.op() {
            let buf = as_buf(buffer);
            let buf_key = UOpKey(buf.clone());

            if let Some(&existing_access) = ret.get(&buf_key)
                && existing_access != OpAccessType::Load
            {
                panic!(
                    "buffer accessed with conflicting ops: {:?} (existing: {:?}, new: {:?})",
                    buf,
                    existing_access,
                    OpAccessType::Load
                );
            }

            ret.insert(buf_key, OpAccessType::Load);
        }

        if let Some(buffer) = node.store_buffer() {
            let buf = as_buf(buffer);
            let buf_key = UOpKey(buf.clone());

            if let Some(&existing_access) = ret.get(&buf_key)
                && existing_access != OpAccessType::Store
            {
                panic!(
                    "buffer accessed with conflicting ops: {:?} (existing: {:?}, new: {:?})",
                    buf,
                    existing_access,
                    OpAccessType::Store
                );
            }

            ret.insert(buf_key, OpAccessType::Store);
        }
    }

    ret
}

// ============================================================================
// PM_ADD_BUFFERS PATTERNS
// ============================================================================

/// Convert DISK BUFFERIZE(BITCAST|CONTIGUOUS) → BUFFER_VIEW (Tinygrad rangeify.py:285-304).
/// For DISK devices, instead of creating a compute kernel, creates a zero-copy typed view
/// with byte offset into the memory-mapped file.
fn late_buffer_view(compute: &Arc<UOp>, bufferize: &Arc<UOp>) -> Option<Arc<UOp>> {
    use morok_ir::uop::cached_property::CachedProperty;
    use morok_ir::uop::properties::VminVmaxProperty;

    let Op::Bufferize { opts, ranges, .. } = bufferize.op() else { return None };

    // Only for DISK device
    if !matches!(&opts.device, Some(d) if d.is_disk()) {
        return None;
    }

    // Compute size from ranges (product of range ends)
    let size: usize = ranges
        .iter()
        .map(|r| {
            if let Op::Range { end, .. } = r.op()
                && let (_, morok_ir::ConstValue::Int(v)) = VminVmaxProperty::get(end)
            {
                return *v as usize;
            }
            if let Op::Const(_) = r.op() {
                return 1; // const 0 index contributes dim of 1
            }
            1
        })
        .product();

    // Walk up from compute to find the INDEX node (Tinygrad rangeify.py:291-295)
    // In Tinygrad, `t` is the BITCAST/CONTIGUOUS itself. We need to look INTO its children
    // for an INDEX, not walk UP past it. The BITCAST's source (after rangeify) should be
    // an INDEX or contain one.
    let mut x = compute.clone();
    loop {
        // Check if any SOURCE of x is an INDEX
        if x.op().sources().iter().any(|s| matches!(s.op(), Op::Index { .. })) {
            break;
        }
        // For BITCAST/CONTIGUOUS (the starting node), look into their source
        if matches!(x.op(), Op::BitCast { .. } | Op::Contiguous { .. }) {
            x = x.op().sources().first()?.clone();
            continue;
        }
        // Don't cross other elementwise ops
        if matches!(x.op(), Op::Unary(..) | Op::Binary(..) | Op::Ternary(..) | Op::Cast { .. }) {
            return None;
        }
        x = x.op().sources().first()?.clone();
    }
    let index = x.op().sources().iter().find(|s| matches!(s.op(), Op::Index { .. }))?.clone();

    // Compute byte offset (Tinygrad rangeify.py:297-298)
    let offset: usize = if let Op::Index { indices, .. } = index.op() {
        if indices.is_empty() {
            // Scalar: offset from first index's constant arg (Tinygrad: x.src[1].arg)
            0
        } else {
            // Shaped: sum of index vmin values
            let mut total: i64 = 0;
            for idx in indices.iter() {
                let (vmin, _) = VminVmaxProperty::get(idx);
                if let morok_ir::ConstValue::Int(v) = vmin {
                    total += v;
                }
            }
            total.max(0) as usize
        }
    } else {
        0
    };

    // Get base buffer (the DISK BUFFER UOp)
    let base = index.base();

    // Create BUFFER_VIEW with compute's dtype
    let buffer_view = UOp::new(Op::BufferView { buffer: base, size, offset }, compute.dtype());

    // Replace BUFFERIZE's first source with the BUFFER_VIEW, keep the range source
    let new_sources: Vec<Arc<UOp>> = std::iter::once(buffer_view).chain(ranges.iter().cloned()).collect();
    Some(UOp::bufferize(new_sources[0].clone(), new_sources[1..].to_vec(), opts.clone()))
}

/// Create pattern matcher for adding buffers (BUFFERIZE → STORE conversion).
///
/// Based on Tinygrad's pm_add_buffers (rangeify.py:358-367) with `allow_locals=False`.
/// Uses a shared KernelContext (like Tinygrad's `ctx=itertools.count(lunique_start)`)
/// to ensure unique buffer IDs across all pattern matches.
pub fn pm_add_buffers_patterns() -> crate::TypedPatternMatcher<super::kernel::KernelContext> {
    crate::patterns! {
        @context super::kernel::KernelContext;
        // Flatten multi-range BUFFERIZE to 1D (Tinygrad: flatten_bufferize, rangeify.py:381-389)
        buf @ Bufferize { compute: _ } if matches!(buf.op(), Op::Bufferize { ranges, .. } if ranges.len() > 1)
            => |buf, _ctx| { flatten_bufferize(buf) },
        // pm_mops rule 1: push movement ops through INDEX (Tinygrad rangeify.py:25-26)
        Index { buffer: mop, indices, gate } if mop.op().is_movement()
            => |mop, indices, gate, _ctx| {
                super::patterns::transform_movement_through_index(mop, indices, gate)
            },
        // pm_mops rule 2: push movement ops through AFTER (Tinygrad rangeify.py:28-29)
        // AFTER(MOVEMENT(x, ...), deps) → MOVEMENT(AFTER(x, deps), ...)
        After { passthrough: mop, deps } if mop.op().is_movement()
            => |mop, deps, _ctx| {
                push_movement_through_after(mop, deps)
            },
        // pm_mops rule 3: strip movement ops from END (Tinygrad rangeify.py:30)
        // END(MOVEMENT(x, ...), ranges) → END(x, ranges)
        End { computation: mop, ranges } if mop.op().is_movement()
            => |mop, ranges, _ctx| {
                let src = &mop.op().sources()[0];
                Some(src.end(ranges.clone()))
            },
        // to_bufferview: DISK BUFFERIZE(BITCAST|CONTIGUOUS) → BUFFER_VIEW (Tinygrad rangeify.py:302-304)
        buf @ Bufferize { compute }
            if matches!(compute.op(), Op::BitCast { .. } | Op::Contiguous { .. })
            => |buf, compute, _ctx| late_buffer_view(compute, buf),
        // BUFFERIZE → STORE conversion (allow_locals=false: treat local as global)
        buf @ Bufferize { compute: _ } => |buf, ctx| {
            bufferize_to_store(buf, ctx, false)
        },
    }
}

/// Create pattern matcher for adding buffers with local buffer support.
///
/// Based on Tinygrad's pm_add_buffers_local (rangeify.py:358-367) with `allow_locals=True`.
/// Uses a shared KernelContext for unique buffer IDs.
pub fn pm_add_buffers_local_patterns() -> crate::TypedPatternMatcher<super::kernel::KernelContext> {
    crate::patterns! {
        @context super::kernel::KernelContext;
        // Flatten multi-range BUFFERIZE to 1D (Tinygrad: flatten_bufferize, rangeify.py:381-389)
        buf @ Bufferize { compute: _ } if matches!(buf.op(), Op::Bufferize { ranges, .. } if ranges.len() > 1)
            => |buf, _ctx| { flatten_bufferize(buf) },
        // pm_mops rule 1: push movement ops through INDEX (Tinygrad rangeify.py:25-26)
        Index { buffer: mop, indices, gate } if mop.op().is_movement()
            => |mop, indices, gate, _ctx| {
                super::patterns::transform_movement_through_index(mop, indices, gate)
            },
        // pm_mops rule 2: push movement ops through AFTER (Tinygrad rangeify.py:28-29)
        After { passthrough: mop, deps } if mop.op().is_movement()
            => |mop, deps, _ctx| {
                push_movement_through_after(mop, deps)
            },
        // pm_mops rule 3: strip movement ops from END (Tinygrad rangeify.py:30)
        End { computation: mop, ranges } if mop.op().is_movement()
            => |mop, ranges, _ctx| {
                let src = &mop.op().sources()[0];
                Some(src.end(ranges.clone()))
            },
        // to_bufferview: DISK BUFFERIZE(BITCAST|CONTIGUOUS) → BUFFER_VIEW (Tinygrad rangeify.py:302-304)
        buf @ Bufferize { compute }
            if matches!(compute.op(), Op::BitCast { .. } | Op::Contiguous { .. })
            => |buf, compute, _ctx| late_buffer_view(compute, buf),
        // BUFFERIZE → STORE conversion (allow_locals=true: create DEFINE_LOCAL for local addrspace)
        buf @ Bufferize { compute: _ } => |buf, ctx| {
            bufferize_to_store(buf, ctx, true)
        },
    }
}
