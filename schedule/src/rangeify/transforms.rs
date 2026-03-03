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

use morok_ir::shape::Shape;
use morok_ir::{AddrSpace, AxisType, BufferizeOpts, ConstValue, DType, Op, UOp, UOpKey};
use smallvec::{SmallVec, smallvec};
use tracing::trace;

use super::context::RangeifyContext;
use super::indexing::{IndexingContext, range_size_as_i64};
use super::kernel::KernelContext;
use crate::passes::linearize_index::{build_linear_index, compute_row_major_strides};

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

/// Result of rangeify transformation including the substitution map.
pub struct RangeifyResult {
    /// The transformed sink node
    pub sink: Arc<UOp>,
    /// Context with range information
    pub context: RangeifyContext,
    /// Maps original UOps to their transformed versions (for global substitution)
    pub becomes_map: HashMap<UOpKey, Arc<UOp>>,
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
/// **Stage 6**: pm_split_store - Split store ranges (CPU only)
/// **Stage 7**: apply_opts - Post-range optimization (happens in optimizer)
#[allow(clippy::mutable_key_type)]
#[tracing::instrument(skip_all)]
pub fn rangeify_with_map(
    sink: Arc<UOp>,
    pcontig_config: Option<&super::kernel::PcontigConfig>,
) -> morok_ir::Result<RangeifyResult> {
    use morok_ir::rewrite::{graph_rewrite_bottom_up_with_map, graph_rewrite_with_map};

    // Aggregate all becomes_maps from rewrite passes
    let mut all_becomes: HashMap<UOpKey, Arc<UOp>> = HashMap::new();

    // Combined early pass (Tinygrad: earliest_rewrites + replace_contiguous, ctx={})
    // MUST run BEFORE range assignment so rangeify sees a cleaned graph.
    // Uses top-down traversal matching Tinygrad's bottom_up=False default.
    let t_stage = std::time::Instant::now();
    let early_combined = super::patterns::early_rewrites().with_context::<super::patterns::ReplaceContiguousCtx>()
        + super::patterns::replace_contiguous();
    let mut contig_ctx = super::patterns::ReplaceContiguousCtx::new();
    let result = graph_rewrite_with_map(&early_combined, sink, &mut contig_ctx);
    let mut sink = result.root;
    all_becomes.extend(result.becomes_map);
    tracing::debug!(
        uop.tree = sink.tree(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "early rewrites + replace contiguous complete"
    );

    // =========================================================================
    // Stage 0: Range assignment to build IndexingContext
    // =========================================================================
    let t_stage = std::time::Instant::now();
    let (rangeified, mut indexing_ctx) = super::indexing::run_rangeify(sink)?;
    sink = rangeified;
    tracing::debug!(
        uop.tree = sink.tree(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "Stage 0: range assignment complete"
    );

    // Split large reductions BEFORE ReduceAxis → REDUCE conversion
    let t_stage = std::time::Instant::now();
    let mut split_config = super::kernel::SplitReduceOpConfig::default();
    let split_matcher = super::patterns::split_reduceop_patterns();
    let result = graph_rewrite_with_map(&split_matcher, sink, &mut split_config);
    sink = result.root;
    all_becomes.extend(result.becomes_map);
    tracing::debug!(
        uop.tree = sink.tree(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "split reduceops complete"
    );

    // =========================================================================
    // Stage 1: Early movement ops (BOTTOM_UP) - Tinygrad: pm_mops + pm_syntactic_sugar
    // MUST RUN FIRST before other optimizations
    // =========================================================================
    let t_stage = std::time::Instant::now();
    let rangeify_matcher = super::patterns::apply_rangeify_patterns();
    let result = graph_rewrite_bottom_up_with_map(&rangeify_matcher, sink, &mut indexing_ctx);
    sink = result.root;
    all_becomes.extend(result.becomes_map);
    tracing::debug!(
        uop.tree = sink.tree(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "Stage 1: rangeify + movement ops complete"
    );

    // =========================================================================
    // Mega-pass: symbolic + reduce_simplify + buffer_folding + buffer_removal
    // (Tinygrad: symbolic + pm_reduce_simplify + pm_const_buffer_folding + pm_remove_bufferize)
    //
    // One fixpoint pass combining all simplification + buffer removal.
    // Uses PcontigConfig as the shared context (buffer_removal needs it;
    // other patterns are lifted via with_context()).
    // =========================================================================
    {
        use super::kernel::PcontigConfig;
        let t_stage = std::time::Instant::now();
        let mega_pass = crate::symbolic::symbolic().with_context::<PcontigConfig>()
            + super::patterns::reduction_simplify_patterns().with_context()
            + super::patterns::absorb_invalid_into_index_gate().with_context()
            + super::patterns::buffer_folding().with_context()
            + super::patterns::dead_axis_removal().with_context()
            + super::patterns::movement_op_patterns().with_context()
            + super::patterns::early_rewrites().with_context()
            + super::patterns::buffer_removal_with_pcontig();
        tracing::debug!(
            total_patterns = mega_pass.len(),
            wildcard_count = mega_pass.wildcard_count(),
            indexed_buckets = mega_pass.indexed_count(),
            "mega-pass pattern stats"
        );
        let mut pcontig = pcontig_config.cloned().unwrap_or_default();
        sink = apply_buffer_removal_protecting_sink(&sink, &mega_pass, &mut pcontig);
        tracing::debug!(
            node_count = sink.toposort().len(),
            elapsed_ms = t_stage.elapsed().as_millis() as u64,
            "mega-pass complete"
        );
    }

    // Stages 2a-6 (load_collapse, split_ranges, symbolic+flatten, simplify_ranges,
    // split_store) now run per-kernel in optimizer::apply_pre_optimization().

    // Buffer limit enforcement
    if let Some(device) = super::patterns::extract_device_from_graph(&sink)
        && let Some(limit) = device.max_buffers()
    {
        let t_stage = std::time::Instant::now();
        let limit_matcher = super::patterns::buffer_limit_patterns(limit);
        let result = graph_rewrite_with_map(&limit_matcher, sink, &mut ());
        sink = result.root;
        all_becomes.extend(result.becomes_map);
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

    Ok(RangeifyResult { sink, context: rangeify_ctx, becomes_map: all_becomes })
}

/// Pattern matcher for range flattening.
///
/// Based on Tinygrad's pm_flatten_range (simplify.py:14-17).
/// Extracts all RANGE operations from nested END/REDUCE/STORE structures.
pub fn pm_flatten_range() -> crate::TypedPatternMatcher {
    crate::patterns! {
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

    // Collect all UOps to find the marked ranges
    let topo = sink.toposort();
    for uop in &topo {
        // Skip protected ranges (e.g., used in ImageDType stores)
        if ctx.protected_ranges.contains(&uop.id) {
            continue;
        }
        if let Some(&mod_val) = ctx.marked_ranges.get(&uop.id)
            && let Op::Range { end, axis_id, axis_type, .. } = uop.op()
        {
            let Some(end_val) = const_uop_to_i64(end) else {
                continue;
            };

            // Create outer range: RANGE(end / mod_val, axis_id shifted by 0, same type)
            let outer_end = end_val / mod_val;
            let outer_range = UOp::range_axis(
                UOp::index_const(outer_end),
                AxisId::Renumbered(axis_id.value() * 2), // Shift to avoid collision
                *axis_type,
            );

            // Create inner range: RANGE(mod_val, axis_id shifted by 1, same type)
            let inner_range =
                UOp::range_axis(UOp::index_const(mod_val), AxisId::Renumbered(axis_id.value() * 2 + 1), *axis_type);

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
// SPLIT STORE (pm_split_store equivalent) - CPU only
// ============================================================================

/// Context for split store - contains target device.
#[derive(Debug, Clone, Default)]
pub struct SplitStoreContext {
    /// Target device (split store only applies to CPU)
    pub device: Option<morok_device::DeviceSpec>,
    /// Range counter for creating unique axis IDs
    range_counter: usize,
}

impl SplitStoreContext {
    /// Create context for the given device.
    pub fn for_device(device: morok_device::DeviceSpec) -> Self {
        Self { device: Some(device), range_counter: 0 }
    }

    /// Generate a unique axis ID for split ranges.
    fn next_axis_id(&mut self) -> morok_ir::AxisId {
        let id = self.range_counter;
        self.range_counter += 1;
        morok_ir::AxisId::Renumbered(1000 + id) // Use high offset to avoid collision
    }
}

/// Split a STORE at comparison cut points (CPU-only optimization).
///
/// When a RANGE has CMPLT comparisons to constants, split the store
/// into multiple stores with disjoint ranges to enable branch elimination.
///
/// Based on Tinygrad's pm_split_store (simplify.py).
#[allow(clippy::mutable_key_type)]
pub fn cut_store_range(ctx: &mut SplitStoreContext, store: &Arc<UOp>, range: &Arc<UOp>) -> Option<Arc<UOp>> {
    use morok_device::DeviceSpec;
    use morok_ir::types::BinaryOp;

    // Guard 1: CPU only
    if ctx.device != Some(DeviceSpec::Cpu) {
        return None;
    }

    // Guard 2: Range end must be constant
    let Op::Range { end, axis_type, .. } = range.op() else {
        return None;
    };
    let range_end = const_uop_to_i64(end)?;
    if range_end <= 0 {
        return None;
    }

    // Find cut points from CMPLT consumers of this range
    let consumer_map = store.get_consumer_map();
    let consumers = consumer_map.get(&UOpKey(range.clone()))?;

    let cuts: Vec<i64> = consumers
        .iter()
        .filter_map(|c| {
            // Must be: Lt(range, const)
            let Op::Binary(BinaryOp::Lt, lhs, rhs) = c.op() else {
                return None;
            };
            if !Arc::ptr_eq(lhs, range) {
                return None;
            }
            const_uop_to_i64(rhs)
        })
        .collect();

    if cuts.is_empty() {
        return None;
    }

    // Build full cut list: [0, ...cuts..., range_end]
    let mut all_cuts: Vec<i64> = vec![0];
    all_cuts.extend(cuts.iter().filter(|&&c| c > 0 && c < range_end));
    all_cuts.push(range_end);
    all_cuts.sort();
    all_cuts.dedup();

    if all_cuts.len() < 3 {
        return None; // Need at least 2 segments
    }

    // Create split stores
    let segments: Vec<Arc<UOp>> = all_cuts
        .windows(2)
        .map(|w| {
            let (start, seg_end) = (w[0], w[1]);
            let seg_size = seg_end - start;

            // Create new range with unique axis ID
            let new_axis_id = ctx.next_axis_id();
            let new_range = UOp::range_axis(UOp::index_const(seg_size), new_axis_id, *axis_type);

            // Build substitution: r → new_r + start
            let offset = if start == 0 { new_range.clone() } else { new_range.add(&UOp::index_const(start)) };

            // Substitute the range in the store
            #[allow(clippy::mutable_key_type)]
            let subs: HashMap<UOpKey, Arc<UOp>> = [(UOpKey(range.clone()), offset)].into_iter().collect();
            let substituted = store.substitute(&subs);

            // Wrap with END containing the new range
            substituted.end(SmallVec::from_elem(new_range, 1))
        })
        .collect();

    Some(UOp::group(segments))
}

/// Pattern matcher for split store optimization (CPU-only).
///
/// Based on Tinygrad's pm_split_store (simplify.py).
/// Matches END(STORE(...), [range]) patterns and splits at comparison cut points.
pub fn pm_split_store() -> crate::TypedPatternMatcher<SplitStoreContext> {
    crate::patterns! {
        @context SplitStoreContext;

        // Match: END(STORE(...), [range]) where range is single
        _end @ End { computation: store @ Store { index: _, value: _, ranges: store_ranges }, ranges: end_ranges }
            if end_ranges.len() == 1 && store_ranges.is_empty()
        => |store, end_ranges| {
            let range = &end_ranges[0];
            cut_store_range(ctx, store, range)
        },
    }
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
        let new_src = transform_single_source(src, &input_ranges, ctx);
        if !Arc::ptr_eq(&new_src, src) {
            any_changed = true;
        }
        new_sources.push(new_src);
    }

    if any_changed { Some(new_sources) } else { None }
}

/// Linearize multi-index INDEX on BUFFERIZE using buffer's ranges as dimensions.
///
/// Tinygrad: `flatten_bufferize` collapses multi-range BUFFERIZE, then `pm_mops`
/// linearizes INDEX through movement ops. We combine both: directly linearize
/// INDEX indices using BUFFERIZE's ranges (the buffer's shape).
///
/// Runs as a BPM pattern in `pm_add_buffers_patterns`, so it sees the original
/// BUFFERIZE child before `bufferize_to_store` transforms it.
fn linearize_index_on_bufferize(node: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Index { buffer, indices, gate } = node.op() else { return None };
    if indices.len() <= 1 {
        return None;
    }

    // Find BUFFERIZE through optional EXPAND/RESHAPE wrapping from dead_axis_removal.
    // After cleanup_dead_axes_bufferize: EXPAND(RESHAPE(BUFFERIZE_reduced)) or direct BUFFERIZE.
    let bufferize = find_bufferize_through_movement(buffer)?;
    let Op::Bufferize { ranges, .. } = bufferize.op() else { return None };

    if indices.len() != ranges.len() {
        return None;
    }

    // Extract dimension sizes from BUFFERIZE ranges.
    // RANGE(end=CONST(n)) → n, CONST(0) → 1 (singleton/broadcast)
    let dims: Vec<i64> = ranges
        .iter()
        .map(|r| {
            if let Some(size) = range_size_as_i64(r) {
                Some(size)
            } else if matches!(r.op(), Op::Const(cv) if cv.0.is_zero()) {
                Some(1)
            } else {
                None
            }
        })
        .collect::<Option<_>>()?;

    if dims.iter().any(|&d| d <= 0) {
        return None;
    }

    let strides = compute_row_major_strides(&dims);
    let flat_idx = build_linear_index(indices, &strides);

    let builder = UOp::index().buffer(buffer.clone()).indices(vec![flat_idx]).dtype(node.dtype());
    match gate {
        Some(g) => builder.gate(g.clone()).call().ok(),
        None => builder.call().ok(),
    }
}

/// Traverse EXPAND/RESHAPE wrappers (from dead_axis_removal) to find the underlying BUFFERIZE.
fn find_bufferize_through_movement(node: &Arc<UOp>) -> Option<Arc<UOp>> {
    let mut current = node.clone();
    for _ in 0..5 {
        match current.op() {
            Op::Bufferize { .. } => return Some(current),
            Op::Expand { src, .. } | Op::Reshape { src, .. } => current = src.clone(),
            _ => return None,
        }
    }
    None
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
    src: &Arc<UOp>,
    input_ranges: &[Arc<UOp>],
    ctx: &mut IndexingContext,
) -> Arc<UOp> {
    // Case 1: Buffer-like op → add multi-index INDEX
    // Unlike Case 2 (BUFFERIZE), we can't linearize here because the buffer's
    // dimensional structure isn't directly available from ctx — the output_ranges
    // may contain PAD validity expressions, not clean RANGE ops.
    // Linearization is deferred to pm_linearize_multi_index.
    if matches!(
        src.op(),
        Op::Buffer { .. } | Op::BufferView { .. } | Op::MStack { .. } | Op::MSelect { .. } | Op::After { .. }
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

        let opts = if output_ranges.len() == realize_axes.len() {
            BufferizeOpts { device: None, addrspace: AddrSpace::Global }
        } else {
            BufferizeOpts { device: None, addrspace: AddrSpace::Local }
        };

        let bufferized = UOp::bufferize(Arc::clone(src), closed_ranges.clone(), opts);

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

/// Apply buffer removal patterns while protecting SINK sources.
fn apply_buffer_removal_protecting_sink(
    sink: &Arc<UOp>,
    matcher: &crate::TypedPatternMatcher<super::kernel::PcontigConfig>,
    ctx: &mut super::kernel::PcontigConfig,
) -> Arc<UOp> {
    let Op::Sink { sources } = sink.op() else {
        return crate::rewrite::graph_rewrite(matcher, sink.clone(), ctx);
    };

    // Collect the rewrite roots — compute subtrees for BUFFERIZE, full nodes otherwise.
    // Using graph_rewrite_roots shares a single engine across all roots, so shared
    // subgraphs (e.g., bitonic sort network) are only processed once.
    let roots: Vec<Arc<UOp>> = sources
        .iter()
        .map(|src| match src.op() {
            Op::Bufferize { compute, .. } => compute.clone(),
            _ => src.clone(),
        })
        .collect();

    let optimized = crate::rewrite::graph_rewrite_roots(matcher, &roots, ctx);

    let new_sources: Vec<Arc<UOp>> = sources
        .iter()
        .zip(optimized)
        .map(|(src, opt)| match src.op() {
            Op::Bufferize { ranges, opts, .. } => UOp::bufferize(opt, ranges.to_vec(), opts.clone()),
            _ => opt,
        })
        .collect();

    UOp::sink(new_sources)
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
            UOp::try_where(cond, lhs.clone(), rhs.clone()).ok()?
        }
    })
}

/// Calculate buffer size from RANGE operations.
fn calculate_size_from_ranges(ranges: &SmallVec<[Arc<UOp>; 4]>) -> usize {
    if ranges.is_empty() {
        return 1;
    }

    ranges
        .iter()
        .map(|r| {
            if let Op::Range { end, .. } = r.op() {
                match end.vmax() {
                    ConstValue::Int(v) if *v > 0 => *v as usize,
                    ConstValue::UInt(v) if *v > 0 => *v as usize,
                    other => panic!(
                        "Cannot allocate buffer with symbolic size: range bound resolved to {:?}. \
                         Buffers require concrete sizes (Tinygrad: 'no symbolic sized buffers')",
                        other
                    ),
                }
            } else {
                1
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
    }
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
    // Filter to only actual RANGE ops, excluding CONST(0) from collapsed singleton dims.
    // Tinygrad alignment: `.end(*rngs)` where `rngs = sorted(idx.ranges, ...)` naturally
    // excludes non-RANGE entries because `.ranges` only collects RANGE UOps.
    let end_ranges: SmallVec<[Arc<UOp>; 4]> =
        ranges.iter().filter(|r| matches!(r.op(), Op::Range { .. })).cloned().collect();

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
            .ok()?;

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
            let zero_idx =
                UOp::index().buffer(buf.clone()).indices(vec![zero_range.clone()]).dtype(sdtype.clone()).call().ok()?;
            let zero_store = zero_idx.store_value(identity).end(smallvec![zero_range.clone()]);
            let buf_zeroed = buf.after(smallvec![zero_store]);

            // Build linear index from BUFFERIZE ranges (not reduce ranges)
            // Filter to only actual RANGE ops (exclude CONST(0) from collapsed dims)
            // Sort by axis_id for correct row-major linearization (Tinygrad: rangeify.py:303)
            let sorted_ranges = sort_ranges_by_axis_id(&end_ranges);
            let linear_index = if sorted_ranges.len() > 1 {
                let dims: Vec<i64> = sorted_ranges.iter().filter_map(range_size_as_i64).collect();
                if dims.len() != sorted_ranges.len() {
                    panic!(
                        "ICE: symbolic ranges in OUTER REDUCE bufferize_to_store \
                                 (resolved {}/{} dims). Symbolic buffer sizes are not supported.",
                        dims.len(),
                        sorted_ranges.len()
                    );
                }
                let strides = compute_row_major_strides(&dims);
                let indices: Vec<Arc<UOp>> = sorted_ranges.iter().cloned().collect();
                build_linear_index(&indices, &strides)
            } else if !sorted_ranges.is_empty() {
                sorted_ranges[0].clone()
            } else {
                UOp::index_const(0)
            };

            // Accumulation: buf[idx] = buf[idx] OP reduce_src (Tinygrad: bufi = buf.index(idx, dtype=sdtype))
            let buf_idx = UOp::index()
                .buffer(buf_zeroed.clone())
                .indices(vec![linear_index])
                .dtype(sdtype.clone())
                .call()
                .ok()?;
            let loaded = UOp::load().buffer(buf_zeroed.clone()).index(buf_idx.clone()).call();
            let accumulated = reduce_op_to_binary(*reduce_op, &loaded, reduce_src)?;

            // Wrap store with both end_ranges AND outer_range
            let do_store = buf_idx.store_value(accumulated).end(end_ranges.clone()).end(smallvec![outer_range]);

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
        // The BUFFER → DEFINE_GLOBAL conversion happens later in split_store
        let device = opts.device.clone().unwrap_or(morok_ir::DeviceSpec::Cpu);
        UOp::new_buffer(device, size, base_dtype.clone())
    } else {
        // For local address space (only when allow_locals=true), create DEFINE_LOCAL directly (like Tinygrad)
        let local_ptr_dtype = base_dtype.clone().ptr(Some(size), opts.addrspace);
        let local_id = ctx.next_local();
        let local_buf = UOp::define_local(local_id, local_ptr_dtype);

        // Broadcast to vector count if needed (Tinygrad: rangeify.py:343)
        // Tinygrad uses x.src[1].dtype.count (the idx's vcount). In Morok, we use
        // compute.dtype().vcount() which should match since the stored value's
        // vectorization determines the indexing granularity.
        let vcount = compute.dtype().vcount();
        if vcount > 1 { local_buf.broadcast(vcount) } else { local_buf }
    };

    // Use ptr=true to keep Ptr dtype for STORE targets (Tinygrad-aligned).
    // This ensures INDEX returns pointer type, which STORE codegen expects.
    // ptr=true is equivalent to setting dtype to buffer.dtype(), but is the
    // idiomatic way per Tinygrad's buf.index(idx, ptr=True).

    // Filter ranges to only include actual RANGE ops (not CONST(0) from collapsed dims).
    //
    // Tinygrad alignment: In Tinygrad, `bufferize_to_store` gets ranges via `idx.ranges`
    // which is a property that traverses the expression tree and collects only actual
    // RANGE UOps. CONST(0) entries (created by `new_range(size=1)` for singleton
    // dimensions, e.g. from keepdim=true reductions) are NOT ranges and are naturally
    // excluded. Morok stores all entries in BUFFERIZE.ranges including CONST(0), so
    // we must filter here to match Tinygrad's behavior.
    let active_ranges: SmallVec<[Arc<UOp>; 4]> =
        ranges.iter().filter(|r| matches!(r.op(), Op::Range { .. })).cloned().collect();

    // Sort active ranges by axis_id for correct row-major linearization (Tinygrad: rangeify.py:303)
    let sorted_ranges = sort_ranges_by_axis_id(&active_ranges);

    let store_target = if !sorted_ranges.is_empty() {
        // Linearize multi-dimensional ranges into single linear index.
        // Buffer is 1D (DEFINE_GLOBAL with total size), so we compute:
        //   linear = r0 * (s1*s2*...) + r1 * (s2*s3*...) + ... + rN
        // using row-major stride calculation.
        //
        // We have direct access to RANGE operations here, so we can extract
        // concrete dimensions. This is the proper place to linearize because
        // later passes (pm_linearize_multi_index) only see the 1D buffer shape.
        if sorted_ranges.len() > 1 {
            // Extract sizes from each RANGE
            let dims: Vec<i64> = sorted_ranges.iter().filter_map(range_size_as_i64).collect();

            if dims.len() != sorted_ranges.len() {
                panic!(
                    "ICE: symbolic ranges in bufferize_to_store \
                     (resolved {}/{} dims). Symbolic buffer sizes are not supported.",
                    dims.len(),
                    sorted_ranges.len()
                );
            }

            // All ranges have concrete sizes - linearize
            let strides = compute_row_major_strides(&dims);
            let indices: Vec<Arc<UOp>> = sorted_ranges.iter().cloned().collect();
            trace!(
                "bufferize_to_store: linearizing {} ranges with dims {:?}, strides {:?}",
                sorted_ranges.len(),
                dims,
                strides
            );
            let linear_index = build_linear_index(&indices, &strides);
            UOp::index()
                .buffer(buffer.clone())
                .indices(vec![linear_index])
                .dtype(sdtype.clone())
                .call()
                .expect("Failed to create INDEX for BUFFERIZE-to-STORE conversion")
        } else {
            // Single range - use directly
            UOp::index()
                .buffer(buffer.clone())
                .indices(sorted_ranges.to_vec())
                .dtype(sdtype.clone())
                .call()
                .expect("Failed to create INDEX for BUFFERIZE-to-STORE conversion")
        }
    } else {
        // Scalar store: create INDEX with buffer + index 0 and explicit sdtype
        UOp::index()
            .buffer(buffer.clone())
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
                        | Op::DefineGlobal { .. }
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
        if result.toposort().iter().any(|x| matches!(x.op(), Op::Range { .. })) {
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
    let pm = super::patterns::build_reduce_collapse_matcher();
    reduce_collapse_with(src, ranges, &pm)
}

/// Collapse REDUCE using extended `pm_reduce_load_collapse` patterns.
///
/// Tinygrad: `reduce_load_collapse(red, u)` — calls `reduce_collapse` with
/// `pm=pm_reduce_load_collapse` which includes `.or_casted()` variants,
/// NE lifting, and the full `pm_load_collapse` non-REDUCE patterns.
pub fn reduce_load_collapse(src: &Arc<UOp>, ranges: &[Arc<UOp>]) -> Option<Arc<UOp>> {
    let pm = super::patterns::build_reduce_load_collapse_matcher();
    reduce_collapse_with(src, ranges, &pm)
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
    // This ensures we only merge ranges that appear in consistent positions across all REDUCEs
    let reduce_ranges: Vec<SmallVec<[Arc<UOp>; 4]>> = u
        .toposort()
        .iter()
        .filter_map(|dep| match dep.op() {
            Op::Reduce { ranges, .. } => Some(ranges.clone()),
            _ => None,
        })
        .collect();

    // Try to merge adjacent pairs for END, or all permutations for REDUCE
    let pairs: Vec<(usize, usize)> = if matches!(u.op(), Op::End { .. }) {
        // For END: only try adjacent pairs (zip pattern from Tinygrad)
        (0..ended_ranges.len() - 1).map(|i| (i, i + 1)).collect()
    } else {
        // For REDUCE: try all permutations
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

        // Check that both ranges have the same axis type (Tinygrad simplify.py:24)
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

        // Tinygrad simplify.py:25-27: Check that both ranges appear in the same REDUCE scopes
        // This prevents invalid merges when ranges have different visibility in different REDUCEs
        let valid_reduce_scope = reduce_ranges.iter().all(|rngs| {
            let r0_in = rngs.iter().any(|rng| Arc::ptr_eq(rng, r0));
            let r1_in = rngs.iter().any(|rng| Arc::ptr_eq(rng, r1));
            r0_in == r1_in // Both present or both absent in this REDUCE
        });
        if !valid_reduce_scope {
            continue;
        }

        // Get range sizes as UOps (supports both constant and symbolic sizes)
        // Skip obviously invalid cases (constant <= 0)
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

        // For constant sizes, check overflow
        if let (Some(s0), Some(s1)) = (const_uop_to_i64(r0_end), const_uop_to_i64(r1_end))
            && s0.checked_mul(s1).is_none()
        {
            continue;
        }

        // Create merged size symbolically: s0 * s1 (Tinygrad simplify.py:28)
        let merged_size_uop = r0_end.mul(r1_end);

        // Create merged range with symbolic size
        let merged_range = r0.with_sources(vec![merged_size_uop]);

        // Create substitutions: r0 = merged // s1, r1 = merged % s1 (Tinygrad simplify.py:29)
        // Using s1 as UOp for symbolic support
        let new_r0 = merged_range.idiv(r1_end);
        let new_r1 = merged_range.mod_(r1_end);

        // Create substitution map
        #[allow(clippy::mutable_key_type)]
        let mut subs: HashMap<UOpKey, Arc<UOp>> = HashMap::new();
        subs.insert(UOpKey(r0.clone()), new_r0);
        subs.insert(UOpKey(r1.clone()), new_r1);

        // Apply substitution and simplify (Tinygrad simplify.py:30-31)
        let rewritten = u.substitute(&subs);
        let matcher = crate::symbolic::symbolic_simple();
        let simplified = crate::rewrite::graph_rewrite(&matcher, rewritten, &mut ());

        // Count divmod operations (Tinygrad simplify.py:34-36)
        use crate::passes::linearize_index::count_divmod;
        let original_divmod = count_divmod(u);
        let new_divmod = count_divmod(&simplified);

        // Only accept if divmod count is reduced or equal
        if new_divmod <= original_divmod {
            return Some(simplified);
        }
    }

    None
}

/// Pattern matcher for range simplification.
///
/// Tries to merge adjacent ranges to reduce divmod operations.
pub fn pm_simplify_ranges() -> crate::TypedPatternMatcher {
    crate::patterns! {
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

/// Create pattern matcher for adding buffers (BUFFERIZE → STORE conversion).
///
/// Based on Tinygrad's pm_add_buffers (rangeify.py:358-367) with `allow_locals=False`.
/// Uses a shared KernelContext (like Tinygrad's `ctx=itertools.count(lunique_start)`)
/// to ensure unique buffer IDs across all pattern matches.
pub fn pm_add_buffers_patterns() -> crate::TypedPatternMatcher<super::kernel::KernelContext> {
    crate::patterns! {
        @context super::kernel::KernelContext;
        // Linearize multi-index INDEX on BUFFERIZE (Tinygrad: flatten_bufferize + pm_mops)
        node if matches!(node.op(), Op::Index { indices, .. } if indices.len() > 1)
            => |node, _ctx| { linearize_index_on_bufferize(node) },
        // pm_mops: push movement ops through INDEX (Tinygrad rangeify.py:24-26)
        Index { buffer: mop, indices, gate } if mop.op().is_movement()
            => |mop, indices, gate, _ctx| {
                super::patterns::transform_movement_through_index(mop, indices, gate)
            },
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
        // Linearize multi-index INDEX on BUFFERIZE (Tinygrad: flatten_bufferize + pm_mops)
        node if matches!(node.op(), Op::Index { indices, .. } if indices.len() > 1)
            => |node, _ctx| { linearize_index_on_bufferize(node) },
        // pm_mops: push movement ops through INDEX (Tinygrad rangeify.py:24-26)
        Index { buffer: mop, indices, gate } if mop.op().is_movement()
            => |mop, indices, gate, _ctx| {
                super::patterns::transform_movement_through_index(mop, indices, gate)
            },
        // BUFFERIZE → STORE conversion (allow_locals=true: create DEFINE_LOCAL for local addrspace)
        buf @ Bufferize { compute: _ } => |buf, ctx| {
            bufferize_to_store(buf, ctx, true)
        },
    }
}
