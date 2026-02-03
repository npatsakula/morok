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

use morok_ir::{AddrSpace, BufferizeOpts, ConstValue, DType, Op, UOp, UOpKey};
use smallvec::SmallVec;
use tracing::{debug, trace};

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
#[tracing::instrument(skip_all, fields(origin.tree = sink.tree()))]
pub fn rangeify_with_map(
    sink: Arc<UOp>,
    pcontig_config: Option<&super::kernel::PcontigConfig>,
) -> morok_ir::Result<RangeifyResult> {
    use morok_ir::rewrite::{graph_rewrite, graph_rewrite_bottom_up_with_map, graph_rewrite_with_map};

    // Aggregate all becomes_maps from rewrite passes
    let mut all_becomes: HashMap<UOpKey, Arc<UOp>> = HashMap::new();

    // =========================================================================
    // Stage 0: Range assignment to build IndexingContext
    // =========================================================================
    let (mut sink, mut indexing_ctx) = super::indexing::run_rangeify(sink)?;
    tracing::debug!(uop.tree = sink.tree(), "Stage 0: range assignment complete");

    // Pre-cleanup: Apply early rewrites (DETACH, CONTIGUOUS_BACKWARD removal)
    let early_matcher = super::patterns::early_rewrites();
    let result = graph_rewrite_bottom_up_with_map(&early_matcher, sink, &mut ());
    sink = result.root;
    all_becomes.extend(result.becomes_map);
    tracing::debug!(uop.tree = sink.tree(), "early rewrites complete");

    // Split large reductions BEFORE ReduceAxis → REDUCE conversion
    let mut split_config = super::kernel::SplitReduceOpConfig::default();
    let split_matcher = super::patterns::split_reduceop_patterns();
    let result = graph_rewrite_with_map(&split_matcher, sink, &mut split_config);
    sink = result.root;
    all_becomes.extend(result.becomes_map);
    tracing::debug!(uop.tree = sink.tree(), "split reduceops complete");

    // =========================================================================
    // Stage 1: Early movement ops (BOTTOM_UP) - Tinygrad: pm_mops + pm_syntactic_sugar
    // MUST RUN FIRST before other optimizations
    // =========================================================================
    // Note: Movement ops are currently integrated in apply_rangeify_patterns
    // For exact alignment, we'd separate pm_mops but current integration works
    let rangeify_matcher = super::patterns::apply_rangeify_patterns();
    let result = graph_rewrite_bottom_up_with_map(&rangeify_matcher, sink, &mut indexing_ctx);
    sink = result.root;
    all_becomes.extend(result.becomes_map);
    tracing::debug!(uop.tree = sink.tree(), "Stage 1: rangeify + movement ops complete");

    // Buffer simplification
    let buffer_simplify = super::patterns::buffer_folding() + super::patterns::dead_axis_removal();
    let result = graph_rewrite_bottom_up_with_map(&buffer_simplify, sink, &mut ());
    sink = result.root;
    all_becomes.extend(result.becomes_map);
    tracing::debug!(uop.tree = sink.tree(), "buffer folding + dead axis removal complete");

    // Apply early rewrites again for RESHAPE to scalar
    let result = graph_rewrite_bottom_up_with_map(&early_matcher, sink, &mut ());
    sink = result.root;
    all_becomes.extend(result.becomes_map);
    tracing::debug!(uop.tree = sink.tree(), "reshape to scalar complete");

    // =========================================================================
    // Stage 2: Load collapse - Tinygrad: pm_load_collapse
    // Collapse load tensor indexing and REDUCE with conditional patterns
    // =========================================================================
    // First: apply pm_load_collapse for gated load patterns
    let load_collapse_matcher = super::patterns::pm_load_collapse();
    sink = graph_rewrite(&load_collapse_matcher, sink, &mut ());
    tracing::debug!(uop.tree = sink.tree(), "Stage 2a: load collapse complete");

    // Then: apply reduction simplify patterns (reduce_unparented, reduce_collapse)
    let reduction_matcher = super::patterns::reduction_simplify_patterns();
    let result = graph_rewrite_bottom_up_with_map(&reduction_matcher, sink, &mut ());
    sink = result.root;
    all_becomes.extend(result.becomes_map);
    tracing::debug!(uop.tree = sink.tree(), "Stage 2b: reduction simplify complete");

    // =========================================================================
    // Stage 3: Split ranges - Tinygrad: pm_split_ranges + pm_flatten_range
    // Range splitting via modulo arithmetic
    // =========================================================================
    // First: mark RANGE % const patterns and substitute at SINK
    let split_matcher = pm_split_ranges();
    let mut split_ctx = SplitRangesContext::default();
    sink = graph_rewrite(&split_matcher, sink, &mut split_ctx);
    tracing::debug!(uop.tree = sink.tree(), "Stage 3a: split ranges complete");

    // Then: flatten nested ranges on REDUCE/STORE/END
    let flatten_matcher = pm_flatten_range();
    sink = graph_rewrite(&flatten_matcher, sink, &mut ());
    tracing::debug!(uop.tree = sink.tree(), "Stage 3b: flatten ranges complete");

    // =========================================================================
    // Stage 4: Initial symbolic - Tinygrad: sym + pm_flatten_range (TOP_DOWN)
    // CRITICAL: Must run BEFORE pm_simplify_ranges
    // NOTE: Using symbolic() (not symbolic_simple()) to include gep_pushing_patterns
    //       This aligns with Tinygrad's sym which includes GEP pushing at every stage.
    // =========================================================================
    let symbolic_with_flatten = crate::symbolic::symbolic() + flatten_matcher;
    sink = graph_rewrite(&symbolic_with_flatten, sink, &mut ());
    tracing::debug!(uop.tree = sink.tree(), "Stage 4: initial symbolic complete");

    // =========================================================================
    // Stage 5: Simplify ranges - Tinygrad: pm_simplify_ranges
    // Merge adjacent ranges to reduce divmod operations
    // =========================================================================
    sink = graph_rewrite(&pm_simplify_ranges(), sink, &mut ());
    tracing::debug!(uop.tree = sink.tree(), "Stage 5: simplify ranges complete");

    // =========================================================================
    // Stage 6: Split store (CPU-only) - Tinygrad: pm_split_store
    // Split stores at comparison cut points to enable branch elimination
    // =========================================================================
    if let Some(device) = super::patterns::extract_device_from_graph(&sink)
        && device == morok_device::DeviceSpec::Cpu
    {
        let mut split_store_ctx = SplitStoreContext::for_device(device);
        let split_store_matcher = pm_split_store();
        sink = graph_rewrite(&split_store_matcher, sink, &mut split_store_ctx);
        tracing::debug!(uop.tree = sink.tree(), "Stage 6: split store complete");
    }

    // =========================================================================
    // Stage 7: Buffer removal with partial contiguous
    // =========================================================================
    let mut pcontig = pcontig_config.cloned().unwrap_or_default();
    let buffer_removal_matcher = super::patterns::buffer_removal_with_pcontig();
    sink = apply_buffer_removal_protecting_sink(&sink, &buffer_removal_matcher, &mut pcontig);
    tracing::debug!(uop.tree = sink.tree(), "Stage 7: buffer removal complete");

    // Buffer limit enforcement
    if let Some(device) = super::patterns::extract_device_from_graph(&sink)
        && let Some(limit) = device.max_buffers()
    {
        let limit_matcher = super::patterns::buffer_limit_patterns(limit);
        let result = graph_rewrite_bottom_up_with_map(&limit_matcher, sink, &mut ());
        sink = result.root;
        all_becomes.extend(result.becomes_map);
        tracing::debug!(uop.tree = sink.tree(), "Stage 7b: buffer limit enforcement complete");
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
            && let Op::Range { end, axis_id, axis_type } = uop.op()
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
    let Op::Range { end, axis_id: _, axis_type } = range.op() else {
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

    let (input_ranges, _) = ctx.get_ranges(x)?;
    let input_ranges = input_ranges.clone(); // Clone to release borrow on ctx

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

/// Transform a single source by adding BUFFERIZE + INDEX if needed.
pub(crate) fn transform_single_source(
    _consumer: &Arc<UOp>,
    src: &Arc<UOp>,
    input_ranges: &[Arc<UOp>],
    ctx: &mut IndexingContext,
) -> Arc<UOp> {
    trace!(
        src.id = src.id,
        src.op = ?std::mem::discriminant(src.op()),
        consumer.id = _consumer.id,
        input_ranges.len = input_ranges.len(),
        "transform_single_source"
    );

    // Case 1: Buffer-like op → add INDEX
    if matches!(
        src.op(),
        Op::Buffer { .. } | Op::BufferView { .. } | Op::MStack { .. } | Op::MSelect { .. } | Op::After { .. }
    ) {
        trace!(src.id = src.id, case = "buffer-like", "adding index");
        return UOp::index()
            .buffer(Arc::clone(src))
            .indices(input_ranges.to_vec())
            .call()
            .expect("Failed to create INDEX for buffer source");
    }

    // Case 1.5: Movement op → transform indices through it and recurse
    // This handles EXPAND [1]->[4] by transforming [Range(0..4)] to [Const(0)]
    // We apply movement op transformation here directly (like Tinygrad's apply_movement_op)
    // Then recursively call transform_single_source to let the appropriate case handle the inner source
    if src.op().is_movement() {
        trace!(src.id = src.id, case = "movement", "processing movement op");
        use super::indexing::apply_movement_op;

        // Get the inner source of the movement op
        let inner_src = &src.op().sources()[0];

        // Get the source shape for transformation
        if let Some(inner_shape) = inner_src.shape().ok().flatten() {
            // Transform indices through the movement op
            let transformed_indices = apply_movement_op(src.op(), inner_shape, input_ranges);

            // Recursively call transform_single_source with transformed indices
            // This will hit the appropriate case (Case 1 for buffer/AFTER, CONST case, etc.)
            return transform_single_source(_consumer, inner_src, &transformed_indices, ctx);
        }

        // Fallback: if we can't get shape, create INDEX on the movement chain
        // (will be handled by pattern matcher later)
        return UOp::index()
            .buffer(Arc::clone(src))
            .indices(input_ranges.to_vec())
            .call()
            .expect("Failed to create INDEX for movement source");
    }

    // Check for REDUCE op that might have been converted from ReduceAxis
    // During graph rewrite, ReduceAxis is converted to REDUCE but realize_map was keyed on ReduceAxis
    // We need to handle both cases
    let realize_axes_opt = ctx.get_realize_axes(src).cloned();

    debug!(
        src.id = src.id,
        src.op = ?std::mem::discriminant(src.op()),
        has_realize_axes = realize_axes_opt.is_some(),
        realize_axes = ?realize_axes_opt,
        "Checking realize_map"
    );

    // Case 2: source needs realization → wrap in BUFFERIZE + INDEX
    if let Some(ref realize_axes) = realize_axes_opt {
        trace!(src.id = src.id, realize_axes = ?realize_axes, case = "realize", "source needs realization");
        // Check if ANY source of this node is also marked for realization
        // If so, skip wrapping here - the inner source should be wrapped first
        for inner_src in src.op().sources() {
            if ctx.should_realize(&inner_src) {
                debug!(src.id = src.id, inner_src.id = inner_src.id, "Skipping - inner source also needs realization");
                // Return the source as-is - inner source will be wrapped when accessed
                return Arc::clone(src);
            }
        }

        debug!(
            src.id = src.id,
            realize_axes = ?realize_axes,
            "Creating BUFFERIZE for realized source"
        );
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

        let bufferized = UOp::bufferize(Arc::clone(src), closed_ranges, opts);

        // Use input ranges AS-IS (Tinygrad indexing.py:78)
        // Do NOT convert Reduce→Loop here - that only happens in limit_bufs for buffer overflow.
        // REDUCE consumers need Reduce-type ranges to share the same loop counter with their source INDEX.
        let index_ranges: Vec<_> = input_ranges
            .iter()
            .enumerate()
            .filter(|(i, _)| realize_axes.contains(i))
            .map(|(_, r)| Arc::clone(r))
            .collect();

        if !index_ranges.is_empty() {
            return UOp::index()
                .buffer(bufferized)
                .indices(index_ranges)
                .call()
                .expect("Failed to create INDEX after BUFFERIZE");
        } else {
            return bufferized;
        }
    }

    // Case 4: Intermediate compute op (Unary, Binary, etc.) → recursively transform sources
    // This ensures that when REDUCE → Unary(Neg) → Buffer, the Buffer gets INDEX
    // Aligns with Tinygrad's approach where patterns run on ALL nodes
    if matches!(src.op(), Op::Unary(..) | Op::Binary(..) | Op::Ternary(..) | Op::Cast { .. } | Op::BitCast { .. }) {
        trace!(
            src.id = src.id,
            src.op = ?std::mem::discriminant(src.op()),
            num_sources = src.op().sources().len(),
            case = "compute",
            "Recursively transforming compute op sources"
        );
        let inner_sources = src.op().sources();
        let mut new_inner_sources = Vec::with_capacity(inner_sources.len());
        let mut any_changed = false;

        for inner_src in inner_sources.iter() {
            let transformed = transform_single_source(_consumer, inner_src, input_ranges, ctx);
            trace!(
                inner_src.id = inner_src.id,
                transformed.id = transformed.id,
                changed = !Arc::ptr_eq(&transformed, inner_src),
                "Inner source transformation"
            );
            if !Arc::ptr_eq(&transformed, inner_src) {
                any_changed = true;
            }
            new_inner_sources.push(transformed);
        }

        if any_changed {
            trace!(src.id = src.id, "rebuilding compute op with new sources");
            return src.with_sources(new_inner_sources);
        }
    }

    // Case 3: no transformation needed
    trace!(src.id = src.id, case = "no-transform", "no transformation needed");
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

    let mut new_sources = Vec::with_capacity(sources.len());
    for src in sources.iter() {
        if let Op::Bufferize { compute, ranges, opts } = src.op() {
            let optimized_compute = crate::rewrite::graph_rewrite(matcher, compute.clone(), ctx);
            let new_bufferize = UOp::bufferize(optimized_compute, ranges.to_vec(), opts.clone());
            new_sources.push(new_bufferize);
        } else {
            let optimized = crate::rewrite::graph_rewrite(matcher, src.clone(), ctx);
            new_sources.push(optimized);
        }
    }

    UOp::sink(new_sources)
}

// ============================================================================
// BUFFERIZE TO STORE CONVERSION
// ============================================================================

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

    // Determine effective address space based on allow_locals parameter
    // Tinygrad has two matchers:
    // - pm_add_buffers (allow_locals=False): treats local as global
    // - pm_add_buffers_local (allow_locals=True): creates DEFINE_LOCAL for local
    let effective_addrspace = if allow_locals { opts.addrspace } else { AddrSpace::Global };

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
        UOp::define_local(local_id, local_ptr_dtype)
    };

    // Create Ptr dtype for STORE target (like Tinygrad's sdtype = x.dtype.ptr(size, addrspace))
    // This ensures INDEX returns pointer type, which STORE codegen expects.
    let ptr_dtype = base_dtype.ptr(Some(size), opts.addrspace);

    let store_target = if !ranges.is_empty() {
        // Linearize multi-dimensional ranges into single linear index.
        // Buffer is 1D (DEFINE_GLOBAL with total size), so we compute:
        //   linear = r0 * (s1*s2*...) + r1 * (s2*s3*...) + ... + rN
        // using row-major stride calculation.
        //
        // We have direct access to RANGE operations here, so we can extract
        // concrete dimensions. This is the proper place to linearize because
        // later passes (pm_linearize_multi_index) only see the 1D buffer shape.
        if ranges.len() > 1 {
            // Extract sizes from each RANGE
            let dims: Vec<i64> = ranges.iter().filter_map(range_size_as_i64).collect();

            if dims.len() == ranges.len() {
                // All ranges have concrete sizes - linearize
                let strides = compute_row_major_strides(&dims);
                let indices: Vec<Arc<UOp>> = ranges.iter().cloned().collect();
                trace!(
                    "bufferize_to_store: linearizing {} ranges with dims {:?}, strides {:?}",
                    ranges.len(),
                    dims,
                    strides
                );
                let linear_index = build_linear_index(&indices, &strides);
                UOp::index()
                    .buffer(buffer.clone())
                    .indices(vec![linear_index])
                    .dtype(ptr_dtype.clone())
                    .call()
                    .expect("Failed to create INDEX for BUFFERIZE-to-STORE conversion")
            } else {
                // Symbolic ranges - can't linearize at compile time
                // Create multi-index INDEX and let pm_linearize_multi_index handle it
                // (which may use vmin/vmax fallback or fail at codegen)
                debug!(
                    "bufferize_to_store: symbolic ranges, creating multi-index INDEX (dims resolved: {}/{})",
                    dims.len(),
                    ranges.len()
                );
                UOp::index()
                    .buffer(buffer.clone())
                    .indices(ranges.to_vec())
                    .dtype(ptr_dtype.clone())
                    .call()
                    .expect("Failed to create INDEX for BUFFERIZE-to-STORE conversion")
            }
        } else {
            // Single range - use directly
            UOp::index()
                .buffer(buffer.clone())
                .indices(ranges.to_vec())
                .dtype(ptr_dtype.clone())
                .call()
                .expect("Failed to create INDEX for BUFFERIZE-to-STORE conversion")
        }
    } else {
        // Scalar store: create INDEX with buffer + index 0 and explicit Ptr dtype
        UOp::index()
            .buffer(buffer.clone())
            .indices(vec![UOp::index_const(0)])
            .dtype(ptr_dtype)
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

    // Determine END ranges: use ONLY the output's ranges (from BUFFERIZE).
    //
    // This matches Tinygrad's approach in rangeify.py:303,337 where:
    // - rngs = sorted(idx.ranges, key=lambda x: x.arg)
    // - do_store = buf.index(idx, dtype=sdtype).store(x.src[0]).end(*rngs)
    //
    // For scalar stores (e.g., REDUCE results), ranges is empty, so END wraps with no ranges.
    // The REDUCE's loop is handled separately by pm_reduce which creates its own END.
    // We must NOT fallback to compute.in_scope_ranges() because that would put the
    // OUTPUT STORE inside the reduce loop (wrong!).
    let end_ranges: SmallVec<[Arc<UOp>; 4]> = ranges.clone();

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

/// Lift range-independent computations outside REDUCE via symbolic simplification.
#[allow(clippy::mutable_key_type)]
pub fn reduce_collapse(src: &Arc<UOp>, ranges: &[Arc<UOp>]) -> Option<Arc<UOp>> {
    if ranges.is_empty() {
        return None;
    }

    let mut substitute_map: HashMap<UOpKey, Arc<UOp>> = HashMap::new();

    for (i, range) in ranges.iter().enumerate() {
        let size_i64 = super::indexing::range_size_as_i64(range)?;

        if size_i64 <= 0 {
            return None;
        }

        let var_name = format!("ridx{}", i);
        let define_var = UOp::define_var(var_name, 0, size_i64 - 1);

        substitute_map.insert(UOpKey(Arc::clone(range)), define_var);
    }

    let substituted = src.substitute(&substitute_map);
    let matcher = crate::symbolic::symbolic_simple();
    let simplified = crate::rewrite::graph_rewrite(&matcher, substituted, &mut ());

    let vars_in_simplified: HashSet<UOpKey> =
        simplified.toposort().into_iter().filter(|uop| matches!(uop.op(), Op::DefineVar { .. })).map(UOpKey).collect();

    let has_var_dependency = substitute_map.values().any(|var| vars_in_simplified.contains(&UOpKey(Arc::clone(var))));

    if has_var_dependency {
        return None;
    }

    if !super::indexing::no_range(&simplified) {
        return None;
    }

    let reverse_map: HashMap<UOpKey, Arc<UOp>> =
        substitute_map.into_iter().map(|(range_key, var)| (UOpKey(var), range_key.0)).collect();

    Some(simplified.substitute(&reverse_map))
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
        let original_divmod = count_divmod(u);
        let new_divmod = count_divmod(&simplified);

        // Only accept if divmod count is reduced or equal
        if new_divmod <= original_divmod {
            return Some(simplified);
        }
    }

    None
}

/// Count IDIV and MOD operations in a UOp graph.
fn count_divmod(uop: &Arc<UOp>) -> usize {
    use morok_ir::types::BinaryOp;
    uop.toposort().iter().filter(|u| matches!(u.op(), Op::Binary(BinaryOp::Idiv | BinaryOp::Mod, _, _))).count()
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
/// This treats all BUFFERIZE ops as global buffers, regardless of their addrspace.
/// Use `pm_add_buffers_local()` when local buffers should be created for local addrspace.
pub fn pm_add_buffers_patterns() -> crate::TypedPatternMatcher<()> {
    use super::kernel::KernelContext;

    // This is a workaround - we create a temporary KernelContext for each match
    // The original code used a shared context, but the new pipeline uses graph_rewrite
    crate::patterns! {
        // BUFFERIZE → STORE conversion (allow_locals=false: treat local as global)
        buf @ Bufferize { compute: _ } => |buf| {
            let mut temp_ctx = KernelContext::new();
            bufferize_to_store(buf, &mut temp_ctx, false)
        },
    }
}

/// Create pattern matcher for adding buffers with local buffer support.
///
/// Based on Tinygrad's pm_add_buffers_local (rangeify.py:358-367) with `allow_locals=True`.
/// This creates DEFINE_LOCAL buffers for local addrspace BUFFERIZE ops.
pub fn pm_add_buffers_local_patterns() -> crate::TypedPatternMatcher<()> {
    use super::kernel::KernelContext;

    crate::patterns! {
        // BUFFERIZE → STORE conversion (allow_locals=true: create DEFINE_LOCAL for local addrspace)
        buf @ Bufferize { compute: _ } => |buf| {
            let mut temp_ctx = KernelContext::new();
            bufferize_to_store(buf, &mut temp_ctx, true)
        },
    }
}
