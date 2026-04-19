//! Range assignment and indexing context for rangeify transformation.
//!
//! This module provides the core range assignment algorithm that converts
//! movement operations into explicit loop ranges.

use std::collections::HashMap;
use std::sync::Arc;

use morok_dtype::DType;
use morok_ir::{AxisId, AxisType, BinaryOp, ConstValue, Op, SInt, UOp, UOpKey};
use tracing::{debug, info_span, instrument, trace, warn};

use crate::argsort;

// ============================================================================
// Context
// ============================================================================

/// (input_ranges, output_ranges) for a UOp.
type UOpRanges = (Vec<Arc<UOp>>, Vec<Arc<UOp>>);

/// Rangeify observability counters.
#[derive(Debug, Clone, Default)]
pub struct RangeifyStats {
    pub recovery_retries: usize,
    pub leaked_pad_ops: usize,
    pub leaked_reduceaxis_ops: usize,
    pub pad_fallback_attempts: usize,
    pub reduceaxis_fallback_attempts: usize,
    pub fallback_suppressed: usize,
}

/// Context for range assignment during rangeify.
#[derive(Default)]
pub struct IndexingContext {
    /// Maps UOps to realize status: Some(axes) = needs realization on axes.
    pub realize_map: HashMap<UOpKey, Option<Vec<usize>>>,
    /// Maps each UOp to its (input_ranges, output_ranges).
    pub range_map: HashMap<UOpKey, UOpRanges>,
    /// Counter for generating unique range IDs.
    range_idx: usize,
    /// Observability counters for fallback/recovery behavior.
    pub stats: RangeifyStats,
}

impl IndexingContext {
    /// Create a new indexing context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create new RANGE with unique ID. Returns const 0 if size is 1.
    ///
    /// Ranges are created with `AxisId::Unrenumbered` to mark them as needing
    /// renumbering. The `renumber_range` pattern will later convert them to
    /// `AxisId::Renumbered` with sequential IDs starting from 0 for each kernel.
    pub fn new_range(&mut self, size: &SInt, axistype: AxisType) -> Arc<UOp> {
        // If size is already a RANGE UOp, return it unchanged (Tinygrad indexing.py:46)
        if let SInt::Symbolic(u) = size
            && matches!(u.op(), Op::Range { .. })
        {
            return Arc::clone(u);
        }
        // Check if size is constant 1
        if let SInt::Const(1) = size {
            return UOp::index_const(0);
        }

        self.new_range_uncollapsed(size, axistype)
    }

    /// Create new RANGE with unique ID, without collapsing size=1 to Const(0).
    /// Use this when you need a proper Range UOp for realization/kernel boundaries.
    pub fn new_range_uncollapsed(&mut self, size: &SInt, axistype: AxisType) -> Arc<UOp> {
        // Create range with Unrenumbered axis_id
        let axis_id = AxisId::Unrenumbered(self.range_idx);
        self.range_idx += 1;

        let size_uop = size.to_uop(morok_dtype::DType::Index);

        UOp::range_axis(size_uop, axis_id, axistype)
    }

    /// Create a new RANGE from an existing UOp end value.
    /// Used when converting REDUCE ranges to LOOP ranges during bufferization.
    ///( when bufferizing, REDUCE ranges become LOOP)
    pub fn new_range_from_uop(&mut self, end: &Arc<UOp>, axis_type: AxisType) -> Arc<UOp> {
        let axis_id = AxisId::Unrenumbered(self.range_idx);
        self.range_idx += 1;
        UOp::range_axis(Arc::clone(end), axis_id, axis_type)
    }

    /// Mark a UOp for realization on all axes.
    pub fn mark_realize_all(&mut self, uop: &Arc<UOp>) -> morok_ir::Result<()> {
        if let Some(shape) = uop.shape()? {
            let axes = (0..shape.len()).collect();
            self.realize_map.insert(UOpKey(Arc::clone(uop)), Some(axes));
        }
        Ok(())
    }

    /// Mark a UOp for realization on specific axes.
    pub fn mark_realize(&mut self, uop: &Arc<UOp>, axes: Vec<usize>) {
        self.realize_map.insert(UOpKey(Arc::clone(uop)), Some(axes));
    }

    /// Check if a UOp is in the realize map.
    pub fn should_realize(&self, uop: &Arc<UOp>) -> bool {
        self.realize_map.contains_key(&UOpKey(Arc::clone(uop)))
    }

    /// Get the realize axes for a UOp.
    pub fn get_realize_axes(&self, uop: &Arc<UOp>) -> Option<&Vec<usize>> {
        self.realize_map.get(&UOpKey(Arc::clone(uop))).and_then(|opt| opt.as_ref())
    }

    /// Get all keys in the realize_map (for debugging).
    #[allow(dead_code)]
    pub fn realize_map_keys(&self) -> Vec<&UOpKey> {
        self.realize_map.keys().collect()
    }

    /// Set the range map for a UOp.
    pub fn set_ranges(&mut self, uop: &Arc<UOp>, input_ranges: Vec<Arc<UOp>>, output_ranges: Vec<Arc<UOp>>) {
        self.range_map.insert(UOpKey(Arc::clone(uop)), (input_ranges, output_ranges));
    }

    /// Get the ranges for a UOp.
    pub fn get_ranges(&self, uop: &Arc<UOp>) -> Option<&UOpRanges> {
        self.range_map.get(&UOpKey(Arc::clone(uop)))
    }

    /// Get the current range counter value.
    pub fn range_counter(&self) -> usize {
        self.range_idx
    }

    /// Record one recovery retry and leaked high-level op counts.
    pub fn record_recovery_retry(&mut self, leaked_pad: usize, leaked_reduceaxis: usize) {
        self.stats.recovery_retries += 1;
        self.stats.leaked_pad_ops += leaked_pad;
        self.stats.leaked_reduceaxis_ops += leaked_reduceaxis;
    }

    /// Record PAD fallback usage.
    pub fn record_pad_fallback(&mut self) -> bool {
        self.stats.pad_fallback_attempts += 1;
        true
    }

    /// Record ReduceAxis fallback usage.
    pub fn record_reduceaxis_fallback(&mut self) -> bool {
        self.stats.reduceaxis_fallback_attempts += 1;
        true
    }
}

// ============================================================================
// Core Algorithm
// ============================================================================

/// Cache for memoizing expensive `graph_rewrite` calls during rangeify.
///
/// Matches upstream `@functools.cache` on `_apply_reshape` and `apply_movement_op`.
/// Keyed by input UOp's `content_hash` (structural, O(1) — pre-computed at creation).
/// Hash-consing ensures same expression = same Arc = same `id` = cache hit.
///
/// Scoped to a single `run_rangeify` call — automatically freed when the function returns.
/// Uses `UOp.id` (identity) as key — collision-free because all UOps are held alive
/// within a single run, and hash consing guarantees same structure = same Arc = same id.
#[derive(Default)]
pub struct SimplifyCache {
    cache: HashMap<u64, Arc<UOp>>,
}

impl SimplifyCache {
    /// Look up or compute a graph_rewrite simplification.
    /// `input` is the UOp to simplify; `f` computes the result if not cached.
    #[inline]
    fn get_or_simplify(&mut self, input: &Arc<UOp>, f: impl FnOnce() -> Arc<UOp>) -> Arc<UOp> {
        if let Some(cached) = self.cache.get(&input.id) {
            return cached.clone();
        }
        let result = f();
        self.cache.insert(input.id, result.clone());
        result
    }
}

/// Run range assignment on a UOp graph. Returns (transformed_sink, context).
#[allow(clippy::mutable_key_type)]
#[instrument(skip(sink), fields(sink_id = sink.id))]
pub fn run_rangeify(sink: Arc<UOp>) -> morok_ir::Result<(Arc<UOp>, IndexingContext)> {
    let mut ctx = IndexingContext::new();
    let mut simplify_cache = SimplifyCache::default();

    // Step 1: Generate realize map via pattern matcher (pm_generate_realize_map)
    // bottom_up=True — patterns see ORIGINAL children
    crate::rewrite::graph_rewrite_bottom_up(pm_generate_realize_map(), sink.clone(), &mut ctx);

    // Step 2: Get toposort (root-to-leaves) and consumer map
    let consumer_map = sink.get_consumer_map();

    // Use forward toposort (root first) for range propagation
    let forward_topo: Vec<_> = sink.toposort().into_iter().rev().collect();

    // Step 3: Assign ranges via forward traversal
    assign_ranges(&forward_topo, &consumer_map, &mut ctx, &mut simplify_cache)?;

    // Step 4: Apply rangeify patterns (pm_apply_rangeify)
    // Converts ReduceAxis→REDUCE, PAD→WHERE, creates BUFFERIZE+INDEX, removes movement ops.
    // Must run bottom_up so patterns see ORIGINAL children (bottom_up=True).
    let rangeify_matcher = super::patterns::apply_rangeify_patterns();
    let mut transformed_sink = crate::rewrite::graph_rewrite_bottom_up(&rangeify_matcher, sink, &mut ctx);

    // Recovery pass for rangeify leaks caused by node reconstruction during rewrite.
    //
    // In some large graphs, a movement op can be reconstructed after initial
    // range assignment (same semantics, different node identity), so it misses
    // `range_map` and escapes conversion/removal in the first pm_apply_rangeify pass.
    // Re-running realize/range assignment on the rewritten graph and applying
    // pm_apply_rangeify once more resolves these misses while preserving parity
    // behavior for normal kernels.
    for pass_idx in 0..2 {
        let mut leaked_pad = 0usize;
        let mut leaked_reduceaxis = 0usize;
        for n in transformed_sink.toposort() {
            match n.op() {
                Op::Pad { .. } => leaked_pad += 1,
                Op::ReduceAxis { .. } => leaked_reduceaxis += 1,
                _ => {}
            }
        }

        if leaked_pad == 0 && leaked_reduceaxis == 0 {
            break;
        }

        ctx.record_recovery_retry(leaked_pad, leaked_reduceaxis);

        tracing::debug!(
            recovery_pass = pass_idx + 1,
            leaked_pad,
            leaked_reduceaxis,
            "run_rangeify: leaked high-level ops detected, re-running assign+apply"
        );

        // Rebuild a fresh context on each retry to avoid stale range_map/realize_map
        // entries from previous passes affecting subsequent recovery behavior.
        let mut retry_ctx = IndexingContext::new();
        retry_ctx.stats = ctx.stats.clone();

        crate::rewrite::graph_rewrite_bottom_up(pm_generate_realize_map(), transformed_sink.clone(), &mut retry_ctx);

        let consumer_map = transformed_sink.get_consumer_map();
        let forward_topo: Vec<_> = transformed_sink.toposort().into_iter().rev().collect();
        let mut pass_cache = SimplifyCache::default();
        assign_ranges(&forward_topo, &consumer_map, &mut retry_ctx, &mut pass_cache)?;

        transformed_sink = crate::rewrite::graph_rewrite_bottom_up(&rangeify_matcher, transformed_sink, &mut retry_ctx);
        ctx = retry_ctx;
    }

    if ctx.stats.recovery_retries > 0
        || ctx.stats.pad_fallback_attempts > 0
        || ctx.stats.reduceaxis_fallback_attempts > 0
    {
        tracing::debug!(
            recovery_retries = ctx.stats.recovery_retries,
            leaked_pad_ops = ctx.stats.leaked_pad_ops,
            leaked_reduceaxis_ops = ctx.stats.leaked_reduceaxis_ops,
            pad_fallback_attempts = ctx.stats.pad_fallback_attempts,
            reduceaxis_fallback_attempts = ctx.stats.reduceaxis_fallback_attempts,
            fallback_suppressed = ctx.stats.fallback_suppressed,
            "rangeify diagnostics"
        );
    }

    let leaked =
        transformed_sink.toposort().into_iter().find(|n| matches!(n.op(), Op::Pad { .. } | Op::ReduceAxis { .. }));
    if leaked.is_some() {
        return Err(morok_ir::Error::SymbolicShapeUnsupported {
            operation: "rangeify leaked high-level PAD/ReduceAxis after recovery".to_string(),
        });
    }

    Ok((transformed_sink, ctx))
}

/// Pattern matcher for generating the realize map (`pm_generate_realize_map`).
///
/// Marks which UOps need to be materialized to buffers:
/// - SINK sources (if not always-contiguous)
/// - COPY, CONTIGUOUS, ASSIGN (always realized)
/// - Sources of COPY, MSTACK, MSELECT, ASSIGN (realized if not always-contiguous)
///
/// Patterns return `None` (no rewrite) — context side-effects mark nodes in the realize map.
fn pm_generate_realize_map() -> &'static crate::TypedPatternMatcher<IndexingContext> {
    crate::cached_patterns! {
        @context IndexingContext;

        // SINK sources → realize non-contiguous bases
        // Tinygrad: ctx.update((x.base, None) for x in s.src if x.base.op not in ALWAYS_CONTIGUOUS)
        x @ Sink { sources: _ } => |x, ctx| {
            for src in x.op().sources() {
                let base = src.base();
                if !is_always_contiguous(&base) {
                    ctx.mark_realize_all(&base).ok();
                }
            }
            None
        },

        // Always realize these ops: COPY, BUFFER_VIEW, CONTIGUOUS, STORE, ASSIGN
        x @ Store { index: _, value: _ } => |x, ctx| { ctx.mark_realize_all(x).ok(); None },
        x @ BufferView { buffer: _ } => |x, ctx| { ctx.mark_realize_all(x).ok(); None },

        // Always realize REDUCE on outer ranges
        x @ Reduce { src: _, ranges, reduce_op: _ } => |x, ctx| {
            if ranges.iter().any(|r| matches!(r.op(), Op::Range { axis_type, .. } if *axis_type == AxisType::Outer)) {
                ctx.mark_realize_all(x).ok();
            }
            None
        },

        x @ Copy { src: _ } => |x, ctx| {
            ctx.mark_realize_all(x).ok();
            // Also realize sources
            for src in x.op().sources() {
                // Tinygrad realize_srcs: guard on src.base.op, realize src.
                if !is_always_contiguous(&src.base()) {
                    ctx.mark_realize_all(&src).ok();
                }
            }
            None
        },
        x @ Contiguous { src: _ } => |x, ctx| { ctx.mark_realize_all(x).ok(); None },
        x @ Assign { target: _, value: _ } => |x, ctx| {
            ctx.mark_realize_all(x).ok();
            for src in x.op().sources() {
                // Tinygrad realize_srcs: guard on src.base.op, realize src.
                if !is_always_contiguous(&src.base()) {
                    ctx.mark_realize_all(&src).ok();
                }
            }
            None
        },

        // MStack/MSelect → realize sources
        x @ MStack { buffers: _ } => |x, ctx| {
            for src in x.op().sources() {
                // Tinygrad realize_srcs: guard on src.base.op, realize src.
                if !is_always_contiguous(&src.base()) {
                    ctx.mark_realize_all(&src).ok();
                }
            }
            None
        },
        x @ MSelect { device_index: _ } => |x, ctx| {
            for src in x.op().sources() {
                // Tinygrad realize_srcs: guard on src.base.op, realize src.
                if !is_always_contiguous(&src.base()) {
                    ctx.mark_realize_all(&src).ok();
                }
            }
            None
        },
    }
}

/// Check if a UOp is always contiguous (doesn't need realization).
///
/// Aligned with ALWAYS_CONTIGUOUS.
/// When the source of a BUFFERIZE is in this set, the BUFFERIZE gets `removable: false`,
/// preventing it from being inlined by buffer removal.
pub(crate) fn is_always_contiguous(uop: &Arc<UOp>) -> bool {
    matches!(
        uop.op(),
        Op::Contiguous { .. }
            | Op::Assign { .. }
            | Op::Copy { .. }
            | Op::Buffer { .. }
            | Op::BufferView { .. }
            | Op::Const(_)
            | Op::Bind { .. }
            | Op::Device(_)
            | Op::MSelect { .. }
            | Op::MStack { .. }
            | Op::Param { .. }
            | Op::DefineLocal(_)
            | Op::DefineReg { .. }
            | Op::Load { .. }
            | Op::Kernel { .. }
    )
}

/// Check if a UOp represents constant true (handles unsimplified OR expressions).
fn is_const_true(uop: &Arc<UOp>) -> bool {
    match uop.op() {
        Op::Const(cv) => matches!(cv.0, ConstValue::Bool(true)),
        Op::Binary(BinaryOp::Or, a, b) => is_const_true(a) && is_const_true(b),
        _ => false,
    }
}

/// Merge ranges from multiple consumers. Creates new ranges and marks realization when needed.
#[instrument(skip(uop, consumer_rngs, ctx), fields(uop_id = uop.id))]
pub(crate) fn merge_consumer_ranges(
    uop: &Arc<UOp>,
    consumer_rngs: &[Vec<Arc<UOp>>],
    ctx: &mut IndexingContext,
) -> morok_ir::Result<Vec<Arc<UOp>>> {
    let Some(shape) = uop.shape()? else {
        return Ok(Vec::new());
    };

    let num_dims = shape.len();

    // Transpose: consumer_rngs[consumer_idx][dim_idx] → all_rngs[dim_idx][consumer_idx]
    let mut all_rngs: Vec<Vec<Arc<UOp>>> = vec![Vec::new(); num_dims];
    for consumer_rng in consumer_rngs {
        for (dim_idx, range) in consumer_rng.iter().enumerate() {
            if dim_idx < num_dims {
                all_rngs[dim_idx].push(Arc::clone(range));
            }
        }
    }

    let mut out_rngs = Vec::new();
    let mut realize_axes = Vec::new();

    // Compute all_all_same FIRST — if ANY dimension
    // has incompatible ranges across consumers, ALL dimensions get realized.
    // With PCONTIG=0 (default): condition per-dim = `all_all_same || (PCONTIG && all_same(dim))`.
    // When all_all_same=False and PCONTIG=0, this is always False → all dims realized.
    let all_all_same = all_rngs.iter().all(|dim_ranges| {
        if dim_ranges.is_empty() {
            return false;
        }
        if dim_ranges.iter().skip(1).all(|r| Arc::ptr_eq(&dim_ranges[0], r)) {
            return true;
        }
        let indices: Vec<_> = dim_ranges.iter().map(|r| r.get_idx()).collect();
        all_ranges_same(&indices)
    });

    for (dim_idx, dim_ranges) in all_rngs.iter().enumerate() {
        if dim_ranges.is_empty() {
            out_rngs.push(ctx.new_range(&shape[dim_idx], AxisType::Loop));
            realize_axes.push(dim_idx);
            continue;
        }

        // FAST PATH: If all ranges are pointer-equal, return original unchanged
        if dim_ranges.iter().skip(1).all(|r| Arc::ptr_eq(&dim_ranges[0], r)) && all_all_same {
            out_rngs.push(Arc::clone(&dim_ranges[0]));
            continue;
            // all_all_same=False but this dim is same → still realize (PCONTIG=0 behavior)
        }

        let indices: Vec<_> = dim_ranges.iter().map(|r| r.get_idx()).collect();
        let valids: Vec<_> = dim_ranges.iter().map(|r| r.get_valid()).collect();
        let ranges_same = all_ranges_same(&indices);

        // if all_all_same or (PCONTIG and all_same): merge
        // With PCONTIG=0 (default): only merge when all_all_same is True.
        if all_all_same {
            debug!(dim_idx, ranges_same, all_all_same, "merge_consumer_ranges: merging dimension");
            let merged_idx = Arc::clone(&indices[0]);
            let merged_valid = if valids.len() == 1 {
                Arc::clone(&valids[0])
            } else {
                valids.iter().skip(1).try_fold(Arc::clone(&valids[0]), |acc, v| acc.try_or_op(v))?
            };

            // Build WHERE(valid, idx, Invalid) and simplify immediately.
            // Without this simplification, unsimplified WHERE/Not chains accumulate
            // and cause oscillation in downstream symbolic passes.
            let merged_range = if is_const_true(&merged_valid) {
                merged_idx
            } else {
                let raw = UOp::try_where(merged_valid, merged_idx, UOp::invalid_marker())?;
                // Uses full `symbolic` here (not symbolic_simple)
                crate::rewrite::graph_rewrite(crate::symbolic::patterns::symbolic(), raw, &mut ())
            };
            out_rngs.push(merged_range);
        } else {
            debug!(dim_idx, "merge_consumer_ranges: creating NEW Loop range (ranges not compatible)");
            out_rngs.push(ctx.new_range(&shape[dim_idx], AxisType::Loop));
            realize_axes.push(dim_idx);
        }
    }

    if !realize_axes.is_empty() {
        if uop.dtype().scalar() == Some(morok_dtype::ScalarDType::Index) {
            debug!(realize_axes = ?realize_axes, "range conflict on Index op - marking axes for realization");
        } else {
            warn!(realize_axes = ?realize_axes, "range conflict detected - marking axes for realization");
        }
        ctx.mark_realize(uop, realize_axes.clone());
    }

    Ok(out_rngs)
}

/// Assign input/output ranges for each UOp via reverse toposort traversal.
#[allow(clippy::mutable_key_type)]
#[instrument(skip_all)]
fn assign_ranges(
    reverse_topo: &[Arc<UOp>],
    consumer_map: &HashMap<UOpKey, Vec<Arc<UOp>>>,
    ctx: &mut IndexingContext,
    simplify_cache: &mut SimplifyCache,
) -> morok_ir::Result<()> {
    // Local variable for ending_ranges - only used within this function
    let mut ending_ranges: HashMap<UOpKey, Vec<Arc<UOp>>> = HashMap::new();

    for x in reverse_topo {
        if matches!(x.op(), Op::Device(_) | Op::Unique(_)) {
            continue;
        }

        // Keep Index-typed dataflow in range assignment so movement-heavy integer
        // branches can be lowered; skip only literal/index-definition helpers.
        if x.dtype().scalar() == Some(morok_dtype::ScalarDType::Index)
            && matches!(x.op(), Op::Const(_) | Op::DefineVar { .. } | Op::Bind { .. } | Op::Index { .. })
        {
            continue;
        }

        // Skip KERNEL internals and MSTACK/MSELECT (treated like sink-level containers).
        if matches!(x.op(), Op::Kernel { .. } | Op::MStack { .. } | Op::MSelect { .. }) {
            continue;
        }

        let _span = info_span!("assign_range", uop_id = x.id, op = x.op().as_ref()).entered();

        let consumers: Vec<_> = consumer_map.get(&UOpKey(x.clone())).cloned().unwrap_or_default();
        let consumer_rngs: Vec<Vec<Arc<UOp>>> =
            consumers.iter().filter_map(|c| ctx.get_ranges(c).map(|(inp, _)| inp.clone())).collect();

        debug!(
            num_consumers = consumers.len(),
            consumer_rngs_len = consumer_rngs.len(),
            consumer_ids = ?consumers.iter().map(|c| c.id).collect::<Vec<_>>(),
            "Consumer info"
        );

        // Inherit ending_ranges from consumers
        // ending_ranges propagate from consumers → producers (backward in data flow)
        let mut inherited_ending: Vec<Arc<UOp>> = Vec::new();
        for consumer in &consumers {
            inherited_ending.extend(ending_ranges.get(&UOpKey(consumer.clone())).cloned().unwrap_or_default());
        }
        if !inherited_ending.is_empty() {
            debug!(
                node_id = x.id,
                inherited_count = inherited_ending.len(),
                consumer_ids = ?consumers.iter().map(|c| c.id).collect::<Vec<_>>(),
                "ending_ranges: node inherits from consumers"
            );
        }
        ending_ranges.insert(UOpKey(x.clone()), inherited_ending);

        let mut out_rngs = if ctx.should_realize(x) {
            // Realized op: create fresh ranges for all dimensions.
            // CONTIGUOUS, COPY, ASSIGN, and ops marked by ending_ranges all land here.
            if let Some(shape) = x.shape()? {
                debug!(
                    node_id = x.id,
                    op = x.op().as_ref(),
                    dims = shape.len(),
                    "REALIZE via realize_map (fresh ranges)"
                );
                let rngs: Vec<_> = shape.iter().map(|s| ctx.new_range(s, AxisType::Loop)).collect();
                let axes: Vec<usize> = (0..shape.len()).collect();
                ctx.realize_map.insert(UOpKey(x.clone()), Some(axes));
                // Clear ending_ranges when realized
                ending_ranges.insert(UOpKey(x.clone()), Vec::new());
                rngs
            } else {
                continue;
            }
        // ReduceAxis uses the same consumer_rngs branching as all other ops
        } else if consumer_rngs.is_empty() {
            continue;
        } else if consumer_rngs.len() == 1 {
            consumer_rngs[0].clone()
        } else {
            merge_consumer_ranges(x, &consumer_rngs, ctx)?
        };

        debug!(should_realize = ctx.should_realize(x), out_rngs_len = out_rngs.len(), "output ranges computed");

        // Check ending_ranges FIRST (before in_rngs computation)
        // ending_ranges realization happens BEFORE input ranges
        // This is critical: in_rngs must be computed from the FINAL out_rngs after realization
        let ending = ending_ranges.get(&UOpKey(x.clone())).cloned().unwrap_or_default();
        if !ending.is_empty() {
            debug!(
                ending_count = ending.len(),
                triggers_realization =
                    matches!(x.op(), Op::ReduceAxis { .. } | Op::Assign { .. }) || is_elementwise_op(x),
                "Ending ranges detected (pre-in_rngs check)"
            );
        }
        // Use ending ranges directly without filtering (matches upstream behavior).
        let filtered_ending = ending.clone();

        if !filtered_ending.is_empty()
            && (matches!(x.op(), Op::ReduceAxis { .. } | Op::Assign { .. }) || is_elementwise_op(x))
        {
            if let Some(shape) = x.shape().ok().flatten() {
                // Start with existing realize_axes (from merge_consumer_ranges)
                let mut realize_axes: Vec<usize> = ctx.get_realize_axes(x).cloned().unwrap_or_default();

                // `if not (PCONTIG > 1) or any(any(rr.arg > e.arg ...) ...)`
                // With PCONTIG=0 (default), `not (0 > 1)` = True, so ALL axes are unconditionally
                // realized when ending_ranges are present. This is critical for layernorm-style
                // patterns where `centered = x - mean` is shared between output and variance paths.
                // The ending_ranges from EXPAND (broadcasting mean/inv_std) must trigger full
                // realization of elementwise ops in the backward slice.
                for (i, _r) in out_rngs.iter().enumerate() {
                    if realize_axes.contains(&i) {
                        continue;
                    }
                    realize_axes.push(i);
                }

                debug!(
                    node_id = x.id,
                    op = x.op().as_ref(),
                    ending_count = ending.len(),
                    realize_axes = ?realize_axes,
                    "SELECTIVE REALIZATION via ending_ranges"
                );

                // Clear ending_ranges after handling
                ending_ranges.insert(UOpKey(x.clone()), Vec::new());

                if !realize_axes.is_empty() {
                    // Mark for realization
                    ctx.mark_realize(x, realize_axes.clone());

                    // Selectively replace only realized axes (preserve others)
                    out_rngs = out_rngs
                        .iter()
                        .enumerate()
                        .map(|(i, r)| {
                            if realize_axes.contains(&i) {
                                if let Some(dim) = shape.get(i) {
                                    ctx.new_range(dim, AxisType::Loop)
                                } else {
                                    Arc::clone(r)
                                }
                            } else {
                                Arc::clone(r)
                            }
                        })
                        .collect();
                }
            } else {
                ending_ranges.insert(UOpKey(x.clone()), Vec::new());
            }
        }

        // NOW compute in_rngs from the FINAL out_rngs (after any realization updates)
        let in_rngs = match x.op() {
            Op::Reshape { src, .. }
            | Op::Permute { src, .. }
            | Op::Expand { src, .. }
            | Op::Pad { src, .. }
            | Op::Shrink { src, .. }
            | Op::Flip { src, .. } => {
                if let Some(in_shape) = src.shape()? {
                    apply_movement_op(x.op(), in_shape, &out_rngs, simplify_cache)
                } else {
                    out_rngs.clone()
                }
            }
            Op::ReduceAxis { src, axes, .. } => {
                if let Some(in_shape) = src.shape()? {
                    // Trace ReduceAxis range assignment details
                    if tracing::enabled!(tracing::Level::TRACE) {
                        let out_shape = x.shape()?;
                        trace!(
                            uop.id = x.id,
                            reduce.axes = ?axes,
                            in_shape.len = in_shape.len(),
                            out_shape.len = ?out_shape.as_ref().map(|s| s.len()),
                            out_rngs.len = out_rngs.len(),
                            "ReduceAxis range assignment"
                        );
                        for (idx, rng) in out_rngs.iter().enumerate() {
                            match rng.op() {
                                Op::Binary(binop, a, b) => {
                                    trace!(
                                        range.index = idx,
                                        range.id = rng.id,
                                        op = "Binary",
                                        binary_op = ?binop,
                                        left.id = a.id,
                                        right.id = b.id,
                                        "ReduceAxis out_rngs entry"
                                    );
                                }
                                Op::Range { axis_id, axis_type, .. } => {
                                    trace!(
                                        range.index = idx,
                                        range.id = rng.id,
                                        op = "Range",
                                        axis.id = ?axis_id,
                                        axis.type_ = ?axis_type,
                                        "ReduceAxis out_rngs entry"
                                    );
                                }
                                _ => {
                                    trace!(
                                        range.index = idx,
                                        range.id = rng.id,
                                        op = ?std::mem::discriminant(rng.op()),
                                        "ReduceAxis out_rngs entry"
                                    );
                                }
                            }
                        }
                    }

                    let mut rngs = Vec::with_capacity(in_shape.len());
                    for (i, s) in in_shape.iter().enumerate() {
                        if axes.contains(&i) {
                            rngs.push(ctx.new_range(s, AxisType::Reduce));
                        } else if i < out_rngs.len() {
                            rngs.push(Arc::clone(&out_rngs[i]));
                            trace!(dim.index = i, range.id = out_rngs[i].id, "ReduceAxis using existing out_rngs");
                        } else {
                            rngs.push(ctx.new_range(s, AxisType::Loop));
                        }
                    }
                    rngs
                } else {
                    out_rngs.clone()
                }
            }
            _ => out_rngs.clone(),
        };

        debug!(in_rngs_len = in_rngs.len(), "input ranges computed");

        // EXPAND marks ranges as ending when broadcasting to static dimensions
        // "if the EXPAND is used to inject a range, we don't mark it as ending_ranges. otherwise we do."
        if let Op::Expand { new_shape, .. } = x.op() {
            // Check if new_shape is all static (no RANGE ops being injected in the shape)
            let shape_is_static = extract_shape_from_uop(new_shape).iter().all(|s| match s {
                SInt::Const(_) | SInt::Infer => true,
                SInt::Symbolic(uop) => !matches!(uop.op(), Op::Range { .. }),
            });

            debug!(
                expand_id = x.id,
                shape_is_static = shape_is_static,
                in_rngs_len = in_rngs.len(),
                out_rngs_len = out_rngs.len(),
                in_rngs_ids = ?in_rngs.iter().map(|r| (r.id, format!("{:?}", std::mem::discriminant(r.op())))).collect::<Vec<_>>(),
                out_rngs_ids = ?out_rngs.iter().map(|r| (r.id, format!("{:?}", std::mem::discriminant(r.op())))).collect::<Vec<_>>(),
                "ending_ranges: EXPAND being processed"
            );

            if shape_is_static {
                // Ranges that changed (in_rngs != out_rngs) are "ending"
                // These are the output ranges that were collapsed to const 0 in in_rngs
                // upstream `.ranges.keys()` returns ALL range types without filtering.
                let mut changed_ranges: Vec<Arc<UOp>> = Vec::new();
                for (inp, out) in in_rngs.iter().zip(out_rngs.iter()) {
                    if !Arc::ptr_eq(inp, out) {
                        changed_ranges.extend(collect_ranges_from_uop(out));
                    }
                }

                if !changed_ranges.is_empty() {
                    debug!(
                        expand_id = x.id,
                        changed_ranges_count = changed_ranges.len(),
                        changed_range_ids = ?changed_ranges.iter().map(|r| r.id).collect::<Vec<_>>(),
                        "ending_ranges: EXPAND marking ranges as ending"
                    );
                    let mut ending = ending_ranges.get(&UOpKey(x.clone())).cloned().unwrap_or_default();
                    ending.extend(changed_ranges);
                    ending_ranges.insert(UOpKey(x.clone()), ending);
                }
            }
        }

        ctx.set_ranges(x, in_rngs, out_rngs);
    }
    Ok(())
}

// ============================================================================
// Movement Op Helpers (from helpers.rs)
// ============================================================================

/// Transform ranges through a movement op (SHRINK, PERMUTE, FLIP, EXPAND, PAD, RESHAPE).
pub fn apply_movement_op(
    op: &Op,
    in_shape: &[SInt],
    rngs: &[Arc<UOp>],
    simplify_cache: &mut SimplifyCache,
) -> Vec<Arc<UOp>> {
    match op {
        Op::Shrink { begins, .. } => {
            // Matches upstream:
            // case Ops.SHRINK: rngs = tuple(a if ss == 0 else a+ss for a,(ss,_) in zip(rngs, arg))
            let begin_uops = extract_shape_uops(begins);
            rngs.iter()
                .zip(begin_uops.iter())
                .map(|(rng, begin)| {
                    // Skip add when begin is zero (concrete optimization)
                    if is_const_zero(begin) {
                        Arc::clone(rng)
                    } else {
                        rng.try_add(begin).expect("SHRINK: try_add failed")
                    }
                })
                .collect()
        }

        Op::Permute { axes, .. } => {
            let inv_perm = argsort(axes);
            inv_perm.iter().map(|&i| Arc::clone(&rngs[i])).collect()
        }

        Op::Flip { axes: flips, .. } => rngs
            .iter()
            .zip(in_shape.iter())
            .zip(flips.iter())
            .map(|((rng, shape), &flip)| {
                if !flip {
                    Arc::clone(rng)
                } else {
                    let shape_uop = shape.to_uop(morok_dtype::DType::Index);
                    let shape_minus_1 = shape_uop.try_sub(&UOp::index_const(1)).unwrap();
                    shape_minus_1.try_sub(rng).unwrap()
                }
            })
            .collect(),

        Op::Expand { new_shape, .. } => {
            let new_shape_vals = extract_shape_from_uop(new_shape);

            // When rngs.len() < new_shape_vals.len(), pad from the left with CONST(0)
            // to align indices with trailing dimensions (same logic as RESHAPE padding).
            let padded_rngs: Vec<Arc<UOp>> = if rngs.len() < new_shape_vals.len() {
                let padding = new_shape_vals.len() - rngs.len();
                let mut v = Vec::with_capacity(new_shape_vals.len());
                for _ in 0..padding {
                    v.push(UOp::index_const(0));
                }
                v.extend(rngs.iter().cloned());
                v
            } else {
                rngs.to_vec()
            };

            // Also pad in_shape from the left with CONST(1) if needed
            let padded_in_shape: Vec<SInt> = if in_shape.len() < new_shape_vals.len() {
                let padding = new_shape_vals.len() - in_shape.len();
                let mut v = Vec::with_capacity(new_shape_vals.len());
                for _ in 0..padding {
                    v.push(SInt::Const(1));
                }
                v.extend(in_shape.iter().cloned());
                v
            } else {
                in_shape.to_vec()
            };

            padded_rngs
                .iter()
                .zip(padded_in_shape.iter())
                .zip(new_shape_vals.iter())
                .map(|((rng, in_sh), out_sh)| {
                    let expanding = match (in_sh, out_sh) {
                        (SInt::Const(1), SInt::Const(n)) if *n > 1 => true,
                        (SInt::Const(1), SInt::Symbolic(_)) => true,
                        _ => false,
                    };
                    if expanding { UOp::index_const(0) } else { Arc::clone(rng) }
                })
                .collect()
        }

        Op::Pad { begin_pads, end_pads, .. } => {
            let begin_uops = extract_shape_uops(begin_pads);
            let end_uops = extract_shape_uops(end_pads);
            rngs.iter()
                .zip(in_shape.iter())
                .zip(begin_uops.iter().zip(end_uops.iter()))
                .map(|((rng, shape), (begin, end))| {
                    if is_const_zero(begin) && is_const_zero(end) {
                        return Arc::clone(rng);
                    }
                    let shape_plus_begin = shape.to_uop(morok_dtype::DType::Index).try_add(begin).unwrap();
                    let valid_low = rng.try_cmplt(begin).unwrap().not();
                    let valid_high = rng.try_cmplt(&shape_plus_begin).unwrap();
                    let valid = valid_low.try_and_op(&valid_high).unwrap();
                    // graph_rewrite(validity, symbolic+pm_simplify_valid)
                    static PAD_SIMPLIFY: std::sync::LazyLock<crate::TypedPatternMatcher> =
                        std::sync::LazyLock::new(|| {
                            crate::symbolic::patterns::symbolic()
                                + crate::symbolic::valid_simplification::pm_simplify_valid()
                        });
                    let valid = simplify_cache.get_or_simplify(&valid, || {
                        crate::rewrite::graph_rewrite(&*PAD_SIMPLIFY, valid.clone(), &mut ())
                    });
                    let adjusted_rng = rng.try_sub(begin).unwrap();
                    UOp::try_where(valid, adjusted_rng, UOp::invalid_marker()).unwrap()
                })
                .collect()
        }

        Op::Reshape { new_shape, .. } => {
            let new_shape_vals = extract_shape_from_uop(new_shape);

            // Optimization: If in_shape == new_shape, this is a no-op reshape
            if in_shape.len() == new_shape_vals.len() {
                let mut is_same_shape = true;
                for (in_dim, out_dim) in in_shape.iter().zip(new_shape_vals.iter()) {
                    match (in_dim, out_dim) {
                        (SInt::Const(a), SInt::Const(b)) if a == b => continue,
                        (SInt::Symbolic(a), SInt::Symbolic(b)) if a.id == b.id => continue,
                        _ => {
                            is_same_shape = false;
                            break;
                        }
                    }
                }
                if is_same_shape {
                    return rngs.to_vec();
                }
            }

            // PLACEHOLDER canonicalization + reshape
            with_placeholder_canonicalization(rngs, |canonical| {
                apply_reshape_core(in_shape, &new_shape_vals, canonical, simplify_cache)
            })
        }

        _ => panic!("apply_movement_op called with non-movement op: {:?}", op),
    }
}

/// Core RESHAPE: flatten `rngs` by `out_shape` strides, decompose into `in_shape` via Mod/Idiv,
/// then run full symbolic simplification.
///
/// Matches upstream `_apply_reshape`.
/// Callers should PLACEHOLDER-canonicalize `rngs` before calling this.
fn apply_reshape_core(
    in_shape: &[SInt],
    out_shape: &[SInt],
    rngs: &[Arc<UOp>],
    simplify_cache: &mut SimplifyCache,
) -> Vec<Arc<UOp>> {
    use morok_ir::rewrite::graph_rewrite;

    // Pad with CONST(0) on the left when rngs.len() < out_shape.len()
    // (trailing-dimension alignment for partial INDEX)
    let padded_rngs: Vec<Arc<UOp>> = if rngs.len() < out_shape.len() {
        let padding = out_shape.len() - rngs.len();
        let mut v = Vec::with_capacity(out_shape.len());
        for _ in 0..padding {
            v.push(UOp::index_const(0));
        }
        v.extend(rngs.iter().cloned());
        v
    } else {
        rngs.to_vec()
    };

    // Flatten: combined = sum(acc_i * rng_i) with acc computed from out_shape
    let mut acc = UOp::index_const(1);
    let mut axes_in = Vec::new();
    for (shape_dim, rng) in out_shape.iter().zip(padded_rngs.iter()).rev() {
        axes_in.push(acc.try_mul(rng).unwrap());
        let dim_uop = shape_dim.to_uop(morok_dtype::DType::Index);
        acc = acc.try_mul(&dim_uop).unwrap();
    }
    let combined = axes_in.into_iter().reduce(|a, b| a.try_add(&b).unwrap()).unwrap_or_else(|| UOp::index_const(0));

    // Unflatten into in_shape via Mod/Idiv
    let mut axes_out = Vec::new();
    let mut remaining = combined;
    for shape_dim in in_shape.iter().rev() {
        let dim_uop = shape_dim.to_uop(morok_dtype::DType::Index);
        axes_out.push(remaining.try_mod(&dim_uop).unwrap());
        remaining = remaining.try_div(&dim_uop).unwrap();
    }
    axes_out.reverse();

    // Simplify ("This simplify is doing a lot of heavy lifting")
    static RESHAPE_SIMPLIFY: std::sync::LazyLock<crate::TypedPatternMatcher> = std::sync::LazyLock::new(|| {
        crate::symbolic::patterns::symbolic()
            + crate::symbolic::valid_simplification::pm_simplify_valid()
            + crate::symbolic::valid_simplification::pm_drop_and_clauses()
    });
    let sink = UOp::sink(axes_out);
    let simplified = simplify_cache.get_or_simplify(&sink, || graph_rewrite(&*RESHAPE_SIMPLIFY, sink.clone(), &mut ()));
    match simplified.op() {
        Op::Sink { sources } => sources.iter().cloned().collect(),
        _ => vec![simplified],
    }
}

/// Reshape ranges from `out_shape` to `in_shape` via flatten + unflatten.
///
/// Public wrapper around `apply_reshape_core` with PLACEHOLDER canonicalization.
/// Used by `flatten_bufferize` to convert multi-dim ranges to 1D.
pub fn apply_reshape_ranges(in_shape: &[SInt], out_shape: &[SInt], rngs: &[Arc<UOp>]) -> Vec<Arc<UOp>> {
    let mut cache = SimplifyCache::default();
    with_placeholder_canonicalization(rngs, |canonical| apply_reshape_core(in_shape, out_shape, canonical, &mut cache))
}

/// Canonicalize RANGE UOps to PLACEHOLDER before calling `f`, then restore.
/// Matches upstream.
fn with_placeholder_canonicalization(rngs: &[Arc<UOp>], f: impl FnOnce(&[Arc<UOp>]) -> Vec<Arc<UOp>>) -> Vec<Arc<UOp>> {
    let sink = UOp::sink(rngs.to_vec());
    // Tinygrad-compatible: canonicalize only live/in-scope ranges.
    let in_scope = sink.in_scope_ranges();
    let ranges_in_expr: Vec<Arc<UOp>> =
        sink.ranges().iter().filter(|r| in_scope.contains(&UOpKey((*r).clone()))).cloned().collect();

    #[allow(clippy::mutable_key_type)]
    let mut sub_map: HashMap<UOpKey, Arc<UOp>> = HashMap::new();
    #[allow(clippy::mutable_key_type)]
    let mut reverse_map: HashMap<UOpKey, Arc<UOp>> = HashMap::new();
    let mut reverse_axis_map: HashMap<usize, Arc<UOp>> = HashMap::new();
    for (i, r) in ranges_in_expr.iter().enumerate() {
        let Op::Range { end, .. } = r.op() else { continue };
        let placeholder = UOp::range_axis(end.clone(), AxisId::Renumbered(i), AxisType::Placeholder);
        sub_map.insert(UOpKey(r.clone()), placeholder.clone());
        reverse_map.insert(UOpKey(placeholder), r.clone());
        reverse_axis_map.insert(i, r.clone());
    }

    if sub_map.is_empty() {
        return f(rngs);
    }

    let canonical_sink = sink.substitute(&sub_map);
    let canonical_rngs: Vec<Arc<UOp>> = match canonical_sink.op() {
        Op::Sink { sources } => sources.iter().cloned().collect(),
        _ => vec![canonical_sink],
    };

    let result = f(&canonical_rngs);

    let result_sink = UOp::sink(result);
    let restored = result_sink.substitute(&reverse_map);
    let mut output: Vec<Arc<UOp>> = match restored.op() {
        Op::Sink { sources } => sources.iter().cloned().collect(),
        _ => vec![restored],
    };

    // If rewrite changed placeholder internals (e.g., `end` expr), structural reverse_map
    // can miss restoration. Recover by axis id to mirror tinygrad intent.
    #[allow(clippy::mutable_key_type)]
    let mut axis_restore_map: HashMap<UOpKey, Arc<UOp>> = HashMap::new();
    for r in UOp::sink(output.clone()).ranges().iter() {
        if let Op::Range { axis_id: AxisId::Renumbered(i), axis_type: AxisType::Placeholder, .. } = r.op()
            && let Some(orig) = reverse_axis_map.get(i)
        {
            axis_restore_map.insert(UOpKey(r.clone()), orig.clone());
        }
    }
    if !axis_restore_map.is_empty() {
        let axis_restored = UOp::sink(output).substitute(&axis_restore_map);
        output = match axis_restored.op() {
            Op::Sink { sources } => sources.iter().cloned().collect(),
            _ => vec![axis_restored],
        };
    }

    debug_assert!(
        !output.iter().any(|r| r
            .in_scope_ranges()
            .iter()
            .any(|rng| matches!(rng.0.op(), Op::Range { axis_type: AxisType::Placeholder, .. }))),
        "Placeholder-typed ranges leaked into output"
    );

    output
}

/// Check if a UOp is a constant zero.
fn is_const_zero(uop: &Arc<UOp>) -> bool {
    matches!(uop.op(), Op::Const(cv) if cv.0 == ConstValue::Int(0))
}

/// Extract shape UOps from a vectorize/const (symbolic-aware).
/// Returns individual element UOps — may be CONST or symbolic expressions.
/// Matches upstream `marg` which extracts via `sgep`.
fn extract_shape_uops(uop: &Arc<UOp>) -> Vec<Arc<UOp>> {
    match uop.op() {
        Op::Cast { src, .. } | Op::BitCast { src, .. } => extract_shape_uops(src),
        Op::Vectorize { elements } => elements.to_vec(),
        Op::Const(_) => vec![uop.clone()],
        Op::VConst { values } => values
            .iter()
            .map(|cv| match cv {
                ConstValue::Int(n) => UOp::index_const(*n),
                ConstValue::UInt(n) => UOp::index_const(*n as i64),
                _ => panic!("Expected int/uint constant in VConst shape uops"),
            })
            .collect(),
        _ => panic!("Expected vectorize or constant for shape uops, got {:?}", uop.op()),
    }
}

/// Extract shape from a UOp (for RESHAPE new_shape, EXPAND new_shape).
fn extract_shape_from_uop(uop: &Arc<UOp>) -> Vec<SInt> {
    match uop.op() {
        Op::Cast { src, .. } | Op::BitCast { src, .. } => extract_shape_from_uop(src),
        Op::Vectorize { elements } => elements
            .iter()
            .map(|elem| match elem.op() {
                Op::Const(cv) => match cv.0 {
                    ConstValue::Int(n) => SInt::Const(n as usize),
                    _ => SInt::Symbolic(Arc::clone(elem)),
                },
                _ => SInt::Symbolic(Arc::clone(elem)),
            })
            .collect(),
        Op::Const(cv) => match cv.0 {
            ConstValue::Int(n) => vec![SInt::Const(n as usize)],
            _ => panic!("Expected int constant for shape"),
        },
        // VConst with empty values = scalar (0-d tensor)
        Op::VConst { values } if values.is_empty() => vec![],
        Op::VConst { values } => values
            .iter()
            .map(|cv| match cv {
                ConstValue::Int(n) => SInt::Const(*n as usize),
                ConstValue::UInt(n) => SInt::Const(*n as usize),
                _ => panic!("Expected int/uint constant in VConst shape"),
            })
            .collect(),
        _ => panic!("Expected vectorize or constant for shape, got {:?}", uop.op()),
    }
}

// ============================================================================
// Range Utilities (from helpers.rs)
// ============================================================================

/// Check if two range lists are pointer-equal (same UOps).
pub fn ranges_equal(ranges1: &[Arc<UOp>], ranges2: &[Arc<UOp>]) -> bool {
    ranges1.len() == ranges2.len() && ranges1.iter().zip(ranges2).all(|(r1, r2)| Arc::ptr_eq(r1, r2))
}

/// Check if two ranges are compatible for merging.
/// Two ranges are compatible if:
/// Check if two range index expressions are compatible (can share the same range).
///
/// Aligned with upstream `all_same()`.
/// Two ranges are compatible only if they are identical (pointer-equal or
/// structurally equal including axis_id). Different REDUCE ranges from
/// different reduction scopes must NOT be considered compatible.
fn ranges_compatible(a: &Arc<UOp>, b: &Arc<UOp>) -> bool {
    Arc::ptr_eq(a, b) || uop_equal(a, b)
}

/// Check if all ranges have identical index expressions (ignoring validity masks).
pub fn all_ranges_same(ranges: &[Arc<UOp>]) -> bool {
    if ranges.is_empty() {
        return true;
    }
    let first_idx = ranges[0].get_idx();
    ranges.iter().skip(1).all(|r| {
        let idx = r.get_idx();
        ranges_compatible(&first_idx, &idx)
    })
}

/// Deep structural equality check for UOps.
pub fn uop_equal(a: &Arc<UOp>, b: &Arc<UOp>) -> bool {
    if Arc::ptr_eq(a, b) {
        return true;
    }
    if std::mem::discriminant(a.op()) != std::mem::discriminant(b.op()) {
        return false;
    }
    if a.dtype() != b.dtype() {
        return false;
    }
    if let (Op::Const(cv_a), Op::Const(cv_b)) = (a.op(), b.op()) {
        return cv_a.0 == cv_b.0;
    }
    if let (
        Op::Range { end: end_a, axis_id: id_a, axis_type: type_a, .. },
        Op::Range { end: end_b, axis_id: id_b, axis_type: type_b, .. },
    ) = (a.op(), b.op())
    {
        return id_a == id_b && type_a == type_b && uop_equal(end_a, end_b);
    }
    let a_srcs = a.op().sources();
    let b_srcs = b.op().sources();
    if a_srcs.len() != b_srcs.len() {
        return false;
    }
    a_srcs.iter().zip(b_srcs.iter()).all(|(sa, sb)| uop_equal(sa, sb))
}

/// Check if range is dead (size ≤ 1). Uses vmax analysis.
pub fn is_dead_axis(range: &Arc<UOp>) -> bool {
    if !matches!(range.op(), Op::Range { .. }) {
        return false;
    }
    match range.vmax() {
        ConstValue::Int(v) => *v <= 0,
        ConstValue::UInt(v) => *v == 0,
        _ => false,
    }
}

/// Check if UOp has no RANGE dependencies.
#[allow(clippy::mutable_key_type)]
pub fn no_range(uop: &Arc<UOp>) -> bool {
    let in_scope_ranges = uop.in_scope_ranges();
    !in_scope_ranges.iter().any(|key| matches!(key.0.op(), Op::Range { .. }))
}

/// Extract RANGE size as i64. Returns None for symbolic ranges.
pub fn range_size_as_i64(range: &Arc<UOp>) -> Option<i64> {
    if let Op::Range { end, .. } = range.op() {
        match end.op() {
            Op::Const(cv) => match cv.0 {
                ConstValue::Int(n) => Some(n),
                ConstValue::UInt(n) => Some(n as i64),
                _ => None,
            },
            _ => None,
        }
    } else {
        None
    }
}

// ============================================================================
// Helpers for patterns (from helpers.rs)
// ============================================================================

/// Check if value is identity for op (Add: 0, Mul: 1, And: -1, Or/Xor: 0).
pub fn is_identity_value(value: &ConstValue, op: &BinaryOp, is_right: bool) -> bool {
    match (op, value) {
        (BinaryOp::Add, ConstValue::Int(0)) => true,
        (BinaryOp::Add, ConstValue::Float(f)) if *f == 0.0 => true,
        (BinaryOp::Sub, ConstValue::Int(0)) if is_right => true,
        (BinaryOp::Sub, ConstValue::Float(f)) if is_right && *f == 0.0 => true,
        (BinaryOp::Mul, ConstValue::Int(1)) => true,
        (BinaryOp::Mul, ConstValue::Float(f)) if *f == 1.0 => true,
        (BinaryOp::Idiv, ConstValue::Int(1)) if is_right => true,
        (BinaryOp::Fdiv, ConstValue::Float(f)) if is_right && *f == 1.0 => true,
        (BinaryOp::Or, ConstValue::Int(0)) => true,
        (BinaryOp::Xor, ConstValue::Int(0)) => true,
        (BinaryOp::And, ConstValue::Int(-1)) => true,
        _ => false,
    }
}

/// Check if value is zero/annihilator for op (Mul: 0, And: 0).
pub fn is_zero_value(value: &ConstValue, op: &BinaryOp) -> bool {
    match (op, value) {
        (BinaryOp::Mul, ConstValue::Int(0)) => true,
        (BinaryOp::Mul, ConstValue::Float(f)) if *f == 0.0 => true,
        (BinaryOp::And, ConstValue::Int(0)) => true,
        _ => false,
    }
}

/// Extract the constant value from a UOp if it's a CONST operation.
pub fn get_const_value(uop: &Arc<UOp>) -> Option<ConstValue> {
    match uop.op() {
        Op::Const(cv) => Some(cv.0),
        _ => None,
    }
}

/// Check if a UOp is a constant with a specific value.
pub fn is_const(uop: &Arc<UOp>, value: &ConstValue) -> bool {
    get_const_value(uop).as_ref() == Some(value)
}

/// Check if a UOp represents a zero-size tensor.
pub fn is_zero_size(uop: &Arc<UOp>) -> bool {
    uop.shape().ok().flatten().map(|shape| shape.iter().any(|dim| matches!(dim, SInt::Const(0)))).unwrap_or(false)
}

/// Check if a dtype is void (used for side-effecting operations).
pub fn is_void(dtype: &DType) -> bool {
    *dtype == DType::Void
}

/// Get the binary operation from a UOp if it's a BINARY operation.
pub fn get_binary_op(uop: &Arc<UOp>) -> Option<BinaryOp> {
    match uop.op() {
        Op::Binary(op, _, _) => Some(*op),
        _ => None,
    }
}

/// Check if a BUFFERIZE operation is for local memory.
pub fn is_local_bufferize(uop: &Arc<UOp>) -> bool {
    if let Op::Bufferize { opts, .. } = uop.op() { opts.addrspace == morok_ir::AddrSpace::Local } else { false }
}

// ============================================================================
// Ending Ranges Helpers (for nested reduction detection)
// ============================================================================

/// Collect all RANGE UOps from an expression tree.
#[allow(clippy::mutable_key_type)]
fn collect_ranges_from_uop(uop: &Arc<UOp>) -> Vec<Arc<UOp>> {
    use std::collections::HashSet;
    let mut ranges = Vec::new();
    let mut seen = HashSet::new();

    for node in uop.toposort() {
        if matches!(node.op(), Op::Range { .. }) {
            let key = UOpKey(Arc::clone(&node));
            if seen.insert(key) {
                ranges.push(node);
            }
        }
    }
    ranges
}

/// Check if UOp is an elementwise operation (matches upstream GroupOp.Elementwise).
fn is_elementwise_op(uop: &Arc<UOp>) -> bool {
    matches!(uop.op(), Op::Binary(..) | Op::Unary(..) | Op::Ternary(..) | Op::Cast { .. } | Op::BitCast { .. })
}
