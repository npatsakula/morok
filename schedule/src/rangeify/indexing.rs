//! Range assignment and indexing context for rangeify transformation.
//!
//! This module provides the core range assignment algorithm that converts
//! movement operations into explicit loop ranges.

use std::collections::HashMap;
use std::sync::Arc;

use morok_dtype::DType;
use morok_ir::{AxisId, AxisType, BinaryOp, ConstValue, Op, SInt, UOp, UOpKey};
use tracing::{debug, info_span, instrument, trace, warn};

// ============================================================================
// Context
// ============================================================================

/// (input_ranges, output_ranges) for a UOp.
type UOpRanges = (Vec<Arc<UOp>>, Vec<Arc<UOp>>);

/// Context for range assignment during rangeify.
#[derive(Default)]
pub struct IndexingContext {
    /// Maps UOps to realize status: Some(axes) = needs realization on axes.
    pub realize_map: HashMap<UOpKey, Option<Vec<usize>>>,
    /// Maps each UOp to its (input_ranges, output_ranges).
    pub range_map: HashMap<UOpKey, UOpRanges>,
    /// Counter for generating unique range IDs.
    range_idx: usize,
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

        let size_uop = match size {
            SInt::Const(n) => UOp::index_const(*n as i64),
            SInt::Symbolic(uop) => Arc::clone(uop),
        };

        UOp::range_axis(size_uop, axis_id, axistype)
    }

    /// Create a new RANGE from an existing UOp end value.
    /// Used when converting REDUCE ranges to LOOP ranges during bufferization.
    /// (Tinygrad rangeify.py:286 - when bufferizing, REDUCE ranges become LOOP)
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
}

// ============================================================================
// Core Algorithm
// ============================================================================

/// Run range assignment on a UOp graph. Returns (transformed_sink, context).
#[allow(clippy::mutable_key_type)]
#[instrument(skip(sink), fields(sink_id = sink.id))]
pub fn run_rangeify(sink: Arc<UOp>) -> morok_ir::Result<(Arc<UOp>, IndexingContext)> {
    let mut ctx = IndexingContext::new();

    // Step 1: Generate realize map - determine which UOps need materialization
    generate_realize_map(&sink, &mut ctx)?;

    // Step 2: Get toposort (root-to-leaves) and consumer map
    let consumer_map = sink.get_consumer_map();

    // Use forward toposort (root first) for range propagation
    let forward_topo: Vec<_> = sink.toposort().into_iter().rev().collect();

    // Step 3: Assign ranges via forward traversal
    assign_ranges(&forward_topo, &consumer_map, &mut ctx)?;

    // Step 4: Apply early rewrites (DETACH, CONTIGUOUS_BACKWARD removal)
    let early_matcher = super::patterns::early_rewrites();
    let transformed_sink = crate::rewrite::graph_rewrite_top_down(&early_matcher, sink, &mut ());

    Ok((transformed_sink, ctx))
}

/// Generate the realize map - mark which UOps need to be materialized to buffers.
#[instrument(skip(sink, ctx))]
fn generate_realize_map(sink: &Arc<UOp>, ctx: &mut IndexingContext) -> morok_ir::Result<()> {
    for node in sink.toposort() {
        trace!(node_id = node.id, op = ?std::mem::discriminant(node.op()), "processing node");
        match node.op() {
            Op::Sink { sources } => {
                for src in sources {
                    if !is_always_contiguous(src) {
                        ctx.mark_realize_all(src)?;
                    }
                }
            }
            Op::Copy { .. } | Op::Contiguous { .. } => {
                ctx.mark_realize_all(&node)?;
            }
            Op::MStack { buffers } => {
                for buf in buffers {
                    if !is_always_contiguous(buf) {
                        ctx.mark_realize_all(buf)?;
                    }
                }
            }
            // NOTE: ReduceAxis does NOT get unconditional realization.
            // Tinygrad only realizes ReduceAxis when:
            // 1. It has ending_ranges (nested reduction detection)
            // 2. Consumer range conflicts (handled by merge_consumer_ranges)
            // 3. Being a SINK source (handled above)
            _ => {}
        }
    }
    Ok(())
}

/// Check if a UOp is always contiguous (doesn't need realization).
fn is_always_contiguous(uop: &Arc<UOp>) -> bool {
    matches!(
        uop.op(),
        Op::Contiguous { .. }
            | Op::Copy { .. }
            | Op::Buffer { .. }
            | Op::BufferView { .. }
            | Op::Const(_)
            | Op::Device(_)
            | Op::MSelect { .. }
            | Op::MStack { .. }
            | Op::DefineGlobal(_)
            | Op::DefineLocal(_)
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

    for (dim_idx, dim_ranges) in all_rngs.iter().enumerate() {
        if dim_ranges.is_empty() {
            out_rngs.push(ctx.new_range(&shape[dim_idx], AxisType::Loop));
            realize_axes.push(dim_idx);
            continue;
        }

        // FAST PATH: If all ranges are pointer-equal, return original unchanged
        if dim_ranges.iter().skip(1).all(|r| Arc::ptr_eq(&dim_ranges[0], r)) {
            out_rngs.push(Arc::clone(&dim_ranges[0]));
            continue;
        }

        // Debug: show which ranges are being compared
        debug!(
            dim_idx = dim_idx,
            range_ids = ?dim_ranges.iter().map(|r| r.id).collect::<Vec<_>>(),
            range_ops = ?dim_ranges.iter().map(|r| format!("{:?}", r.op())).collect::<Vec<_>>(),
            "merge_consumer_ranges: ranges differ at dimension"
        );

        let indices: Vec<_> = dim_ranges.iter().map(|r| r.get_idx()).collect();
        let valids: Vec<_> = dim_ranges.iter().map(|r| r.get_valid()).collect();

        if all_ranges_same(&indices) {
            let merged_idx = Arc::clone(&indices[0]);
            let merged_valid = if valids.len() == 1 {
                Arc::clone(&valids[0])
            } else {
                valids.iter().skip(1).try_fold(Arc::clone(&valids[0]), |acc, v| acc.try_or_op(v))?
            };

            let merged_range = if is_const_true(&merged_valid) {
                merged_idx
            } else {
                UOp::try_where(merged_valid, merged_idx, UOp::invalid_marker())?
            };

            out_rngs.push(merged_range);
        } else {
            out_rngs.push(ctx.new_range(&shape[dim_idx], AxisType::Loop));
            realize_axes.push(dim_idx);
        }
    }

    if !realize_axes.is_empty() {
        warn!(realize_axes = ?realize_axes, "range conflict detected - marking for realization");
        ctx.mark_realize(uop, realize_axes);
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
) -> morok_ir::Result<()> {
    // Local variable for ending_ranges - only used within this function (like Tinygrad)
    let mut ending_ranges: HashMap<UOpKey, Vec<Arc<UOp>>> = HashMap::new();

    for x in reverse_topo {
        if matches!(x.op(), Op::Device(_) | Op::Unique(_)) {
            continue;
        }

        let _span = info_span!("assign_range",
            uop_id = x.id,
            op = ?std::mem::discriminant(x.op())
        )
        .entered();

        let consumers: Vec<_> = consumer_map.get(&UOpKey(x.clone())).cloned().unwrap_or_default();
        let consumer_rngs: Vec<Vec<Arc<UOp>>> =
            consumers.iter().filter_map(|c| ctx.get_ranges(c).map(|(inp, _)| inp.clone())).collect();

        debug!(
            num_consumers = consumers.len(),
            consumer_rngs_len = consumer_rngs.len(),
            consumer_ids = ?consumers.iter().map(|c| c.id).collect::<Vec<_>>(),
            "Consumer info"
        );

        // Inherit ending_ranges from consumers (like Tinygrad line 173)
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
            if let Some(shape) = x.shape()? {
                let rngs: Vec<_> = shape.iter().map(|s| ctx.new_range(s, AxisType::Loop)).collect();
                let axes: Vec<usize> = (0..shape.len()).collect();
                ctx.realize_map.insert(UOpKey(x.clone()), Some(axes));
                // Clear ending_ranges when realized (like Tinygrad line 185)
                ending_ranges.insert(UOpKey(x.clone()), Vec::new());
                rngs
            } else {
                continue;
            }
        } else if let Op::Bufferize { ranges, .. } = x.op() {
            ranges.to_vec()
        } else if let Op::ReduceAxis { .. } = x.op() {
            // Use the ReduceAxis's OUTPUT shape, not input shape.
            // For keepdim=true, output shape is [1], not [] - we must use actual output shape.
            // Tinygrad inherits output ranges from consumers or creates based on output shape.
            if consumer_rngs.is_empty() {
                if let Some(out_shape) = x.shape()? {
                    out_shape.iter().map(|s| ctx.new_range(s, AxisType::Loop)).collect()
                } else {
                    continue;
                }
            } else if consumer_rngs.len() == 1 {
                consumer_rngs[0].clone()
            } else {
                merge_consumer_ranges(x, &consumer_rngs, ctx)?
            }
        } else if consumer_rngs.is_empty() {
            continue;
        } else if consumer_rngs.len() == 1 {
            consumer_rngs[0].clone()
        } else {
            merge_consumer_ranges(x, &consumer_rngs, ctx)?
        };

        debug!(should_realize = ctx.should_realize(x), out_rngs_len = out_rngs.len(), "output ranges computed");

        // Check ending_ranges FIRST (before in_rngs computation)
        // Tinygrad lines 224-234: ending_ranges realization happens BEFORE input ranges
        // This is critical: in_rngs must be computed from the FINAL out_rngs after realization
        let ending = ending_ranges.get(&UOpKey(x.clone())).cloned().unwrap_or_default();
        if !ending.is_empty() {
            debug!(
                ending_count = ending.len(),
                triggers_realization = matches!(x.op(), Op::ReduceAxis { .. }) || is_elementwise_op(x),
                "Ending ranges detected (pre-in_rngs check)"
            );
        }
        // Filter ending ranges: for ReduceAxis, ignore REDUCE ranges from other reductions
        // since they're context-specific and shouldn't trigger realization here.
        let filtered_ending: Vec<_> =
            if matches!(x.op(), Op::ReduceAxis { .. }) {
                ending
                    .iter()
                    .filter(|e| {
                        if let Op::Range { axis_type, .. } = e.op() { *axis_type != AxisType::Reduce } else { true }
                    })
                    .cloned()
                    .collect()
            } else {
                ending.clone()
            };

        if !filtered_ending.is_empty() && (matches!(x.op(), Op::ReduceAxis { .. }) || is_elementwise_op(x)) {
            if let Some(shape) = x.shape().ok().flatten() {
                // Start with existing realize_axes (from merge_consumer_ranges)
                let mut realize_axes: Vec<usize> = ctx.get_realize_axes(x).cloned().unwrap_or_default();

                // Get axis_ids from ending ranges for comparison (matching Tinygrad's rr.arg > e.arg)
                let ending_axis_ids: Vec<AxisId> = filtered_ending
                    .iter()
                    .filter_map(|e| if let Op::Range { axis_id, .. } = e.op() { Some(*axis_id) } else { None })
                    .collect();

                // For each axis, check if any range in out_rngs has axis_id > ending axis_id
                // This matches Tinygrad's: any(any(rr.arg > e.arg for e in ending_ranges[x]) for rr in r.ranges)
                for (i, r) in out_rngs.iter().enumerate() {
                    if realize_axes.contains(&i) {
                        continue;
                    }

                    // Check ranges in this output expression
                    let should_realize = r.ranges().iter().any(|rr| {
                        if let Op::Range { axis_id, .. } = rr.op() {
                            ending_axis_ids.iter().any(|e_id| axis_id > e_id)
                        } else {
                            false
                        }
                    });

                    if should_realize {
                        realize_axes.push(i);
                    }
                }

                debug!(
                    node_id = x.id,
                    op = ?std::mem::discriminant(x.op()),
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
                                ctx.new_range_uncollapsed(&shape[i], AxisType::Loop)
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
                    apply_movement_op(x.op(), in_shape, &out_rngs)
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

        // EXPAND marks ranges as ending when broadcasting to static dimensions (Tinygrad lines 249-252)
        // "if the EXPAND is used to inject a range, we don't mark it as ending_ranges. otherwise we do."
        if let Op::Expand { new_shape, .. } = x.op() {
            // Check if new_shape is all static (no RANGE ops being injected in the shape)
            let shape_is_static = extract_shape_from_uop(new_shape).iter().all(|s| matches!(s, SInt::Const(_)));

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
                let mut changed_ranges: Vec<Arc<UOp>> = Vec::new();
                for (inp, out) in in_rngs.iter().zip(out_rngs.iter()) {
                    if !Arc::ptr_eq(inp, out) {
                        // The output range is being collapsed - collect any RANGE ops in it
                        // Filter out REDUCE ranges - they're internal to reductions and shouldn't
                        // propagate as ending ranges to other operations.
                        let ranges = collect_ranges_from_uop(out);
                        for r in ranges {
                            if let Op::Range { axis_type, .. } = r.op()
                                && *axis_type != AxisType::Reduce
                            {
                                changed_ranges.push(r);
                            }
                        }
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
pub fn apply_movement_op(op: &Op, in_shape: &[SInt], rngs: &[Arc<UOp>]) -> Vec<Arc<UOp>> {
    match op {
        Op::Shrink { begins, .. } => {
            let begin_vals = extract_shape_values(begins);
            rngs.iter()
                .zip(begin_vals.iter())
                .map(|(rng, &begin)| {
                    if begin == 0 {
                        Arc::clone(rng)
                    } else {
                        let begin_uop = UOp::index_const(begin as i64);
                        rng.try_add(&begin_uop).unwrap()
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
                    let shape_minus_1 = match shape {
                        SInt::Const(n) => UOp::index_const(*n as i64 - 1),
                        SInt::Symbolic(uop) => {
                            let one = UOp::index_const(1);
                            uop.try_sub(&one).unwrap()
                        }
                    };
                    shape_minus_1.try_sub(rng).unwrap()
                }
            })
            .collect(),

        Op::Expand { new_shape, .. } => {
            let new_shape_vals = extract_shape_from_uop(new_shape);
            rngs.iter()
                .zip(in_shape.iter())
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
            let begin_vals = extract_shape_values(begin_pads);
            let end_vals = extract_shape_values(end_pads);
            rngs.iter()
                .zip(in_shape.iter())
                .zip(begin_vals.iter().zip(end_vals.iter()))
                .map(|((rng, shape), (&begin, &end))| {
                    if begin == 0 && end == 0 {
                        return Arc::clone(rng);
                    }
                    let begin_uop = UOp::index_const(begin as i64);
                    let shape_plus_begin = match shape {
                        SInt::Const(n) => UOp::index_const(*n as i64 + begin as i64),
                        SInt::Symbolic(uop) => uop.try_add(&begin_uop).unwrap(),
                    };
                    let too_low = rng.try_cmplt(&begin_uop).unwrap();
                    let true_val = UOp::const_(DType::Bool, ConstValue::Bool(true));
                    let valid_low = too_low.try_xor_op(&true_val).unwrap();
                    let valid_high = rng.try_cmplt(&shape_plus_begin).unwrap();
                    let valid = valid_low.try_and_op(&valid_high).unwrap();
                    let adjusted_rng = rng.try_sub(&begin_uop).unwrap();
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

            // Flatten output indices
            let mut acc = UOp::index_const(1);
            let mut axes_in = Vec::new();

            trace!(
                new_shape = ?new_shape_vals,
                rngs.len = rngs.len(),
                "Reshape flatten start"
            );

            for (shape_dim, rng) in new_shape_vals.iter().zip(rngs.iter()).rev() {
                trace!(
                    shape_dim = ?shape_dim,
                    rng.id = rng.id,
                    rng.op = ?std::mem::discriminant(rng.op()),
                    acc.op = ?std::mem::discriminant(acc.op()),
                    "Reshape flatten iteration"
                );
                let weighted = acc.try_mul(rng).unwrap();
                trace!(
                    weighted.id = weighted.id,
                    weighted.op = ?std::mem::discriminant(weighted.op()),
                    "Reshape flatten weighted"
                );
                axes_in.push(weighted);
                acc = match shape_dim {
                    SInt::Const(n) => {
                        let n_uop = UOp::index_const(*n as i64);
                        let new_acc = acc.try_mul(&n_uop).unwrap();
                        trace!(
                            multiplier = n,
                            new_acc.op = ?std::mem::discriminant(new_acc.op()),
                            "Reshape flatten acc update"
                        );
                        new_acc
                    }
                    SInt::Symbolic(uop) => acc.try_mul(uop).unwrap(),
                };
            }
            let combined_axes =
                axes_in.into_iter().reduce(|a, b| a.try_add(&b).unwrap()).unwrap_or_else(|| UOp::index_const(0));

            trace!(combined_axes.id = combined_axes.id, "reshape flatten combined");

            // Unflatten into input shape dimensions
            // Apply range-based simplification:
            //   x % n → x when 0 <= vmin(x) && vmax(x) < n
            //   x / n → 0 when 0 <= vmin(x) && vmax(x) < n
            use morok_ir::types::ConstValue;
            use morok_ir::uop::cached_property::CachedProperty;
            use morok_ir::uop::properties::VminVmaxProperty;

            fn simplify_mod(x: &Arc<UOp>, n: i64) -> Arc<UOp> {
                let (vmin, vmax) = VminVmaxProperty::get(x);
                if let (ConstValue::Int(min), ConstValue::Int(max)) = (vmin, vmax)
                    && *min >= 0
                    && *max < n
                {
                    // x is always in range [0, n), so x % n = x
                    tracing::trace!(n, min, max, "simplify_mod: SIMPLIFIED to identity");
                    return Arc::clone(x);
                }
                tracing::trace!(n, ?vmin, ?vmax, x.tree = %x.tree(), "simplify_mod: NOT simplified");
                let n_uop = UOp::index_const(n);
                x.try_mod(&n_uop).unwrap()
            }

            fn simplify_div(x: &Arc<UOp>, n: i64) -> Arc<UOp> {
                let (vmin, vmax) = VminVmaxProperty::get(x);
                if let (ConstValue::Int(min), ConstValue::Int(max)) = (vmin, vmax)
                    && *min >= 0
                    && *max < n
                    && n > 0
                {
                    // x is always in range [0, n), so x / n = 0
                    return UOp::index_const(0);
                }
                let n_uop = UOp::index_const(n);
                x.try_div(&n_uop).unwrap()
            }

            let mut axes_out = Vec::new();
            let mut combined = combined_axes;
            for shape_dim in in_shape.iter().rev() {
                match shape_dim {
                    SInt::Const(n) => {
                        let mod_result = simplify_mod(&combined, *n as i64);
                        axes_out.push(mod_result);
                        combined = simplify_div(&combined, *n as i64);
                    }
                    SInt::Symbolic(uop) => {
                        let mod_result = combined.try_mod(uop).unwrap();
                        axes_out.push(mod_result);
                        combined = combined.try_div(uop).unwrap();
                    }
                }
            }
            axes_out.reverse();

            // Apply symbolic simplification to the output ranges (bottom-up to ensure children are simplified first)
            // This is critical for simplifying Range(n) % n → Range(n) and Range(n) / n → 0
            // Like Tinygrad: graph_rewrite(UOp.sink(*axes_out), symbolic+pm_simplify_valid, name="reshape").src
            use crate::symbolic::patterns::symbolic_simple;
            use morok_ir::rewrite::graph_rewrite_bottom_up;

            let sink = UOp::sink(axes_out);
            let simplified_sink = graph_rewrite_bottom_up(&symbolic_simple(), sink, &mut ());

            // Extract simplified sources
            match simplified_sink.op() {
                Op::Sink { sources } => sources.iter().cloned().collect(),
                _ => vec![simplified_sink],
            }
        }

        _ => panic!("apply_movement_op called with non-movement op: {:?}", op),
    }
}

/// Extract shape values from a UOp (for SHRINK begins/ends, PAD pads).
fn extract_shape_values(uop: &Arc<UOp>) -> Vec<usize> {
    match uop.op() {
        Op::Vectorize { elements } => elements
            .iter()
            .map(|elem| match elem.op() {
                Op::Const(cv) => match cv.0 {
                    ConstValue::Int(n) => n as usize,
                    _ => panic!("Expected int constant in vectorize"),
                },
                _ => panic!("Expected constant element in vectorize"),
            })
            .collect(),
        Op::Const(cv) => match cv.0 {
            ConstValue::Int(n) => vec![n as usize],
            _ => panic!("Expected int constant"),
        },
        _ => panic!("Expected vectorize or constant for shape values, got {:?}", uop.op()),
    }
}

/// Extract shape from a UOp (for RESHAPE new_shape, EXPAND new_shape).
fn extract_shape_from_uop(uop: &Arc<UOp>) -> Vec<SInt> {
    match uop.op() {
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
        _ => panic!("Expected vectorize or constant for shape, got {:?}", uop.op()),
    }
}

/// Compute inverse permutation (argsort).
fn argsort(perm: &[usize]) -> Vec<usize> {
    let mut inv = vec![0; perm.len()];
    for (i, &p) in perm.iter().enumerate() {
        inv[p] = i;
    }
    inv
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
/// 1. They are pointer-equal
/// 2. They are structurally equal (same UOp graph)
/// 3. They are both REDUCE ranges with the same `end` value (different axis_ids are OK)
fn ranges_compatible(a: &Arc<UOp>, b: &Arc<UOp>) -> bool {
    if Arc::ptr_eq(a, b) {
        return true;
    }
    if uop_equal(a, b) {
        return true;
    }
    // Special case: Two REDUCE ranges with same end value are compatible
    // (they represent iteration over the same axis dimension, just from different contexts)
    match (a.op(), b.op()) {
        (
            Op::Range { axis_type: AxisType::Reduce, end: end_a, .. },
            Op::Range { axis_type: AxisType::Reduce, end: end_b, .. },
        ) => Arc::ptr_eq(end_a, end_b) || uop_equal(end_a, end_b),
        _ => false,
    }
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
        Op::Range { end: end_a, axis_id: id_a, axis_type: type_a },
        Op::Range { end: end_b, axis_id: id_b, axis_type: type_b },
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

/// Check if op should always run (CONTIGUOUS, COPY, ASSIGN, NOOP).
pub fn is_always_run_op(op: &Op) -> bool {
    matches!(op, Op::Contiguous { .. } | Op::Copy { .. } | Op::Assign { .. } | Op::Noop)
}

/// Check if op is cheap to inline (CONST, Unary, Binary, Ternary, Cast, Gep, Vectorize).
pub fn is_cheap_to_inline(op: &Op) -> bool {
    matches!(
        op,
        Op::Const(_)
            | Op::Unique(_)
            | Op::Device(_)
            | Op::Noop
            | Op::DefineVar { .. }
            | Op::DefineReg { .. }
            | Op::VConst { .. }
            // Note: Op::Unary excluded - needs buffering for reduce sources
            | Op::Binary(..)
            | Op::Ternary(..)
            | Op::Cast { .. }
            | Op::BitCast { .. }
            | Op::Gep { .. }
            | Op::Vectorize { .. }
            | Op::PointerIndex { .. }
    )
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

/// Check if UOp is an elementwise operation.
fn is_elementwise_op(uop: &Arc<UOp>) -> bool {
    matches!(uop.op(), Op::Binary(..) | Op::Unary(..) | Op::Ternary(..) | Op::Cast { .. })
}
