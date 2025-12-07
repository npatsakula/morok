//! Range assignment and indexing context for rangeify transformation.

use std::collections::HashMap;
use std::sync::Arc;

use morok_ir::{AxisId, AxisType, BinaryOp, ConstValue, Op, SInt, UOp, UOpKey};

use super::helpers;

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

        // Create range with Unrenumbered axis_id
        let axis_id = AxisId::Unrenumbered(self.range_idx);
        self.range_idx += 1;

        let size_uop = match size {
            SInt::Const(n) => UOp::index_const(*n as i64),
            SInt::Symbolic(uop) => Arc::clone(uop),
        };

        UOp::range_axis(size_uop, axis_id, axistype)
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

/// Run range assignment on a UOp graph. Returns (transformed_sink, context).
#[allow(clippy::mutable_key_type)]
pub fn run_rangeify(sink: Arc<UOp>) -> morok_ir::Result<(Arc<UOp>, IndexingContext)> {
    let mut ctx = IndexingContext::new();

    // Step 1: Generate realize map - determine which UOps need materialization
    generate_realize_map(&sink, &mut ctx)?;

    // Step 2: Get toposort (root-to-leaves) and consumer map
    // We need forward traversal so that BUFFERIZE is processed first,
    // then its compute (ADD), then the inputs (RESHAPE, BUFFER).
    // This allows ranges to propagate from BUFFERIZE → ADD → RESHAPE.
    let consumer_map = sink.get_consumer_map();

    // Use forward toposort (root first) for range propagation
    // The sink.toposort() gives leaves-to-root, so we reverse it
    let forward_topo: Vec<_> = sink.toposort().into_iter().rev().collect();

    // Step 3: Assign ranges via forward traversal
    assign_ranges(&forward_topo, &consumer_map, &mut ctx)?;

    // Step 4: Apply early rewrites (DETACH, CONTIGUOUS_BACKWARD removal)
    let early_matcher = super::patterns::early_rewrites();
    let transformed_sink = crate::rewrite::graph_rewrite(&early_matcher, sink, &mut ());

    Ok((transformed_sink, ctx))
}

/// Generate the realize map - mark which UOps need to be materialized to buffers.
fn generate_realize_map(sink: &Arc<UOp>, ctx: &mut IndexingContext) -> morok_ir::Result<()> {
    // Traverse graph and mark realization points
    for node in sink.toposort() {
        match node.op() {
            // Always realize SINK sources
            Op::Sink { sources } => {
                for src in sources {
                    if !is_always_contiguous(src) {
                        ctx.mark_realize_all(src)?;
                    }
                }
            }

            // Always realize these operations
            Op::Copy { .. } | Op::Contiguous { .. } => {
                ctx.mark_realize_all(&node)?;
            }

            // Realize sources of these operations
            Op::MStack { buffers } => {
                for buf in buffers {
                    if !is_always_contiguous(buf) {
                        ctx.mark_realize_all(buf)?;
                    }
                }
            }

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
///
/// This is needed because `try_or_op(const(true), const(true))` creates
/// `Binary(Or, true, true)` without simplification, not `Const(true)`.
fn is_const_true(uop: &Arc<UOp>) -> bool {
    match uop.op() {
        Op::Const(cv) => matches!(cv.0, ConstValue::Bool(true)),
        // Handle unsimplified true | true = true
        Op::Binary(BinaryOp::Or, a, b) => is_const_true(a) && is_const_true(b),
        _ => false,
    }
}

/// Merge ranges from multiple consumers. Creates new ranges and marks realization when needed.
///
/// This function handles multi-consumer scenarios where different consumers may have
/// different ranges for the same dimension. It:
/// 1. Merges compatible ranges (same indices) by OR-ing validity masks
/// 2. Creates new ranges for incompatible dimensions and marks them for realization
///
/// Returns the merged ranges and updates the context's realize_map if needed.
pub(crate) fn merge_consumer_ranges(
    uop: &Arc<UOp>,
    consumer_rngs: &[Vec<Arc<UOp>>],
    ctx: &mut IndexingContext,
) -> morok_ir::Result<Vec<Arc<UOp>>> {
    // Get shape to know how many dimensions
    let Some(shape) = uop.shape()? else {
        // No shape - return empty ranges (should not happen in practice)
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

    // Process each dimension
    for (dim_idx, dim_ranges) in all_rngs.iter().enumerate() {
        if dim_ranges.is_empty() {
            // No ranges for this dimension - create new one
            out_rngs.push(ctx.new_range(&shape[dim_idx], AxisType::Loop));
            realize_axes.push(dim_idx);
            continue;
        }

        // FAST PATH: If all ranges are pointer-equal, return original unchanged
        // This preserves Arc identity for tests like test_merge_consumer_ranges_identical_1d
        if dim_ranges.iter().skip(1).all(|r| Arc::ptr_eq(&dim_ranges[0], r)) {
            out_rngs.push(Arc::clone(&dim_ranges[0]));
            continue;
        }

        // Extract index and valid components
        let indices: Vec<_> = dim_ranges.iter().map(|r| r.get_idx()).collect();
        let valids: Vec<_> = dim_ranges.iter().map(|r| r.get_valid()).collect();

        // Check if all indices are the same
        if helpers::all_ranges_same(&indices) {
            // Compatible - merge validity masks
            let merged_idx = Arc::clone(&indices[0]);

            // OR all validity masks: valid1 | valid2 | ... | validN
            let merged_valid = if valids.len() == 1 {
                Arc::clone(&valids[0])
            } else {
                valids.iter().skip(1).try_fold(Arc::clone(&valids[0]), |acc, v| acc.try_or_op(v))?
            };

            // Check if merged valid is constant true (no validity check needed)
            // Uses is_const_true to handle unsimplified true | true expressions
            let merged_range = if is_const_true(&merged_valid) {
                // Always valid - use idx directly
                merged_idx
            } else {
                // Wrap index with merged validity: WHERE(merged_valid, merged_idx, INVALID)
                UOp::try_where(merged_valid, merged_idx, UOp::invalid_marker())?
            };

            out_rngs.push(merged_range);
        } else {
            // Incompatible - create new range and mark for realization
            out_rngs.push(ctx.new_range(&shape[dim_idx], AxisType::Loop));
            realize_axes.push(dim_idx);
        }
    }

    // Update realize map with axes that need materialization
    if !realize_axes.is_empty() {
        ctx.mark_realize(uop, realize_axes);
    }

    Ok(out_rngs)
}

/// Assign input/output ranges for each UOp via reverse toposort traversal.
#[allow(clippy::mutable_key_type)]
fn assign_ranges(
    reverse_topo: &[Arc<UOp>],
    consumer_map: &HashMap<UOpKey, Vec<Arc<UOp>>>,
    ctx: &mut IndexingContext,
) -> morok_ir::Result<()> {
    for x in reverse_topo {
        // Skip certain ops
        if matches!(x.op(), Op::Device(_) | Op::Unique(_)) {
            continue;
        }

        // Get consumers for this UOp
        let consumers: Vec<_> = consumer_map.get(&UOpKey(x.clone())).cloned().unwrap_or_default();

        // Collect consumer ranges
        let consumer_rngs: Vec<Vec<Arc<UOp>>> =
            consumers.iter().filter_map(|c| ctx.get_ranges(c).map(|(inp, _)| inp.clone())).collect();

        // Determine output ranges
        let out_rngs = if ctx.should_realize(x) {
            // Create new ranges for realized ops
            if let Some(shape) = x.shape()? {
                let rngs: Vec<_> = shape.iter().map(|s| ctx.new_range(s, AxisType::Loop)).collect();

                // Mark all axes as realized
                let axes: Vec<usize> = (0..shape.len()).collect();
                ctx.realize_map.insert(UOpKey(x.clone()), Some(axes));

                rngs
            } else {
                continue;
            }
        } else if let Op::Bufferize { ranges, .. } = x.op() {
            // BUFFERIZE has no shape but already contains ranges
            // Use those ranges directly as output ranges
            // This allows range propagation to the compute inside BUFFERIZE
            ranges.to_vec()
        } else if consumer_rngs.is_empty() {
            // No consumers have ranges
            continue;
        } else if consumer_rngs.len() == 1 {
            // Single consumer - inherit ranges
            consumer_rngs[0].clone()
        } else {
            // Multiple consumers - merge ranges
            merge_consumer_ranges(x, &consumer_rngs, ctx)?
        };

        // Determine input ranges by applying movement op transformations
        let in_rngs = match x.op() {
            // Movement ops transform ranges
            Op::Reshape { src, .. }
            | Op::Permute { src, .. }
            | Op::Expand { src, .. }
            | Op::Pad { src, .. }
            | Op::Shrink { src, .. }
            | Op::Flip { src, .. } => {
                if let Some(in_shape) = src.shape()? {
                    helpers::apply_movement_op(x.op(), in_shape, &out_rngs)
                } else {
                    out_rngs.clone()
                }
            }

            // REDUCE_AXIS creates ranges for reduction axes
            //
            // The source has shape [d0, d1, ..., dn] and the output has reduced dims.
            // For each axis in the source:
            // - If it's a reduce axis: create a new REDUCE-type range
            // - If it's not a reduce axis: use the corresponding output range
            //
            // Note: out_rngs comes from downstream RESHAPE (e.g., from remove_singleton_dims
            // when keepdim=false). Since morok always uses keepdim=true internally then RESHAPE,
            // out_rngs has the same number of dims as the REDUCE_AXIS output. Each position i
            // in the source corresponds to position i in out_rngs.
            //
            // Follows Tinygrad's approach (indexing.py:256):
            //   rngs = tuple(new_range(s, REDUCE) if i in axes else r for i,(r,s) in enumerate(zip(rngs, shape)))
            Op::ReduceAxis { src, axes, .. } => {
                if let Some(in_shape) = src.shape()? {
                    let mut rngs = Vec::with_capacity(in_shape.len());

                    for (i, s) in in_shape.iter().enumerate() {
                        if axes.contains(&i) {
                            // Reduce axis: create new REDUCE-type range
                            rngs.push(ctx.new_range(s, AxisType::Reduce));
                        } else {
                            // Non-reduce axis: use output range at same position
                            // Position i in source maps to position i in out_rngs (keepdim=true)
                            if i < out_rngs.len() {
                                rngs.push(Arc::clone(&out_rngs[i]));
                            } else {
                                // Fallback: create LOOP range
                                rngs.push(ctx.new_range(s, AxisType::Loop));
                            }
                        }
                    }
                    rngs
                } else {
                    out_rngs.clone()
                }
            }

            // All other ops pass through ranges unchanged
            _ => out_rngs.clone(),
        };

        // Store the range mapping
        ctx.set_ranges(x, in_rngs, out_rngs);
    }
    Ok(())
}
