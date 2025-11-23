//! Range assignment and indexing context for rangeify transformation.
//!
//! This module implements the core range assignment algorithm that determines
//! which ranges (loop indices) each UOp operates on.

use std::collections::HashMap;
use std::rc::Rc;

use morok_dtype::DType;
use morok_ir::{AxisType, ConstValue, Op, SInt, UOp, UOpKey};

use super::helpers;

/// Represents (input_ranges, output_ranges) for a UOp.
///
/// - Input ranges: The ranges used to compute this UOp's value
/// - Output ranges: The ranges this UOp operates on (from consumers)
type UOpRanges = (Vec<Rc<UOp>>, Vec<Rc<UOp>>);

/// Context for range assignment during rangeify.
///
/// Tracks which UOps need to be realized (materialized to buffers) and
/// assigns input/output ranges for each UOp in the graph.
#[derive(Default)]
pub struct IndexingContext {
    /// Maps UOps to their realize status.
    ///
    /// - `None`: Not yet processed or doesn't need realization
    /// - `Some(axes)`: Needs realization on the specified axes
    pub realize_map: HashMap<UOpKey, Option<Vec<usize>>>,

    /// Maps each UOp to its (input_ranges, output_ranges).
    ///
    /// Input ranges: The ranges used to compute this UOp's value
    /// Output ranges: The ranges this UOp operates on (from consumers)
    pub range_map: HashMap<UOpKey, UOpRanges>,

    /// Counter for generating unique range IDs.
    range_idx: usize,
}

impl IndexingContext {
    /// Create a new indexing context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new RANGE UOp with a unique ID.
    ///
    /// If size is 1, returns a constant 0 instead (optimization).
    pub fn new_range(&mut self, size: &SInt, axistype: AxisType) -> Rc<UOp> {
        // Check if size is constant 1
        if let SInt::Const(1) = size {
            return UOp::const_(DType::Index, ConstValue::Int(0));
        }

        // Create range with unique ID
        let range_id = self.range_idx;
        self.range_idx += 1;

        let size_uop = match size {
            SInt::Const(n) => UOp::const_(DType::Index, ConstValue::Int(*n as i64)),
            SInt::Symbolic(uop) => Rc::clone(uop),
        };

        UOp::range_axis(size_uop, range_id, axistype)
    }

    /// Mark a UOp for realization on all axes.
    pub fn mark_realize_all(&mut self, uop: &Rc<UOp>) -> morok_ir::Result<()> {
        if let Some(shape) = uop.shape()? {
            let axes = (0..shape.len()).collect();
            self.realize_map.insert(UOpKey(Rc::clone(uop)), Some(axes));
        }
        Ok(())
    }

    /// Mark a UOp for realization on specific axes.
    pub fn mark_realize(&mut self, uop: &Rc<UOp>, axes: Vec<usize>) {
        self.realize_map.insert(UOpKey(Rc::clone(uop)), Some(axes));
    }

    /// Check if a UOp is in the realize map.
    pub fn should_realize(&self, uop: &Rc<UOp>) -> bool {
        self.realize_map.contains_key(&UOpKey(Rc::clone(uop)))
    }

    /// Get the realize axes for a UOp.
    pub fn get_realize_axes(&self, uop: &Rc<UOp>) -> Option<&Vec<usize>> {
        self.realize_map.get(&UOpKey(Rc::clone(uop))).and_then(|opt| opt.as_ref())
    }

    /// Set the range map for a UOp.
    pub fn set_ranges(&mut self, uop: &Rc<UOp>, input_ranges: Vec<Rc<UOp>>, output_ranges: Vec<Rc<UOp>>) {
        self.range_map.insert(UOpKey(Rc::clone(uop)), (input_ranges, output_ranges));
    }

    /// Get the ranges for a UOp.
    pub fn get_ranges(&self, uop: &Rc<UOp>) -> Option<&UOpRanges> {
        self.range_map.get(&UOpKey(Rc::clone(uop)))
    }

    /// Get the current range counter value.
    pub fn range_counter(&self) -> usize {
        self.range_idx
    }
}

/// Run the range assignment algorithm on a UOp graph.
///
/// This is the core of the rangeify transformation. It:
/// 1. Determines which UOps need to be realized (materialized to buffers)
/// 2. Assigns input/output ranges for each UOp
/// 3. Returns a transformed graph with BUFFERIZE and INDEX operations
///
/// # Algorithm
///
/// Performs a reverse toposort traversal (bottom-up from leaves):
/// - For each UOp, determine output ranges based on consumers
/// - Apply movement op transformations to get input ranges
/// - Create new ranges at realization points
///
/// # Arguments
///
/// * `sink` - The sink UOp representing the entire graph
///
/// # Returns
///
/// A tuple of (transformed_sink, indexing_context)
#[allow(clippy::mutable_key_type)]
pub fn run_rangeify(sink: Rc<UOp>) -> morok_ir::Result<(Rc<UOp>, IndexingContext)> {
    let mut ctx = IndexingContext::new();

    // Step 1: Generate realize map - determine which UOps need materialization
    generate_realize_map(&sink, &mut ctx)?;

    // Step 2: Get reverse toposort and consumer map
    let consumer_map = sink.get_consumer_map();
    let reverse_topo = sink.reverse_toposort(&consumer_map);

    // Step 3: Assign ranges via reverse traversal
    assign_ranges(&reverse_topo, &consumer_map, &mut ctx)?;

    // Step 4: Apply early rewrites (DETACH, CONTIGUOUS_BACKWARD removal)
    let early_matcher = super::patterns::early_rewrites();
    let transformed_sink = crate::rewrite::graph_rewrite(&early_matcher, sink);

    Ok((transformed_sink, ctx))
}

/// Generate the realize map - mark which UOps need to be materialized to buffers.
fn generate_realize_map(sink: &Rc<UOp>, ctx: &mut IndexingContext) -> morok_ir::Result<()> {
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
fn is_always_contiguous(uop: &Rc<UOp>) -> bool {
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

/// Merge ranges from multiple consumers of a UOp.
///
/// When a UOp has multiple consumers with different indexing patterns, this function
/// determines which ranges to assign. For each dimension:
///
/// 1. Extract index and valid components from all consumer ranges
/// 2. Check if all indices are structurally identical (ignoring validity masks)
/// 3. If identical: merge by OR-ing validity masks
/// 4. If different: create new range and mark axis for realization
///
/// # Algorithm
///
/// Based on Tinygrad's multi-consumer range merging (schedule/indexing.py:198-222).
///
/// # Arguments
///
/// * `uop` - The UOp whose consumer ranges are being merged
/// * `consumer_rngs` - List of range lists from each consumer
/// * `ctx` - Indexing context for creating new ranges and tracking realization
///
/// # Returns
///
/// Merged output ranges for the UOp.
///
/// # Example
///
/// ```ignore
/// // x = tensor[10, 20]
/// // Consumer 1: x[i, j]   (same indices)
/// // Consumer 2: x[i, j]   (same indices)
/// // Result: Merge with OR of validity masks, no realization
///
/// // x = tensor[10, 20]
/// // Consumer 1: x[i, j]
/// // Consumer 2: x[i, k]   (different j vs k)
/// // Result: Create new range for dim 1, realize axis 1
/// ```
fn merge_consumer_ranges(
    uop: &Rc<UOp>,
    consumer_rngs: &[Vec<Rc<UOp>>],
    ctx: &mut IndexingContext,
) -> morok_ir::Result<Vec<Rc<UOp>>> {
    // Get shape to know how many dimensions
    let Some(shape) = uop.shape()? else {
        // No shape - return empty ranges (should not happen in practice)
        return Ok(Vec::new());
    };

    let num_dims = shape.len();

    // Transpose: consumer_rngs[consumer_idx][dim_idx] → all_rngs[dim_idx][consumer_idx]
    let mut all_rngs: Vec<Vec<Rc<UOp>>> = vec![Vec::new(); num_dims];
    for consumer_rng in consumer_rngs {
        for (dim_idx, range) in consumer_rng.iter().enumerate() {
            if dim_idx < num_dims {
                all_rngs[dim_idx].push(Rc::clone(range));
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

        // Extract index and valid components
        let indices: Vec<_> = dim_ranges.iter().map(|r| r.get_idx()).collect();
        let valids: Vec<_> = dim_ranges.iter().map(|r| r.get_valid()).collect();

        // Check if all indices are the same
        if helpers::all_ranges_same(&indices) {
            // Compatible - merge validity masks
            let merged_idx = Rc::clone(&indices[0]);

            // OR all validity masks: valid1 | valid2 | ... | validN
            let merged_valid = if valids.len() == 1 {
                Rc::clone(&valids[0])
            } else {
                valids.iter().skip(1).try_fold(Rc::clone(&valids[0]), |acc, v| acc.try_or_op(v))?
            };

            // Check if merged valid is constant true (no validity check needed)
            let merged_range = if let Op::Const(cv) = merged_valid.op()
                && let ConstValue::Bool(true) = cv.0
            {
                // Always valid - use idx directly
                merged_idx
            } else {
                // Wrap index with merged validity: WHERE(merged_valid, merged_idx, INVALID)
                UOp::where_op(merged_valid, merged_idx, UOp::invalid_marker())?
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
    reverse_topo: &[Rc<UOp>],
    consumer_map: &HashMap<UOpKey, Vec<Rc<UOp>>>,
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
        let consumer_rngs: Vec<Vec<Rc<UOp>>> =
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
            Op::ReduceAxis { src, axes, .. } => {
                if let Some(in_shape) = src.shape()? {
                    let mut rngs = out_rngs.clone();
                    // Extend with reduction ranges
                    for (i, s) in in_shape.iter().enumerate() {
                        if axes.contains(&i) && i >= rngs.len() {
                            rngs.push(ctx.new_range(s, AxisType::Reduce));
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
