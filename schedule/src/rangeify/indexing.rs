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

        UOp::new(Op::Range { end: size_uop, axis_id: range_id, axis_type: axistype }, DType::Index)
    }

    /// Mark a UOp for realization on all axes.
    pub fn mark_realize_all(&mut self, uop: &Rc<UOp>) {
        if let Some(shape) = uop.shape() {
            let axes = (0..shape.len()).collect();
            self.realize_map.insert(UOpKey(Rc::clone(uop)), Some(axes));
        }
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
pub fn run_rangeify(sink: Rc<UOp>) -> (Rc<UOp>, IndexingContext) {
    let mut ctx = IndexingContext::new();

    // Step 1: Generate realize map - determine which UOps need materialization
    generate_realize_map(&sink, &mut ctx);

    // Step 2: Get reverse toposort and consumer map
    let consumer_map = sink.get_consumer_map();
    let reverse_topo = sink.reverse_toposort(&consumer_map);

    // Step 3: Assign ranges via reverse traversal
    assign_ranges(&reverse_topo, &consumer_map, &mut ctx);

    // Step 4: Apply transformations (convert to BUFFERIZE/INDEX)
    let transformed_sink = apply_rangeify_transform(sink, &ctx);

    (transformed_sink, ctx)
}

/// Generate the realize map - mark which UOps need to be materialized to buffers.
fn generate_realize_map(sink: &Rc<UOp>, ctx: &mut IndexingContext) {
    // Traverse graph and mark realization points
    for node in sink.toposort() {
        match node.op() {
            // Always realize SINK sources
            Op::Sink { sources } => {
                for src in sources {
                    if !is_always_contiguous(src) {
                        ctx.mark_realize_all(src);
                    }
                }
            }

            // Always realize these operations
            Op::Copy { .. } | Op::Contiguous { .. } => {
                ctx.mark_realize_all(&node);
            }

            // Realize sources of these operations
            Op::MStack { buffers } => {
                for buf in buffers {
                    if !is_always_contiguous(buf) {
                        ctx.mark_realize_all(buf);
                    }
                }
            }

            _ => {}
        }
    }
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

/// Assign input/output ranges for each UOp via reverse toposort traversal.
#[allow(clippy::mutable_key_type)]
fn assign_ranges(reverse_topo: &[Rc<UOp>], consumer_map: &HashMap<UOpKey, Vec<Rc<UOp>>>, ctx: &mut IndexingContext) {
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
            if let Some(shape) = x.shape() {
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
            // For now, just take ranges from first consumer
            // TODO: Implement proper range merging logic
            consumer_rngs[0].clone()
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
                if let Some(in_shape) = src.shape() {
                    helpers::apply_movement_op(x.op(), in_shape, &out_rngs)
                } else {
                    out_rngs.clone()
                }
            }

            // REDUCE_AXIS creates ranges for reduction axes
            Op::ReduceAxis { src, axes, .. } => {
                if let Some(in_shape) = src.shape() {
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
}

/// Apply the rangeify transformation to convert movement ops to BUFFERIZE/INDEX.
///
/// NOTE: This is now a no-op. The actual transformation happens in transform::rangeify()
/// via pattern matching after range assignment is complete.
fn apply_rangeify_transform(sink: Rc<UOp>, _ctx: &IndexingContext) -> Rc<UOp> {
    // Transformation is now handled by pattern matching in transform::rangeify()
    // This function is kept for backward compatibility but does nothing
    sink
}
