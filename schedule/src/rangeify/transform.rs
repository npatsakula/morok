//! Transform helpers for converting movement ops to BUFFERIZE + INDEX.
//!
//! This module provides the core transformation functions that convert
//! movement operations into explicit buffer materialization and indexing.

use std::rc::Rc;

use morok_ir::{AddrSpace, BufferizeOpts, Op, UOp, UOpKey};

use super::indexing::IndexingContext;

/// Transform a UOp's sources by adding BUFFERIZE + INDEX where needed.
///
/// This is the core function that converts movement ops to explicit indexing.
/// It processes each source of a UOp and:
/// 1. Adds INDEX to buffer-like sources
/// 2. Wraps realizable sources in BUFFERIZE + INDEX
/// 3. Passes through other sources unchanged
///
/// Returns `Some(new_sources)` if any source was transformed, `None` otherwise.
pub fn transform_sources_with_bufferize(x: &Rc<UOp>, ctx: &IndexingContext) -> Option<Vec<Rc<UOp>>> {
    // Skip already processed ops
    if matches!(x.op(), Op::Bufferize { .. } | Op::Index { .. } | Op::After { .. }) {
        return None;
    }

    let sources = x.op().sources();
    if sources.is_empty() {
        return None;
    }

    // Get input ranges for this consumer
    let (input_ranges, _) = ctx.get_ranges(x)?;

    let mut new_sources = Vec::with_capacity(sources.len());
    let mut any_changed = false;

    for src in sources.iter() {
        let new_src = transform_single_source(x, src, input_ranges, ctx);
        if !Rc::ptr_eq(&new_src, src) {
            any_changed = true;
        }
        new_sources.push(new_src);
    }

    if any_changed { Some(new_sources) } else { None }
}

/// Transform a single source by adding BUFFERIZE + INDEX if needed.
pub(crate) fn transform_single_source(
    _consumer: &Rc<UOp>,
    src: &Rc<UOp>,
    input_ranges: &[Rc<UOp>],
    ctx: &IndexingContext,
) -> Rc<UOp> {
    // Case 1: Source is a buffer-like op → add INDEX
    if matches!(
        src.op(),
        Op::Buffer { .. } | Op::BufferView { .. } | Op::MStack { .. } | Op::MSelect { .. } | Op::After { .. }
    ) {
        // Add INDEX with input ranges from consumer
        return UOp::index(Rc::clone(src), input_ranges.to_vec()).expect("Failed to create INDEX for buffer source");
    }

    // Case 2: Source needs realization → wrap in BUFFERIZE + INDEX
    if let Some(realize_axes) = ctx.get_realize_axes(src) {
        let (_, output_ranges) = ctx.get_ranges(src).expect("Realized op must have ranges");

        // Select ranges to close (only realized axes)
        let closed_ranges: Vec<_> = output_ranges
            .iter()
            .enumerate()
            .filter(|(i, _)| realize_axes.contains(i))
            .map(|(_, r)| Rc::clone(r))
            .collect();

        // Determine buffer options
        let opts = if output_ranges.len() == realize_axes.len() {
            // Full realization → GLOBAL address space
            // Use default device (will be determined later)
            BufferizeOpts { device: None, addrspace: AddrSpace::Global }
        } else {
            // Partial realization → LOCAL address space
            BufferizeOpts { device: None, addrspace: AddrSpace::Local }
        };

        // Create BUFFERIZE
        let bufferized = UOp::bufferize(Rc::clone(src), closed_ranges, opts);

        // Add INDEX with consumer's input ranges (filtered to realized axes)
        let index_ranges: Vec<_> = input_ranges
            .iter()
            .enumerate()
            .filter(|(i, _)| realize_axes.contains(i))
            .map(|(_, r)| Rc::clone(r))
            .collect();

        if !index_ranges.is_empty() {
            return UOp::index(bufferized, index_ranges).expect("Failed to create INDEX after BUFFERIZE");
        } else {
            return bufferized;
        }
    }

    // Case 3: No transformation needed
    Rc::clone(src)
}

/// Check if a movement op should be removed.
///
/// A movement op should be removed if:
/// 1. It has ranges assigned in the range_map, OR
/// 2. Its source is already an INDEX operation
///
/// This indicates the transformation has been applied.
pub fn should_remove_movement_op(x: &Rc<UOp>, ctx: &IndexingContext) -> bool {
    // Remove if op has ranges assigned
    if ctx.range_map.contains_key(&UOpKey(Rc::clone(x))) {
        return true;
    }

    // Remove if source is already indexed
    if let Some(src) = x.op().sources().first()
        && matches!(src.op(), Op::Index { .. })
    {
        return true;
    }

    false
}

/// Main rangeify transformation entry point.
///
/// Converts movement operations (RESHAPE, PERMUTE, EXPAND, PAD, SHRINK, FLIP)
/// into BUFFERIZE + INDEX operations with explicit loop ranges.
///
/// # Algorithm
///
/// 1. **Range assignment** - Determine input/output ranges for each UOp
/// 2. **Early rewrites** - Cleanup (DETACH, CONTIGUOUS_BACKWARD removal)
/// 3. **Core rangeify** - Convert movement ops to BUFFERIZE + INDEX
/// 4. **Buffer folding** - Noop removal, constant folding
/// 5. **Dead axis removal** - Remove size-1 dimensions
/// 6. **Cost-based removal** - Remove unnecessary buffers
/// 7. **Symbolic simplification** - Optimize index expressions
/// 8. **Buffer limit enforcement** - Force bufferization when device limits exceeded
/// 9. **Reduction simplifications** - reduce_unparented, split_reduceop
///
/// Future phases will add:
/// 10. **Kernel splitting** - Split at STORE boundaries
///
/// # Arguments
///
/// * `sink` - The root of the computation graph to transform
///
/// # Returns
///
/// A tuple of:
/// - The transformed computation graph
/// - A RangeifyContext tracking the transformation
///
/// # Example
///
/// ```ignore
/// use schedule::rangeify::rangeify;
/// use morok_ir::UOp;
///
/// let x = UOp::const_(DType::Float32, ConstValue::Float(1.0));
/// let (result, ctx) = rangeify(x, None);
/// println!("Generated {} ranges", ctx.range_counter);
/// ```
pub fn rangeify(
    sink: Rc<UOp>,
    pcontig_config: Option<&super::buffer_cost::PcontigConfig>,
) -> morok_ir::Result<(Rc<UOp>, super::context::RangeifyContext)> {
    use std::cell::RefCell;
    use std::rc::Rc as StdRc;

    // Step 1: Run range assignment to build IndexingContext
    let (mut sink, indexing_ctx) = super::indexing::run_rangeify(sink)?;

    // Step 2: Wrap context for pattern access via closure capture
    let ctx = StdRc::new(RefCell::new(indexing_ctx));

    // Step 3: Apply early rewrites (DETACH, CONTIGUOUS_BACKWARD removal)
    let early_matcher = super::patterns::early_rewrites();
    sink = crate::rewrite::graph_rewrite(&early_matcher, sink);

    // Step 4: Apply core rangeify patterns (BUFFERIZE insertion, movement removal)
    {
        let rangeify_matcher = super::patterns::apply_rangeify_patterns(StdRc::clone(&ctx));
        sink = crate::rewrite::graph_rewrite(&rangeify_matcher, sink);
        // rangeify_matcher is dropped here, releasing its reference to ctx
    }

    // Step 5: Apply buffer folding optimizations
    // These patterns simplify and optimize BUFFERIZE operations
    let buffer_folding_matcher = super::patterns::buffer_folding();
    sink = crate::rewrite::graph_rewrite(&buffer_folding_matcher, sink);

    // Step 6: Remove dead axes (size-1 dimensions)
    // This simplifies buffer structure after folding
    let dead_axis_matcher = super::patterns::dead_axis_removal();
    sink = crate::rewrite::graph_rewrite(&dead_axis_matcher, sink);

    // Step 7: Cost-based buffer removal with partial contiguous
    // Remove buffers that don't provide performance benefits
    // Use partial contiguous to selectively materialize dimensions when beneficial
    let config = pcontig_config.cloned().unwrap_or_default();
    let buffer_removal_matcher = super::patterns::buffer_removal_with_pcontig(&config);
    sink = crate::rewrite::graph_rewrite(&buffer_removal_matcher, sink);

    // Step 8: Symbolic simplification
    // Optimize index expressions created during rangeify
    let symbolic_matcher = crate::symbolic::symbolic_simple();
    sink = crate::rewrite::graph_rewrite(&symbolic_matcher, sink);

    // Step 8.5: Buffer limit enforcement
    // Enforce device-specific buffer limits by forcing bufferization when exceeded
    // Only applies if device has buffer limits (Metal: 31, WebGPU: 8)
    if let Some(device) = super::buffer_limits::extract_device_from_graph(&sink)
        && let Some(limit) = device.max_buffers()
    {
        let limit_matcher = super::buffer_limits::buffer_limit_patterns(limit);
        sink = crate::rewrite::graph_rewrite(&limit_matcher, sink);
    }

    // Step 9: Reduction simplifications
    // Apply reduce_unparented and split_reduceop optimizations
    let split_config = super::split_reduceop::SplitReduceOpConfig::default();
    let reduction_matcher = super::patterns::reduction_simplify_patterns(&split_config);
    sink = crate::rewrite::graph_rewrite(&reduction_matcher, sink);

    // Step 10: Extract final context
    let final_indexing_ctx = StdRc::try_unwrap(ctx).ok().expect("Context should have no other references").into_inner();

    // Step 11: Build RangeifyContext for return
    let rangeify_ctx = super::context::RangeifyContext {
        range_counter: final_indexing_ctx.range_counter(),
        range_map: std::collections::HashMap::new(), // Could populate from indexing_ctx if needed
    };

    Ok((sink, rangeify_ctx))
}
