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

/// Check if a UOp is a chain of movement ops ending in a buffer-like op.
///
/// This detects patterns like `RESHAPE(BUFFER)`, `PERMUTE(EXPAND(BUFFER))`, etc.
/// These need INDEX wrapping so that `movement_op_patterns` can later transform
/// the indices appropriately.
fn is_movement_chain_on_buffer(uop: &Rc<UOp>) -> bool {
    let mut current = uop.clone();
    while current.op().is_movement() {
        if let Some(src) = current.op().sources().first() {
            current = src.clone();
        } else {
            return false;
        }
    }
    matches!(
        current.op(),
        Op::Buffer { .. } | Op::BufferView { .. } | Op::MStack { .. } | Op::MSelect { .. } | Op::After { .. }
    )
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

    // Case 1.5: Movement op chain on buffer-like source → add INDEX
    // This enables movement_op_patterns to later transform the indices.
    // Example: RESHAPE(BUFFER).INDEX([ranges]) will become BUFFER.INDEX(transformed_ranges)
    if src.op().is_movement() && is_movement_chain_on_buffer(src) {
        return UOp::index(Rc::clone(src), input_ranges.to_vec())
            .expect("Failed to create INDEX for movement buffer source");
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

/// Apply buffer removal patterns while protecting SINK sources from removal.
///
/// SINK sources are root BUFFERIZE operations that represent user-requested
/// output materialization. These should NOT be removed even if their compute
/// is "cheap to inline".
///
/// This function:
/// 1. Extracts SINK sources (the BUFFERIZEs to protect)
/// 2. For each BUFFERIZE, applies buffer_removal only to its compute graph
/// 3. Reconstructs the BUFFERIZEs with optimized compute
/// 4. Reconstructs SINK with the protected BUFFERIZEs
fn apply_buffer_removal_protecting_sink(
    sink: &Rc<UOp>,
    matcher: &crate::pattern::matcher::PatternMatcher<super::buffer_cost::PcontigConfig>,
    ctx: &mut super::buffer_cost::PcontigConfig,
) -> Rc<UOp> {
    // If not a SINK, apply buffer_removal normally
    let Op::Sink { sources } = sink.op() else {
        return crate::rewrite::graph_rewrite(matcher, sink.clone(), ctx);
    };

    // Process each SINK source
    let mut new_sources = Vec::with_capacity(sources.len());
    for src in sources.iter() {
        // If source is BUFFERIZE, optimize its compute but keep the BUFFERIZE wrapper
        if let Op::Bufferize { compute, ranges, opts } = src.op() {
            // Apply buffer_removal to the compute graph only
            let optimized_compute = crate::rewrite::graph_rewrite(matcher, compute.clone(), ctx);

            // Reconstruct BUFFERIZE with optimized compute (preserve the BUFFERIZE wrapper)
            let new_bufferize = UOp::bufferize(optimized_compute, ranges.to_vec(), opts.clone());
            new_sources.push(new_bufferize);
        } else {
            // Non-BUFFERIZE source: apply buffer_removal normally
            let optimized = crate::rewrite::graph_rewrite(matcher, src.clone(), ctx);
            new_sources.push(optimized);
        }
    }

    // Reconstruct SINK with protected sources
    UOp::sink(new_sources)
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
/// 3. **Core rangeify** - Convert movement ops to BUFFERIZE + INDEX (bottom-up)
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
    // Step 1: Run range assignment to build IndexingContext
    let (mut sink, mut indexing_ctx) = super::indexing::run_rangeify(sink)?;

    // Step 2: Apply early rewrites (DETACH, CONTIGUOUS_BACKWARD removal)
    let early_matcher = super::patterns::early_rewrites();
    sink = crate::rewrite::graph_rewrite(&early_matcher, sink, &mut ());

    // Step 3: Apply core rangeify transformation (bottom-up)
    // This must be bottom-up so that ADD gets transformed before BUFFERIZE.
    // Using graph_rewrite_bottom_up to ensure children are processed first.
    // NOTE: ReduceAxis → REDUCE conversion is now part of this step
    let rangeify_matcher = super::patterns::apply_rangeify_patterns();
    sink = crate::rewrite::graph_rewrite_bottom_up(&rangeify_matcher, sink, &mut indexing_ctx);

    // Step 4: Buffer simplification (folding + dead axis removal)
    // - Folds noop BUFFERIZE (INDEX with same ranges)
    // - Removes size-1 dimensions from BUFFERIZE
    let buffer_simplify = super::patterns::buffer_folding() + super::patterns::dead_axis_removal();
    sink = crate::rewrite::graph_rewrite(&buffer_simplify, sink, &mut ());

    // Step 5: Cost-based buffer removal with partial contiguous
    // Remove buffers that don't provide performance benefits
    // Use partial contiguous to selectively materialize dimensions when beneficial
    //
    // IMPORTANT: SINK sources (root BUFFERIZE ops) should NOT be removed.
    // They represent user-requested output materialization.
    // We protect them by extracting SINK sources, applying buffer_removal to their
    // compute graphs only, then reconstructing SINK.
    let mut pcontig = pcontig_config.cloned().unwrap_or_default();
    let buffer_removal_matcher = super::patterns::buffer_removal_with_pcontig();
    sink = apply_buffer_removal_protecting_sink(&sink, &buffer_removal_matcher, &mut pcontig);

    // Step 6: Symbolic simplification
    // Optimize index expressions created during rangeify
    let symbolic_matcher = crate::symbolic::symbolic_simple();
    sink = crate::rewrite::graph_rewrite(&symbolic_matcher, sink, &mut ());

    // Step 6.5: Buffer limit enforcement
    // Enforce device-specific buffer limits by forcing bufferization when exceeded
    // Only applies if device has buffer limits (Metal: 31, WebGPU: 8)
    if let Some(device) = super::buffer_limits::extract_device_from_graph(&sink)
        && let Some(limit) = device.max_buffers()
    {
        let limit_matcher = super::buffer_limits::buffer_limit_patterns(limit);
        sink = crate::rewrite::graph_rewrite(&limit_matcher, sink, &mut ());
    }

    // Step 7: Reduction simplifications
    // Apply reduce_unparented and split_reduceop optimizations
    let mut split_config = super::split_reduceop::SplitReduceOpConfig::default();
    let reduction_matcher = super::patterns::reduction_simplify_patterns();
    sink = crate::rewrite::graph_rewrite(&reduction_matcher, sink, &mut split_config);

    // Step 8: Build RangeifyContext for return
    let rangeify_ctx = super::context::RangeifyContext {
        range_counter: indexing_ctx.range_counter(),
        range_map: std::collections::HashMap::new(), // Could populate from indexing_ctx if needed
    };

    Ok((sink, rangeify_ctx))
}
