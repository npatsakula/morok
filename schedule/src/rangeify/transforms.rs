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
#[allow(clippy::mutable_key_type)]
#[tracing::instrument(skip_all, fields(origin.tree = sink.tree()))]
pub fn rangeify_with_map(
    sink: Arc<UOp>,
    pcontig_config: Option<&super::kernel::PcontigConfig>,
) -> morok_ir::Result<RangeifyResult> {
    use morok_ir::rewrite::{graph_rewrite_bottom_up_with_map, graph_rewrite_with_map};

    // Aggregate all becomes_maps from rewrite passes
    let mut all_becomes: HashMap<UOpKey, Arc<UOp>> = HashMap::new();

    // Step 1: Run range assignment to build IndexingContext
    let (mut sink, mut indexing_ctx) = super::indexing::run_rangeify(sink)?;
    tracing::debug!(uop.tree = sink.tree(), "range assignment complete");

    // Step 2: Apply early rewrites (DETACH, CONTIGUOUS_BACKWARD removal)
    let early_matcher = super::patterns::early_rewrites();
    let result = graph_rewrite_bottom_up_with_map(&early_matcher, sink, &mut ());
    sink = result.root;
    all_becomes.extend(result.becomes_map);
    tracing::debug!(uop.tree = sink.tree(), "early rewrites complete");

    // Step 2.5: Split large reductions BEFORE ReduceAxis → REDUCE conversion
    // split_reduceop needs ReduceAxis (not REDUCE), so it must run before Step 3
    let mut split_config = super::kernel::SplitReduceOpConfig::default();
    let split_matcher = super::patterns::split_reduceop_patterns();
    let result = graph_rewrite_with_map(&split_matcher, sink, &mut split_config);
    sink = result.root;
    all_becomes.extend(result.becomes_map);
    tracing::debug!(uop.tree = sink.tree(), "split reduceops complete");

    // Step 3: Apply core rangeify transformation (bottom-up)
    // This includes: bufferize transform, movement op removal, ReduceAxis → REDUCE conversion
    let rangeify_matcher = super::patterns::apply_rangeify_patterns();
    let result = graph_rewrite_bottom_up_with_map(&rangeify_matcher, sink, &mut indexing_ctx);
    sink = result.root;
    all_becomes.extend(result.becomes_map);
    tracing::debug!(uop.tree = sink.tree(), "rangeify complete");

    // Step 4: Buffer simplification
    let buffer_simplify = super::patterns::buffer_folding() + super::patterns::dead_axis_removal();
    let result = graph_rewrite_with_map(&buffer_simplify, sink, &mut ());
    sink = result.root;
    all_becomes.extend(result.becomes_map);
    tracing::debug!(uop.tree = sink.tree(), "buffer folding + dead axis removal complete");

    // Step 4.5: Apply early rewrites again for RESHAPE to scalar
    let result = graph_rewrite_bottom_up_with_map(&early_matcher, sink, &mut ());
    sink = result.root;
    all_becomes.extend(result.becomes_map);
    tracing::debug!(uop.tree = sink.tree(), "reshape to scalar complete");

    // Step 5: Cost-based buffer removal with partial contiguous
    let mut pcontig = pcontig_config.cloned().unwrap_or_default();
    let buffer_removal_matcher = super::patterns::buffer_removal_with_pcontig();
    sink = apply_buffer_removal_protecting_sink(&sink, &buffer_removal_matcher, &mut pcontig);
    // Note: apply_buffer_removal_protecting_sink doesn't use _with_map yet, could add later
    tracing::debug!(uop.tree = sink.tree(), "buffer removal complete");

    // Step 6: Symbolic simplification
    let symbolic_matcher = crate::symbolic::symbolic_simple();
    let result = graph_rewrite_with_map(&symbolic_matcher, sink, &mut ());
    sink = result.root;
    all_becomes.extend(result.becomes_map);
    tracing::debug!(uop.tree = sink.tree(), "symbolic simplification complete");

    // Step 6.5: Buffer limit enforcement
    if let Some(device) = super::patterns::extract_device_from_graph(&sink)
        && let Some(limit) = device.max_buffers()
    {
        let limit_matcher = super::patterns::buffer_limit_patterns(limit);
        let result = graph_rewrite_with_map(&limit_matcher, sink, &mut ());
        sink = result.root;
        all_becomes.extend(result.becomes_map);
        tracing::debug!(uop.tree = sink.tree(), "buffer limit enforcement complete");
    }

    // Step 7: Reduction simplifications (reduce_unparented, reduce_collapse)
    // These match Op::Reduce, so must run AFTER ReduceAxis → REDUCE conversion
    let reduction_matcher = super::patterns::reduction_simplify_patterns();
    let result = graph_rewrite_with_map(&reduction_matcher, sink, &mut ());
    sink = result.root;
    all_becomes.extend(result.becomes_map);
    tracing::debug!(uop.tree = sink.tree(), "reduction simplification complete");

    // Step 8: Build RangeifyContext for return
    let rangeify_ctx = RangeifyContext { range_counter: indexing_ctx.range_counter(), range_map: HashMap::new() };

    Ok(RangeifyResult { sink, context: rangeify_ctx, becomes_map: all_becomes })
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
pub fn bufferize_to_store(bufferize_op: &Arc<UOp>, ctx: &mut KernelContext) -> Option<Arc<UOp>> {
    let (compute, ranges, opts) = match bufferize_op.op() {
        Op::Bufferize { compute, ranges, opts } => {
            tracing::debug!(
                bufferize_id = bufferize_op.id,
                compute_id = compute.id,
                ranges_len = ranges.len(),
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

    let buffer = if let Some(existing_buffer) = ctx.get_buffer(bufferize_op) {
        existing_buffer.clone()
    } else if opts.addrspace == AddrSpace::Global {
        // Create BUFFER node (like Tinygrad's UOp.new_buffer)
        // The BUFFER → DEFINE_GLOBAL conversion happens later in split_store
        let device = opts.device.clone().unwrap_or(morok_ir::DeviceSpec::Cpu);
        UOp::new_buffer(device, size, base_dtype.clone())
    } else {
        // For local address space, create DEFINE_LOCAL directly (like Tinygrad)
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

    let mut do_store = if !end_ranges.is_empty() { UOp::end(store, end_ranges) } else { store };

    if opts.addrspace == AddrSpace::Local {
        do_store = UOp::barrier(do_store, SmallVec::new());
    }

    let result = UOp::after(buffer.clone(), SmallVec::from_elem(do_store, 1));
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

    let casted = UOp::cast(Arc::clone(value), scalar_type);

    if target_dtype.is_vector() {
        let count = target_dtype.count();
        let elements: SmallVec<[Arc<UOp>; 4]> = (0..count).map(|_| casted.clone()).collect();
        Some(UOp::vectorize(elements))
    } else {
        Some(casted)
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

    let mut all_range_sources: Vec<Arc<UOp>> = r.op().sources().iter().skip(off).cloned().collect();

    let innermost_computation = if matches!(r.op(), Op::End { .. }) {
        let mut computation = Arc::clone(&r.op().sources()[0]);

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

    let mut new_sources: Vec<Arc<UOp>> =
        if let Some(inner_comp) = innermost_computation { vec![inner_comp] } else { r.op().sources()[..off].to_vec() };
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
/// Based on Tinygrad's pm_add_buffers (rangeify.py:358-367).
pub fn pm_add_buffers_patterns() -> crate::TypedPatternMatcher<()> {
    use super::kernel::KernelContext;

    // This is a workaround - we create a temporary KernelContext for each match
    // The original code used a shared context, but the new pipeline uses graph_rewrite
    crate::patterns! {
        // BUFFERIZE → STORE conversion
        buf @ Bufferize { compute: _ } => |buf| {
            let mut temp_ctx = KernelContext::new();
            bufferize_to_store(buf, &mut temp_ctx)
        },
    }
}
