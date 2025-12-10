//! Consolidated kernel splitting and pipeline orchestration.
//!
//! This module contains:
//! - KernelContext for tracking state during kernel splitting
//! - split_store for splitting computation at STORE boundaries
//! - run_kernel_split_pipeline for full pipeline orchestration
//! - Buffer cost analysis (PcontigConfig, collect_accessed_buffers, etc.)
//! - Two-stage reduction splitting (split_reduceop)
//!
//! Consolidated from: kernel_context.rs, split_kernel.rs, pipeline.rs,
//! buffer_cost.rs, split_reduceop.rs

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use morok_ir::{AddrSpace, AxisType, Op, SInt, UOp, UOpKey};
use smallvec::SmallVec;
use tracing::{debug, trace};

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Configuration for partial contiguous optimization.
#[derive(Debug, Clone, Copy)]
pub struct PcontigConfig {
    /// 0=disabled, 1=basic, 2=enabled (default), 3+=aggressive
    pub level: u8,
    /// Max buffers before keeping BUFFERIZE (default: 3)
    pub max_buffers_threshold: usize,
    /// Max output/input ratio for partial contiguous (default: 10.0)
    pub out_in_ratio_threshold: f64,
}

impl Default for PcontigConfig {
    fn default() -> Self {
        Self { level: 2, max_buffers_threshold: 3, out_in_ratio_threshold: 10.0 }
    }
}

/// Configuration for split_reduceop optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitReduceOpConfig {
    /// Minimum input/output ratio to trigger splitting (default: 32768)
    pub split_threshold: usize,
    /// Max output buffer size as 2^N elements (default: 22 = 4M elements)
    pub output_size_bits: u32,
    /// Max split divisor (default: 256)
    pub max_divisor: usize,
    /// Min split divisor (default: 8)
    pub min_divisor: usize,
    /// Enable/disable the optimization (default: true)
    pub enabled: bool,
}

impl Default for SplitReduceOpConfig {
    fn default() -> Self {
        Self { split_threshold: 32768, output_size_bits: 22, max_divisor: 256, min_divisor: 8, enabled: true }
    }
}

impl SplitReduceOpConfig {
    pub fn max_output_size(&self) -> usize {
        2_usize.pow(self.output_size_bits)
    }
}

// ============================================================================
// KERNEL CONTEXT
// ============================================================================

/// Represents a dependency between two kernels.
#[derive(Clone, Debug)]
pub struct KernelDependency {
    pub buffer_id: u64,
    pub producer: Arc<UOp>,
    pub consumer: Arc<UOp>,
}

/// Context for tracking state during kernel splitting.
#[derive(Clone)]
pub struct KernelContext {
    pub global_counter: usize,
    pub local_counter: usize,
    pub buffer_map: HashMap<UOpKey, Arc<UOp>>,
    pub vars: HashSet<UOpKey>,
    pub range_counter: usize,
    pub kernel_deps: Vec<KernelDependency>,
}

impl KernelContext {
    pub fn new() -> Self {
        Self {
            global_counter: 0,
            local_counter: 0,
            buffer_map: HashMap::new(),
            vars: HashSet::new(),
            range_counter: 0,
            kernel_deps: Vec::new(),
        }
    }

    pub fn add_dependency(&mut self, buffer_id: u64, producer: Arc<UOp>, consumer: Arc<UOp>) {
        self.kernel_deps.push(KernelDependency { buffer_id, producer, consumer });
    }

    pub fn next_global(&mut self) -> usize {
        let id = self.global_counter;
        self.global_counter += 1;
        id
    }

    pub fn next_local(&mut self) -> usize {
        let id = self.local_counter;
        self.local_counter += 1;
        id
    }

    pub fn next_range(&mut self) -> usize {
        let id = self.range_counter;
        self.range_counter += 1;
        id
    }

    pub fn has_buffer(&self, buf: &Arc<UOp>) -> bool {
        self.buffer_map.contains_key(&UOpKey(buf.clone()))
    }

    pub fn get_buffer(&self, buf: &Arc<UOp>) -> Option<&Arc<UOp>> {
        self.buffer_map.get(&UOpKey(buf.clone()))
    }

    pub fn map_buffer(&mut self, original: Arc<UOp>, replacement: Arc<UOp>) {
        self.buffer_map.insert(UOpKey(original), replacement);
    }

    pub fn add_var(&mut self, var: Arc<UOp>) {
        self.vars.insert(UOpKey(var));
    }

    pub fn has_var(&self, var: &Arc<UOp>) -> bool {
        self.vars.contains(&UOpKey(var.clone()))
    }
}

impl Default for KernelContext {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// BUFFER COST ANALYSIS
// ============================================================================

/// Collect BUFFER, BUFFERIZE(GLOBAL), MSTACK, and MSELECT operations.
#[allow(clippy::mutable_key_type)]
pub fn collect_accessed_buffers(src: &Arc<UOp>) -> Vec<Arc<UOp>> {
    let mut buffers = Vec::new();
    let mut visited = HashSet::new();

    fn visit(uop: &Arc<UOp>, buffers: &mut Vec<Arc<UOp>>, visited: &mut HashSet<UOpKey>) -> bool {
        let key = UOpKey(Arc::clone(uop));
        if !visited.insert(key) {
            return true;
        }

        match uop.op() {
            Op::Bufferize { opts, .. } if opts.addrspace == AddrSpace::Global => {
                buffers.push(Arc::clone(uop));
                return false;
            }
            Op::Buffer { .. } | Op::MStack { .. } | Op::MSelect { .. } => {
                buffers.push(Arc::clone(uop));
            }
            _ => {}
        }

        for child in uop.op().sources() {
            visit(&child, buffers, visited);
        }
        true
    }

    visit(src, &mut buffers, &mut visited);

    let mut seen = HashSet::new();
    buffers.retain(|b| seen.insert(UOpKey(Arc::clone(b))));
    buffers
}

/// Collect all REDUCE operations in a computation tree.
#[allow(clippy::mutable_key_type)]
pub fn collect_reduces(src: &Arc<UOp>) -> Vec<Arc<UOp>> {
    let mut reduces = Vec::new();
    let mut visited = HashSet::new();

    fn visit(uop: &Arc<UOp>, reduces: &mut Vec<Arc<UOp>>, visited: &mut HashSet<UOpKey>) {
        let key = UOpKey(Arc::clone(uop));
        if !visited.insert(key) {
            return;
        }

        if matches!(uop.op(), Op::Reduce { .. }) {
            reduces.push(Arc::clone(uop));
        }

        for child in uop.op().sources() {
            visit(&child, reduces, visited);
        }
    }

    visit(src, &mut reduces, &mut visited);
    reduces
}

/// Collect all INDEX operations in a computation tree.
#[allow(clippy::mutable_key_type)]
pub fn collect_indexes(src: &Arc<UOp>) -> Vec<Arc<UOp>> {
    let mut indexes = Vec::new();
    let mut visited = HashSet::new();

    fn visit(uop: &Arc<UOp>, indexes: &mut Vec<Arc<UOp>>, visited: &mut HashSet<UOpKey>) {
        let key = UOpKey(Arc::clone(uop));
        if !visited.insert(key) {
            return;
        }

        if matches!(uop.op(), Op::Index { .. }) {
            indexes.push(Arc::clone(uop));
        }

        for child in uop.op().sources() {
            visit(&child, indexes, visited);
        }
    }

    visit(src, &mut indexes, &mut visited);
    indexes
}

/// Calculate buffer size in bytes.
pub fn calculate_buffer_size(buffer: &Arc<UOp>) -> Option<usize> {
    use morok_ir::ConstValue;

    match buffer.op() {
        Op::Bufferize { ranges, .. } => {
            let mut product = 1usize;
            for range in ranges {
                match range.op() {
                    Op::Range { end, .. } => match end.op() {
                        Op::Const(cv) => match cv.0 {
                            ConstValue::Int(n) if n > 0 => {
                                product = product.checked_mul(n as usize)?;
                            }
                            _ => return None,
                        },
                        _ => return None,
                    },
                    _ => return None,
                }
            }
            let element_size = buffer.dtype().bytes();
            Some(product.checked_mul(element_size)?)
        }
        Op::Buffer { size, .. } => Some(*size),
        Op::MStack { .. } | Op::MSelect { .. } => Some(1),
        _ => None,
    }
}

/// Calculate output/input size ratio.
pub fn calculate_out_in_ratio(output_size: usize, input_buffers: &[Arc<UOp>]) -> Option<f64> {
    let mut input_sum = 0usize;

    for buf in input_buffers {
        match calculate_buffer_size(buf) {
            Some(size) => input_sum = input_sum.checked_add(size)?,
            None => return None,
        }
    }

    let ratio = (output_size + 1) as f64 / (input_sum + 1) as f64;
    Some(ratio)
}

/// Check if any buffer is accessed within a reduce scope.
#[allow(clippy::mutable_key_type)]
pub fn has_buffer_in_reduce(reduces: &[Arc<UOp>]) -> bool {
    if reduces.is_empty() {
        return false;
    }

    let reduce_sources: Vec<Arc<UOp>> = reduces
        .iter()
        .filter_map(|r| if let Op::Reduce { src, .. } = r.op() { Some(Arc::clone(src)) } else { None })
        .collect();

    if reduce_sources.is_empty() {
        return false;
    }

    let sink = UOp::sink(reduce_sources);
    let mut visited = HashSet::new();
    let mut found_buffer = false;

    fn visit(uop: &Arc<UOp>, found: &mut bool, visited: &mut HashSet<UOpKey>) -> bool {
        if *found {
            return false;
        }

        let key = UOpKey(Arc::clone(uop));
        if !visited.insert(key) {
            return true;
        }

        match uop.op() {
            Op::Buffer { .. } | Op::Bufferize { .. } => {
                *found = true;
                return false;
            }
            _ => {}
        }

        for child in uop.op().sources() {
            if !visit(&child, found, visited) {
                return false;
            }
        }
        true
    }

    visit(&sink, &mut found_buffer, &mut visited);
    found_buffer
}

/// Filter indexes to only those accessing LOCAL bufferize operations.
pub fn collect_local_indexes(indexes: &[Arc<UOp>]) -> Vec<Arc<UOp>> {
    indexes
        .iter()
        .filter(|idx| {
            matches!(idx.op(), Op::Index { buffer, .. }
                if matches!(buffer.op(), Op::Bufferize { opts, .. }
                    if opts.addrspace == AddrSpace::Local))
        })
        .map(Arc::clone)
        .collect()
}

/// Extract ranges that must be materialized.
#[allow(clippy::mutable_key_type)]
pub fn extract_exclude_ranges(local_indexes: &[Arc<UOp>]) -> HashSet<UOpKey> {
    let mut exclude = HashSet::new();

    for idx in local_indexes {
        if let Op::Index { indices, .. } = idx.op() {
            for range in indices {
                for r in range.in_scope_ranges() {
                    exclude.insert(r.clone());
                }
            }
        }
    }

    exclude
}

/// Partition ranges into materialize vs substitute.
#[allow(clippy::type_complexity, clippy::mutable_key_type)]
pub fn partition_ranges(
    buf_ranges: &[Arc<UOp>],
    idx_ranges: &[Arc<UOp>],
    exclude_ranges: &HashSet<UOpKey>,
) -> (Vec<(Arc<UOp>, Arc<UOp>)>, Vec<(Arc<UOp>, Arc<UOp>)>) {
    let mut materialize = Vec::new();
    let mut substitute = Vec::new();

    for (buf_rng, idx_rng) in buf_ranges.iter().zip(idx_ranges.iter()) {
        if matches!(buf_rng.op(), Op::Const(_)) {
            continue;
        }

        let buf_key = UOpKey(Arc::clone(buf_rng));

        let should_materialize = exclude_ranges.contains(&buf_key)
            || idx_rng.in_scope_ranges().iter().any(|r| {
                if let Op::Range { axis_type, .. } = r.0.op() { matches!(axis_type, AxisType::Reduce) } else { false }
            });

        let pair = (Arc::clone(buf_rng), Arc::clone(idx_rng));

        if should_materialize {
            materialize.push(pair);
        } else {
            substitute.push(pair);
        }
    }

    (materialize, substitute)
}

/// Apply partial contiguous transformation.
#[allow(clippy::mutable_key_type)]
pub fn apply_partial_contiguous(
    src: &Arc<UOp>,
    materialize: Vec<(Arc<UOp>, Arc<UOp>)>,
    substitute: Vec<(Arc<UOp>, Arc<UOp>)>,
) -> Option<Arc<UOp>> {
    use morok_ir::BufferizeOpts;

    if substitute.is_empty() {
        return None;
    }

    let subs_map: HashMap<UOpKey, Arc<UOp>> = substitute.into_iter().map(|(k, v)| (UOpKey(k), v)).collect();
    let substituted = src.substitute(&subs_map);

    if materialize.is_empty() {
        return Some(substituted);
    }

    let (mat_buf_rngs, mat_idx_rngs): (Vec<_>, Vec<_>) = materialize.into_iter().unzip();

    let opts = BufferizeOpts::local();
    let bufferized = UOp::bufferize(substituted, mat_buf_rngs, opts);

    UOp::index(bufferized, mat_idx_rngs).ok()
}

// ============================================================================
// SPLIT KERNEL
// ============================================================================

/// Find first COPY or BUFFER_VIEW operation.
fn find_copy_or_buffer_view(uop: &Arc<UOp>) -> Option<Arc<UOp>> {
    for node in uop.toposort() {
        if matches!(node.op(), Op::Copy { .. } | Op::BufferView { .. }) {
            return Some(node);
        }
    }
    None
}

/// Split STORE and END operations into individual kernels.
pub fn split_store(uop: &Arc<UOp>, ctx: &mut KernelContext) -> Option<Arc<UOp>> {
    use super::patterns::{rangeify_codegen_patterns, to_define_global_patterns};
    use super::transforms::find_bufs;
    use crate::rewrite::graph_rewrite_bottom_up;

    trace!(uop_id = uop.id, op = ?std::mem::discriminant(uop.op()), "split_store: entering");

    // Handle AFTER wrapping STORE/END
    if let Op::After { deps, .. } = uop.op() {
        for dep in deps.iter() {
            match dep.op() {
                Op::Kernel { .. } => return Some(dep.clone()),
                Op::Store { .. } | Op::StoreGated { .. } => return split_store(dep, ctx),
                Op::End { computation, .. } if matches!(computation.op(), Op::Store { .. } | Op::StoreGated { .. }) => {
                    return split_store(dep, ctx);
                }
                Op::Barrier { src, .. } => {
                    if let Op::End { computation, .. } = src.op()
                        && matches!(computation.op(), Op::Store { .. } | Op::StoreGated { .. })
                    {
                        return split_store(src, ctx);
                    }
                    if matches!(src.op(), Op::Store { .. } | Op::StoreGated { .. }) {
                        return split_store(src, ctx);
                    }
                }
                _ => continue,
            }
        }
        return None;
    }

    // Skip if has non-OUTER ranges
    if uop.has_non_outer_ranges() {
        return None;
    }

    // Verify operation type
    let computation = match uop.op() {
        Op::End { computation, ranges } => match computation.op() {
            Op::Store { .. } | Op::StoreGated { .. } => {
                for r in ranges.iter() {
                    if let Op::Range { axis_type, .. } = r.op()
                        && *axis_type == AxisType::Outer
                    {
                        return None;
                    }
                }
                uop.clone()
            }
            _ => return None,
        },
        Op::Store { .. } | Op::StoreGated { .. } => uop.clone(),
        _ => return None,
    };

    // Apply transformation pipeline
    let transformed = {
        let matcher = to_define_global_patterns();
        graph_rewrite_bottom_up(&matcher, computation, ctx)
    };

    let transformed = {
        let movement_matcher = super::patterns::movement_op_patterns();
        graph_rewrite_bottom_up(&movement_matcher, transformed, &mut ())
    };

    let transformed = {
        let codegen_matcher = rangeify_codegen_patterns();
        graph_rewrite_bottom_up(&codegen_matcher, transformed, &mut ())
    };

    // Validate no buffer access cycles
    #[allow(clippy::mutable_key_type)]
    let _buf_accesses = find_bufs(&transformed);

    // Create kernel AST
    let ast = if let Some(special_op) = find_copy_or_buffer_view(&transformed) {
        special_op
    } else {
        UOp::sink(vec![transformed])
    };

    // Build kernel sources
    let mut sources: SmallVec<[Arc<UOp>; 4]> = SmallVec::new();

    // morok_ir::uop::debug::print_ast(&ast, "KERNEL AST", 2);

    // Collect buffer sources
    let mut buffer_sources: Vec<Arc<UOp>> = Vec::new();

    for value in ctx.buffer_map.values() {
        let actual_buffer = match value.op() {
            Op::After { passthrough, .. } => passthrough.clone(),
            _ => value.clone(),
        };
        buffer_sources.push(actual_buffer);
    }

    buffer_sources.sort_by_key(|b| match b.op() {
        Op::DefineGlobal(id) => *id as u64,
        Op::DefineLocal(id) => (*id as u64) + (1u64 << 32),
        Op::Buffer { .. } => b.id + (1u64 << 48),
        _ => b.id,
    });

    for node in buffer_sources {
        sources.push(node);
    }

    // Collect variable sources
    let mut var_sources: Vec<Arc<UOp>> =
        ast.toposort().into_iter().filter(|node| matches!(node.op(), Op::DefineVar { .. })).collect();

    var_sources.sort_by(|a, b| {
        let name_a = match a.op() {
            Op::DefineVar { name, .. } => name,
            _ => unreachable!(),
        };
        let name_b = match b.op() {
            Op::DefineVar { name, .. } => name,
            _ => unreachable!(),
        };
        name_a.cmp(name_b)
    });

    for node in var_sources {
        sources.push(node);
    }

    for var_key in &ctx.vars {
        sources.push(var_key.0.clone());
    }

    let kernel = UOp::kernel(sources.clone(), ast.clone());
    debug!(
        kernel_id = kernel.id,
        num_sources = sources.len(),
        buffer_map_size = ctx.buffer_map.len(),
        "split_store: created kernel"
    );
    for (i, src) in sources.iter().enumerate() {
        debug!(
            kernel_id = kernel.id,
            source_idx = i,
            source_id = src.id,
            source_op = ?src.op(),
            source_dtype = ?src.dtype(),
            "split_store: kernel source"
        );
    }

    Some(kernel)
}

// ============================================================================
// PIPELINE
// ============================================================================

/// Run the kernel splitting pipeline.
pub fn run_kernel_split_pipeline(root: Arc<UOp>) -> (Arc<UOp>, KernelContext) {
    use super::transforms::bufferize_to_store;

    let mut ctx = KernelContext::new();

    let after_bufferize = transform_bottom_up(&root, &mut ctx, bufferize_to_store);

    // DEBUG: Print after bufferize_to_store
    if std::env::var("MOROK_DEBUG_RANGEIFY").is_ok() {
        morok_ir::uop::debug::print_ast(&after_bufferize, "AFTER BUFFERIZE_TO_STORE", 15);
    }

    let after_split = transform_bottom_up(&after_bufferize, &mut ctx, split_store);

    resolve_kernel_dependencies(&after_split, &mut ctx);

    (after_split, ctx)
}

fn resolve_kernel_dependencies(root: &Arc<UOp>, ctx: &mut KernelContext) {
    let mut buffer_producers: HashMap<u64, Arc<UOp>> = HashMap::new();

    for node in root.toposort() {
        if let Op::After { passthrough, deps } = node.op()
            && let Op::DefineGlobal(_) = passthrough.op()
        {
            for dep in deps {
                if let Op::Kernel { .. } = dep.op() {
                    buffer_producers.insert(passthrough.id, dep.clone());
                    break;
                }
            }
        }
    }

    for node in root.toposort() {
        if let Op::Kernel { sources, ast } = node.op() {
            for src in sources {
                let buffer_id = match src.op() {
                    Op::After { passthrough, .. } => {
                        if let Op::DefineGlobal(_) = passthrough.op() {
                            Some(passthrough.id)
                        } else {
                            None
                        }
                    }
                    Op::DefineGlobal(_) => Some(src.id),
                    _ => None,
                };

                if let Some(buf_id) = buffer_id
                    && let Some(producer) = buffer_producers.get(&buf_id)
                    && !Arc::ptr_eq(producer, &node)
                {
                    ctx.add_dependency(buf_id, producer.clone(), node.clone());
                }
            }

            for ast_node in ast.toposort() {
                if let Op::Load { buffer, .. } = ast_node.op() {
                    let buffer_id = match buffer.op() {
                        Op::DefineGlobal(_) => Some(buffer.id),
                        _ => None,
                    };

                    if let Some(buf_id) = buffer_id
                        && let Some(producer) = buffer_producers.get(&buf_id)
                        && !Arc::ptr_eq(producer, &node)
                    {
                        let already_tracked = ctx.kernel_deps.iter().any(|d| {
                            d.buffer_id == buf_id
                                && Arc::ptr_eq(&d.producer, producer)
                                && Arc::ptr_eq(&d.consumer, &node)
                        });
                        if !already_tracked {
                            ctx.add_dependency(buf_id, producer.clone(), node.clone());
                        }
                    }
                }
            }
        }
    }
}

fn transform_bottom_up<F>(uop: &Arc<UOp>, ctx: &mut KernelContext, transform_fn: F) -> Arc<UOp>
where
    F: Fn(&Arc<UOp>, &mut KernelContext) -> Option<Arc<UOp>> + Copy,
{
    let sources = uop.op().sources();

    if sources.is_empty() {
        return transform_fn(uop, ctx).unwrap_or_else(|| uop.clone());
    }

    let mut transformed_sources = Vec::with_capacity(sources.len());
    let mut any_changed = false;

    for src in sources {
        let transformed = transform_bottom_up(&src, ctx, transform_fn);
        if !Arc::ptr_eq(&transformed, &src) {
            any_changed = true;
        }
        transformed_sources.push(transformed);
    }

    let reconstructed = if any_changed { uop.with_sources(transformed_sources) } else { uop.clone() };

    transform_fn(&reconstructed, ctx).unwrap_or(reconstructed)
}

// ============================================================================
// SPLIT REDUCEOP
// ============================================================================

/// Extract all RANGE axis IDs from a UOp tree.
pub fn collect_range_ids(indexed: &Arc<UOp>) -> Vec<usize> {
    let mut range_ids: Vec<usize> = indexed
        .toposort()
        .into_iter()
        .filter_map(|node| if let Op::Range { axis_id, .. } = node.op() { Some(axis_id.value()) } else { None })
        .collect();

    range_ids.sort_unstable();
    range_ids.dedup();
    range_ids
}

#[derive(Debug, Clone)]
struct SplitCandidate {
    dimension: usize,
    divisor: usize,
    #[allow(dead_code)]
    output_size: usize,
}

fn detect_expanded_dimensions(source: &Arc<UOp>, input_shape: &[SInt]) -> Vec<bool> {
    let ranges: Vec<Arc<UOp>> = input_shape
        .iter()
        .enumerate()
        .map(|(axis_id, dim)| match dim {
            SInt::Const(n) if *n > 1 => {
                let end = UOp::index_const(*n as i64);
                UOp::range_axis(end, morok_ir::AxisId::Unrenumbered(axis_id), morok_ir::AxisType::Loop)
            }
            _ => UOp::index_const(0),
        })
        .collect();

    let indexed = match UOp::index(Arc::clone(source), ranges) {
        Ok(idx) => idx,
        Err(_) => return vec![false; input_shape.len()],
    };

    let base = source.base();
    let noop = UOp::noop();
    #[allow(clippy::mutable_key_type)]
    let mut substitutions = HashMap::new();
    substitutions.insert(UOpKey(base), noop);

    let substituted = indexed.substitute(&substitutions);

    use super::patterns::movement_op_patterns;
    use crate::rewrite::graph_rewrite;

    let pm_mops = movement_op_patterns();
    let transformed = graph_rewrite(&pm_mops, substituted, &mut ());

    let surviving_range_ids = collect_range_ids(&transformed);
    let surviving_set: HashSet<usize> = surviving_range_ids.into_iter().collect();

    input_shape.iter().enumerate().map(|(axis_id, _)| !surviving_set.contains(&axis_id)).collect()
}

fn find_split_candidates(
    reduce: &Arc<UOp>,
    input_shape: &[SInt],
    is_expanded: &[bool],
    config: &SplitReduceOpConfig,
) -> Vec<SplitCandidate> {
    let Op::ReduceAxis { axes: reduce_axes, .. } = reduce.op() else {
        return vec![];
    };

    let output_shape = match reduce.shape() {
        Ok(Some(shape)) => shape,
        _ => return vec![],
    };

    let output_size: usize = output_shape.iter().filter_map(|s| s.as_const()).product();

    let mut candidates = Vec::new();

    for &axis in reduce_axes {
        if axis >= is_expanded.len() || is_expanded[axis] {
            continue;
        }

        let dim_size = match &input_shape[axis] {
            SInt::Const(n) => *n,
            SInt::Symbolic(_) => continue,
        };

        for divisor in (config.min_divisor..=config.max_divisor).rev() {
            if dim_size % divisor != 0 {
                continue;
            }

            let new_output_size = output_size * divisor;

            if new_output_size > config.max_output_size() {
                continue;
            }

            candidates.push(SplitCandidate { dimension: axis, divisor, output_size: new_output_size });
        }
    }

    candidates
}

fn apply_split_transformation(
    source: &Arc<UOp>,
    reduce: &Arc<UOp>,
    candidate: &SplitCandidate,
    input_shape: &[SInt],
) -> Option<Arc<UOp>> {
    let Op::ReduceAxis { reduce_op, axes: reduce_axes, .. } = reduce.op() else {
        return None;
    };

    let dim_to_split = candidate.dimension;
    let divisor = candidate.divisor;
    let dim_size = input_shape[dim_to_split].as_const()?;
    let remainder = dim_size / divisor;

    let mut splitted_shape: SmallVec<[SInt; 4]> = SmallVec::new();
    for (i, dim) in input_shape.iter().enumerate() {
        if i == dim_to_split {
            splitted_shape.push(SInt::Const(divisor));
            splitted_shape.push(SInt::Const(remainder));
        } else {
            splitted_shape.push(dim.clone());
        }
    }

    let reshaped = source.try_reshape(&splitted_shape).ok()?;

    let mut permutation: Vec<usize> = (0..splitted_shape.len()).filter(|&i| i != dim_to_split).collect();
    permutation.push(dim_to_split);

    let permuted = reshaped.try_permute(permutation.clone()).ok()?;

    let adjusted_axes: Vec<usize> = reduce_axes
        .iter()
        .map(|&axis| {
            if axis < dim_to_split {
                axis
            } else if axis == dim_to_split {
                dim_to_split + 1
            } else {
                axis + 1
            }
        })
        .collect();

    let permuted_axes: Vec<usize> =
        adjusted_axes.iter().map(|&old_axis| permutation.iter().position(|&p| p == old_axis).unwrap()).collect();

    let first_reduce = permuted.try_reduce_axis(*reduce_op, permuted_axes).ok()?;

    let contiguous = UOp::contiguous(first_reduce);

    let output_shape = contiguous.shape().ok()??;
    let split_axis = output_shape.len() - 1;

    let second_reduce = contiguous.try_reduce_axis(*reduce_op, vec![split_axis]).ok()?;

    let final_shape = reduce.shape().ok()??;

    second_reduce.try_reshape(final_shape).ok()
}

/// Split large REDUCE_AXIS into two stages.
pub fn split_reduceop(reduce: &Arc<UOp>, config: &SplitReduceOpConfig) -> Option<Arc<UOp>> {
    if !config.enabled {
        return None;
    }

    let Op::ReduceAxis { src: source, .. } = reduce.op() else {
        return None;
    };

    let input_shape = source.shape().ok()??;
    let output_shape = reduce.shape().ok()??;

    if !input_shape.iter().all(|s| s.is_const()) {
        return None;
    }

    let input_size: usize = input_shape.iter().map(|s| s.as_const().unwrap()).product();
    let output_size: usize = output_shape.iter().map(|s| s.as_const().unwrap()).product();

    if output_size == 0 {
        return None;
    }

    let ratio = input_size / output_size;
    if ratio < config.split_threshold {
        return None;
    }

    let is_expanded = detect_expanded_dimensions(source, input_shape);
    let candidates = find_split_candidates(reduce, input_shape, &is_expanded, config);

    if candidates.is_empty() {
        return None;
    }

    apply_split_transformation(source, reduce, &candidates[0], input_shape)
}
