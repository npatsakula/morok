//! Consolidated kernel splitting and pipeline orchestration.
//!
//! This module contains:
//! - KernelContext for tracking state during kernel splitting
//! - split_store for splitting computation at STORE boundaries
//! - run_kernel_split_pipeline for full pipeline orchestration
//! - PcontigConfig for partial contiguous buffer removal
//! - Two-stage reduction splitting (split_reduceop)
//!
//! Consolidated from: kernel_context.rs, split_kernel.rs, pipeline.rs,
//! buffer_cost.rs, split_reduceop.rs

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use indexmap::IndexMap;
use morok_ir::{AxisType, Op, SInt, UOp, UOpKey};
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

/// Context for tracking state during kernel splitting.
///
/// Simplified from original 8 fields to 6 fields, removing Morok-specific
/// `kernel_deps` and `buffer_id_mapping` that are no longer needed after
/// aligning with Tinygrad's fix_assign approach.
#[derive(Clone)]
pub struct KernelContext {
    pub global_counter: usize,
    pub local_counter: usize,
    pub buffer_map: HashMap<UOpKey, Arc<UOp>>,
    pub vars: HashSet<UOpKey>,
    pub range_counter: usize,
    /// Mapping from DefineGlobal UOp ID to original BUFFER UOp ID.
    /// Used by to_define_global_patterns for buffer tracking.
    pub define_to_buffer_id: HashMap<u64, u64>,
}

impl KernelContext {
    pub fn new() -> Self {
        Self {
            global_counter: 0,
            local_counter: 0,
            buffer_map: HashMap::new(),
            vars: HashSet::new(),
            range_counter: 0,
            define_to_buffer_id: HashMap::new(),
        }
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
// LOCAL ADD BUFFER CONTEXT (Per-Kernel Context)
// ============================================================================

/// Per-kernel context for tracking state during kernel splitting.
///
/// Based on Tinygrad's `LocalAddBufferContext` (rangeify.py:376-383).
/// This is used within `split_store` for each individual kernel being created.
///
/// IMPORTANT: Uses IndexMap for `map` to maintain insertion order.
/// This is critical because DEFINE_GLOBAL indices are assigned in the order
/// patterns match, and kernel sources must be in the same order for correct
/// buffer indexing during execution.
#[derive(Default)]
pub struct LocalAddBufferContext {
    /// DEFINE_GLOBAL counter
    pub dg: usize,
    /// Buffer → AFTER mapping (IndexMap maintains insertion order)
    pub map: IndexMap<UOpKey, Arc<UOp>>,
    /// Bound variables
    pub vars: HashMap<UOpKey, ()>,
    /// Range renumber counter
    pub range: usize,
    /// Optimization hints extracted from CONTIGUOUS.opts (Tinygrad: ctx.opts)
    pub opts: Vec<morok_ir::ContiguousHint>,
}

impl LocalAddBufferContext {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get next DEFINE_GLOBAL index.
    pub fn next_dg(&mut self) -> usize {
        let id = self.dg;
        self.dg += 1;
        id
    }

    /// Get next range renumber index.
    pub fn next_range(&mut self) -> usize {
        let id = self.range;
        self.range += 1;
        id
    }

    /// Track a variable (like unbind_kernel in Tinygrad).
    pub fn add_var(&mut self, var: Arc<UOp>) {
        self.vars.insert(UOpKey(var), ());
    }

    /// Map a buffer to its AFTER wrapper.
    pub fn map_buffer(&mut self, buf: Arc<UOp>, after: Arc<UOp>) {
        self.map.insert(UOpKey(buf), after);
    }

    /// Check if buffer is already mapped.
    pub fn has_buffer(&self, buf: &Arc<UOp>) -> bool {
        self.map.contains_key(&UOpKey(buf.clone()))
    }
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
///
/// Based on Tinygrad's split_store (rangeify.py:480-507).
/// Simplified from 280 lines to ~80 lines using LocalAddBufferContext.
pub fn split_store(_ctx: &mut Vec<Arc<UOp>>, x: &Arc<UOp>) -> Option<Arc<UOp>> {
    use super::patterns::{
        local_to_define_global_patterns, movement_op_patterns, pm_syntactic_sugar, rangeify_codegen_patterns,
    };
    use crate::rewrite::graph_rewrite_bottom_up;

    trace!(uop_id = x.id, op = ?std::mem::discriminant(x.op()), "split_store: entering");

    // Guard 1: Skip if has non-OUTER ranges (like Tinygrad rangeify.py:482)
    let has_non_outer = x
        .in_scope_ranges()
        .iter()
        .any(|r| matches!(r.0.op(), Op::Range { axis_type, .. } if *axis_type != AxisType::Outer));
    if has_non_outer {
        return None;
    }

    // Guard 2: Skip END where LAST range is OUTER (like Tinygrad rangeify.py:485)
    // Tinygrad: `if x.op is Ops.END and x.src[1].arg[-1] == AxisType.OUTER: return None`
    if let Op::End { ranges, .. } = x.op()
        && let Some(r) = ranges.last()
        && matches!(r.op(), Op::Range { axis_type: AxisType::Outer, .. })
    {
        return None;
    }

    // Verify operation type (only STORE and END(STORE) are valid)
    let is_valid = match x.op() {
        Op::Store { .. } => true,
        Op::End { computation, .. } => matches!(computation.op(), Op::Store { .. }),
        _ => false,
    };
    if !is_valid {
        return None;
    }

    // Per-kernel context (LocalAddBufferContext)
    let mut lctx = LocalAddBufferContext::new();

    // Sequential rewrites (like Tinygrad's combined patterns)
    // 1. to_define_global: BUFFER → DEFINE_GLOBAL, handle AFTER, BIND
    let ret = {
        let matcher = local_to_define_global_patterns();
        graph_rewrite_bottom_up(&matcher, x.clone(), &mut lctx)
    };

    // 2. movement_op_patterns + pm_syntactic_sugar (pm_mops + pm_syntactic_sugar equivalent)
    // Tinygrad: graph_rewrite(sink, pm_mops+pm_syntactic_sugar, name="early movement ops", bottom_up=True)
    let ret = {
        let matcher = movement_op_patterns() + pm_syntactic_sugar();
        graph_rewrite_bottom_up(&matcher, ret, &mut ())
    };

    // 3. rangeify_codegen (CONTIGUOUS removal, NOOP → zero, hint extraction)
    let ret = {
        let matcher = rangeify_codegen_patterns();
        graph_rewrite_bottom_up(&matcher, ret, &mut lctx)
    };

    // Find COPY/BUFFER_VIEW or wrap in SINK (like Tinygrad rangeify.py:495-501)
    let ast = find_copy_or_buffer_view(&ret).unwrap_or_else(|| UOp::sink(vec![ret]));

    // Build KERNEL from context (like Tinygrad rangeify.py:504)
    // Sources: lctx.map.values() (buffer → AFTER mappings) + lctx.vars.keys() (bound variables)
    let sources: SmallVec<[Arc<UOp>; 4]> =
        lctx.map.values().cloned().chain(lctx.vars.keys().map(|k| k.0.clone())).collect();

    let kernel = UOp::kernel(sources.clone(), ast.clone());
    debug!(
        kernel_id = kernel.id,
        num_sources = sources.len(),
        map_size = lctx.map.len(),
        vars_size = lctx.vars.len(),
        "split_store: created kernel"
    );

    Some(kernel)
}

/// Fix inter-kernel dependencies (like Tinygrad's fix_assign).
///
/// Based on Tinygrad rangeify.py:568-580.
/// When kernel B reads from a buffer that kernel A writes to, this function
/// ensures kernel B's AFTER node depends on kernel A's AFTER node.
///
/// Uses buf_uop() to walk through AFTER chains and get underlying buffer IDs.
fn fix_assign(root: &Arc<UOp>) -> Arc<UOp> {
    // Map buf_uop().id -> AFTER node that produces it
    let mut kernel_assign: HashMap<u64, Arc<UOp>> = HashMap::new();
    #[allow(clippy::mutable_key_type)]
    let mut assign_rep: HashMap<UOpKey, Arc<UOp>> = HashMap::new();

    for u in root.toposort() {
        let Op::After { passthrough, deps } = u.op() else {
            continue;
        };

        // Use buf_uop() to get underlying buffer ID (handles AFTER chains)
        let buf_id = passthrough.buf_uop().id;
        kernel_assign.insert(buf_id, u.clone());

        // Get kernel from deps (first dep that is a kernel, may be wrapped in End)
        let Some(kernel) = deps.iter().find_map(|d| match d.op() {
            Op::Kernel { .. } => Some(d.clone()),
            Op::End { computation, .. } if matches!(computation.op(), Op::Kernel { .. }) => Some(computation.clone()),
            _ => None,
        }) else {
            continue;
        };

        let Op::Kernel { sources, .. } = kernel.op() else {
            continue;
        };

        for s in sources {
            // Check kernel sources for buffer dependencies
            if !matches!(s.op(), Op::Buffer { .. } | Op::DefineGlobal(_) | Op::After { .. }) {
                continue;
            }
            let s_buf_id = s.buf_uop().id;
            if s_buf_id == buf_id {
                continue;
            }
            let Some(a) = kernel_assign.get(&s_buf_id) else {
                continue;
            };

            // Cycle detection (like Tinygrad rangeify.py:577-578)
            if u.toposort().iter().any(|x| matches!(x.op(), Op::After { .. }) && x.buf_uop().id == s_buf_id) {
                panic!("cycle detected in graph, kernel for buffer must either depend on AFTER or BUFFER");
            }

            // Add dependency: a.replace(src=a.src+(u,))
            if let Op::After { passthrough: a_passthrough, deps: a_deps } = a.op() {
                let mut new_deps = a_deps.clone();
                new_deps.push(u.clone());
                let new_a = a_passthrough.after(new_deps);
                assign_rep.insert(UOpKey(a.clone()), new_a.clone());
                kernel_assign.insert(s_buf_id, new_a);
            }
        }
    }

    if assign_rep.is_empty() { root.clone() } else { root.substitute(&assign_rep) }
}

// ============================================================================
// PIPELINE
// ============================================================================

/// Run the kernel splitting pipeline.
///
/// Based on Tinygrad's get_rangeify_map (rangeify.py:565-580).
/// Simplified from ~200 lines to ~40 lines.
///
/// # Returns
/// Returns `(result, KernelContext)` tuple for backward compatibility with 30+ callers.
pub fn run_kernel_split_pipeline(root: Arc<UOp>) -> (Arc<UOp>, KernelContext) {
    use super::transforms::pm_add_buffers_local_patterns;
    use crate::rewrite::graph_rewrite_bottom_up;

    let ctx = KernelContext::new(); // Keep for compatibility

    // Phase 1: bufferize -> store (like Tinygrad rangeify.py:565)
    // Using pm_add_buffers_local_patterns (allow_locals=true) to create DEFINE_LOCAL
    // for local address space BUFFERIZE ops. The raw stage (allow_locals=false)
    // is used in the full optimizer pipeline after filtering.
    let after_buffers = {
        let matcher = pm_add_buffers_local_patterns();
        graph_rewrite_bottom_up(&matcher, root, &mut ())
    };

    trace!(tree = %after_buffers.tree_full(), "after pm_add_buffers");

    // Phase 2: split kernels (like Tinygrad rangeify.py:566)
    // We manually transform STORE/END → KERNEL to avoid reprocessing transformed nodes
    let after_split = split_all_stores(&after_buffers);

    // Phase 3: fix_assign (like Tinygrad rangeify.py:568-580)
    let result = fix_assign(&after_split);

    (result, ctx) // Keep tuple return for compatibility
}

/// Split all STORE/END operations into KERNELs.
///
/// This function manually traverses the tree and transforms STORE/END nodes,
/// avoiding the issue where graph_rewrite_bottom_up would reprocess
/// transformed nodes that still contain END operations.
fn split_all_stores(root: &Arc<UOp>) -> Arc<UOp> {
    use morok_ir::UOpKey;

    #[allow(clippy::mutable_key_type)]
    let mut replacements: HashMap<UOpKey, Arc<UOp>> = HashMap::new();
    let mut ctx = Vec::new();

    // Find all AFTER nodes that contain STORE/END in their deps
    for node in root.toposort() {
        if let Op::After { passthrough, deps } = node.op() {
            // Transform each dep that is STORE, END(STORE), or BARRIER(END(STORE))
            let mut new_deps = SmallVec::new();
            let mut any_changed = false;

            for dep in deps {
                let transformed_dep = transform_store_to_kernel(dep, &mut ctx);
                if !Arc::ptr_eq(&transformed_dep, dep) {
                    any_changed = true;
                }
                new_deps.push(transformed_dep);
            }

            if any_changed {
                let new_after = passthrough.after(new_deps);
                replacements.insert(UOpKey(node.clone()), new_after);
            }
        }
    }

    if replacements.is_empty() { root.clone() } else { root.substitute(&replacements) }
}

/// Transform a single STORE/END/BARRIER node into a KERNEL.
fn transform_store_to_kernel(node: &Arc<UOp>, ctx: &mut Vec<Arc<UOp>>) -> Arc<UOp> {
    match node.op() {
        Op::Store { .. } => {
            if let Some(kernel) = split_store(ctx, node) {
                // Wrap kernel in END for proper structure
                kernel.end(SmallVec::new())
            } else {
                node.clone()
            }
        }
        Op::End { computation, ranges } => {
            if matches!(computation.op(), Op::Store { .. }) {
                if let Some(kernel) = split_store(ctx, node) {
                    // Keep END wrapper with original ranges
                    kernel.end(ranges.clone())
                } else {
                    node.clone()
                }
            } else {
                node.clone()
            }
        }
        Op::Barrier { src, deps } => {
            // Recursively transform the barrier source
            let transformed_src = transform_store_to_kernel(src, ctx);
            if Arc::ptr_eq(&transformed_src, src) { node.clone() } else { transformed_src.barrier(deps.clone()) }
        }
        _ => node.clone(),
    }
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

    let indexed = match UOp::index().buffer(Arc::clone(source)).indices(ranges).call() {
        Ok(idx) => idx,
        Err(_) => return vec![false; input_shape.len()],
    };

    let base = source.base();
    let noop = UOp::noop();
    #[allow(clippy::mutable_key_type)]
    let mut substitutions = HashMap::new();
    substitutions.insert(UOpKey(base), noop);

    let substituted = indexed.substitute(&substitutions);

    use super::patterns::{movement_op_patterns, pm_syntactic_sugar};
    use crate::rewrite::graph_rewrite_top_down;

    // Tinygrad: pm_mops + pm_syntactic_sugar (early movement ops)
    let pm_mops = movement_op_patterns() + pm_syntactic_sugar();
    let transformed = graph_rewrite_top_down(&pm_mops, substituted, &mut ());

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

    let contiguous = first_reduce.contiguous();

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
