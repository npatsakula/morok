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
    /// Mapping from original DefineGlobal UOp ID to renumbered UOp ID.
    /// Used for buffer registry dual-registration so consumers can find
    /// shared buffers using the original (pre-renumbered) ID from AFTER nodes.
    pub buffer_id_mapping: HashMap<u64, u64>,
    /// Mapping from DefineGlobal UOp ID to original BUFFER UOp ID.
    /// Each DefineGlobal created from a BUFFER gets one entry (never overwritten).
    /// Used to find input buffers when `buffer_map` entries are overwritten.
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
            kernel_deps: Vec::new(),
            buffer_id_mapping: HashMap::new(),
            define_to_buffer_id: HashMap::new(),
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

    // Handle AFTER wrapping STORE/END - PRESERVE THE WRAPPER (like Tinygrad)
    // Transform deps in-place and return new AFTER with transformed deps
    if let Op::After { passthrough, deps } = uop.op() {
        let mut new_deps = SmallVec::new();
        let mut any_transformed = false;

        for dep in deps.iter() {
            match dep.op() {
                Op::Kernel { .. } => {
                    // Already a kernel - keep as is
                    new_deps.push(dep.clone());
                }
                Op::End { computation, .. } if matches!(computation.op(), Op::Kernel { .. }) => {
                    // Already END(Kernel) - keep as is
                    new_deps.push(dep.clone());
                }
                Op::Store { .. } | Op::StoreGated { .. } => {
                    // Transform STORE to KERNEL, wrap in END
                    if let Some(kernel) = split_store(dep, ctx) {
                        let end = UOp::end(kernel, SmallVec::new());
                        new_deps.push(end);
                        any_transformed = true;
                    } else {
                        new_deps.push(dep.clone());
                    }
                }
                Op::End { computation, ranges }
                    if matches!(computation.op(), Op::Store { .. } | Op::StoreGated { .. }) =>
                {
                    // Transform END(STORE) to END(KERNEL)
                    if let Some(kernel) = split_store(computation, ctx) {
                        let end = UOp::end(kernel, ranges.clone());
                        new_deps.push(end);
                        any_transformed = true;
                    } else {
                        new_deps.push(dep.clone());
                    }
                }
                Op::Barrier { src, deps: barrier_deps } => {
                    // Handle BARRIER wrapping END(STORE)
                    let transformed_src = if let Op::End { computation, ranges } = src.op()
                        && matches!(computation.op(), Op::Store { .. } | Op::StoreGated { .. })
                    {
                        if let Some(kernel) = split_store(computation, ctx) {
                            let end = UOp::end(kernel, ranges.clone());
                            any_transformed = true;
                            end
                        } else {
                            src.clone()
                        }
                    } else if matches!(src.op(), Op::Store { .. } | Op::StoreGated { .. }) {
                        if let Some(kernel) = split_store(src, ctx) {
                            let end = UOp::end(kernel, SmallVec::new());
                            any_transformed = true;
                            end
                        } else {
                            src.clone()
                        }
                    } else {
                        src.clone()
                    };
                    let new_barrier = UOp::barrier(transformed_src, barrier_deps.clone());
                    new_deps.push(new_barrier);
                }
                _ => {
                    new_deps.push(dep.clone());
                }
            }
        }

        if any_transformed {
            // Return NEW AFTER with transformed deps - PRESERVES WRAPPER
            return Some(UOp::after(passthrough.clone(), new_deps));
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
        // Debug: show BUFFER nodes in computation before transformation
        if tracing::enabled!(tracing::Level::DEBUG) {
            let buffer_nodes: Vec<_> = computation
                .toposort()
                .into_iter()
                .filter(|n| matches!(n.op(), Op::Buffer { .. }))
                .map(|n| n.id)
                .collect();
            debug!(
                computation.id = computation.id,
                buffer_nodes = ?buffer_nodes,
                "BUFFER nodes before to_define_global"
            );
        }
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

    // eprintln!("=== KERNEL AST ===\n{}", ast.tree_full());

    // Collect buffer sources by traversing the kernel's AST.
    // We find buffers in the AST, then check ctx.buffer_map for AFTER wrappers.
    // If a buffer has an AFTER wrapper (from a producer kernel), we use the AFTER
    // as the kernel source to track inter-kernel dependencies.
    let mut buffer_sources: Vec<Arc<UOp>> = Vec::new();
    let mut seen_buffer_ids: HashSet<u64> = HashSet::new();

    for node in ast.toposort() {
        let buffer = match node.op() {
            Op::DefineGlobal(_) | Op::DefineLocal(_) => Some(node.clone()),
            Op::Index { buffer, .. } | Op::Load { buffer, .. } => {
                // Get the actual buffer from Index/Load
                match buffer.op() {
                    Op::DefineGlobal(_) | Op::DefineLocal(_) => Some(buffer.clone()),
                    _ => None,
                }
            }
            Op::Store { buffer, .. } | Op::StoreGated { buffer, .. } => {
                // Get output buffer from Store
                match buffer.op() {
                    Op::DefineGlobal(_) | Op::DefineLocal(_) => Some(buffer.clone()),
                    Op::Index { buffer: inner, .. } => {
                        if matches!(inner.op(), Op::DefineGlobal(_) | Op::DefineLocal(_)) {
                            Some(inner.clone())
                        } else {
                            None
                        }
                    }
                    _ => None,
                }
            }
            _ => None,
        };

        if let Some(buf) = buffer
            && seen_buffer_ids.insert(buf.id)
        {
            // Check if this buffer has an AFTER wrapper in buffer_map.
            // If so, use the AFTER as the kernel source (tracks inter-kernel deps).
            // Otherwise, use the bare DefineGlobal/DefineLocal.
            let source = ctx.get_buffer(&buf).cloned().unwrap_or_else(|| buf.clone());
            trace!(
                buffer.id = buf.id,
                buffer.op = ?std::mem::discriminant(buf.op()),
                has_after = ctx.has_buffer(&buf),
                source.id = source.id,
                source.op = ?std::mem::discriminant(source.op()),
                "Collecting buffer source"
            );
            buffer_sources.push(source);
        }
    }

    // Sort buffer sources by their underlying DefineGlobal/DefineLocal index.
    // AFTER wrappers need to be unwrapped to get the underlying buffer.
    buffer_sources.sort_by_key(|b| {
        let inner = match b.op() {
            Op::After { passthrough, .. } => passthrough,
            _ => b,
        };
        match inner.op() {
            Op::DefineGlobal(id) => *id as u64,
            Op::DefineLocal(id) => (*id as u64) + (1u64 << 32),
            Op::Buffer { .. } => inner.id + (1u64 << 48),
            _ => inner.id,
        }
    });

    // Renumber DefineGlobal/DefineLocal indices to match buffer positions.
    // This ensures kernel AST references match the buffer list order.
    let (ast, buffer_sources, id_mapping) = renumber_define_globals(ast, &buffer_sources);

    // Store the ID mapping for buffer registry dual-registration.
    // This allows consumers to find shared buffers using original IDs from AFTER nodes.
    ctx.buffer_id_mapping.extend(id_mapping);

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

/// Fix inter-kernel dependencies (like Tinygrad's fix_assign).
///
/// When kernel B reads from a buffer that kernel A writes to, this function
/// ensures kernel B's AFTER node depends on kernel A's AFTER node.
///
/// This handles cases like `Eq(data, Expand(ReduceAxis(data)))` where:
/// 1. ReduceAxis creates kernel A writing to buffer X
/// 2. Eq creates kernel B reading from buffer X
/// 3. Kernel B's AFTER must depend on Kernel A's AFTER
fn fix_assign(root: &Arc<UOp>) -> Arc<UOp> {
    use morok_ir::UOpKey;

    if tracing::enabled!(tracing::Level::DEBUG) {
        let toposort_nodes: Vec<_> = root.toposort();
        let after_count = toposort_nodes.iter().filter(|n| matches!(n.op(), Op::After { .. })).count();
        let kernel_count = toposort_nodes.iter().filter(|n| matches!(n.op(), Op::Kernel { .. })).count();
        debug!(
            root.id = root.id,
            total_nodes = toposort_nodes.len(),
            after_count,
            kernel_count,
            "fix_assign starting"
        );
    }

    // Step 1: Map buffer_id -> AFTER node that writes to it
    let mut kernel_assign: HashMap<u64, Arc<UOp>> = HashMap::new();

    for node in root.toposort() {
        if let Op::After { passthrough, deps } = node.op() {
            trace!(
                after.id = node.id,
                passthrough.id = passthrough.id,
                passthrough.op = ?std::mem::discriminant(passthrough.op()),
                num_deps = deps.len(),
                "Found AFTER node"
            );
            let buf_id = match passthrough.op() {
                Op::DefineGlobal(_) => passthrough.id,
                _ => continue,
            };
            // Only record if there's a Kernel dep (this AFTER produces a buffer)
            // Note: The kernel may be wrapped in an End node: AFTER(DefineGlobal, [End(Kernel, ranges)])
            let has_kernel = deps.iter().any(|d| match d.op() {
                Op::Kernel { .. } => true,
                Op::End { computation, .. } => matches!(computation.op(), Op::Kernel { .. }),
                _ => false,
            });
            if has_kernel {
                trace!(buffer.id = buf_id, after.id = node.id, "recording producer");
                kernel_assign.insert(buf_id, node.clone());
            }
        }
    }

    if kernel_assign.is_empty() {
        trace!("no kernel producers found");
        return root.clone();
    }

    // Step 2: Find AFTER nodes whose kernels read from buffers written by other kernels
    #[allow(clippy::mutable_key_type)]
    let mut assign_rep: HashMap<UOpKey, Arc<UOp>> = HashMap::new();

    for node in root.toposort() {
        if let Op::After { passthrough, deps } = node.op() {
            let this_buf_id = match passthrough.op() {
                Op::DefineGlobal(_) => passthrough.id,
                _ => continue,
            };

            // Find the kernel in deps (may be wrapped in End node)
            let kernel = deps.iter().find_map(|d| match d.op() {
                Op::Kernel { .. } => Some(d.clone()),
                Op::End { computation, .. } if matches!(computation.op(), Op::Kernel { .. }) => {
                    Some(computation.clone())
                }
                _ => None,
            });
            let Some(kernel) = kernel else {
                continue;
            };

            // Check each source of the kernel for buffer dependencies
            if let Op::Kernel { sources, .. } = kernel.op() {
                trace!(
                    after.id = node.id,
                    buffer.id = this_buf_id,
                    num_sources = sources.len(),
                    "Checking kernel sources"
                );
                for src in sources {
                    // Get buffer ID from source (could be DefineGlobal or AFTER wrapping one)
                    let src_buf_id = match src.op() {
                        Op::DefineGlobal(_) => src.id,
                        Op::After { passthrough: p, .. } if matches!(p.op(), Op::DefineGlobal(_)) => p.id,
                        _ => continue,
                    };

                    trace!(
                        source.id = src.id,
                        source.buffer_id = src_buf_id,
                        has_producer = kernel_assign.contains_key(&src_buf_id),
                        "Checking source buffer"
                    );

                    // Skip if same buffer as this AFTER produces
                    if src_buf_id == this_buf_id {
                        continue;
                    }

                    // Check if another kernel writes to this buffer
                    if let Some(producer_after) = kernel_assign.get(&src_buf_id) {
                        // Don't add dependency on self
                        if Arc::ptr_eq(producer_after, &node) {
                            continue;
                        }

                        // Add the producer AFTER as a dependency if not already present
                        let mut new_deps = deps.clone();
                        if !new_deps.iter().any(|d| Arc::ptr_eq(d, producer_after)) {
                            debug!(
                                consumer.after_id = node.id,
                                producer.after_id = producer_after.id,
                                producer.buffer_id = src_buf_id,
                                "Adding inter-kernel dependency"
                            );
                            new_deps.push(producer_after.clone());
                            let new_after = UOp::after(passthrough.clone(), new_deps);
                            assign_rep.insert(UOpKey(node.clone()), new_after.clone());
                            // Update kernel_assign to point to the new AFTER
                            kernel_assign.insert(this_buf_id, new_after);
                        }
                    }
                }
            }
        }
    }

    debug!(num_replacements = assign_rep.len(), "fix_assign completed");

    if assign_rep.is_empty() { root.clone() } else { root.substitute(&assign_rep) }
}

// ============================================================================
// PIPELINE
// ============================================================================

/// Run the kernel splitting pipeline.
pub fn run_kernel_split_pipeline(root: Arc<UOp>) -> (Arc<UOp>, KernelContext) {
    use super::transforms::bufferize_to_store;

    let mut ctx = KernelContext::new();
    let mut memo: HashMap<u64, Arc<UOp>> = HashMap::new();

    // Phase 1: Convert BUFFERIZE to STORE with memoization for DAG handling
    let after_bufferize = transform_bottom_up_memo(&root, &mut ctx, bufferize_to_store, &mut memo);

    // Trace after bufferize_to_store
    trace!(tree = %after_bufferize.tree_full(), "after bufferize_to_store");

    // Phase 2: Split STORE to KERNEL with fresh memoization
    memo.clear();
    let after_split = transform_bottom_up_memo(&after_bufferize, &mut ctx, split_store, &mut memo);

    // Phase 2.5: Collect producer AFTERs from memo that have kernel deps
    // These AFTERs may have been transformed but not connected to the final output
    let producer_afters: Vec<Arc<UOp>> = memo
        .values()
        .filter(|uop| {
            if let Op::After { deps, .. } = uop.op() {
                deps.iter().any(|d| {
                    matches!(d.op(), Op::Kernel { .. })
                        || matches!(d.op(), Op::End { computation, .. } if matches!(computation.op(), Op::Kernel { .. }))
                })
            } else {
                false
            }
        })
        .cloned()
        .collect();

    // Phase 2.6: Ensure producer AFTERs are in the AST by adding to SINK
    // This handles the case where producer kernels are embedded in consumer computations
    let after_split = if !producer_afters.is_empty() {
        ensure_producer_afters_in_sink(&after_split, &producer_afters)
    } else {
        after_split
    };

    // Phase 3: Fix inter-kernel dependencies (like Tinygrad's fix_assign)
    let after_split = fix_assign(&after_split);

    // Phase 4: Resolve kernel dependencies for scheduling
    resolve_kernel_dependencies(&after_split, &mut ctx);

    (after_split, ctx)
}

/// Ensure producer AFTERs are connected to the SINK.
///
/// When kernel B reads from a buffer produced by kernel A, kernel A's AFTER
/// may have been transformed but not connected to the output. This function
/// ensures all producer AFTERs appear in the final AST by adding them to the SINK.
fn ensure_producer_afters_in_sink(root: &Arc<UOp>, producer_afters: &[Arc<UOp>]) -> Arc<UOp> {
    // Find existing AFTER nodes in the AST to avoid duplicates
    let existing_afters: HashSet<u64> = root
        .toposort()
        .into_iter()
        .filter(|n| matches!(n.op(), Op::After { .. }))
        .map(|n| {
            // Key by passthrough buffer ID for deduplication
            if let Op::After { passthrough, .. } = n.op() { passthrough.id } else { n.id }
        })
        .collect();

    // Find producer AFTERs that aren't already in the AST
    let missing_afters: Vec<_> = producer_afters
        .iter()
        .filter(|a| {
            if let Op::After { passthrough, .. } = a.op() { !existing_afters.contains(&passthrough.id) } else { false }
        })
        .cloned()
        .collect();

    if missing_afters.is_empty() {
        return root.clone();
    }

    debug!(num_missing = missing_afters.len(), "adding missing producer afters to sink");
    for after in &missing_afters {
        if let Op::After { passthrough, .. } = after.op() {
            trace!(after.id = after.id, passthrough.id = passthrough.id, "missing after");
        }
    }

    // Add missing AFTERs to the SINK
    if let Op::Sink { sources } = root.op() {
        let mut new_sources: Vec<Arc<UOp>> = sources.iter().cloned().collect();
        new_sources.extend(missing_afters);
        UOp::sink(new_sources)
    } else {
        // Root is not a SINK - wrap it with missing AFTERs
        let mut new_sources = vec![root.clone()];
        new_sources.extend(missing_afters);
        UOp::sink(new_sources)
    }
}

fn resolve_kernel_dependencies(root: &Arc<UOp>, ctx: &mut KernelContext) {
    let mut buffer_producers: HashMap<u64, Arc<UOp>> = HashMap::new();

    for node in root.toposort() {
        if let Op::After { passthrough, deps } = node.op()
            && let Op::DefineGlobal(_) = passthrough.op()
        {
            // Look for Kernel in deps (may be wrapped in End node)
            for dep in deps {
                let kernel = match dep.op() {
                    Op::Kernel { .. } => Some(dep.clone()),
                    Op::End { computation, .. } if matches!(computation.op(), Op::Kernel { .. }) => {
                        Some(computation.clone())
                    }
                    _ => None,
                };
                if let Some(k) = kernel {
                    trace!(
                        buffer.id = passthrough.id,
                        kernel.id = k.id,
                        "Buffer produced by kernel"
                    );
                    buffer_producers.insert(passthrough.id, k);
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
                    trace!(
                        consumer.kernel_id = node.id,
                        producer.kernel_id = producer.id,
                        buffer.id = buf_id,
                        "Kernel dependency found"
                    );
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

/// Memoized version of transform_bottom_up for proper DAG handling.
/// Without memoization, the same node referenced from multiple places
/// gets processed independently, breaking multi-kernel dependencies.
fn transform_bottom_up_memo<F>(
    uop: &Arc<UOp>,
    ctx: &mut KernelContext,
    transform_fn: F,
    memo: &mut HashMap<u64, Arc<UOp>>,
) -> Arc<UOp>
where
    F: Fn(&Arc<UOp>, &mut KernelContext) -> Option<Arc<UOp>> + Copy,
{
    // Check memo first - return cached result for same UOp id
    if let Some(cached) = memo.get(&uop.id) {
        return cached.clone();
    }

    let sources = uop.op().sources();

    if sources.is_empty() {
        let result = transform_fn(uop, ctx).unwrap_or_else(|| uop.clone());
        memo.insert(uop.id, result.clone());
        return result;
    }

    let mut transformed_sources = Vec::with_capacity(sources.len());
    let mut any_changed = false;

    for src in sources {
        let transformed = transform_bottom_up_memo(&src, ctx, transform_fn, memo);
        if !Arc::ptr_eq(&transformed, &src) {
            any_changed = true;
        }
        transformed_sources.push(transformed);
    }

    let reconstructed = if any_changed { uop.with_sources(transformed_sources) } else { uop.clone() };

    let result = transform_fn(&reconstructed, ctx).unwrap_or(reconstructed);
    memo.insert(uop.id, result.clone());
    result
}

/// Renumber DefineGlobal/DefineLocal indices in AST to match buffer positions.
///
/// When kernels are split, each kernel collects only the DefineGlobal nodes
/// present in its AST. However, the indices (the arg in DefineGlobal(arg))
/// are assigned globally during transformation. This creates a mismatch:
/// - Kernel sources: [DefineGlobal(0), DefineGlobal(3)]
/// - Buffer list built from sources: positions [0, 1]
/// - But AST references DefineGlobal(3), which codegen interprets as index 3
///
/// This function renumbers the AST so DefineGlobal indices match their
/// position in buffer_sources: 0, 1, 2, ... for sequential buffer access.
///
/// Returns:
/// - Rewritten AST with renumbered DefineGlobal/DefineLocal indices
/// - New buffer sources with renumbered nodes
/// - Mapping from original UOp ID to new UOp ID (for buffer registry dual-registration)
fn renumber_define_globals(ast: Arc<UOp>, buffer_sources: &[Arc<UOp>]) -> (Arc<UOp>, Vec<Arc<UOp>>, HashMap<u64, u64>) {
    // Helper to extract the inner buffer from an AFTER wrapper or return the node itself.
    fn get_inner_buffer(src: &Arc<UOp>) -> &Arc<UOp> {
        match src.op() {
            Op::After { passthrough, .. } => passthrough,
            _ => src,
        }
    }

    // Build old UOp id -> new index mapping
    // DefineGlobals come first (indices 0..num_globals-1)
    // DefineLocals come after (indices num_globals..num_globals+num_locals-1)
    let mut id_to_new_idx: HashMap<u64, usize> = HashMap::new();
    let mut global_idx = 0usize;
    let mut local_start = 0usize;

    // First pass: count globals to know where locals start
    for src in buffer_sources {
        let inner = get_inner_buffer(src);
        if matches!(inner.op(), Op::DefineGlobal(_)) {
            local_start += 1;
        }
    }

    // Second pass: assign new indices
    let mut local_idx = local_start;
    for src in buffer_sources {
        let inner = get_inner_buffer(src);
        match inner.op() {
            Op::DefineGlobal(_) => {
                id_to_new_idx.insert(inner.id, global_idx);
                global_idx += 1;
            }
            Op::DefineLocal(_) => {
                id_to_new_idx.insert(inner.id, local_idx);
                local_idx += 1;
            }
            _ => {}
        }
    }

    // Build new buffer_sources with renumbered nodes.
    // For AFTER wrappers, create new AFTER with renumbered inner buffer.
    let new_sources: Vec<Arc<UOp>> = buffer_sources
        .iter()
        .map(|src| {
            let inner = get_inner_buffer(src);
            if let Some(&new_idx) = id_to_new_idx.get(&inner.id) {
                let new_inner = match inner.op() {
                    Op::DefineGlobal(_) => UOp::define_global(new_idx, inner.dtype().clone()),
                    Op::DefineLocal(_) => UOp::define_local(new_idx, inner.dtype().clone()),
                    _ => return src.clone(),
                };
                // Preserve AFTER wrapper if present
                let result = match src.op() {
                    Op::After { deps, .. } => UOp::after(new_inner.clone(), deps.clone()),
                    _ => new_inner.clone(),
                };
                trace!(
                    source.id = src.id,
                    inner.id = inner.id,
                    new_inner.id = new_inner.id,
                    result.id = result.id,
                    is_after = matches!(src.op(), Op::After { .. }),
                    "Renumbering DefineGlobal"
                );
                result
            } else {
                src.clone()
            }
        })
        .collect();

    // Build UOp id -> new UOp mapping for AST rewrite.
    // Map from INNER buffer ID (what appears in AST) to renumbered INNER buffer.
    let old_to_new: HashMap<u64, Arc<UOp>> = buffer_sources
        .iter()
        .zip(new_sources.iter())
        .map(|(old, new)| {
            let old_inner = get_inner_buffer(old);
            let new_inner = get_inner_buffer(new);
            (old_inner.id, new_inner.clone())
        })
        .collect();

    // Rewrite AST: replace all DefineGlobal/DefineLocal references
    fn rewrite_ast(
        node: &Arc<UOp>,
        old_to_new: &HashMap<u64, Arc<UOp>>,
        memo: &mut HashMap<u64, Arc<UOp>>,
    ) -> Arc<UOp> {
        if let Some(cached) = memo.get(&node.id) {
            return cached.clone();
        }

        // If this node is a DefineGlobal/Local that needs replacement
        if let Some(replacement) = old_to_new.get(&node.id) {
            memo.insert(node.id, replacement.clone());
            return replacement.clone();
        }

        let sources = node.op().sources();
        if sources.is_empty() {
            memo.insert(node.id, node.clone());
            return node.clone();
        }

        let mut new_sources = Vec::with_capacity(sources.len());
        let mut changed = false;
        for src in &sources {
            let new_src = rewrite_ast(src, old_to_new, memo);
            if !Arc::ptr_eq(&new_src, src) {
                changed = true;
            }
            new_sources.push(new_src);
        }

        let result = if changed { node.with_sources(new_sources) } else { node.clone() };
        memo.insert(node.id, result.clone());
        result
    }

    let mut memo = HashMap::new();
    let new_ast = rewrite_ast(&ast, &old_to_new, &mut memo);

    // Build old_id -> new_id mapping for buffer registry dual-registration.
    // This allows consumers to find shared buffers using the original (pre-renumbered) ID.
    // For AFTER wrappers, we map the inner buffer IDs.
    let id_mapping: HashMap<u64, u64> = buffer_sources
        .iter()
        .zip(new_sources.iter())
        .filter_map(|(old, new)| {
            let old_inner = get_inner_buffer(old);
            let new_inner = get_inner_buffer(new);
            if matches!(old_inner.op(), Op::DefineGlobal(_) | Op::DefineLocal(_)) {
                Some((old_inner.id, new_inner.id))
            } else {
                None
            }
        })
        .collect();

    (new_ast, new_sources, id_mapping)
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
