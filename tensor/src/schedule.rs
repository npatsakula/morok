//! Kernel scheduling types and execution.
//!
//! This module provides types and functions for managing the execution
//! schedule of tensor operations. After the rangeify pipeline transforms
//! the computation graph into KERNEL operations, we need to:
//!
//! 1. Extract kernel operations from the transformed graph
//! 2. Allocate buffers for intermediate results (PARAM/DEFINE_LOCAL)
//! 3. Execute kernels in dependency order
//!
//! The scheduling process converts from lazy tensor operations to
//! executable kernels with properly allocated device buffers.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use morok_device::Buffer;
use morok_device::device::Device;
use morok_device::registry;
use morok_dtype::{DType, DeviceSpec};
use morok_ir::{ConstValueHash, Op, UOp};
use tracing::{debug, trace};

use crate::error::*;
use crate::{Error, Result};
use snafu::ResultExt;

fn source_primary_buffer_id(src: &Arc<UOp>) -> Option<u64> {
    match src.op() {
        Op::Buffer { .. } | Op::Param { .. } | Op::After { .. } => Some(src.buf_uop().id),
        Op::MSelect { buffer, device_index } => {
            if let Op::MStack { buffers } = buffer.op() {
                buffers.get(*device_index).map(|b| b.buf_uop().id).or_else(|| Some(src.buf_uop().id))
            } else {
                Some(src.buf_uop().id)
            }
        }
        Op::MStack { buffers } => buffers.first().map(|b| b.buf_uop().id),
        _ => None,
    }
}

fn source_dependency_buffer_ids(src: &Arc<UOp>) -> Vec<u64> {
    match src.op() {
        Op::MStack { buffers } => buffers.iter().map(|b| b.buf_uop().id).collect(),
        Op::MSelect { buffer, device_index } => {
            if let Op::MStack { buffers } = buffer.op() {
                let mut ids: Vec<u64> = buffers.iter().map(|b| b.buf_uop().id).collect();
                if let Some(selected) = buffers.get(*device_index) {
                    ids.push(selected.buf_uop().id);
                }
                ids.sort_unstable();
                ids.dedup();
                return ids;
            }
            vec![src.buf_uop().id]
        }
        Op::Buffer { .. } | Op::Param { .. } | Op::After { .. } => vec![src.buf_uop().id],
        _ => vec![],
    }
}

fn analyze_kernel_dependencies(kernels: &[Arc<UOp>], root: &Arc<UOp>) -> Vec<HashSet<usize>> {
    // Build kernel ID → index mapping
    let kernel_idx: HashMap<u64, usize> = kernels.iter().enumerate().map(|(i, k)| (k.id, i)).collect();

    // Map buffer_id → writer kernel index (the kernel that writes to this buffer)
    let mut buf_to_writer: HashMap<u64, usize> = HashMap::new();

    // Find AFTER nodes and map buffers to their writer kernels
    for node in root.toposort() {
        if let Op::After { passthrough, deps } = node.op() {
            // Find the kernel in deps (may be wrapped in END)
            let kernel = deps.iter().find_map(|d| match d.op() {
                Op::Kernel { .. } => Some(d.clone()),
                Op::End { computation, .. } if matches!(computation.op(), Op::Kernel { .. }) => {
                    Some(computation.clone())
                }
                _ => None,
            });

            if let Some(k) = kernel
                && let Some(&idx) = kernel_idx.get(&k.id)
            {
                // Use buf_uop() to get underlying buffer ID (handles AFTER chains)
                let buf_id = passthrough.buf_uop().id;
                buf_to_writer.insert(buf_id, idx);
            }
        }
    }

    // Build dependency edges from kernel sources
    // A kernel depends on the writer of any buffer it reads from
    let mut dependencies: Vec<HashSet<usize>> = vec![HashSet::new(); kernels.len()];
    for (idx, kernel) in kernels.iter().enumerate() {
        if let Op::Kernel { sources, .. } = kernel.op() {
            for src in sources {
                for buf_id in source_dependency_buffer_ids(src) {
                    // If another kernel writes this buffer, we depend on it
                    if let Some(&writer_idx) = buf_to_writer.get(&buf_id)
                        && writer_idx != idx
                    {
                        dependencies[idx].insert(writer_idx);
                    }
                }
            }
        }
    }

    dependencies
}

/// Input buffers collected before schedule creation.
///
/// Maps BUFFER UOp ID → Buffer for input tensors.
/// This allows schedule creation to find input buffers without
/// global registry lookups (RAII migration).
pub type InputBuffers = HashMap<u64, Buffer>;

/// A bound range variable that needs to be iterated over.
///
/// This represents a BIND(DEFINE_VAR, RANGE) node from the kernel AST.
/// The scheduler will expand these into concrete iteration values.
#[derive(Clone, Debug)]
pub struct BoundRange {
    /// Variable name (e.g., "range_0")
    pub var_name: String,
    /// The RANGE UOp that defines the iteration space
    pub range_uop: Arc<UOp>,
}

/// A single executable kernel with its buffers and variable bindings.
///
/// Each ScheduleItem represents one kernel that needs to be compiled
/// and executed. The kernel AST contains STORE operations that write
/// results to buffers.
///
/// For kernels with OUTER ranges (represented as BIND variables), the
/// scheduler will expand one ScheduleItem into N items with concrete
/// variable values in fixedvars.
#[derive(Clone)]
pub struct ScheduleItem {
    /// The KERNEL wrapper UOp (for buffer allocation)
    pub kernel: Arc<UOp>,

    /// The inner kernel AST (SINK containing STORE ops) - for codegen
    pub ast: Arc<UOp>,

    /// Device buffers for this kernel (in order expected by codegen)
    pub buffers: Vec<Buffer>,

    /// UOp IDs under which each buffer was registered in buffer index.
    /// Same length as `buffers`. Used for cleanup - to remove buffers from
    /// the global registry, we need to know what key they were registered under.
    pub buffer_uop_ids: Vec<u64>,

    /// Fixed variable values for this specific kernel invocation.
    /// Maps variable name (e.g., "range_0") to concrete i64 value.
    /// Empty for unexpanded schedule items.
    pub fixedvars: HashMap<String, i64>,

    /// Bound ranges that need to be expanded into iterations.
    /// Non-empty only for unexpanded schedule items.
    /// After expansion, this will be empty and fixedvars will be populated.
    pub bound_ranges: Vec<BoundRange>,

    /// KERNEL UOp IDs that must complete before this kernel can execute.
    /// Empty for kernels without dependencies (first in chain or independent).
    /// Dependencies are implicit in kernel ordering after topological sort.
    pub dependencies: Vec<u64>,

    /// Additional UOp IDs registered as aliases in buffer index.
    /// These are IDs where the same buffer was registered under a different key
    /// for lookup convenience. They need to be cleaned up along with buffer_uop_ids.
    pub alias_registered_ids: Vec<u64>,
}

/// Full execution schedule (list of kernels in dependency order).
pub type Schedule = Vec<ScheduleItem>;

/// Result of schedule creation, including output buffer identification.
pub struct ScheduleResult {
    /// The schedule items in dependency order.
    pub items: Schedule,
    /// UOp IDs of output buffers, in SINK source order.
    /// Extracted directly from the SINK's sources via `buf_uop()`.
    /// For single-tensor realize, contains one ID.
    pub output_uop_ids: Vec<u64>,
}

/// Buffers collected for a single kernel.
struct KernelBuffers {
    /// Device buffers in codegen order.
    buffers: Vec<Buffer>,
    /// UOp IDs for each buffer.
    uop_ids: Vec<u64>,
    /// Additional alias IDs for cleanup.
    alias_ids: Vec<u64>,
}

/// Sort kernels by dependencies (producers before consumers).
///
/// Uses Kahn's algorithm for topological sort based on buffer dependencies
/// derived from the graph structure (AFTER nodes and kernel sources).
/// This ensures producer kernels are processed before consumers, which is
/// critical for buffer sharing: the producer allocates the buffer first,
/// then the consumer finds it in the registry via `get_or_create_buffer`.
fn sort_kernels_by_dependencies(
    kernels: &[Arc<UOp>],
    root: &Arc<UOp>,
) -> Result<(Vec<Arc<UOp>>, HashMap<u64, Vec<u64>>)> {
    debug!(num_kernels = kernels.len(), "sorting kernels by dependencies");

    let dependencies = analyze_kernel_dependencies(kernels, root);

    // Kahn's algorithm for topological sort
    let mut in_degree: Vec<usize> = dependencies.iter().map(|deps| deps.len()).collect();
    let mut dependents: Vec<Vec<usize>> = vec![vec![]; kernels.len()];

    for (consumer, deps) in dependencies.iter().enumerate() {
        for &producer in deps {
            dependents[producer].push(consumer);
        }
    }

    let mut queue: VecDeque<usize> =
        in_degree.iter().enumerate().filter(|&(_, &deg)| deg == 0).map(|(idx, _)| idx).collect();

    let mut sorted_indices = Vec::new();
    while let Some(idx) = queue.pop_front() {
        sorted_indices.push(idx);
        for &dependent in &dependents[idx] {
            in_degree[dependent] -= 1;
            if in_degree[dependent] == 0 {
                queue.push_back(dependent);
            }
        }
    }

    if sorted_indices.len() < kernels.len() {
        return DependencyCyclesSnafu.fail();
    }

    let sorted: Vec<Arc<UOp>> = sorted_indices.iter().map(|&idx| kernels[idx].clone()).collect();

    let dependency_ids_by_kernel: HashMap<u64, Vec<u64>> = kernels
        .iter()
        .enumerate()
        .map(|(idx, kernel)| {
            let mut deps: Vec<u64> = dependencies[idx].iter().map(|&dep_idx| kernels[dep_idx].id).collect();
            deps.sort_unstable();
            (kernel.id, deps)
        })
        .collect();

    debug!(num_sorted = sorted.len(), "kernels sorted");

    Ok((sorted, dependency_ids_by_kernel))
}

/// Extract kernels from transformed graph and create schedule.
///
/// This function walks the transformed UOp graph (after rangeify and
/// kernel splitting) and extracts all KERNEL operations. For each kernel,
/// it identifies the buffers needed from the kernel's sources.
///
/// # Arguments
///
/// * `transformed` - The UOp graph after rangeify + kernel splitting
/// * `input_buffers` - Pre-collected input buffers (for RAII migration).
///   If a buffer is found here, it's used directly instead of registry lookup.
///
/// # Returns
///
/// A schedule of kernels to execute, with buffers attached.
///
/// # Errors
///
/// Returns error if:
/// - No kernels found after scheduling pipeline
/// - Buffer not found in registry for a kernel source
pub fn create_schedule(
    transformed: Arc<UOp>,
    input_buffers: &InputBuffers,
    var_vals: &HashMap<String, i64>,
) -> Result<ScheduleResult> {
    // Step 1: Find all KERNEL operations
    let mut kernels = Vec::new();
    for node in transformed.toposort() {
        if matches!(node.op(), Op::Kernel { .. }) {
            kernels.push(node);
        }
    }

    if kernels.is_empty() {
        return NoKernelsFoundSnafu.fail();
    }

    // Step 1.5: Sort kernels by dependencies (producers before consumers)
    // This ensures producer kernels are processed first, so they allocate buffers
    // before consumers look them up.
    let (kernels, dependency_ids_by_kernel) = sort_kernels_by_dependencies(&kernels, &transformed)?;

    // Track allocated intermediate buffers locally (no global registry needed)
    let mut allocated_buffers: HashMap<u64, Buffer> = HashMap::new();

    // Step 2: For each kernel, collect buffers from sources
    let mut schedule = Vec::new();

    for kernel_uop in kernels {
        let (sources, inner_ast) = match kernel_uop.op() {
            Op::Kernel { sources, ast } => (sources, ast),
            _ => unreachable!("filtered to only kernels above"),
        };

        // Step 3: Map sources to actual Buffers and extract bound ranges
        // Sources are Buffer/Param/After nodes.
        let kb = collect_kernel_buffers(sources, &kernel_uop, input_buffers, &mut allocated_buffers)?;

        // Collect bound ranges from kernel AST (DEFINE_VAR nodes needing expansion).
        // Skip DefineVars that have bound values in var_vals — those are user Variables,
        // not OUTER ranges needing schedule expansion.
        let bound_ranges = collect_bound_ranges(inner_ast, var_vals)?;

        let dependencies = dependency_ids_by_kernel.get(&kernel_uop.id).cloned().unwrap_or_default();

        debug!(kernel.id = kernel_uop.id, num_sources = sources.len(), "Kernel created");

        // Populate fixedvars with only the user Variables referenced by this kernel's AST.
        let fixedvars: HashMap<String, i64> = if var_vals.is_empty() {
            HashMap::new()
        } else {
            let nodes = inner_ast.toposort();
            let ast_var_names: HashSet<&str> = nodes
                .iter()
                .filter_map(|n| match n.op() {
                    Op::DefineVar { name, .. } => Some(name.as_str()),
                    _ => None,
                })
                .collect();
            var_vals
                .iter()
                .filter(|(name, _)| ast_var_names.contains(name.as_str()))
                .map(|(k, v)| (k.clone(), *v))
                .collect()
        };

        // Use inner_ast (the kernel's internal AST) for codegen, kernel_uop for buffer allocation
        schedule.push(ScheduleItem {
            kernel: kernel_uop.clone(),
            ast: inner_ast.clone(),
            buffers: kb.buffers,
            buffer_uop_ids: kb.uop_ids,
            fixedvars,
            bound_ranges,
            dependencies,
            alias_registered_ids: kb.alias_ids,
        });
    }

    // Identify output buffers directly from SINK source order.
    // Each SINK source is AFTER(BUFFER, KERNEL) — buf_uop() extracts the BUFFER.
    // This handles diamond patterns correctly: a buffer can be both an output
    // AND consumed by another kernel without being filtered out.
    let output_uop_ids: Vec<u64> = match transformed.op() {
        Op::Sink { sources } => sources.iter().map(|src| src.buf_uop().id).collect(),
        _ => vec![transformed.buf_uop().id],
    };

    Ok(ScheduleResult { items: schedule, output_uop_ids })
}

/// Extract device from the first input buffer in kernel sources.
///
/// This follows Tinygrad's pattern where `ctx[0].device` (first buffer's device)
/// determines the device for kernel compilation and output buffer allocation.
///
/// Falls back to CPU if no input buffers are found.
fn find_first_input_buffer_device(
    sources: &[Arc<UOp>],
    input_buffers: &InputBuffers,
    allocated_buffers: &HashMap<u64, Buffer>,
) -> Result<Arc<Device>> {
    let alloc_registry = registry::registry();

    for src in sources {
        if let Some(buf_id) = source_primary_buffer_id(src) {
            let buffer = allocated_buffers.get(&buf_id).cloned().or_else(|| input_buffers.get(&buf_id).cloned());
            if let Some(buffer) = buffer {
                let device_spec = buffer.allocator().device_spec();
                return morok_runtime::DEVICE_FACTORIES
                    .device(&device_spec, alloc_registry)
                    .context(DeviceFactorySnafu);
            }
        }
    }

    // Fallback to CPU if no input buffers found
    morok_runtime::DEVICE_FACTORIES.device(&DeviceSpec::Cpu, alloc_registry).context(DeviceFactorySnafu)
}

/// Collect buffers for a kernel from its sources.
///
/// This walks the kernel sources and identifies:
/// - Input buffers (Op::Buffer) - get from input_buffers
/// - Intermediate buffers (Op::Param) - allocate and track
/// - Shared buffers (Op::After) - look up from allocated_buffers (producer kernel)
///
/// For input buffers (PARAM that maps to an original BUFFER),
/// we reuse the existing buffer from input_buffers instead of allocating.
///
/// For shared buffers (AFTER nodes), we look up the buffer using buf_uop()
/// which walks through AFTER chains to get the underlying buffer ID.
///
/// Output/intermediate buffers are allocated on the same device as the first input buffer
/// (following Tinygrad's pattern). Newly allocated buffers are tracked in `allocated_buffers`.
fn collect_kernel_buffers(
    sources: &[Arc<UOp>],
    kernel: &Arc<UOp>,
    input_buffers: &InputBuffers,
    allocated_buffers: &mut HashMap<u64, Buffer>,
) -> Result<KernelBuffers> {
    // Get AST for buffer size computation
    let ast = match kernel.op() {
        Op::Kernel { ast, .. } => ast,
        _ => {
            return ExpectedKernelOpSnafu.fail();
        }
    };

    // Get target device from first input buffer (Tinygrad pattern: ctx[0].device)
    let target_device = find_first_input_buffer_device(sources, input_buffers, allocated_buffers)?;

    let mut buffers = Vec::new();
    let mut uop_ids = Vec::new();
    let mut alias_ids = Vec::new();

    for src in sources {
        match src.op() {
            Op::After { passthrough, .. } => {
                // Shared buffer from producer kernel.
                // Use buf_uop() to get underlying buffer ID (handles AFTER chains).
                let buf_id = passthrough.buf_uop().id;
                if buf_id != src.id {
                    alias_ids.push(src.id);
                }

                // Look up from allocated_buffers or input_buffers
                let existing = allocated_buffers.get(&buf_id).cloned().or_else(|| input_buffers.get(&buf_id).cloned());

                if let Some(buffer) = existing {
                    trace!(
                        buf_id,
                        buffer.id = ?buffer.id(),
                        "Found shared buffer from AFTER"
                    );

                    // Track under buf_id if not already tracked
                    allocated_buffers.entry(buf_id).or_insert_with(|| buffer.clone());

                    buffers.push(buffer);
                    uop_ids.push(buf_id);
                } else {
                    trace!(buf_id, "after buffer not found in allocated_buffers or input_buffers");
                    return Err(Error::BufferNotFound { uop_id: buf_id });
                }
            }
            Op::MSelect { .. } | Op::MStack { .. } => {
                let canonical_id = source_primary_buffer_id(src).expect("multi-device source should resolve buffer id");
                if canonical_id != src.id {
                    alias_ids.push(src.id);
                }

                let existing =
                    allocated_buffers.get(&canonical_id).cloned().or_else(|| input_buffers.get(&canonical_id).cloned());

                if let Some(buffer) = existing {
                    trace!(canonical_id, buffer.id = ?buffer.id(), "Found shared buffer from MSELECT/MSTACK source");
                    allocated_buffers.entry(canonical_id).or_insert_with(|| buffer.clone());
                    buffers.push(buffer);
                    uop_ids.push(canonical_id);
                } else {
                    trace!(canonical_id, "multi-device source buffer not found in allocated_buffers or input_buffers");
                    return Err(Error::BufferNotFound { uop_id: canonical_id });
                }
            }
            // Kernel sources are always Buffer/Param/After.
            Op::DefineLocal(_id) => {
                // Allocate local/shared memory buffer on same device as inputs
                let ptr_dtype = src.dtype();
                let size = compute_buffer_size(ast, src)?;

                // Extract the base scalar dtype from the Ptr type
                let scalar_dtype = match ptr_dtype {
                    morok_dtype::DType::Ptr { base, .. } => *base,
                    other => {
                        return ExpectedPtrDtypeSnafu { context: "DEFINE_LOCAL", actual: other.clone() }.fail();
                    }
                };

                let buffer =
                    Buffer::new(target_device.allocator.clone(), scalar_dtype.clone(), vec![size], Default::default());

                // Track in allocated_buffers (no registry needed)
                allocated_buffers.insert(src.id, buffer.clone());

                buffers.push(buffer);
                uop_ids.push(src.id);
            }
            Op::Buffer { size, .. } | Op::Param { size, .. } => {
                let canonical_id = src.buf_uop().id;
                if canonical_id != src.id {
                    alias_ids.push(src.id);
                }

                // BUFFER/PARAM can be either input (from input_buffers) or output (needs allocation)
                // Try input_buffers first, then allocated_buffers, then allocate new
                if let Some(buffer) =
                    input_buffers.get(&canonical_id).cloned().or_else(|| input_buffers.get(&src.id).cloned())
                {
                    buffers.push(buffer);
                    uop_ids.push(canonical_id);
                } else if let Some(buffer) =
                    allocated_buffers.get(&canonical_id).cloned().or_else(|| allocated_buffers.get(&src.id).cloned())
                {
                    buffers.push(buffer);
                    uop_ids.push(canonical_id);
                } else {
                    // Output buffer - allocate new buffer
                    trace!(src.id = src.id, canonical_id, size, "Allocating output BUFFER/PARAM");
                    let scalar_dtype = src.dtype();

                    let buffer = Buffer::new(
                        target_device.allocator.clone(),
                        scalar_dtype.clone(),
                        vec![*size],
                        Default::default(),
                    );

                    // Track in allocated_buffers
                    allocated_buffers.insert(canonical_id, buffer.clone());
                    buffers.push(buffer);
                    uop_ids.push(canonical_id);
                }
            }
            Op::Bind { .. } | Op::DefineVar { .. } => {
                // Variable binding - not a buffer, skip
                continue;
            }
            _ => {
                // Unknown source type - skip
                continue;
            }
        }
    }

    alias_ids.sort_unstable();
    alias_ids.dedup();
    Ok(KernelBuffers { buffers, uop_ids, alias_ids })
}

/// Collect bound ranges from kernel AST.
///
/// This walks the kernel AST and identifies DEFINE_VAR nodes that need schedule expansion.
///
/// Note: Loop ranges (OUTER/GLOBAL/LOOP) now generate inline for-loops directly in codegen,
/// so they don't create DefineVars and don't need schedule expansion (Tinygrad approach).
///
/// # Arguments
///
/// * `ast` - The kernel AST (SINK or special operation)
///
/// # Returns
///
/// A vector of BoundRange structs, one for each DefineVar found.
fn collect_bound_ranges(ast: &Arc<UOp>, var_vals: &HashMap<String, i64>) -> Result<Vec<BoundRange>> {
    let nodes = ast.toposort();
    let mut bound_ranges = Vec::new();

    // Collect DefineVar nodes that need schedule expansion (OUTER ranges).
    // Skip DefineVars that have a bound value in var_vals — those are user Variables
    // whose value goes directly to fixedvars, not range expansion.
    for node in &nodes {
        if let Op::DefineVar { name, max_val, .. } = node.op() {
            // If this variable has a concrete bound value, it's a user Variable.
            // Its value is already in fixedvars — no schedule expansion needed.
            if var_vals.contains_key(name) {
                continue;
            }

            // OUTER range variable — expand from 0 to max_val
            let range_end = UOp::const_(
                morok_ir::DType::Scalar(morok_dtype::ScalarDType::Index),
                morok_ir::ConstValue::Int(*max_val + 1),
            );
            let range_uop = UOp::range_axis(range_end, morok_ir::AxisId::Renumbered(0), morok_ir::AxisType::Outer);

            bound_ranges.push(BoundRange { var_name: name.clone(), range_uop });
        }
    }

    Ok(bound_ranges)
}

/// Expand schedule items with bound ranges into individual iterations.
///
/// This function implements Tinygrad's schedule expansion pattern.
/// For each kernel with OUTER ranges (represented as bound_ranges),
/// it generates N schedule items where N is the product of all range sizes.
/// Each expanded item has concrete values in fixedvars.
///
/// Based on Tinygrad's schedule.py:97-116.
///
/// # Arguments
///
/// * `schedule` - The unexpanded schedule with potential bound_ranges
///
/// # Returns
///
/// An expanded schedule where all items have empty bound_ranges and
/// populated fixedvars for each iteration.
pub fn expand_schedule(schedule: Schedule) -> Schedule {
    let mut expanded = Vec::new();

    for item in schedule {
        if item.bound_ranges.is_empty() {
            // No bound ranges - already expanded or no standalone DefineVars
            expanded.push(item);
        } else {
            // Expand into multiple schedule items
            // Note: bound_ranges now only contains standalone DefineVars (not BIND+OUTER)
            // BIND+OUTER DefineVars are handled by CPU codegen as inlined loops
            // Extract iteration counts from each bound range
            let iteration_counts: Vec<i64> = item
                .bound_ranges
                .iter()
                .map(|br| {
                    // Extract range end from RANGE operation
                    match br.range_uop.op() {
                        Op::Range { end, .. } => extract_const_int(end).unwrap_or(1),
                        _ => 1,
                    }
                })
                .collect();

            // Compute total iterations (product of all range sizes)
            let total_iterations: usize = iteration_counts.iter().product::<i64>() as usize;

            // Generate one schedule item per iteration
            for iter_idx in 0..total_iterations {
                // Convert flat index to multi-dimensional indices
                let indices = compute_multi_index(iter_idx, &iteration_counts);

                // Create fixedvars mapping for this iteration
                let mut fixedvars = item.fixedvars.clone();
                for (i, br) in item.bound_ranges.iter().enumerate() {
                    fixedvars.insert(br.var_name.clone(), indices[i]);
                }

                // Create expanded schedule item
                expanded.push(ScheduleItem {
                    kernel: item.kernel.clone(),
                    ast: item.ast.clone(),
                    buffers: item.buffers.clone(),
                    buffer_uop_ids: item.buffer_uop_ids.clone(),
                    fixedvars,
                    bound_ranges: vec![], // Expanded items have no bound ranges
                    dependencies: item.dependencies.clone(),
                    alias_registered_ids: item.alias_registered_ids.clone(),
                });
            }
        }
    }

    expanded
}

/// Extract i64 constant from a UOp.
///
/// Used to get range end values from CONST nodes.
fn extract_const_int(uop: &Arc<UOp>) -> Option<i64> {
    match uop.op() {
        Op::Const(ConstValueHash(morok_ir::ConstValue::Int(v))) => Some(*v),
        Op::Const(ConstValueHash(morok_ir::ConstValue::UInt(v))) => Some(*v as i64),
        _ => None,
    }
}

/// Convert flat iteration index to multi-dimensional indices (row-major order).
///
/// # Arguments
///
/// * `flat_idx` - The flat iteration index (0..total_iterations)
/// * `dimensions` - The size of each dimension (range sizes)
///
/// # Returns
///
/// A vector of indices, one per dimension.
///
/// # Example
///
/// ```ignore
/// // For a 2x3 iteration space (total 6 iterations):
/// compute_multi_index(0, &[2, 3]) // [0, 0]
/// compute_multi_index(1, &[2, 3]) // [0, 1]
/// compute_multi_index(2, &[2, 3]) // [0, 2]
/// compute_multi_index(3, &[2, 3]) // [1, 0]
/// compute_multi_index(4, &[2, 3]) // [1, 1]
/// compute_multi_index(5, &[2, 3]) // [1, 2]
/// ```
fn compute_multi_index(flat_idx: usize, dimensions: &[i64]) -> Vec<i64> {
    let mut indices = Vec::with_capacity(dimensions.len());
    let mut remaining = flat_idx;

    // Compute strides (row-major order)
    let mut strides = Vec::with_capacity(dimensions.len());
    let mut stride = 1usize;
    for &dim in dimensions.iter().rev() {
        strides.push(stride);
        stride *= dim as usize;
    }
    strides.reverse();

    // Extract each index
    for &stride in &strides {
        let idx = (remaining / stride) as i64;
        indices.push(idx);
        remaining %= stride;
    }

    indices
}

/// Compute buffer size from the buffer definition's dtype.
///
/// Buffer size is embedded in the Ptr dtype by debuf() during rangeify.
/// This follows Tinygrad's pattern where size is stored in `dtype.ptr(size=...)`.
fn compute_buffer_size(_ast: &Arc<UOp>, buffer_def: &Arc<UOp>) -> Result<usize> {
    // Extract size from Ptr dtype (set by debuf() in split_patterns.rs)
    match buffer_def.dtype() {
        DType::Ptr { size: Some(s), .. } => Ok(s),
        DType::Ptr { size: None, .. } => BufferPtrNoSizeSnafu.fail(),
        other => ExpectedPtrDtypeSnafu { context: "buffer_size", actual: other.clone() }.fail(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use morok_ir::{DType, DeviceSpec};
    use smallvec::SmallVec;

    #[test]
    fn test_schedule_item_creation() {
        use morok_ir::ConstValue;

        // Create a simple kernel for testing
        let buffer = UOp::new_buffer(DeviceSpec::Cpu, 10, DType::Float32);
        let idx = UOp::index_const(0);
        let value = UOp::const_(DType::Float32, ConstValue::Float(1.0));
        let index = UOp::index().buffer(buffer.clone()).indices(vec![idx]).call().unwrap();
        let store = index.store(value);
        let sink = UOp::sink(vec![store]);

        let mut sources = SmallVec::new();
        sources.push(buffer);
        let kernel = UOp::kernel(sources, sink.clone());

        // ScheduleItem should be creatable
        let item = ScheduleItem {
            kernel: kernel.clone(),
            ast: sink,
            buffers: vec![],
            buffer_uop_ids: vec![],
            fixedvars: HashMap::new(),
            bound_ranges: vec![],
            dependencies: vec![],
            alias_registered_ids: vec![],
        };

        assert!(matches!(item.kernel.op(), Op::Kernel { .. }));
    }

    #[test]
    fn test_collect_kernel_buffers_after_uses_canonical_buffer_id() {
        let buffer_uop = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
        let after = buffer_uop.after(SmallVec::new());
        let sink = UOp::sink(vec![UOp::native_const(0.0f32)]);
        let mut sources = SmallVec::new();
        sources.push(after.clone());
        let kernel = UOp::kernel(sources.clone(), sink);

        let alloc = morok_device::registry::cpu().expect("cpu allocator");
        let input_buf = Buffer::new(alloc, DType::Float32, vec![4], Default::default());
        let mut input_buffers = InputBuffers::new();
        input_buffers.insert(buffer_uop.id, input_buf.clone());
        let mut allocated = HashMap::new();

        let kb = collect_kernel_buffers(&sources, &kernel, &input_buffers, &mut allocated).expect("collect buffers");

        assert_eq!(kb.uop_ids, vec![buffer_uop.id]);
        assert_eq!(kb.buffers.len(), 1);
        assert_eq!(kb.buffers[0].id(), input_buf.id());
        assert!(kb.alias_ids.contains(&after.id));
    }

    #[test]
    fn test_collect_kernel_buffers_mselect_uses_canonical_buffer_id() {
        let buffer_uop = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
        let mstack = UOp::mstack(SmallVec::from_vec(vec![buffer_uop.clone()]));
        let mselect = mstack.mselect(0);
        let sink = UOp::sink(vec![UOp::native_const(0.0f32)]);
        let mut sources = SmallVec::new();
        sources.push(mselect.clone());
        let kernel = UOp::kernel(sources.clone(), sink);

        let alloc = morok_device::registry::cpu().expect("cpu allocator");
        let input_buf = Buffer::new(alloc, DType::Float32, vec![4], Default::default());
        let mut input_buffers = InputBuffers::new();
        input_buffers.insert(buffer_uop.id, input_buf.clone());
        let mut allocated = HashMap::new();

        let kb = collect_kernel_buffers(&sources, &kernel, &input_buffers, &mut allocated).expect("collect buffers");

        assert_eq!(kb.uop_ids, vec![buffer_uop.id]);
        assert_eq!(kb.buffers.len(), 1);
        assert_eq!(kb.buffers[0].id(), input_buf.id());
        assert!(kb.alias_ids.contains(&mselect.id));
    }

    #[test]
    fn test_create_schedule_preserves_kernel_dependencies() {
        let buffer_uop = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);

        let sink1 = UOp::sink(vec![UOp::native_const(1.0f32)]);
        let mut sources1 = SmallVec::new();
        sources1.push(buffer_uop.clone());
        let kernel1 = UOp::kernel(sources1, sink1);

        let mut deps = SmallVec::new();
        deps.push(kernel1.clone());
        let after = buffer_uop.after(deps);

        let sink2 = UOp::sink(vec![UOp::native_const(2.0f32)]);
        let mut sources2 = SmallVec::new();
        sources2.push(after);
        let kernel2 = UOp::kernel(sources2, sink2);

        let transformed = UOp::sink(vec![kernel1.clone(), kernel2.clone()]);

        let alloc = morok_device::registry::cpu().expect("cpu allocator");
        let input_buf = Buffer::new(alloc, DType::Float32, vec![4], Default::default());
        let mut input_buffers = InputBuffers::new();
        input_buffers.insert(buffer_uop.id, input_buf);

        let result = create_schedule(transformed, &input_buffers, &HashMap::new()).expect("create schedule");

        assert_eq!(result.items.len(), 2);
        let k1_item = result.items.iter().find(|it| it.kernel.id == kernel1.id).expect("k1 item");
        let k2_item = result.items.iter().find(|it| it.kernel.id == kernel2.id).expect("k2 item");

        assert!(k1_item.dependencies.is_empty());
        assert_eq!(k2_item.dependencies, vec![kernel1.id]);
    }
}
