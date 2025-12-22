//! Kernel scheduling types and execution.
//!
//! This module provides types and functions for managing the execution
//! schedule of tensor operations. After the rangeify pipeline transforms
//! the computation graph into KERNEL operations, we need to:
//!
//! 1. Extract kernel operations from the transformed graph
//! 2. Allocate buffers for intermediate results (DEFINE_GLOBAL/DEFINE_LOCAL)
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
use morok_schedule::rangeify::{KernelContext, KernelDependency};
use tracing::{debug, trace};

use crate::error::*;
use crate::{Error, Result};
use snafu::ResultExt;

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

    /// Mapping from DEFINE_GLOBAL UOp ID to original BUFFER UOp (for input buffers).
    /// This allows reusing existing buffers instead of allocating new ones.
    /// Key: DEFINE_GLOBAL UOp ID, Value: original BUFFER UOp for looking up in buffer index.
    pub source_buffers: HashMap<u64, Arc<UOp>>,

    /// KERNEL UOp IDs that must complete before this kernel can execute.
    /// Empty for the first kernel in a dependency chain.
    /// Populated from KernelContext.kernel_deps during schedule creation.
    pub dependencies: Vec<u64>,

    /// Additional UOp IDs registered as aliases in buffer index.
    /// These are IDs where the same buffer was registered under a different key
    /// for lookup convenience. They need to be cleaned up along with buffer_uop_ids.
    pub alias_registered_ids: Vec<u64>,
}

/// Full execution schedule (list of kernels in dependency order).
pub type Schedule = Vec<ScheduleItem>;

/// Sort kernels by dependencies (producers before consumers).
///
/// Uses Kahn's algorithm for topological sort based on kernel_deps.
/// This ensures producer kernels are processed before consumers, which is
/// critical for buffer sharing: the producer allocates the buffer first,
/// then the consumer finds it in the registry via `get_or_create_buffer`.
fn sort_kernels_by_dependencies(kernels: &[Arc<UOp>], kernel_deps: &[KernelDependency]) -> Vec<Arc<UOp>> {
    debug!(num_kernels = kernels.len(), num_deps = kernel_deps.len(), "sorting kernels by dependencies");

    // Build kernel ID set
    let kernel_ids: HashSet<u64> = kernels.iter().map(|k| k.id).collect();

    // Build dependency graph (only for kernels in our list)
    let mut in_degree: HashMap<u64, usize> = HashMap::new();
    let mut dependents: HashMap<u64, Vec<u64>> = HashMap::new();

    for kernel in kernels {
        in_degree.entry(kernel.id).or_insert(0);
        dependents.entry(kernel.id).or_default();
    }

    for dep in kernel_deps {
        // Only count dependencies between kernels in our list
        if kernel_ids.contains(&dep.consumer.id) && kernel_ids.contains(&dep.producer.id) {
            *in_degree.entry(dep.consumer.id).or_insert(0) += 1;
            dependents.entry(dep.producer.id).or_default().push(dep.consumer.id);
        }
    }

    // Kahn's algorithm
    let mut queue: VecDeque<u64> = in_degree.iter().filter(|&(_, deg)| *deg == 0).map(|(&id, _)| id).collect();

    let kernel_map: HashMap<u64, Arc<UOp>> = kernels.iter().map(|k| (k.id, k.clone())).collect();

    let mut sorted = Vec::new();
    while let Some(id) = queue.pop_front() {
        if let Some(kernel) = kernel_map.get(&id) {
            sorted.push(kernel.clone());
        }
        for &dependent in dependents.get(&id).unwrap_or(&vec![]) {
            if let Some(deg) = in_degree.get_mut(&dependent) {
                *deg -= 1;
                if *deg == 0 {
                    queue.push_back(dependent);
                }
            }
        }
    }

    // If sorting didn't include all kernels (possible cycle or disconnected graph),
    // fall back to original order for any missing kernels
    if sorted.len() < kernels.len() {
        let sorted_ids: HashSet<u64> = sorted.iter().map(|k| k.id).collect();
        for kernel in kernels {
            if !sorted_ids.contains(&kernel.id) {
                sorted.push(kernel.clone());
            }
        }
    }

    debug!(num_sorted = sorted.len(), "kernels sorted");

    sorted
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
/// * `kernel_ctx` - The KernelContext from kernel splitting, containing buffer_map
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
    kernel_ctx: KernelContext,
    input_buffers: &InputBuffers,
) -> Result<Schedule> {
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
    let kernels = sort_kernels_by_dependencies(&kernels, &kernel_ctx.kernel_deps);

    // Track allocated intermediate buffers locally (no global registry needed)
    let mut allocated_buffers: HashMap<u64, Buffer> = HashMap::new();

    // Step 2: Build reverse mapping from DEFINE_GLOBAL → original BUFFER/BUFFERIZE
    // This allows us to reuse existing input buffers instead of allocating new ones.
    // For outputs (BUFFERIZE), we don't have a pre-existing buffer, so we skip those.
    let mut define_to_buffer: HashMap<u64, Arc<UOp>> = HashMap::new();
    for (key, value) in &kernel_ctx.buffer_map {
        // Unwrap AFTER if present (for output buffers)
        let actual_value = match value.op() {
            Op::After { passthrough, .. } => passthrough.clone(),
            _ => value.clone(),
        };

        // If actual_value is DEFINE_GLOBAL, map it back to the original key
        // BUT only if the key is a BUFFER (input), not BUFFERIZE (output)
        if let Op::DefineGlobal(_) = actual_value.op()
            && matches!(key.0.op(), Op::Buffer { .. })
        {
            // Input buffer - map DEFINE_GLOBAL → BUFFER for reuse
            define_to_buffer.insert(actual_value.id, key.0.clone());
        }
        // For BUFFERIZE (output), we don't add to define_to_buffer
        // This causes collect_kernel_buffers to allocate a new buffer
    }

    // Build reverse mapping: new_id -> old_id
    // This is needed because kernel sources use POST-renumbered IDs,
    // but define_to_buffer has PRE-renumbered IDs.
    let reverse_id_mapping: HashMap<u64, u64> =
        kernel_ctx.buffer_id_mapping.iter().map(|(&old_id, &new_id)| (new_id, old_id)).collect();

    trace!(
        define_to_buffer_entries = define_to_buffer.len(),
        reverse_id_mapping_entries = reverse_id_mapping.len(),
        "Buffer mappings"
    );

    // Step 2: For each kernel, collect buffers from sources
    let mut schedule = Vec::new();

    for kernel_uop in kernels {
        let (sources, inner_ast) = match kernel_uop.op() {
            Op::Kernel { sources, ast } => (sources, ast),
            _ => unreachable!("filtered to only kernels above"),
        };

        // Step 3: Map sources to actual Buffers and extract bound ranges
        // Sources can be:
        // - DEFINE_GLOBAL(id) - intermediate buffer to allocate
        // - DEFINE_LOCAL(id) - local/shared memory to allocate
        // - BUFFER - input buffer from registry
        // - DEFINE_VAR - variable binding for OUTER ranges

        // Collect and allocate all buffers for this kernel
        // Buffers are allocated ONCE here, then reused across all iterations
        let (buffers, buffer_uop_ids, alias_registered_ids) = collect_kernel_buffers(
            sources,
            &kernel_uop,
            &define_to_buffer,
            &kernel_ctx.buffer_id_mapping,
            &reverse_id_mapping,
            &kernel_ctx.define_to_buffer_id,
            input_buffers,
            &mut allocated_buffers,
        )?;

        // Collect bound ranges from kernel AST (BIND nodes with DEFINE_VAR)
        let bound_ranges = collect_bound_ranges(inner_ast)?;

        // Build source_buffers mapping for this kernel
        let mut source_buffers = HashMap::new();
        for src in sources {
            if let Op::DefineGlobal(_) = src.op()
                && let Some(original_buffer) = define_to_buffer.get(&src.id)
            {
                source_buffers.insert(src.id, original_buffer.clone());
            }
        }

        // Extract dependencies from kernel_ctx.kernel_deps
        // Find all kernels that this kernel depends on (producer kernels)
        let dependencies: Vec<u64> = kernel_ctx
            .kernel_deps
            .iter()
            .filter(|dep| dep.consumer.id == kernel_uop.id)
            .map(|dep| dep.producer.id)
            .collect();

        debug!(
            kernel.id = kernel_uop.id,
            dependencies = ?dependencies,
            "Kernel dependencies"
        );

        // Use inner_ast (the kernel's internal AST) for codegen, kernel_uop for buffer allocation
        schedule.push(ScheduleItem {
            kernel: kernel_uop.clone(),
            ast: inner_ast.clone(),
            buffers,
            buffer_uop_ids,
            fixedvars: HashMap::new(),
            bound_ranges,
            source_buffers,
            dependencies,
            alias_registered_ids,
        });
    }

    Ok(schedule)
}

/// Extract device from the first input buffer in kernel sources.
///
/// This follows Tinygrad's pattern where `ctx[0].device` (first buffer's device)
/// determines the device for kernel compilation and output buffer allocation.
///
/// Falls back to CPU if no input buffers are found.
fn find_first_input_buffer_device(
    sources: &[Arc<UOp>],
    define_to_buffer: &HashMap<u64, Arc<UOp>>,
    reverse_id_mapping: &HashMap<u64, u64>,
    input_buffers: &InputBuffers,
    allocated_buffers: &HashMap<u64, Buffer>,
) -> Result<Arc<Device>> {
    let alloc_registry = registry::registry();

    for src in sources {
        match src.op() {
            // Direct input buffer
            Op::Buffer { .. } => {
                // Try allocated_buffers, then input_buffers (no registry needed)
                let buffer = allocated_buffers.get(&src.id).cloned().or_else(|| input_buffers.get(&src.id).cloned());
                if let Some(buffer) = buffer {
                    let device_spec = buffer.allocator().device_spec();
                    return morok_runtime::DEVICE_FACTORIES
                        .device(&device_spec, alloc_registry)
                        .context(DeviceFactorySnafu);
                }
            }
            // DEFINE_GLOBAL that maps to an input buffer
            // Use reverse mapping to handle renumbered IDs
            Op::DefineGlobal(_) => {
                let lookup_result = define_to_buffer
                    .get(&src.id)
                    .or_else(|| reverse_id_mapping.get(&src.id).and_then(|old_id| define_to_buffer.get(old_id)));

                if let Some(original_buffer) = lookup_result {
                    // Try allocated_buffers, then input_buffers (no registry needed)
                    let buffer = allocated_buffers
                        .get(&original_buffer.id)
                        .cloned()
                        .or_else(|| input_buffers.get(&original_buffer.id).cloned());
                    if let Some(buffer) = buffer {
                        let device_spec = buffer.allocator().device_spec();
                        return morok_runtime::DEVICE_FACTORIES
                            .device(&device_spec, alloc_registry)
                            .context(DeviceFactorySnafu);
                    }
                }
            }
            _ => continue,
        }
    }

    // Fallback to CPU if no input buffers found
    morok_runtime::DEVICE_FACTORIES.device(&DeviceSpec::Cpu, alloc_registry).context(DeviceFactorySnafu)
}

/// Collect buffers for a kernel from its sources.
///
/// This walks the kernel sources and identifies:
/// - Input buffers (Op::Buffer) - get from input_buffers
/// - Intermediate buffers (Op::DefineGlobal, Op::DefineLocal) - allocate and track
/// - Shared buffers (Op::After) - look up from allocated_buffers (producer kernel)
///
/// For input buffers (DEFINE_GLOBAL that maps to an original BUFFER),
/// we reuse the existing buffer from input_buffers instead of allocating.
///
/// For shared buffers (AFTER nodes), we look up the buffer using the original
/// (pre-renumbered) DefineGlobal ID from the AFTER passthrough, then translate
/// to the renumbered ID using buffer_id_mapping.
///
/// Output/intermediate buffers are allocated on the same device as the first input buffer
/// (following Tinygrad's pattern). Newly allocated buffers are tracked in `allocated_buffers`.
#[allow(clippy::too_many_arguments)]
fn collect_kernel_buffers(
    sources: &[Arc<UOp>],
    kernel: &Arc<UOp>,
    define_to_buffer: &HashMap<u64, Arc<UOp>>,
    buffer_id_mapping: &HashMap<u64, u64>,
    reverse_id_mapping: &HashMap<u64, u64>,
    define_to_buffer_id: &HashMap<u64, u64>,
    input_buffers: &InputBuffers,
    allocated_buffers: &mut HashMap<u64, Buffer>,
) -> Result<(Vec<Buffer>, Vec<u64>, Vec<u64>)> {
    // Get AST for buffer size computation
    let ast = match kernel.op() {
        Op::Kernel { ast, .. } => ast,
        _ => {
            return ExpectedKernelOpSnafu.fail();
        }
    };

    // Get target device from first input buffer (Tinygrad pattern: ctx[0].device)
    // Pass reverse_id_mapping to handle renumbered IDs
    let target_device = find_first_input_buffer_device(
        sources,
        define_to_buffer,
        reverse_id_mapping,
        input_buffers,
        allocated_buffers,
    )?;

    let mut buffers = Vec::new();
    let mut uop_ids = Vec::new();
    let mut alias_ids = Vec::new();

    for src in sources {
        match src.op() {
            Op::After { passthrough, .. } => {
                // Shared buffer from producer kernel.
                // passthrough.id is the ORIGINAL (pre-renumbered) buffer identity.
                let original_id = passthrough.id;

                // Look up using the renumbered ID (what producer allocated under)
                let lookup_id = buffer_id_mapping.get(&original_id).copied().unwrap_or(original_id);

                // Look up from allocated_buffers (no registry needed)
                if let Some(existing) = allocated_buffers.get(&lookup_id).cloned() {
                    trace!(
                        original_id,
                        lookup_id,
                        buffer.id = ?existing.id(),
                        "Found shared buffer from AFTER"
                    );

                    // Also track under original_id for future lookups
                    if original_id != lookup_id {
                        allocated_buffers.insert(original_id, existing.clone());
                        alias_ids.push(original_id); // Track for cleanup
                    }

                    buffers.push(existing);
                    uop_ids.push(lookup_id);
                } else {
                    trace!(original_id, lookup_id, "after buffer not found in allocated_buffers");
                    return Err(Error::BufferNotFound { uop_id: original_id });
                }
            }
            Op::DefineGlobal(_id) => {
                // Check if this DEFINE_GLOBAL maps to an original BUFFER (input buffer)
                // First try direct lookup, then try with reverse mapping (new_id -> old_id)
                // because define_to_buffer has PRE-renumbered IDs but kernel sources use POST-renumbered IDs.
                let lookup_result = define_to_buffer
                    .get(&src.id)
                    .or_else(|| reverse_id_mapping.get(&src.id).and_then(|old_id| define_to_buffer.get(old_id)));

                if let Some(original_buffer) = lookup_result {
                    trace!(src.id = src.id, original_buffer.id = original_buffer.id, "input buffer (reusing existing)");
                    // This is an input buffer - get from input_buffers
                    if let Some(buffer) = input_buffers.get(&original_buffer.id).cloned() {
                        // Also track under the DEFINE_GLOBAL's ID for later lookups
                        allocated_buffers.insert(src.id, buffer.clone());
                        buffers.push(buffer);
                        uop_ids.push(src.id);
                    } else {
                        return Err(Error::BufferNotFound { uop_id: original_buffer.id });
                    }
                } else {
                    // Try define_to_buffer_id mapping (tracks all DefineGlobal → BUFFER id)
                    // This handles the case where buffer_map was overwritten by a later kernel
                    let buffer_id_via_mapping = define_to_buffer_id
                        .get(&src.id)
                        .or_else(|| reverse_id_mapping.get(&src.id).and_then(|old_id| define_to_buffer_id.get(old_id)));

                    if let Some(&original_buffer_id) = buffer_id_via_mapping
                        && let Some(buffer) = input_buffers.get(&original_buffer_id).cloned()
                    {
                        trace!(
                            src.id = src.id,
                            original_buffer_id,
                            buffer.id = ?buffer.id(),
                            "Input buffer via define_to_buffer_id"
                        );
                        // Also track under the DEFINE_GLOBAL's ID for later lookups
                        allocated_buffers.insert(src.id, buffer.clone());
                        buffers.push(buffer);
                        uop_ids.push(src.id);
                        continue;
                    }

                    // Check if buffer already exists in allocated_buffers (shared from another kernel)
                    if let Some(existing) = allocated_buffers.get(&src.id).cloned() {
                        trace!(
                            src.id = src.id,
                            buffer.id = ?existing.id(),
                            "Shared buffer from allocated_buffers"
                        );
                        buffers.push(existing);
                        uop_ids.push(src.id);
                        continue;
                    }

                    trace!(src.id = src.id, "creating output/intermediate buffer");
                    // This is an output/intermediate buffer - allocate on the same device as input
                    let ptr_dtype = src.dtype();
                    let size = compute_buffer_size(ast, src)?;

                    // Extract the base scalar dtype from the Ptr type
                    let scalar_dtype = match ptr_dtype {
                        morok_dtype::DType::Ptr { base, .. } => *base,
                        other => {
                            return ExpectedPtrDtypeSnafu { context: "DEFINE_GLOBAL", actual: other.clone() }.fail();
                        }
                    };

                    let buffer = Buffer::new(
                        target_device.allocator.clone(),
                        scalar_dtype.clone(),
                        vec![size],
                        Default::default(),
                    );

                    // Track in allocated_buffers (no registry needed)
                    allocated_buffers.insert(src.id, buffer.clone());

                    buffers.push(buffer);
                    uop_ids.push(src.id);
                }
            }
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
            Op::Buffer { .. } => {
                // Input buffer - get from input_buffers
                if let Some(buffer) = input_buffers.get(&src.id).cloned() {
                    buffers.push(buffer);
                    uop_ids.push(src.id);
                } else {
                    return Err(Error::BufferNotFound { uop_id: src.id });
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

    Ok((buffers, uop_ids, alias_ids))
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
fn collect_bound_ranges(ast: &Arc<UOp>) -> Result<Vec<BoundRange>> {
    let nodes = ast.toposort();
    let mut bound_ranges = Vec::new();

    // Collect all DefineVar nodes - they need schedule expansion
    // (Loop ranges no longer create DefineVars - they become inline loops)
    for node in &nodes {
        if let Op::DefineVar { name, max_val, .. } = node.op() {
            // Create a synthetic RANGE UOp for this variable
            // Range goes from 0 to max_val+1 (exclusive upper bound)
            let range_end = UOp::const_(
                morok_ir::DType::Scalar(morok_dtype::ScalarDType::Index),
                morok_ir::ConstValue::Int(*max_val + 1),
            );
            let range_uop = UOp::range_axis(
                range_end,
                morok_ir::AxisId::Renumbered(0), // Dummy axis ID
                morok_ir::AxisType::Outer,
            );

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
                    source_buffers: item.source_buffers.clone(),
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
        let index = UOp::const_(DType::Int32, ConstValue::Int(0));
        let value = UOp::const_(DType::Float32, ConstValue::Float(1.0));
        let store = UOp::store(buffer.clone(), index, value);
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
            source_buffers: HashMap::new(),
            dependencies: vec![],
            alias_registered_ids: vec![],
        };

        assert!(matches!(item.kernel.op(), Op::Kernel { .. }));
    }
}
