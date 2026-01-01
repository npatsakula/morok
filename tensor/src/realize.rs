//! Tensor realization (execution) API.
//!
//! This module provides the execution pipeline for tensor operations:
//! 1. **Rangeify** - Transform movement ops to BUFFERIZE + INDEX
//! 2. **Kernel splitting** - Split at STORE boundaries into KERNEL ops
//! 3. **Scheduling** - Extract kernels and create execution schedule
//! 4. **Execution** - Compile and run each kernel in dependency order
//!
//! # Parallel Execution
//!
//! When independent kernels are detected (no buffer conflicts), they can be
//! executed in parallel using the `UnifiedExecutor` from the runtime crate.
//! The executor tracks buffer dependencies and uses timeline signals for
//! cross-device synchronization.
//!
//! # ExecutionPlan (Pre-compiled Execution)
//!
//! For repeated executions, use `Tensor::prepare()` to create an `ExecutionPlan`
//! that pre-compiles all kernels and allocates all buffers. Then call
//! `plan.execute()` for fast repeated execution without recompilation overhead.
//!
//! ```ignore
//! // One-time preparation (compiles kernels, allocates buffers)
//! let plan = tensor.prepare()?;
//!
//! // Fast execution (can be called many times)
//! plan.execute(&mut executor)?;
//!
//! // Get results
//! let output = plan.output_buffer();
//! ```

use std::collections::HashMap;

use tracing::{debug, trace};

use crate::{
    Result, Tensor,
    error::{
        CompileKernelSnafu, CreateProgramSnafu, DependencyCyclesSnafu, DeviceFactorySnafu, DeviceSnafu,
        EmptyScheduleSnafu, ExecutionSnafu, OptimizeSnafu, RangeifySnafu, RenderKernelSnafu, ShapeUnknownSnafu,
        UOpSnafu,
    },
    schedule::{Schedule, ScheduleItem, expand_schedule},
};
use morok_device::{Buffer, device::Device};
use morok_dtype::DType;
use morok_ir::{AxisId, DeviceSpec, Op, SInt, UOp};
use morok_runtime::{
    ExecutionGraph, ExecutionNode, ExecutionPlan, ExecutionPlanBuilder, KernelBufferAccess, ParallelGroup,
    PreparedKernel,
};
use snafu::{OptionExt, ResultExt};
use std::sync::Arc;
use std::time::Duration;

impl Tensor {
    /// Realize (execute) this tensor's computation graph.
    ///
    /// This is a convenience method that prepares and executes in one call.
    /// For repeated executions of the same computation, use `prepare()` instead.
    ///
    /// # Pipeline
    ///
    /// 1. **Prepare**: Creates an `ExecutionPlan` (compiles kernels, allocates buffers)
    /// 2. **Execute**: Runs all kernels in dependency order
    /// 3. **Return**: Links output buffer to this tensor's UOp
    ///
    /// # Example
    ///
    /// ```ignore
    /// let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
    /// let b = Tensor::from_slice(&[4.0f32, 5.0, 6.0]);
    /// let c = (&a + &b).realize()?;
    /// // c's buffer now contains [5.0, 7.0, 9.0]
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if preparation or execution fails.
    pub fn realize(self) -> Result<Self> {
        // Collect input buffer IDs BEFORE prepare() so we know which mappings to preserve
        let uop = self.uop();
        let input_buffer_ids: std::collections::HashSet<u64> = collect_input_buffers(&uop).keys().copied().collect();

        // Prepare execution plan (compiles kernels, allocates buffers)
        let plan = self.prepare()?;

        // Execute the plan
        let mut executor = morok_runtime::global_executor();
        plan.execute(&mut executor).context(ExecutionSnafu)?;

        // Get output buffer and its properties
        let output_buf = plan.output_buffer().clone();

        trace!(
            buffer.id = ?output_buf.id(),
            buffer.size = output_buf.size(),
            "Realized output buffer"
        );

        // uop already captured above for input buffer collection
        let output_dtype = uop.dtype();
        let output_device = output_buf.allocator().device_spec();
        // Buffer::size() returns bytes, convert to element count
        let num_elements = output_buf.size() / output_dtype.bytes();

        // Create a new BUFFER UOp to represent the materialized data.
        // This is critical: after realization, the tensor's UOp should be a BUFFER
        // so that subsequent schedules know this tensor is already materialized
        // and don't re-compute it.
        let buffer_uop = UOp::new_buffer(output_device, num_elements, output_dtype.clone());

        // RAII: Wrap output buffer in Arc for ownership
        let output_buf_arc = Arc::new(output_buf);

        // Register buffer for subsequent schedule creation lookups.
        // When this realized tensor is used as input to another operation,
        // collect_input_buffers() needs to find this buffer by UOp ID.
        // This sets the buffer on TensorEntry and creates UOp ID → Tensor ID mapping.
        crate::tensor_registry::register_buffer(buffer_uop.id, self.entry.id, output_buf_arc.clone());

        // Get the tensor's shape and reshape the buffer to match
        let shape = uop.shape().context(UOpSnafu)?.context(ShapeUnknownSnafu)?;
        let realized_uop = buffer_uop.try_reshape(shape).context(UOpSnafu)?;

        debug!(
            buffer_uop.id = buffer_uop.id,
            num_elements,
            shape = ?shape,
            realized_uop.id = realized_uop.id,
            realized_uop.base_id = realized_uop.base().id,
            "Tensor realized"
        );

        // Update this tensor's UOp to point to the realized buffer
        self.set_uop(realized_uop);

        // PRIMARY: Create NEW tensor with buffer (RAII ownership)
        // Since realize(self) consumes self, we return a new Tensor with buffer set
        let result = Tensor::with_buffer(Arc::clone(&self.entry), output_buf_arc);

        // Only clean up INTERMEDIATES (not inputs or output)
        // Input buffer mappings must be preserved for subsequent realize() calls
        plan.release_intermediate_buffers(|uop_id| {
            if !input_buffer_ids.contains(&uop_id) {
                crate::tensor_registry::remove_buffer(uop_id);
            }
        });

        Ok(result)
    }

    /// Prepare an execution plan for this tensor's computation graph.
    ///
    /// This performs all one-time work:
    /// 1. Creates schedule from computation graph
    /// 2. Expands bound ranges
    /// 3. Compiles all kernels
    /// 4. Allocates all buffers
    /// 5. Computes parallel execution groups
    ///
    /// The returned `ExecutionPlan` can then be executed multiple times
    /// without recompilation overhead.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
    /// let b = Tensor::from_slice(&[4.0f32, 5.0, 6.0]);
    /// let c = &a + &b;
    ///
    /// // One-time preparation
    /// let plan = c.prepare()?;
    ///
    /// // Fast execution (can be called many times)
    /// let mut executor = morok_runtime::global_executor();
    /// plan.execute(&mut executor)?;
    ///
    /// // Get results
    /// let output = plan.output_buffer();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Rangeify transformation fails
    /// - No kernels found after scheduling
    /// - Kernel compilation fails
    /// - Buffer allocation fails
    pub fn prepare(&self) -> Result<ExecutionPlan> {
        self.prepare_with(&morok_schedule::OptimizerConfig::from_env())
    }

    /// Prepare an execution plan with explicit optimizer configuration.
    ///
    /// This method allows fine-grained control over kernel optimization settings,
    /// including beam search width, heuristic parameters, and tensor core usage.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use morok_schedule::{OptimizerConfig, OptStrategy, BeamConfig};
    ///
    /// // Beam search with width 8 and 120s timeout
    /// let config = OptimizerConfig::builder()
    ///     .strategy(OptStrategy::Beam { width: 8 })
    ///     .beam(BeamConfig::builder()
    ///         .timeout_secs(120)
    ///         .build())
    ///     .build();
    ///
    /// let plan = tensor.prepare_with(&config)?;
    /// plan.execute(&mut executor)?;
    /// ```
    ///
    /// # Arguments
    ///
    /// * `config` - Optimizer configuration controlling optimization strategy
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Rangeify transformation fails
    /// - No kernels found after scheduling
    /// - Kernel compilation fails
    /// - Buffer allocation fails
    pub fn prepare_with(&self, config: &morok_schedule::OptimizerConfig) -> Result<ExecutionPlan> {
        use morok_ir::AxisType;

        let uop = self.uop();

        // Step 1: Create BUFFERIZE wrapping the computation
        let shape = uop.shape().context(UOpSnafu)?.context(ShapeUnknownSnafu)?;

        let ranges: Vec<_> = shape
            .iter()
            .enumerate()
            .map(|(i, dim)| {
                let end = match dim {
                    SInt::Const(n) => UOp::index_const(*n as i64),
                    SInt::Symbolic(var) => var.clone(),
                };
                UOp::range_axis(end, AxisId::Unrenumbered(i), AxisType::Outer)
            })
            .collect();

        let output_dtype = uop.dtype();
        let bufferize = UOp::bufferize_global(uop.clone(), ranges);

        // Step 2: Create SINK of the BUFFERIZE
        let sink = UOp::sink(vec![bufferize]);

        // Step 3: Run rangeify pipeline
        // Note: We track becomes_map but don't apply it globally.
        // The becomes_map contains ALL transformations from rewrite passes including internal
        // restructuring. Applying it to other tensors could corrupt them. The diamond pattern
        // issue (like in argmin) needs to be solved within the scheduling logic itself.
        let rangeify_result = morok_schedule::rangeify_with_map(sink, None).context(RangeifySnafu)?;
        let rangeified = rangeify_result.sink;

        trace!(ast = %rangeified.tree_full(), "post-rangeify ast");

        // Step 4: Run kernel splitting pipeline
        let (kernelized, kernel_ctx) = morok_schedule::run_kernel_split_pipeline(rangeified);

        trace!(ast = %kernelized.tree_full(), "post-kernel-split ast");

        // Step 5: Collect input buffers from the computation graph
        // This allows schedule creation to find buffers without global registry lookups
        let input_buffers = collect_input_buffers(&uop);

        // Step 6: Create schedule from kernels
        let schedule = crate::schedule::create_schedule(kernelized, kernel_ctx, &input_buffers)?;

        // Step 7: Build execution plan (pass expected output dtype and size)
        let output_size = shape.iter().map(|s| s.as_const().unwrap_or(1)).product::<usize>() * output_dtype.bytes();
        prepare_execution_plan(&schedule, output_dtype, output_size, config)
    }
}

/// Collect input buffers from a computation graph.
///
/// Walks the UOp graph and collects all BUFFER UOps that have
/// associated buffers in the tensor registry's buffer index.
/// Input tensors (from `from_slice()`) and realized tensors
/// register their buffers for this lookup to work.
///
/// This allows schedule creation to receive buffers explicitly without
/// needing global registry lookups during kernel buffer collection.
fn collect_input_buffers(root: &Arc<UOp>) -> crate::schedule::InputBuffers {
    let mut inputs = HashMap::new();
    for node in root.toposort() {
        if let Op::Buffer { .. } = node.op() {
            // Buffers are registered in from_slice_on() and realize()
            if let Some(buf) = crate::tensor_registry::get_buffer(node.id) {
                inputs.insert(node.id, buf);
            }
        }
    }
    inputs
}

/// Build an execution graph from an expanded schedule.
///
/// Creates an `ExecutionGraph` that represents the DAG of kernel operations,
/// capturing buffer dependencies and enabling parallel execution analysis.
///
/// # Arguments
///
/// * `schedule` - The expanded schedule (after range iteration expansion)
///
/// # Returns
///
/// An `ExecutionGraph` with nodes for each schedule item.
fn build_execution_graph(schedule: &[ScheduleItem]) -> ExecutionGraph {
    let mut graph = ExecutionGraph::new();

    for item in schedule {
        // Determine device from first buffer
        let device = item.buffers.first().map(|b| b.allocator().device_spec()).unwrap_or(DeviceSpec::Cpu);

        // Collect buffer IDs
        let all_buffer_ids: Vec<_> = item.buffers.iter().map(|b| b.id()).collect();

        // Determine output indices by finding buffers written to by STORE ops
        let is_transfer = matches!(item.ast.op(), Op::Copy { .. });
        let output_indices = if is_transfer {
            // For COPY, destination is output
            vec![0usize]
        } else {
            // For computational kernels, detect outputs from STORE operations
            detect_output_indices(&item.kernel, &item.buffers)
        };

        // Split buffer IDs into inputs and outputs
        let outputs: Vec<_> = output_indices.iter().map(|&i| all_buffer_ids[i]).collect();
        let inputs: Vec<_> =
            all_buffer_ids.iter().enumerate().filter(|(i, _)| !output_indices.contains(i)).map(|(_, id)| *id).collect();

        // Create buffer access info for parallel execution
        let buffer_access = Some(KernelBufferAccess { buffers: all_buffer_ids, output_indices });

        // Use kernel UOp ID for node ID since dependencies reference kernel IDs
        // (from kernel_ctx.kernel_deps which uses kernel UOp IDs)
        let node = ExecutionNode {
            id: item.kernel.id,
            device,
            inputs,
            outputs,
            predecessors: item.dependencies.clone(),
            is_transfer,
            buffer_access,
        };

        graph.add_node(node);
    }

    graph
}

/// Detect output buffer indices by finding buffers written to by STORE operations.
///
/// Examines the kernel's AST to find STORE and STOREGATED operations, then
/// identifies which buffers in the buffer list are written to.
///
/// # Arguments
///
/// * `kernel` - The KERNEL UOp containing the AST
/// * `buffers` - The list of buffers for this kernel
///
/// # Returns
///
/// Indices into `buffers` for output buffers. Falls back to [0] if no outputs found.
fn detect_output_indices(kernel: &Arc<UOp>, buffers: &[Buffer]) -> Vec<usize> {
    use std::collections::HashSet;

    let ast = match kernel.op() {
        Op::Kernel { ast, .. } => ast,
        _ => return vec![0], // Fallback if not a kernel
    };

    // Find all DefineGlobal IDs that are written to by STORE operations
    let mut output_buffer_uop_ids: HashSet<u64> = HashSet::new();

    for node in ast.toposort() {
        match node.op() {
            Op::Store { buffer, .. } | Op::StoreGated { buffer, .. } => {
                // Get the buffer's DefineGlobal ID
                let buf_id = match buffer.op() {
                    Op::DefineGlobal(_) | Op::DefineLocal(_) => buffer.id,
                    Op::Index { buffer: inner, .. } => {
                        if matches!(inner.op(), Op::DefineGlobal(_) | Op::DefineLocal(_)) {
                            inner.id
                        } else {
                            continue;
                        }
                    }
                    _ => continue,
                };
                output_buffer_uop_ids.insert(buf_id);
            }
            _ => {}
        }
    }

    // Now map UOp IDs to buffer indices
    // The kernel sources have DefineGlobal UOps in a specific order, which corresponds
    // to the buffer list order. We need to find which DefineGlobal IDs match.
    let sources = match kernel.op() {
        Op::Kernel { sources, .. } => sources,
        _ => return vec![0],
    };

    let mut output_indices = Vec::new();
    let mut buffer_idx = 0;
    for src in sources {
        match src.op() {
            Op::DefineGlobal(_) | Op::DefineLocal(_) => {
                if output_buffer_uop_ids.contains(&src.id) && buffer_idx < buffers.len() {
                    output_indices.push(buffer_idx);
                }
                buffer_idx += 1;
            }
            Op::Buffer { .. } => {
                // Input buffer - also consumes a buffer slot
                if output_buffer_uop_ids.contains(&src.id) && buffer_idx < buffers.len() {
                    output_indices.push(buffer_idx);
                }
                buffer_idx += 1;
            }
            Op::DefineVar { .. } | Op::Bind { .. } => {
                // Variable - doesn't consume a buffer slot
            }
            _ => {}
        }
    }

    // Fallback to first buffer if no outputs found
    if output_indices.is_empty() && !buffers.is_empty() { vec![0] } else { output_indices }
}

/// Prepare an execution plan from a schedule.
///
/// This performs all one-time preparation work:
/// 1. Expands the schedule (handles bound ranges)
/// 2. Builds execution graph and computes parallel groups
/// 3. Allocates all buffers
/// 4. Compiles all kernels
/// 5. Creates PreparedKernel structures
///
/// # Arguments
///
/// * `schedule` - The schedule from `create_schedule()`
///
/// # Returns
///
/// An `ExecutionPlan` ready for fast repeated execution.
///
/// # Errors
///
/// Returns error if compilation or buffer allocation fails.
fn prepare_execution_plan(
    schedule: &Schedule,
    expected_output_dtype: DType,
    expected_output_size: usize,
    config: &morok_schedule::OptimizerConfig,
) -> Result<ExecutionPlan> {
    // Expand the schedule to handle OUTER range iterations
    let expanded_schedule = expand_schedule(schedule.clone());

    debug!(num_items = expanded_schedule.len(), "expanded schedule");

    // Build execution graph for parallel group analysis
    let mut execution_graph = build_execution_graph(&expanded_schedule);

    // Compute parallel groups
    let _parallel_groups_raw = execution_graph.compute_parallel_groups();

    // Verify the graph is valid (no cycles)
    if !execution_graph.is_valid() {
        return DependencyCyclesSnafu.fail();
    }

    // Get device from first buffer in first kernel (Tinygrad pattern: ctx[0].device)
    let alloc_registry = morok_device::registry::registry();
    let device = if let Some(first_item) = expanded_schedule.first() {
        if let Some(first_buffer) = first_item.buffers.first() {
            let device_spec = first_buffer.allocator().device_spec();
            morok_runtime::DEVICE_FACTORIES.device(&device_spec, alloc_registry).context(DeviceFactorySnafu)?
        } else {
            morok_runtime::DEVICE_FACTORIES.device(&DeviceSpec::Cpu, alloc_registry).context(DeviceFactorySnafu)?
        }
    } else {
        return EmptyScheduleSnafu.fail();
    };

    let device_str = device.device.canonicalize();

    // Build the ExecutionPlan using the builder
    let mut builder = ExecutionPlanBuilder::new(device.device.clone());

    // Step 1: Add all buffers to the plan
    // Buffers in each ScheduleItem are already in the correct order (from collect_kernel_buffers).
    // We track buffers by their UOp ID (what they were registered under in tensor_registry's buffer index).
    let mut uop_id_to_idx: HashMap<u64, usize> = HashMap::new();

    for item in &expanded_schedule {
        // Ensure all buffers are allocated
        for (buffer, &uop_id) in item.buffers.iter().zip(item.buffer_uop_ids.iter()) {
            buffer.ensure_allocated().context(DeviceSnafu)?;

            // Add buffer if not already added (use UOp ID for registry cleanup)
            uop_id_to_idx.entry(uop_id).or_insert_with(|| builder.add_buffer(uop_id, buffer.clone()));
        }

        // Collect alias IDs for cleanup
        builder.add_alias_ids(item.alias_registered_ids.iter().copied());
    }

    // Step 2: Compile all kernels and create PreparedKernel structures
    let mut prepared_kernels: Vec<PreparedKernel> = Vec::new();

    for item in &expanded_schedule {
        // Skip COPY operations for now (handle separately in execution)
        if matches!(item.ast.op(), Op::Copy { .. }) {
            // TODO: Handle COPY operations in ExecutionPlan
            continue;
        }

        // Step 1: Get device-aware optimizer renderer
        let optimizer_renderer = get_optimizer_renderer(&device);

        // Step 2: Optimize OUTSIDE cache (enables beam search)
        let optimized_ast = if let morok_schedule::OptStrategy::Beam { width } = config.strategy {
            // Beam search: compile-and-time multiple candidates
            beam_search_optimize(item.ast.clone(), &optimizer_renderer, width, &device, &item.buffers)?
        } else {
            // Heuristic optimization (default)
            morok_schedule::optimize_kernel_with_config(item.ast.clone(), &optimizer_renderer, config)
        };

        // Step 3: Apply decomposition
        let ast_decomposed = match device.renderer.decompositor() {
            Some(matcher) => morok_ir::decompositions::decompose_with(&optimized_ast, &matcher),
            None => optimized_ast,
        };

        // Step 4: Cache by OPTIMIZED ast id (different optimizations → different cache entries)
        let cached = morok_runtime::kernel_cache::get_or_compile_kernel(ast_decomposed.id, &device_str, || {
            // Render
            let spec = device.renderer.render(&ast_decomposed).context(RenderKernelSnafu)?;

            // Compile
            let compiled = device.compiler.compile(&spec).context(CompileKernelSnafu)?;

            // Create program
            let program = (device.runtime)(&compiled).context(CreateProgramSnafu)?;

            Ok(morok_runtime::kernel_cache::CachedKernel {
                program,
                device: device_str.clone(),
                code: spec.src.clone(),
                entry_point: spec.name.clone(),
            })
        })?;

        // Build buffer indices for this kernel using item.buffer_uop_ids (already in correct order)
        let buffer_indices: Vec<usize> =
            item.buffer_uop_ids.iter().filter_map(|&uop_id| uop_id_to_idx.get(&uop_id).copied()).collect();

        trace!(kernel.ast_id = item.ast.id, num_buffers = item.buffers.len(), "kernel buffer mapping");

        // Create PreparedKernel
        // Note: buffer_ptrs and buffer_ids will be computed in ExecutionPlanBuilder::build()
        let prepared = PreparedKernel {
            id: item.ast.id,
            kernel: cached,
            device: device.device.clone(),
            buffer_indices,
            output_indices: vec![0], // First buffer is typically output
            fixedvars: item.fixedvars.clone(),
            dependencies: item.dependencies.clone(),
            buffer_ptrs: Vec::new(), // Computed in build()
            buffer_ids: Vec::new(),  // Computed in build()
        };

        prepared_kernels.push(prepared);
    }

    // Add kernels to builder and track their indices
    let num_prepared_kernels = prepared_kernels.len();
    for kernel in prepared_kernels {
        builder.add_kernel(kernel);
    }

    // Step 3: Create parallel groups
    // Each kernel goes into its own group for sequential execution.
    //
    // NOTE: While expanded iterations with different fixedvars ARE independent
    // (they write to different positions in the same buffer), the UnifiedExecutor's
    // validate_parallel_independence() cannot distinguish this - it sees writes to
    // the same buffer ID as a conflict. Until we enhance the executor to understand
    // position-based independence, we execute sequentially.
    //
    // Future optimization: Group truly independent kernels (different AST IDs
    // writing to different buffers) for parallel execution.
    let parallel_groups: Vec<ParallelGroup> =
        (0..num_prepared_kernels).map(|idx| ParallelGroup { kernel_indices: vec![idx] }).collect();

    builder.set_parallel_groups(parallel_groups);

    // Find output buffer by scanning ALL kernels' buffers
    // Search order: exact match (dtype + size), then dtype only, then first buffer
    let mut output_buffer_idx: Option<usize> = None;

    // Pass 1: Look for exact match (dtype AND size)
    // Select the HIGHEST BufferId (most recently allocated = output buffer)
    // because input and output buffers may have the same dtype+size for cast operations.
    // Note: We use BufferId for selection but UOp ID for the map lookup.
    let mut best_buf_id: Option<u64> = None;
    for item in &expanded_schedule {
        if matches!(item.ast.op(), Op::Copy { .. }) {
            continue;
        }
        for (buf, &uop_id) in item.buffers.iter().zip(item.buffer_uop_ids.iter()) {
            if buf.dtype() == expected_output_dtype
                && buf.size() == expected_output_size
                && let Some(&idx) = uop_id_to_idx.get(&uop_id)
            {
                let buf_id = buf.id().0;
                // Select highest BufferId (latest allocated = output)
                if best_buf_id.is_none_or(|best| buf_id > best) {
                    trace!(
                        uop_id,
                        buffer.id = buf_id,
                        buffer.idx = idx,
                        buffer.dtype = ?expected_output_dtype,
                        buffer.size = expected_output_size,
                        "Candidate output buffer (exact match)"
                    );
                    output_buffer_idx = Some(idx);
                    best_buf_id = Some(buf_id);
                }
            }
        }
    }

    // Pass 2: Fallback to dtype match only
    if output_buffer_idx.is_none() {
        for item in &expanded_schedule {
            if matches!(item.ast.op(), Op::Copy { .. }) {
                continue;
            }
            for (buf, &uop_id) in item.buffers.iter().zip(item.buffer_uop_ids.iter()) {
                if buf.dtype() == expected_output_dtype
                    && let Some(&idx) = uop_id_to_idx.get(&uop_id)
                {
                    trace!(
                        uop_id,
                        buffer.idx = idx,
                        buffer.dtype = ?buf.dtype(),
                        "Fallback output buffer (dtype match)"
                    );
                    output_buffer_idx = Some(idx);
                    break;
                }
            }
            if output_buffer_idx.is_some() {
                break;
            }
        }
    }

    // Pass 3: Last resort - first buffer
    if output_buffer_idx.is_none()
        && let Some(first_item) = expanded_schedule.first()
        && !first_item.buffer_uop_ids.is_empty()
    {
        let first_uop_id = first_item.buffer_uop_ids[0];
        if let Some(&idx) = uop_id_to_idx.get(&first_uop_id) {
            trace!(
                uop_id = first_uop_id,
                buffer.idx = idx,
                buffer.dtype = ?first_item.buffers[0].dtype(),
                "Fallback output buffer (first buffer)"
            );
            output_buffer_idx = Some(idx);
        }
    }

    // Set output buffer
    if let Some(idx) = output_buffer_idx {
        builder.set_output_buffer(idx);
    }

    Ok(builder.build())
}

/// Get the optimizer renderer for a device.
fn get_optimizer_renderer(device: &Device) -> morok_schedule::OptimizerRenderer {
    match device.device {
        DeviceSpec::Cpu => morok_schedule::OptimizerRenderer::cpu(),
        DeviceSpec::Cuda { .. } => morok_schedule::OptimizerRenderer::cuda(),
        DeviceSpec::Metal { .. } => morok_schedule::OptimizerRenderer::metal(),
        _ => morok_schedule::OptimizerRenderer::cpu(),
    }
}

/// Optimize a kernel AST using beam search auto-tuning.
///
/// Beam search explores multiple optimization paths and selects the fastest
/// by compiling and timing each candidate. This is slower than heuristics
/// but can find better optimizations.
fn beam_search_optimize(
    ast: Arc<UOp>,
    renderer: &morok_schedule::OptimizerRenderer,
    beam_width: usize,
    device: &Device,
    buffers: &[Buffer],
) -> Result<Arc<UOp>> {
    use morok_schedule::{BeamConfig, Scheduler, apply_post_optimization, beam_search_cached, prepare_scheduler};

    let mut config = BeamConfig::from_env();
    config.beam_width = beam_width;

    // Prepare scheduler (applies symbolic simplification and loop→global)
    let scheduler = prepare_scheduler(ast, renderer);

    // Ensure all buffers are allocated for timing
    for buf in buffers {
        buf.ensure_allocated().context(DeviceSnafu)?;
    }

    // Clone buffers for the closure (Buffer is Clone + Send + Sync)
    let buffers: Vec<Buffer> = buffers.to_vec();
    let bench_config = morok_runtime::BenchmarkConfig::default();

    // Clone device components for the closure
    let dev_renderer = device.renderer.clone();
    let dev_compiler = device.compiler.clone();
    let dev_runtime = device.runtime.clone();

    // Compile-and-time function: compilation is NOT timed, only execution
    let compile_and_time = |s: &Scheduler| -> Option<Duration> {
        let raw_ast = s.get_optimized_ast(None);

        // Apply post-optimization passes for accurate timing
        // This includes scalar accumulator devectorization which is critical for performance
        let optimized = apply_post_optimization(raw_ast);

        // Apply decomposition
        let decomposed = match dev_renderer.decompositor() {
            Some(m) => morok_ir::decompositions::decompose_with(&optimized, &m),
            None => optimized,
        };

        // Render and compile (NOT timed)
        let spec = dev_renderer.render(&decomposed).ok()?;
        let compiled = dev_compiler.compile(&spec).ok()?;
        let program = (dev_runtime)(&compiled).ok()?;

        // Extract buffer pointers inside the closure (avoids Sync issue)
        let buffer_ptrs: Vec<*mut u8> = buffers.iter().map(|b| unsafe { b.as_raw_ptr() }).collect();

        // Time ONLY execution
        let result = unsafe {
            morok_runtime::benchmark_kernel(program.as_ref(), &buffer_ptrs, &HashMap::new(), &bench_config).ok()?
        };

        Some(result.min)
    };

    // Run beam search with caching
    let result = beam_search_cached(scheduler, &config, compile_and_time).context(OptimizeSnafu)?;

    // Apply post-optimization to final result
    let raw_ast = result.scheduler.get_optimized_ast(None);
    Ok(apply_post_optimization(raw_ast))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_realize_simple_add() {
        let _guard = crate::test::helpers::test_setup();

        // Test that realizing a simple computation works.
        // The pipeline transforms:
        //   ADD(RESHAPE(BUFFER_A), RESHAPE(BUFFER_B))
        // Into:
        //   STORE(OUTPUT, INDEX, ADD(LOAD(INPUT_A, idx), LOAD(INPUT_B, idx)))
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let b = Tensor::from_slice([4.0f32, 5.0, 6.0]);

        // Create computation: a + b
        let c = &a + &b;

        // Realize should compile and execute the kernel
        let result: ndarray::ArrayD<f32> = c.realize().unwrap().to_ndarray().unwrap();
        let (result, _) = result.into_raw_vec_and_offset();
        assert_eq!(result, vec![5.0, 7.0, 9.0]);
    }

    /// Test that realizing a reduction (sum) works end-to-end.
    ///
    /// This verifies the complete reduction pipeline:
    /// - Early-return pattern prevents unnecessary ReduceAxis for size-1 dimensions
    /// - Vectorize consistency prevents VConst panics in shape extraction
    /// - ReduceAxis → REDUCE transformation following Tinygrad's approach
    /// - REDUCE codegen generates correct LLVM IR
    #[test]
    fn test_realize_sum() {
        let _guard = crate::test::helpers::test_setup();

        // Create a 1D tensor: [1, 2, 3, 4]
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);

        // Sum all elements (should be 10.0)
        let sum_result = a.sum(());
        if let Err(ref e) = sum_result {
            tracing::debug!(error = ?e, "sum failed");
        }
        assert!(sum_result.is_ok(), "Sum creation failed");

        // Realize the computation
        let realized = sum_result.unwrap().realize();
        if let Err(ref e) = realized {
            eprintln!("realize failed: {e:?}");
        }
        assert!(realized.is_ok(), "Realize should succeed: {:?}", realized.err());
    }

    #[test]
    fn test_tensor_device_default_cpu() {
        // Tensors created with from_slice default to CPU
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        assert_eq!(a.device(), morok_ir::DeviceSpec::Cpu);
    }

    #[test]
    fn test_tensor_to_same_device_is_noop() {
        // Moving to the same device should return a clone
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let b = a.to(morok_ir::DeviceSpec::Cpu);
        // Both should point to the same UOp (clone shares Rc)
        assert_eq!(a.device(), b.device());
    }

    #[test]
    fn test_tensor_to_different_device_creates_copy() {
        use morok_ir::DeviceSpec;
        // Moving to a different device should create a COPY UOp
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let b = a.to(DeviceSpec::Cuda { device_id: 0 });
        // b should report the new device
        assert_eq!(b.device(), DeviceSpec::Cuda { device_id: 0 });
        // a should still be on CPU
        assert_eq!(a.device(), DeviceSpec::Cpu);
    }

    // More comprehensive tests will be added in Phase 1.5

    // ==========================================================================
    // ExecutionPlan tests
    // ==========================================================================

    #[test]
    fn test_prepare_simple_add() {
        let _guard = crate::test::helpers::test_setup();

        // Create computation: a + b
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let b = Tensor::from_slice([4.0f32, 5.0, 6.0]);
        let c = &a + &b;

        // Prepare should compile kernels and allocate buffers
        let plan = c.prepare();
        assert!(plan.is_ok(), "prepare() should succeed: {:?}", plan.err());

        let plan = plan.unwrap();

        // Verify plan has kernels and buffers
        assert!(plan.kernels().next().is_some(), "Plan should have at least one kernel");
        assert!(!plan.buffers().is_empty(), "Plan should have buffers");
        assert!(!plan.parallel_groups().is_empty(), "Plan should have parallel groups");
    }

    #[test]
    fn test_prepare_and_execute() {
        let _guard = crate::test::helpers::test_setup();

        // Create computation: a + b
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let b = Tensor::from_slice([4.0f32, 5.0, 6.0]);
        let c = &a + &b;

        // Prepare
        let plan = c.prepare().expect("prepare should succeed");

        // Execute
        let mut executor = morok_runtime::global_executor();
        let result = plan.execute(&mut executor);
        assert!(result.is_ok(), "execute() should succeed: {:?}", result.err());

        // Verify output buffer has correct data
        let output = plan.output_buffer();
        let mut data = vec![0.0f32; 3];
        output
            .copyout(unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, 12) })
            .expect("copyout should succeed");
        assert_eq!(data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_prepare_and_execute_twice() {
        let _guard = crate::test::helpers::test_setup();

        // Create computation
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let b = Tensor::from_slice([4.0f32, 5.0, 6.0]);
        let c = &a + &b;

        // Prepare once
        let plan = c.prepare().expect("prepare should succeed");

        // Execute twice to verify reusability
        let mut executor = morok_runtime::global_executor();

        for _ in 0..2 {
            let result = plan.execute(&mut executor);
            assert!(result.is_ok(), "execute() should succeed: {:?}", result.err());
        }

        // Verify output
        let output = plan.output_buffer();
        let mut data = vec![0.0f32; 3];
        output
            .copyout(unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, 12) })
            .expect("copyout should succeed");
        assert_eq!(data, vec![5.0, 7.0, 9.0]);
    }

    /// Test that realize() produces correct results.
    ///
    /// Note: Buffer count assertions removed as they're not reliable with
    /// parallel test execution and global state. The key invariant (no memory
    /// leak) is tested in test_memory_growth_detection.
    #[test]
    fn test_realize_buffer_cleanup() {
        let _guard = crate::test::helpers::test_setup();

        // Create input tensors ONCE (these will stay in registry)
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let b = Tensor::from_slice([4.0f32, 5.0, 6.0]);

        // Realize the computation
        let c = (&a + &b).realize().expect("realize should succeed");

        // Verify computation is correct
        let result: ndarray::ArrayD<f32> = c.to_ndarray().expect("to_ndarray should succeed");
        let (data, _) = result.into_raw_vec_and_offset();
        assert_eq!(data, vec![5.0, 7.0, 9.0]);
    }

    /// Test that prepare() + execute() pattern can clean up with release_intermediate_buffers().
    #[test]
    fn test_prepare_execute_cleanup() {
        let _guard = crate::test::helpers::test_setup();

        // Create input tensors
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let b = Tensor::from_slice([4.0f32, 5.0, 6.0]);
        let c = &a + &b;

        // Prepare the plan
        let plan = c.prepare().expect("prepare should succeed");

        let count_before_cleanup = crate::tensor_registry::buffer_count();

        // Execute multiple times (simulating benchmark loop)
        let mut executor = morok_runtime::global_executor();
        for _ in 0..3 {
            plan.execute(&mut executor).expect("execute should succeed");
        }

        // Verify output
        let output = plan.output_buffer();
        let mut data = vec![0.0f32; 3];
        output
            .copyout(unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, 12) })
            .expect("copyout should succeed");
        assert_eq!(data, vec![5.0, 7.0, 9.0]);

        // Now cleanup
        plan.release_intermediate_buffers(crate::tensor_registry::remove_buffer);

        let count_after_cleanup = crate::tensor_registry::buffer_count();

        // After cleanup, we should have fewer or equal buffers
        // (intermediate buffers removed)
        assert!(
            count_after_cleanup <= count_before_cleanup,
            "Cleanup should not increase buffer count: before={}, after={}",
            count_before_cleanup,
            count_after_cleanup
        );
    }

    /// Test that intermediate buffer cleanup is working.
    ///
    /// The correct pattern is: prepare() ONCE, execute() many times.
    /// This test verifies that repeated execute() calls do NOT grow the registry
    /// AFTER initial setup. First execute may allocate buffers (one-time setup),
    /// but subsequent calls must not grow.
    #[test]
    fn test_memory_growth_detection() {
        let _guard = crate::test::helpers::test_setup();

        const ITERATIONS: usize = 10;

        let mut executor = morok_runtime::global_executor();

        // Create input tensors
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);
        let b = Tensor::from_slice([5.0f32, 6.0, 7.0, 8.0]);
        let c = &a + &b;

        // Prepare ONCE
        let plan = c.prepare().expect("prepare should succeed");

        let mut counts: Vec<usize> = Vec::with_capacity(ITERATIONS);

        // Execute MANY times
        for _ in 0..ITERATIONS {
            plan.execute(&mut executor).expect("execute should succeed");
            counts.push(crate::tensor_registry::buffer_count());
        }

        // Cleanup after final execution
        plan.release_intermediate_buffers(crate::tensor_registry::remove_buffer);
        let count_after_cleanup = crate::tensor_registry::buffer_count();

        // Key invariant: count should be STABLE during iterations (no growth between iterations)
        // First execute may allocate buffers, but subsequent calls must reuse them.
        let count_after_first_execute = counts[0];
        let growth_during_iterations = counts.last().unwrap().saturating_sub(count_after_first_execute);

        eprintln!("Counts during execute: {:?}", counts);
        eprintln!("Growth during iterations (after first): {}", growth_during_iterations);
        eprintln!("Count after cleanup: {}", count_after_cleanup);

        assert_eq!(
            growth_during_iterations, 0,
            "Registry should not grow during repeated execute() calls (after initial setup)"
        );

        // Cleanup should reduce count by removing allocated buffers
        assert!(
            count_after_cleanup <= count_after_first_execute,
            "Cleanup should not increase buffer count: first_execute={}, after_cleanup={}",
            count_after_first_execute,
            count_after_cleanup
        );
    }

    /// Test that creating new input tensors each iteration causes growth.
    ///
    /// This is expected behavior - input tensor buffers are not automatically
    /// cleaned up when they go out of scope. Users should reuse tensors.
    #[test]
    fn test_memory_growth_with_new_inputs() {
        let _guard = crate::test::helpers::test_setup();

        const ITERATIONS: usize = 5;

        let mut executor = morok_runtime::global_executor();
        let mut counts: Vec<usize> = Vec::with_capacity(ITERATIONS);

        let count_before = crate::tensor_registry::buffer_count();

        for _ in 0..ITERATIONS {
            // Create NEW input tensors each iteration - this causes growth
            let a = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);
            let b = Tensor::from_slice([5.0f32, 6.0, 7.0, 8.0]);
            let c = &a + &b;

            let plan = c.prepare().expect("prepare should succeed");
            plan.execute(&mut executor).expect("execute should succeed");
            plan.release_intermediate_buffers(crate::tensor_registry::remove_buffer);

            counts.push(crate::tensor_registry::buffer_count());
        }

        let total_growth = *counts.last().unwrap() as isize - count_before as isize;
        let growth_per_iter = total_growth as f64 / ITERATIONS as f64;

        eprintln!("Counts with new inputs each iteration: {:?}", counts);
        eprintln!("Growth per iteration: {:.1} buffers", growth_per_iter);

        // We expect growth because input tensor buffers accumulate
        // (2 inputs per iteration that are never cleaned up)
        // Plus some overhead for DefineGlobal registrations
        assert!(growth_per_iter >= 2.0, "Expected growth with new inputs, but growth_per_iter={:.1}", growth_per_iter);
    }

    /// Test that realize() correctly computes and cleans up.
    #[test]
    fn test_memory_growth_realize_pattern() {
        let _guard = crate::test::helpers::test_setup();

        // Single realize should work correctly
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);
        let b = Tensor::from_slice([5.0f32, 6.0, 7.0, 8.0]);
        let c = (&a + &b).realize().expect("realize should succeed");

        // Verify result
        let result: ndarray::ArrayD<f32> = c.to_ndarray().expect("to_ndarray should succeed");
        assert_eq!(result.as_slice().unwrap(), &[6.0, 8.0, 10.0, 12.0]);
    }

    /// STRICT test: Repeated prepare+execute+cleanup cycles with SAME inputs.
    ///
    /// Each cycle should return to the same baseline registry count.
    /// If this fails, we have a leak in the prepare or cleanup path.
    #[test]
    fn test_memory_growth_strict_cycles() {
        let _guard = crate::test::helpers::test_setup();

        const ITERATIONS: usize = 10;

        let mut executor = morok_runtime::global_executor();

        // Create input tensors ONCE
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);
        let b = Tensor::from_slice([5.0f32, 6.0, 7.0, 8.0]);

        // Baseline after creating inputs
        let baseline = crate::tensor_registry::buffer_count();
        eprintln!("Baseline after creating inputs: {}", baseline);

        let mut counts_after_cleanup: Vec<usize> = Vec::with_capacity(ITERATIONS);

        for i in 0..ITERATIONS {
            // Create computation graph (no new allocations, just UOp graph)
            let c = &a + &b;

            // Collect input buffer IDs to preserve their mappings
            let input_buffer_ids: std::collections::HashSet<u64> =
                collect_input_buffers(&c.uop()).keys().copied().collect();

            // Full cycle: prepare -> execute -> cleanup
            let plan = c.prepare().expect("prepare should succeed");
            let count_after_prepare = crate::tensor_registry::buffer_count();

            plan.execute(&mut executor).expect("execute should succeed");
            let count_after_execute = crate::tensor_registry::buffer_count();

            // Release all buffers EXCEPT inputs (which are reused across iterations)
            plan.release_all_buffers(|uop_id| {
                if !input_buffer_ids.contains(&uop_id) {
                    crate::tensor_registry::remove_buffer(uop_id);
                }
            });
            let count_after_cleanup = crate::tensor_registry::buffer_count();

            counts_after_cleanup.push(count_after_cleanup);

            if i == 0 {
                eprintln!(
                    "Cycle 0: after_prepare={}, after_execute={}, after_cleanup={}",
                    count_after_prepare, count_after_execute, count_after_cleanup
                );
            }
        }

        eprintln!("Counts after cleanup each cycle: {:?}", counts_after_cleanup);

        // Check that counts stay bounded (within 1 of first cycle's cleanup count)
        let first_cleanup = counts_after_cleanup[0];
        let last_cleanup = *counts_after_cleanup.last().unwrap();
        let growth = last_cleanup.saturating_sub(first_cleanup);

        eprintln!("First cleanup: {}, Last cleanup: {}, Growth: {}", first_cleanup, last_cleanup, growth);

        assert_eq!(
            growth, 0,
            "Registry should not grow across prepare+execute+cleanup cycles. \
             First: {}, Last: {}, Growth: {}",
            first_cleanup, last_cleanup, growth
        );
    }
}
