//! Unified parallel execution for heterogeneous devices.
//!
//! The `UnifiedExecutor` handles kernel execution across any mix of devices
//! (CPU, CUDA, Metal, etc.) with proper synchronization and dependency tracking.
//!
//! # Design Principles
//!
//! 1. **Single abstraction** - One executor handles any device mix
//! 2. **Device-agnostic sync** - Timeline signals abstract over device-specific primitives
//! 3. **Zero overhead for single-device** - Fast path skips synchronization when possible
//! 4. **Buffer dependency tracking** - Following Tinygrad's `_access_resources()` pattern
//!
//! # Example
//!
//! ```ignore
//! let mut executor = UnifiedExecutor::new();
//! executor.add_device(DeviceSpec::Cpu)?;
//!
//! // Execute schedule - handles dependencies automatically
//! let output_id = executor.execute(&schedule)?;
//! ```
//!
//! # Execution Graph
//!
//! For complex schedules with multiple devices, the executor builds an execution
//! graph (DAG) where nodes are kernel operations and edges are buffer dependencies.
//! Independent kernels on the same device can be batched, and kernels on different
//! devices can run in parallel (with appropriate synchronization).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use tracing::debug;

use morok_device::device::Device;
use morok_device::registry::DeviceRegistry;
use morok_device::{Allocator, Buffer, BufferId, CpuTimelineSignal, TimelineSignal};
use morok_dtype::DeviceSpec;
use snafu::ResultExt;

use crate::error::Result;
use crate::execution_plan::PreparedKernel;

// ============================================================================
// Dependency Tracking Types (Tinygrad-Aligned)
// ============================================================================

/// A dependency token representing a completed operation.
///
/// Unlike timeline values, dependency tokens are opaque references to past operations.
/// Only tokens from ALREADY-COMPLETED operations are returned by `query_resources`,
/// ensuring we never wait on unsignaled timelines.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DependencyToken {
    /// Unique operation identifier (kernel ID or transfer ID).
    pub operation_id: u64,
    /// Device that executed the operation.
    pub device: DeviceSpec,
    /// Timeline value that was signaled upon completion.
    pub timeline: u64,
}

/// Query result from `query_resources` - dependencies to wait on.
///
/// This struct is returned by the pure query phase and contains only
/// references to ALREADY-SIGNALED operations.
#[derive(Debug, Default)]
pub struct DependencyQuery {
    /// Tokens of operations that must complete before the current operation.
    /// All tokens in this list represent operations that have ALREADY been signaled.
    pub wait_on: Vec<DependencyToken>,
}

/// Deferred update to apply AFTER operation completes and signals.
///
/// This struct captures the state changes that must be applied to dependency
/// maps after an operation successfully completes. By deferring updates,
/// we ensure parallel operations don't see incorrect state.
#[derive(Debug)]
pub struct DependencyUpdate {
    /// Buffer IDs written by this operation.
    writes: Vec<BufferId>,
    /// Buffer IDs read by this operation.
    reads: Vec<BufferId>,
    /// Token representing this operation (to be stored in maps).
    token: DependencyToken,
}

/// Per-device execution context.
///
/// Each device has its own timeline signal, queue, and allocator.
/// This enables parallel execution across devices with proper synchronization.
pub struct DeviceContext {
    /// Device specification (CPU, CUDA:0, etc.).
    pub device: DeviceSpec,
    /// Device abstraction for rendering/compiling/executing.
    pub device_handle: Arc<Device>,
    /// Timeline signal for this device's operations.
    pub signal: Arc<dyn TimelineSignal>,
    /// Current timeline value (monotonically increasing).
    pub timeline: AtomicU64,
    /// Allocator for this device.
    pub allocator: Arc<dyn Allocator>,
}

impl std::fmt::Debug for DeviceContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceContext")
            .field("device", &self.device)
            .field("timeline", &self.timeline.load(Ordering::Relaxed))
            .finish()
    }
}

impl DeviceContext {
    /// Create a new device context.
    pub fn new(device: Arc<Device>, signal: Arc<dyn TimelineSignal>) -> Self {
        let allocator = device.allocator.clone();
        let device_spec = device.device.clone();
        Self { device: device_spec, device_handle: device, signal, timeline: AtomicU64::new(0), allocator }
    }

    /// Get the next timeline value and increment.
    pub fn next_timeline(&self) -> u64 {
        self.timeline.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Get the current timeline value.
    pub fn current_timeline(&self) -> u64 {
        self.timeline.load(Ordering::Relaxed)
    }

    /// Signal that operations up to the given timeline value are complete.
    pub fn signal_completion(&self, value: u64) {
        self.signal.set(value);
    }

    /// Wait for operations up to the given timeline value to complete.
    pub fn wait_for(&self, value: u64) -> Result<()> {
        self.signal.wait(value, 0).context(crate::error::DeviceSnafu)?;
        Ok(())
    }
}

/// Cross-device synchronization strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncStrategy {
    /// Same device - no synchronization needed (operations are ordered).
    None,
    /// Same device type, different instance (e.g., CUDA:0 → CUDA:1).
    /// Use peer-to-peer events if available.
    PeerToPeer,
    /// Different device types (e.g., CUDA → CPU).
    /// Use CPU-mediated polling.
    CpuMediated,
}

/// A node in the execution graph representing a kernel or transfer operation.
#[derive(Debug, Clone)]
pub struct ExecutionNode {
    /// Unique identifier for this node (typically the kernel AST ID).
    pub id: u64,
    /// Device this operation executes on.
    pub device: DeviceSpec,
    /// Buffer IDs read by this operation.
    pub inputs: Vec<BufferId>,
    /// Buffer IDs written by this operation.
    pub outputs: Vec<BufferId>,
    /// IDs of nodes that must complete before this one (dependencies).
    pub predecessors: Vec<u64>,
    /// Whether this is a data transfer (COPY) or a computational kernel.
    pub is_transfer: bool,
    /// Buffer access information for parallel execution.
    /// Contains the full buffer list and output indices for dependency tracking.
    pub buffer_access: Option<KernelBufferAccess>,
}

/// Buffer access information for parallel kernel execution.
///
/// This struct captures which buffers a kernel accesses and which are outputs,
/// enabling precise dependency tracking in `execute_parallel_group`.
#[derive(Debug, Clone)]
pub struct KernelBufferAccess {
    /// All buffer IDs accessed by this kernel (inputs and outputs).
    pub buffers: Vec<BufferId>,
    /// Indices into `buffers` that are outputs (written by the kernel).
    /// Other indices are inputs (read-only).
    pub output_indices: Vec<usize>,
}

/// Execution graph representing a DAG of kernel operations.
///
/// The graph is built from a schedule and captures buffer dependencies
/// between kernels. Independent kernels can be executed in parallel.
#[derive(Debug, Default)]
pub struct ExecutionGraph {
    /// Nodes in the graph, indexed by ID.
    nodes: HashMap<u64, ExecutionNode>,
    /// Execution order (topologically sorted node IDs).
    execution_order: Vec<u64>,
    /// Nodes grouped by device for batched execution.
    device_groups: HashMap<DeviceSpec, Vec<u64>>,
}

impl ExecutionGraph {
    /// Create a new empty execution graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a node to the graph.
    pub fn add_node(&mut self, node: ExecutionNode) {
        let id = node.id;
        let device = node.device.clone();
        self.nodes.insert(id, node);
        self.device_groups.entry(device).or_default().push(id);
    }

    /// Get a node by ID.
    pub fn node(&self, id: u64) -> Option<&ExecutionNode> {
        self.nodes.get(&id)
    }

    /// Get all nodes.
    pub fn nodes(&self) -> impl Iterator<Item = &ExecutionNode> {
        self.nodes.values()
    }

    /// Compute topological order and find parallelizable groups.
    ///
    /// Returns groups of nodes that can be executed in parallel.
    /// Each group contains nodes with no dependencies on each other.
    pub fn compute_parallel_groups(&mut self) -> Vec<Vec<u64>> {
        // Build in-degree map
        let mut in_degree: HashMap<u64, usize> = HashMap::new();
        let mut successors: HashMap<u64, Vec<u64>> = HashMap::new();

        for node in self.nodes.values() {
            in_degree.entry(node.id).or_insert(0);
            for &pred in &node.predecessors {
                successors.entry(pred).or_default().push(node.id);
                *in_degree.entry(node.id).or_insert(0) += 1;
            }
        }

        // Kahn's algorithm with level grouping
        let mut groups = Vec::new();
        let mut ready: Vec<u64> = in_degree.iter().filter(|&(_, deg)| *deg == 0).map(|(&id, _)| id).collect();

        while !ready.is_empty() {
            // All nodes in ready can be executed in parallel
            groups.push(ready.clone());

            // Record execution order
            self.execution_order.extend(ready.iter().copied());

            // Find next batch
            let mut next_ready = Vec::new();
            for id in ready {
                if let Some(succs) = successors.get(&id) {
                    for &succ in succs {
                        let deg = in_degree.get_mut(&succ).unwrap();
                        *deg -= 1;
                        if *deg == 0 {
                            next_ready.push(succ);
                        }
                    }
                }
            }
            ready = next_ready;
        }

        groups
    }

    /// Get nodes grouped by device.
    pub fn device_groups(&self) -> &HashMap<DeviceSpec, Vec<u64>> {
        &self.device_groups
    }

    /// Check if all nodes have been visited (no cycles).
    pub fn is_valid(&self) -> bool {
        self.execution_order.len() == self.nodes.len()
    }

    /// Find independent kernels that can run in parallel.
    ///
    /// Two kernels are independent if:
    /// 1. Neither depends on the other (no path in DAG)
    /// 2. They don't write to the same buffer
    /// 3. One doesn't read what the other writes
    pub fn find_independent_kernels(&self, node_ids: &[u64]) -> Vec<Vec<u64>> {
        if node_ids.len() <= 1 {
            return vec![node_ids.to_vec()];
        }

        // Build conflict map (which nodes conflict with which)
        let mut conflicts: HashMap<u64, HashSet<u64>> = HashMap::new();

        for &id1 in node_ids {
            for &id2 in node_ids {
                if id1 >= id2 {
                    continue;
                }

                let node1 = match self.nodes.get(&id1) {
                    Some(n) => n,
                    None => continue,
                };
                let node2 = match self.nodes.get(&id2) {
                    Some(n) => n,
                    None => continue,
                };

                // Check for output conflicts
                let outputs1: HashSet<_> = node1.outputs.iter().collect();
                let outputs2: HashSet<_> = node2.outputs.iter().collect();

                // Same output buffer = conflict
                if !outputs1.is_disjoint(&outputs2) {
                    conflicts.entry(id1).or_default().insert(id2);
                    conflicts.entry(id2).or_default().insert(id1);
                    continue;
                }

                // Read-write conflict
                let inputs1: HashSet<_> = node1.inputs.iter().collect();
                let inputs2: HashSet<_> = node2.inputs.iter().collect();

                if !inputs1.is_disjoint(&outputs2) || !inputs2.is_disjoint(&outputs1) {
                    conflicts.entry(id1).or_default().insert(id2);
                    conflicts.entry(id2).or_default().insert(id1);
                    continue;
                }

                // Check DAG dependency (predecessor relationship)
                if node1.predecessors.contains(&id2) || node2.predecessors.contains(&id1) {
                    conflicts.entry(id1).or_default().insert(id2);
                    conflicts.entry(id2).or_default().insert(id1);
                }
            }
        }

        // Greedy graph coloring to find independent sets
        let mut groups: Vec<Vec<u64>> = Vec::new();
        let mut assigned: HashSet<u64> = HashSet::new();

        for &id in node_ids {
            if assigned.contains(&id) {
                continue;
            }

            // Try to add to existing group
            let mut added = false;
            for group in &mut groups {
                let node_conflicts = conflicts.get(&id).cloned().unwrap_or_default();
                if group.iter().all(|&g| !node_conflicts.contains(&g)) {
                    group.push(id);
                    assigned.insert(id);
                    added = true;
                    break;
                }
            }

            if !added {
                groups.push(vec![id]);
                assigned.insert(id);
            }
        }

        groups
    }
}

/// Unified executor for heterogeneous device execution.
///
/// Manages device contexts and buffer dependencies to enable parallel
/// execution across any mix of devices.
///
/// # Tinygrad-Aligned Dependency Model
///
/// The executor uses a split query/update pattern for dependency tracking:
/// - `query_resources`: Pure function that returns dependencies without mutation
/// - `update_resources`: Applies deferred updates AFTER operation signals completion
///
/// This separation prevents race conditions in parallel execution by ensuring:
/// 1. All dependency queries happen before parallel execution starts
/// 2. Map updates only occur after operations complete and signal
/// 3. No operation ever waits on an unsignaled timeline
pub struct UnifiedExecutor {
    /// Per-device execution contexts.
    contexts: HashMap<DeviceSpec, DeviceContext>,

    /// Buffer write dependency map: BufferId → DependencyToken.
    /// Tracks the last operation that wrote to each buffer.
    /// Tokens are ONLY inserted after the operation signals completion.
    write_deps: HashMap<BufferId, DependencyToken>,

    /// Buffer read dependency map: BufferId → Vec<DependencyToken>.
    /// Tracks all operations currently reading each buffer.
    /// Tokens are ONLY inserted after operations signal completion.
    read_deps: HashMap<BufferId, Vec<DependencyToken>>,

    /// Device registry for looking up allocators.
    registry: &'static DeviceRegistry,
}

impl std::fmt::Debug for UnifiedExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnifiedExecutor")
            .field("contexts", &self.contexts.keys().collect::<Vec<_>>())
            .field("write_deps", &self.write_deps.len())
            .field("read_deps", &self.read_deps.len())
            .finish()
    }
}

impl UnifiedExecutor {
    /// Create a new unified executor.
    pub fn new(registry: &'static DeviceRegistry) -> Self {
        Self { contexts: HashMap::new(), write_deps: HashMap::new(), read_deps: HashMap::new(), registry }
    }

    /// Add a device to the executor.
    ///
    /// Creates the device context with timeline signal and queues.
    pub fn add_device(&mut self, device_spec: DeviceSpec) -> Result<()> {
        if self.contexts.contains_key(&device_spec) {
            return Ok(()); // Already added
        }

        // Create device handle
        let device = crate::DEVICE_FACTORIES.device(&device_spec, self.registry)?;

        // Create timeline signal based on device type
        let signal: Arc<dyn TimelineSignal> = match &device_spec {
            DeviceSpec::Cpu => Arc::new(CpuTimelineSignal::new()),
            #[cfg(feature = "cuda")]
            DeviceSpec::Cuda { .. } => {
                // TODO: Create CUDA timeline signal with events
                // For now, fall back to CPU signal (works, but less efficient)
                Arc::new(CpuTimelineSignal::new())
            }
            _ => Arc::new(CpuTimelineSignal::new()),
        };

        let ctx = DeviceContext::new(device, signal);
        self.contexts.insert(device_spec, ctx);

        Ok(())
    }

    /// Get the device context for a device specification.
    pub fn context(&self, device: &DeviceSpec) -> Option<&DeviceContext> {
        self.contexts.get(device)
    }

    /// Get the device context mutably.
    pub fn context_mut(&mut self, device: &DeviceSpec) -> Option<&mut DeviceContext> {
        self.contexts.get_mut(device)
    }

    /// Determine the synchronization strategy between two devices.
    pub fn sync_strategy(from: &DeviceSpec, to: &DeviceSpec) -> SyncStrategy {
        if from == to {
            SyncStrategy::None
        } else if std::mem::discriminant(from) == std::mem::discriminant(to) {
            // Same device type (e.g., both CUDA)
            SyncStrategy::PeerToPeer
        } else {
            // Different device types
            SyncStrategy::CpuMediated
        }
    }

    /// Query buffer dependencies for an operation (PURE - no state mutation).
    ///
    /// This is the first phase of the split query/update pattern. It returns:
    /// 1. Dependencies to wait on (from ALREADY-COMPLETED operations)
    /// 2. A deferred update to apply AFTER this operation completes
    ///
    /// # Arguments
    ///
    /// * `buffers` - All buffers accessed by this operation
    /// * `output_indices` - Indices of buffers being written (outputs)
    /// * `device` - Device executing this operation
    /// * `operation_id` - Unique identifier for this operation
    /// * `timeline` - Timeline value for this operation
    /// * `skip_same_device` - When true, skip same-device deps (hardware queue ordering).
    ///   Use `true` for sequential execution, `false` for parallel groups (rayon).
    ///
    /// # Returns
    ///
    /// Tuple of (DependencyQuery, DependencyUpdate).
    /// - Query contains tokens to wait on (all already signaled)
    /// - Update should be applied AFTER operation signals completion
    pub fn query_resources(
        &self,
        buffer_ids: &[BufferId],
        output_indices: &[usize],
        device: &DeviceSpec,
        operation_id: u64,
        timeline: u64,
        skip_same_device: bool,
    ) -> (DependencyQuery, DependencyUpdate) {
        let mut wait_on = Vec::new();
        let mut writes = Vec::new();
        let mut reads = Vec::new();

        for (i, &buf_id) in buffer_ids.iter().enumerate() {
            let is_output = output_indices.contains(&i);

            // Collect EXISTING writer dependency (already completed)
            if let Some(token) = self.write_deps.get(&buf_id) {
                // Skip same-device deps only when hardware queue provides ordering
                if !skip_same_device || &token.device != device {
                    wait_on.push(token.clone());
                }
            }

            if is_output {
                writes.push(buf_id);
                // If writing, also wait for all current readers (WAR hazard)
                if let Some(readers) = self.read_deps.get(&buf_id) {
                    for token in readers {
                        if !skip_same_device || &token.device != device {
                            wait_on.push(token.clone());
                        }
                    }
                }
            } else {
                reads.push(buf_id);
            }
        }

        // Deduplicate wait dependencies
        let unique: HashSet<_> = wait_on.drain(..).collect();
        wait_on.extend(unique);

        let token = DependencyToken { operation_id, device: device.clone(), timeline };
        (DependencyQuery { wait_on }, DependencyUpdate { writes, reads, token })
    }

    /// Apply deferred dependency updates AFTER operation completes and signals.
    ///
    /// This is the second phase of the split query/update pattern. Call this
    /// ONLY after the operation has signaled completion via `signal_completion`.
    ///
    /// # Critical Ordering
    ///
    /// The call sequence MUST be:
    /// 1. `query_resources` - get dependencies and prepare update
    /// 2. Wait for dependencies (`wait_for_tokens`)
    /// 3. Execute operation
    /// 4. `signal_completion` - signal that operation is done
    /// 5. `update_resources` - apply the deferred update
    ///
    /// This ordering ensures no operation ever waits on an unsignaled timeline.
    pub fn update_resources(&mut self, update: DependencyUpdate) {
        // Update write dependencies
        for buf_id in update.writes {
            self.write_deps.insert(buf_id, update.token.clone());
            // Clear readers since we've overwritten the buffer
            self.read_deps.remove(&buf_id);
        }

        // Update read dependencies
        for buf_id in update.reads {
            self.read_deps.entry(buf_id).or_default().push(update.token.clone());
        }
    }

    /// Wait for all dependency tokens to be signaled.
    ///
    /// Since tokens only come from `query_resources` which returns tokens from
    /// ALREADY-COMPLETED operations, this should never block indefinitely.
    pub fn wait_for_tokens(&self, tokens: &[DependencyToken]) -> Result<()> {
        for token in tokens {
            if let Some(ctx) = self.contexts.get(&token.device) {
                ctx.signal.wait(token.timeline, 0).context(crate::error::DeviceSnafu)?;
            }
        }
        Ok(())
    }

    /// Clear all dependency tracking.
    ///
    /// Call this between independent execution batches.
    pub fn clear_deps(&mut self) {
        self.write_deps.clear();
        self.read_deps.clear();
    }

    /// Check if all operations on a single device.
    ///
    /// Returns `Some(device)` if all buffers are on the same device,
    /// enabling the fast single-device path.
    pub fn single_device_check(&self, buffers: &[&Buffer]) -> Option<DeviceSpec> {
        if buffers.is_empty() {
            return None;
        }

        let first_device = buffers[0].allocator().device_spec();

        for buffer in buffers.iter().skip(1) {
            if buffer.allocator().device_spec() != first_device {
                return None;
            }
        }

        Some(first_device)
    }

    /// Synchronize all devices.
    ///
    /// Waits for all pending operations to complete on all devices.
    pub fn synchronize_all(&self) -> Result<()> {
        for ctx in self.contexts.values() {
            let current = ctx.current_timeline();
            if current > 0 {
                ctx.wait_for(current)?;
            }
        }
        Ok(())
    }

    /// Execute a kernel with dependency tracking (sequential execution).
    ///
    /// Uses the Tinygrad-aligned query/update pattern:
    /// 1. Query dependencies (pure, no mutation)
    /// 2. Wait for cross-device dependencies (same-device skipped - hardware queue ordering)
    /// 3. Execute the kernel
    /// 4. Signal completion
    /// 5. Update dependency maps (AFTER signal)
    ///
    /// # Arguments
    ///
    /// * `kernel_id` - Unique identifier for this kernel execution
    /// * `device` - Device to execute on
    /// * `buffers` - All buffers accessed by this kernel
    /// * `output_indices` - Indices of output buffers in the buffers slice
    /// * `execute_fn` - Function that performs the actual kernel execution
    ///
    /// # Returns
    ///
    /// The timeline value for this execution (can be used for synchronization).
    pub fn execute_kernel<F>(
        &mut self,
        kernel_id: u64,
        device: &DeviceSpec,
        buffer_ids: &[BufferId],
        output_indices: &[usize],
        execute_fn: F,
    ) -> Result<u64>
    where
        F: FnOnce() -> Result<()>,
    {
        // 1. Ensure device context exists
        if !self.contexts.contains_key(device) {
            self.add_device(device.clone())?;
        }

        // 2. Get next timeline value for this execution
        let timeline = self.contexts.get(device).unwrap().next_timeline();

        // 3. Query dependencies (skip_same_device=true for sequential execution)
        // Hardware queue provides ordering for same-device operations
        let (query, update) = self.query_resources(
            buffer_ids,
            output_indices,
            device,
            kernel_id,
            timeline,
            true, // skip_same_device: hardware queue provides ordering
        );

        // 4. Wait for cross-device dependencies (all already signaled)
        self.wait_for_tokens(&query.wait_on)?;

        // 5. Execute the kernel
        execute_fn()?;

        // 6. Signal completion
        if let Some(ctx) = self.contexts.get(device) {
            ctx.signal_completion(timeline);
        }

        // 7. Update maps AFTER signal (critical ordering!)
        self.update_resources(update);

        Ok(timeline)
    }

    /// Execute a buffer transfer (COPY operation) with dependency tracking.
    ///
    /// Uses the Tinygrad-aligned query/update pattern for transfers:
    /// 1. Query dependencies for src (read) and dst (write)
    /// 2. Wait for all dependencies
    /// 3. Perform the transfer
    /// 4. Signal completion
    /// 5. Update dependency maps
    ///
    /// Handles cross-device transfers with appropriate synchronization:
    /// - Same device: Direct copy using device's copy queue
    /// - Same vendor (e.g., CUDA:0 → CUDA:1): Peer-to-peer transfer
    /// - Different vendors (e.g., CUDA → CPU): Stage through host memory
    ///
    /// # Arguments
    ///
    /// * `src` - Source buffer
    /// * `dst` - Destination buffer (must be pre-allocated)
    /// * `src_device` - Device the source buffer is on
    /// * `dst_device` - Device the destination buffer is on
    ///
    /// # Returns
    ///
    /// The timeline value for this transfer operation.
    pub fn execute_transfer(
        &mut self,
        src: &Buffer,
        dst: &mut Buffer,
        src_device: &DeviceSpec,
        dst_device: &DeviceSpec,
    ) -> Result<u64> {
        // Use a unique transfer ID based on buffer IDs
        let transfer_id = src.id().0.wrapping_add(dst.id().0);

        // Ensure both device contexts exist
        if !self.contexts.contains_key(src_device) {
            self.add_device(src_device.clone())?;
        }
        if !self.contexts.contains_key(dst_device) {
            self.add_device(dst_device.clone())?;
        }

        // Get timeline for destination device (where the result will be used)
        let timeline = self.contexts.get(dst_device).unwrap().next_timeline();

        // Query dependencies using the new pattern
        // For a transfer: src is input (index 0), dst is output (index 1)
        // We pass buffer IDs for dependency tracking
        let buffer_ids: [BufferId; 2] = [src.id(), dst.id()];
        let output_indices = [1usize]; // dst is the output

        // For transfers, we don't skip same-device since transfers are explicit synchronization points
        let (query, update) =
            self.query_resources(&buffer_ids, &output_indices, dst_device, transfer_id, timeline, false);

        // Wait for all dependencies
        self.wait_for_tokens(&query.wait_on)?;

        // Perform the transfer based on sync strategy
        match Self::sync_strategy(src_device, dst_device) {
            SyncStrategy::None => {
                // Same device - direct copy
                dst.copy_from(src).context(crate::error::DeviceSnafu)?;
            }
            SyncStrategy::PeerToPeer => {
                // Same vendor (e.g., both CUDA) - use peer-to-peer if available
                // For now, fall back to copy_from which handles this
                dst.copy_from(src).context(crate::error::DeviceSnafu)?;
            }
            SyncStrategy::CpuMediated => {
                // Different vendors - stage through CPU
                // First, wait for source device operations to complete
                if let Some(src_ctx) = self.contexts.get(src_device) {
                    let src_timeline = src_ctx.current_timeline();
                    if src_timeline > 0 {
                        src_ctx.wait_for(src_timeline)?;
                    }
                }

                // copy_from handles the staging internally for cross-device copies
                dst.copy_from(src).context(crate::error::DeviceSnafu)?;

                // Wait for destination device operations if needed
                if let Some(dst_ctx) = self.contexts.get(dst_device) {
                    let dst_timeline = dst_ctx.current_timeline();
                    if dst_timeline > 0 {
                        dst_ctx.wait_for(dst_timeline)?;
                    }
                }
            }
        }

        // Signal completion
        if let Some(ctx) = self.contexts.get(dst_device) {
            ctx.signal_completion(timeline);
        }

        // Update dependency maps AFTER signal
        self.update_resources(update);

        Ok(timeline)
    }

    /// Check if kernels can be executed in parallel.
    ///
    /// Returns true if the kernels have no conflicting buffer accesses.
    /// This is used by the parallel execution path.
    pub fn can_parallelize(&self, kernels: &[(u64, &[&Buffer], &[usize])]) -> bool {
        // Check for output buffer conflicts
        let mut output_buffers: HashSet<morok_device::BufferId> = HashSet::new();

        for (_kernel_id, buffers, output_indices) in kernels {
            for &idx in *output_indices {
                let buf_id = buffers[idx].id();
                if output_buffers.contains(&buf_id) {
                    return false; // Same buffer written by multiple kernels
                }
                output_buffers.insert(buf_id);
            }
        }

        // Check for read-write conflicts (output of one is input of another)
        for (_kernel_id, buffers, output_indices) in kernels {
            for (i, buffer) in buffers.iter().enumerate() {
                if !output_indices.contains(&i) {
                    // This is an input buffer
                    if output_buffers.contains(&buffer.id()) {
                        return false; // Reading a buffer being written
                    }
                }
            }
        }

        true
    }
}

impl UnifiedExecutor {
    /// Execute kernels by indices using scoped parallelism.
    ///
    /// This is the new clean API that:
    /// - Uses `rayon::scope` for borrowed parallel execution (no cloning)
    /// - Takes kernel indices instead of pre-built contexts
    /// - Borrows all data from `PreparedKernel` (fixedvars, buffer_ptrs, etc.)
    ///
    /// # Arguments
    ///
    /// * `all_kernels` - All prepared kernels in the execution plan
    /// * `indices` - Indices of kernels to execute in this group
    /// * `buffers` - All buffers in the execution plan (for validation)
    ///
    /// # Returns
    ///
    /// Vector of (kernel_id, timeline) pairs for each executed kernel.
    pub fn execute_kernels_by_indices(
        &mut self,
        all_kernels: &[PreparedKernel],
        indices: &[usize],
        buffers: &[Buffer],
    ) -> Result<Vec<(u64, u64)>> {
        use std::sync::Mutex;

        if indices.is_empty() {
            return Ok(vec![]);
        }

        // Collect kernel references for this group
        let kernels: Vec<&PreparedKernel> = indices.iter().map(|&idx| &all_kernels[idx]).collect();

        // Single kernel fast path - avoid rayon overhead
        if kernels.len() == 1 {
            let kernel = kernels[0];

            // Ensure device context exists
            if !self.contexts.contains_key(&kernel.device) {
                self.add_device(kernel.device.clone())?;
            }

            let timeline = self.contexts.get(&kernel.device).unwrap().next_timeline();

            // Query dependencies
            let (query, update) = self.query_resources(
                &kernel.buffer_ids,
                &kernel.output_indices,
                &kernel.device,
                kernel.id,
                timeline,
                true, // skip_same_device for sequential
            );

            // Wait for dependencies
            self.wait_for_tokens(&query.wait_on)?;

            // Execute kernel - borrow directly, no cloning!
            unsafe {
                kernel.kernel.program.execute(&kernel.buffer_ptrs, &kernel.fixedvars, None, None).map_err(|e| {
                    crate::error::Error::Execution { reason: format!("Kernel {} failed: {}", kernel.id, e) }
                })?;
            }

            // Debug buffer contents after kernel execution
            if tracing::enabled!(tracing::Level::DEBUG) {
                for (i, (&ptr, &buf_id)) in kernel.buffer_ptrs.iter().zip(kernel.buffer_ids.iter()).enumerate() {
                    let buf = &buffers[kernel.buffer_indices[i]];
                    let elem_size = buf.dtype().bytes();
                    let num_elems = buf.size() / elem_size;
                    let show_elems = 5.min(num_elems);
                    if buf.dtype().is_float() {
                        let slice = unsafe { std::slice::from_raw_parts(ptr as *const f32, show_elems) };
                        debug!(
                            kernel.id = kernel.id,
                            buffer.index = i,
                            buffer.id = ?buf_id,
                            buffer.dtype = "f32",
                            buffer.values = ?slice,
                            "Buffer contents after kernel"
                        );
                    } else {
                        let slice = unsafe { std::slice::from_raw_parts(ptr as *const i32, show_elems) };
                        debug!(
                            kernel.id = kernel.id,
                            buffer.index = i,
                            buffer.id = ?buf_id,
                            buffer.dtype = "i32",
                            buffer.values = ?slice,
                            "Buffer contents after kernel"
                        );
                    }
                }
            }

            // Signal completion
            if let Some(ctx) = self.contexts.get(&kernel.device) {
                ctx.signal_completion(timeline);
            }

            // Update dependency maps
            self.update_resources(update);

            return Ok(vec![(kernel.id, timeline)]);
        }

        // PHASE 0: Ensure all device contexts exist
        for kernel in &kernels {
            if !self.contexts.contains_key(&kernel.device) {
                self.add_device(kernel.device.clone())?;
            }
        }

        // PHASE 1: Validate no buffer conflicts
        self.validate_kernel_independence(&kernels, buffers)?;

        // PHASE 2: Query dependencies for all kernels
        let mut queries_updates: Vec<(DependencyQuery, DependencyUpdate)> = Vec::with_capacity(kernels.len());
        let mut timelines: Vec<(u64, DeviceSpec, u64)> = Vec::with_capacity(kernels.len());

        for kernel in &kernels {
            let timeline = self.contexts.get(&kernel.device).unwrap().next_timeline();
            timelines.push((kernel.id, kernel.device.clone(), timeline));

            let (query, update) = self.query_resources(
                &kernel.buffer_ids,
                &kernel.output_indices,
                &kernel.device,
                kernel.id,
                timeline,
                false, // parallel execution needs ALL deps
            );
            queries_updates.push((query, update));
        }

        // PHASE 3: Wait for all dependencies
        for (query, _) in &queries_updates {
            self.wait_for_tokens(&query.wait_on)?;
        }

        // PHASE 4: Execute in parallel using rayon::scope
        //
        // Problem: *mut u8 is not Sync, so &[*mut u8] is not Send.
        // Solution: Transmute &[*mut u8] to &[usize] which IS Send.
        //
        // SAFETY: *mut u8 and usize are guaranteed same size and alignment.
        // The pointers remain valid for the duration of ExecutionPlan's lifetime.
        debug_assert_eq!(std::mem::size_of::<*mut u8>(), std::mem::size_of::<usize>());
        debug_assert_eq!(std::mem::align_of::<*mut u8>(), std::mem::align_of::<usize>());

        let errors: Mutex<Vec<crate::error::Error>> = Mutex::new(Vec::new());

        // Pre-transmute pointer slices to usize slices (zero-copy, zero-alloc)
        #[allow(clippy::type_complexity)]
        let ptr_slices: Vec<(
            &[usize],
            u64,
            &std::sync::Arc<crate::kernel_cache::CachedKernel>,
            &std::collections::HashMap<String, i64>,
        )> = kernels
            .iter()
            .map(|k| {
                let usize_slice: &[usize] =
                    unsafe { std::mem::transmute::<&[*mut u8], &[usize]>(k.buffer_ptrs.as_slice()) };
                (usize_slice, k.id, &k.kernel, &k.fixedvars)
            })
            .collect();

        rayon::scope(|s| {
            for &(usize_slice, id, program, fixedvars) in &ptr_slices {
                let errors_ref = &errors;
                s.spawn(move |_| {
                    // Transmute back to pointer slice (zero-copy)
                    let ptrs: &[*mut u8] = unsafe { std::mem::transmute::<&[usize], &[*mut u8]>(usize_slice) };

                    let result = unsafe { program.program.execute(ptrs, fixedvars, None, None) };

                    if let Err(e) = result {
                        errors_ref
                            .lock()
                            .unwrap()
                            .push(crate::error::Error::Execution { reason: format!("Kernel {} failed: {}", id, e) });
                    }
                });
            }
        });

        // Check for errors
        let errs = errors.into_inner().unwrap();
        if let Some(e) = errs.into_iter().next() {
            return Err(e);
        }

        // PHASE 5: Signal completions and update state
        for (_, device, timeline) in &timelines {
            if let Some(dev_ctx) = self.contexts.get(device) {
                dev_ctx.signal_completion(*timeline);
            }
        }

        for (_, update) in queries_updates {
            self.update_resources(update);
        }

        Ok(timelines.iter().map(|(id, _, t)| (*id, *t)).collect())
    }

    /// Validate that kernels have no buffer conflicts for parallel execution.
    fn validate_kernel_independence(&self, kernels: &[&PreparedKernel], buffers: &[Buffer]) -> Result<()> {
        let mut all_outputs: HashSet<BufferId> = HashSet::new();
        let mut all_inputs: HashSet<BufferId> = HashSet::new();

        for kernel in kernels {
            let outputs: HashSet<_> =
                kernel.output_indices.iter().map(|&i| buffers[kernel.buffer_indices[i]].id()).collect();

            let inputs: HashSet<_> = kernel
                .buffer_indices
                .iter()
                .enumerate()
                .filter(|(i, _)| !kernel.output_indices.contains(i))
                .map(|(_, &buf_idx)| buffers[buf_idx].id())
                .collect();

            // Check for write-write conflict (WAW)
            if !all_outputs.is_disjoint(&outputs) {
                return Err(crate::error::Error::Execution {
                    reason: "Write conflict: multiple kernels write same buffer".into(),
                });
            }

            // Check for read-write conflict (RAW/WAR)
            if !outputs.is_disjoint(&all_inputs) || !inputs.is_disjoint(&all_outputs) {
                return Err(crate::error::Error::Execution { reason: "Read-write conflict in parallel group".into() });
            }

            all_outputs.extend(outputs);
            all_inputs.extend(inputs);
        }

        Ok(())
    }
}

/// Global executor instance.
///
/// For most use cases, a single global executor is sufficient.
/// Thread-safety is handled by timeline signals and dependency tracking.
static EXECUTOR: once_cell::sync::Lazy<parking_lot::Mutex<UnifiedExecutor>> =
    once_cell::sync::Lazy::new(|| parking_lot::Mutex::new(UnifiedExecutor::new(morok_device::registry::registry())));

/// Get access to the global executor.
pub fn global_executor() -> parking_lot::MutexGuard<'static, UnifiedExecutor> {
    EXECUTOR.lock()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sync_strategy() {
        assert_eq!(UnifiedExecutor::sync_strategy(&DeviceSpec::Cpu, &DeviceSpec::Cpu), SyncStrategy::None);

        assert_eq!(
            UnifiedExecutor::sync_strategy(&DeviceSpec::Cuda { device_id: 0 }, &DeviceSpec::Cuda { device_id: 1 }),
            SyncStrategy::PeerToPeer
        );

        assert_eq!(
            UnifiedExecutor::sync_strategy(&DeviceSpec::Cpu, &DeviceSpec::Cuda { device_id: 0 }),
            SyncStrategy::CpuMediated
        );
    }

    #[test]
    fn test_executor_creation() {
        let registry = morok_device::registry::registry();
        let executor = UnifiedExecutor::new(registry);
        assert!(executor.contexts.is_empty());
    }

    #[test]
    fn test_execution_graph_empty() {
        let mut graph = ExecutionGraph::new();
        let groups = graph.compute_parallel_groups();
        assert!(groups.is_empty());
        assert!(graph.is_valid());
    }

    #[test]
    fn test_execution_graph_single_node() {
        let mut graph = ExecutionGraph::new();
        graph.add_node(ExecutionNode {
            id: 1,
            device: DeviceSpec::Cpu,
            inputs: vec![],
            outputs: vec![BufferId(100)],
            predecessors: vec![],
            is_transfer: false,
            buffer_access: None,
        });

        let groups = graph.compute_parallel_groups();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0], vec![1]);
        assert!(graph.is_valid());
    }

    #[test]
    fn test_execution_graph_linear_chain() {
        let mut graph = ExecutionGraph::new();

        // A → B → C (linear dependency)
        graph.add_node(ExecutionNode {
            id: 1,
            device: DeviceSpec::Cpu,
            inputs: vec![],
            outputs: vec![BufferId(100)],
            predecessors: vec![],
            is_transfer: false,
            buffer_access: None,
        });
        graph.add_node(ExecutionNode {
            id: 2,
            device: DeviceSpec::Cpu,
            inputs: vec![BufferId(100)],
            outputs: vec![BufferId(101)],
            predecessors: vec![1],
            is_transfer: false,
            buffer_access: None,
        });
        graph.add_node(ExecutionNode {
            id: 3,
            device: DeviceSpec::Cpu,
            inputs: vec![BufferId(101)],
            outputs: vec![BufferId(102)],
            predecessors: vec![2],
            is_transfer: false,
            buffer_access: None,
        });

        let groups = graph.compute_parallel_groups();
        assert_eq!(groups.len(), 3); // Each node in its own group (no parallelism)
        assert!(graph.is_valid());
    }

    #[test]
    fn test_execution_graph_parallel_nodes() {
        let mut graph = ExecutionGraph::new();

        // A and B are independent, both feed into C
        //   A ──┐
        //       └──→ C
        //   B ──┘
        graph.add_node(ExecutionNode {
            id: 1,
            device: DeviceSpec::Cpu,
            inputs: vec![],
            outputs: vec![BufferId(100)],
            predecessors: vec![],
            is_transfer: false,
            buffer_access: None,
        });
        graph.add_node(ExecutionNode {
            id: 2,
            device: DeviceSpec::Cpu,
            inputs: vec![],
            outputs: vec![BufferId(101)],
            predecessors: vec![],
            is_transfer: false,
            buffer_access: None,
        });
        graph.add_node(ExecutionNode {
            id: 3,
            device: DeviceSpec::Cpu,
            inputs: vec![BufferId(100), BufferId(101)],
            outputs: vec![BufferId(102)],
            predecessors: vec![1, 2],
            is_transfer: false,
            buffer_access: None,
        });

        let groups = graph.compute_parallel_groups();
        assert_eq!(groups.len(), 2); // First group has A,B; second has C
        assert!(groups[0].contains(&1));
        assert!(groups[0].contains(&2));
        assert_eq!(groups[1], vec![3]);
        assert!(graph.is_valid());
    }

    #[test]
    fn test_find_independent_kernels() {
        let mut graph = ExecutionGraph::new();

        // Two independent kernels
        graph.add_node(ExecutionNode {
            id: 1,
            device: DeviceSpec::Cpu,
            inputs: vec![BufferId(100)],
            outputs: vec![BufferId(200)],
            predecessors: vec![],
            is_transfer: false,
            buffer_access: None,
        });
        graph.add_node(ExecutionNode {
            id: 2,
            device: DeviceSpec::Cpu,
            inputs: vec![BufferId(101)],
            outputs: vec![BufferId(201)],
            predecessors: vec![],
            is_transfer: false,
            buffer_access: None,
        });

        let independent = graph.find_independent_kernels(&[1, 2]);
        // Both should be in the same group since they have no conflicts
        assert_eq!(independent.len(), 1);
        assert!(independent[0].contains(&1));
        assert!(independent[0].contains(&2));
    }

    #[test]
    fn test_find_independent_kernels_with_conflict() {
        let mut graph = ExecutionGraph::new();

        // Two kernels writing to same buffer = conflict
        graph.add_node(ExecutionNode {
            id: 1,
            device: DeviceSpec::Cpu,
            inputs: vec![BufferId(100)],
            outputs: vec![BufferId(200)], // Same output
            predecessors: vec![],
            is_transfer: false,
            buffer_access: None,
        });
        graph.add_node(ExecutionNode {
            id: 2,
            device: DeviceSpec::Cpu,
            inputs: vec![BufferId(101)],
            outputs: vec![BufferId(200)], // Same output
            predecessors: vec![],
            is_transfer: false,
            buffer_access: None,
        });

        let independent = graph.find_independent_kernels(&[1, 2]);
        // Should be in different groups due to output conflict
        assert_eq!(independent.len(), 2);
    }

    #[test]
    fn test_find_independent_kernels_read_write_conflict() {
        let mut graph = ExecutionGraph::new();

        // One kernel writes to buffer that another reads = conflict
        graph.add_node(ExecutionNode {
            id: 1,
            device: DeviceSpec::Cpu,
            inputs: vec![BufferId(100)],
            outputs: vec![BufferId(200)],
            predecessors: vec![],
            is_transfer: false,
            buffer_access: None,
        });
        graph.add_node(ExecutionNode {
            id: 2,
            device: DeviceSpec::Cpu,
            inputs: vec![BufferId(200)], // Reads output of kernel 1
            outputs: vec![BufferId(201)],
            predecessors: vec![],
            is_transfer: false,
            buffer_access: None,
        });

        let independent = graph.find_independent_kernels(&[1, 2]);
        // Should be in different groups due to read-write conflict
        assert_eq!(independent.len(), 2);
    }
}
