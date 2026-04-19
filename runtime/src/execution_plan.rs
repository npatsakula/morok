//! Pre-compiled execution plan for kernel execution.
//!
//! `ExecutionPlan` separates one-time preparation (kernel compilation, buffer
//! allocation) from fast repeated execution.
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │              PREPARATION (one-time)                      │
//! │  Schedule → expand → compile_kernels → build()          │
//! │                       ↓                                  │
//! │                ExecutionPlan                             │
//! └─────────────────────────────────────────────────────────┘
//!                         ↓
//! ┌─────────────────────────────────────────────────────────┐
//! │              EXECUTION (fast path)                       │
//! │  for kernel in kernels:                                  │
//! │      program.execute(ptrs, vals, global, local)          │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```ignore
//! let plan = tensor.prepare()?;
//! plan.execute()?;
//! let output = plan.output_buffer();
//! ```

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use morok_device::{Buffer, BufferId};
use morok_dtype::DeviceSpec;
use morok_ir::UOp;
use rayon::prelude::*;

use crate::error::Result;
use crate::kernel_cache::CachedKernel;
use crate::profiler::KernelProfile;

// ============================================================================
// Core Structures
// ============================================================================

/// A pre-compiled kernel ready for execution.
///
/// Variable values are stored as positional `vals: Vec<i64>` rather than a named
/// HashMap, matching Tinygrad's `vals: tuple[int, ...]` parameter style.
#[derive(Clone)]
pub struct PreparedKernel {
    /// Unique identifier (from original AST).
    pub id: u64,

    pub ast: Arc<UOp>,

    /// Compiled kernel program (Arc-shared from cache).
    pub kernel: Arc<CachedKernel>,

    /// Device this kernel executes on.
    pub device: DeviceSpec,

    /// Indices into `ExecutionPlan::buffers` for this kernel's buffers.
    /// Ordered as expected by the kernel (matches codegen buffer order).
    pub buffer_indices: Vec<usize>,

    /// Indices of output buffers within `buffer_indices`.
    pub output_indices: Vec<usize>,

    /// Variable values in positional order (matches `var_names` in CachedKernel).
    pub vals: Vec<i64>,

    /// Kernel IDs that must complete before this one (dependencies).
    pub dependencies: Vec<u64>,

    /// Pre-computed raw buffer pointers for zero-allocation execution.
    /// Computed once during prepare(), stable for the lifetime of ExecutionPlan.
    /// SAFETY: Pointers are valid as long as ExecutionPlan owns the buffers.
    pub buffer_ptrs: Vec<*mut u8>,

    /// Pre-computed buffer IDs for dependency tracking.
    pub buffer_ids: Vec<BufferId>,
}

// SAFETY: PreparedKernel is immutable during plan execution.
// - `buffer_ptrs` are precomputed and never mutated after build().
// - `vals` are read-only during `execute`/`execute_profiled`.
// - Safety of concurrent execution is enforced by execution-level hazard filtering
//   (RAW/WAW/WAR + alias/view overlap checks) before kernels are parallelized.
unsafe impl Send for PreparedKernel {}
unsafe impl Sync for PreparedKernel {}

/// Pre-compiled execution plan for a computation graph.
///
/// Created once via `prepare()`, then executed multiple times.
/// The plan owns all its buffers and compiled kernels.
pub struct ExecutionPlan {
    /// Pre-compiled kernels in topological order.
    kernels: Vec<PreparedKernel>,

    /// ALL buffers owned by this plan (inputs, intermediates, outputs).
    buffers: Vec<Buffer>,

    /// Mapping: AST id → buffer index (for kernel buffer binding).
    ast_to_buffer: HashMap<u64, usize>,

    /// Indices of output buffers in `buffers` (matches SINK source order).
    output_buffer_indices: Vec<usize>,

    /// Primary device for this plan.
    device: DeviceSpec,

    /// Additional UOp IDs registered as aliases that need cleanup.
    alias_ids: Vec<u64>,

    /// Topological execution levels after hazard filtering.
    /// Kernels within each level are safe to run in parallel.
    execution_levels: Vec<ExecutionLevel>,
}

#[derive(Debug, Clone)]
struct ExecutionLevel {
    kernel_indices: Vec<usize>,
    contains_thread_id_kernel: bool,
}

#[derive(Debug, Clone)]
struct KernelAccess {
    reads: HashSet<BufferId>,
    writes: HashSet<BufferId>,
    has_thread_id_parallelism: bool,
}

// ============================================================================
// ExecutionPlan Implementation
// ============================================================================

impl ExecutionPlan {
    #[inline]
    fn execute_kernel(kernel: &PreparedKernel) -> Result<()> {
        unsafe {
            kernel
                .kernel
                .program
                .execute(&kernel.buffer_ptrs, &kernel.vals, kernel.kernel.global_size, kernel.kernel.local_size)
                .map_err(|e| crate::error::Error::Execution { reason: format!("Kernel {} failed: {}", kernel.id, e) })
        }
    }

    #[inline]
    fn can_parallelize_level(level: &ExecutionLevel) -> bool {
        level.kernel_indices.len() > 1
            && !level.contains_thread_id_kernel
            // Avoid nested rayon scheduling from callers already inside rayon pools.
            && rayon::current_thread_index().is_none()
    }

    /// Get the first (or only) output buffer after execution.
    pub fn output_buffer(&self) -> &Buffer {
        &self.buffers[self.output_buffer_indices[0]]
    }

    /// Get output buffer by position (matches SINK source order for batch).
    pub fn output_buffer_at(&self, position: usize) -> &Buffer {
        &self.buffers[self.output_buffer_indices[position]]
    }

    /// Get all output buffers.
    pub fn output_buffers(&self) -> Vec<&Buffer> {
        self.output_buffer_indices.iter().map(|&i| &self.buffers[i]).collect()
    }

    /// Number of outputs in this plan.
    pub fn num_outputs(&self) -> usize {
        self.output_buffer_indices.len()
    }

    /// Get a buffer by AST id (for reading intermediate results).
    pub fn buffer(&self, ast_id: u64) -> Option<&Buffer> {
        self.ast_to_buffer.get(&ast_id).map(|&idx| &self.buffers[idx])
    }

    /// Get a mutable buffer by AST id (for `copyin()` on input buffers).
    pub fn buffer_mut_by_id(&mut self, ast_id: u64) -> Option<&mut Buffer> {
        self.ast_to_buffer.get(&ast_id).copied().map(|idx| &mut self.buffers[idx])
    }

    /// Get the primary device for this plan.
    pub fn device(&self) -> &DeviceSpec {
        &self.device
    }

    /// Get all buffers owned by this plan.
    pub fn buffers(&self) -> &[Buffer] {
        &self.buffers
    }

    /// Get mutable access to all buffers owned by this plan.
    pub fn buffers_mut(&mut self) -> &mut [Buffer] {
        &mut self.buffers
    }

    /// Get a mutable buffer by its index in the buffers array.
    pub fn buffer_at_mut(&mut self, index: usize) -> Option<&mut Buffer> {
        self.buffers.get_mut(index)
    }

    /// Get all prepared kernels.
    pub fn prepared_kernels(&self) -> &[PreparedKernel] {
        &self.kernels
    }

    /// Iterate over compiled kernels (for inspecting generated source code).
    pub fn kernels(&self) -> impl Iterator<Item = &CachedKernel> {
        self.kernels.iter().map(|pk| pk.kernel.as_ref())
    }

    /// Execute the plan.
    ///
    /// This is the fast path — a tight loop over pre-compiled kernels with
    /// pre-computed buffer pointers. No allocation, no validation, no
    /// device context lookup.
    pub fn execute(&self) -> Result<()> {
        for level in &self.execution_levels {
            if Self::can_parallelize_level(level) {
                level.kernel_indices.par_iter().try_for_each(|&idx| Self::execute_kernel(&self.kernels[idx]))?;
            } else {
                for &idx in &level.kernel_indices {
                    Self::execute_kernel(&self.kernels[idx])?;
                }
            }
        }
        Ok(())
    }

    /// Execute the plan with per-kernel timing.
    ///
    /// Returns a [`KernelProfile`] for each kernel in execution order.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let plan = tensor.prepare()?;
    /// let profiles = plan.execute_profiled()?;
    ///
    /// // Sort by time descending
    /// let mut sorted = profiles;
    /// sorted.sort_by(|a, b| b.elapsed.cmp(&a.elapsed));
    /// for p in &sorted[..10.min(sorted.len())] {
    ///     println!("{:>8.3}ms  {}", p.elapsed.as_secs_f64() * 1000.0, p.kernel.entry_point);
    /// }
    /// ```
    pub fn execute_profiled(&self) -> Result<Vec<KernelProfile>> {
        let mut profiles = Vec::with_capacity(self.kernels.len());
        for level in &self.execution_levels {
            if Self::can_parallelize_level(level) {
                let level_profiles: Result<Vec<KernelProfile>> = level
                    .kernel_indices
                    .par_iter()
                    .map(|&idx| {
                        let kernel = &self.kernels[idx];
                        let start = Instant::now();
                        Self::execute_kernel(kernel)?;
                        Ok(KernelProfile {
                            kernel: Arc::clone(&kernel.kernel),
                            device: kernel.device.clone(),
                            num_buffers: kernel.buffer_ptrs.len(),
                            elapsed: start.elapsed(),
                        })
                    })
                    .collect();
                profiles.extend(level_profiles?);
            } else {
                for &idx in &level.kernel_indices {
                    let kernel = &self.kernels[idx];
                    let start = Instant::now();
                    Self::execute_kernel(kernel)?;
                    profiles.push(KernelProfile {
                        kernel: Arc::clone(&kernel.kernel),
                        device: kernel.device.clone(),
                        num_buffers: kernel.buffer_ptrs.len(),
                        elapsed: start.elapsed(),
                    });
                }
            }
        }
        Ok(profiles)
    }

    /// Re-execute the plan with different variable bindings.
    ///
    /// The kernel code is NOT recompiled; only the `vals` passed to each kernel
    /// are updated. Buffers must be allocated to max variable values (which is
    /// the default when using `Variable::bind()`).
    ///
    /// # Safety contract
    ///
    /// Variable values **must** fall within `[min_val, max_val]` bounds defined
    /// at `Variable::new()` time. Exceeding `max_val` causes out-of-bounds buffer
    /// access (buffers are allocated to `max_val`). Use `Variable::bind()` to
    /// validate bounds before calling this method.
    ///
    /// Variables not present in `var_vals` keep their existing values from
    /// `prepare()` (or the previous `execute_with_vars` call). Internal
    /// variables like `thread_id` are left untouched.
    pub fn execute_with_vars(&mut self, var_vals: &[(&str, i64)]) -> Result<()> {
        // Build a map for O(1) lookup (avoids O(V*K) linear scan per kernel)
        let vals_map: HashMap<&str, i64> = var_vals.iter().copied().collect();
        for kernel in &mut self.kernels {
            for (idx, name) in kernel.kernel.var_names.iter().enumerate() {
                if let Some(&v) = vals_map.get(name.as_str()) {
                    kernel.vals[idx] = v;
                }
            }
        }
        self.execute()
    }

    /// Re-execute the plan with different variable bindings and per-kernel timing.
    ///
    /// Updates kernel `vals` the same way as [`Self::execute_with_vars`] and then
    /// executes via [`Self::execute_profiled`].
    pub fn execute_with_vars_profiled(&mut self, var_vals: &[(&str, i64)]) -> Result<Vec<KernelProfile>> {
        // Build a map for O(1) lookup (avoids O(V*K) linear scan per kernel)
        let vals_map: HashMap<&str, i64> = var_vals.iter().copied().collect();
        for kernel in &mut self.kernels {
            for (idx, name) in kernel.kernel.var_names.iter().enumerate() {
                if let Some(&v) = vals_map.get(name.as_str()) {
                    kernel.vals[idx] = v;
                }
            }
        }
        self.execute_profiled()
    }

    /// Get the first output buffer index.
    pub fn output_buffer_idx(&self) -> usize {
        self.output_buffer_indices[0]
    }

    /// Get the AST ID to buffer index mapping.
    pub fn ast_to_buffer_map(&self) -> &HashMap<u64, usize> {
        &self.ast_to_buffer
    }

    /// Release intermediate buffers from the global buffer registry.
    ///
    /// Call this after you're done executing the plan to free intermediate
    /// buffers from the global registry. The output buffer is preserved.
    pub fn release_intermediate_buffers<F>(&self, remove_fn: F)
    where
        F: Fn(u64),
    {
        self.release_buffers_impl(remove_fn, true);
    }

    /// Release ALL buffers from the global registry, including the output.
    pub fn release_all_buffers<F>(&self, remove_fn: F)
    where
        F: Fn(u64),
    {
        self.release_buffers_impl(remove_fn, false);
    }

    fn release_buffers_impl<F>(&self, remove_fn: F, skip_output: bool)
    where
        F: Fn(u64),
    {
        let output_buf_ids: std::collections::HashSet<u64> = if skip_output {
            self.output_buffer_indices.iter().filter_map(|&idx| self.buffers.get(idx).map(|b| b.id().0)).collect()
        } else {
            std::collections::HashSet::new()
        };

        for (&ast_id, &buf_idx) in &self.ast_to_buffer {
            if skip_output && output_buf_ids.contains(&self.buffers[buf_idx].id().0) {
                continue;
            }
            remove_fn(ast_id);
        }

        for &alias_id in &self.alias_ids {
            remove_fn(alias_id);
        }
    }
}

fn kernel_access(kernel: &PreparedKernel) -> KernelAccess {
    let mut writes = HashSet::new();
    let mut reads = HashSet::new();

    let mut output_mask = vec![false; kernel.buffer_ids.len()];
    for &out_idx in &kernel.output_indices {
        if out_idx < output_mask.len() {
            output_mask[out_idx] = true;
        }
    }

    if output_mask.iter().all(|&is_output| !is_output) {
        // Conservatively treat unknown outputs as writes to all buffers.
        writes.extend(kernel.buffer_ids.iter().copied());
    } else {
        for (idx, buffer_id) in kernel.buffer_ids.iter().copied().enumerate() {
            if output_mask[idx] {
                writes.insert(buffer_id);
            } else {
                reads.insert(buffer_id);
            }
        }
    }

    let has_thread_id_parallelism = kernel.kernel.var_names.iter().any(|name| name == "thread_id")
        && kernel.kernel.global_size.map(|[tc, _, _]| tc > 1).unwrap_or(false);

    KernelAccess { reads, writes, has_thread_id_parallelism }
}

fn accesses_conflict(lhs: &KernelAccess, rhs: &KernelAccess) -> bool {
    // WAW, RAW, WAR hazards.
    !lhs.writes.is_disjoint(&rhs.writes) || !lhs.writes.is_disjoint(&rhs.reads) || !lhs.reads.is_disjoint(&rhs.writes)
}

fn partition_level_by_hazards(level_indices: &[usize], accesses: &[KernelAccess]) -> Vec<ExecutionLevel> {
    let mut groups: Vec<Vec<usize>> = Vec::new();

    for &idx in level_indices {
        let mut placed = false;
        for group in &mut groups {
            if group.iter().all(|&other| !accesses_conflict(&accesses[idx], &accesses[other])) {
                group.push(idx);
                placed = true;
                break;
            }
        }
        if !placed {
            groups.push(vec![idx]);
        }
    }

    groups
        .into_iter()
        .map(|kernel_indices| ExecutionLevel {
            contains_thread_id_kernel: kernel_indices.iter().any(|&idx| accesses[idx].has_thread_id_parallelism),
            kernel_indices,
        })
        .collect()
}

fn build_execution_levels(kernels: &[PreparedKernel]) -> Vec<ExecutionLevel> {
    if kernels.is_empty() {
        return Vec::new();
    }

    let id_to_idx: HashMap<u64, usize> = kernels.iter().enumerate().map(|(idx, kernel)| (kernel.id, idx)).collect();

    let mut in_degree = vec![0usize; kernels.len()];
    let mut successors: Vec<Vec<usize>> = vec![Vec::new(); kernels.len()];

    for (idx, kernel) in kernels.iter().enumerate() {
        for dep_id in &kernel.dependencies {
            if let Some(&dep_idx) = id_to_idx.get(dep_id) {
                in_degree[idx] += 1;
                successors[dep_idx].push(idx);
            }
        }
    }

    let accesses: Vec<KernelAccess> = kernels.iter().map(kernel_access).collect();
    let mut levels = Vec::new();
    let mut ready: Vec<usize> =
        in_degree.iter().enumerate().filter_map(|(idx, &deg)| if deg == 0 { Some(idx) } else { None }).collect();
    let mut visited = 0usize;

    while !ready.is_empty() {
        ready.sort_unstable();
        visited += ready.len();
        levels.extend(partition_level_by_hazards(&ready, &accesses));

        let mut next_ready = Vec::new();
        for idx in ready {
            for &succ in &successors[idx] {
                in_degree[succ] -= 1;
                if in_degree[succ] == 0 {
                    next_ready.push(succ);
                }
            }
        }
        ready = next_ready;
    }

    if visited != kernels.len() {
        return kernels
            .iter()
            .enumerate()
            .map(|(idx, kernel)| ExecutionLevel {
                kernel_indices: vec![idx],
                contains_thread_id_kernel: kernel.kernel.var_names.iter().any(|name| name == "thread_id"),
            })
            .collect();
    }

    levels
}

impl std::fmt::Debug for ExecutionPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExecutionPlan")
            .field("kernels", &self.kernels.len())
            .field("buffers", &self.buffers.len())
            .field("device", &self.device)
            .field("execution_levels", &self.execution_levels.len())
            .finish()
    }
}

impl std::fmt::Debug for PreparedKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedKernel")
            .field("id", &self.id)
            .field("device", &self.device)
            .field("buffer_indices", &self.buffer_indices)
            .field("output_indices", &self.output_indices)
            .field("vals", &self.vals)
            .field("dependencies", &self.dependencies)
            .finish()
    }
}

// ============================================================================
// Builder for ExecutionPlan
// ============================================================================

/// Builder for creating ExecutionPlan from schedule data.
pub struct ExecutionPlanBuilder {
    kernels: Vec<PreparedKernel>,
    buffers: Vec<Buffer>,
    ast_to_buffer: HashMap<u64, usize>,
    output_buffer_indices: Vec<usize>,
    device: DeviceSpec,
    alias_ids: Vec<u64>,
}

impl ExecutionPlanBuilder {
    /// Create a new builder.
    pub fn new(device: DeviceSpec) -> Self {
        Self {
            kernels: Vec::new(),
            buffers: Vec::new(),
            ast_to_buffer: HashMap::new(),
            output_buffer_indices: Vec::new(),
            device,
            alias_ids: Vec::new(),
        }
    }

    /// Add alias IDs that need cleanup.
    pub fn add_alias_ids(&mut self, ids: impl IntoIterator<Item = u64>) {
        self.alias_ids.extend(ids);
    }

    /// Add a buffer to the plan. Returns the buffer index.
    pub fn add_buffer(&mut self, ast_id: u64, buffer: Buffer) -> usize {
        let idx = self.buffers.len();
        self.buffers.push(buffer);
        self.ast_to_buffer.insert(ast_id, idx);
        idx
    }

    /// Map an additional AST/buffer UOp ID to an existing buffer index.
    pub fn map_buffer(&mut self, ast_id: u64, idx: usize) {
        self.ast_to_buffer.insert(ast_id, idx);
    }

    /// Replace a buffer at the given index (for BUFFER_VIEW sub-buffer views).
    pub fn replace_buffer(&mut self, idx: usize, buffer: Buffer) {
        self.buffers[idx] = buffer;
    }

    /// Set single output buffer index.
    pub fn set_output_buffer(&mut self, idx: usize) {
        self.output_buffer_indices = vec![idx];
    }

    /// Set multiple output buffer indices (batch scheduling).
    pub fn set_output_buffers(&mut self, indices: Vec<usize>) {
        self.output_buffer_indices = indices;
    }

    /// Add a prepared kernel.
    pub fn add_kernel(&mut self, kernel: PreparedKernel) {
        self.kernels.push(kernel);
    }

    /// Build the ExecutionPlan.
    ///
    /// Finalizes by computing pre-allocated buffer pointers and buffer IDs
    /// for zero-allocation execution.
    pub fn build(mut self) -> ExecutionPlan {
        for kernel in &mut self.kernels {
            kernel.buffer_ptrs =
                kernel.buffer_indices.iter().map(|&idx| unsafe { self.buffers[idx].as_raw_ptr() }).collect();

            kernel.buffer_ids = kernel.buffer_indices.iter().map(|&idx| self.buffers[idx].id()).collect();
        }

        if self.output_buffer_indices.is_empty() && !self.buffers.is_empty() {
            self.output_buffer_indices = vec![0];
        }

        let execution_levels = build_execution_levels(&self.kernels);

        ExecutionPlan {
            kernels: self.kernels,
            buffers: self.buffers,
            ast_to_buffer: self.ast_to_buffer,
            output_buffer_indices: self.output_buffer_indices,
            device: self.device,
            alias_ids: self.alias_ids,
            execution_levels,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use morok_device::device::Program;

    #[derive(Debug)]
    struct NoopProgram;

    impl Program for NoopProgram {
        unsafe fn execute(
            &self,
            _buffers: &[*mut u8],
            _vals: &[i64],
            _global_size: Option<[usize; 3]>,
            _local_size: Option<[usize; 3]>,
        ) -> morok_device::Result<()> {
            Ok(())
        }

        fn name(&self) -> &str {
            "noop"
        }
    }

    fn make_kernel(
        id: u64,
        dependencies: Vec<u64>,
        buffer_ids: Vec<u64>,
        output_indices: Vec<usize>,
        threaded: bool,
    ) -> PreparedKernel {
        let (var_names, vals, global_size) = if threaded {
            (vec!["thread_id".to_string()], vec![0], Some([4, 1, 1]))
        } else {
            (Vec::new(), Vec::new(), None)
        };

        let cached = Arc::new(CachedKernel {
            program: Box::new(NoopProgram),
            device: "CPU".to_string(),
            code: String::new(),
            entry_point: format!("k{id}"),
            var_names,
            globals: (0..buffer_ids.len()).collect(),
            outs: output_indices.clone(),
            ins: Vec::new(),
            global_size,
            local_size: None,
        });

        PreparedKernel {
            id,
            ast: UOp::sink(vec![]),
            kernel: cached,
            device: DeviceSpec::Cpu,
            buffer_indices: (0..buffer_ids.len()).collect(),
            output_indices,
            vals,
            dependencies,
            buffer_ptrs: Vec::new(),
            buffer_ids: buffer_ids.into_iter().map(BufferId).collect(),
        }
    }

    #[test]
    fn test_builder_basic() {
        let builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
        let plan = builder.build();

        assert!(plan.kernels.is_empty());
        assert!(plan.buffers.is_empty());
        assert_eq!(plan.device, DeviceSpec::Cpu);
        assert!(plan.execution_levels.is_empty());
    }

    #[test]
    fn test_builder_map_buffer_alias() {
        let alloc = morok_device::registry::cpu().expect("cpu allocator");
        let buf = Buffer::new(alloc, morok_dtype::DType::Float32, vec![8], Default::default());

        let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
        let idx = builder.add_buffer(10, buf);
        builder.map_buffer(11, idx);
        let plan = builder.build();

        assert_eq!(plan.ast_to_buffer_map().get(&10), Some(&idx));
        assert_eq!(plan.ast_to_buffer_map().get(&11), Some(&idx));
        assert_eq!(plan.buffers().len(), 1);
    }

    #[test]
    fn test_execution_levels_parallel_dependency_layers() {
        let kernels = vec![
            make_kernel(1, vec![], vec![10], vec![0], false),
            make_kernel(2, vec![], vec![20], vec![0], false),
            make_kernel(3, vec![1, 2], vec![30, 10, 20], vec![0], false),
        ];

        let levels = build_execution_levels(&kernels);
        assert_eq!(levels.len(), 2);
        assert_eq!(levels[0].kernel_indices, vec![0, 1]);
        assert_eq!(levels[1].kernel_indices, vec![2]);
        assert!(!levels[0].contains_thread_id_kernel);
    }

    #[test]
    fn test_execution_levels_split_raw_hazard() {
        let kernels = vec![
            make_kernel(1, vec![], vec![7], vec![0], false),
            // Writes buffer 8, reads buffer 7 -> RAW hazard against kernel 1.
            make_kernel(2, vec![], vec![8, 7], vec![0], false),
        ];

        let levels = build_execution_levels(&kernels);
        assert_eq!(levels.len(), 2);
        assert_eq!(levels[0].kernel_indices, vec![0]);
        assert_eq!(levels[1].kernel_indices, vec![1]);
    }

    #[test]
    fn test_execution_levels_split_alias_overlap_waw() {
        let kernels =
            vec![make_kernel(1, vec![], vec![42], vec![0], false), make_kernel(2, vec![], vec![42], vec![0], false)];

        let levels = build_execution_levels(&kernels);
        assert_eq!(levels.len(), 2);
        assert_eq!(levels[0].kernel_indices, vec![0]);
        assert_eq!(levels[1].kernel_indices, vec![1]);
    }

    #[test]
    fn test_execution_levels_thread_id_disables_outer_parallel() {
        let kernels =
            vec![make_kernel(1, vec![], vec![10], vec![0], true), make_kernel(2, vec![], vec![20], vec![0], false)];

        let levels = build_execution_levels(&kernels);
        assert_eq!(levels.len(), 1);
        assert!(levels[0].contains_thread_id_kernel);
        assert!(!ExecutionPlan::can_parallelize_level(&levels[0]));
    }
}
