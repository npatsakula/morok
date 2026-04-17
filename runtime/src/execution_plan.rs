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

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use morok_device::{Buffer, BufferId};
use morok_dtype::DeviceSpec;
use morok_ir::UOp;

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
}

// ============================================================================
// ExecutionPlan Implementation
// ============================================================================

impl ExecutionPlan {
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
        for kernel in &self.kernels {
            unsafe {
                kernel
                    .kernel
                    .program
                    .execute(&kernel.buffer_ptrs, &kernel.vals, kernel.kernel.global_size, kernel.kernel.local_size)
                    .map_err(|e| crate::error::Error::Execution {
                        reason: format!("Kernel {} failed: {}", kernel.id, e),
                    })?;
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
        for kernel in &self.kernels {
            let start = Instant::now();
            unsafe {
                kernel
                    .kernel
                    .program
                    .execute(&kernel.buffer_ptrs, &kernel.vals, kernel.kernel.global_size, kernel.kernel.local_size)
                    .map_err(|e| crate::error::Error::Execution {
                        reason: format!("Kernel {} failed: {}", kernel.id, e),
                    })?;
            }
            profiles.push(KernelProfile {
                kernel: Arc::clone(&kernel.kernel),
                device: kernel.device.clone(),
                num_buffers: kernel.buffer_ptrs.len(),
                elapsed: start.elapsed(),
            });
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

impl std::fmt::Debug for ExecutionPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExecutionPlan")
            .field("kernels", &self.kernels.len())
            .field("buffers", &self.buffers.len())
            .field("device", &self.device)
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

        ExecutionPlan {
            kernels: self.kernels,
            buffers: self.buffers,
            ast_to_buffer: self.ast_to_buffer,
            output_buffer_indices: self.output_buffer_indices,
            device: self.device,
            alias_ids: self.alias_ids,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_basic() {
        let builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
        let plan = builder.build();

        assert!(plan.kernels.is_empty());
        assert!(plan.buffers.is_empty());
        assert_eq!(plan.device, DeviceSpec::Cpu);
    }
}
