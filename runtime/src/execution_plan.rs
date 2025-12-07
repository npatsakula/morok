//! Pre-compiled execution plan for parallel kernel execution.
//!
//! This module provides `ExecutionPlan` - a structure that separates
//! one-time preparation (kernel compilation, buffer allocation, parallel
//! group computation) from fast repeated execution.
//!
//! # Design
//!
//! Like Python's code objects or PyTorch's traced graphs, `ExecutionPlan`
//! captures all the work needed to execute a computation graph:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    PREPARATION (one-time)                        │
//! │  Schedule → expand → compile_kernels → compute_parallel_groups  │
//! │                           ↓                                      │
//! │                    ExecutionPlan                                 │
//! └─────────────────────────────────────────────────────────────────┘
//!                               ↓
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    EXECUTION (fast path)                         │
//! │  for group in parallel_groups:                                   │
//! │      if single_kernel → execute_kernel()                         │
//! │      else → execute_parallel_group()                             │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```ignore
//! // One-time preparation (compiles kernels, allocates buffers)
//! let plan = ExecutionPlan::prepare(&schedule)?;
//!
//! // Fast execution (can be called many times)
//! plan.execute(&mut executor)?;
//!
//! // Get results
//! let output = plan.output_buffer();
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use morok_device::Buffer;
use morok_dtype::DeviceSpec;

use crate::error::Result;
use crate::executor::{ParallelKernelContext, UnifiedExecutor};
use crate::kernel_cache::CachedKernel;

// ============================================================================
// Core Structures
// ============================================================================

/// A pre-compiled kernel ready for execution.
///
/// This is the "lowered" form of a `ScheduleItem` - all compilation is complete,
/// only buffer binding and execution remain.
#[derive(Clone)]
pub struct PreparedKernel {
    /// Unique identifier (from original AST).
    pub id: u64,

    /// Compiled kernel program (Arc-shared from cache).
    pub kernel: Arc<CachedKernel>,

    /// Device this kernel executes on.
    pub device: DeviceSpec,

    /// Indices into `ExecutionPlan::buffers` for this kernel's buffers.
    /// Ordered as expected by the kernel (matches codegen buffer order).
    pub buffer_indices: Vec<usize>,

    /// Indices of output buffers within `buffer_indices`.
    /// Used for dependency tracking.
    pub output_indices: Vec<usize>,

    /// Fixed variable values for this kernel invocation.
    /// Pre-expanded from bound_ranges during preparation.
    pub fixedvars: HashMap<String, i64>,

    /// Kernel IDs that must complete before this one (dependencies).
    pub dependencies: Vec<u64>,
}

/// A group of kernels that can execute in parallel.
///
/// Within a group, kernels have no buffer conflicts and can be
/// dispatched via `execute_parallel_group`.
#[derive(Clone, Debug)]
pub struct ParallelGroup {
    /// Indices into `ExecutionPlan::kernels` for kernels in this group.
    pub kernel_indices: Vec<usize>,
}

/// Pre-compiled execution plan for a computation graph.
///
/// Created once via `ExecutionPlan::prepare()`, then executed multiple times
/// with the same buffers. The plan owns all its buffers.
pub struct ExecutionPlan {
    /// Pre-compiled kernels in topological order.
    kernels: Vec<PreparedKernel>,

    /// Parallel groups for execution.
    /// Each group contains kernel indices that can run in parallel.
    parallel_groups: Vec<ParallelGroup>,

    /// ALL buffers owned by this plan (inputs, intermediates, outputs).
    /// Allocated during prepare(), reused across execute() calls.
    buffers: Vec<Buffer>,

    /// Mapping: AST id → buffer index (for kernel buffer binding).
    ast_to_buffer: HashMap<u64, usize>,

    /// Index of output buffer in `buffers`.
    output_buffer_idx: usize,

    /// Primary device for this plan.
    device: DeviceSpec,
}

// ============================================================================
// Wrapper for raw pointer to enable Send
// ============================================================================

/// Wrapper to make raw pointers Send for parallel execution.
///
/// # Safety
///
/// The underlying pointer must remain valid for the duration of use.
/// This is guaranteed because `ExecutionPlan` owns all its buffers.
#[derive(Clone, Copy)]
struct SendPtr(*mut u8);

// SAFETY: The pointer comes from buffers owned by ExecutionPlan,
// which keeps them alive for the duration of execute().
unsafe impl Send for SendPtr {}

impl SendPtr {
    /// Create a new SendPtr from a raw pointer.
    ///
    /// # Safety
    ///
    /// The caller must ensure the pointer remains valid.
    unsafe fn new(ptr: *mut u8) -> Self {
        Self(ptr)
    }

    /// Get the underlying raw pointer.
    fn as_ptr(self) -> *mut u8 {
        self.0
    }
}

// ============================================================================
// ExecutionPlan Implementation
// ============================================================================

impl ExecutionPlan {
    /// Get the output buffer after execution.
    pub fn output_buffer(&self) -> &Buffer {
        &self.buffers[self.output_buffer_idx]
    }

    /// Get a buffer by AST id (for reading intermediate results).
    pub fn buffer(&self, ast_id: u64) -> Option<&Buffer> {
        self.ast_to_buffer.get(&ast_id).map(|&idx| &self.buffers[idx])
    }

    /// Get the primary device for this plan.
    pub fn device(&self) -> &DeviceSpec {
        &self.device
    }

    /// Get all buffers owned by this plan.
    pub fn buffers(&self) -> &[Buffer] {
        &self.buffers
    }

    /// Get all prepared kernels.
    pub fn prepared_kernels(&self) -> &[PreparedKernel] {
        &self.kernels
    }

    /// Iterate over compiled kernels (for inspecting generated source code).
    ///
    /// Each `CachedKernel` contains:
    /// - `code`: Generated source code (LLVM IR, etc.)
    /// - `device`: Target device string
    /// - `entry_point`: Kernel entry function name
    ///
    /// # Example
    ///
    /// ```ignore
    /// for kernel in plan.kernels() {
    ///     println!("=== {} ===", kernel.entry_point);
    ///     println!("{}", kernel.code);
    /// }
    /// ```
    pub fn kernels(&self) -> impl Iterator<Item = &CachedKernel> {
        self.kernels.iter().map(|pk| pk.kernel.as_ref())
    }

    /// Get all parallel groups.
    pub fn parallel_groups(&self) -> &[ParallelGroup] {
        &self.parallel_groups
    }

    /// Execute the plan. All buffers already allocated during prepare().
    ///
    /// This is the fast path - no compilation, no allocation, just execution:
    /// 1. Iterate through parallel groups
    /// 2. Single kernel → `executor.execute_kernel()` (avoids rayon overhead)
    /// 3. Multiple kernels → `executor.execute_parallel_group()` (parallel)
    ///
    /// # Arguments
    ///
    /// * `executor` - The UnifiedExecutor for dependency tracking
    ///
    /// # Returns
    ///
    /// Ok(()) on success, error if any kernel fails.
    pub fn execute(&self, executor: &mut UnifiedExecutor) -> Result<()> {
        for group in &self.parallel_groups {
            self.execute_group(executor, group)?;
        }
        Ok(())
    }

    /// Execute a single parallel group.
    fn execute_group(&self, executor: &mut UnifiedExecutor, group: &ParallelGroup) -> Result<()> {
        if group.kernel_indices.is_empty() {
            return Ok(());
        }

        if group.kernel_indices.len() == 1 {
            // Single kernel - sequential path (no rayon overhead)
            let kernel = &self.kernels[group.kernel_indices[0]];
            self.execute_single_kernel(executor, kernel)?;
        } else {
            // Multiple kernels - parallel execution
            self.execute_parallel_kernels(executor, group)?;
        }

        Ok(())
    }

    /// Execute a single kernel with dependency tracking.
    fn execute_single_kernel(&self, executor: &mut UnifiedExecutor, kernel: &PreparedKernel) -> Result<()> {
        // Collect buffer references
        let buffers: Vec<&Buffer> = kernel.buffer_indices.iter().map(|&idx| &self.buffers[idx]).collect();

        // Extract pointers for kernel execution
        let pointers: Vec<*mut u8> = buffers
            .iter()
            .map(|b| {
                // SAFETY: buffers are allocated and owned by ExecutionPlan
                unsafe { b.as_raw_ptr() }
            })
            .collect();

        // Clone data for the closure
        let fixedvars = kernel.fixedvars.clone();
        let program = Arc::clone(&kernel.kernel);

        executor.execute_kernel(kernel.id, &kernel.device, &buffers, &kernel.output_indices, || {
            // SAFETY: pointers are valid for the duration of this closure
            unsafe {
                program
                    .program
                    .execute(&pointers, &fixedvars, None, None)
                    .map_err(|e| crate::error::Error::Execution { reason: format!("Kernel execution failed: {}", e) })
            }
        })?;

        Ok(())
    }

    /// Execute multiple kernels in parallel.
    fn execute_parallel_kernels(&self, executor: &mut UnifiedExecutor, group: &ParallelGroup) -> Result<()> {
        // Build ParallelKernelContext for each kernel
        let contexts: Vec<ParallelKernelContext> = group
            .kernel_indices
            .iter()
            .map(|&idx| {
                let kernel = &self.kernels[idx];
                let buffers: Vec<Buffer> =
                    kernel.buffer_indices.iter().map(|&buf_idx| self.buffers[buf_idx].clone()).collect();

                ParallelKernelContext {
                    node_id: kernel.id,
                    device: kernel.device.clone(),
                    buffers,
                    output_indices: kernel.output_indices.clone(),
                }
            })
            .collect();

        // Build execution functions
        // We need to extract pointers BEFORE creating closures to avoid lifetime issues
        #[allow(clippy::type_complexity)]
        let kernel_data: Vec<(Arc<CachedKernel>, Vec<SendPtr>, HashMap<String, i64>)> = group
            .kernel_indices
            .iter()
            .map(|&idx| {
                let kernel = &self.kernels[idx];

                // Extract pointers
                let pointers: Vec<SendPtr> = kernel
                    .buffer_indices
                    .iter()
                    .map(|&buf_idx| {
                        let ptr = unsafe { self.buffers[buf_idx].as_raw_ptr() };
                        // SAFETY: buffers are owned by ExecutionPlan and live for duration of execute()
                        unsafe { SendPtr::new(ptr) }
                    })
                    .collect();

                (Arc::clone(&kernel.kernel), pointers, kernel.fixedvars.clone())
            })
            .collect();

        // Create execution closures
        let execute_fns: Vec<Box<dyn FnOnce() -> Result<()> + Send>> = kernel_data
            .into_iter()
            .map(|(program, pointers, fixedvars)| {
                Box::new(move || {
                    let raw_ptrs: Vec<*mut u8> = pointers.iter().map(|p| p.as_ptr()).collect();
                    // SAFETY: pointers are valid for the duration of this closure
                    unsafe {
                        program.program.execute(&raw_ptrs, &fixedvars, None, None).map_err(|e| {
                            crate::error::Error::Execution { reason: format!("Kernel execution failed: {}", e) }
                        })
                    }
                }) as Box<dyn FnOnce() -> Result<()> + Send>
            })
            .collect();

        // Execute in parallel
        executor.execute_parallel_group(&contexts, execute_fns)?;

        Ok(())
    }
}

impl std::fmt::Debug for ExecutionPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExecutionPlan")
            .field("kernels", &self.kernels.len())
            .field("parallel_groups", &self.parallel_groups.len())
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
            .field("dependencies", &self.dependencies)
            .finish()
    }
}

// ============================================================================
// Builder for ExecutionPlan
// ============================================================================

/// Builder for creating ExecutionPlan from schedule data.
///
/// This is used by the tensor crate to construct ExecutionPlan
/// since it has access to Schedule and related types.
pub struct ExecutionPlanBuilder {
    kernels: Vec<PreparedKernel>,
    parallel_groups: Vec<ParallelGroup>,
    buffers: Vec<Buffer>,
    ast_to_buffer: HashMap<u64, usize>,
    output_buffer_idx: usize,
    device: DeviceSpec,
}

impl ExecutionPlanBuilder {
    /// Create a new builder.
    pub fn new(device: DeviceSpec) -> Self {
        Self {
            kernels: Vec::new(),
            parallel_groups: Vec::new(),
            buffers: Vec::new(),
            ast_to_buffer: HashMap::new(),
            output_buffer_idx: 0,
            device,
        }
    }

    /// Add a buffer to the plan.
    ///
    /// Returns the buffer index.
    pub fn add_buffer(&mut self, ast_id: u64, buffer: Buffer) -> usize {
        let idx = self.buffers.len();
        self.buffers.push(buffer);
        self.ast_to_buffer.insert(ast_id, idx);
        idx
    }

    /// Set the output buffer index.
    pub fn set_output_buffer(&mut self, idx: usize) {
        self.output_buffer_idx = idx;
    }

    /// Add a prepared kernel.
    pub fn add_kernel(&mut self, kernel: PreparedKernel) {
        self.kernels.push(kernel);
    }

    /// Set parallel groups.
    pub fn set_parallel_groups(&mut self, groups: Vec<ParallelGroup>) {
        self.parallel_groups = groups;
    }

    /// Build the ExecutionPlan.
    pub fn build(self) -> ExecutionPlan {
        ExecutionPlan {
            kernels: self.kernels,
            parallel_groups: self.parallel_groups,
            buffers: self.buffers,
            ast_to_buffer: self.ast_to_buffer,
            output_buffer_idx: self.output_buffer_idx,
            device: self.device,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_send_ptr() {
        let mut data: Vec<u8> = vec![1, 2, 3, 4];
        let ptr = data.as_mut_ptr();

        // SAFETY: ptr is valid for this test
        let send_ptr = unsafe { SendPtr::new(ptr) };

        // Verify we can get the pointer back
        assert_eq!(send_ptr.as_ptr(), ptr);
    }

    #[test]
    fn test_builder_basic() {
        let builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
        let plan = builder.build();

        assert!(plan.kernels.is_empty());
        assert!(plan.parallel_groups.is_empty());
        assert!(plan.buffers.is_empty());
        assert_eq!(plan.device, DeviceSpec::Cpu);
    }

    #[test]
    fn test_parallel_group_debug() {
        let group = ParallelGroup { kernel_indices: vec![0, 1, 2] };

        let debug_str = format!("{:?}", group);
        assert!(debug_str.contains("kernel_indices"));
    }
}
