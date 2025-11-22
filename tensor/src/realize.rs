//! Tensor realization (execution) API.
//!
//! This module provides the execution pipeline for tensor operations:
//! 1. **Rangeify** - Transform movement ops to BUFFERIZE + INDEX
//! 2. **Kernel splitting** - Split at STORE boundaries into KERNEL ops
//! 3. **Scheduling** - Extract kernels and create execution schedule
//! 4. **Execution** - Compile and run each kernel in dependency order

use crate::{Result, Tensor};
use morok_runtime::{CompiledKernel, LlvmKernel};
use morok_device::Buffer;
use morok_ir::UOp;

impl Tensor {
    /// Realize (execute) this tensor's computation graph.
    ///
    /// This executes the full compilation pipeline:
    /// 1. Creates a SINK of the computation graph
    /// 2. Runs rangeify pipeline (movement ops → BUFFERIZE + INDEX)
    /// 3. Runs kernel splitting (BUFFERIZE → KERNEL operations)
    /// 4. Creates schedule of kernels to execute
    /// 5. Compiles and executes each kernel
    ///
    /// After realization, the tensor's buffer contains the computed results.
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
    /// Returns error if:
    /// - Rangeify transformation fails
    /// - No kernels found after scheduling
    /// - Buffer allocation fails
    /// - Kernel compilation fails
    /// - Kernel execution fails
    pub fn realize(self) -> Result<Self> {
        // Step 1: Create SINK of this tensor
        let sink = UOp::sink(vec![self.uop.clone()]);

        // Step 2: Run rangeify pipeline (Phases 1-4)
        // This transforms movement ops (RESHAPE, PERMUTE, etc.) into
        // BUFFERIZE + INDEX operations with explicit ranges
        let (rangeified, _context) = morok_schedule::rangeify(sink).map_err(|e| {
            crate::Error::Runtime {
                message: format!("Rangeify failed: {}", e),
            }
        })?;

        // Step 3: Run kernel splitting pipeline (Phase 5)
        // This transforms BUFFERIZE → KERNEL operations by splitting
        // at STORE boundaries
        let kernelized = morok_schedule::run_kernel_split_pipeline(rangeified);

        // Step 4: Create schedule from kernels
        // Extracts KERNEL operations and creates ScheduleItems with buffers
        let schedule = crate::schedule::create_schedule(kernelized)?;

        // Step 5: Execute schedule
        // Compiles and runs each kernel in dependency order
        execute_schedule(&schedule)?;

        Ok(self)
    }
}

/// Execute a schedule of kernels.
///
/// For each kernel in the schedule:
/// 1. Allocates buffers for intermediate results (DEFINE_GLOBAL/DEFINE_LOCAL)
/// 2. Renders kernel AST to LLVM IR
/// 3. JIT compiles the kernel
/// 4. Executes the kernel with buffer pointers
///
/// # Arguments
///
/// * `schedule` - The schedule of kernels to execute
///
/// # Errors
///
/// Returns error if any step fails (allocation, compilation, execution).
fn execute_schedule(schedule: &crate::schedule::Schedule) -> Result<()> {
    for item in schedule {
        // Step 1: Allocate buffers for this kernel
        // This handles DEFINE_GLOBAL/DEFINE_LOCAL allocations
        let buffers = crate::schedule::allocate_kernel_buffers(&item.ast)?;

        // Step 2: Ensure all buffers are allocated
        for buffer in &buffers {
            buffer.ensure_allocated().map_err(|e| crate::Error::Device {
                message: format!("Buffer allocation failed: {}", e),
            })?;
        }

        // Step 3: Render kernel AST to LLVM IR
        let rendered = morok_codegen::llvm::render(&item.ast, Some("kernel"))
            .map_err(|e| crate::Error::Codegen {
                message: format!("Failed to render kernel: {}", e),
            })?;

        // Step 4: JIT compile
        let kernel = LlvmKernel::compile(&rendered).map_err(|e| {
            crate::Error::Runtime {
                message: format!("Failed to compile kernel: {}", e),
            }
        })?;

        // Step 5: Collect buffer pointers
        let pointers: Vec<*mut u8> = buffers
            .iter()
            .map(|b| unsafe { get_buffer_ptr(b) })
            .collect::<Result<Vec<_>>>()?;

        // Step 6: Execute kernel
        unsafe {
            kernel.execute(&pointers).map_err(|e| crate::Error::Runtime {
                message: format!("Kernel execution failed: {}", e),
            })?;
        }
    }

    Ok(())
}

/// Get raw pointer to buffer data.
///
/// # Safety
///
/// This is unsafe because we're extracting a raw pointer from the buffer.
/// The caller must ensure exclusive access during kernel execution.
unsafe fn get_buffer_ptr(buffer: &Buffer) -> Result<*mut u8> {
    Ok(unsafe { buffer.as_raw_ptr() })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // TODO: Enable once rangeify/kernel splitting is fully integrated
    fn test_realize_simple_add() {
        // Test that realizing a simple computation works
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
        let b = Tensor::from_slice(&[4.0f32, 5.0, 6.0]);

        // Create computation: a + b
        let c = &a + &b;

        // Realize should compile and execute the kernel
        let result = c.realize();
        if let Err(ref e) = result {
            eprintln!("Realize failed: {:?}", e);
        }
        assert!(result.is_ok());
    }

    // More comprehensive tests will be added in Phase 1.5
}
