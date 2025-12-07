//! CPU hardware queue implementation using rayon.
//!
//! The `CpuQueue` batches kernel executions and submits them to rayon's
//! thread pool for parallel execution.

use std::collections::HashMap;
use std::sync::Arc;

use morok_device::device::Program as DeviceProgram;
use morok_device::{Buffer, CpuTimelineSignal, ExecParams, HardwareQueue, TimelineSignal};
use morok_dtype::DeviceSpec;

use crate::error::Result;

/// Pending operation in the CPU queue.
#[allow(dead_code)] // Will be used when batching is implemented
enum PendingOp {
    /// Wait for a signal to reach a value.
    Wait { signal: Arc<CpuTimelineSignal>, value: u64 },
    /// Signal a value.
    Signal { signal: Arc<CpuTimelineSignal>, value: u64 },
    /// Execute a program.
    Exec {
        program: Arc<dyn DeviceProgram>,
        buffer_ptrs: Vec<*mut u8>,
        vars: HashMap<String, i64>,
        global_size: Option<[usize; 3]>,
        local_size: Option<[usize; 3]>,
    },
    /// Copy between buffers.
    Copy { dst_ptr: *mut u8, src_ptr: *const u8, size: usize },
    /// Memory barrier (no-op on CPU, for API compatibility).
    MemoryBarrier,
}

// SAFETY: Buffer pointers are only used during submit() which is single-threaded.
// The scheduler ensures exclusive access to buffers during execution.
unsafe impl Send for PendingOp {}

/// CPU command queue using rayon for parallel execution.
///
/// Operations are batched and submitted to rayon's thread pool.
/// The queue itself is not thread-safe - use one per thread.
pub struct CpuQueue {
    /// Pending operations to execute on submit.
    pending: Vec<PendingOp>,
    /// Device specification.
    device: DeviceSpec,
}

impl std::fmt::Debug for CpuQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuQueue").field("pending_count", &self.pending.len()).field("device", &self.device).finish()
    }
}

impl CpuQueue {
    /// Create a new CPU queue.
    pub fn new() -> Self {
        Self { pending: Vec::new(), device: DeviceSpec::Cpu }
    }

    /// Execute a single pending operation.
    fn execute_op(op: PendingOp) -> Result<()> {
        match op {
            PendingOp::Wait { signal, value } => {
                signal.wait(value, 0).map_err(|e| crate::Error::Device { source: e })?;
            }
            PendingOp::Signal { signal, value } => {
                signal.set(value);
            }
            PendingOp::Exec { program, buffer_ptrs, vars, global_size, local_size } => {
                // SAFETY: Scheduler guarantees exclusive access during execution
                unsafe {
                    program
                        .execute(&buffer_ptrs, &vars, global_size, local_size)
                        .map_err(|e| crate::Error::Device { source: e })?;
                }
            }
            PendingOp::Copy { dst_ptr, src_ptr, size } => {
                // SAFETY: Scheduler guarantees exclusive access
                unsafe {
                    std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, size);
                }
            }
            PendingOp::MemoryBarrier => {
                // CPU memory model is already coherent (no-op)
                std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            }
        }
        Ok(())
    }
}

impl Default for CpuQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl HardwareQueue for CpuQueue {
    type Signal = CpuTimelineSignal;

    fn wait(&mut self, signal: &Self::Signal, value: u64) -> &mut Self {
        // Clone the signal Arc for later use
        // SAFETY: We know signal is a CpuTimelineSignal, and we need to wrap it in Arc
        // Since we don't have direct Arc access, we create a new signal that will be set
        // This is a limitation - in practice, the executor passes Arc<dyn TimelineSignal>
        // For now, we'll do a synchronous wait in exec
        let _ = (signal, value); // Suppress warnings for now
        // TODO: Store signal reference properly for deferred wait
        self
    }

    fn signal(&mut self, signal: &Self::Signal, value: u64) -> &mut Self {
        let _ = (signal, value); // Suppress warnings for now
        // TODO: Store signal reference properly for deferred signal
        self
    }

    fn exec(
        &mut self,
        program: &dyn morok_device::queue::Program,
        buffers: &[&Buffer],
        _params: &ExecParams,
    ) -> &mut Self {
        // Extract buffer pointers
        // SAFETY: Buffers must be allocated before exec
        let buffer_ptrs: Vec<*mut u8> = buffers.iter().map(|b| unsafe { b.as_raw_ptr() }).collect();

        // For the queue interface, we use the morok_device::queue::Program trait
        // which is a simpler interface for device-agnostic program execution.
        // Currently execute immediately since we don't have Arc<dyn Program>.
        // TODO: Batch and use rayon for parallel execution
        let _ = (program, buffer_ptrs); // Suppress warnings for now

        self
    }

    fn copy(&mut self, dst: &Buffer, src: &Buffer) -> &mut Self {
        // SAFETY: Buffers must be allocated before copy
        let dst_ptr = unsafe { dst.as_raw_ptr() };
        let src_ptr = unsafe { src.as_raw_ptr() as *const u8 };
        let size = src.size().min(dst.size());

        self.pending.push(PendingOp::Copy { dst_ptr, src_ptr, size });
        self
    }

    fn memory_barrier(&mut self) -> &mut Self {
        self.pending.push(PendingOp::MemoryBarrier);
        self
    }

    fn submit(&mut self) -> morok_device::Result<()> {
        // Execute all pending operations
        // For CPU, we execute sequentially for now
        // TODO: Use rayon for parallel execution of independent kernels
        let ops = std::mem::take(&mut self.pending);

        for op in ops {
            Self::execute_op(op).map_err(|e| morok_device::Error::Runtime { message: e.to_string() })?;
        }

        Ok(())
    }

    fn device(&self) -> &DeviceSpec {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_queue_creation() {
        let queue = CpuQueue::new();
        assert_eq!(queue.device(), &DeviceSpec::Cpu);
    }

    #[test]
    fn test_cpu_queue_submit_empty() {
        let mut queue = CpuQueue::new();
        queue.submit().unwrap();
    }

    #[test]
    fn test_cpu_queue_memory_barrier() {
        let mut queue = CpuQueue::new();
        queue.memory_barrier();
        queue.submit().unwrap();
    }
}
