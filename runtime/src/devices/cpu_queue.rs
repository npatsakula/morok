//! CPU hardware queue implementation using rayon.
//!
//! The `CpuQueue` batches kernel executions and submits them to rayon's
//! thread pool for parallel execution.

use std::sync::Arc;

use morok_device::device::Program as DeviceProgram;
use morok_device::{Buffer, CpuTimelineSignal, ExecParams, HardwareQueue, TimelineSignal};
use morok_dtype::DeviceSpec;

use crate::error::Result;

/// Pending operation in the CPU queue.
#[allow(dead_code)] // Will be used when batching is implemented
enum PendingOp {
    /// Wait for a signal to reach a value.
    Wait { signal: CpuTimelineSignal, value: u64 },
    /// Signal a value.
    Signal { signal: CpuTimelineSignal, value: u64 },
    /// Execute a program.
    Exec {
        program: Arc<dyn DeviceProgram>,
        buffer_ptrs: Vec<*mut u8>,
        vals: Vec<i64>,
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
    /// Deferred builder-style errors surfaced by submit().
    errors: Vec<String>,
    /// Device specification.
    device: DeviceSpec,
}

impl std::fmt::Debug for CpuQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuQueue")
            .field("pending_count", &self.pending.len())
            .field("error_count", &self.errors.len())
            .field("device", &self.device)
            .finish()
    }
}

impl CpuQueue {
    /// Create a new CPU queue.
    pub fn new() -> Self {
        Self { pending: Vec::new(), errors: Vec::new(), device: DeviceSpec::Cpu }
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
            PendingOp::Exec { program, buffer_ptrs, vals, global_size, local_size } => {
                // SAFETY: Scheduler guarantees exclusive access during execution
                unsafe {
                    program
                        .execute(&buffer_ptrs, &vals, global_size, local_size)
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
        self.pending.push(PendingOp::Wait { signal: signal.clone(), value });
        self
    }

    fn signal(&mut self, signal: &Self::Signal, value: u64) -> &mut Self {
        self.pending.push(PendingOp::Signal { signal: signal.clone(), value });
        self
    }

    fn exec(&mut self, program: Arc<dyn DeviceProgram>, buffers: &[&Buffer], params: &ExecParams) -> &mut Self {
        let mut buffer_ptrs = Vec::with_capacity(buffers.len());
        for buffer in buffers {
            if let Err(err) = buffer.ensure_allocated() {
                self.errors.push(format!("CPU queue exec buffer allocation failed: {err}"));
                return self;
            }
            // SAFETY: the buffer was just allocated and the queued op executes before
            // submit() returns; scheduler ownership rules protect aliasing.
            buffer_ptrs.push(unsafe { buffer.as_raw_ptr() });
        }

        self.pending.push(PendingOp::Exec {
            program,
            buffer_ptrs,
            vals: params.vals.clone(),
            global_size: Some(params.global_size),
            local_size: params.local_size,
        });

        self
    }

    fn copy(&mut self, dst: &Buffer, src: &Buffer) -> &mut Self {
        if src.size() != dst.size() {
            self.errors.push(format!(
                "CPU queue copy size mismatch: src={} bytes, dst={} bytes",
                src.size(),
                dst.size()
            ));
            return self;
        }
        if let Err(err) = dst.ensure_allocated() {
            self.errors.push(format!("CPU queue copy dst allocation failed: {err}"));
            return self;
        }
        if let Err(err) = src.ensure_allocated() {
            self.errors.push(format!("CPU queue copy src allocation failed: {err}"));
            return self;
        }
        // SAFETY: both buffers are allocated; the queued op executes before submit() returns
        // and scheduler ownership rules protect aliasing.
        let dst_ptr = unsafe { dst.as_raw_ptr() };
        let src_ptr = unsafe { src.as_raw_ptr() as *const u8 };
        let size = src.size();

        self.pending.push(PendingOp::Copy { dst_ptr, src_ptr, size });
        self
    }

    fn memory_barrier(&mut self) -> &mut Self {
        self.pending.push(PendingOp::MemoryBarrier);
        self
    }

    fn submit(&mut self) -> morok_device::Result<()> {
        if !self.errors.is_empty() {
            let errors = std::mem::take(&mut self.errors);
            self.pending.clear();
            return Err(morok_device::Error::Runtime { message: errors.join("; ") });
        }

        // CPU queue is intentionally serial; parallelism across kernels is handled by ExecutionPlan.
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
#[path = "../test/unit/cpu_queue.rs"]
mod tests;
