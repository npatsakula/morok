//! Hardware command queue abstraction for parallel execution.
//!
//! This module provides a device-agnostic interface for command queues,
//! abstracting over CUDA streams, Metal command buffers, CPU task queues, etc.
//!
//! # Design
//!
//! The `HardwareQueue` trait uses a builder pattern for chaining operations:
//!
//! ```ignore
//! queue
//!     .wait(&signal, 1)       // Wait for dependency
//!     .exec(&kernel, &bufs)   // Execute kernel
//!     .signal(&signal, 2)     // Signal completion
//!     .submit()?;             // Submit to hardware
//! ```
//!
//! # Queue Types
//!
//! Most devices have two queue types:
//! - **Compute queue**: For kernel execution
//! - **Copy queue**: For DMA transfers (optional, some devices share)

use std::sync::Arc;

use morok_dtype::DeviceSpec;

use crate::buffer::Buffer;
use crate::error::Result;
use crate::sync::TimelineSignal;

/// Kernel execution parameters.
#[derive(Debug, Clone)]
pub struct ExecParams {
    /// Global work size (total number of work items per dimension).
    pub global_size: [usize; 3],
    /// Local work size (work group size per dimension).
    pub local_size: [usize; 3],
}

impl ExecParams {
    /// Create 1D execution parameters.
    pub fn new_1d(global: usize, local: usize) -> Self {
        Self { global_size: [global, 1, 1], local_size: [local, 1, 1] }
    }

    /// Create 2D execution parameters.
    pub fn new_2d(global: [usize; 2], local: [usize; 2]) -> Self {
        Self { global_size: [global[0], global[1], 1], local_size: [local[0], local[1], 1] }
    }

    /// Create 3D execution parameters.
    pub fn new_3d(global: [usize; 3], local: [usize; 3]) -> Self {
        Self { global_size: global, local_size: local }
    }
}

impl Default for ExecParams {
    fn default() -> Self {
        Self { global_size: [1, 1, 1], local_size: [1, 1, 1] }
    }
}

/// Compiled program that can be executed on a queue.
///
/// This is a thin wrapper around device-specific program handles
/// (JIT function pointers, CUDA modules, etc.).
pub trait Program: Send + Sync + std::fmt::Debug {
    /// Get the device this program is compiled for.
    fn device(&self) -> &DeviceSpec;

    /// Get the program name (for debugging).
    fn name(&self) -> &str;
}

/// Hardware command queue for submitting operations to a device.
///
/// Queues batch operations and submit them to hardware atomically.
/// All operations are non-blocking until `submit()` is called.
///
/// # Thread Safety
///
/// Queues are `Send` but not necessarily `Sync`. Each queue should be
/// owned by a single thread/task at a time.
pub trait HardwareQueue: Send + std::fmt::Debug {
    /// The timeline signal type used by this queue.
    type Signal: TimelineSignal;

    /// Wait for a signal to reach a value before executing subsequent operations.
    ///
    /// This creates a dependency: operations after this call won't start
    /// until the signal reaches `value`.
    fn wait(&mut self, signal: &Self::Signal, value: u64) -> &mut Self;

    /// Signal a value after all previous operations complete.
    ///
    /// Operations submitted after this call may start before the signal is set.
    fn signal(&mut self, signal: &Self::Signal, value: u64) -> &mut Self;

    /// Execute a compiled program with the given buffers and parameters.
    ///
    /// # Arguments
    ///
    /// * `program` - The compiled program to execute
    /// * `buffers` - Buffer arguments (raw pointers extracted internally)
    /// * `params` - Execution parameters (grid size, etc.)
    ///
    /// # Safety
    ///
    /// Caller must ensure:
    /// - All buffers are allocated
    /// - No conflicting buffer accesses (handled by executor)
    fn exec(&mut self, program: &dyn Program, buffers: &[&Buffer], params: &ExecParams) -> &mut Self;

    /// Copy data between buffers.
    ///
    /// Both buffers must be accessible from this queue's device.
    /// For cross-device copies, use the executor's transfer mechanism.
    fn copy(&mut self, dst: &Buffer, src: &Buffer) -> &mut Self;

    /// Insert a memory barrier.
    ///
    /// Ensures all previous memory operations are visible to subsequent operations.
    /// Mostly needed for CPU and some GPU memory models.
    fn memory_barrier(&mut self) -> &mut Self;

    /// Submit all batched operations to the hardware.
    ///
    /// This is the only blocking point - it submits work but doesn't wait
    /// for completion. Use signals to synchronize.
    fn submit(&mut self) -> Result<()>;

    /// Get the device this queue belongs to.
    fn device(&self) -> &DeviceSpec;
}

/// Factory for creating hardware queues.
///
/// Each device implementation provides a factory that creates queues
/// for that device type.
pub trait QueueFactory: Send + Sync + std::fmt::Debug {
    /// The queue type produced by this factory.
    type Queue: HardwareQueue;

    /// The signal type used by queues from this factory.
    type Signal: TimelineSignal;

    /// Create a new compute queue.
    fn create_compute_queue(&self) -> Result<Self::Queue>;

    /// Create a new copy/DMA queue if supported.
    ///
    /// Returns `None` if the device doesn't support separate copy queues.
    fn create_copy_queue(&self) -> Result<Option<Self::Queue>>;

    /// Create a new timeline signal.
    fn create_signal(&self) -> Result<Arc<Self::Signal>>;

    /// Get the device specification.
    fn device(&self) -> &DeviceSpec;
}

/// Type-erased queue for use in the unified executor.
///
/// This wraps a concrete `HardwareQueue` implementation and provides
/// a common interface that doesn't require knowing the signal type.
pub struct DynQueue {
    inner: Box<dyn DynQueueInner>,
}

impl std::fmt::Debug for DynQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynQueue").field("device", &self.inner.device()).finish()
    }
}

impl DynQueue {
    /// Create a new type-erased queue from a concrete implementation.
    pub fn new<Q: HardwareQueue + 'static>(queue: Q) -> Self
    where
        Q::Signal: 'static,
    {
        Self { inner: Box::new(DynQueueWrapper { queue, _phantom: std::marker::PhantomData }) }
    }

    /// Wait for a type-erased signal.
    pub fn wait(&mut self, signal: &dyn TimelineSignal, value: u64) -> &mut Self {
        self.inner.wait_dyn(signal, value);
        self
    }

    /// Signal completion.
    pub fn signal(&mut self, signal: &dyn TimelineSignal, value: u64) -> &mut Self {
        self.inner.signal_dyn(signal, value);
        self
    }

    /// Execute a program.
    pub fn exec(&mut self, program: &dyn Program, buffers: &[&Buffer], params: &ExecParams) -> &mut Self {
        self.inner.exec_dyn(program, buffers, params);
        self
    }

    /// Copy between buffers.
    pub fn copy(&mut self, dst: &Buffer, src: &Buffer) -> &mut Self {
        self.inner.copy_dyn(dst, src);
        self
    }

    /// Insert memory barrier.
    pub fn memory_barrier(&mut self) -> &mut Self {
        self.inner.memory_barrier_dyn();
        self
    }

    /// Submit to hardware.
    pub fn submit(&mut self) -> Result<()> {
        self.inner.submit_dyn()
    }

    /// Get the device.
    pub fn device(&self) -> &DeviceSpec {
        self.inner.device()
    }
}

/// Internal trait for type erasure.
trait DynQueueInner: Send + std::fmt::Debug {
    fn wait_dyn(&mut self, signal: &dyn TimelineSignal, value: u64);
    fn signal_dyn(&mut self, signal: &dyn TimelineSignal, value: u64);
    fn exec_dyn(&mut self, program: &dyn Program, buffers: &[&Buffer], params: &ExecParams);
    fn copy_dyn(&mut self, dst: &Buffer, src: &Buffer);
    fn memory_barrier_dyn(&mut self);
    fn submit_dyn(&mut self) -> Result<()>;
    fn device(&self) -> &DeviceSpec;
}

/// Wrapper for concrete queue types.
struct DynQueueWrapper<Q: HardwareQueue> {
    queue: Q,
    _phantom: std::marker::PhantomData<Q::Signal>,
}

impl<Q: HardwareQueue> std::fmt::Debug for DynQueueWrapper<Q> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynQueueWrapper").field("queue", &self.queue).finish()
    }
}

impl<Q: HardwareQueue + 'static> DynQueueInner for DynQueueWrapper<Q>
where
    Q::Signal: 'static,
{
    fn wait_dyn(&mut self, _signal: &dyn TimelineSignal, _value: u64) {
        // Note: In a full implementation, we'd need to downcast the signal
        // to the concrete type. For now, this is a placeholder that demonstrates
        // the interface. The real implementation will use concrete types
        // in the executor where the signal type is known.
        //
        // The type-erased DynQueue is mainly for heterogeneous collections;
        // most code paths will use concrete types directly.
    }

    fn signal_dyn(&mut self, _signal: &dyn TimelineSignal, _value: u64) {
        // See wait_dyn comment
    }

    fn exec_dyn(&mut self, program: &dyn Program, buffers: &[&Buffer], params: &ExecParams) {
        self.queue.exec(program, buffers, params);
    }

    fn copy_dyn(&mut self, dst: &Buffer, src: &Buffer) {
        self.queue.copy(dst, src);
    }

    fn memory_barrier_dyn(&mut self) {
        self.queue.memory_barrier();
    }

    fn submit_dyn(&mut self) -> Result<()> {
        self.queue.submit()
    }

    fn device(&self) -> &DeviceSpec {
        self.queue.device()
    }
}
