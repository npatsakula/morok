//! Runtime execution for morok kernels.
//!
//! Provides generic kernel execution interface with backend-specific implementations
//! (LLVM JIT, Cranelift JIT, native shared libraries, CUDA, etc.).
//!
//! # Parallel Execution
//!
//! The `executor` module provides the `UnifiedExecutor` for parallel kernel
//! execution across heterogeneous devices (CPU, GPU, etc.).
//!
//! # Benchmarking
//!
//! The `benchmark` module provides timing utilities for measuring kernel
//! execution performance, used by beam search auto-tuning.

pub mod benchmark;
pub mod cranelift;
pub mod device_registry;
pub mod devices;
pub mod error;
pub mod execution_plan;
pub mod executor;
pub mod kernel_cache;
pub mod llvm;

#[cfg(test)]
pub mod test;

pub use benchmark::{BenchmarkConfig, BenchmarkResult, benchmark_kernel, benchmark_kernel_with_cutoff};
pub use device_registry::DEVICE_FACTORIES;
pub use devices::cpu::create_cpu_device;
pub use devices::cpu_queue::CpuQueue;
pub use error::*;
pub use execution_plan::{ExecutionPlan, ExecutionPlanBuilder, ParallelGroup, PreparedKernel};
pub use executor::{
    DeviceContext, ExecutionGraph, ExecutionNode, KernelBufferAccess, SyncStrategy, UnifiedExecutor, global_executor,
};
pub use kernel_cache::*;
pub use llvm::*;
