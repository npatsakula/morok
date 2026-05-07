//! Runtime execution for morok kernels.
//!
//! Provides generic kernel execution interface with backend-specific implementations
//! (LLVM JIT, native shared libraries, CUDA, etc.).
//!
//! # Execution Model
//!
//! `execution_plan` is the canonical runtime path and executes prepared
//! operations in dependency order with hazard-aware host parallelism.
//! The `executor` module remains available for explicit parallel scheduling
//! scenarios.
//!
//! # Benchmarking
//!
//! The `benchmark` module provides timing utilities for measuring kernel
//! execution performance, used by beam search auto-tuning.

pub mod benchmark;
pub mod clang;
pub mod custom_function;
pub mod device_registry;
pub mod devices;
pub(crate) mod dispatch;
pub mod error;
pub mod execution_plan;
pub mod executor;
pub mod jit_loader;
pub mod kernel_cache;
pub mod llvm;
#[cfg(feature = "mlir")]
pub mod mlir;
pub mod profiler;

#[cfg(test)]
pub mod test;

pub use benchmark::{BenchmarkConfig, BenchmarkResult, benchmark_kernel, warmup_thread_pool};
pub use custom_function::run_custom_function;
pub use device_registry::DEVICE_FACTORIES;
pub use devices::cpu::{CpuBackend, create_cpu_device, create_cpu_device_with_backend};
pub use devices::cpu_queue::CpuQueue;
pub use error::*;
pub use execution_plan::{
    ExecutionPlan, ExecutionPlanBuilder, PreparedBufferView, PreparedCopy, PreparedCustomFunction, PreparedKernel,
    PreparedOp,
};
pub use executor::{
    DeviceContext, ExecutionGraph, ExecutionNode, KernelBufferAccess, SyncStrategy, UnifiedExecutor, global_executor,
};
pub use kernel_cache::*;
pub use llvm::*;
pub use profiler::KernelProfile;
