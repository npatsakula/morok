//! Runtime execution for morok kernels.
//!
//! Provides generic kernel execution interface with backend-specific implementations
//! (LLVM JIT, native shared libraries, CUDA, etc.).

pub mod devices;
pub mod error;
pub mod kernel;
pub mod kernel_cache;
pub mod llvm;

#[cfg(test)]
pub mod test;

pub use devices::cpu::create_cpu_device;
pub use error::*;
pub use kernel::*;
pub use kernel_cache::*;
pub use llvm::*;
