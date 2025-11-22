//! Runtime execution for morok kernels.
//!
//! Provides generic kernel execution interface with backend-specific implementations
//! (LLVM JIT, native shared libraries, CUDA, etc.).

pub mod error;
pub mod kernel;
pub mod llvm;

#[cfg(test)]
pub mod test;

pub use error::*;
pub use kernel::*;
pub use llvm::*;
