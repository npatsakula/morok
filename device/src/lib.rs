//! Device abstraction layer for tensor operations.
//!
//! This module provides a clean abstraction over different compute devices (CPU, CUDA, etc.)
//! with support for:
//! - Lazy buffer allocation
//! - Buffer views with zero-copy slicing
//! - LRU caching of allocations for performance
//! - Device-agnostic copy operations
//!
//! # Examples
//!
//! ```no_run
//! use morok_device::allocator::BufferOptions;
//! use morok_device::{Buffer, registry};
//! use morok_dtype::DType;
//!
//! // Get a CPU device
//! let cpu = registry::cpu().unwrap();
//!
//! // Create a buffer with lazy allocation
//! let buffer = Buffer::new(cpu, DType::Float32, vec![10, 10], BufferOptions::default());
//!
//! // Allocation happens on first use
//! buffer.ensure_allocated().unwrap();
//! ```

pub mod allocator;
pub mod buffer;
pub mod error;
pub mod registry;

pub use buffer::Buffer;
pub use error::{Error, Result};

#[cfg(test)]
mod test;

// Re-export commonly used types
#[cfg(feature = "cuda")]
pub use allocator::CudaAllocator;
pub use allocator::{Allocator, BufferOptions, CpuAllocator};
#[cfg(feature = "cuda")]
pub use registry::cuda;
pub use registry::{DeviceSpec, cpu, get_device};
