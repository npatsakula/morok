//! Generic kernel execution interface.

use crate::Result;

/// A compiled kernel ready for execution.
///
/// This trait abstracts over different kernel compilation backends:
/// - LLVM JIT (from IR)
/// - Native shared libraries (.so/.dll)
/// - Pre-compiled LLVM bitcode
/// - CUDA PTX
/// - Metal/Vulkan compute shaders
pub trait CompiledKernel {
    /// Execute the kernel with buffer pointers.
    ///
    /// # Safety
    ///
    /// All buffer pointers must be valid for the duration of kernel execution.
    /// The number and types of buffers must match the kernel's signature.
    unsafe fn execute(&self, buffers: &[*mut u8]) -> Result<()>;

    /// Get the kernel name for debugging/profiling.
    fn name(&self) -> &str;
}
