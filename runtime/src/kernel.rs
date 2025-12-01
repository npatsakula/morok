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
    /// This is a convenience method that calls execute_with_vars with empty fixedvars.
    ///
    /// # Safety
    ///
    /// All buffer pointers must be valid for the duration of kernel execution.
    /// The number and types of buffers must match the kernel's signature.
    unsafe fn execute(&self, buffers: &[*mut u8]) -> Result<()> {
        unsafe { self.execute_with_vars(buffers, &std::collections::HashMap::new()) }
    }

    /// Execute the kernel with buffer pointers and variable values.
    ///
    /// This is the main execution method. Variables are OUTER range iteration values
    /// that are passed as additional i64 parameters to the kernel.
    ///
    /// # Safety
    ///
    /// All buffer pointers must be valid for the duration of kernel execution.
    /// The number and types of buffers must match the kernel's signature.
    /// Variable values must match the kernel's DEFINE_VAR parameters.
    unsafe fn execute_with_vars(&self, buffers: &[*mut u8], vars: &std::collections::HashMap<String, i64>) -> Result<()>;

    /// Get the kernel name for debugging/profiling.
    fn name(&self) -> &str;
}
