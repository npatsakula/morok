//! GPU-specific LLVM code generation (stub).
//!
//! This module provides the foundation for GPU renderers that use LLVM
//! as their code generation backend (e.g., AMD GPUs via AMDGPU).
//!
//! Unlike CPU rendering which inlines outer loops, GPU rendering:
//! - Uses grid parallelism for outer loops (kernel is called N times)
//! - May require different memory address spaces
//! - May use different optimization attributes

// Placeholder for future GPU implementations
// This could be extended for:
// - AMDLLVMRenderer (AMDGPU backend)
// - Other LLVM-based GPU targets

/// Trait for GPU-specific LLVM renderers.
///
/// GPU renderers differ from CPU in how they handle:
/// - Outer loops (grid parallelism vs inlining)
/// - Memory address spaces (global, shared, local)
/// - Thread/workgroup dimensions
pub trait GpuLlvmRenderer {
    /// Returns false - GPU uses grid parallelism instead of inlined loops.
    fn inline_outer_loops(&self) -> bool {
        false
    }

    /// Get the global address space for this GPU target.
    fn global_address_space(&self) -> u32 {
        1 // Common default for many GPU targets
    }

    /// Get the shared/local address space for this GPU target.
    fn shared_address_space(&self) -> u32 {
        3 // Common default for many GPU targets
    }
}
