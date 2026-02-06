//! GPU-specific LLVM IR text generation.
//!
//! Placeholder for future GPU backends (HIP, CUDA, Metal).
//! When implementing GPU support, see Tinygrad's AMDLLVMRenderer for patterns:
//! - Work item IDs: @llvm.amdgcn.workgroup.id.x / @llvm.amdgcn.workitem.id.x
//! - Barriers: @llvm.amdgcn.s.barrier() with fences
//! - Shared memory: addrspace(3) global
//! - WMMA: @llvm.amdgcn.wmma / @llvm.amdgcn.mfma intrinsics

pub mod ops;

pub use ops::render_uop;
