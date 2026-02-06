//! GPU-specific LLVM IR operation rendering.
//!
//! Placeholder for future GPU backends (HIP, CUDA, Metal).
//! When implementing GPU support, see Tinygrad's AMDLLVMRenderer for patterns:
//! - Work item IDs: @llvm.amdgcn.workgroup.id.x / @llvm.amdgcn.workitem.id.x
//! - Barriers: @llvm.amdgcn.s.barrier() with fences
//! - Shared memory: addrspace(3) global
//! - WMMA: @llvm.amdgcn.wmma / @llvm.amdgcn.mfma intrinsics

use crate::llvm::common::RenderContext;
use morok_ir::UOp;
use std::sync::Arc;

/// Render a UOp to LLVM IR string for GPU backend.
///
/// Currently unimplemented - returns None for all ops.
pub fn render_uop(_uop: &Arc<UOp>, _ctx: &mut RenderContext, _kernel: &mut Vec<String>) -> Option<()> {
    None
}
