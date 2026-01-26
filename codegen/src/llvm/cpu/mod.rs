//! CPU-specific LLVM code generation.
//!
//! This module provides CPU renderer which generates LLVM IR for CPU execution.
//! Uses text-based IR generation (following Tinygrad's approach).

use morok_ir::UOp;

use crate::{RenderedKernel, Renderer};
use std::sync::Arc;

/// CPU LLVM renderer using text-based IR generation.
///
/// Generates LLVM IR for CPU execution using text-based IR generation.
/// This renderer delegates to `LlvmTextRenderer` for actual IR generation,
/// which produces the kernel signature: `void @kernel(ptr %args, ptr %vars)`
pub struct CpuLlvmRenderer;

impl Default for CpuLlvmRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuLlvmRenderer {
    /// Create a new CPU LLVM renderer.
    pub fn new() -> Self {
        Self
    }
}

impl Renderer for CpuLlvmRenderer {
    fn render(&self, uop: &Arc<UOp>, name: Option<&str>) -> crate::Result<RenderedKernel> {
        crate::llvm::text::render(uop, name)
    }

    fn backend_name(&self) -> &str {
        "llvm"
    }

    fn decompositor(&self) -> Option<morok_ir::pattern::TypedPatternMatcher<()>> {
        None
    }
}
