//! CPU-specific LLVM code generation.
//!
//! This module provides the CPU renderer which generates LLVM IR for CPU execution.
//! Uses text-based IR generation (following Tinygrad's approach) instead of inkwell builder API.

use inkwell::context::Context;
use std::sync::Arc;

use morok_ir::UOp;

use crate::{RenderedKernel, Renderer};

/// CPU LLVM renderer.
///
/// Generates LLVM IR for CPU execution using text-based IR generation.
/// This renderer delegates to `LlvmTextRenderer` for actual IR generation,
/// which produces the same kernel signature: `void @kernel(ptr %args, ptr %vars)`
pub struct CpuLlvmRenderer<'ctx> {
    #[allow(dead_code)]
    context: &'ctx Context,
}

impl<'ctx> CpuLlvmRenderer<'ctx> {
    /// Create a new CPU LLVM renderer with a given context.
    ///
    /// Note: The context is kept for API compatibility but text-based
    /// generation doesn't require it. The runtime will create its own
    /// context when parsing the generated IR.
    pub fn new(context: &'ctx Context) -> Self {
        Self { context }
    }

    /// Get the inkwell context.
    pub fn context(&self) -> &'ctx Context {
        self.context
    }
}

impl<'ctx> Renderer for CpuLlvmRenderer<'ctx> {
    fn render(&self, uop: &Arc<UOp>, name: Option<&str>) -> crate::Result<RenderedKernel> {
        // Delegate to text-based renderer
        crate::llvm::text::render(uop, name)
    }

    fn backend_name(&self) -> &str {
        "llvm"
    }

    fn decompositor(&self) -> Option<morok_ir::pattern::TypedPatternMatcher<()>> {
        None
    }
}
