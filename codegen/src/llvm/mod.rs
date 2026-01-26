//! LLVM IR code generation.
//!
//! This module generates LLVM IR code from optimized UOp graphs.
//!
//! # Module Structure
//!
//! - `cpu/`: CPU-specific renderer wrapper (delegates to text-based codegen)
//! - `text/`: Text-based LLVM IR generation (string-based, Tinygrad-style)
//! - `gpu/`: GPU-specific renderer stubs (for future use)

pub mod cpu;
pub mod gpu;
pub mod text;

pub use cpu::CpuLlvmRenderer as LlvmRenderer;
pub use text::LlvmTextRenderer;

use crate::{RenderedKernel, Renderer};
use morok_ir::UOp;
use std::sync::Arc;

/// Render a UOp graph to LLVM IR.
pub fn render(uop: &Arc<UOp>, name: Option<&str>) -> crate::Result<RenderedKernel> {
    cpu::CpuLlvmRenderer::render(&cpu::CpuLlvmRenderer, uop, name)
}
