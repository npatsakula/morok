//! LLVM IR code generation.
//!
//! This module generates LLVM IR code from optimized UOp graphs.
//!
//! # Module Structure
//!
//! - `common/`: Shared infrastructure (types, builders, intrinsics, loop generation)
//! - `cpu/`: CPU-specific renderer and ops (inkwell-based)
//! - `text/`: Text-based LLVM IR generation (string-based, Tinygrad-style)
//! - `gpu/`: GPU-specific renderer stubs (for future use)
//! - `error`: Error types
//! - `helpers`: ValueMap and LoopContext

pub mod common;
pub mod cpu;
pub mod error;
pub mod gpu;
pub mod helpers;
pub mod text;

// Re-export CPU renderer as LlvmRenderer for backward compatibility
pub use cpu::CpuLlvmRenderer as LlvmRenderer;
pub use error::{Error, Result};

// Re-export text renderer
pub use text::LlvmTextRenderer;

use crate::{RenderedKernel, with_context};
use morok_ir::UOp;
use std::sync::Arc;

/// Render a UOp graph to LLVM IR using the thread-local context.
pub fn render(uop: &Arc<UOp>, name: Option<&str>) -> crate::Result<RenderedKernel> {
    with_context(|context| {
        let renderer = cpu::CpuLlvmRenderer::new(context);
        crate::Renderer::render(&renderer, uop, name)
    })
}
