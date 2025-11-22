//! Core traits for code generation.

use crate::{Result, RenderedKernel};
use morok_ir::UOp;
use std::rc::Rc;

/// Backend-agnostic code generation interface.
///
/// Implementers generate executable code from optimized UOp graphs.
/// Different backends (LLVM, CUDA, Metal, etc.) implement this trait
/// to generate code in their respective formats.
pub trait Renderer {
    /// Render a UOp graph into executable code.
    ///
    /// Takes an optimized UOp graph (typically from the scheduler/optimizer)
    /// and generates code that can be compiled and executed.
    ///
    /// # Arguments
    ///
    /// * `uop` - The root UOp of the computation graph
    /// * `name` - Optional name for the kernel (used for debugging/caching)
    ///
    /// # Returns
    ///
    /// A `RenderedKernel` containing the generated code and metadata.
    fn render(&self, uop: &Rc<UOp>, name: Option<&str>) -> Result<RenderedKernel>;

    /// Get the backend name (e.g., "llvm", "cuda", "metal").
    fn backend_name(&self) -> &str;

    /// Check if the backend supports a specific operation.
    fn supports_op(&self, op: &morok_ir::Op) -> bool;
}
