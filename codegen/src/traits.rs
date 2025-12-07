//! Core traits for code generation.

use crate::{RenderedKernel, Result};
use morok_ir::UOp;
use morok_ir::pattern::PatternMatcher;
use std::sync::Arc;

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
    fn render(&self, uop: &Arc<UOp>, name: Option<&str>) -> Result<RenderedKernel>;

    /// Get the backend name (e.g., "llvm", "cuda", "metal").
    fn backend_name(&self) -> &str;

    /// Returns decomposition patterns for operations this backend doesn't support.
    ///
    /// Backends that support all transcendental operations natively (e.g., LLVM)
    /// should return `None`. Backends that need decomposition (e.g., CPU interpreter)
    /// should return a `PatternMatcher` containing the decomposition rules.
    ///
    /// # Example
    ///
    /// ```ignore
    /// fn decompositor(&self) -> Option<PatternMatcher<()>> {
    ///     // LLVM has native transcendentals
    ///     None
    /// }
    /// ```
    fn decompositor(&self) -> Option<PatternMatcher<()>>;
}
