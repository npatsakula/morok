//! Kernel metadata for optimized ASTs.
//!
//! This module defines metadata structures attached to optimized UOps
//! to track optimization history and kernel properties.

use std::sync::Arc;

use super::types::Opt;

/// Metadata attached to optimized kernel ASTs.
///
/// Contains information about the kernel's name, applied optimizations,
/// and backend configuration flags. This is attached to the final AST
/// via `UOp::with_metadata()` after optimization is complete.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelInfo {
    /// Human-readable kernel name (e.g., "r_g16l16R32u4").
    ///
    /// Format: `{prefix}_{axis_encodings}`
    /// - Prefix: "r" for reduce kernels, "E" for elementwise
    /// - Axis encodings: Each range as `{letter}{size}` (e.g., "g16", "l8", "R32")
    pub name: String,

    /// Sequence of optimizations that were successfully applied.
    ///
    /// Used for reproducing optimization strategies, debugging, and
    /// performance analysis.
    pub applied_opts: Vec<Opt>,

    /// Flag indicating whether local memory should be avoided.
    ///
    /// Set by NOLOCALS optimization or for CPU backends that don't
    /// support shared/local memory.
    pub dont_use_locals: bool,
}

impl KernelInfo {
    /// Create new kernel metadata.
    pub fn new(name: impl Into<String>, applied_opts: Vec<Opt>, dont_use_locals: bool) -> Arc<Self> {
        Arc::new(Self { name: name.into(), applied_opts, dont_use_locals })
    }

    /// Get the function name (ASCII-only version of name).
    ///
    /// Converts special characters (including ANSI color codes) to hex
    /// codes for valid function identifiers in generated code.
    ///
    /// Based on Tinygrad's `to_function_name()`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let info = KernelInfo::new("r_g16l16", vec![], false);
    /// assert_eq!(info.function_name(), "r_g16l16");
    ///
    /// // With ANSI colors
    /// let info = KernelInfo::new("r\x1b[34mg16\x1b[0m", vec![], false);
    /// assert!(info.function_name().contains("1B")); // ESC = 0x1B
    /// ```
    pub fn function_name(&self) -> String {
        self.name
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() || c == '_' { c.to_string() } else { format!("{:02X}", c as u32) })
            .collect()
    }
}

#[cfg(test)]
#[path = "../test/unit/optimizer/kernel_info_internal.rs"]
mod tests;
