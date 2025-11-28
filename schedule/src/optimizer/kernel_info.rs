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
mod tests {
    use super::*;

    #[test]
    fn test_function_name_ascii() {
        let info = KernelInfo::new("test_kernel", vec![], false);
        assert_eq!(info.function_name(), "test_kernel");
    }

    #[test]
    fn test_function_name_with_underscores() {
        let info = KernelInfo::new("r_g16l16R32u4", vec![], false);
        assert_eq!(info.function_name(), "r_g16l16R32u4");
    }

    #[test]
    fn test_function_name_unicode() {
        // ANSI escape codes should be converted to hex
        let info = KernelInfo::new("r\x1b[34mg16\x1b[0m", vec![], false);
        let func_name = info.function_name();
        assert!(func_name.contains("1B")); // ESC character (0x1B)
        assert!(func_name.contains("5B")); // '[' character (0x5B)
    }

    #[test]
    fn test_function_name_special_chars() {
        let info = KernelInfo::new("test-kernel+v2", vec![], false);
        let func_name = info.function_name();
        assert!(func_name.contains("2D")); // '-' = 0x2D
        assert!(func_name.contains("2B")); // '+' = 0x2B
    }
}
