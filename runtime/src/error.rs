//! Error types for runtime execution.

use snafu::Snafu;

/// Result type for runtime operations.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Errors that can occur during runtime execution.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum Error {
    /// Codegen error occurred.
    #[snafu(display("Codegen error: {source}"))]
    Codegen { source: morok_codegen::Error },

    /// JIT compilation failed.
    #[snafu(display("JIT compilation failed: {reason}"))]
    JitCompilation { reason: String },

    /// Function not found in module.
    #[snafu(display("Function '{name}' not found in module"))]
    FunctionNotFound { name: String },

    /// Buffer allocation failed.
    #[snafu(display("Buffer allocation failed: {reason}"))]
    BufferAllocation { reason: String },

    /// Invalid buffer size.
    #[snafu(display("Invalid buffer size: {size}"))]
    InvalidBufferSize { size: usize },

    /// Execution error.
    #[snafu(display("Execution error: {reason}"))]
    Execution { reason: String },

    /// LLVM error.
    #[snafu(display("LLVM error: {reason}"))]
    LlvmError { reason: String },

    /// Unsupported device type.
    #[snafu(display("Unsupported device type: {device}"))]
    UnsupportedDevice { device: String },

    /// Device error (from morok_device crate).
    #[snafu(display("Device error: {source}"))]
    Device { source: morok_device::Error },
}

impl From<morok_device::Error> for Error {
    fn from(source: morok_device::Error) -> Self {
        Error::Device { source }
    }
}
