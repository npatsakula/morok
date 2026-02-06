//! Cranelift-specific error types.
//!
//! Provides type-safe error handling for Cranelift code generation operations,
//! matching the pattern established in the LLVM module.

use snafu::Snafu;

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Errors that can occur during Cranelift code generation.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum Error {
    // ========================================================================
    // Unsupported operations
    // ========================================================================
    /// Operation not supported by Cranelift backend.
    #[snafu(display("Unsupported: {what}"))]
    Unsupported { what: &'static str },

    /// Operation requires decomposition before codegen.
    #[snafu(display("Transcendental {op} should be decomposed before codegen"))]
    RequiresDecomposition { op: &'static str },

    /// Unknown operation encountered.
    #[snafu(display("Unknown operation: {op}"))]
    UnknownOp { op: String },

    // ========================================================================
    // Graph structure errors
    // ========================================================================
    /// UOp does not produce a value (e.g., STORE, SINK used where value expected).
    #[snafu(display("UOp {op} (id={id}) does not produce a value"))]
    NoValue { op: String, id: u64 },

    /// Required value not found.
    #[snafu(display("{what} not found (id={id})"))]
    NotFound { what: &'static str, id: u64 },

    /// Loop context not found for range.
    #[snafu(display("Loop context for range id={id} not found"))]
    LoopContextNotFound { id: u64 },

    /// REDUCE range must be a RANGE operation.
    #[snafu(display("REDUCE range must be RANGE op, got id={id}"))]
    InvalidReduceRange { id: u64 },

    // ========================================================================
    // Type errors
    // ========================================================================
    /// Type not supported for Cranelift codegen.
    #[snafu(display("Type {dtype} not supported"))]
    UnsupportedType { dtype: String },
}

/// Convert Cranelift error to parent crate error.
impl From<Error> for crate::Error {
    fn from(e: Error) -> Self {
        match e {
            Error::Unsupported { what } => crate::Error::UnsupportedOp { op: what.to_string() },
            Error::RequiresDecomposition { op } => {
                crate::Error::UnsupportedOp { op: format!("Transcendental {} should be decomposed before codegen", op) }
            }
            Error::UnknownOp { op } => crate::Error::UnsupportedOp { op },
            Error::NoValue { op, id } => crate::Error::Missing { what: format!("value for {} (id={})", op, id) },
            Error::NotFound { what, id } => crate::Error::Missing { what: format!("{} (id={})", what, id) },
            Error::LoopContextNotFound { id } => {
                crate::Error::Missing { what: format!("loop context for range (id={})", id) }
            }
            Error::InvalidReduceRange { id } => {
                crate::Error::UnsupportedOp { op: format!("REDUCE range must be RANGE op, got id={}", id) }
            }
            Error::UnsupportedType { dtype } => crate::Error::TypeError { reason: dtype },
        }
    }
}
