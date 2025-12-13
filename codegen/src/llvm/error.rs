//! LLVM-specific error types.
//!
//! Provides type-safe error handling for LLVM code generation operations.

use inkwell::builder::BuilderError;
use snafu::Snafu;

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Errors that can occur during LLVM code generation.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum Error {
    // ========================================================================
    // Builder operations
    // ========================================================================
    /// GEP (GetElementPtr) instruction failed.
    #[snafu(display("LLVM build_gep failed"))]
    BuildGep { source: BuilderError },

    /// Load instruction failed.
    #[snafu(display("LLVM build_load failed"))]
    BuildLoad { source: BuilderError },

    /// Store instruction failed.
    #[snafu(display("LLVM build_store failed"))]
    BuildStore { source: BuilderError },

    /// Function call instruction failed.
    #[snafu(display("LLVM build_call failed for '{intrinsic}'"))]
    BuildCall { intrinsic: String, source: BuilderError },

    /// PHI node creation failed.
    #[snafu(display("LLVM build_phi failed"))]
    BuildPhi { source: BuilderError },

    /// Branch instruction failed.
    #[snafu(display("LLVM build_branch failed"))]
    BuildBranch { source: BuilderError },

    /// Arithmetic operation failed.
    #[snafu(display("LLVM arithmetic operation failed"))]
    Arithmetic { source: BuilderError },

    /// Comparison operation failed.
    #[snafu(display("LLVM comparison failed"))]
    Comparison { source: BuilderError },

    /// Cast/conversion operation failed.
    #[snafu(display("LLVM cast failed"))]
    Cast { source: BuilderError },

    /// Return instruction failed.
    #[snafu(display("LLVM build_return failed"))]
    BuildReturn { source: BuilderError },

    /// Select instruction failed.
    #[snafu(display("LLVM build_select failed"))]
    BuildSelect { source: BuilderError },

    /// Alloca instruction failed.
    #[snafu(display("LLVM build_alloca failed"))]
    BuildAlloca { source: BuilderError },

    /// Value extraction from builder result failed.
    #[snafu(display("Failed to extract {expected} from builder result"))]
    ValueExtractionFailed { expected: &'static str },

    /// Function parameter retrieval failed.
    #[snafu(display("Function parameter at index {index} not found"))]
    InvalidFunctionParameter { index: u32 },

    // ========================================================================
    // Intrinsic errors
    // ========================================================================
    /// LLVM intrinsic not found.
    #[snafu(display("Intrinsic '{name}' not found"))]
    IntrinsicNotFound { name: String },

    /// Failed to get intrinsic declaration.
    #[snafu(display("Failed to get declaration for intrinsic '{name}'"))]
    IntrinsicDeclaration { name: String },

    // ========================================================================
    // Graph structure errors
    // ========================================================================
    /// UOp does not produce a value (e.g., STORE, SINK used where value expected).
    #[snafu(display("UOp {op} (id={id}) does not produce a value"))]
    NoValue { op: String, id: u64 },

    /// Required value not found in ValueMap.
    #[snafu(display("{what} not found in ValueMap (id={id})"))]
    NotInValueMap { what: &'static str, id: u64 },

    /// Loop context not found for range.
    #[snafu(display("Loop context for range id={id} not found"))]
    LoopContextNotFound { id: u64 },

    // ========================================================================
    // Builder context errors
    // ========================================================================
    /// No insert block available in builder.
    #[snafu(display("No insert block available"))]
    NoInsertBlock,

    /// No parent function available.
    #[snafu(display("No parent function available"))]
    NoParentFunction,

    // ========================================================================
    // Type errors
    // ========================================================================
    /// Type cannot be represented in LLVM.
    #[snafu(display("Type '{dtype}' cannot be converted to LLVM basic type"))]
    InvalidLlvmType { dtype: &'static str },

    /// Type not supported for intrinsic operations.
    #[snafu(display("Type {dtype} not supported for intrinsics"))]
    UnsupportedIntrinsicType { dtype: String },

    /// RANGE end value must be integer.
    #[snafu(display("RANGE end value must be integer, got {actual}"))]
    RangeEndNotInteger { actual: String },

    /// Invalid comparison operation.
    #[snafu(display("Comparison op {op} is not valid"))]
    InvalidComparisonOp { op: String },

    // ========================================================================
    // Unsupported operations
    // ========================================================================
    /// Operation not supported.
    #[snafu(display("Unsupported: {what}"))]
    Unsupported { what: &'static str },

    // ========================================================================
    // Module errors
    // ========================================================================
    /// LLVM module verification failed.
    #[snafu(display("Module verification failed: {message}"))]
    ModuleVerification { message: String },
}

/// Convert LLVM error to parent crate error.
impl From<Error> for crate::Error {
    fn from(e: Error) -> Self {
        crate::Error::LlvmError { reason: e.to_string() }
    }
}
