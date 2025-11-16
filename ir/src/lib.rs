//! Intermediate Representation (IR) for the Morok compiler.
//!
//! This crate defines the core IR data structures and operations used throughout
//! the Morok compiler pipeline.
//!
//! # Module Organization
//!
//! - [`types`] - Fundamental type definitions (ConstValue, operation types, etc.)
//! - [`op`] - Operation enum defining all IR operations
//! - [`uop`] - UOp (micro-operation) struct and implementation
//! - [`indexing`] - Multi-dimensional indexing support
//! - [`error`] - Error types and result handling
//! - [`ops`] - Operation implementations (arithmetic, bitwise, etc.)
//! - [`shape`] - Shape inference utilities
//! - [`sint`] - Symbolic integers

// Module declarations
pub mod error;
pub mod indexing;
pub mod op;
pub mod ops;
pub mod shape;
pub mod sint;
pub mod types;
pub mod uop;

#[cfg(test)]
pub mod test;

// Re-exports for backward compatibility
// All types remain accessible at the crate root
pub use error::{Error, IndexTypeMismatchSnafu, Result};
pub use indexing::IndexSpec;
pub use op::Op;
pub use sint::{SInt, sint_max, sint_min, sint_prod};
pub use types::{
    AddrSpace, AxisType, BinaryOp, BufferizeOpts, ConstValue, ConstValueHash, ReduceOp, TernaryOp, UnaryOp,
    WmmaMetadata,
};
pub use uop::{IntoUOp, UOp};

// Re-export external types for convenience
pub use morok_device::DeviceSpec;
pub use morok_dtype::DType;
