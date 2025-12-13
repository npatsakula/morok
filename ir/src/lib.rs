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
//! - [`uop::constructors`] - UOp constructor methods by semantic category
//! - [`indexing`] - Multi-dimensional indexing support
//! - [`error`] - Error types and result handling
//! - [`shape`] - Shape inference utilities
//! - [`sint`] - Symbolic integers

// Make this crate available as `morok_ir` for proc-macro generated code
extern crate self as morok_ir;

// Module declarations
pub mod decompositions;
pub mod error;
pub mod indexing;
pub mod kernel_info;
pub mod op;
pub mod prelude;
pub mod shape;
pub mod sint;
pub mod types;
pub mod uop;

pub mod provenance;

#[macro_use]
pub mod pattern;
pub mod rewrite;

#[cfg(any(test, feature = "proptest"))]
pub mod test;

// Re-exports for backward compatibility
// All types remain accessible at the crate root
pub use error::{Error, IndexTypeMismatchSnafu, Result};
pub use indexing::IndexSpec;
pub use op::Op;
pub use sint::{SInt, sint_max, sint_min, sint_prod};
pub use types::{
    AddrSpace, AxisId, AxisType, BinaryOp, BufferizeOpts, ConstValue, ConstValueHash, ReduceOp, TernaryOp, UnaryOp,
    WmmaMetadata,
};
pub use uop::{IntoUOp, UOp, UOpKey};

// Re-export pattern matching and rewriting infrastructure
pub use pattern::{BindingStore, BindingStoreExt, PatternMatcher, UPat, VarIntern};
pub use rewrite::{graph_rewrite, graph_rewrite_bottom_up};

// Re-export external types for convenience
pub use morok_dtype::DType;
pub use morok_dtype::DeviceSpec;
