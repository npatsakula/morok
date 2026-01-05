//! Common imports for working with UOp graphs.
//!
//! This module provides a convenient way to import the most commonly used types
//! when working with the IR:
//!
//! ```rust,ignore
//! use morok_ir::prelude::*;
//! ```

// Core types
pub use crate::Op;
pub use crate::uop::{IntoUOp, UOp, UOpKey};

// Operation types
pub use crate::types::{BinaryOp, ConstValue, ConstValueHash, ReduceOp, TernaryOp, UnaryOp};

// Shape and indexing
pub use crate::indexing::IndexSpec;
pub use crate::sint::SInt;

// Re-exports from dependencies
pub use morok_dtype::DType;
pub use morok_dtype::DeviceSpec;

pub use strum::AsRefStr;
