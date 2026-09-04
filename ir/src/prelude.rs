//! Common imports for working with UOp graphs.
//!
//! This module provides a convenient way to import the most commonly used types
//! when working with the IR:
//!
//! ```rust,ignore
//! use svod_ir::prelude::*;
//! ```

// Core types
pub use crate::Op;
pub use crate::origin::{OriginId, OriginScope, OriginSet};
pub use crate::uop::{IntoUOp, UOp, UOpKey};

// Operation types
pub use crate::types::{
    BinaryOp, BinaryStageIdentity, CallInfo, ConstValue, ConstValueHash, CustomFunctionKind, InsArg, ReduceOp,
    RendererDevice, SourceStageIdentity, StageAbiParam, StageAbiParamKind, StageDigest, TernaryOp, UnaryOp,
};

// Shape and indexing
pub use crate::indexing::IndexSpec;
pub use crate::sint::SInt;

// Re-exports from dependencies
pub use svod_dtype::DType;
pub use svod_dtype::DeviceSpec;

pub use strum::AsRefStr;
