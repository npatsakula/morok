//! Common infrastructure for LLVM code generation.
//!
//! This module provides shared utilities used by both CPU and GPU renderers.

pub mod builders;
pub mod intrinsics;
pub mod loop_gen;
pub mod types;

pub use builders::*;
pub use intrinsics::*;
pub use loop_gen::*;
pub use types::*;
