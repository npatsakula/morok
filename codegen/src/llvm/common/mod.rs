//! Common utilities for LLVM IR text generation.
//!
//! Shared between CPU and GPU backends.

mod ctx;
pub mod types;

pub use ctx::{PendingReduce, RenderContext};
pub use types::{addr_space_num, lcast, lconst, ldt};
