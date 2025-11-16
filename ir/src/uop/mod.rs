//! UOp (micro-operation) implementation.
//!
//! This module contains the UOp struct and all related functionality for creating
//! and manipulating operations in the IR.
//!
//! # Module Organization
//!
//! - [`core`] - UOp struct and fundamental operations
//! - [`hash_consing`] - Caching infrastructure for deduplication
//! - [`constructors`] - Helper methods for creating UOps

pub mod core;
pub mod hash_consing;
pub mod constructors;

// Re-export the main types
pub use core::{IntoUOp, UOp};
