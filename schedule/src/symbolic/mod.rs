//! Symbolic simplification patterns.
//!
//! This module provides symbolic simplification patterns for UOp graphs,
//! including identity folding, constant folding, and algebraic simplification.

pub mod dce;
pub mod patterns;

pub use patterns::{symbolic, symbolic_simple};
