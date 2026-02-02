//! Symbolic simplification patterns.
//!
//! This module provides symbolic simplification patterns for UOp graphs,
//! including identity folding, constant folding, and algebraic simplification.

pub mod dce;
pub mod fast_div;
pub mod patterns;

pub use fast_div::fast_division_patterns;
pub use patterns::{symbolic, symbolic_simple};
