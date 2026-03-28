//! Symbolic simplification patterns.
//!
//! This module provides symbolic simplification patterns for UOp graphs,
//! including identity folding, constant folding, and algebraic simplification.

pub mod dce;
pub mod divmod;
pub mod fast_div;
pub mod index_lowering;
pub mod patterns;
pub mod valid_simplification;

pub use fast_div::fast_division_patterns;
pub use index_lowering::pm_lower_index_dtype;
pub use patterns::{symbolic, symbolic_simple};
pub use valid_simplification::pm_simplify_valid;
