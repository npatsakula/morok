//! Backend-agnostic IR transformation passes.
//!
//! This module contains transformations that normalize the IR before
//! code generation, eliminating backend-specific logic from codegen.
//!
//! # Passes
//!
//! - [`linearize_index`] - Multi-index → single linear offset normalization
//!
//! These passes run during post-optimization, after rangeify and before codegen.

pub mod linearize_index;

#[cfg(test)]
pub(crate) use linearize_index::pm_linearize_multi_index;
pub use linearize_index::{build_linear_index, compute_row_major_strides, count_divmod, extract_index_dimension};
