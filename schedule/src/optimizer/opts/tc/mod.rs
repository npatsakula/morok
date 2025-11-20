//! Tensor Core (TC) optimization - Hardware-accelerated matrix multiplication.
//!
//! Implements pattern matching, selection, and transformation for tensor core operations.
//! Tensor cores are specialized hardware units for accelerating matrix multiplications,
//! available on NVIDIA (WMMA), AMD (Matrix Cores), Intel, and Apple (AMX) hardware.
//!
//! # Architecture
//!
//! The TC optimization pipeline has four stages:
//!
//! 1. **Pattern Matching** (`matching`) - Detect matmul patterns (REDUCE(ADD, MUL(A, B)))
//! 2. **Selection** (`selection`) - Filter tensor cores by dtype and choose best match
//! 3. **Swizzle** (`swizzle`) - Apply data layout permutations for optimal access
//! 4. **Application** (`apply`) - Transform AST with WMMA construction
//!
//! # Usage
//!
//! ```ignore
//! use morok_schedule::optimizer::opts::tc;
//!
//! // Apply tensor core optimization
//! tc::apply(scheduler, tc_select, tc_opt, use_tensor_cores)?;
//! ```

pub mod apply;
pub mod matching;
pub mod selection;
pub mod swizzle;

pub use apply::apply;
