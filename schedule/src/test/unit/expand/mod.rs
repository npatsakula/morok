//! Pre-expander test suite (expand.rs).
//!
//! Tests for the UNROLL/CONTRACT expansion system, ported from Tinygrad's
//! TestExpander class (test_uop_graph.py:663-811).

pub mod do_contract;
pub mod do_expand;
pub mod edge_cases;
pub mod fix_reduce;
pub mod fix_store;
pub mod helpers;
pub mod swizzle;
