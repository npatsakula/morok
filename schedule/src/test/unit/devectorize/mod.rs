//! Devectorizer test suite (devectorize.rs).
//!
//! Comprehensive unit tests for the devectorize pass, which transforms
//! vectorized memory operations into contiguous vector loads/stores.
//!
//! Unlike Tinygrad which tests devectorizer indirectly through full pipeline,
//! we create dedicated unit tests for each phase and pattern.
//!
//! # Test Organization
//!
//! - `helpers`: Test builders and assertion helpers
//! - `expand_index`: Phase 1 - expand_vector_index tests
//! - `load_store`: Phase 2 - PTRCAT distribution and split tests
//! - `bool_storage`: Phase 3 - bool->uint8 conversion tests
//! - `gep_patterns`: GEP/CAT/VECTORIZE pattern tests
//! - `alu_devectorization`: no_vectorized_alu tests
//! - `pipeline`: End-to-end devectorize() tests
//! - `edge_cases`: Corner cases and regression tests

pub mod alu_devectorization;
pub mod bool_storage;
pub mod edge_cases;
pub mod expand_index;
pub mod gep_patterns;
pub mod helpers;
pub mod load_store;
pub mod new_patterns;
pub mod pipeline;
