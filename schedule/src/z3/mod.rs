//! Z3/SMT verification for symbolic optimizations.
//!
//! This module provides Z3-based verification that pattern rewrites preserve semantics.
//! It follows Tinygrad's architecture but adapted for Rust idioms.

pub mod alu;
pub mod convert;
pub mod verify;

pub use verify::{CounterExample, verify_equivalence};

/// Check if Z3 is available and working.
pub fn is_z3_available() -> bool {
    // Try to create a simple Z3 solver
    std::panic::catch_unwind(|| {
        let _solver = z3::Solver::new();
    })
    .is_ok()
}
