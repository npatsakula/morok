//! Property-based tests for symbolic optimization and rangeify.
//!
//! Uses proptest to verify algebraic laws, metamorphic properties,
//! and multi-oracle correctness.

#[cfg(feature = "z3")]
mod oracles;
mod symbolic_meta;
mod symbolic_props;
