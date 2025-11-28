//! Property-based tests for IR operations.
//!
//! Uses proptest to verify invariants across wide input spaces.

#[cfg(test)]
mod dtype_props;

pub mod generators;
pub mod shrinking;
