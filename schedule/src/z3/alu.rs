//! Custom ALU operations for Z3.
//!
//! Handles semantic differences between Rust/C operations and Z3's default operations.

use z3::ast::{Bool, Int};

/// C-style truncated division (matches Rust's default).
///
/// Z3's default division is Euclidean (always rounds toward negative infinity),
/// but Rust/C use truncated division (rounds toward zero).
///
/// Formula:
/// - If a < 0 and b > 0: (a + (b-1)) / b
/// - If a < 0 and b < 0: (a - (b+1)) / b
/// - Otherwise: a / b
pub fn z3_cdiv(a: &Int, b: &Int) -> Int {
    let zero = Int::from_i64(0);

    Bool::ite(&a.lt(&zero), &Bool::ite(&zero.lt(b), &((a + (b - 1)) / b), &((a - (b + 1)) / b)), &(a / b))
}

/// C-style modulo (matches Rust's default).
///
/// Defined as: a - cdiv(a, b) * b
pub fn z3_cmod(a: &Int, b: &Int) -> Int {
    a - z3_cdiv(a, b) * b
}

#[cfg(test)]
#[path = "../test/unit/z3/alu_internal.rs"]
mod tests;
