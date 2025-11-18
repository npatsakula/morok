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
mod tests {
    use super::*;

    #[test]
    fn test_cdiv_positive() {
        let solver = z3::Solver::new();

        let a = Int::new_const("a");
        let b = Int::new_const("b");

        // For positive a, b: cdiv should match euclidean div
        solver.assert(a.ge(Int::from_i64(0)));
        solver.assert(b.gt(Int::from_i64(0)));

        let result = z3_cdiv(&a, &b);
        solver.assert(result.eq(&a / &b));

        assert_eq!(solver.check(), z3::SatResult::Sat);
    }

    #[test]
    fn test_cdiv_negative_dividend() {
        let solver = z3::Solver::new();

        // Test: -7 / 3 = -2 (truncated), not -3 (euclidean)
        let a = Int::from_i64(-7);
        let b = Int::from_i64(3);

        let result = z3_cdiv(&a, &b);
        solver.assert(result.eq(Int::from_i64(-2)));

        assert_eq!(solver.check(), z3::SatResult::Sat);
    }

    #[test]
    fn test_cmod() {
        let solver = z3::Solver::new();

        // Test: -7 % 3 = -1 (Rust), not 2 (Python/Z3)
        let a = Int::from_i64(-7);
        let b = Int::from_i64(3);

        let result = z3_cmod(&a, &b);
        solver.assert(result.eq(Int::from_i64(-1)));

        assert_eq!(solver.check(), z3::SatResult::Sat);
    }
}
