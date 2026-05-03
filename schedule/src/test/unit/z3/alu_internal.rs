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
