use super::*;
use morok_dtype::DType;

use morok_ir::types::ConstValue;

#[test]
fn test_verify_identity_add_zero() {
    // x + 0 = x
    let x = UOp::var("x", DType::Int32, 0, 100);
    let zero = UOp::const_(DType::Int32, ConstValue::Int(0));
    let x_plus_zero = x.try_add(&zero).unwrap();

    verify_equivalence(&x_plus_zero, &x).expect("x + 0 should equal x");
}

#[test]
fn test_verify_commutativity() {
    // x + y = y + x
    let x = UOp::var("x", DType::Int32, 0, 100);
    let y = UOp::var("y", DType::Int32, 0, 100);
    let x_plus_y = x.try_add(&y).unwrap();
    let y_plus_x = y.try_add(&x).unwrap();

    verify_equivalence(&x_plus_y, &y_plus_x).expect("x + y should equal y + x");
}

#[test]
fn test_verify_detect_inequality() {
    // x + 1 ≠ x (should find counterexample)
    let x = UOp::var("x", DType::Int32, 0, 100);
    let one = UOp::const_(DType::Int32, ConstValue::Int(1));
    let x_plus_one = x.try_add(&one).unwrap();

    let result = verify_equivalence(&x_plus_one, &x);
    assert!(result.is_err(), "x + 1 should not equal x");

    if let Err(CounterExample::Found { message, model }) = result {
        tracing::debug!(message = %message, model = %model, "z3 counterexample found");
    }
}

#[test]
fn test_verify_self_folding() {
    // x - x = 0
    let x = UOp::var("x", DType::Int32, 0, 100);
    let x_minus_x = x.try_sub(&x).unwrap();
    let zero = UOp::const_(DType::Int32, ConstValue::Int(0));

    verify_equivalence(&x_minus_x, &zero).expect("x - x should equal 0");
}
