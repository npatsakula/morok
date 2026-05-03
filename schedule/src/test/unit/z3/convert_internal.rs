use super::*;

#[test]
fn test_convert_const_int() {
    let mut z3ctx = Z3Context::new();

    let uop = UOp::const_(DType::Int32, ConstValue::Int(42));
    let z3_expr = z3ctx.convert_uop(&uop).expect("Should convert");

    assert!(z3_expr.as_int().is_some());
}

#[test]
fn test_convert_simple_add() {
    let mut z3ctx = Z3Context::new();

    let a = UOp::const_(DType::Int32, ConstValue::Int(10));
    let b = UOp::const_(DType::Int32, ConstValue::Int(20));
    let add = a.try_add(&b).expect("Should create ADD");

    let z3_expr = z3ctx.convert_uop(&add).expect("Should convert");
    assert!(z3_expr.as_int().is_some());
}

#[test]
fn test_convert_variable() {
    let mut z3ctx = Z3Context::new();

    let var = UOp::var("x", DType::Int32, 0, 100);
    let z3_expr = z3ctx.convert_uop(&var).expect("Should convert");

    assert!(z3_expr.as_int().is_some());

    // Solver should have constraints for variable bounds
    assert_eq!(z3ctx.solver.check(), z3::SatResult::Sat);
}
