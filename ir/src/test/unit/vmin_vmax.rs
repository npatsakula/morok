//! Unit tests for vmin/vmax range analysis.

use crate::{BinaryOp, ConstValue, Op, TernaryOp, UOp, UnaryOp};
use morok_dtype::DType;

// ============================================================================
// Test Constants
// ============================================================================

#[test]
fn test_vmin_vmax_const() {
    let five = UOp::const_(DType::Int32, ConstValue::Int(5));
    assert_eq!(five.vmin(), &ConstValue::Int(5));
    assert_eq!(five.vmax(), &ConstValue::Int(5));

    let neg_three = UOp::const_(DType::Int32, ConstValue::Int(-3));
    assert_eq!(neg_three.vmin(), &ConstValue::Int(-3));
    assert_eq!(neg_three.vmax(), &ConstValue::Int(-3));

    let pi = UOp::const_(DType::Float32, ConstValue::Float(3.14159));
    assert_eq!(pi.vmin(), &ConstValue::Float(3.14159));
    assert_eq!(pi.vmax(), &ConstValue::Float(3.14159));

    let bool_true = UOp::const_(DType::Bool, ConstValue::Bool(true));
    assert_eq!(bool_true.vmin(), &ConstValue::Bool(true));
    assert_eq!(bool_true.vmax(), &ConstValue::Bool(true));
}

// ============================================================================
// Test Arithmetic Operations
// ============================================================================

#[test]
fn test_vmin_vmax_add() {
    let a = UOp::const_(DType::Int32, ConstValue::Int(2));
    let b = UOp::const_(DType::Int32, ConstValue::Int(3));
    let sum = UOp::new(Op::Binary(BinaryOp::Add, a, b), DType::Int32);

    assert_eq!(sum.vmin(), &ConstValue::Int(5));
    assert_eq!(sum.vmax(), &ConstValue::Int(5));
}

#[test]
fn test_vmin_vmax_sub() {
    let a = UOp::const_(DType::Int32, ConstValue::Int(10));
    let b = UOp::const_(DType::Int32, ConstValue::Int(3));
    let diff = UOp::new(Op::Binary(BinaryOp::Sub, a, b), DType::Int32);

    assert_eq!(diff.vmin(), &ConstValue::Int(7));
    assert_eq!(diff.vmax(), &ConstValue::Int(7));
}

#[test]
fn test_vmin_vmax_mul() {
    let a = UOp::const_(DType::Int32, ConstValue::Int(-2));
    let b = UOp::const_(DType::Int32, ConstValue::Int(3));
    let prod = UOp::new(Op::Binary(BinaryOp::Mul, a, b), DType::Int32);

    assert_eq!(prod.vmin(), &ConstValue::Int(-6));
    assert_eq!(prod.vmax(), &ConstValue::Int(-6));
}

#[test]
fn test_vmin_vmax_mul_range() {
    // Test multiplication with ranges
    let a = UOp::new(Op::DefineVar { name: "a".to_string(), min_val: -2, max_val: 3 }, DType::Int32);
    let b = UOp::new(Op::DefineVar { name: "b".to_string(), min_val: -1, max_val: 4 }, DType::Int32);
    let prod = UOp::new(Op::Binary(BinaryOp::Mul, a, b), DType::Int32);

    // Check all 4 corners: -2*-1=2, -2*4=-8, 3*-1=-3, 3*4=12
    // Min is -8, max is 12
    assert_eq!(prod.vmin(), &ConstValue::Int(-8));
    assert_eq!(prod.vmax(), &ConstValue::Int(12));
}

#[test]
fn test_vmin_vmax_max() {
    let a = UOp::const_(DType::Int32, ConstValue::Int(5));
    let b = UOp::const_(DType::Int32, ConstValue::Int(10));
    let max_val = UOp::new(Op::Binary(BinaryOp::Max, a, b), DType::Int32);

    assert_eq!(max_val.vmin(), &ConstValue::Int(10));
    assert_eq!(max_val.vmax(), &ConstValue::Int(10));
}

#[test]
fn test_vmin_vmax_idiv() {
    let a = UOp::const_(DType::Int32, ConstValue::Int(15));
    let b = UOp::const_(DType::Int32, ConstValue::Int(3));
    let div = UOp::new(Op::Binary(BinaryOp::Idiv, a, b), DType::Int32);

    assert_eq!(div.vmin(), &ConstValue::Int(5));
    assert_eq!(div.vmax(), &ConstValue::Int(5));
}

#[test]
fn test_vmin_vmax_mod() {
    let a = UOp::const_(DType::Int32, ConstValue::Int(17));
    let b = UOp::const_(DType::Int32, ConstValue::Int(5));
    let modulo = UOp::new(Op::Binary(BinaryOp::Mod, a, b), DType::Int32);

    assert_eq!(modulo.vmin(), &ConstValue::Int(2));
    assert_eq!(modulo.vmax(), &ConstValue::Int(2));
}

// ============================================================================
// Test Unary Operations
// ============================================================================

#[test]
fn test_vmin_vmax_neg() {
    let five = UOp::const_(DType::Int32, ConstValue::Int(5));
    let neg = UOp::new(Op::Unary(UnaryOp::Neg, five), DType::Int32);

    assert_eq!(neg.vmin(), &ConstValue::Int(-5));
    assert_eq!(neg.vmax(), &ConstValue::Int(-5));
}

#[test]
fn test_vmin_vmax_neg_range() {
    let var = UOp::new(Op::DefineVar { name: "x".to_string(), min_val: -3, max_val: 5 }, DType::Int32);
    let neg = UOp::new(Op::Unary(UnaryOp::Neg, var), DType::Int32);

    // Negation flips the range
    assert_eq!(neg.vmin(), &ConstValue::Int(-5));
    assert_eq!(neg.vmax(), &ConstValue::Int(3));
}

// ============================================================================
// Test Comparison Operations
// ============================================================================

#[test]
fn test_vmin_vmax_cmplt() {
    let a = UOp::const_(DType::Int32, ConstValue::Int(5));
    let b = UOp::const_(DType::Int32, ConstValue::Int(10));
    let cmp = UOp::new(Op::Binary(BinaryOp::Lt, a, b), DType::Bool);

    // 5 < 10 is always true, so range is [true, true]
    assert_eq!(cmp.vmin(), &ConstValue::Bool(true));
    assert_eq!(cmp.vmax(), &ConstValue::Bool(true));
}

#[test]
fn test_vmin_vmax_eq() {
    let a = UOp::const_(DType::Int32, ConstValue::Int(5));
    let b = UOp::const_(DType::Int32, ConstValue::Int(5));
    let eq = UOp::new(Op::Binary(BinaryOp::Eq, a, b), DType::Bool);

    // 5 == 5 is always true, so range is [true, true]
    assert_eq!(eq.vmin(), &ConstValue::Bool(true));
    assert_eq!(eq.vmax(), &ConstValue::Bool(true));
}

// ============================================================================
// Test Bitwise Operations
// ============================================================================

#[test]
fn test_vmin_vmax_and_bool() {
    let a = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let b = UOp::const_(DType::Bool, ConstValue::Bool(false));
    let and = UOp::new(Op::Binary(BinaryOp::And, a, b), DType::Bool);

    // true & false = false
    assert_eq!(and.vmin(), &ConstValue::Bool(false));
    assert_eq!(and.vmax(), &ConstValue::Bool(false));
}

#[test]
fn test_vmin_vmax_or_bool() {
    let a = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let b = UOp::const_(DType::Bool, ConstValue::Bool(false));
    let or = UOp::new(Op::Binary(BinaryOp::Or, a, b), DType::Bool);

    // true | false = true
    assert_eq!(or.vmin(), &ConstValue::Bool(true));
    assert_eq!(or.vmax(), &ConstValue::Bool(true));
}

#[test]
fn test_vmin_vmax_and_int() {
    let a = UOp::const_(DType::Int32, ConstValue::Int(15)); // 0b1111
    let b = UOp::const_(DType::Int32, ConstValue::Int(7)); // 0b0111
    let and = UOp::new(Op::Binary(BinaryOp::And, a, b), DType::Int32);

    // 15 & 7 = 7
    assert_eq!(and.vmin(), &ConstValue::Int(7));
    assert_eq!(and.vmax(), &ConstValue::Int(7));
}

#[test]
fn test_vmin_vmax_shl() {
    let a = UOp::const_(DType::Int32, ConstValue::Int(3));
    let b = UOp::const_(DType::Int32, ConstValue::Int(2));
    let shl = UOp::new(Op::Binary(BinaryOp::Shl, a, b), DType::Int32);

    // 3 << 2 = 12
    assert_eq!(shl.vmin(), &ConstValue::Int(12));
    assert_eq!(shl.vmax(), &ConstValue::Int(12));
}

#[test]
fn test_vmin_vmax_shr() {
    let a = UOp::const_(DType::Int32, ConstValue::Int(12));
    let b = UOp::const_(DType::Int32, ConstValue::Int(2));
    let shr = UOp::new(Op::Binary(BinaryOp::Shr, a, b), DType::Int32);

    // 12 >> 2 = 3
    assert_eq!(shr.vmin(), &ConstValue::Int(3));
    assert_eq!(shr.vmax(), &ConstValue::Int(3));
}

// ============================================================================
// Test Special Operations
// ============================================================================

#[test]
fn test_vmin_vmax_define_var() {
    let var = UOp::new(Op::DefineVar { name: "x".to_string(), min_val: -10, max_val: 20 }, DType::Int32);

    assert_eq!(var.vmin(), &ConstValue::Int(-10));
    assert_eq!(var.vmax(), &ConstValue::Int(20));
}

#[test]
fn test_vmin_vmax_range() {
    let end = UOp::const_(DType::Int32, ConstValue::Int(10));
    let range = UOp::new(Op::Range { end, axis_id: 0, axis_type: crate::types::AxisType::Loop }, DType::Int32);

    // RANGE goes from 0 to end-1
    assert_eq!(range.vmin(), &ConstValue::Int(0));
    assert_eq!(range.vmax(), &ConstValue::Int(9));
}

#[test]
fn test_vmin_vmax_cast() {
    let float_val = UOp::const_(DType::Float32, ConstValue::Float(5.7));
    let int_val = UOp::new(Op::Cast { src: float_val.clone(), dtype: DType::Float32 }, DType::Int32);

    // Cast from 5.7 to int = 5
    assert_eq!(int_val.vmin(), &ConstValue::Int(5));
    assert_eq!(int_val.vmax(), &ConstValue::Int(5));
}

#[test]
fn test_vmin_vmax_cast_range() {
    let var = UOp::new(Op::DefineVar { name: "x".to_string(), min_val: -1000, max_val: 1000 }, DType::Int32);
    // Cast to Int8 which has range [-128, 127]
    let casted = UOp::new(Op::Cast { src: var.clone(), dtype: DType::Int32 }, DType::Int8);

    // Should be clamped to Int8 bounds
    assert_eq!(casted.vmin(), &ConstValue::Int(-128));
    assert_eq!(casted.vmax(), &ConstValue::Int(127));
}

// ============================================================================
// Test Ternary Operations
// ============================================================================

#[test]
fn test_vmin_vmax_where_true() {
    let cond = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let true_val = UOp::const_(DType::Int32, ConstValue::Int(10));
    let false_val = UOp::const_(DType::Int32, ConstValue::Int(5));
    let where_op = UOp::new(Op::Ternary(TernaryOp::Where, cond, true_val, false_val), DType::Int32);

    // Condition is always true, so result is true_val
    assert_eq!(where_op.vmin(), &ConstValue::Int(10));
    assert_eq!(where_op.vmax(), &ConstValue::Int(10));
}

#[test]
fn test_vmin_vmax_where_false() {
    let cond = UOp::const_(DType::Bool, ConstValue::Bool(false));
    let true_val = UOp::const_(DType::Int32, ConstValue::Int(10));
    let false_val = UOp::const_(DType::Int32, ConstValue::Int(5));
    let where_op = UOp::new(Op::Ternary(TernaryOp::Where, cond, true_val, false_val), DType::Int32);

    // Condition is always false, so result is false_val
    assert_eq!(where_op.vmin(), &ConstValue::Int(5));
    assert_eq!(where_op.vmax(), &ConstValue::Int(5));
}

#[test]
fn test_vmin_vmax_where_range() {
    // Condition can be either true or false
    let cond = UOp::new(Op::DefineVar { name: "cond".to_string(), min_val: 0, max_val: 1 }, DType::Bool);
    let true_val = UOp::const_(DType::Int32, ConstValue::Int(10));
    let false_val = UOp::const_(DType::Int32, ConstValue::Int(5));
    let where_op = UOp::new(Op::Ternary(TernaryOp::Where, cond, true_val, false_val), DType::Int32);

    // Could be either branch
    assert_eq!(where_op.vmin(), &ConstValue::Int(5));
    assert_eq!(where_op.vmax(), &ConstValue::Int(10));
}

#[test]
fn test_vmin_vmax_mulacc() {
    let a = UOp::const_(DType::Int32, ConstValue::Int(3));
    let b = UOp::const_(DType::Int32, ConstValue::Int(4));
    let c = UOp::const_(DType::Int32, ConstValue::Int(5));
    let mulacc = UOp::new(Op::Ternary(TernaryOp::MulAcc, a, b, c), DType::Int32);

    // 3 * 4 + 5 = 17
    assert_eq!(mulacc.vmin(), &ConstValue::Int(17));
    assert_eq!(mulacc.vmax(), &ConstValue::Int(17));
}

// ============================================================================
// Test Complex Expressions
// ============================================================================

#[test]
fn test_vmin_vmax_complex_expression() {
    // Test: (x + 5) * 2 where x in [0, 10]
    let x = UOp::new(Op::DefineVar { name: "x".to_string(), min_val: 0, max_val: 10 }, DType::Int32);
    let five = UOp::const_(DType::Int32, ConstValue::Int(5));
    let two = UOp::const_(DType::Int32, ConstValue::Int(2));

    let x_plus_5 = UOp::new(Op::Binary(BinaryOp::Add, x, five), DType::Int32);
    let result = UOp::new(Op::Binary(BinaryOp::Mul, x_plus_5, two), DType::Int32);

    // x in [0, 10] => x+5 in [5, 15] => (x+5)*2 in [10, 30]
    assert_eq!(result.vmin(), &ConstValue::Int(10));
    assert_eq!(result.vmax(), &ConstValue::Int(30));
}

#[test]
fn test_vmin_vmax_nested_max() {
    // Test: max(max(a, b), c) where a=3, b=7, c=5
    let a = UOp::const_(DType::Int32, ConstValue::Int(3));
    let b = UOp::const_(DType::Int32, ConstValue::Int(7));
    let c = UOp::const_(DType::Int32, ConstValue::Int(5));

    let max_ab = UOp::new(Op::Binary(BinaryOp::Max, a, b), DType::Int32);
    let max_abc = UOp::new(Op::Binary(BinaryOp::Max, max_ab, c), DType::Int32);

    assert_eq!(max_abc.vmin(), &ConstValue::Int(7));
    assert_eq!(max_abc.vmax(), &ConstValue::Int(7));
}

// ============================================================================
// Test Float Operations
// ============================================================================

#[test]
fn test_vmin_vmax_float_ops() {
    let a = UOp::const_(DType::Float32, ConstValue::Float(2.5));
    let b = UOp::const_(DType::Float32, ConstValue::Float(1.5));

    let sum = UOp::new(Op::Binary(BinaryOp::Add, a.clone(), b.clone()), DType::Float32);
    assert_eq!(sum.vmin(), &ConstValue::Float(4.0));
    assert_eq!(sum.vmax(), &ConstValue::Float(4.0));

    let diff = UOp::new(Op::Binary(BinaryOp::Sub, a.clone(), b.clone()), DType::Float32);
    assert_eq!(diff.vmin(), &ConstValue::Float(1.0));
    assert_eq!(diff.vmax(), &ConstValue::Float(1.0));

    let prod = UOp::new(Op::Binary(BinaryOp::Mul, a.clone(), b.clone()), DType::Float32);
    assert_eq!(prod.vmin(), &ConstValue::Float(3.75));
    assert_eq!(prod.vmax(), &ConstValue::Float(3.75));

    let div = UOp::new(Op::Binary(BinaryOp::Fdiv, a.clone(), b.clone()), DType::Float32);
    // 2.5 / 1.5 = 1.666...
    if let ConstValue::Float(min_val) = div.vmin() {
        assert!((min_val - 1.6666666666666667).abs() < 1e-10);
    } else {
        panic!("Expected float result");
    }
}

// ============================================================================
// Test Edge Cases
// ============================================================================

#[test]
fn test_vmin_vmax_division_by_zero_range() {
    // Test division when divisor range includes zero
    let a = UOp::const_(DType::Int32, ConstValue::Int(10));
    let b = UOp::new(
        Op::DefineVar {
            name: "b".to_string(),
            min_val: -1,
            max_val: 1, // Includes zero!
        },
        DType::Int32,
    );
    let div = UOp::new(Op::Binary(BinaryOp::Idiv, a, b), DType::Int32);

    // Division by zero range returns dtype bounds
    assert_eq!(div.vmin(), &ConstValue::Int(i32::MIN as i64));
    assert_eq!(div.vmax(), &ConstValue::Int(i32::MAX as i64));
}

#[test]
fn test_vmin_vmax_mod_by_zero_range() {
    // Test modulo when divisor range includes zero
    let a = UOp::const_(DType::Int32, ConstValue::Int(10));
    let b = UOp::new(
        Op::DefineVar {
            name: "b".to_string(),
            min_val: -1,
            max_val: 1, // Includes zero!
        },
        DType::Int32,
    );
    let modulo = UOp::new(Op::Binary(BinaryOp::Mod, a, b), DType::Int32);

    // Modulo by zero range returns dtype bounds
    assert_eq!(modulo.vmin(), &ConstValue::Int(i32::MIN as i64));
    assert_eq!(modulo.vmax(), &ConstValue::Int(i32::MAX as i64));
}

#[test]
fn test_vmin_vmax_shift_overflow() {
    let a = UOp::const_(DType::Int32, ConstValue::Int(1));
    let b = UOp::const_(DType::Int32, ConstValue::Int(64)); // Shift by 64 or more
    let shl = UOp::new(Op::Binary(BinaryOp::Shl, a, b), DType::Int32);

    // Shift by >= 64 returns dtype bounds
    assert_eq!(shl.vmin(), &ConstValue::Int(i32::MIN as i64));
    assert_eq!(shl.vmax(), &ConstValue::Int(i32::MAX as i64));
}
