//! Unit tests for vmin/vmax range analysis.

use std::f32::consts::PI;

use crate::{AxisId, ConstValue, Op, UOp};
use morok_dtype::DType;

// ============================================================================
// Test Constants
// ============================================================================

#[test]
fn test_vmin_vmax_const() {
    assert_eq!(UOp::native_const(5i32).vmin(), &ConstValue::Int(5));
    assert_eq!(UOp::native_const(5i32).vmax(), &ConstValue::Int(5));

    assert_eq!(UOp::native_const(-3i32).vmin(), &ConstValue::Int(-3));
    assert_eq!(UOp::native_const(-3i32).vmax(), &ConstValue::Int(-3));

    assert_eq!(UOp::native_const(PI).vmin(), &ConstValue::Float(PI as f64));
    assert_eq!(UOp::native_const(PI).vmax(), &ConstValue::Float(PI as f64));

    assert_eq!(UOp::native_const(true).vmin(), &ConstValue::Bool(true));
    assert_eq!(UOp::native_const(true).vmax(), &ConstValue::Bool(true));
}

// ============================================================================
// Test Arithmetic Operations
// ============================================================================

#[test]
fn test_vmin_vmax_add() {
    let sum = UOp::native_const(2i32).try_add(&UOp::native_const(3i32)).unwrap();

    assert_eq!(sum.vmin(), &ConstValue::Int(5));
    assert_eq!(sum.vmax(), &ConstValue::Int(5));
}

#[test]
fn test_vmin_vmax_sub() {
    let a = UOp::native_const(10i32);
    let b = UOp::native_const(3i32);
    let diff = a.try_sub(&b).unwrap();

    assert_eq!(diff.vmin(), &ConstValue::Int(7));
    assert_eq!(diff.vmax(), &ConstValue::Int(7));
}

#[test]
fn test_vmin_vmax_mul() {
    let a = UOp::native_const(-2i32);
    let b = UOp::native_const(3i32);
    let prod = a.try_mul(&b).unwrap();

    assert_eq!(prod.vmin(), &ConstValue::Int(-6));
    assert_eq!(prod.vmax(), &ConstValue::Int(-6));
}

#[test]
fn test_vmin_vmax_mul_range() {
    // Test multiplication with ranges
    let a = UOp::define_var("a".to_string(), 0, 3);
    let b = UOp::define_var("b".to_string(), 0, 4);
    let prod = a.try_mul(&b).unwrap();

    // a ∈ [0, 3], b ∈ [0, 4]
    // Check all 4 corners: 0*0=0, 0*4=0, 3*0=0, 3*4=12
    // Min is 0, max is 12
    assert_eq!(prod.vmin(), &ConstValue::Int(0));
    assert_eq!(prod.vmax(), &ConstValue::Int(12));
}

#[test]
fn test_vmin_vmax_max() {
    let a = UOp::native_const(5i32);
    let b = UOp::native_const(10i32);
    let max_val = a.try_max(&b).unwrap();

    assert_eq!(max_val.vmin(), &ConstValue::Int(10));
    assert_eq!(max_val.vmax(), &ConstValue::Int(10));
}

#[test]
fn test_vmin_vmax_idiv() {
    let a = UOp::native_const(15i32);
    let b = UOp::native_const(3i32);
    let div = a.try_div(&b).unwrap();

    assert_eq!(div.vmin(), &ConstValue::Int(5));
    assert_eq!(div.vmax(), &ConstValue::Int(5));
}

#[test]
fn test_vmin_vmax_mod() {
    let a = UOp::native_const(17i32);
    let b = UOp::native_const(5i32);
    let modulo = a.try_mod(&b).unwrap();

    assert_eq!(modulo.vmin(), &ConstValue::Int(2));
    assert_eq!(modulo.vmax(), &ConstValue::Int(2));
}

// ============================================================================
// Test Unary Operations
// ============================================================================

#[test]
fn test_vmin_vmax_neg() {
    let five = UOp::native_const(5i32);
    let neg = five.neg();

    assert_eq!(neg.vmin(), &ConstValue::Int(-5));
    assert_eq!(neg.vmax(), &ConstValue::Int(-5));
}

#[test]
fn test_vmin_vmax_neg_range() {
    let var = UOp::define_var("x".to_string(), 0, 5);
    let neg = var.neg();

    // var ∈ [0, 5], so neg(var) ∈ [-5, 0]
    assert_eq!(neg.vmin(), &ConstValue::Int(-5));
    assert_eq!(neg.vmax(), &ConstValue::Int(0));
}

// ============================================================================
// Test Comparison Operations
// ============================================================================

#[test]
fn test_vmin_vmax_cmplt() {
    let a = UOp::native_const(5i32);
    let b = UOp::native_const(10i32);
    let cmp = a.try_cmplt(&b).unwrap();

    // 5 < 10 is always true, so range is [true, true]
    assert_eq!(cmp.vmin(), &ConstValue::Bool(true));
    assert_eq!(cmp.vmax(), &ConstValue::Bool(true));
}

#[test]
fn test_vmin_vmax_eq() {
    let a = UOp::native_const(5i32);
    let b = UOp::native_const(5i32);
    let eq = a.try_cmpeq(&b).unwrap();

    // 5 == 5 is always true, so range is [true, true]
    assert_eq!(eq.vmin(), &ConstValue::Bool(true));
    assert_eq!(eq.vmax(), &ConstValue::Bool(true));
}

// ============================================================================
// Test Bitwise Operations
// ============================================================================

#[test]
fn test_vmin_vmax_and_bool() {
    let and = UOp::native_const(true).try_and_op(&UOp::native_const(false)).unwrap();

    // true & false = false
    assert_eq!(and.vmin(), &ConstValue::Bool(false));
    assert_eq!(and.vmax(), &ConstValue::Bool(false));
}

#[test]
fn test_vmin_vmax_or_bool() {
    let or = UOp::native_const(true).try_or_op(&UOp::native_const(false)).unwrap();

    // true | false = true
    assert_eq!(or.vmin(), &ConstValue::Bool(true));
    assert_eq!(or.vmax(), &ConstValue::Bool(true));
}

#[test]
fn test_vmin_vmax_and_int() {
    let a = UOp::native_const(15i32); // 0b1111
    let b = UOp::native_const(7i32); // 0b0111
    let and = a.try_and_op(&b).unwrap();

    // 15 & 7 = 7
    assert_eq!(and.vmin(), &ConstValue::Int(7));
    assert_eq!(and.vmax(), &ConstValue::Int(7));
}

#[test]
fn test_vmin_vmax_shl() {
    let a = UOp::native_const(3i32);
    let b = UOp::native_const(2i32);
    let shl = a.try_shl_op(&b).unwrap();

    // 3 << 2 = 12
    assert_eq!(shl.vmin(), &ConstValue::Int(12));
    assert_eq!(shl.vmax(), &ConstValue::Int(12));
}

#[test]
fn test_vmin_vmax_shr() {
    let a = UOp::native_const(12i32);
    let b = UOp::native_const(2i32);
    let shr = a.try_shr_op(&b).unwrap();

    // 12 >> 2 = 3
    assert_eq!(shr.vmin(), &ConstValue::Int(3));
    assert_eq!(shr.vmax(), &ConstValue::Int(3));
}

// ============================================================================
// Test Special Operations
// ============================================================================

#[test]
fn test_vmin_vmax_define_var() {
    let var = UOp::define_var("x".to_string(), 0, 20);

    assert_eq!(var.vmin(), &ConstValue::Int(0));
    assert_eq!(var.vmax(), &ConstValue::Int(20));
}

#[test]
fn test_vmin_vmax_define_var_with_min() {
    // Test variable with non-zero min_val
    let var = UOp::define_var("x".to_string(), 5, 20);

    assert_eq!(var.vmin(), &ConstValue::Int(5));
    assert_eq!(var.vmax(), &ConstValue::Int(20));
}

#[test]
fn test_vmin_vmax_range() {
    let end = UOp::native_const(10i32);
    let range = UOp::new(
        Op::Range { end, axis_id: AxisId::Renumbered(0), axis_type: crate::types::AxisType::Loop },
        DType::Int32,
    );

    // RANGE goes from 0 to end-1
    assert_eq!(range.vmin(), &ConstValue::Int(0));
    assert_eq!(range.vmax(), &ConstValue::Int(9));
}

#[test]
fn test_vmin_vmax_cast() {
    let float_val = UOp::native_const(5.7f32);
    let int_val = UOp::cast(float_val.clone(), DType::Int32);

    // Cast from 5.7 to int = 5
    assert_eq!(int_val.vmin(), &ConstValue::Int(5));
    assert_eq!(int_val.vmax(), &ConstValue::Int(5));
}

#[test]
fn test_vmin_vmax_cast_range() {
    let var = UOp::define_var("x".to_string(), 0, 1000);
    // Cast to Int8 which has range [-128, 127]
    let casted = UOp::cast(var.clone(), DType::Int8);

    // Should be clamped to Int8 bounds
    assert_eq!(casted.vmin(), &ConstValue::Int(0));
    assert_eq!(casted.vmax(), &ConstValue::Int(127));
}

// ============================================================================
// Test Ternary Operations
// ============================================================================

#[test]
fn test_vmin_vmax_where_true() {
    let where_op = UOp::try_where(UOp::native_const(true), UOp::native_const(10i32), UOp::native_const(5i32)).unwrap();

    // Condition is always true, so result is true_val
    assert_eq!(where_op.vmin(), &ConstValue::Int(10));
    assert_eq!(where_op.vmax(), &ConstValue::Int(10));
}

#[test]
fn test_vmin_vmax_where_false() {
    let where_op = UOp::try_where(UOp::native_const(false), UOp::native_const(10i32), UOp::native_const(5i32)).unwrap();

    // Condition is always false, so result is false_val
    assert_eq!(where_op.vmin(), &ConstValue::Int(5));
    assert_eq!(where_op.vmax(), &ConstValue::Int(5));
}

#[test]
fn test_vmin_vmax_where_range() {
    // Condition can be either true or false
    let cond = UOp::define_var("cond".to_string(), 0, 1);
    let true_val = UOp::native_const(10i32);
    let false_val = UOp::native_const(5i32);
    let where_op = UOp::try_where(cond, true_val, false_val).unwrap();

    // Could be either branch
    assert_eq!(where_op.vmin(), &ConstValue::Int(5));
    assert_eq!(where_op.vmax(), &ConstValue::Int(10));
}

#[test]
fn test_vmin_vmax_mulacc() {
    let a = UOp::native_const(3i32);
    let b = UOp::native_const(4i32);
    let c = UOp::native_const(5i32);
    let mulacc = UOp::try_mulacc(a, b, c).unwrap();

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
    let x = UOp::var("x", DType::Int32, 0, 10);
    let five = UOp::native_const(5i32);
    let two = UOp::native_const(2i32);

    let x_plus_5 = x.try_add(&five).unwrap();
    let result = x_plus_5.try_mul(&two).unwrap();

    // x in [0, 10] => x+5 in [5, 15] => (x+5)*2 in [10, 30]
    assert_eq!(result.vmin(), &ConstValue::Int(10));
    assert_eq!(result.vmax(), &ConstValue::Int(30));
}

#[test]
fn test_vmin_vmax_nested_max() {
    // Test: max(max(a, b), c) where a=3, b=7, c=5
    let a = UOp::native_const(3i32);
    let b = UOp::native_const(7i32);
    let c = UOp::native_const(5i32);

    let max_ab = a.try_max(&b).unwrap();
    let max_abc = max_ab.try_max(&c).unwrap();

    assert_eq!(max_abc.vmin(), &ConstValue::Int(7));
    assert_eq!(max_abc.vmax(), &ConstValue::Int(7));
}

// ============================================================================
// Test Float Operations
// ============================================================================

#[test]
fn test_vmin_vmax_float_ops() {
    let a = UOp::native_const(2.5f32);
    let b = UOp::native_const(1.5f32);

    let sum = a.try_add(&b).unwrap();
    assert_eq!(sum.vmin(), &ConstValue::Float(4.0));
    assert_eq!(sum.vmax(), &ConstValue::Float(4.0));

    let diff = a.try_sub(&b).unwrap();
    assert_eq!(diff.vmin(), &ConstValue::Float(1.0));
    assert_eq!(diff.vmax(), &ConstValue::Float(1.0));

    let prod = a.try_mul(&b).unwrap();
    assert_eq!(prod.vmin(), &ConstValue::Float(3.75));
    assert_eq!(prod.vmax(), &ConstValue::Float(3.75));

    let div = a.try_div(&b).unwrap();
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
    let a = UOp::native_const(10i32);
    let b = UOp::var("b", DType::Int32, 0, 1); // Includes zero!
    let div = a.try_div(&b).unwrap();

    // Division by zero range returns dtype bounds
    assert_eq!(div.vmin(), &ConstValue::Int(i32::MIN as i64));
    assert_eq!(div.vmax(), &ConstValue::Int(i32::MAX as i64));
}

#[test]
fn test_vmin_vmax_mod_by_zero_range() {
    // Test modulo when divisor range includes zero
    let a = UOp::native_const(10i32);
    let b = UOp::var("b", DType::Int32, 0, 1); // Includes zero!
    let modulo = a.try_mod(&b).unwrap();

    // Modulo by zero range returns dtype bounds
    assert_eq!(modulo.vmin(), &ConstValue::Int(i32::MIN as i64));
    assert_eq!(modulo.vmax(), &ConstValue::Int(i32::MAX as i64));
}

#[test]
fn test_vmin_vmax_shift_overflow() {
    let a = UOp::native_const(1i32);
    let b = UOp::native_const(64i32); // Shift by 64 or more
    let shl = a.try_shl_op(&b).unwrap();

    // Shift by >= 64 returns dtype bounds
    assert_eq!(shl.vmin(), &ConstValue::Int(i32::MIN as i64));
    assert_eq!(shl.vmax(), &ConstValue::Int(i32::MAX as i64));
}
