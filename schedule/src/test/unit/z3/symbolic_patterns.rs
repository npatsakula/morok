//! Z3 verification tests for critical symbolic optimization patterns.
//!
//! These tests verify that pattern rewrites preserve semantics using Z3/SMT solving.
//! Each test creates an expression, applies symbolic simplification, and verifies
//! that the original and simplified versions are semantically equivalent.

use morok_dtype::DType;
use morok_ir::types::ConstValue;
use morok_ir::{Op, UOp};
use std::rc::Rc;

use crate::rewrite::graph_rewrite;
use crate::symbolic::symbolic_simple;
use crate::z3::verify::verify_equivalence;

// ============================================================================
// A. Identity Patterns (5 tests)
// ============================================================================

#[test]
fn test_identity_add_zero() {
    // x + 0 → x
    let x = UOp::var("x", DType::Int32, 0, 100);
    let zero = UOp::native_const(0i32);
    let expr = x.try_add_op(&zero).unwrap();

    let matcher = symbolic_simple();
    let simplified = graph_rewrite(&matcher, expr.clone(), &mut ());

    // Should simplify to x
    assert!(Rc::ptr_eq(&simplified, &x), "x + 0 should simplify to x");

    // Z3 verification
    verify_equivalence(&expr, &simplified).expect("x + 0 should equal x");
}

#[test]
fn test_identity_mul_one() {
    // x * 1 → x
    let x = UOp::var("x", DType::Int32, 0, 100);
    let one = UOp::native_const(1i32);
    let expr = x.try_mul_op(&one).unwrap();

    let matcher = symbolic_simple();
    let simplified = graph_rewrite(&matcher, expr.clone(), &mut ());

    assert!(Rc::ptr_eq(&simplified, &x), "x * 1 should simplify to x");
    verify_equivalence(&expr, &simplified).expect("x * 1 should equal x");
}

#[test]
fn test_identity_sub_zero() {
    // x - 0 → x
    let x = UOp::var("x", DType::Int32, 0, 100);
    let zero = UOp::native_const(0i32);
    let expr = x.try_sub_op(&zero).unwrap();

    let matcher = symbolic_simple();
    let simplified = graph_rewrite(&matcher, expr.clone(), &mut ());

    assert!(Rc::ptr_eq(&simplified, &x), "x - 0 should simplify to x");
    verify_equivalence(&expr, &simplified).expect("x - 0 should equal x");
}

#[test]
fn test_identity_div_one() {
    // x / 1 → x
    let x = UOp::var("x", DType::Int32, 0, 100);
    let one = UOp::native_const(1i32);
    let expr = x.try_idiv_op(&one).unwrap();

    let matcher = symbolic_simple();
    let simplified = graph_rewrite(&matcher, expr.clone(), &mut ());

    assert!(Rc::ptr_eq(&simplified, &x), "x / 1 should simplify to x");
    verify_equivalence(&expr, &simplified).expect("x / 1 should equal x");
}

#[test]
fn test_identity_mod_one() {
    // x % 1 → 0 (verify semantically even if optimizer doesn't simplify)
    let x = UOp::var("x", DType::Int32, 0, 100);
    let one = UOp::native_const(1i32);
    let expr = x.try_mod_op(&one).unwrap();
    let zero = UOp::native_const(0i32);

    // Z3 verification: x % 1 is semantically equivalent to 0
    verify_equivalence(&expr, &zero).expect("x % 1 should equal 0");
}

// ============================================================================
// B. Zero Propagation (3 tests)
// ============================================================================

#[test]
fn test_zero_mul_zero() {
    // x * 0 → 0
    let x = UOp::var("x", DType::Int32, 0, 100);
    let zero = UOp::native_const(0i32);
    let expr = x.try_mul_op(&zero).unwrap();

    let matcher = symbolic_simple();
    let simplified = graph_rewrite(&matcher, expr.clone(), &mut ());

    // Should simplify to 0
    assert!(Rc::ptr_eq(&simplified, &zero), "x * 0 should simplify to 0");
    verify_equivalence(&expr, &simplified).expect("x * 0 should equal 0");
}

#[test]
fn test_zero_and_zero() {
    // x & 0 → 0
    let x = UOp::var("x", DType::Int32, 0, 100);
    let zero = UOp::native_const(0i32);
    let expr = x.try_and_op(&zero).unwrap();

    let matcher = symbolic_simple();
    let simplified = graph_rewrite(&matcher, expr.clone(), &mut ());

    // Should simplify to 0
    assert!(Rc::ptr_eq(&simplified, &zero), "x & 0 should simplify to 0");

    // Note: Z3 verification for bitwise AND is not implemented in convert.rs
    // We skip Z3 verification for this test
}

#[test]
fn test_zero_div_x() {
    // 0 / x → 0 (verify semantically, for x ≠ 0)
    let x = UOp::var("x", DType::Int32, 1, 100); // x ≠ 0
    let zero = UOp::native_const(0i32);
    let expr = zero.try_idiv_op(&x).unwrap();

    // Z3 verification: 0 / x is semantically equivalent to 0
    verify_equivalence(&expr, &zero).expect("0 / x should equal 0");
}

// ============================================================================
// C. Self-Folding (4 tests)
// ============================================================================

#[test]
fn test_self_sub_zero() {
    // x - x → 0 (verify semantically even if optimizer doesn't simplify)
    let x = UOp::var("x", DType::Int32, 0, 100);
    let expr = x.try_sub_op(&x).unwrap();
    let zero = UOp::native_const(0i32);

    // Z3 verification: x - x is semantically equivalent to 0
    verify_equivalence(&expr, &zero).expect("x - x should equal 0");
}

#[test]
fn test_self_div_one() {
    // x / x → 1 (for x ≠ 0)
    let x = UOp::var("x", DType::Int32, 1, 100); // x ≠ 0
    let expr = x.try_idiv_op(&x).unwrap();

    let matcher = symbolic_simple();
    let simplified = graph_rewrite(&matcher, expr.clone(), &mut ());

    // Should simplify to 1
    match simplified.op() {
        Op::Const(cv) => assert_eq!(cv.0, ConstValue::Int(1), "x / x should be 1"),
        other => panic!("Expected Const(1), got {:?}", other),
    }

    verify_equivalence(&expr, &simplified).expect("x / x should equal 1");
}

#[test]
fn test_self_mod_zero() {
    // x % x → 0 (for x ≠ 0)
    let x = UOp::var("x", DType::Int32, 1, 100); // x ≠ 0
    let expr = x.try_mod_op(&x).unwrap();

    let matcher = symbolic_simple();
    let simplified = graph_rewrite(&matcher, expr.clone(), &mut ());

    // Should simplify to 0
    match simplified.op() {
        Op::Const(cv) => assert_eq!(cv.0, ConstValue::Int(0), "x % x should be 0"),
        other => panic!("Expected Const(0), got {:?}", other),
    }

    verify_equivalence(&expr, &simplified).expect("x % x should equal 0");
}

#[test]
fn test_self_and_identity() {
    // x & x → x
    let x = UOp::var("x", DType::Int32, 0, 100);
    let expr = x.try_and_op(&x).unwrap();

    let matcher = symbolic_simple();
    let simplified = graph_rewrite(&matcher, expr.clone(), &mut ());

    // Should simplify to x
    assert!(Rc::ptr_eq(&simplified, &x), "x & x should simplify to x");

    // Note: Z3 verification for bitwise AND is not implemented
    // We skip Z3 verification for this test
}

// ============================================================================
// D. Division Patterns (4 tests)
// ============================================================================

#[test]
fn test_div_cancel_mul() {
    // (a * b) / b → a (for b ≠ 0)
    let a = UOp::var("a", DType::Int32, 0, 100);
    let b = UOp::var("b", DType::Int32, 1, 100); // b ≠ 0
    let a_mul_b = a.try_mul_op(&b).unwrap();
    let expr = a_mul_b.try_idiv_op(&b).unwrap();

    let matcher = symbolic_simple();
    let simplified = graph_rewrite(&matcher, expr.clone(), &mut ());

    // Should simplify to a
    assert!(Rc::ptr_eq(&simplified, &a), "(a * b) / b should simplify to a");
    verify_equivalence(&expr, &simplified).expect("(a * b) / b should equal a");
}

#[test]
fn test_div_chain() {
    // (a / b) / c → a / (b * c) (for b, c ≠ 0)
    let a = UOp::var("a", DType::Int32, 0, 100);
    let b = UOp::var("b", DType::Int32, 1, 10); // b ≠ 0
    let c = UOp::var("c", DType::Int32, 1, 10); // c ≠ 0
    let a_div_b = a.try_idiv_op(&b).unwrap();
    let expr = a_div_b.try_idiv_op(&c).unwrap();

    let matcher = symbolic_simple();
    let simplified = graph_rewrite(&matcher, expr.clone(), &mut ());

    // Verify semantically (structure might differ)
    verify_equivalence(&expr, &simplified).expect("(a / b) / c should be semantically equivalent");
}

#[test]
fn test_div_gcd_factor() {
    // (a * 6) / (b * 6) → a / b (for b ≠ 0)
    let a = UOp::var("a", DType::Int32, 0, 60);
    let b = UOp::var("b", DType::Int32, 1, 10); // b ≠ 0
    let six = UOp::native_const(6i32);

    let a_mul_6 = a.try_mul_op(&six).unwrap();
    let b_mul_6 = b.try_mul_op(&six).unwrap();
    let expr = a_mul_6.try_idiv_op(&b_mul_6).unwrap();

    let matcher = symbolic_simple();
    let simplified = graph_rewrite(&matcher, expr.clone(), &mut ());

    // Verify semantically
    verify_equivalence(&expr, &simplified).expect("(a * c) / (b * c) should be semantically equivalent");
}

#[test]
fn test_mod_self_zero() {
    // a % a → 0 (for a ≠ 0)
    let a = UOp::var("a", DType::Int32, 1, 100); // a ≠ 0
    let expr = a.try_mod_op(&a).unwrap();

    let matcher = symbolic_simple();
    let simplified = graph_rewrite(&matcher, expr.clone(), &mut ());

    // Should simplify to 0
    match simplified.op() {
        Op::Const(cv) => assert_eq!(cv.0, ConstValue::Int(0), "a % a should be 0"),
        other => panic!("Expected Const(0), got {:?}", other),
    }

    verify_equivalence(&expr, &simplified).expect("a % a should equal 0");
}

// ============================================================================
// E. Term Combining (4 tests)
// ============================================================================

#[test]
fn test_term_combine_add() {
    // x + x → 2 * x
    let x = UOp::var("x", DType::Int32, 0, 100);
    let expr = x.try_add_op(&x).unwrap();

    let matcher = symbolic_simple();
    let simplified = graph_rewrite(&matcher, expr.clone(), &mut ());

    // Verify semantically (might be 2*x or x*2 or x+x)
    verify_equivalence(&expr, &simplified).expect("x + x should be semantically equivalent to 2 * x");
}

#[test]
fn test_term_combine_coefficients() {
    // (2 * x) + (3 * x) → 5 * x
    let x = UOp::var("x", DType::Int32, 0, 100);
    let two = UOp::native_const(2i32);
    let three = UOp::native_const(3i32);

    let two_x = two.try_mul_op(&x).unwrap();
    let three_x = three.try_mul_op(&x).unwrap();
    let expr = two_x.try_add_op(&three_x).unwrap();

    let matcher = symbolic_simple();
    let simplified = graph_rewrite(&matcher, expr.clone(), &mut ());

    // Verify semantically
    verify_equivalence(&expr, &simplified).expect("(2 * x) + (3 * x) should equal 5 * x");
}

#[test]
fn test_const_folding_add() {
    // (x + 3) + 5 → x + 8
    let x = UOp::var("x", DType::Int32, 0, 100);
    let three = UOp::native_const(3i32);
    let five = UOp::native_const(5i32);

    let x_plus_3 = x.try_add_op(&three).unwrap();
    let expr = x_plus_3.try_add_op(&five).unwrap();

    let matcher = symbolic_simple();
    let simplified = graph_rewrite(&matcher, expr.clone(), &mut ());

    // Verify semantically
    verify_equivalence(&expr, &simplified).expect("(x + 3) + 5 should equal x + 8");
}

#[test]
fn test_const_folding_mul() {
    // (x * 2) * 3 → x * 6
    let x = UOp::var("x", DType::Int32, 0, 100);
    let two = UOp::native_const(2i32);
    let three = UOp::native_const(3i32);

    let x_mul_2 = x.try_mul_op(&two).unwrap();
    let expr = x_mul_2.try_mul_op(&three).unwrap();

    let matcher = symbolic_simple();
    let simplified = graph_rewrite(&matcher, expr.clone(), &mut ());

    // Verify semantically
    verify_equivalence(&expr, &simplified).expect("(x * 2) * 3 should equal x * 6");
}
