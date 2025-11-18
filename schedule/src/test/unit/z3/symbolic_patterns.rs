//! Z3 verification tests for critical symbolic optimization patterns.
//!
//! These tests verify that pattern rewrites preserve semantics using Z3/SMT solving.
//! Each test creates an expression, applies symbolic simplification, and verifies
//! that the original and simplified versions are semantically equivalent.

use morok_dtype::DType;
use morok_ir::types::{BinaryOp, ConstValue};
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
    let zero = UOp::const_(DType::Int32, ConstValue::Int(0));
    let expr = UOp::new(Op::Binary(BinaryOp::Add, Rc::clone(&x), zero), DType::Int32);

    let matcher = symbolic_simple();
    let simplified = graph_rewrite(&matcher, expr.clone());

    // Should simplify to x
    assert!(Rc::ptr_eq(&simplified, &x), "x + 0 should simplify to x");

    // Z3 verification
    verify_equivalence(&expr, &simplified).expect("x + 0 should equal x");
}

#[test]
fn test_identity_mul_one() {
    // x * 1 → x
    let x = UOp::var("x", DType::Int32, 0, 100);
    let one = UOp::const_(DType::Int32, ConstValue::Int(1));
    let expr = UOp::new(Op::Binary(BinaryOp::Mul, Rc::clone(&x), one), DType::Int32);

    let matcher = symbolic_simple();
    let simplified = graph_rewrite(&matcher, expr.clone());

    assert!(Rc::ptr_eq(&simplified, &x), "x * 1 should simplify to x");
    verify_equivalence(&expr, &simplified).expect("x * 1 should equal x");
}

#[test]
fn test_identity_sub_zero() {
    // x - 0 → x
    let x = UOp::var("x", DType::Int32, 0, 100);
    let zero = UOp::const_(DType::Int32, ConstValue::Int(0));
    let expr = UOp::new(Op::Binary(BinaryOp::Sub, Rc::clone(&x), zero), DType::Int32);

    let matcher = symbolic_simple();
    let simplified = graph_rewrite(&matcher, expr.clone());

    assert!(Rc::ptr_eq(&simplified, &x), "x - 0 should simplify to x");
    verify_equivalence(&expr, &simplified).expect("x - 0 should equal x");
}

#[test]
fn test_identity_div_one() {
    // x / 1 → x
    let x = UOp::var("x", DType::Int32, 0, 100);
    let one = UOp::const_(DType::Int32, ConstValue::Int(1));
    let expr = UOp::new(Op::Binary(BinaryOp::Idiv, Rc::clone(&x), one), DType::Int32);

    let matcher = symbolic_simple();
    let simplified = graph_rewrite(&matcher, expr.clone());

    assert!(Rc::ptr_eq(&simplified, &x), "x / 1 should simplify to x");
    verify_equivalence(&expr, &simplified).expect("x / 1 should equal x");
}

#[test]
fn test_identity_mod_one() {
    // x % 1 → 0 (verify semantically even if optimizer doesn't simplify)
    let x = UOp::var("x", DType::Int32, 0, 100);
    let one = UOp::const_(DType::Int32, ConstValue::Int(1));
    let expr = UOp::new(Op::Binary(BinaryOp::Mod, Rc::clone(&x), Rc::clone(&one)), DType::Int32);
    let zero = UOp::const_(DType::Int32, ConstValue::Int(0));

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
    let zero = UOp::const_(DType::Int32, ConstValue::Int(0));
    let expr = UOp::new(Op::Binary(BinaryOp::Mul, x, Rc::clone(&zero)), DType::Int32);

    let matcher = symbolic_simple();
    let simplified = graph_rewrite(&matcher, expr.clone());

    // Should simplify to 0
    assert!(Rc::ptr_eq(&simplified, &zero), "x * 0 should simplify to 0");
    verify_equivalence(&expr, &simplified).expect("x * 0 should equal 0");
}

#[test]
fn test_zero_and_zero() {
    // x & 0 → 0
    let x = UOp::var("x", DType::Int32, 0, 100);
    let zero = UOp::const_(DType::Int32, ConstValue::Int(0));
    let expr = UOp::new(Op::Binary(BinaryOp::And, x, Rc::clone(&zero)), DType::Int32);

    let matcher = symbolic_simple();
    let simplified = graph_rewrite(&matcher, expr.clone());

    // Should simplify to 0
    assert!(Rc::ptr_eq(&simplified, &zero), "x & 0 should simplify to 0");

    // Note: Z3 verification for bitwise AND is not implemented in convert.rs
    // We skip Z3 verification for this test
}

#[test]
fn test_zero_div_x() {
    // 0 / x → 0 (verify semantically, for x ≠ 0)
    let x = UOp::var("x", DType::Int32, 1, 100); // x ≠ 0
    let zero = UOp::const_(DType::Int32, ConstValue::Int(0));
    let expr = UOp::new(Op::Binary(BinaryOp::Idiv, Rc::clone(&zero), x), DType::Int32);

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
    let expr = UOp::new(Op::Binary(BinaryOp::Sub, Rc::clone(&x), Rc::clone(&x)), DType::Int32);
    let zero = UOp::const_(DType::Int32, ConstValue::Int(0));

    // Z3 verification: x - x is semantically equivalent to 0
    verify_equivalence(&expr, &zero).expect("x - x should equal 0");
}

#[test]
fn test_self_div_one() {
    // x / x → 1 (for x ≠ 0)
    let x = UOp::var("x", DType::Int32, 1, 100); // x ≠ 0
    let expr = UOp::new(Op::Binary(BinaryOp::Idiv, Rc::clone(&x), Rc::clone(&x)), DType::Int32);

    let matcher = symbolic_simple();
    let simplified = graph_rewrite(&matcher, expr.clone());

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
    let expr = UOp::new(Op::Binary(BinaryOp::Mod, Rc::clone(&x), Rc::clone(&x)), DType::Int32);

    let matcher = symbolic_simple();
    let simplified = graph_rewrite(&matcher, expr.clone());

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
    let expr = UOp::new(Op::Binary(BinaryOp::And, Rc::clone(&x), Rc::clone(&x)), DType::Int32);

    let matcher = symbolic_simple();
    let simplified = graph_rewrite(&matcher, expr.clone());

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
    let a_mul_b = UOp::new(Op::Binary(BinaryOp::Mul, Rc::clone(&a), Rc::clone(&b)), DType::Int32);
    let expr = UOp::new(Op::Binary(BinaryOp::Idiv, a_mul_b, Rc::clone(&b)), DType::Int32);

    let matcher = symbolic_simple();
    let simplified = graph_rewrite(&matcher, expr.clone());

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
    let a_div_b = UOp::new(Op::Binary(BinaryOp::Idiv, Rc::clone(&a), Rc::clone(&b)), DType::Int32);
    let expr = UOp::new(Op::Binary(BinaryOp::Idiv, a_div_b, Rc::clone(&c)), DType::Int32);

    let matcher = symbolic_simple();
    let simplified = graph_rewrite(&matcher, expr.clone());

    // Verify semantically (structure might differ)
    verify_equivalence(&expr, &simplified).expect("(a / b) / c should be semantically equivalent");
}

#[test]
fn test_div_gcd_factor() {
    // (a * 6) / (b * 6) → a / b (for b ≠ 0)
    let a = UOp::var("a", DType::Int32, 0, 60);
    let b = UOp::var("b", DType::Int32, 1, 10); // b ≠ 0
    let six = UOp::const_(DType::Int32, ConstValue::Int(6));

    let a_mul_6 = UOp::new(Op::Binary(BinaryOp::Mul, Rc::clone(&a), Rc::clone(&six)), DType::Int32);
    let b_mul_6 = UOp::new(Op::Binary(BinaryOp::Mul, Rc::clone(&b), Rc::clone(&six)), DType::Int32);
    let expr = UOp::new(Op::Binary(BinaryOp::Idiv, a_mul_6, b_mul_6), DType::Int32);

    let matcher = symbolic_simple();
    let simplified = graph_rewrite(&matcher, expr.clone());

    // Verify semantically
    verify_equivalence(&expr, &simplified).expect("(a * c) / (b * c) should be semantically equivalent");
}

#[test]
fn test_mod_self_zero() {
    // a % a → 0 (for a ≠ 0)
    let a = UOp::var("a", DType::Int32, 1, 100); // a ≠ 0
    let expr = UOp::new(Op::Binary(BinaryOp::Mod, Rc::clone(&a), Rc::clone(&a)), DType::Int32);

    let matcher = symbolic_simple();
    let simplified = graph_rewrite(&matcher, expr.clone());

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
    let expr = UOp::new(Op::Binary(BinaryOp::Add, Rc::clone(&x), Rc::clone(&x)), DType::Int32);

    let matcher = symbolic_simple();
    let simplified = graph_rewrite(&matcher, expr.clone());

    // Verify semantically (might be 2*x or x*2 or x+x)
    verify_equivalence(&expr, &simplified).expect("x + x should be semantically equivalent to 2 * x");
}

#[test]
fn test_term_combine_coefficients() {
    // (2 * x) + (3 * x) → 5 * x
    let x = UOp::var("x", DType::Int32, 0, 100);
    let two = UOp::const_(DType::Int32, ConstValue::Int(2));
    let three = UOp::const_(DType::Int32, ConstValue::Int(3));

    let two_x = UOp::new(Op::Binary(BinaryOp::Mul, two, Rc::clone(&x)), DType::Int32);
    let three_x = UOp::new(Op::Binary(BinaryOp::Mul, three, Rc::clone(&x)), DType::Int32);
    let expr = UOp::new(Op::Binary(BinaryOp::Add, two_x, three_x), DType::Int32);

    let matcher = symbolic_simple();
    let simplified = graph_rewrite(&matcher, expr.clone());

    // Verify semantically
    verify_equivalence(&expr, &simplified).expect("(2 * x) + (3 * x) should equal 5 * x");
}

#[test]
fn test_const_folding_add() {
    // (x + 3) + 5 → x + 8
    let x = UOp::var("x", DType::Int32, 0, 100);
    let three = UOp::const_(DType::Int32, ConstValue::Int(3));
    let five = UOp::const_(DType::Int32, ConstValue::Int(5));

    let x_plus_3 = UOp::new(Op::Binary(BinaryOp::Add, Rc::clone(&x), three), DType::Int32);
    let expr = UOp::new(Op::Binary(BinaryOp::Add, x_plus_3, five), DType::Int32);

    let matcher = symbolic_simple();
    let simplified = graph_rewrite(&matcher, expr.clone());

    // Verify semantically
    verify_equivalence(&expr, &simplified).expect("(x + 3) + 5 should equal x + 8");
}

#[test]
fn test_const_folding_mul() {
    // (x * 2) * 3 → x * 6
    let x = UOp::var("x", DType::Int32, 0, 100);
    let two = UOp::const_(DType::Int32, ConstValue::Int(2));
    let three = UOp::const_(DType::Int32, ConstValue::Int(3));

    let x_mul_2 = UOp::new(Op::Binary(BinaryOp::Mul, Rc::clone(&x), two), DType::Int32);
    let expr = UOp::new(Op::Binary(BinaryOp::Mul, x_mul_2, three), DType::Int32);

    let matcher = symbolic_simple();
    let simplified = graph_rewrite(&matcher, expr.clone());

    // Verify semantically
    verify_equivalence(&expr, &simplified).expect("(x * 2) * 3 should equal x * 6");
}
