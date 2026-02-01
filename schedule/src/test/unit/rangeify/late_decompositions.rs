//! Tests for late decomposition patterns (get_late_rewrite_patterns).
//!
//! Based on Tinygrad's decompositions.py:321-367.

use morok_ir::types::ConstValue;
use morok_ir::{BinaryOp, Op, UOp, UnaryOp};

use crate::rangeify::patterns::{pm_mod_to_and, pm_mul_to_shl, pm_neg_from_mul};
use crate::rewrite::graph_rewrite_bottom_up;

// ============================================================================
// MOD → AND PATTERNS
// ============================================================================

#[test]
fn test_mod_power_of_two_becomes_and() {
    let matcher = pm_mod_to_and();

    // x % 8 → x & 7
    let x = UOp::range(UOp::index_const(100), 0);
    let modulo = x.mod_(&UOp::index_const(8));

    let result = graph_rewrite_bottom_up(&matcher, modulo, &mut ());

    // Should be And(x, 7)
    if let Op::Binary(BinaryOp::And, lhs, rhs) = result.op() {
        assert!(std::sync::Arc::ptr_eq(lhs, &x), "LHS should be x");
        if let Op::Const(c) = rhs.op() {
            assert_eq!(c.0, ConstValue::Int(7), "RHS should be 7");
        } else {
            panic!("Expected constant 7, got {:?}", rhs.op());
        }
    } else {
        panic!("Expected And operation, got {:?}", result.op());
    }
}

#[test]
fn test_mod_non_power_of_two_unchanged() {
    let matcher = pm_mod_to_and();

    // x % 7 should NOT change (7 is not power of two)
    let x = UOp::range(UOp::index_const(100), 0);
    let modulo = x.mod_(&UOp::index_const(7));

    let result = graph_rewrite_bottom_up(&matcher, modulo.clone(), &mut ());

    // Should still be Mod
    assert!(matches!(result.op(), Op::Binary(BinaryOp::Mod, _, _)), "x % 7 should remain Mod");
}

#[test]
fn test_mod_power_of_two_various_sizes() {
    let matcher = pm_mod_to_and();

    for power in [2, 4, 16, 32, 64, 128, 256, 512, 1024] {
        let x = UOp::range(UOp::index_const(10000), 0);
        let modulo = x.mod_(&UOp::index_const(power));

        let result = graph_rewrite_bottom_up(&matcher, modulo, &mut ());

        if let Op::Binary(BinaryOp::And, _, rhs) = result.op() {
            if let Op::Const(c) = rhs.op() {
                assert_eq!(c.0, ConstValue::Int(power - 1), "Expected mask {} for modulus {}", power - 1, power);
            }
        } else {
            panic!("Expected And for x % {}, got {:?}", power, result.op());
        }
    }
}

// ============================================================================
// MUL → SHL PATTERNS
// ============================================================================

#[test]
fn test_mul_power_of_two_becomes_shl() {
    let matcher = pm_mul_to_shl();

    // x * 8 → x << 3
    let x = UOp::range(UOp::index_const(100), 0);
    let mul = x.mul(&UOp::index_const(8));

    let result = graph_rewrite_bottom_up(&matcher, mul, &mut ());

    // Should be Shl(x, 3)
    if let Op::Binary(BinaryOp::Shl, lhs, rhs) = result.op() {
        assert!(std::sync::Arc::ptr_eq(lhs, &x), "LHS should be x");
        if let Op::Const(c) = rhs.op() {
            assert_eq!(c.0, ConstValue::Int(3), "RHS should be 3 (log2(8))");
        } else {
            panic!("Expected constant 3, got {:?}", rhs.op());
        }
    } else {
        panic!("Expected Shl operation, got {:?}", result.op());
    }
}

#[test]
fn test_mul_non_power_of_two_unchanged() {
    let matcher = pm_mul_to_shl();

    // x * 7 should NOT change (7 is not power of two)
    let x = UOp::range(UOp::index_const(100), 0);
    let mul = x.mul(&UOp::index_const(7));

    let result = graph_rewrite_bottom_up(&matcher, mul.clone(), &mut ());

    // Should still be Mul
    assert!(matches!(result.op(), Op::Binary(BinaryOp::Mul, _, _)), "x * 7 should remain Mul");
}

#[test]
fn test_mul_by_one_returns_identity() {
    let matcher = pm_mul_to_shl();

    // x * 1 → x (handled specially, not converted to shift)
    let x = UOp::range(UOp::index_const(100), 0);
    let mul = x.mul(&UOp::index_const(1));

    let result = graph_rewrite_bottom_up(&matcher, mul, &mut ());

    assert!(std::sync::Arc::ptr_eq(&result, &x), "x * 1 should return x");
}

// ============================================================================
// NEG FROM MUL PATTERNS
// ============================================================================

#[test]
fn test_mul_neg_one_becomes_neg() {
    let matcher = pm_neg_from_mul();

    // x * -1 → NEG(x)
    let x = UOp::range(UOp::index_const(100), 0);
    let mul = x.mul(&UOp::index_const(-1));

    let result = graph_rewrite_bottom_up(&matcher, mul, &mut ());

    // Should be Neg(x)
    if let Op::Unary(UnaryOp::Neg, inner) = result.op() {
        assert!(std::sync::Arc::ptr_eq(inner, &x), "Inner should be x");
    } else {
        panic!("Expected Neg operation, got {:?}", result.op());
    }
}

#[test]
fn test_mul_pos_one_unchanged_by_neg_pattern() {
    let matcher = pm_neg_from_mul();

    // x * 1 should NOT match the neg pattern
    let x = UOp::range(UOp::index_const(100), 0);
    let mul = x.mul(&UOp::index_const(1));

    let result = graph_rewrite_bottom_up(&matcher, mul.clone(), &mut ());

    // Should still be Mul
    assert!(matches!(result.op(), Op::Binary(BinaryOp::Mul, _, _)), "x * 1 should remain Mul");
}
