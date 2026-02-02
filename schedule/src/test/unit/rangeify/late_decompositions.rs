//! Tests for late decomposition patterns (get_late_rewrite_patterns).
//!
//! Based on Tinygrad's decompositions.py:321-367.

use morok_ir::types::ConstValue;
use morok_ir::{BinaryOp, Op, UOp, UnaryOp};

use crate::rangeify::patterns::{
    pm_comparison_negations, pm_div_to_shr, pm_fdiv_to_mul, pm_mod_to_and, pm_mul_to_shl, pm_neg_from_mul,
};
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

// ============================================================================
// DIV → SHR PATTERNS (Tinygrad: decompositions.py:340-344)
// ============================================================================

#[test]
fn test_div_power_of_two_becomes_shr() {
    let matcher = pm_div_to_shr();

    // x // 8 → x >> 3 (for non-negative x)
    // Use a range with non-negative vmin
    let x = UOp::range(UOp::index_const(100), 0);
    let div = x.idiv(&UOp::index_const(8));

    let result = graph_rewrite_bottom_up(&matcher, div, &mut ());

    // Should be Shr(x, 3)
    if let Op::Binary(BinaryOp::Shr, lhs, rhs) = result.op() {
        assert!(std::sync::Arc::ptr_eq(lhs, &x), "LHS should be x");
        if let Op::Const(c) = rhs.op() {
            assert_eq!(c.0, ConstValue::Int(3), "RHS should be 3 (log2(8))");
        } else {
            panic!("Expected constant 3, got {:?}", rhs.op());
        }
    } else {
        panic!("Expected Shr operation, got {:?}", result.op());
    }
}

#[test]
fn test_div_non_power_of_two_unchanged() {
    let matcher = pm_div_to_shr();

    // x // 7 should NOT change (7 is not power of two)
    let x = UOp::range(UOp::index_const(100), 0);
    let div = x.idiv(&UOp::index_const(7));

    let result = graph_rewrite_bottom_up(&matcher, div.clone(), &mut ());

    // Should still be Idiv
    assert!(matches!(result.op(), Op::Binary(BinaryOp::Idiv, _, _)), "x // 7 should remain Idiv");
}

#[test]
fn test_div_by_one_unchanged() {
    let matcher = pm_div_to_shr();

    // x // 1 should NOT be converted to shift (guard in pattern)
    let x = UOp::range(UOp::index_const(100), 0);
    let div = x.idiv(&UOp::index_const(1));

    let result = graph_rewrite_bottom_up(&matcher, div.clone(), &mut ());

    // Should still be Idiv (trivial case skipped)
    assert!(matches!(result.op(), Op::Binary(BinaryOp::Idiv, _, _)), "x // 1 should remain Idiv");
}

#[test]
fn test_div_power_of_two_various_sizes() {
    let matcher = pm_div_to_shr();

    for (power, shift) in [(2, 1), (4, 2), (16, 4), (32, 5), (64, 6), (128, 7), (256, 8)] {
        let x = UOp::range(UOp::index_const(10000), 0);
        let div = x.idiv(&UOp::index_const(power));

        let result = graph_rewrite_bottom_up(&matcher, div, &mut ());

        if let Op::Binary(BinaryOp::Shr, _, rhs) = result.op() {
            if let Op::Const(c) = rhs.op() {
                assert_eq!(c.0, ConstValue::Int(shift), "Expected shift {} for divisor {}", shift, power);
            }
        } else {
            panic!("Expected Shr for x // {}, got {:?}", power, result.op());
        }
    }
}

// ============================================================================
// FDIV → MUL PATTERNS (Tinygrad: decompositions.py:364-366)
// ============================================================================

#[test]
fn test_fdiv_constant_becomes_mul_reciprocal() {
    let matcher = pm_fdiv_to_mul();

    // x / 2.0 → x * 0.5
    let x = UOp::native_const(10.0f32);
    let div = x.try_div(&UOp::native_const(2.0f32)).unwrap();

    let result = graph_rewrite_bottom_up(&matcher, div, &mut ());

    // Should be Mul(x, 0.5)
    if let Op::Binary(BinaryOp::Mul, _, rhs) = result.op() {
        if let Op::Const(c) = rhs.op() {
            match c.0 {
                ConstValue::Float(f) => assert!((f - 0.5).abs() < 1e-6, "Expected 0.5, got {}", f),
                _ => panic!("Expected float constant"),
            }
        } else {
            panic!("Expected constant, got {:?}", rhs.op());
        }
    } else {
        panic!("Expected Mul operation, got {:?}", result.op());
    }
}

#[test]
fn test_fdiv_by_zero_prevented_at_construction() {
    // Division by zero is prevented at UOp construction time
    let x = UOp::native_const(10.0f32);
    let result = x.try_div(&UOp::native_const(0.0f32));

    // Should return error, not create a div-by-zero operation
    assert!(result.is_err(), "Division by zero should fail at construction");
}

#[test]
fn test_fdiv_various_constants() {
    let matcher = pm_fdiv_to_mul();

    for (divisor, expected_recip) in [(4.0f32, 0.25f32), (5.0, 0.2), (10.0, 0.1), (0.5, 2.0)] {
        let x = UOp::native_const(100.0f32);
        let div = x.try_div(&UOp::native_const(divisor)).unwrap();

        let result = graph_rewrite_bottom_up(&matcher, div, &mut ());

        if let Op::Binary(BinaryOp::Mul, _, rhs) = result.op() {
            if let Op::Const(c) = rhs.op() {
                match c.0 {
                    ConstValue::Float(f) => {
                        assert!((f - expected_recip as f64).abs() < 1e-6, "Expected {}, got {}", expected_recip, f);
                    }
                    _ => panic!("Expected float constant"),
                }
            }
        } else {
            panic!("Expected Mul for x / {}, got {:?}", divisor, result.op());
        }
    }
}

// ============================================================================
// COMPARISON NEGATION PATTERNS (Tinygrad: decompositions.py:354-361)
// ============================================================================

#[test]
fn test_not_lt_becomes_reversed_lt() {
    let matcher = pm_comparison_negations();

    // !(x < 5) → (4 < x)
    let x = UOp::range(UOp::index_const(100), 0);
    let five = UOp::index_const(5);
    let lt = x.try_cmplt(&five).unwrap();
    let not_lt = lt.not();

    let result = graph_rewrite_bottom_up(&matcher, not_lt, &mut ());

    // Should be Lt(4, x)
    if let Op::Binary(BinaryOp::Lt, lhs, rhs) = result.op() {
        // LHS should be constant 4
        if let Op::Const(c) = lhs.op() {
            assert_eq!(c.0, ConstValue::Int(4), "LHS should be 4");
        } else {
            panic!("Expected constant 4, got {:?}", lhs.op());
        }
        // RHS should be x
        assert!(std::sync::Arc::ptr_eq(rhs, &x), "RHS should be x");
    } else {
        panic!("Expected Lt operation, got {:?}", result.op());
    }
}

#[test]
fn test_not_reversed_lt_becomes_lt() {
    let matcher = pm_comparison_negations();

    // !(5 < x) → (x < 6)
    let x = UOp::range(UOp::index_const(100), 0);
    let five = UOp::index_const(5);
    let lt = five.try_cmplt(&x).unwrap();
    let not_lt = lt.not();

    let result = graph_rewrite_bottom_up(&matcher, not_lt, &mut ());

    // Should be Lt(x, 6)
    if let Op::Binary(BinaryOp::Lt, lhs, rhs) = result.op() {
        // LHS should be x
        assert!(std::sync::Arc::ptr_eq(lhs, &x), "LHS should be x");
        // RHS should be constant 6
        if let Op::Const(c) = rhs.op() {
            assert_eq!(c.0, ConstValue::Int(6), "RHS should be 6");
        } else {
            panic!("Expected constant 6, got {:?}", rhs.op());
        }
    } else {
        panic!("Expected Lt operation, got {:?}", result.op());
    }
}

#[test]
fn test_range_compression() {
    let matcher = pm_comparison_negations();

    // (3 < x) & (x < 5) → x == 4
    let x = UOp::range(UOp::index_const(100), 0);
    let three = UOp::index_const(3);
    let five = UOp::index_const(5);

    let gt_three = three.try_cmplt(&x).unwrap(); // 3 < x
    let lt_five = x.try_cmplt(&five).unwrap(); // x < 5
    let combined = gt_three.try_and_op(&lt_five).unwrap();

    let result = graph_rewrite_bottom_up(&matcher, combined, &mut ());

    // Should be Eq(x, 4)
    if let Op::Binary(BinaryOp::Eq, lhs, rhs) = result.op() {
        // One side should be x, other should be 4
        let (var_side, const_side) = if matches!(lhs.op(), Op::Const(_)) { (rhs, lhs) } else { (lhs, rhs) };

        assert!(std::sync::Arc::ptr_eq(var_side, &x), "Variable side should be x");
        if let Op::Const(c) = const_side.op() {
            assert_eq!(c.0, ConstValue::Int(4), "Constant should be 4");
        } else {
            panic!("Expected constant 4, got {:?}", const_side.op());
        }
    } else {
        panic!("Expected Eq operation, got {:?}", result.op());
    }
}

#[test]
fn test_negated_mul_comparison() {
    let matcher = pm_comparison_negations();

    // x*-1 < 5 → -5 < x
    let x = UOp::range(UOp::index_const(100), 0);
    let neg_one = UOp::index_const(-1);
    let five = UOp::index_const(5);

    let neg_x = x.mul(&neg_one);
    let lt = neg_x.try_cmplt(&five).unwrap();

    let result = graph_rewrite_bottom_up(&matcher, lt, &mut ());

    // Should be Lt(-5, x)
    if let Op::Binary(BinaryOp::Lt, lhs, rhs) = result.op() {
        // LHS should be constant -5
        if let Op::Const(c) = lhs.op() {
            assert_eq!(c.0, ConstValue::Int(-5), "LHS should be -5");
        } else {
            panic!("Expected constant -5, got {:?}", lhs.op());
        }
        // RHS should be x
        assert!(std::sync::Arc::ptr_eq(rhs, &x), "RHS should be x");
    } else {
        panic!("Expected Lt operation, got {:?}", result.op());
    }
}
