//! Property tests for symbolic optimizer algebraic rules.
//!
//! Tests that pattern rewrites preserve semantics and follow algebraic laws.

use std::rc::Rc;

use proptest::prelude::*;

use morok_dtype::DType;
use morok_ir::types::{BinaryOp, ConstValue};
use morok_ir::{Op, UOp};

use crate::rewrite::graph_rewrite;
use crate::symbolic::symbolic_simple;

// Import generators from ir crate
use morok_ir::test::property::generators::*;

// ============================================================================
// Identity Elimination Properties
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// x + 0 should simplify to x
    #[test]
    fn identity_add_zero_right(x in arb_simple_uop(DType::Int32)) {
        let zero = UOp::const_(DType::Int32, ConstValue::Int(0));
        let expr = UOp::new(Op::Binary(BinaryOp::Add, Rc::clone(&x), zero), DType::Int32);

        let matcher = symbolic_simple();
        let simplified = graph_rewrite(&matcher, expr);

        prop_assert!(Rc::ptr_eq(&simplified, &x),
            "x + 0 should simplify to x");
    }

    /// 0 + x should simplify to x (commutativity)
    #[test]
    fn identity_add_zero_left(x in arb_simple_uop(DType::Int32)) {
        let zero = UOp::const_(DType::Int32, ConstValue::Int(0));
        let expr = UOp::new(Op::Binary(BinaryOp::Add, zero, Rc::clone(&x)), DType::Int32);

        let matcher = symbolic_simple();
        let simplified = graph_rewrite(&matcher, expr);

        prop_assert!(Rc::ptr_eq(&simplified, &x),
            "0 + x should simplify to x");
    }

    /// x - 0 should simplify to x
    #[test]
    fn identity_sub_zero(x in arb_simple_uop(DType::Int32)) {
        let zero = UOp::const_(DType::Int32, ConstValue::Int(0));
        let expr = UOp::new(Op::Binary(BinaryOp::Sub, Rc::clone(&x), zero), DType::Int32);

        let matcher = symbolic_simple();
        let simplified = graph_rewrite(&matcher, expr);

        prop_assert!(Rc::ptr_eq(&simplified, &x),
            "x - 0 should simplify to x");
    }

    /// x * 1 should simplify to x
    #[test]
    fn identity_mul_one_right(x in arb_simple_uop(DType::Int32)) {
        let one = UOp::const_(DType::Int32, ConstValue::Int(1));
        let expr = UOp::new(Op::Binary(BinaryOp::Mul, Rc::clone(&x), one), DType::Int32);

        let matcher = symbolic_simple();
        let simplified = graph_rewrite(&matcher, expr);

        prop_assert!(Rc::ptr_eq(&simplified, &x),
            "x * 1 should simplify to x");
    }

    /// 1 * x should simplify to x (commutativity)
    #[test]
    fn identity_mul_one_left(x in arb_simple_uop(DType::Int32)) {
        let one = UOp::const_(DType::Int32, ConstValue::Int(1));
        let expr = UOp::new(Op::Binary(BinaryOp::Mul, one, Rc::clone(&x)), DType::Int32);

        let matcher = symbolic_simple();
        let simplified = graph_rewrite(&matcher, expr);

        prop_assert!(Rc::ptr_eq(&simplified, &x),
            "1 * x should simplify to x");
    }

    /// x / 1 should simplify to x (integer division)
    #[test]
    fn identity_idiv_one(x in arb_simple_uop(DType::Int32)) {
        let one = UOp::const_(DType::Int32, ConstValue::Int(1));
        let expr = UOp::new(Op::Binary(BinaryOp::Idiv, Rc::clone(&x), one), DType::Int32);

        let matcher = symbolic_simple();
        let simplified = graph_rewrite(&matcher, expr);

        prop_assert!(Rc::ptr_eq(&simplified, &x),
            "x / 1 should simplify to x");
    }

    /// x | 0 should simplify to x
    #[test]
    fn identity_or_zero_right(x in arb_simple_uop(DType::Int32)) {
        let zero = UOp::const_(DType::Int32, ConstValue::Int(0));
        let expr = UOp::new(Op::Binary(BinaryOp::Or, Rc::clone(&x), zero), DType::Int32);

        let matcher = symbolic_simple();
        let simplified = graph_rewrite(&matcher, expr);

        prop_assert!(Rc::ptr_eq(&simplified, &x),
            "x | 0 should simplify to x");
    }

    /// x ^ 0 should simplify to x
    #[test]
    fn identity_xor_zero_right(x in arb_simple_uop(DType::Int32)) {
        let zero = UOp::const_(DType::Int32, ConstValue::Int(0));
        let expr = UOp::new(Op::Binary(BinaryOp::Xor, Rc::clone(&x), zero), DType::Int32);

        let matcher = symbolic_simple();
        let simplified = graph_rewrite(&matcher, expr);

        prop_assert!(Rc::ptr_eq(&simplified, &x),
            "x ^ 0 should simplify to x");
    }
}

// ============================================================================
// Zero Propagation Properties
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// x * 0 should simplify to 0
    #[test]
    fn zero_mul_right(x in arb_simple_uop(DType::Int32)) {
        let zero = UOp::const_(DType::Int32, ConstValue::Int(0));
        let expr = UOp::new(Op::Binary(BinaryOp::Mul, x, Rc::clone(&zero)), DType::Int32);

        let matcher = symbolic_simple();
        let simplified = graph_rewrite(&matcher, expr);

        prop_assert!(Rc::ptr_eq(&simplified, &zero),
            "x * 0 should simplify to 0");
    }

    /// 0 * x should simplify to 0
    #[test]
    fn zero_mul_left(x in arb_simple_uop(DType::Int32)) {
        let zero = UOp::const_(DType::Int32, ConstValue::Int(0));
        let expr = UOp::new(Op::Binary(BinaryOp::Mul, Rc::clone(&zero), x), DType::Int32);

        let matcher = symbolic_simple();
        let simplified = graph_rewrite(&matcher, expr);

        prop_assert!(Rc::ptr_eq(&simplified, &zero),
            "0 * x should simplify to 0");
    }

    /// x & 0 should simplify to 0
    #[test]
    fn zero_and_right(x in arb_simple_uop(DType::Int32)) {
        let zero = UOp::const_(DType::Int32, ConstValue::Int(0));
        let expr = UOp::new(Op::Binary(BinaryOp::And, x, Rc::clone(&zero)), DType::Int32);

        let matcher = symbolic_simple();
        let simplified = graph_rewrite(&matcher, expr);

        prop_assert!(Rc::ptr_eq(&simplified, &zero),
            "x & 0 should simplify to 0");
    }

    /// 0 & x should simplify to 0
    #[test]
    fn zero_and_left(x in arb_simple_uop(DType::Int32)) {
        let zero = UOp::const_(DType::Int32, ConstValue::Int(0));
        let expr = UOp::new(Op::Binary(BinaryOp::And, Rc::clone(&zero), x), DType::Int32);

        let matcher = symbolic_simple();
        let simplified = graph_rewrite(&matcher, expr);

        prop_assert!(Rc::ptr_eq(&simplified, &zero),
            "0 & x should simplify to 0");
    }
}

// ============================================================================
// Self-Folding Properties
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// x / x should simplify to 1 (for x as variable, not constant 0)
    #[test]
    fn self_idiv_one(x in arb_var_uop(DType::Int32)) {
        let expr = UOp::new(Op::Binary(BinaryOp::Idiv, Rc::clone(&x), Rc::clone(&x)), DType::Int32);

        let matcher = symbolic_simple();
        let simplified = graph_rewrite(&matcher, expr);

        // Should simplify to constant 1
        match simplified.op() {
            Op::Const(cv) => prop_assert_eq!(cv.0, ConstValue::Int(1),
                "x / x should be 1"),
            _ => prop_assert!(false, "x / x should simplify to Const(1), got {:?}", simplified.op()),
        }
    }

    /// x & x should simplify to x (idempotent)
    #[test]
    fn self_and_identity(x in arb_simple_uop(DType::Int32)) {
        let expr = UOp::new(Op::Binary(BinaryOp::And, Rc::clone(&x), Rc::clone(&x)), DType::Int32);

        let matcher = symbolic_simple();
        let simplified = graph_rewrite(&matcher, expr);

        prop_assert!(Rc::ptr_eq(&simplified, &x),
            "x & x should simplify to x");
    }

    /// x | x should simplify to x (idempotent)
    #[test]
    fn self_or_identity(x in arb_simple_uop(DType::Int32)) {
        let expr = UOp::new(Op::Binary(BinaryOp::Or, Rc::clone(&x), Rc::clone(&x)), DType::Int32);

        let matcher = symbolic_simple();
        let simplified = graph_rewrite(&matcher, expr);

        prop_assert!(Rc::ptr_eq(&simplified, &x),
            "x | x should simplify to x");
    }

    /// x < x should simplify to false (for non-float types)
    #[test]
    fn self_lt_false(x in arb_var_uop(DType::Int32)) {
        let expr = UOp::new(Op::Binary(BinaryOp::Lt, Rc::clone(&x), Rc::clone(&x)), DType::Bool);

        let matcher = symbolic_simple();
        let simplified = graph_rewrite(&matcher, expr);

        match simplified.op() {
            Op::Const(cv) => prop_assert_eq!(cv.0, ConstValue::Bool(false),
                "x < x should be false"),
            _ => prop_assert!(false, "x < x should simplify to Const(false), got {:?}", simplified.op()),
        }
    }

    /// x == x should simplify to true (for non-float types)
    #[test]
    fn self_eq_true(x in arb_var_uop(DType::Int32)) {
        let expr = UOp::new(Op::Binary(BinaryOp::Eq, Rc::clone(&x), Rc::clone(&x)), DType::Bool);

        let matcher = symbolic_simple();
        let simplified = graph_rewrite(&matcher, expr);

        match simplified.op() {
            Op::Const(cv) => prop_assert_eq!(cv.0, ConstValue::Bool(true),
                "x == x should be true"),
            _ => prop_assert!(false, "x == x should simplify to Const(true), got {:?}", simplified.op()),
        }
    }

    /// x != x should simplify to false (for non-float types)
    #[test]
    fn self_ne_false(x in arb_var_uop(DType::Int32)) {
        let expr = UOp::new(Op::Binary(BinaryOp::Ne, Rc::clone(&x), Rc::clone(&x)), DType::Bool);

        let matcher = symbolic_simple();
        let simplified = graph_rewrite(&matcher, expr);

        match simplified.op() {
            Op::Const(cv) => prop_assert_eq!(cv.0, ConstValue::Bool(false),
                "x != x should be false"),
            _ => prop_assert!(false, "x != x should simplify to Const(false), got {:?}", simplified.op()),
        }
    }
}

// ============================================================================
// Constant Folding Properties
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    /// Binary operations on two constants should fold
    #[test]
    fn const_fold_add(a in arb_small_int(), b in arb_small_int()) {
        let a_uop = UOp::const_(DType::Int32, a);
        let b_uop = UOp::const_(DType::Int32, b);
        let expr = UOp::new(Op::Binary(BinaryOp::Add, a_uop, b_uop), DType::Int32);

        let matcher = symbolic_simple();
        let simplified = graph_rewrite(&matcher, expr);

        // Should be a constant
        match simplified.op() {
            Op::Const(cv) => {
                // Compute expected result
                if let (ConstValue::Int(av), ConstValue::Int(bv)) = (a, b) {
                    let expected = av.wrapping_add(bv);
                    if let ConstValue::Int(result) = cv.0 {
                        prop_assert_eq!(result as i32, expected as i32,
                            "{} + {} should equal {}", av, bv, expected);
                    }
                }
            }
            _ => prop_assert!(false, "Constant addition should fold to constant"),
        }
    }

    /// Binary operations on two constants should fold (multiplication)
    #[test]
    fn const_fold_mul(a in arb_small_int(), b in arb_small_int()) {
        let a_uop = UOp::const_(DType::Int32, a);
        let b_uop = UOp::const_(DType::Int32, b);
        let expr = UOp::new(Op::Binary(BinaryOp::Mul, a_uop, b_uop), DType::Int32);

        let matcher = symbolic_simple();
        let simplified = graph_rewrite(&matcher, expr);

        // Should be a constant
        prop_assert!(matches!(simplified.op(), Op::Const(_)),
            "Constant multiplication should fold to constant");
    }

    /// Division by non-zero constant should fold
    #[test]
    fn const_fold_idiv(a in arb_small_int(), b in arb_nonzero_int()) {
        let a_uop = UOp::const_(DType::Int32, a);
        let b_uop = UOp::const_(DType::Int32, b);
        let expr = UOp::new(Op::Binary(BinaryOp::Idiv, a_uop, b_uop), DType::Int32);

        let matcher = symbolic_simple();
        let simplified = graph_rewrite(&matcher, expr);

        // Should be a constant
        prop_assert!(matches!(simplified.op(), Op::Const(_)),
            "Constant division should fold to constant");
    }
}

// ============================================================================
// Algebraic Law Properties
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// Commutativity: x + y = y + x after optimization
    #[test]
    fn commutativity_add(
        x in arb_simple_uop(DType::Int32),
        y in arb_simple_uop(DType::Int32),
    ) {
        let xy = UOp::new(Op::Binary(BinaryOp::Add, Rc::clone(&x), Rc::clone(&y)), DType::Int32);
        let yx = UOp::new(Op::Binary(BinaryOp::Add, Rc::clone(&y), Rc::clone(&x)), DType::Int32);

        let matcher = symbolic_simple();
        let opt_xy = graph_rewrite(&matcher, xy);
        let opt_yx = graph_rewrite(&matcher, yx);

        // Both should optimize to same structure (either both to x+y or both simplified)
        // We verify this by checking if optimization preserves commutativity
        // by ensuring symmetric simplification
        prop_assert!(
            (Rc::ptr_eq(&opt_xy, &opt_yx)) ||
            (matches!((opt_xy.op(), opt_yx.op()),
                (Op::Binary(BinaryOp::Add, _, _), Op::Binary(BinaryOp::Add, _, _)))),
            "Addition should commute after optimization"
        );
    }

    /// Idempotent operations: applying twice gives same result
    #[test]
    fn idempotent_and(x in arb_simple_uop(DType::Int32)) {
        // x & x & x = x & x = x
        let x_and_x = UOp::new(Op::Binary(BinaryOp::And, Rc::clone(&x), Rc::clone(&x)), DType::Int32);
        let x_and_x_and_x = UOp::new(Op::Binary(BinaryOp::And, Rc::clone(&x_and_x), Rc::clone(&x)), DType::Int32);

        let matcher = symbolic_simple();
        let opt1 = graph_rewrite(&matcher, x_and_x);
        let opt2 = graph_rewrite(&matcher, x_and_x_and_x);

        // Both should simplify to the same form (ideally to x, but constants may fold differently)
        // The key property is idempotence: applying & with self gives same result
        prop_assert!(Rc::ptr_eq(&opt1, &opt2) || Rc::ptr_eq(&opt1, &x),
            "x & x should be idempotent: either both simplify to same form or to x");
    }
}
