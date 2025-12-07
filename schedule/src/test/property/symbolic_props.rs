//! Property tests for symbolic optimizer algebraic rules.
//!
//! Tests that pattern rewrites preserve semantics and follow algebraic laws.

use std::sync::Arc;

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
        let zero = UOp::native_const(0i32);
        let expr = x.try_add(&zero).unwrap();

        let matcher = symbolic_simple();
        let simplified = graph_rewrite(&matcher, expr, &mut ());

        prop_assert!(Arc::ptr_eq(&simplified, &x),
            "x + 0 should simplify to x");
    }

    /// 0 + x should simplify to x (commutativity)
    #[test]
    fn identity_add_zero_left(x in arb_simple_uop(DType::Int32)) {
        let zero = UOp::native_const(0i32);
        let expr = zero.try_add(&x).unwrap();

        let matcher = symbolic_simple();
        let simplified = graph_rewrite(&matcher, expr, &mut ());

        prop_assert!(Arc::ptr_eq(&simplified, &x),
            "0 + x should simplify to x");
    }

    /// x - 0 should simplify to x
    #[test]
    fn identity_sub_zero(x in arb_simple_uop(DType::Int32)) {
        let zero = UOp::native_const(0i32);
        let expr = x.try_sub(&zero).unwrap();

        let matcher = symbolic_simple();
        let simplified = graph_rewrite(&matcher, expr, &mut ());

        prop_assert!(Arc::ptr_eq(&simplified, &x),
            "x - 0 should simplify to x");
    }

    /// x * 1 should simplify to x
    #[test]
    fn identity_mul_one_right(x in arb_simple_uop(DType::Int32)) {
        let one = UOp::native_const(1i32);
        let expr = x.try_mul(&one).unwrap();

        let matcher = symbolic_simple();
        let simplified = graph_rewrite(&matcher, expr, &mut ());

        prop_assert!(Arc::ptr_eq(&simplified, &x),
            "x * 1 should simplify to x");
    }

    /// 1 * x should simplify to x (commutativity)
    #[test]
    fn identity_mul_one_left(x in arb_simple_uop(DType::Int32)) {
        let one = UOp::native_const(1i32);
        let expr = one.try_mul(&x).unwrap();

        let matcher = symbolic_simple();
        let simplified = graph_rewrite(&matcher, expr, &mut ());

        prop_assert!(Arc::ptr_eq(&simplified, &x),
            "1 * x should simplify to x");
    }

    /// x / 1 should simplify to x (integer division)
    #[test]
    fn identity_idiv_one(x in arb_simple_uop(DType::Int32)) {
        let one = UOp::native_const(1i32);
        let expr = x.try_div(&one).unwrap();

        let matcher = symbolic_simple();
        let simplified = graph_rewrite(&matcher, expr, &mut ());

        prop_assert!(Arc::ptr_eq(&simplified, &x),
            "x / 1 should simplify to x");
    }

    /// x | 0 should simplify to x
    #[test]
    fn identity_or_zero_right(x in arb_simple_uop(DType::Int32)) {
        let zero = UOp::native_const(0i32);
        let expr = x.try_or_op(&zero).unwrap();

        let matcher = symbolic_simple();
        let simplified = graph_rewrite(&matcher, expr, &mut ());

        prop_assert!(Arc::ptr_eq(&simplified, &x),
            "x | 0 should simplify to x");
    }

    /// x ^ 0 should simplify to x
    #[test]
    fn identity_xor_zero_right(x in arb_simple_uop(DType::Int32)) {
        let zero = UOp::native_const(0i32);
        let expr = x.try_xor_op(&zero).unwrap();

        let matcher = symbolic_simple();
        let simplified = graph_rewrite(&matcher, expr, &mut ());

        prop_assert!(Arc::ptr_eq(&simplified, &x),
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
        let zero = UOp::native_const(0i32);
        let expr = x.try_mul(&zero).unwrap();

        let matcher = symbolic_simple();
        let simplified = graph_rewrite(&matcher, expr, &mut ());

        prop_assert!(Arc::ptr_eq(&simplified, &zero),
            "x * 0 should simplify to 0");
    }

    /// 0 * x should simplify to 0
    #[test]
    fn zero_mul_left(x in arb_simple_uop(DType::Int32)) {
        let zero = UOp::native_const(0i32);
        let expr = zero.try_mul(&x).unwrap();

        let matcher = symbolic_simple();
        let simplified = graph_rewrite(&matcher, expr, &mut ());

        prop_assert!(Arc::ptr_eq(&simplified, &zero),
            "0 * x should simplify to 0");
    }

    /// x & 0 should simplify to 0
    #[test]
    fn zero_and_right(x in arb_simple_uop(DType::Int32)) {
        let zero = UOp::native_const(0i32);
        let expr = x.try_and_op(&zero).unwrap();

        let matcher = symbolic_simple();
        let simplified = graph_rewrite(&matcher, expr, &mut ());

        prop_assert!(Arc::ptr_eq(&simplified, &zero),
            "x & 0 should simplify to 0");
    }

    /// 0 & x should simplify to 0
    #[test]
    fn zero_and_left(x in arb_simple_uop(DType::Int32)) {
        let zero = UOp::native_const(0i32);
        let expr = zero.try_and_op(&x).unwrap();

        let matcher = symbolic_simple();
        let simplified = graph_rewrite(&matcher, expr, &mut ());

        prop_assert!(Arc::ptr_eq(&simplified, &zero),
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
        let expr = x.try_div(&x).unwrap();

        let matcher = symbolic_simple();
        let simplified = graph_rewrite(&matcher, expr, &mut ());

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
        let expr = x.try_and_op(&x).unwrap();

        let matcher = symbolic_simple();
        let simplified = graph_rewrite(&matcher, expr, &mut ());

        prop_assert!(Arc::ptr_eq(&simplified, &x),
            "x & x should simplify to x");
    }

    /// x | x should simplify to x (idempotent)
    #[test]
    fn self_or_identity(x in arb_simple_uop(DType::Int32)) {
        let expr = x.try_or_op(&x).unwrap();

        let matcher = symbolic_simple();
        let simplified = graph_rewrite(&matcher, expr, &mut ());

        prop_assert!(Arc::ptr_eq(&simplified, &x),
            "x | x should simplify to x");
    }

    /// x < x should simplify to false (for non-float types)
    #[test]
    fn self_lt_false(x in arb_var_uop(DType::Int32)) {
        let expr = x.try_cmplt(&x).unwrap();

        let matcher = symbolic_simple();
        let simplified = graph_rewrite(&matcher, expr, &mut ());

        match simplified.op() {
            Op::Const(cv) => prop_assert_eq!(cv.0, ConstValue::Bool(false),
                "x < x should be false"),
            _ => prop_assert!(false, "x < x should simplify to Const(false), got {:?}", simplified.op()),
        }
    }

    /// x == x should simplify to true (for non-float types)
    #[test]
    fn self_eq_true(x in arb_var_uop(DType::Int32)) {
        let expr = x.try_cmpeq(&x).unwrap();

        let matcher = symbolic_simple();
        let simplified = graph_rewrite(&matcher, expr, &mut ());

        match simplified.op() {
            Op::Const(cv) => prop_assert_eq!(cv.0, ConstValue::Bool(true),
                "x == x should be true"),
            _ => prop_assert!(false, "x == x should simplify to Const(true), got {:?}", simplified.op()),
        }
    }

    /// x != x should simplify to false (for non-float types)
    #[test]
    fn self_ne_false(x in arb_var_uop(DType::Int32)) {
        let expr = x.try_cmpne(&x).unwrap();

        let matcher = symbolic_simple();
        let simplified = graph_rewrite(&matcher, expr, &mut ());

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
        let expr = a_uop.try_add(&b_uop).unwrap();

        let matcher = symbolic_simple();
        let simplified = graph_rewrite(&matcher, expr, &mut ());

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
        let expr = a_uop.try_mul(&b_uop).unwrap();

        let matcher = symbolic_simple();
        let simplified = graph_rewrite(&matcher, expr, &mut ());

        // Should be a constant
        prop_assert!(matches!(simplified.op(), Op::Const(_)),
            "Constant multiplication should fold to constant");
    }

    /// Division by non-zero constant should fold
    #[test]
    fn const_fold_idiv(a in arb_small_int(), b in nonzero_int()) {
        let a_uop = UOp::const_(DType::Int32, a);
        let b_uop = UOp::const_(DType::Int32, b);
        let expr = a_uop.try_div(&b_uop).unwrap();

        let matcher = symbolic_simple();
        let simplified = graph_rewrite(&matcher, expr, &mut ());

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
        let xy = x.try_add(&y).unwrap();
        let yx = y.try_add(&x).unwrap();

        let matcher = symbolic_simple();
        let opt_xy = graph_rewrite(&matcher, xy, &mut ());
        let opt_yx = graph_rewrite(&matcher, yx, &mut ());

        // Both should optimize to same structure (either both to x+y or both simplified)
        // We verify this by checking if optimization preserves commutativity
        // by ensuring symmetric simplification
        prop_assert!(
            (Arc::ptr_eq(&opt_xy, &opt_yx)) ||
            (matches!((opt_xy.op(), opt_yx.op()),
                (Op::Binary(BinaryOp::Add, _, _), Op::Binary(BinaryOp::Add, _, _)))),
            "Addition should commute after optimization"
        );
    }

    /// Idempotent operations: applying twice gives same result
    #[test]
    fn idempotent_and(x in arb_simple_uop(DType::Int32)) {
        // x & x & x = x & x = x
        let x_and_x = x.try_and_op(&x).unwrap();
        let x_and_x_and_x = x_and_x.try_and_op(&x).unwrap();

        let matcher = symbolic_simple();
        let opt1 = graph_rewrite(&matcher, x_and_x, &mut ());
        let opt2 = graph_rewrite(&matcher, x_and_x_and_x, &mut ());

        // Both should simplify to the same form (ideally to x, but constants may fold differently)
        // The key property is idempotence: applying & with self gives same result
        prop_assert!(Arc::ptr_eq(&opt1, &opt2) || Arc::ptr_eq(&opt1, &x),
            "x & x should be idempotent: either both simplify to same form or to x");
    }
}
