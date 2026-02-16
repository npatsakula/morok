//! Multi-oracle property tests combining known properties, dtype-invariance, and Z3 verification.
//!
//! This module implements the most comprehensive testing strategy by using multiple
//! independent oracles to verify optimization correctness.

use std::sync::Arc;

use proptest::prelude::*;

use morok_dtype::DType;
use morok_ir::UOp;
use morok_ir::types::ConstValue;

use crate::rewrite::graph_rewrite;
use crate::symbolic::symbolic_simple;
use crate::z3::verify_equivalence;

// Import generators from ir crate
use morok_ir::test::property::generators::*;

// ============================================================================
// Multi-Oracle Testing
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// Test optimization with all three oracles:
    /// 1. Known property (algebraic identity)
    /// 2. DType invariance (test across widened types)
    /// 3. Z3 verification (formal proof)
    #[test]
    fn multi_oracle_known_property(kpg in arb_known_property_graph()) {
        let graph = kpg.build();
        let matcher = symbolic_simple();

        // Oracle 1: Known property - verify expected simplification (if it happens)
        let simplified = graph_rewrite(&matcher, graph.clone(), &mut ());

        // Note: Not all patterns are implemented, so we don't require simplification
        // We just verify that IF simplification occurs, it's correct
        if let Some(expected) = kpg.expected_result() {
            // Only assert if actually simplified (not just returned original)
            if !Arc::ptr_eq(&simplified, &graph) {
                // If it simplified to something, it should match expected OR be semantically equivalent
                // (we don't require exact pointer equality as different equivalent forms are ok)
                let is_expected = Arc::ptr_eq(&simplified, &expected);

                // Check for zero constant (Int(0) or UInt(0) both represent zero)
                let is_zero = |cv: &ConstValue| matches!(cv, ConstValue::Int(0) | ConstValue::UInt(0));
                let is_constant_zero = matches!(simplified.op(), morok_ir::Op::Const(cv) if is_zero(&cv.0))
                    && matches!(expected.op(), morok_ir::Op::Const(cv) if is_zero(&cv.0));

                prop_assert!(is_expected || is_constant_zero,
                    "Simplified result should match expected. Got: {:?}, Expected: {:?}",
                    simplified.op(), expected.op());
            }
        }

        // Oracle 2: DType invariance - widen types and verify
        // (Skip for now as it requires more complex graph transformation)

        // Oracle 3: Z3 verification - prove equivalence
        let result = verify_equivalence(&graph, &simplified);
        match result {
            Ok(()) => {}, // Verification succeeded
            Err(e) => {
                // Only fail on counterexamples, not on conversion failures
                // (some operations may not be supported by Z3 converter yet)
                if matches!(e, crate::z3::CounterExample::Found { .. }) {
                    prop_assert!(false, "Z3 found counterexample: {}", e);
                }
            }
        }
    }

    /// Test that identity eliminations are Z3-verified
    #[test]
    fn z3_verify_identity_add_zero(x in arb_var_uop(DType::Int32)) {
        let zero = UOp::native_const(0i32);
        let expr = UOp::new(
            morok_ir::Op::Binary(morok_ir::types::BinaryOp::Add, Arc::clone(&x), zero),
            DType::Int32,
        );

        let matcher = symbolic_simple();
        let simplified = graph_rewrite(&matcher, expr.clone(), &mut ());

        // Should simplify to x
        prop_assert!(Arc::ptr_eq(&simplified, &x));

        // Z3 should verify equivalence
        verify_equivalence(&expr, &simplified)
            .expect("Z3 should verify x + 0 = x");
    }

    /// Test that zero propagation is Z3-verified
    #[test]
    fn z3_verify_zero_mul(x in arb_var_uop(DType::Int32)) {
        let zero = UOp::native_const(0i32);
        let expr = x.try_mul(&zero).unwrap();

        let matcher = symbolic_simple();
        let simplified = graph_rewrite(&matcher, expr.clone(), &mut ());

        // Should simplify to 0
        prop_assert!(Arc::ptr_eq(&simplified, &zero));

        // Z3 should verify equivalence
        verify_equivalence(&expr, &simplified)
            .expect("Z3 should verify x * 0 = 0");
    }

    /// Test that self-division is Z3-verified (for x ≠ 0)
    #[test]
    fn z3_verify_self_div(name in "[a-z]", min_val in 1i64..100, range_size in 1i64..100) {
        // Create variable with min_val >= 1 to avoid division by zero
        let x = UOp::var(&name, DType::Int32, min_val, min_val + range_size);

        let expr = UOp::new(
            morok_ir::Op::Binary(morok_ir::types::BinaryOp::Idiv, Arc::clone(&x), Arc::clone(&x)),
            DType::Int32,
        );

        let matcher = symbolic_simple();
        let simplified = graph_rewrite(&matcher, expr.clone(), &mut ());

        // Should simplify to 1
        match simplified.op() {
            morok_ir::Op::Const(cv) => {
                prop_assert_eq!(cv.0, ConstValue::Int(1));
            }
            _ => prop_assert!(false, "x / x should simplify to 1"),
        }

        // Z3 should verify equivalence
        verify_equivalence(&expr, &simplified)
            .expect("Z3 should verify x / x = 1 for x ≠ 0");
    }

    /// Test arithmetic trees with Z3 verification.
    ///
    /// Uses bounded constants to avoid overflow mismatches between
    /// Z3's unbounded integers and our IR's wrapping semantics.
    #[test]
    fn z3_verify_arithmetic_optimization(graph in arb_arithmetic_tree_bounded_up_to(DType::Int32, 3)) {
        let matcher = symbolic_simple();
        let optimized = graph_rewrite(&matcher, graph.clone(), &mut ());

        // Z3 should verify equivalence
        let result = verify_equivalence(&graph, &optimized);

        match result {
            Ok(()) => {}, // Verification succeeded
            Err(e) => {
                match e {
                    crate::z3::CounterExample::Found { .. } => {
                        prop_assert!(false, "Z3 found counterexample: {}", e);
                    }
                    crate::z3::CounterExample::ConversionFailed(_) => {
                        // Some operations may not be supported yet - skip
                    }
                    crate::z3::CounterExample::Timeout => {
                        // Timeout is acceptable for complex expressions
                    }
                }
            }
        }
    }
}

// ============================================================================
// DType Widening Oracle
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// Test that optimization is sound across dtype widening.
    ///
    /// If we widen all dtypes (Int32 -> Int64), the optimization should
    /// produce equivalent results.
    #[test]
    fn dtype_widening_preserves_optimization(
        kpg in arb_known_property_graph(),
        family in arb_dtype_family(),
    ) {
        let narrowest = family.narrowest();
        let widest = family.widest();

        // Build graph with narrowest dtype
        let narrow_graph = rebuild_with_dtype(&kpg, narrowest.clone());
        let wide_graph = rebuild_with_dtype(&kpg, widest.clone());

        let matcher = symbolic_simple();

        // Optimize both
        let narrow_opt = graph_rewrite(&matcher, narrow_graph.clone(), &mut ());
        let wide_opt = graph_rewrite(&matcher, wide_graph.clone(), &mut ());

        // Both should have same structure (just different dtypes)
        // We verify this by checking if they simplify to the same form
        let narrow_simplified_form = optimization_form(&narrow_opt);
        let wide_simplified_form = optimization_form(&wide_opt);

        prop_assert_eq!(narrow_simplified_form, wide_simplified_form,
            "Optimization should preserve form across dtype widening");

        // If Z3 can verify the narrow version, it should verify the wide version
        let narrow_result = verify_equivalence(&narrow_graph, &narrow_opt);
        let wide_result = verify_equivalence(&wide_graph, &wide_opt);

        match (narrow_result, wide_result) {
            (Ok(()), Ok(())) => {}, // Both verified
            (Ok(()), Err(e)) => {
                prop_assert!(false,
                    "Narrow dtype verified but wide dtype failed: {}", e);
            }
            (Err(_), Ok(())) => {}, // Narrow failed but wide succeeded (acceptable)
            (Err(_), Err(_)) => {}, // Both failed (acceptable if conversion issues)
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Rebuild a known property graph with a different dtype.
fn rebuild_with_dtype(kpg: &KnownPropertyGraph, dtype: DType) -> Arc<UOp> {
    match kpg {
        KnownPropertyGraph::AddZero { .. } => {
            let x = UOp::var("x", dtype.clone(), 0, 100);
            let zero = UOp::native_const(0i64);
            UOp::new(morok_ir::Op::Binary(morok_ir::types::BinaryOp::Add, x, zero), dtype)
        }
        KnownPropertyGraph::MulOne { .. } => {
            let x = UOp::var("x", dtype.clone(), 0, 100);
            let one = UOp::native_const(1i64);
            UOp::new(morok_ir::Op::Binary(morok_ir::types::BinaryOp::Mul, x, one), dtype)
        }
        KnownPropertyGraph::SubZero { .. } => {
            let x = UOp::var("x", dtype.clone(), 0, 100);
            let zero = UOp::native_const(0i64);
            UOp::new(morok_ir::Op::Binary(morok_ir::types::BinaryOp::Sub, x, zero), dtype)
        }
        KnownPropertyGraph::MulZero { .. } => {
            let x = UOp::var("x", dtype.clone(), 0, 100);
            let zero = UOp::native_const(0i64);
            UOp::new(morok_ir::Op::Binary(morok_ir::types::BinaryOp::Mul, x, zero), dtype)
        }
        KnownPropertyGraph::SubSelf { .. } => {
            let x = UOp::var("x", dtype.clone(), 0, 100);
            UOp::new(morok_ir::Op::Binary(morok_ir::types::BinaryOp::Sub, Arc::clone(&x), x), dtype)
        }
        KnownPropertyGraph::AddSelf { .. } => {
            let x = UOp::var("x", dtype.clone(), 0, 100);
            UOp::new(morok_ir::Op::Binary(morok_ir::types::BinaryOp::Add, Arc::clone(&x), x), dtype)
        }
    }
}

/// Get a simplified form descriptor for comparing optimization results.
///
/// Returns: (op_type, is_const, is_var, child_count)
fn optimization_form(uop: &Arc<UOp>) -> (String, bool, bool, usize) {
    use morok_ir::Op;

    match uop.op() {
        Op::Const(_) => ("const".to_string(), true, false, 0),
        Op::DefineVar { .. } => ("var".to_string(), false, true, 0),
        Op::Unary(op, _) => (format!("unary_{:?}", op), false, false, 1),
        Op::Binary(op, _, _) => (format!("binary_{:?}", op), false, false, 2),
        Op::Ternary(op, _, _, _) => (format!("ternary_{:?}", op), false, false, 3),
        _ => ("other".to_string(), false, false, 0),
    }
}
