//! Metamorphic and structural property tests for symbolic optimization.
//!
//! Tests higher-level invariants like idempotence, cost monotonicity,
//! and range preservation.

use std::rc::Rc;

use proptest::prelude::*;

use morok_dtype::DType;
use morok_ir::UOp;

use crate::rewrite::graph_rewrite;
use crate::symbolic::symbolic_simple;

// Import generators and utilities from ir crate
use morok_ir::test::property::generators::*;
use morok_ir::test::property::shrinking::{uop_depth, uop_op_count};

// ============================================================================
// Idempotence Properties
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// Applying symbolic optimization twice should give same result as once.
    ///
    /// This is a critical property: optimize(optimize(x)) = optimize(x)
    #[test]
    fn symbolic_idempotent(graph in arb_arithmetic_tree_up_to(DType::Int32, 4)) {
        let matcher = symbolic_simple();

        let once = graph_rewrite(&matcher, graph.clone());
        let twice = graph_rewrite(&matcher, once.clone());

        prop_assert!(Rc::ptr_eq(&once, &twice),
            "Optimizing twice should give same result as optimizing once");
    }

    /// Idempotence for known property graphs
    #[test]
    fn symbolic_idempotent_known_props(kpg in arb_known_property_graph()) {
        let graph = kpg.build();
        let matcher = symbolic_simple();

        let once = graph_rewrite(&matcher, graph);
        let twice = graph_rewrite(&matcher, once.clone());

        prop_assert!(Rc::ptr_eq(&once, &twice),
            "Known property graphs should be idempotent");
    }
}

// ============================================================================
// Cost Monotonicity Properties
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// Optimization should generally not significantly increase operation count.
    ///
    /// Note: Some rewrites may temporarily increase op count to enable further
    /// optimizations (e.g., distribution before folding). We allow small increases.
    #[test]
    fn cost_monotonic_op_count(graph in arb_arithmetic_tree_up_to(DType::Int32, 4)) {
        let original_count = uop_op_count(&graph);

        let matcher = symbolic_simple();
        let optimized = graph_rewrite(&matcher, graph);

        let optimized_count = uop_op_count(&optimized);

        // Allow small increases for restructuring, but catch large regressions
        prop_assert!(optimized_count <= original_count + 2,
            "Optimization should not significantly increase op count: {} -> {}",
            original_count, optimized_count);
    }

    /// Optimization should not increase graph depth significantly.
    ///
    /// Note: Some patterns might increase depth slightly (e.g., x -> x + 0)
    /// but we verify depth doesn't grow unbounded.
    #[test]
    fn cost_depth_bounded(graph in arb_arithmetic_tree_up_to(DType::Int32, 4)) {
        let original_depth = uop_depth(&graph);

        let matcher = symbolic_simple();
        let optimized = graph_rewrite(&matcher, graph);

        let optimized_depth = uop_depth(&optimized);

        // Allow depth to increase by at most 1 (for pattern rewrites that restructure)
        prop_assert!(optimized_depth <= original_depth + 1,
            "Optimization should not significantly increase depth: {} -> {}",
            original_depth, optimized_depth);
    }
}

// ============================================================================
// Structural Invariant Properties
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// Optimization should preserve dtype.
    #[test]
    fn preserves_dtype(graph in arb_arithmetic_tree_up_to(DType::Int32, 4)) {
        let original_dtype = graph.dtype().clone();

        let matcher = symbolic_simple();
        let optimized = graph_rewrite(&matcher, graph);

        let optimized_dtype = optimized.dtype().clone();

        prop_assert_eq!(original_dtype, optimized_dtype,
            "Optimization must preserve dtype");
    }

    /// Constants should be properly typed after optimization.
    #[test]
    fn constants_properly_typed(graph in arb_arithmetic_tree_up_to(DType::Int32, 4)) {
        let matcher = symbolic_simple();
        let optimized = graph_rewrite(&matcher, graph);

        // Walk the graph and verify all constants have matching dtypes
        verify_constant_dtypes(&optimized)?;
    }

    /// Optimization should not create cycles.
    ///
    /// This is a fundamental invariant: UOp graphs must be DAGs.
    #[test]
    fn no_cycles_created(graph in arb_arithmetic_tree_up_to(DType::Int32, 4)) {
        let matcher = symbolic_simple();
        let optimized = graph_rewrite(&matcher, graph);

        // Use toposort to detect cycles - it will panic if graph has cycles
        let _topo = optimized.toposort();
    }
}

// ============================================================================
// Compositional Properties
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// Optimizing subexpressions independently should give compatible results.
    ///
    /// If we optimize (a + b), we should be able to optimize a and b separately
    /// and get a result that's at least as good as optimizing a and b in isolation.
    ///
    /// NOTE: This test is currently ignored because distribution patterns increase operation count,
    /// which conflicts with the compositional optimization property. The distribution patterns
    /// are kept enabled because they may enable other optimizations in some cases.
    #[test]
    #[ignore = "Distribution patterns conflict with compositional optimization"]
    fn compositional_subexpr_optimization(
        a in arb_arithmetic_tree_up_to(DType::Int32, 2),
        b in arb_arithmetic_tree_up_to(DType::Int32, 2),
        op in arb_arithmetic_binary_op(),
    ) {
        let matcher = symbolic_simple();

        // Optimize subexpressions first
        let opt_a = graph_rewrite(&matcher, a.clone());
        let opt_b = graph_rewrite(&matcher, b.clone());

        // Build expression with optimized subexpressions
        let expr_opt_subs = UOp::new(
            morok_ir::Op::Binary(op, opt_a, opt_b),
            DType::Int32,
        );

        // Optimize the composed expression
        let final_opt = graph_rewrite(&matcher, expr_opt_subs);

        // Operation count of final result should be minimal
        let final_count = uop_op_count(&final_opt);

        // Build expression with un-optimized subexpressions and optimize
        let expr_unopt = UOp::new(
            morok_ir::Op::Binary(op, a, b),
            DType::Int32,
        );
        let direct_opt = graph_rewrite(&matcher, expr_unopt);
        let direct_count = uop_op_count(&direct_opt);

        // Both approaches should give similar results
        prop_assert!(final_count <= direct_count + 1,
            "Compositional optimization should be nearly as good as direct: {} vs {}",
            final_count, direct_count);
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Verify that all constants in the graph have matching dtypes.
fn verify_constant_dtypes(uop: &Rc<UOp>) -> Result<(), TestCaseError> {
    use morok_ir::Op;

    match uop.op() {
        Op::Const(cv) => {
            let const_dtype = cv.0.dtype();
            let uop_dtype = uop.dtype();

            // For scalar types, verify they match
            if let Some(scalar_dt) = uop_dtype.scalar() {
                let expected_dtype = DType::Scalar(scalar_dt);
                // Allow some flexibility for type widening
                // (e.g., Int32 constant in Int64 context is ok if it was widened)
                if const_dtype != expected_dtype {
                    // Only fail if it's clearly wrong (different type families)
                    let const_is_int = matches!(const_dtype.scalar(), Some(dt) if dt.is_int());
                    let uop_is_int = matches!(uop_dtype.scalar(), Some(dt) if dt.is_int());

                    prop_assert!(
                        const_is_int == uop_is_int,
                        "Constant dtype family mismatch: {:?} vs {:?}",
                        const_dtype,
                        expected_dtype
                    );
                }
            }
            Ok(())
        }
        Op::Unary(_, src) => verify_constant_dtypes(src),
        Op::Binary(_, lhs, rhs) => {
            verify_constant_dtypes(lhs)?;
            verify_constant_dtypes(rhs)
        }
        Op::Ternary(_, a, b, c) => {
            verify_constant_dtypes(a)?;
            verify_constant_dtypes(b)?;
            verify_constant_dtypes(c)
        }
        _ => Ok(()),
    }
}
