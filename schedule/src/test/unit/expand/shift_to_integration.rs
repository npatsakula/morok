//! Integration tests for shift_to → expand.rs pipeline.
//!
//! These tests verify the complete optimization path:
//! OptOps (UNROLL/UPCAST) → shift_to substitution → pre_expand → fix_reduce_unroll
//!
//! The key transformation is:
//! 1. shift_to replaces Range with arithmetic: `replaced_rng * amount + new_rng`
//! 2. fix_reduce_unroll extracts the reduce range and moves unroll/upcast to CONTRACT
//! 3. do_contract collapses CONTRACT(UNROLL) via GEP swizzle

use std::sync::Arc;

use morok_dtype::DType;
use morok_ir::types::ConstValue;
use morok_ir::{AxisId, AxisType, BinaryOp, Op, ReduceOp, UOp};
use smallvec::smallvec;

use crate::expand::pre_expand;
use crate::optimizer::Renderer;
use crate::optimizer::Scheduler;

/// Helper: Create a simple reduction sum([0..size]).
fn create_simple_reduce(size: usize, axis_id: usize) -> Arc<UOp> {
    let end = UOp::const_(DType::Index, ConstValue::Int(size as i64));
    let range = UOp::range_axis(end, AxisId::Renumbered(axis_id), AxisType::Reduce);
    let src = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    src.reduce(smallvec![range], ReduceOp::Add)
}

/// Helper: Manually create the shift_to substitution expression.
///
/// shift_to(rng, amount, new_type, top=false) creates:
///   replaced_rng * amount + new_rng
///
/// where:
/// - replaced_rng has size = old_size / amount (same axis_id/type as original)
/// - new_rng has size = amount (new axis_id, new_type)
fn create_shift_to_expr(
    original_axis_id: usize,
    original_size: usize,
    amount: usize,
    new_axis_id: usize,
    new_type: AxisType,
) -> (Arc<UOp>, Arc<UOp>, Arc<UOp>) {
    let new_size = original_size / amount;

    // replaced_rng: Reduce range with reduced size
    let replaced_end = UOp::const_(DType::Index, ConstValue::Int(new_size as i64));
    let replaced_rng = UOp::range_axis(replaced_end, AxisId::Renumbered(original_axis_id), AxisType::Reduce);

    // new_rng: New range with the unroll/upcast type
    let new_end = UOp::const_(DType::Index, ConstValue::Int(amount as i64));
    let new_rng = UOp::range_axis(new_end, AxisId::Renumbered(new_axis_id), new_type);

    // Substitution expression: replaced_rng * amount + new_rng
    let amount_const = UOp::const_(DType::Index, ConstValue::Int(amount as i64));
    let mul = UOp::new(Op::Binary(BinaryOp::Mul, replaced_rng.clone(), amount_const), DType::Index);
    let sub_expr = UOp::new(Op::Binary(BinaryOp::Add, mul, new_rng.clone()), DType::Index);

    (replaced_rng, new_rng, sub_expr)
}

// =============================================================================
// Test 1: UNROLL OptOp → expand pipeline
// =============================================================================

/// Test that shift_to with UNROLL type creates expressions that expand correctly.
///
/// Simulates: REDUCE([0..16]) with UNROLL(0, 4)
/// Expected flow:
/// 1. shift_to creates: reduce_rng(4) * 4 + Range(Unroll, 4)
/// 2. fix_reduce_unroll extracts reduce_rng, moves Unroll to CONTRACT
/// 3. REDUCE.src gets wrapped in CONTRACT
#[test]
fn test_expand_with_shift_to_unroll() {
    // Create the shift_to substitution expression manually
    // Original: Range(Reduce, 16), UNROLL by 4 → Range(Reduce, 4) * 4 + Range(Unroll, 4)
    let (_replaced_rng, _new_rng, sub_expr) = create_shift_to_expr(
        0,  // original axis_id
        16, // original size
        4,  // amount (unroll factor)
        1,  // new axis_id for unroll range
        AxisType::Unroll,
    );

    // Create REDUCE with the substitution expression in ranges
    let src = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    let reduce = src.reduce(smallvec![sub_expr], ReduceOp::Add);

    // Run pre_expand (full pipeline)
    let result = pre_expand(&reduce);

    // Verify the result
    match result.op() {
        Op::Reduce { src: fixed_src, ranges, .. } => {
            // 1. Ranges should contain only the reduced Range(Reduce, 4)
            // The Unroll range should be moved to CONTRACT wrapper
            assert!(
                ranges.iter().any(|r| {
                    if let Op::Range { axis_type, end, .. } = r.op() {
                        *axis_type == AxisType::Reduce
                            && matches!(end.op(), Op::Const(cv) if cv.0 == ConstValue::Int(4))
                    } else {
                        false
                    }
                }),
                "Should have reduced Range(Reduce, 4), got ranges: {:?}",
                ranges.iter().map(|r| r.op()).collect::<Vec<_>>()
            );

            // 2. Source should be wrapped (CONTRACT or expanded VECTORIZE)
            // CONTRACT(Const) → VECTORIZE after do_contract
            assert!(
                matches!(fixed_src.op(), Op::Contract { .. } | Op::Vectorize { .. }),
                "Source should be CONTRACT or VECTORIZE (expanded), got {:?}",
                fixed_src.op()
            );

            // 3. No UNROLL or arithmetic expressions should remain in ranges
            for range in ranges.iter() {
                assert!(
                    !matches!(range.op(), Op::Unroll { .. } | Op::Binary(..)),
                    "Ranges should not contain UNROLL or Binary, got {:?}",
                    range.op()
                );
            }
        }
        other => panic!("Expected REDUCE, got {:?}", other),
    }
}

// =============================================================================
// Test 2: UPCAST OptOp → expand pipeline
// =============================================================================

/// Test that shift_to with UPCAST type creates vectorized accumulators.
///
/// Simulates: mat[M,K].sum(axis=K) with UPCAST(K_axis, 4)
/// Expected flow:
/// 1. shift_to creates: reduce_rng(K/4) * 4 + Range(Upcast, 4)
/// 2. fix_reduce_unroll extracts reduce_rng, detects Upcast → Vector dtype
/// 3. REDUCE gets vectorized accumulator for K-vectorization
#[test]
fn test_expand_with_shift_to_upcast() {
    // Create reduction with output dimension (Loop) to enable vectorized accumulator
    // Output dimension is required for fix_reduce_unroll to set Vector dtype
    let loop_end = UOp::const_(DType::Index, ConstValue::Int(16));
    let loop_range = UOp::range_axis(loop_end, AxisId::Renumbered(2), AxisType::Loop);

    // Create the shift_to substitution expression for UPCAST
    // Original: Range(Reduce, 64), UPCAST by 4 → Range(Reduce, 16) * 4 + Range(Upcast, 4)
    let (_replaced_rng, _new_rng, sub_expr) = create_shift_to_expr(
        0,  // original axis_id
        64, // original size (K dimension)
        4,  // amount (upcast/vectorization factor)
        1,  // new axis_id for upcast range
        AxisType::Upcast,
    );

    // Create REDUCE with both the upcast expression and loop range
    let src = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    let reduce = src.reduce(smallvec![sub_expr, loop_range.clone()], ReduceOp::Add);

    // Run pre_expand (full pipeline)
    let result = pre_expand(&reduce);

    // Verify the result
    match result.op() {
        Op::Reduce { src: fixed_src, ranges, .. } => {
            // 1. Source should be wrapped in CONTRACT (for upcast axes)
            // CONTRACT(Const) → VECTORIZE after do_contract
            assert!(
                matches!(fixed_src.op(), Op::Contract { .. } | Op::Vectorize { .. }),
                "Source should be CONTRACT or VECTORIZE for upcast, got {:?}",
                fixed_src.op()
            );

            // 2. Loop range should be preserved (output dimension)
            assert!(
                ranges.iter().any(|r| matches!(r.op(), Op::Range { axis_type: AxisType::Loop, .. })),
                "Should preserve Loop range for output dimension"
            );

            // 3. Reduce range should have reduced size (64/4 = 16)
            let has_reduced_range = ranges.iter().any(|r| {
                if let Op::Range { axis_type: AxisType::Reduce, end, .. } = r.op() {
                    matches!(end.op(), Op::Const(cv) if cv.0 == ConstValue::Int(16))
                } else {
                    false
                }
            });
            assert!(has_reduced_range, "Should have reduced Range(Reduce, 16)");
        }
        other => panic!("Expected REDUCE, got {:?}", other),
    }
}

// =============================================================================
// Test 3: Nested shift_to (multiple UNROLL applications)
// =============================================================================

/// Test that nested shift_to expressions (from multiple UNROLL/UPCAST) work.
///
/// Simulates: REDUCE with UNROLL(4) then UPCAST(2)
/// Creates: ADD(ADD(MUL(reduce_rng, 4), Range(Unroll)), Range(Upcast))
#[test]
fn test_expand_with_nested_shift_to() {
    // Create nested expression: two shift_to applications
    // First shift_to: Range(Reduce, 16) → Range(Reduce, 4) * 4 + Range(Unroll, 4)
    let (_replaced1, _new1, inner_expr) =
        create_shift_to_expr(0, 16, 4, 1, AxisType::Unroll);

    // Second shift_to applied to the inner expression conceptually
    // For testing, we manually create the outer layer:
    // ADD(inner_expr, Range(Upcast, 2))
    let upcast_end = UOp::const_(DType::Index, ConstValue::Int(2));
    let upcast_range = UOp::range_axis(upcast_end, AxisId::Renumbered(2), AxisType::Upcast);
    let nested_expr = UOp::new(Op::Binary(BinaryOp::Add, inner_expr, upcast_range), DType::Index);

    // Add output dimension for proper handling
    let loop_end = UOp::const_(DType::Index, ConstValue::Int(8));
    let loop_range = UOp::range_axis(loop_end, AxisId::Renumbered(3), AxisType::Loop);

    let src = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    let reduce = src.reduce(smallvec![nested_expr, loop_range], ReduceOp::Add);

    // Run pre_expand
    let result = pre_expand(&reduce);

    // Verify
    match result.op() {
        Op::Reduce { src: fixed_src, ranges, .. } => {
            // Source should be expanded
            assert!(
                matches!(fixed_src.op(), Op::Contract { .. } | Op::Vectorize { .. }),
                "Source should be CONTRACT or VECTORIZE, got {:?}",
                fixed_src.op()
            );

            // Should extract the innermost Reduce range
            assert!(
                ranges.iter().any(|r| matches!(r.op(), Op::Range { axis_type: AxisType::Reduce, .. })),
                "Should extract Reduce range from nested expression"
            );

            // Loop range preserved
            assert!(ranges.iter().any(|r| matches!(r.op(), Op::Range { axis_type: AxisType::Loop, .. })));
        }
        other => panic!("Expected REDUCE, got {:?}", other),
    }
}

// =============================================================================
// Test 4: Full Scheduler integration (real shift_to call)
// =============================================================================

/// Test the complete path through Scheduler.shift_to → pre_expand.
///
/// This uses the actual Scheduler infrastructure instead of manual expression building.
#[test]
fn test_scheduler_shift_to_integration() {
    // Create a simple kernel AST with STORE(REDUCE)
    let reduce = create_simple_reduce(16, 0);

    // Create simple index for STORE
    let index = UOp::index_const(0);

    // Create STORE with the REDUCE as value
    let loop_end = UOp::const_(DType::Index, ConstValue::Int(1));
    let loop_range = UOp::range_axis(loop_end, AxisId::Renumbered(1), AxisType::Loop);
    let store = index.store_with_ranges(reduce.clone(), smallvec![loop_range.clone()]);

    // Wrap in SINK (required for Scheduler)
    let ast = UOp::sink(vec![store]);

    // Create Scheduler with CPU renderer (no device limits)
    let renderer = Renderer::cpu();
    let mut scheduler = Scheduler::new(ast.clone(), renderer);

    // Get the reduce range
    let reduce_range = scheduler.rngs().iter().find(|r| {
        matches!(r.op(), Op::Range { axis_type: AxisType::Reduce, .. })
    }).cloned();

    if let Some(rng) = reduce_range {
        // Apply shift_to with UNROLL
        let result = scheduler.shift_to(rng, 4, AxisType::Unroll, false, None);
        assert!(result.is_ok(), "shift_to should succeed");

        // Get the optimized AST and run pre_expand
        let optimized = scheduler.get_optimized_ast(None);
        let expanded = pre_expand(&optimized);

        // Verify the expanded AST has proper structure
        // The REDUCE inside should have CONTRACT-wrapped source
        let mut found_reduce = false;
        for node in expanded.toposort() {
            if let Op::Reduce { src, ranges, .. } = node.op() {
                found_reduce = true;
                // Source should be expanded (CONTRACT or VECTORIZE)
                assert!(
                    matches!(src.op(), Op::Contract { .. } | Op::Vectorize { .. } | Op::Const(_)),
                    "REDUCE.src should be CONTRACT/VECTORIZE/Const after expansion, got {:?}",
                    src.op()
                );
                // Ranges should not contain arithmetic expressions
                for range in ranges.iter() {
                    assert!(
                        !matches!(range.op(), Op::Binary(..)),
                        "REDUCE.ranges should not contain Binary after expansion"
                    );
                }
            }
        }
        assert!(found_reduce, "Should find REDUCE in expanded AST");
    }
}

// =============================================================================
// Test 5: extract_ranges_from_expr pattern matching
// =============================================================================

/// Test that extract_ranges_from_expr correctly identifies all patterns.
#[test]
fn test_extract_ranges_patterns() {
    use crate::expand::fix_reduce_unroll;

    // Pattern 1: top=false: ADD(MUL(reduce_rng, Const), new_rng)
    let reduce_end = UOp::const_(DType::Index, ConstValue::Int(4));
    let reduce_rng = UOp::range_axis(reduce_end, AxisId::Renumbered(0), AxisType::Reduce);
    let amount = UOp::const_(DType::Index, ConstValue::Int(4));
    let mul = UOp::new(Op::Binary(BinaryOp::Mul, reduce_rng.clone(), amount), DType::Index);

    let unroll_end = UOp::const_(DType::Index, ConstValue::Int(4));
    let unroll_rng = UOp::range_axis(unroll_end, AxisId::Renumbered(1), AxisType::Unroll);
    let add = UOp::new(Op::Binary(BinaryOp::Add, mul, unroll_rng.clone()), DType::Index);

    let src = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    let reduce = src.reduce(smallvec![add], ReduceOp::Add);

    let result = fix_reduce_unroll(&reduce);
    assert!(result.is_some(), "Pattern 1 (top=false) should be handled");

    // Pattern 2: top=true: ADD(MUL(new_rng, Const), reduce_rng)
    let mul_top = UOp::new(
        Op::Binary(BinaryOp::Mul, unroll_rng, UOp::const_(DType::Index, ConstValue::Int(4))),
        DType::Index,
    );
    let add_top = UOp::new(Op::Binary(BinaryOp::Add, mul_top, reduce_rng), DType::Index);

    let reduce_top = src.reduce(smallvec![add_top], ReduceOp::Add);
    let result_top = fix_reduce_unroll(&reduce_top);
    assert!(result_top.is_some(), "Pattern 2 (top=true) should be handled");
}
