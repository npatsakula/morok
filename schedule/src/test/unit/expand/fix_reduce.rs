//! Tests for fix_reduce_unroll (REDUCE with UNROLL/Upcast handling).
//!
//! fix_reduce_unroll handles REDUCE operations with non-Range entries in their ranges:
//! - Range(Unroll/Upcast) → move to CONTRACT wrapper
//! - Arithmetic expressions from shift_to → extract the Reduce range
//! - Sets Vector dtype for Upcast axes when output dimensions remain

use super::helpers::*;
use morok_dtype::DType;
use morok_ir::types::ConstValue;
use morok_ir::{AxisId, AxisType, BinaryOp, Op, ReduceOp, UOp};
use smallvec::smallvec;

// =============================================================================
// Passthrough Tests
// =============================================================================

/// Test: REDUCE with only Reduce ranges passes through unchanged.
#[test]
fn test_fix_reduce_simple_passthrough() {
    // Create REDUCE with only Reduce-type Range
    let end = UOp::const_(DType::Index, ConstValue::Int(32));
    let range = UOp::range_axis(end, AxisId::Renumbered(0), AxisType::Reduce);
    let src = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    let reduce = UOp::reduce(src, smallvec![range.clone()], ReduceOp::Add);

    // Apply expander
    let result = expander_rewrite(&reduce);

    // Should pass through (ranges unchanged, no CONTRACT wrapper)
    match result.op() {
        Op::Reduce { src: fixed_src, ranges, .. } => {
            assert_eq!(ranges.len(), 1, "Should still have one range");
            assert!(matches!(ranges[0].op(), Op::Range { axis_type: AxisType::Reduce, .. }));
            // Source should NOT be wrapped in CONTRACT
            assert!(!matches!(fixed_src.op(), Op::Contract { .. }), "Should not have CONTRACT wrapper");
        }
        other => panic!("Expected REDUCE, got {:?}", other),
    }
}

// =============================================================================
// Range(Unroll) Tests
// =============================================================================

/// Test: REDUCE with Range(Unroll) → CONTRACT wrapper.
#[test]
#[tracing_test::traced_test]
fn test_fix_reduce_range_unroll() {
    // Create REDUCE with Range(Unroll)
    let end = UOp::const_(DType::Index, ConstValue::Int(4));
    let unroll_range = UOp::range_axis(end.clone(), AxisId::Renumbered(1), AxisType::Unroll);
    let reduce_range =
        UOp::range_axis(UOp::const_(DType::Index, ConstValue::Int(32)), AxisId::Renumbered(0), AxisType::Reduce);
    let src = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    let reduce = UOp::reduce(src, smallvec![reduce_range, unroll_range], ReduceOp::Add);

    // Apply expander
    let result = expander_rewrite(&reduce);

    // Range(Unroll) should be converted and handled:
    // 1. Phase1: Range(Unroll) → UNROLL
    // 2. fix_reduce_unroll: UNROLL moved from ranges to CONTRACT wrapper on source
    // 3. do_contract: CONTRACT(Const) → VECTORIZE
    // Final: REDUCE(VECTORIZE, [Range(Reduce)])
    match result.op() {
        Op::Reduce { src: fixed_src, ranges, .. } => {
            // Reduce ranges should only contain the Reduce-type range
            // UNROLL should be removed from ranges
            for range in ranges.iter() {
                assert!(!matches!(range.op(), Op::Unroll { .. }), "UNROLL should be removed from ranges");
            }
            // Source should be expanded (CONTRACT → VECTORIZE for Const source)
            assert!(
                matches!(fixed_src.op(), Op::Contract { .. } | Op::Vectorize { .. }),
                "Source should be CONTRACT or VECTORIZE (expanded), got {:?}",
                fixed_src.op()
            );
        }
        other => panic!("Expected REDUCE, got {:?}", other),
    }
}

// =============================================================================
// Range(Upcast) Tests
// =============================================================================

/// Test: REDUCE with Range(Upcast) and output dimensions → Vector dtype.
#[test]
fn test_fix_reduce_range_upcast() {
    // Create REDUCE with Range(Upcast) AND a Loop range (output dimension)
    let upcast_end = UOp::const_(DType::Index, ConstValue::Int(4));
    let upcast_range = UOp::range_axis(upcast_end, AxisId::Renumbered(1), AxisType::Upcast);
    let loop_end = UOp::const_(DType::Index, ConstValue::Int(16));
    let loop_range = UOp::range_axis(loop_end, AxisId::Renumbered(2), AxisType::Loop);
    let src = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    let reduce = UOp::reduce(src, smallvec![upcast_range, loop_range], ReduceOp::Add);

    // Apply expander
    let result = expander_rewrite(&reduce);

    // With Upcast and output dimensions:
    // 1. Range(Upcast) → UNROLL
    // 2. fix_reduce_unroll: UNROLL moved to CONTRACT wrapper on source
    // 3. do_contract: CONTRACT(Const) → VECTORIZE
    match result.op() {
        Op::Reduce { src: fixed_src, ranges, .. } => {
            // Source should be expanded (CONTRACT → VECTORIZE for Const source)
            assert!(
                matches!(fixed_src.op(), Op::Contract { .. } | Op::Vectorize { .. }),
                "Source should be CONTRACT or VECTORIZE (expanded), got {:?}",
                fixed_src.op()
            );
            // Loop range should be preserved
            assert!(ranges.iter().any(|r| matches!(r.op(), Op::Range { axis_type: AxisType::Loop, .. })));
        }
        other => panic!("Expected REDUCE, got {:?}", other),
    }
}

// =============================================================================
// Arithmetic Expression Tests
// =============================================================================

/// Test: REDUCE with nested Binary expression from shift_to.
///
/// Pattern: ADD(ADD(MUL(reduce_range, 4), range_upcast1), range_upcast2)
#[test]
fn test_fix_reduce_arithmetic_expr() {
    // Create nested shift_to expression
    let end = UOp::const_(DType::Index, ConstValue::Int(64));
    let reduce_range = UOp::range_axis(end.clone(), AxisId::Renumbered(0), AxisType::Reduce);

    // Inner shift_to: MUL(reduce_range, 4) + Range(Upcast)
    let upcast_end = UOp::const_(DType::Index, ConstValue::Int(4));
    let upcast_range = UOp::range_axis(upcast_end.clone(), AxisId::Renumbered(1), AxisType::Upcast);
    let mul = UOp::new(
        Op::Binary(BinaryOp::Mul, reduce_range.clone(), UOp::const_(DType::Index, ConstValue::Int(4))),
        DType::Index,
    );
    let inner_add = UOp::new(Op::Binary(BinaryOp::Add, mul, upcast_range), DType::Index);

    // Outer shift_to: inner_add + Range(Upcast)
    let upcast_range2 = UOp::range_axis(upcast_end, AxisId::Renumbered(2), AxisType::Upcast);
    let nested_binary = UOp::new(Op::Binary(BinaryOp::Add, inner_add, upcast_range2), DType::Index);

    let src = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    let reduce = UOp::reduce(src, smallvec![nested_binary], ReduceOp::Add);

    // Apply expander
    let result = expander_rewrite(&reduce);

    // Should extract the reduce_range and handle upcast axes
    // Note: fix_reduce_unroll extracts ranges from arithmetic expressions
    // CONTRACT(Const) → VECTORIZE after do_contract
    match result.op() {
        Op::Reduce { src: fixed_src, ranges, .. } => {
            // Source should be expanded (CONTRACT → VECTORIZE for Const source)
            assert!(
                matches!(fixed_src.op(), Op::Contract { .. } | Op::Vectorize { .. }),
                "Source should be CONTRACT or VECTORIZE (expanded), got {:?}",
                fixed_src.op()
            );
            // Ranges should contain the extracted reduce_range
            assert!(
                ranges.iter().any(|r| matches!(r.op(), Op::Range { axis_type: AxisType::Reduce, .. })),
                "Should extract Reduce range from expression"
            );
        }
        other => panic!("Expected REDUCE, got {:?}", other),
    }
}
