//! Tests for fix_reduce_unroll (REDUCE with UNROLL handling).
//!
//! fix_reduce_unroll handles REDUCE operations with UNROLL ops in their ranges.
//! After Phase 1 converts Range(Unroll/Upcast) → UNROLL ops, fix_reduce_unroll:
//! - Partitions ranges into RANGE ops vs UNROLL ops
//! - Moves UNROLL ops to CONTRACT wrapper on source
//! - Returns REDUCE with only RANGE ops in ranges

use super::helpers::*;
use morok_dtype::DType;
use morok_ir::types::ConstValue;
use morok_ir::{AxisId, AxisType, Op, ReduceOp, UOp};
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
    let reduce = src.reduce(smallvec![range.clone()], ReduceOp::Add);

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
    let reduce = src.reduce(smallvec![reduce_range, unroll_range], ReduceOp::Add);

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
    let reduce = src.reduce(smallvec![upcast_range, loop_range], ReduceOp::Add);

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

