//! Tests for fix_reduce_unroll (REDUCE with UNROLL handling).
//!
//! fix_reduce_unroll handles REDUCE operations with UNROLL ops in their ranges.
//! After Phase 1 converts Range(Unroll/Upcast) → UNROLL ops, fix_reduce_unroll:
//! - Partitions ranges into RANGE ops vs UNROLL ops
//! - Moves UNROLL ops to CONTRACT wrapper on source
//! - Returns REDUCE with only RANGE ops in ranges
//!
//! Value assertions verify CONTRACT wrapping behavior.

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

/// Test: REDUCE with only Loop ranges passes through unchanged.
#[test]
fn test_fix_reduce_loop_passthrough() {
    // Create REDUCE with only Loop-type Range (output dimension)
    let end = UOp::const_(DType::Index, ConstValue::Int(16));
    let range = UOp::range_axis(end, AxisId::Renumbered(0), AxisType::Loop);
    let src = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let reduce = src.reduce(smallvec![range.clone()], ReduceOp::Add);

    // Apply expander
    let result = expander_rewrite(&reduce);

    // Should pass through (no UNROLL in ranges)
    match result.op() {
        Op::Reduce { src: fixed_src, ranges, .. } => {
            assert_eq!(ranges.len(), 1, "Should still have one range");
            assert!(matches!(ranges[0].op(), Op::Range { axis_type: AxisType::Loop, .. }));
            // Source should NOT be wrapped in CONTRACT
            assert!(!matches!(fixed_src.op(), Op::Contract { .. }), "Should not have CONTRACT wrapper");
        }
        other => panic!("Expected REDUCE, got {:?}", other),
    }
}

// =============================================================================
// Range(Unroll) Tests
// =============================================================================

/// Test: REDUCE with Range(Unroll) → CONTRACT wrapper on source.
///
/// This is the core fix_reduce_unroll behavior:
/// REDUCE(src, [Range(Reduce), Range(Unroll)]) →
/// REDUCE(CONTRACT(src, unroll_axes), [Range(Reduce)])
#[test]
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

/// Test: REDUCE with Range(Unroll) → source vectorized to match unroll count.
#[test]
fn test_fix_reduce_unroll_vectorizes_source() {
    // Create REDUCE with Range(Unroll) of size 4
    let unroll_end = UOp::const_(DType::Index, ConstValue::Int(4));
    let unroll_range = UOp::range_axis(unroll_end, AxisId::Renumbered(1), AxisType::Unroll);
    let src = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let reduce = src.reduce(smallvec![unroll_range], ReduceOp::Add);

    // Apply expander
    let result = expander_rewrite(&reduce);

    // After fix_reduce_unroll:
    // - UNROLL moved to CONTRACT wrapper
    // - CONTRACT(Const) → VECTORIZE of 4 copies
    match result.op() {
        Op::Reduce { src: fixed_src, ranges, .. } => {
            // No UNROLL in ranges
            assert!(ranges.is_empty() || !ranges.iter().any(|r| matches!(r.op(), Op::Unroll { .. })));
            // Source should be VECTORIZE with 4 elements (broadcast)
            if let Op::Vectorize { elements } = fixed_src.op() {
                assert_eq!(elements.len(), 4, "Should broadcast to 4 elements");
                // All elements should be the same constant
                for elem in elements.iter() {
                    if let Op::Const(cv) = elem.op() {
                        assert_eq!(cv.0, ConstValue::Float(1.0), "Should preserve value");
                    } else {
                        panic!("VECTORIZE elements should be constants");
                    }
                }
            }
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

// =============================================================================
// Multiple UNROLL Tests
// =============================================================================

/// Test: REDUCE with multiple Range(Unroll) → combined CONTRACT axes.
#[test]
fn test_fix_reduce_multiple_unrolls() {
    // Create REDUCE with two Range(Unroll)
    let unroll1_end = UOp::const_(DType::Index, ConstValue::Int(2));
    let unroll1_range = UOp::range_axis(unroll1_end, AxisId::Renumbered(1), AxisType::Unroll);
    let unroll2_end = UOp::const_(DType::Index, ConstValue::Int(2));
    let unroll2_range = UOp::range_axis(unroll2_end, AxisId::Renumbered(2), AxisType::Unroll);
    let src = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let reduce = src.reduce(smallvec![unroll1_range, unroll2_range], ReduceOp::Add);

    // Apply expander
    let result = expander_rewrite(&reduce);

    // Both UNROLL ranges should be combined in CONTRACT
    // Total vectorization: 2 * 2 = 4
    match result.op() {
        Op::Reduce { src: fixed_src, ranges, .. } => {
            // No UNROLL in ranges
            assert!(ranges.is_empty() || !ranges.iter().any(|r| matches!(r.op(), Op::Unroll { .. })));
            // Source should be vectorized to 4 elements
            assert_eq!(
                fixed_src.dtype().vcount(),
                4,
                "Combined UNROLL should vectorize to 4 elements"
            );
        }
        other => panic!("Expected REDUCE, got {:?}", other),
    }
}

// =============================================================================
// Mixed Ranges Tests
// =============================================================================

/// Test: REDUCE with mixed Range(Reduce), Range(Unroll), Range(Loop).
#[test]
fn test_fix_reduce_mixed_ranges() {
    // Create REDUCE with all three range types
    let reduce_end = UOp::const_(DType::Index, ConstValue::Int(32));
    let reduce_range = UOp::range_axis(reduce_end, AxisId::Renumbered(0), AxisType::Reduce);

    let unroll_end = UOp::const_(DType::Index, ConstValue::Int(4));
    let unroll_range = UOp::range_axis(unroll_end, AxisId::Renumbered(1), AxisType::Unroll);

    let loop_end = UOp::const_(DType::Index, ConstValue::Int(8));
    let loop_range = UOp::range_axis(loop_end, AxisId::Renumbered(2), AxisType::Loop);

    let src = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    let reduce = src.reduce(smallvec![reduce_range, unroll_range, loop_range], ReduceOp::Add);

    // Apply expander
    let result = expander_rewrite(&reduce);

    // After fix_reduce_unroll:
    // - UNROLL moved to CONTRACT wrapper
    // - Reduce and Loop ranges preserved in REDUCE.ranges
    match result.op() {
        Op::Reduce { src: fixed_src, ranges, .. } => {
            // Should have Reduce and Loop ranges (UNROLL removed)
            let has_reduce = ranges.iter().any(|r| matches!(r.op(), Op::Range { axis_type: AxisType::Reduce, .. }));
            let has_loop = ranges.iter().any(|r| matches!(r.op(), Op::Range { axis_type: AxisType::Loop, .. }));
            let has_unroll = ranges.iter().any(|r| matches!(r.op(), Op::Unroll { .. }));

            assert!(has_reduce, "Should preserve Reduce range");
            assert!(has_loop, "Should preserve Loop range");
            assert!(!has_unroll, "UNROLL should be removed from ranges");

            // Source should be expanded
            assert!(
                matches!(fixed_src.op(), Op::Contract { .. } | Op::Vectorize { .. }),
                "Source should be expanded"
            );
        }
        other => panic!("Expected REDUCE, got {:?}", other),
    }
}

// =============================================================================
// UNROLL Source Tests (non-constant sources)
// =============================================================================

/// Test: REDUCE with UNROLL-wrapped source and Range(Unroll).
///
/// When the source is already UNROLL-wrapped and we have Range(Unroll),
/// the CONTRACT wrapping should combine properly.
#[test]
fn test_fix_reduce_unroll_source_with_unroll_range() {
    // Create UNROLL-wrapped source: UNROLL(VCONST([1,2,3,4]), [(1,4)])
    let src = create_unroll_values(1, vec![1, 2, 3, 4]);
    // Cast to Float32 for reduce (since UNROLL is Int64)
    let src_float = src.cast(DType::Float32);

    // Create Range(Unroll) with same axis
    let unroll_end = UOp::const_(DType::Index, ConstValue::Int(4));
    let unroll_range = UOp::range_axis(unroll_end, AxisId::Renumbered(1), AxisType::Unroll);

    let reduce = src_float.reduce(smallvec![unroll_range], ReduceOp::Add);

    // Apply expander
    let result = expander_rewrite(&reduce);

    // The UNROLL source combined with CONTRACT should produce
    // a properly vectorized REDUCE
    match result.op() {
        Op::Reduce { ranges, .. } => {
            // UNROLL should be removed from ranges
            assert!(!ranges.iter().any(|r| matches!(r.op(), Op::Unroll { .. })));
        }
        other => panic!("Expected REDUCE, got {:?}", other),
    }
}

// =============================================================================
// Edge Cases
// =============================================================================

/// Test: REDUCE with single Range(Unroll) only.
#[test]
fn test_fix_reduce_single_unroll_only() {
    // Create REDUCE with only Range(Unroll) - no Reduce range
    let unroll_end = UOp::const_(DType::Index, ConstValue::Int(4));
    let unroll_range = UOp::range_axis(unroll_end, AxisId::Renumbered(0), AxisType::Unroll);
    let src = UOp::const_(DType::Float32, ConstValue::Float(2.0));
    let reduce = src.reduce(smallvec![unroll_range], ReduceOp::Add);

    // Apply expander
    let result = expander_rewrite(&reduce);

    // After expansion: REDUCE with empty ranges (or only output ranges)
    // Source should be vectorized
    match result.op() {
        Op::Reduce { src: fixed_src, ranges, .. } => {
            // UNROLL removed from ranges
            assert!(!ranges.iter().any(|r| matches!(r.op(), Op::Unroll { .. })));
            // Source vectorized
            assert_eq!(fixed_src.dtype().vcount(), 4, "Source should be vec4");
        }
        other => panic!("Expected REDUCE, got {:?}", other),
    }
}

/// Test: REDUCE with Range(Unroll) size 1 should effectively be passthrough.
#[test]
fn test_fix_reduce_unroll_size_1() {
    // Create REDUCE with Range(Unroll) of size 1
    let unroll_end = UOp::const_(DType::Index, ConstValue::Int(1));
    let unroll_range = UOp::range_axis(unroll_end, AxisId::Renumbered(0), AxisType::Unroll);
    let src = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let reduce = src.reduce(smallvec![unroll_range], ReduceOp::Add);

    // Apply expander
    let result = expander_rewrite(&reduce);

    // Size-1 UNROLL should still work (trivial case)
    assert!(matches!(result.op(), Op::Reduce { .. }));
}
