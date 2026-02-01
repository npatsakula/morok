//! Tests for end_unrolls (END with UNROLL handling).
//!
//! end_unrolls transforms END operations that have UNROLL in their ranges:
//!
//! END(computation, [Range, UNROLL, Range]) →
//! END(CONTRACT(computation, unroll_axes), [Range, Range])
//!
//! The CONTRACT wraps the computation, then do_contract processes it:
//! - For void dtype (vcount=1): CONTRACT(src) → src directly
//! - For non-UNROLL src: CONTRACT(src) → VECTORIZE(src, src, ...)
//! - For UNROLL src: CONTRACT(UNROLL) → GEP extraction
//!
//! Based on Tinygrad's expander.py:78-82.

use morok_dtype::DType;
use morok_ir::{AxisId, AxisType, Op, UOp};

use super::helpers::*;

// =============================================================================
// Basic Functionality Tests
// =============================================================================

/// Test: END(comp, [UNROLL(...)]) → UNROLL removed from ranges.
///
/// For void computation, CONTRACT(void, src) is simplified to src
/// because void has vcount=1 (Morok optimizes this directly).
#[test]
fn test_end_single_unroll() {
    let computation = UOp::noop();
    let unroll = create_unroll_iota(0, 4);
    let end = create_end(computation, vec![unroll]);

    let result = phase2_only(&end);

    // Key check: UNROLL should be removed from ranges
    match result.op() {
        Op::End { computation: c, ranges } => {
            // UNROLL must be removed from ranges
            for r in ranges.iter() {
                assert!(!matches!(r.op(), Op::Unroll { .. }), "UNROLL should be removed from ranges");
            }
            // For void dtype, CONTRACT is simplified to src directly
            // (Morok: vcount=1 shortcut, Tinygrad: VECTORIZE→unwrap)
            // So computation may be NOOP, CONTRACT, or the result of contraction
            assert!(
                matches!(c.op(), Op::Noop | Op::Contract { .. } | Op::Vectorize { .. }),
                "Computation should be processed, got {:?}",
                c.op()
            );
        }
        other => panic!("Expected END, got {:?}", other),
    }
}

/// Test: END with multiple UNROLLs → all UNROLL axes combined and removed.
#[test]
fn test_end_multiple_unrolls() {
    let computation = UOp::noop();
    let unroll1 = create_unroll_iota(0, 2);
    let unroll2 = create_unroll_iota(1, 3);

    let end = create_end(computation, vec![unroll1, unroll2]);
    let result = phase2_only(&end);

    match result.op() {
        Op::End { ranges, .. } => {
            // All UNROLLs must be removed from ranges
            for r in ranges.iter() {
                assert!(!matches!(r.op(), Op::Unroll { .. }), "UNROLL should be removed");
            }
        }
        other => panic!("Expected END, got {:?}", other),
    }
}

/// Test: END with mixed [Range, UNROLL, Range] → only non-UNROLL ranges remain.
#[test]
fn test_end_mixed_ranges() {
    let computation = UOp::noop();

    // Mixed: Range, UNROLL, Range
    let range1 = UOp::range_axis(
        UOp::const_(DType::Index, morok_ir::types::ConstValue::Int(8)),
        AxisId::Renumbered(0),
        AxisType::Reduce,
    );
    let unroll = create_unroll_iota(1, 4);
    let range2 = UOp::range_axis(
        UOp::const_(DType::Index, morok_ir::types::ConstValue::Int(16)),
        AxisId::Renumbered(2),
        AxisType::Loop,
    );

    let end = create_end(computation, vec![range1.clone(), unroll, range2.clone()]);
    let result = phase2_only(&end);

    match result.op() {
        Op::End { ranges, .. } => {
            // Only Range ops should remain (UNROLL removed)
            assert_eq!(ranges.len(), 2, "Should have 2 non-UNROLL ranges");
            for r in ranges.iter() {
                assert!(matches!(r.op(), Op::Range { .. }), "Only Range ops should remain");
            }
        }
        other => panic!("Expected END, got {:?}", other),
    }
}

// =============================================================================
// Passthrough Tests (pattern should NOT fire)
// =============================================================================

/// Test: END with no UNROLL passes through unchanged.
#[test]
fn test_end_no_unroll_passthrough() {
    let computation = UOp::noop();
    let range = UOp::range_axis(
        UOp::const_(DType::Index, morok_ir::types::ConstValue::Int(16)),
        AxisId::Renumbered(0),
        AxisType::Reduce,
    );

    let end = create_end(computation.clone(), vec![range]);
    let result = phase2_only(&end);

    match result.op() {
        Op::End { computation: c, ranges } => {
            // No CONTRACT needed without UNROLL
            assert!(matches!(c.op(), Op::Noop), "Computation should stay NOOP");
            assert_eq!(ranges.len(), 1, "Range should be preserved");
        }
        other => panic!("Expected END, got {:?}", other),
    }
}

/// Test: END with empty ranges passes through unchanged.
#[test]
fn test_end_empty_ranges_passthrough() {
    let computation = UOp::noop();
    let end = create_end(computation.clone(), vec![]);
    let result = phase2_only(&end);

    match result.op() {
        Op::End { computation: c, ranges } => {
            assert!(matches!(c.op(), Op::Noop), "Computation should stay NOOP");
            assert!(ranges.is_empty(), "Ranges should stay empty");
        }
        other => panic!("Expected END, got {:?}", other),
    }
}

// =============================================================================
// Axis Handling Tests
// =============================================================================

/// Test: UNROLL with multi-axis [(0,2), (1,3)] → all axes contracted.
#[test]
fn test_end_unroll_multi_axis() {
    let computation = UOp::noop();
    let unroll = create_unroll_multi_axis(vec![(0, 2), (1, 3)]);

    let end = create_end(computation, vec![unroll]);
    let result = phase2_only(&end);

    match result.op() {
        Op::End { ranges, .. } => {
            // UNROLL should be removed
            assert!(ranges.is_empty() || ranges.iter().all(|r| !matches!(r.op(), Op::Unroll { .. })));
        }
        other => panic!("Expected END, got {:?}", other),
    }
}

/// Test: Void dtype computation is handled correctly.
///
/// For void dtype, CONTRACT(void, src) simplifies to src.
#[test]
fn test_end_void_dtype() {
    let computation = UOp::noop();
    let unroll = create_unroll_iota(0, 4);

    let end = create_end(computation, vec![unroll]);
    let result = phase2_only(&end);

    assert_is_end(&result);
    // For void, CONTRACT simplifies away
    if let Op::End { computation: c, ranges } = result.op() {
        // UNROLL must be removed
        for r in ranges.iter() {
            assert!(!matches!(r.op(), Op::Unroll { .. }));
        }
        // Computation may be NOOP (CONTRACT simplified) or CONTRACT
        assert!(matches!(c.op(), Op::Noop | Op::Contract { .. } | Op::Vectorize { .. }));
    }
}

/// Test: Non-UNROLL range order is preserved.
#[test]
fn test_end_preserves_non_unroll_order() {
    let computation = UOp::noop();

    let range_a = UOp::range_axis(
        UOp::const_(DType::Index, morok_ir::types::ConstValue::Int(8)),
        AxisId::Renumbered(0),
        AxisType::Reduce,
    );
    let unroll = create_unroll_iota(1, 4);
    let range_b = UOp::range_axis(
        UOp::const_(DType::Index, morok_ir::types::ConstValue::Int(16)),
        AxisId::Renumbered(2),
        AxisType::Loop,
    );

    let end = create_end(computation, vec![range_a, unroll, range_b]);
    let result = phase2_only(&end);

    match result.op() {
        Op::End { ranges, .. } => {
            assert_eq!(ranges.len(), 2, "Should have 2 non-UNROLL ranges");
            // Verify order: first should be axis 0 (Reduce), second axis 2 (Loop)
            if let Op::Range { axis_id: AxisId::Renumbered(id0), axis_type: t0, .. } = ranges[0].op() {
                assert_eq!(*id0, 0);
                assert_eq!(*t0, AxisType::Reduce);
            }
            if let Op::Range { axis_id: AxisId::Renumbered(id1), axis_type: t1, .. } = ranges[1].op() {
                assert_eq!(*id1, 2);
                assert_eq!(*t1, AxisType::Loop);
            }
        }
        other => panic!("Expected END, got {:?}", other),
    }
}

// =============================================================================
// Integration Tests
// =============================================================================

/// Test: END with UNROLL through full expander.
#[test]
fn test_end_unroll_full_expander() {
    let computation = UOp::noop();
    let unroll = create_unroll_iota(0, 4);

    let end = create_end(computation, vec![unroll]);
    let result = expander_rewrite(&end);

    // After full expansion, UNROLL should be processed
    match result.op() {
        Op::End { ranges, .. } => {
            for r in ranges.iter() {
                assert!(!matches!(r.op(), Op::Unroll { .. }), "UNROLL should be processed");
            }
        }
        _ => {
            assert!(count_unrolls(&result) == 0, "UNROLLs should be expanded");
        }
    }
}

/// Test: Nested END operations with UNROLL.
#[test]
fn test_end_nested() {
    let inner_comp = UOp::noop();
    let inner_unroll = create_unroll_iota(0, 2);
    let inner_end = create_end(inner_comp, vec![inner_unroll]);

    let outer_unroll = create_unroll_iota(1, 3);
    let outer_end = create_end(inner_end, vec![outer_unroll]);

    let result = phase2_only(&outer_end);

    // Both UNROLLs should be handled
    let unroll_count = count_unrolls(&result);
    assert!(unroll_count == 0, "All UNROLLs should be processed, found {}", unroll_count);
}
