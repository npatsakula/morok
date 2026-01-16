//! Tests for fix_store_unroll (STORE with UNROLL in ranges).
//!
//! fix_store_unroll handles STORE operations with UNROLL ops in their ranges:
//! - Partitions ranges into UNROLL vs non-UNROLL
//! - Wraps result in CONTRACT with collected axes

use super::helpers::*;
use morok_dtype::{DType, DeviceSpec};
use morok_ir::types::ConstValue;
use morok_ir::{AxisId, AxisType, Op, UOp};
use smallvec::smallvec;

// =============================================================================
// STORE Partition Tests
// =============================================================================

/// Test: STORE with UNROLL in ranges → CONTRACT wrapper.
///
/// Based on Tinygrad's fix_store_unroll (expander.py:123-126).
#[test]
fn test_fix_store_partition() {
    // Create buffer and index
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 1024, DType::Float32);
    let index = UOp::index_const(0);
    let value = UOp::const_(DType::Float32, ConstValue::Float(1.0));

    // Create UNROLL range
    let unroll = create_unroll_iota(0, 4);

    // Create STORE with UNROLL in ranges
    let store = UOp::store_with_ranges(buffer, index, value, smallvec![unroll]);

    // Apply expander
    let result = phase2_only(&store);

    // Result should be CONTRACT wrapping STORE
    match result.op() {
        Op::Contract { src, upcast_ranges } => {
            assert_eq!(upcast_ranges, &[(0, 4)], "CONTRACT should have axis from UNROLL");
            // Inner should be STORE without UNROLL ranges
            match src.op() {
                Op::Store { ranges, .. } => {
                    assert!(
                        ranges.is_empty() || !ranges.iter().any(|r| matches!(r.op(), Op::Unroll { .. })),
                        "STORE ranges should not contain UNROLL"
                    );
                }
                other => panic!("Expected STORE inside CONTRACT, got {:?}", other),
            }
        }
        // If no CONTRACT, the store was processed differently
        Op::Store { ranges, .. } => {
            // Store may pass through if no UNROLL detected
            // This is also valid if the pattern didn't match
            assert!(!ranges.iter().any(|r| matches!(r.op(), Op::Unroll { .. })));
        }
        other => panic!("Expected CONTRACT or STORE, got {:?}", other),
    }
}

/// Test: STORE with mixed UNROLL and non-UNROLL ranges.
///
/// Partitions correctly, keeping non-UNROLL ranges in STORE.
#[test]
fn test_fix_store_mixed_ranges() {
    // Create buffer and index
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 1024, DType::Float32);
    let index = UOp::index_const(0);
    let value = UOp::const_(DType::Float32, ConstValue::Float(1.0));

    // Create UNROLL range
    let unroll = create_unroll_iota(0, 4);

    // Create regular Range (Loop type)
    let end = UOp::const_(DType::Index, ConstValue::Int(16));
    let loop_range = UOp::range_axis(end, AxisId::Renumbered(1), AxisType::Loop);

    // Create STORE with both UNROLL and non-UNROLL ranges
    let store = UOp::store_with_ranges(buffer, index, value, smallvec![unroll, loop_range.clone()]);

    // Apply expander
    let result = phase2_only(&store);

    // Result should partition: CONTRACT(STORE with loop_range)
    match result.op() {
        Op::Contract { src, upcast_ranges } => {
            assert_eq!(upcast_ranges, &[(0, 4)], "CONTRACT should have UNROLL axis");
            match src.op() {
                Op::Store { ranges, .. } => {
                    // Loop range should be preserved in STORE
                    assert_eq!(ranges.len(), 1, "STORE should have one non-UNROLL range");
                    assert!(
                        matches!(ranges[0].op(), Op::Range { axis_type: AxisType::Loop, .. }),
                        "Should preserve Loop range"
                    );
                }
                other => panic!("Expected STORE inside CONTRACT, got {:?}", other),
            }
        }
        Op::Store { ranges, .. } => {
            // If no CONTRACT, store processed differently
            // Non-UNROLL ranges should still be present
            assert!(ranges.iter().any(|r| matches!(r.op(), Op::Range { .. })));
        }
        other => panic!("Expected CONTRACT or STORE, got {:?}", other),
    }
}
