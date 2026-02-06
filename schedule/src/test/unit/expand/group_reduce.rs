//! Tests for pm_group_for_reduce (GROUP_REDUCE → shared memory pattern).
//!
//! GROUP_REDUCE enables two-stage reduction for tensor core optimizations:
//! 1. First reduce within each group (partial reduce with non-GROUP_REDUCE ranges)
//! 2. Bufferize to LOCAL memory indexed by LOCAL + GROUP_REDUCE ranges
//! 3. INDEX from shared memory with renumbered ranges (axis_id + 100)
//! 4. Final REDUCE across the new ranges
//!
//! Based on Tinygrad's fix_group_for_reduce (expander.py:128-141).

use super::helpers::*;
use morok_dtype::DType;
use morok_ir::types::ConstValue;
use morok_ir::{AddrSpace, AxisId, AxisType, Op, ReduceOp, UOp};
use smallvec::smallvec;
use std::sync::Arc;

/// Create a GROUP_REDUCE range for testing.
fn create_group_reduce_range(axis_id: usize, size: i64) -> Arc<UOp> {
    let end = UOp::const_(DType::Index, ConstValue::Int(size));
    UOp::range_axis(end, AxisId::Renumbered(axis_id), AxisType::GroupReduce)
}

/// Create a LOCAL range for testing.
fn create_local_range(axis_id: usize, size: i64) -> Arc<UOp> {
    let end = UOp::const_(DType::Index, ConstValue::Int(size));
    UOp::range_axis(end, AxisId::Renumbered(axis_id), AxisType::Local)
}

/// Create a Reduce range for testing.
fn create_reduce_range(axis_id: usize, size: i64) -> Arc<UOp> {
    let end = UOp::const_(DType::Index, ConstValue::Int(size));
    UOp::range_axis(end, AxisId::Renumbered(axis_id), AxisType::Reduce)
}

// =============================================================================
// Passthrough Tests
// =============================================================================

/// Test: REDUCE with only regular Reduce ranges should pass through unchanged.
#[test]
fn test_passthrough_no_group_reduce() {
    let reduce_range = create_reduce_range(0, 32);
    let src = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    let reduce = src.reduce(smallvec![reduce_range], ReduceOp::Add);

    let result = expander_rewrite(&reduce);

    // Should pass through - no GROUP_REDUCE means no transformation
    match result.op() {
        Op::Reduce { ranges, .. } => {
            assert_eq!(ranges.len(), 1, "Should have single range");
            assert!(
                matches!(ranges[0].op(), Op::Range { axis_type: AxisType::Reduce, .. }),
                "Range should be Reduce type"
            );
        }
        other => panic!("Expected REDUCE, got {:?}", other),
    }
}

// =============================================================================
// Basic Transformation Tests
// =============================================================================

/// Test: REDUCE with GROUP_REDUCE range transforms to shared memory pattern.
#[test]
fn test_group_reduce_basic_transformation() {
    let group_range = create_group_reduce_range(0, 16);
    let src = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    let reduce = src.reduce(smallvec![group_range], ReduceOp::Add);

    let result = expander_rewrite(&reduce);

    // After transformation, the tree should contain:
    // 1. A BUFFERIZE with LOCAL address space
    // 2. A final REDUCE with renumbered ranges (axis_id >= 100)

    let all_nodes = result.toposort();

    // Check for LOCAL bufferize
    let has_local_buf = all_nodes.iter().any(|n| {
        matches!(n.op(), Op::Bufferize { opts, .. }
            if opts.addrspace == AddrSpace::Local)
    });
    assert!(has_local_buf, "Should create LOCAL BUFFERIZE for shared memory");

    // The outer op should be REDUCE
    if let Op::Reduce { ranges, .. } = result.op() {
        // All ranges should be renumbered (axis_id + 100)
        for range in ranges.iter() {
            if let Op::Range { axis_id, axis_type, .. } = range.op() {
                assert_eq!(*axis_type, AxisType::Reduce, "Final ranges should be Reduce type");
                assert!(
                    axis_id.value() >= 100,
                    "Ranges should be renumbered (axis_id >= 100), got {}",
                    axis_id.value()
                );
            }
        }
    } else {
        panic!("Expected REDUCE at top level, got {:?}", result.op());
    }
}

/// Test: GROUP_REDUCE with mixed ranges (GROUP_REDUCE + regular Reduce).
#[test]
fn test_group_reduce_with_mixed_ranges() {
    let group_range = create_group_reduce_range(0, 16);
    let reduce_range = create_reduce_range(1, 32);
    let src = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let reduce = src.reduce(smallvec![group_range, reduce_range], ReduceOp::Add);

    let result = expander_rewrite(&reduce);
    let all_nodes = result.toposort();

    // Should still create LOCAL bufferize for GROUP_REDUCE part
    let has_local_buf = all_nodes.iter().any(|n| {
        matches!(n.op(), Op::Bufferize { opts, .. }
            if opts.addrspace == AddrSpace::Local)
    });
    assert!(has_local_buf, "Should create LOCAL BUFFERIZE");

    // Check that transformation happened
    let has_group_reduce_in_final = all_nodes.iter().any(|n| {
        if let Op::Reduce { ranges, .. } = n.op() {
            ranges.iter().any(|r| matches!(r.op(), Op::Range { axis_type: AxisType::GroupReduce, .. }))
        } else {
            false
        }
    });
    assert!(!has_group_reduce_in_final, "GROUP_REDUCE should be transformed out of final REDUCEs");
}

// =============================================================================
// LOCAL Range Integration Tests
// =============================================================================

/// Test: GROUP_REDUCE with upstream LOCAL ranges includes them in buffer indices.
#[test]
fn test_group_reduce_with_local_ranges() {
    // Create a computation that includes LOCAL ranges in the dependency graph
    let local_range = create_local_range(0, 32);
    let group_range = create_group_reduce_range(1, 16);

    // Source that depends on local_range (by using it in arithmetic)
    // This simulates a computation indexed by local thread ID
    let local_float = local_range.clone().cast(DType::Float32);
    let scaled_src = local_float.try_mul(&UOp::const_(DType::Float32, ConstValue::Float(2.0))).unwrap();

    // Reduce over GROUP_REDUCE range
    let reduce = scaled_src.reduce(smallvec![group_range.clone()], ReduceOp::Add);

    let result = expander_rewrite(&reduce);
    let all_nodes = result.toposort();

    // Should contain BUFFERIZE with LOCAL address space
    let has_local_buf = all_nodes.iter().any(|n| {
        matches!(n.op(), Op::Bufferize { opts, .. }
            if opts.addrspace == AddrSpace::Local)
    });
    assert!(has_local_buf, "Should have LOCAL BUFFERIZE for GROUP_REDUCE");

    // The BUFFERIZE should have ranges that include the LOCAL range
    // (fix_group_for_reduce collects upstream_locals and includes them in buf_ranges)
    for node in all_nodes.iter() {
        if let Op::Bufferize { ranges, opts, .. } = node.op() {
            if opts.addrspace == AddrSpace::Local {
                // Check that LOCAL range type appears in buffer ranges
                let has_local_in_ranges =
                    ranges.iter().any(|r| matches!(r.op(), Op::Range { axis_type: AxisType::Local, .. }));
                assert!(has_local_in_ranges, "BUFFERIZE ranges should include LOCAL range for shared memory indexing");
            }
        }
    }
}

// =============================================================================
// Reduce Operation Preservation Tests
// =============================================================================

/// Test: Different reduce operations (Add, Max, Mul) are preserved.
#[test]
fn test_group_reduce_preserves_reduce_op() {
    for reduce_op in [ReduceOp::Add, ReduceOp::Max, ReduceOp::Mul] {
        let group_range = create_group_reduce_range(0, 8);
        let src = UOp::const_(DType::Float32, ConstValue::Float(1.0));
        let reduce = src.reduce(smallvec![group_range], reduce_op);

        let result = expander_rewrite(&reduce);

        // Final REDUCE should preserve the operation type
        if let Op::Reduce { reduce_op: final_op, .. } = result.op() {
            assert_eq!(*final_op, reduce_op, "Reduce operation should be preserved");
        } else {
            panic!("Expected REDUCE at top level for {:?}", reduce_op);
        }
    }
}

// =============================================================================
// Pattern Matcher Integration Tests
// =============================================================================

/// Test: pm_group_for_reduce is properly integrated in the pipeline.
#[test]
fn test_pm_group_for_reduce_in_pipeline() {
    use crate::expand::pre_expand;

    let group_range = create_group_reduce_range(0, 16);
    let src = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    let reduce = src.reduce(smallvec![group_range], ReduceOp::Add);

    // Run through pre_expand (which includes pm_group_for_reduce)
    let expanded = pre_expand(&reduce);

    // After pre_expand, GROUP_REDUCE should be transformed
    let has_group_reduce_in_reduce = expanded.toposort().iter().any(|n| {
        if let Op::Reduce { ranges, .. } = n.op() {
            ranges.iter().any(|r| matches!(r.op(), Op::Range { axis_type: AxisType::GroupReduce, .. }))
        } else {
            false
        }
    });
    assert!(!has_group_reduce_in_reduce, "GROUP_REDUCE should be transformed by pm_group_for_reduce");
}

// =============================================================================
// Edge Cases
// =============================================================================

/// Test: Multiple GROUP_REDUCE ranges.
#[test]
fn test_multiple_group_reduce_ranges() {
    let group_range1 = create_group_reduce_range(0, 8);
    let group_range2 = create_group_reduce_range(1, 4);
    let src = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let reduce = src.reduce(smallvec![group_range1, group_range2], ReduceOp::Add);

    let result = expander_rewrite(&reduce);
    let all_nodes = result.toposort();

    // Should create LOCAL BUFFERIZE
    let has_local_buf = all_nodes.iter().any(|n| {
        matches!(n.op(), Op::Bufferize { opts, .. }
            if opts.addrspace == AddrSpace::Local)
    });
    assert!(has_local_buf, "Should create LOCAL BUFFERIZE for multiple GROUP_REDUCE");
}

/// Test: GROUP_REDUCE with no other ranges.
#[test]
fn test_group_reduce_only() {
    let group_range = create_group_reduce_range(0, 32);
    let src = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    let reduce = src.reduce(smallvec![group_range], ReduceOp::Add);

    let result = expander_rewrite(&reduce);

    // Should still transform correctly
    if let Op::Reduce { .. } = result.op() {
        // Good - final result is REDUCE
    } else {
        panic!("Expected REDUCE at top level, got {:?}", result.op());
    }
}
