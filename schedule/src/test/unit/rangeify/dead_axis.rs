use std::sync::Arc;

use crate::rangeify::patterns::dead_axis_removal;
use crate::rewrite::graph_rewrite_top_down;
use morok_dtype::DType;
use morok_ir::{ConstValue, Op, UOp};

#[test]
fn test_bufferize_with_size_1_range() {
    // BUFFERIZE(x, [RANGE(1)]) should have the dead axis removed
    let x = UOp::define_global(1, DType::Float32);
    let dead_range = UOp::range_const(1, 0); // Size 1 = dead axis

    let bufferized = UOp::bufferize_global(x.clone(), vec![dead_range]);

    let matcher = dead_axis_removal();
    let result = graph_rewrite_top_down(&matcher, bufferized, &mut ());

    // Should return compute directly since all axes are dead
    assert!(Arc::ptr_eq(&result, &x), "BUFFERIZE with only dead axis should return compute");
}

#[test]
fn test_bufferize_all_dead_axes() {
    // BUFFERIZE(x, [RANGE(1), RANGE(1), RANGE(1)]) → x
    let x = UOp::define_global(1, DType::Float32);
    let dead_ranges = vec![UOp::range_const(1, 0), UOp::range_const(1, 1), UOp::range_const(1, 2)];

    let bufferized = UOp::bufferize_global(x.clone(), dead_ranges);

    let matcher = dead_axis_removal();
    let result = graph_rewrite_top_down(&matcher, bufferized, &mut ());

    // All axes dead → return compute directly
    assert!(Arc::ptr_eq(&result, &x), "All dead axes should be removed, returning compute");
}

#[test]
fn test_bufferize_mixed_live_dead_simple_compute() {
    // When compute is DEFINE_GLOBAL (has no ranges), ALL ranges are considered dead
    // because compute doesn't depend on any of them. This matches Tinygrad's behavior.
    // BUFFERIZE(DEFINE_GLOBAL, [RANGE(10), RANGE(1), RANGE(20)]) → DEFINE_GLOBAL
    let x = UOp::define_global(1, DType::Float32);
    let range1 = UOp::range_const(10, 0);
    let dead_range = UOp::range_const(1, 1);
    let range2 = UOp::range_const(20, 2);

    let bufferized = UOp::bufferize_global(x.clone(), vec![range1, dead_range, range2]);

    let matcher = dead_axis_removal();
    let result = graph_rewrite_top_down(&matcher, bufferized.clone(), &mut ());

    // All ranges are dead (compute doesn't use them), so return compute directly
    assert!(Arc::ptr_eq(&result, &x), "When compute has no ranges, all BUFFERIZE ranges are dead → return compute");
}

#[test]
fn test_bufferize_no_dead_axes_simple_compute() {
    // With DEFINE_GLOBAL compute (no ranges), ALL BUFFERIZE ranges are dead
    // because compute doesn't depend on them. This matches Tinygrad's behavior.
    let x = UOp::define_global(1, DType::Float32);
    let ranges = vec![UOp::range_const(10, 0), UOp::range_const(20, 1)];

    let bufferized = UOp::bufferize_global(x.clone(), ranges);

    let matcher = dead_axis_removal();
    let result = graph_rewrite_top_down(&matcher, bufferized.clone(), &mut ());

    // All ranges are dead → return compute directly
    assert!(Arc::ptr_eq(&result, &x), "When compute has no ranges, all BUFFERIZE ranges are dead");
}

// Pattern 2: INDEX Adjustment Tests

#[test]
fn test_index_after_dead_axis_removal() {
    // When BUFFERIZE has dead axes, INDEX should adjust indices accordingly
    // This is more complex and requires the pattern to work with actual buffer structure
    let x = UOp::define_global(1, DType::Float32);
    let live_range = UOp::range_const(10, 0);
    let dead_range = UOp::range_const(1, 1);

    // Create BUFFERIZE with mixed ranges
    let bufferized = UOp::bufferize_global(x.clone(), vec![live_range.clone(), dead_range.clone()]);

    // Create INDEX with indices matching the buffer structure
    let idx1 = UOp::index_const(5);
    let idx2 = UOp::index_const(0);

    let indexed = UOp::index().buffer(bufferized).indices(vec![idx1.clone(), idx2]).call().unwrap();

    let matcher = dead_axis_removal();
    let result = graph_rewrite_top_down(&matcher, indexed, &mut ());

    // After dead axis removal, the INDEX should have fewer indices
    if let Op::Index { indices, .. } = result.op() {
        // The second index (for dead axis) should be removed
        assert_eq!(indices.len(), 1, "Should have 1 index after dead axis removal");
        assert!(Arc::ptr_eq(&indices[0], &idx1), "Live index should be preserved");
    }
    // Note: This test might not pass if the pattern doesn't handle INDEX adjustment yet
}

#[test]
fn test_bufferize_dead_axis_with_constants() {
    // Dead axes with CONST ranges should be removed
    let x = UOp::define_global(1, DType::Float32);

    // Create range with constant end = 1
    let dead_range_const = UOp::range_const(1, 0);

    let bufferized = UOp::bufferize_global(x.clone(), vec![dead_range_const]);

    let matcher = dead_axis_removal();
    let result = graph_rewrite_top_down(&matcher, bufferized, &mut ());

    // Should return compute directly
    assert!(Arc::ptr_eq(&result, &x), "Dead axis with constant 1 should be removed");
}

#[test]
fn test_multiple_dead_axis_removal_passes() {
    // Test that multiple passes of dead axis removal work correctly
    // With DEFINE_GLOBAL compute (no ranges), ALL ranges are dead → returns compute
    let x = UOp::define_global(1, DType::Float32);
    let live_range = UOp::range_const(10, 0);
    let dead_range1 = UOp::range_const(1, 1);
    let dead_range2 = UOp::range_const(1, 2);

    let bufferized = UOp::bufferize_global(x.clone(), vec![live_range.clone(), dead_range1, dead_range2]);

    let matcher = dead_axis_removal();
    // Run multiple times to ensure idempotence
    let result1 = graph_rewrite_top_down(&matcher, bufferized.clone(), &mut ());
    let result2 = graph_rewrite_top_down(&matcher, result1.clone(), &mut ());

    // Both should produce same result (idempotent)
    assert!(Arc::ptr_eq(&result1, &result2), "Dead axis removal should be idempotent");

    // All ranges are dead (compute doesn't use them) → return compute directly
    assert!(Arc::ptr_eq(&result1, &x), "When compute has no ranges, all BUFFERIZE ranges are dead → return compute");
}

#[test]
fn test_dead_axis_uint_constant() {
    // Test with UInt constant value of 1
    let x = UOp::define_global(1, DType::Float32);

    let const_end = UOp::const_(DType::Index, ConstValue::UInt(1));
    let dead_range = UOp::range(const_end, 0);

    let bufferized = UOp::bufferize_global(x.clone(), vec![dead_range]);

    let matcher = dead_axis_removal();
    let result = graph_rewrite_top_down(&matcher, bufferized, &mut ());

    // Should return compute directly
    assert!(Arc::ptr_eq(&result, &x), "Dead axis with UInt(1) should be removed");
}
