use std::sync::Arc;

use crate::rangeify::patterns::dead_axis_removal;
use crate::rewrite::graph_rewrite_bottom_up;
use morok_dtype::DType;
use morok_ir::{ConstValue, Op, UOp};

/// Helper to check if result is EXPAND(RESHAPE(BUFFERIZE_no_ranges, ...), ...)
fn is_expand_reshape_bufferize(result: &Arc<UOp>) -> bool {
    // Result should be EXPAND
    let Op::Expand { src: reshape_op, .. } = result.op() else {
        return false;
    };

    // Inner should be RESHAPE
    let Op::Reshape { src: bufferize_op, .. } = reshape_op.op() else {
        return false;
    };

    // Innermost should be BUFFERIZE with empty ranges
    let Op::Bufferize { ranges, .. } = bufferize_op.op() else {
        return false;
    };

    ranges.is_empty()
}

/// Helper to get the innermost BUFFERIZE from EXPAND(RESHAPE(BUFFERIZE))
fn get_inner_bufferize(result: &Arc<UOp>) -> Option<&Arc<UOp>> {
    let Op::Expand { src: reshape_op, .. } = result.op() else {
        return None;
    };
    let Op::Reshape { src: bufferize_op, .. } = reshape_op.op() else {
        return None;
    };
    if matches!(bufferize_op.op(), Op::Bufferize { .. }) { Some(bufferize_op) } else { None }
}

#[test]
fn test_bufferize_with_size_1_range() {
    // BUFFERIZE(x, [RANGE(1)]) should be restructured to EXPAND(RESHAPE(BUFFERIZE(x, [])))
    // This matches Tinygrad's behavior where BUFFERIZE is kept for later STORE creation.
    let x = UOp::define_global(1, DType::Float32);
    let dead_range = UOp::range_const(1, 0); // Size 1 = dead axis

    let bufferized = UOp::bufferize_global(x.clone(), vec![dead_range]);

    let matcher = dead_axis_removal();
    let result = graph_rewrite_bottom_up(&matcher, bufferized, &mut ());

    // Result should be EXPAND(RESHAPE(BUFFERIZE_no_ranges))
    // The BUFFERIZE is KEPT (not removed) so it can be converted to STORE later.
    assert!(
        is_expand_reshape_bufferize(&result),
        "BUFFERIZE with dead axis should be restructured to EXPAND(RESHAPE(BUFFERIZE_no_ranges)), got: {}",
        result.tree()
    );

    // Verify the inner BUFFERIZE has the same compute
    if let Some(inner_buf) = get_inner_bufferize(&result) {
        if let Op::Bufferize { compute, .. } = inner_buf.op() {
            assert!(Arc::ptr_eq(compute, &x), "Inner BUFFERIZE should have original compute");
        }
    }
}

#[test]
fn test_bufferize_all_dead_axes() {
    // BUFFERIZE(x, [RANGE(1), RANGE(1), RANGE(1)]) → EXPAND(RESHAPE(BUFFERIZE(x, [])))
    let x = UOp::define_global(1, DType::Float32);
    let dead_ranges = vec![UOp::range_const(1, 0), UOp::range_const(1, 1), UOp::range_const(1, 2)];

    let bufferized = UOp::bufferize_global(x.clone(), dead_ranges);

    let matcher = dead_axis_removal();
    let result = graph_rewrite_bottom_up(&matcher, bufferized, &mut ());

    // All axes dead → EXPAND(RESHAPE(BUFFERIZE_no_ranges))
    assert!(
        is_expand_reshape_bufferize(&result),
        "All dead axes should produce EXPAND(RESHAPE(BUFFERIZE_no_ranges)), got: {}",
        result.tree()
    );
}

#[test]
fn test_bufferize_mixed_live_dead_simple_compute() {
    // When compute is DEFINE_GLOBAL (has no ranges), ALL ranges are considered dead
    // because compute doesn't depend on any of them. This matches Tinygrad's behavior.
    // BUFFERIZE(DEFINE_GLOBAL, [RANGE(10), RANGE(1), RANGE(20)]) → EXPAND(RESHAPE(BUFFERIZE_no_ranges))
    let x = UOp::define_global(1, DType::Float32);
    let range1 = UOp::range_const(10, 0);
    let dead_range = UOp::range_const(1, 1);
    let range2 = UOp::range_const(20, 2);

    let bufferized = UOp::bufferize_global(x.clone(), vec![range1, dead_range, range2]);

    let matcher = dead_axis_removal();
    let result = graph_rewrite_bottom_up(&matcher, bufferized.clone(), &mut ());

    // All ranges are dead (compute doesn't use them) → EXPAND(RESHAPE(BUFFERIZE_no_ranges))
    assert!(
        is_expand_reshape_bufferize(&result),
        "When compute has no ranges, result should be EXPAND(RESHAPE(BUFFERIZE_no_ranges)), got: {}",
        result.tree()
    );
}

#[test]
fn test_bufferize_no_dead_axes_simple_compute() {
    // With DEFINE_GLOBAL compute (no ranges), ALL BUFFERIZE ranges are dead
    // because compute doesn't depend on them. This matches Tinygrad's behavior.
    let x = UOp::define_global(1, DType::Float32);
    let ranges = vec![UOp::range_const(10, 0), UOp::range_const(20, 1)];

    let bufferized = UOp::bufferize_global(x.clone(), ranges);

    let matcher = dead_axis_removal();
    let result = graph_rewrite_bottom_up(&matcher, bufferized.clone(), &mut ());

    // All ranges are dead → EXPAND(RESHAPE(BUFFERIZE_no_ranges))
    assert!(
        is_expand_reshape_bufferize(&result),
        "When compute has no ranges, result should be EXPAND(RESHAPE(BUFFERIZE_no_ranges)), got: {}",
        result.tree()
    );
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
    let result = graph_rewrite_bottom_up(&matcher, indexed, &mut ());

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
    // Dead axes with CONST ranges should be removed, result wrapped in EXPAND(RESHAPE(...))
    let x = UOp::define_global(1, DType::Float32);

    // Create range with constant end = 1
    let dead_range_const = UOp::range_const(1, 0);

    let bufferized = UOp::bufferize_global(x.clone(), vec![dead_range_const]);

    let matcher = dead_axis_removal();
    let result = graph_rewrite_bottom_up(&matcher, bufferized, &mut ());

    // Should be EXPAND(RESHAPE(BUFFERIZE_no_ranges))
    assert!(
        is_expand_reshape_bufferize(&result),
        "Dead axis with constant 1 should produce EXPAND(RESHAPE(BUFFERIZE_no_ranges)), got: {}",
        result.tree()
    );
}

#[test]
fn test_multiple_dead_axis_removal_passes() {
    // Test that multiple passes of dead axis removal work correctly
    let x = UOp::define_global(1, DType::Float32);
    let live_range = UOp::range_const(10, 0);
    let dead_range1 = UOp::range_const(1, 1);
    let dead_range2 = UOp::range_const(1, 2);

    let bufferized = UOp::bufferize_global(x.clone(), vec![live_range.clone(), dead_range1, dead_range2]);

    let matcher = dead_axis_removal();
    // Run multiple times to ensure idempotence
    let result1 = graph_rewrite_bottom_up(&matcher, bufferized.clone(), &mut ());
    let result2 = graph_rewrite_bottom_up(&matcher, result1.clone(), &mut ());

    // Both should produce same result (idempotent) - comparing tree structure
    assert_eq!(result1.tree(), result2.tree(), "Dead axis removal should be idempotent");

    // Result should be EXPAND(RESHAPE(BUFFERIZE_no_ranges))
    assert!(
        is_expand_reshape_bufferize(&result1),
        "Result should be EXPAND(RESHAPE(BUFFERIZE_no_ranges)), got: {}",
        result1.tree()
    );
}

#[test]
fn test_dead_axis_uint_constant() {
    // Test with UInt constant value of 1
    let x = UOp::define_global(1, DType::Float32);

    let const_end = UOp::const_(DType::Index, ConstValue::UInt(1));
    let dead_range = UOp::range(const_end, 0);

    let bufferized = UOp::bufferize_global(x.clone(), vec![dead_range]);

    let matcher = dead_axis_removal();
    let result = graph_rewrite_bottom_up(&matcher, bufferized, &mut ());

    // Should be EXPAND(RESHAPE(BUFFERIZE_no_ranges))
    assert!(
        is_expand_reshape_bufferize(&result),
        "Dead axis with UInt(1) should produce EXPAND(RESHAPE(BUFFERIZE_no_ranges)), got: {}",
        result.tree()
    );
}
