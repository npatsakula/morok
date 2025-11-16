use std::rc::Rc;

use crate::rangeify::patterns::dead_axis_removal;
use crate::rewrite::graph_rewrite;
use morok_dtype::DType;
use morok_ir::{AddrSpace, AxisType, BufferizeOpts, ConstValue, Op, UOp};

// Helper functions
fn create_const(val: i64) -> Rc<UOp> {
    UOp::const_(DType::Int32, ConstValue::Int(val))
}

fn create_range(end: i64, axis_id: usize) -> Rc<UOp> {
    UOp::new(
        Op::Range {
            end: create_const(end),
            axis_id,
            axis_type: AxisType::Loop,
        },
        DType::Index,
    )
}

fn create_bufferize(compute: Rc<UOp>, ranges: Vec<Rc<UOp>>) -> Rc<UOp> {
    UOp::bufferize(
        compute,
        ranges,
        BufferizeOpts {
            device: None,
            addrspace: AddrSpace::Global,
        },
    )
}

// Pattern 1: BUFFERIZE Dead Axis Removal Tests

#[test]
fn test_bufferize_with_size_1_range() {
    // BUFFERIZE(x, [RANGE(1)]) should have the dead axis removed
    let x = UOp::define_global(1, DType::Float32);
    let dead_range = create_range(1, 0); // Size 1 = dead axis

    let bufferized = create_bufferize(x.clone(), vec![dead_range]);

    let matcher = dead_axis_removal();
    let result = graph_rewrite(&matcher, bufferized);

    // Should return compute directly since all axes are dead
    assert!(Rc::ptr_eq(&result, &x), "BUFFERIZE with only dead axis should return compute");
}

#[test]
fn test_bufferize_all_dead_axes() {
    // BUFFERIZE(x, [RANGE(1), RANGE(1), RANGE(1)]) → x
    let x = UOp::define_global(1, DType::Float32);
    let dead_ranges = vec![
        create_range(1, 0),
        create_range(1, 1),
        create_range(1, 2),
    ];

    let bufferized = create_bufferize(x.clone(), dead_ranges);

    let matcher = dead_axis_removal();
    let result = graph_rewrite(&matcher, bufferized);

    // All axes dead → return compute directly
    assert!(Rc::ptr_eq(&result, &x), "All dead axes should be removed, returning compute");
}

#[test]
fn test_bufferize_mixed_live_dead() {
    // BUFFERIZE(x, [RANGE(10), RANGE(1), RANGE(20)]) should keep only live axes
    let x = UOp::define_global(1, DType::Float32);
    let live_range1 = create_range(10, 0);
    let dead_range = create_range(1, 1); // Dead
    let live_range2 = create_range(20, 2);

    let bufferized = create_bufferize(
        x.clone(),
        vec![live_range1.clone(), dead_range, live_range2.clone()],
    );

    let matcher = dead_axis_removal();
    let result = graph_rewrite(&matcher, bufferized.clone());

    // Result should be BUFFERIZE with only live ranges
    if let Op::Bufferize { ranges, .. } = result.op() {
        assert_eq!(ranges.len(), 2, "Should have 2 live ranges");
        assert!(Rc::ptr_eq(&ranges[0], &live_range1), "First live range should be preserved");
        assert!(Rc::ptr_eq(&ranges[1], &live_range2), "Second live range should be preserved");
    } else {
        panic!("Expected BUFFERIZE after dead axis removal");
    }
}

#[test]
fn test_bufferize_no_dead_axes() {
    // BUFFERIZE(x, [RANGE(10), RANGE(20)]) should remain unchanged
    let x = UOp::define_global(1, DType::Float32);
    let live_ranges = vec![
        create_range(10, 0),
        create_range(20, 1),
    ];

    let bufferized = create_bufferize(x, live_ranges);

    let matcher = dead_axis_removal();
    let result = graph_rewrite(&matcher, bufferized.clone());

    // Should remain unchanged (no dead axes to remove)
    assert!(Rc::ptr_eq(&result, &bufferized), "No dead axes means no changes");
}

// Pattern 2: INDEX Adjustment Tests

#[test]
fn test_index_after_dead_axis_removal() {
    // When BUFFERIZE has dead axes, INDEX should adjust indices accordingly
    // This is more complex and requires the pattern to work with actual buffer structure
    let x = UOp::define_global(1, DType::Float32);
    let live_range = create_range(10, 0);
    let dead_range = create_range(1, 1);

    // Create BUFFERIZE with mixed ranges
    let bufferized = create_bufferize(
        x,
        vec![live_range.clone(), dead_range.clone()],
    );

    // Create INDEX with indices matching the buffer structure
    let idx1 = UOp::const_(DType::Index, ConstValue::Int(5));
    let idx2 = UOp::const_(DType::Index, ConstValue::Int(0));

    let indexed = UOp::index(bufferized, vec![idx1.clone(), idx2])
        .expect("Failed to create INDEX");

    let matcher = dead_axis_removal();
    let result = graph_rewrite(&matcher, indexed);

    // After dead axis removal, the INDEX should have fewer indices
    if let Op::Index { indices, .. } = result.op() {
        // The second index (for dead axis) should be removed
        assert_eq!(indices.len(), 1, "Should have 1 index after dead axis removal");
        assert!(Rc::ptr_eq(&indices[0], &idx1), "Live index should be preserved");
    }
    // Note: This test might not pass if the pattern doesn't handle INDEX adjustment yet
}

#[test]
fn test_bufferize_dead_axis_with_constants() {
    // Dead axes with CONST ranges should be removed
    let x = UOp::define_global(1, DType::Float32);

    // Create range with constant end = 1
    let const_end = UOp::const_(DType::Index, ConstValue::Int(1));
    let dead_range_const = UOp::new(
        Op::Range {
            end: const_end,
            axis_id: 0,
            axis_type: AxisType::Loop,
        },
        DType::Index,
    );

    let bufferized = create_bufferize(x.clone(), vec![dead_range_const]);

    let matcher = dead_axis_removal();
    let result = graph_rewrite(&matcher, bufferized);

    // Should return compute directly
    assert!(Rc::ptr_eq(&result, &x), "Dead axis with constant 1 should be removed");
}

#[test]
fn test_multiple_dead_axis_removal_passes() {
    // Test that multiple passes of dead axis removal work correctly
    let x = UOp::define_global(1, DType::Float32);
    let live_range = create_range(10, 0);
    let dead_range1 = create_range(1, 1);
    let dead_range2 = create_range(1, 2);

    let bufferized = create_bufferize(
        x.clone(),
        vec![live_range.clone(), dead_range1, dead_range2],
    );

    let matcher = dead_axis_removal();
    // Run multiple times to ensure idempotence
    let result1 = graph_rewrite(&matcher, bufferized.clone());
    let result2 = graph_rewrite(&matcher, result1.clone());

    // Both should produce same result (idempotent)
    assert!(Rc::ptr_eq(&result1, &result2), "Dead axis removal should be idempotent");

    // Result should have only the live range
    if let Op::Bufferize { ranges, .. } = result1.op() {
        assert_eq!(ranges.len(), 1, "Should have 1 live range");
        assert!(Rc::ptr_eq(&ranges[0], &live_range), "Live range should be preserved");
    } else {
        panic!("Expected BUFFERIZE after dead axis removal");
    }
}

#[test]
fn test_dead_axis_uint_constant() {
    // Test with UInt constant value of 1
    let x = UOp::define_global(1, DType::Float32);

    let const_end = UOp::const_(DType::Index, ConstValue::UInt(1));
    let dead_range = UOp::new(
        Op::Range {
            end: const_end,
            axis_id: 0,
            axis_type: AxisType::Loop,
        },
        DType::Index,
    );

    let bufferized = create_bufferize(x.clone(), vec![dead_range]);

    let matcher = dead_axis_removal();
    let result = graph_rewrite(&matcher, bufferized);

    // Should return compute directly
    assert!(Rc::ptr_eq(&result, &x), "Dead axis with UInt(1) should be removed");
}
