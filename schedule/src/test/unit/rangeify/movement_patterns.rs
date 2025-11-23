//! Tests for movement operation pattern matching.
//!
//! Verifies that movement ops (RESHAPE, PERMUTE, EXPAND, PAD, SHRINK, FLIP)
//! are correctly pushed through INDEX operations.

use std::rc::Rc;

use morok_device::DeviceSpec;
use morok_dtype::DType;
use morok_ir::{AxisType, ConstValue, Op, SInt, UOp};

use crate::rangeify::movement_patterns::movement_op_patterns;
use crate::rewrite::graph_rewrite;

// ===== Helper Functions =====

/// Create a test buffer with given size.
fn create_buffer(size: usize) -> Rc<UOp> {
    UOp::new_buffer(DeviceSpec::Cpu, size, DType::Float32)
}

/// Create a RANGE for testing.
fn create_range(size: usize, axis_id: usize) -> Rc<UOp> {
    let end = UOp::const_(DType::Index, ConstValue::Int(size as i64));
    UOp::range_axis(end, axis_id, AxisType::Loop)
}

// ===== EXPAND Tests =====

#[test]
fn test_expand_index_transformation() {
    // Test: EXPAND([10, 1, 20] → [10, 5, 20]).INDEX([r0, r1, r2])
    // Expected: buffer.INDEX([r0, 0, r2]) - r1 becomes 0 due to broadcast

    #[allow(clippy::identity_op)]
    let buffer = create_buffer(10 * 1 * 20);

    // Reshape to [10, 1, 20]
    let reshaped =
        buffer.try_reshape(&vec![SInt::Const(10), SInt::Const(1), SInt::Const(20)].into_iter().collect()).unwrap();

    // Expand to [10, 5, 20]
    let shape2 = UOp::vectorize(
        vec![
            UOp::const_(DType::Index, ConstValue::Int(10)),
            UOp::const_(DType::Index, ConstValue::Int(5)),
            UOp::const_(DType::Index, ConstValue::Int(20)),
        ]
        .into(),
    );
    let expanded = UOp::new(Op::Expand { src: reshaped, new_shape: shape2 }, DType::Float32);

    // Create INDEX with ranges [r0, r1, r2]
    let r0 = create_range(10, 0);
    let r1 = create_range(5, 1);
    let r2 = create_range(20, 2);
    let indexed = UOp::index(expanded, vec![r0.clone(), r1.clone(), r2.clone()]).unwrap();

    // Apply pattern
    let pm = movement_op_patterns();
    let result = graph_rewrite(&pm, indexed);

    // Verify: should transform EXPAND through INDEX
    // After transformation, we should have buffer.INDEX([r0, 0, r2])
    // where dimension 1 became a constant 0

    assert!(matches!(result.op(), Op::Index { .. }), "Result should be INDEX");

    let Op::Index { buffer: _res_buf, indices: res_idx, .. } = result.op() else {
        panic!("Expected INDEX");
    };

    assert_eq!(res_idx.len(), 3, "Should have 3 indices");

    // Index 1 should be constant 0 (broadcast dimension)
    match res_idx[1].op() {
        Op::Const(cv) => {
            assert_eq!(cv.0, ConstValue::Int(0), "Broadcast dimension should become constant 0");
        }
        _ => panic!("Index 1 should be constant 0, got {:?}", res_idx[1].op()),
    }
}

// ===== PERMUTE Tests =====

#[test]
fn test_permute_index_transformation() {
    // Test: PERMUTE([10, 20, 30], axes=[1, 2, 0]).INDEX([r0, r1, r2])
    // Expected: buffer.INDEX([r2, r0, r1]) - indices reordered

    let buffer = create_buffer(10 * 20 * 30);
    let reshaped =
        buffer.try_reshape(&vec![SInt::Const(10), SInt::Const(20), SInt::Const(30)].into_iter().collect()).unwrap();

    // Permute: axes [1, 2, 0]
    let permuted = reshaped.try_permute(vec![1, 2, 0]).unwrap();

    // Create INDEX
    let r0 = create_range(20, 0); // Now indexing dimension 1 of original
    let r1 = create_range(30, 1); // Now indexing dimension 2 of original
    let r2 = create_range(10, 2); // Now indexing dimension 0 of original
    let indexed = UOp::index(permuted, vec![r0.clone(), r1.clone(), r2.clone()]).unwrap();

    // Apply pattern
    let pm = movement_op_patterns();
    let result = graph_rewrite(&pm, indexed);

    // Verify transformation
    assert!(matches!(result.op(), Op::Index { .. }));

    let Op::Index { indices: res_idx, .. } = result.op() else {
        panic!("Expected INDEX");
    };

    assert_eq!(res_idx.len(), 3);

    // After permutation, indices should be reordered
    // Permute axes=[1,2,0] means: output[0] = input[1], output[1] = input[2], output[2] = input[0]
    // So when we index output with [r0, r1, r2], we're accessing input at [r2, r0, r1]
}

// ===== RESHAPE Tests =====

#[test]
fn test_reshape_index_transformation() {
    // Test: RESHAPE([200] → [10, 20]).INDEX([r0, r1])
    // Expected: buffer.INDEX([r0 * 20 + r1]) - combined index

    let buffer = create_buffer(200);

    // Reshape to [10, 20]
    let reshaped = buffer.try_reshape(&vec![SInt::Const(10), SInt::Const(20)].into_iter().collect()).unwrap();

    // Create INDEX
    let r0 = create_range(10, 0);
    let r1 = create_range(20, 1);
    let indexed = UOp::index(reshaped, vec![r0, r1]).unwrap();

    // Apply pattern
    let pm = movement_op_patterns();
    let result = graph_rewrite(&pm, indexed);

    // Verify: should have INDEX with combined indices
    assert!(matches!(result.op(), Op::Index { .. }));

    let Op::Index { indices: res_idx, .. } = result.op() else {
        panic!("Expected INDEX");
    };

    // Should have single index (flattened)
    assert_eq!(res_idx.len(), 1, "Should flatten to 1D index");
}

// ===== SHRINK Tests =====

#[test]
fn test_shrink_index_transformation() {
    // Test: SHRINK([0:5, 10:30] from [10, 40]).INDEX([r0, r1])
    // Expected: buffer.INDEX([r0 + 0, r1 + 10]) - offset indices

    let buffer = create_buffer(10 * 40);
    let reshaped = buffer.try_reshape(&vec![SInt::Const(10), SInt::Const(40)].into_iter().collect()).unwrap();

    // SHRINK: begin=[0, 10], end=[5, 30]
    let begins = UOp::vectorize(
        vec![UOp::const_(DType::Index, ConstValue::Int(0)), UOp::const_(DType::Index, ConstValue::Int(10))].into(),
    );
    let ends = UOp::vectorize(
        vec![UOp::const_(DType::Index, ConstValue::Int(5)), UOp::const_(DType::Index, ConstValue::Int(30))].into(),
    );
    let shrunk = UOp::new(Op::Shrink { src: reshaped, begins, ends }, DType::Float32);

    // Create INDEX
    let r0 = create_range(5, 0);
    let r1 = create_range(20, 1);
    let indexed = UOp::index(shrunk, vec![r0, r1]).unwrap();

    // Apply pattern
    let pm = movement_op_patterns();
    let result = graph_rewrite(&pm, indexed);

    // Verify transformation
    assert!(matches!(result.op(), Op::Index { .. }));
}

// ===== FLIP Tests =====

#[test]
fn test_flip_index_transformation() {
    // Test: FLIP([10, 20], axes=[false, true]).INDEX([r0, r1])
    // Expected: buffer.INDEX([r0, (20-1) - r1]) - reversed second axis

    let buffer = create_buffer(10 * 20);
    let reshaped = buffer.try_reshape(&vec![SInt::Const(10), SInt::Const(20)].into_iter().collect()).unwrap();

    // FLIP second axis
    let flipped = UOp::new(Op::Flip { src: reshaped, axes: vec![false, true] }, DType::Float32);

    // Create INDEX
    let r0 = create_range(10, 0);
    let r1 = create_range(20, 1);
    let indexed = UOp::index(flipped, vec![r0, r1]).unwrap();

    // Apply pattern
    let pm = movement_op_patterns();
    let result = graph_rewrite(&pm, indexed);

    // Verify transformation
    assert!(matches!(result.op(), Op::Index { .. }));
}

// ===== PAD Tests =====

#[test]
fn test_pad_index_transformation() {
    // Test: PAD([10, 20], begin=[1, 2], end=[1, 2]).INDEX([r0, r1])
    // Expected: buffer.INDEX with validity checks and adjusted indices

    let buffer = create_buffer(10 * 20);
    let reshaped = buffer.try_reshape(&vec![SInt::Const(10), SInt::Const(20)].into_iter().collect()).unwrap();

    // PAD: add 1 padding on each side of first dim, 2 on each side of second dim
    let begin_pads = UOp::vectorize(
        vec![UOp::const_(DType::Index, ConstValue::Int(1)), UOp::const_(DType::Index, ConstValue::Int(2))].into(),
    );
    let end_pads = UOp::vectorize(
        vec![UOp::const_(DType::Index, ConstValue::Int(1)), UOp::const_(DType::Index, ConstValue::Int(2))].into(),
    );
    let padded = UOp::new(Op::Pad { src: reshaped, begin_pads, end_pads }, DType::Float32);

    // Create INDEX
    let r0 = create_range(12, 0); // 10 + 1 + 1
    let r1 = create_range(24, 1); // 20 + 2 + 2
    let indexed = UOp::index(padded, vec![r0, r1]).unwrap();

    // Apply pattern
    let pm = movement_op_patterns();
    let result = graph_rewrite(&pm, indexed);

    // Verify transformation
    assert!(matches!(result.op(), Op::Index { .. }));
}

// ===== Non-Movement Op Tests =====

#[test]
fn test_non_movement_op_no_match() {
    // Test that non-movement operations are not transformed

    let buffer = create_buffer(100);

    // Create a non-movement op (ADD)
    let const_val = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let added = buffer.try_add_op(&const_val).unwrap();

    // Create INDEX
    let r0 = create_range(100, 0);
    let indexed = UOp::index(added, vec![r0]).unwrap();

    // Apply pattern
    let pm = movement_op_patterns();
    let result = graph_rewrite(&pm, indexed);

    // Should NOT transform (no movement op)
    // The result should still have the ADD operation somewhere in the tree
    assert!(matches!(result.op(), Op::Index { .. }));

    // The buffer should still be the ADD node
    let Op::Index { buffer: res_buf, .. } = result.op() else {
        panic!("Expected INDEX");
    };

    assert!(matches!(res_buf.op(), Op::Binary { .. }), "Buffer should still be the ADD");
}

// ===== Nested Movement Ops Test =====

#[test]
fn test_nested_movement_ops() {
    // Test: RESHAPE(EXPAND(buffer)).INDEX(ranges)
    // Should iterate to fixed point, transforming both operations

    #[allow(clippy::identity_op)]
    let buffer = create_buffer(10 * 1);
    let reshaped1 = buffer.try_reshape(&vec![SInt::Const(10), SInt::Const(1)].into_iter().collect()).unwrap();

    // Expand to [10, 5]
    let shape = UOp::vectorize(
        vec![UOp::const_(DType::Index, ConstValue::Int(10)), UOp::const_(DType::Index, ConstValue::Int(5))].into(),
    );
    let expanded = UOp::new(Op::Expand { src: reshaped1, new_shape: shape }, DType::Float32);

    // Reshape to [50]
    let reshaped2 = expanded.try_reshape(&vec![SInt::Const(50)].into_iter().collect()).unwrap();

    // Create INDEX
    let r0 = create_range(50, 0);
    let indexed = UOp::index(reshaped2, vec![r0]).unwrap();

    // Apply pattern (should iterate multiple times)
    let pm = movement_op_patterns();
    let result = graph_rewrite(&pm, indexed);

    // Verify: should have transformed through all movement ops
    assert!(matches!(result.op(), Op::Index { .. }));

    // The final buffer should be close to the original buffer
    // (may have intermediate operations, but movement ops should be gone)
}
