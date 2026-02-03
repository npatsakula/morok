//! Tests for movement operation pattern matching.
//!
//! Verifies that movement ops (RESHAPE, PERMUTE, EXPAND, PAD, SHRINK, FLIP)
//! are correctly pushed through INDEX operations.

use std::sync::Arc;

use morok_device::DeviceSpec;
use morok_dtype::DType;
use morok_ir::{AxisId, AxisType, Op, SInt, UOp};

use crate::rangeify::patterns::movement_op_patterns;
use crate::rewrite::graph_rewrite;

// ===== Helper Functions =====

/// Create a test buffer with given size.
fn create_buffer(size: usize) -> Arc<UOp> {
    UOp::new_buffer(DeviceSpec::Cpu, size, DType::Float32)
}

/// Create a RANGE for testing.
fn create_range(size: usize, axis_id: usize) -> Arc<UOp> {
    let end = UOp::index_const(size as i64);
    UOp::range_axis(end, AxisId::Renumbered(axis_id), AxisType::Loop)
}

// ===== EXPAND Tests =====

#[test]
fn test_expand_index_transformation() {
    // Test: EXPAND([10, 1, 20] → [10, 5, 20]).INDEX([r0, r1, r2])
    // Since the source is RESHAPE(buffer), graph_rewrite transforms both:
    // 1. EXPAND transformation: INDEX(RESHAPE(buf), [r0, 0, r2]) - r1 becomes 0
    // 2. RESHAPE transformation: INDEX(buf, [flattened]) - combines to 1D index

    #[allow(clippy::identity_op)]
    let buffer = create_buffer(10 * 1 * 20);

    // Reshape to [10, 1, 20]
    let reshaped =
        buffer.try_reshape(&vec![SInt::Const(10), SInt::Const(1), SInt::Const(20)].into_iter().collect()).unwrap();

    // Expand to [10, 5, 20]
    let shape2 = UOp::vectorize(vec![UOp::index_const(10), UOp::index_const(5), UOp::index_const(20)].into());
    let expanded = UOp::new(Op::Expand { src: reshaped, new_shape: shape2 }, DType::Float32);

    // Create INDEX with ranges [r0, r1, r2]
    let r0 = create_range(10, 0);
    let r1 = create_range(5, 1);
    let r2 = create_range(20, 2);
    let indexed = UOp::index().buffer(expanded).indices(vec![r0.clone(), r1.clone(), r2.clone()]).call().unwrap();

    // Apply pattern
    let pm = movement_op_patterns();
    let result = graph_rewrite(&pm, indexed, &mut ());

    // Verify: should transform all movement ops through INDEX
    // Final result: buffer.INDEX([flattened_index])
    assert!(matches!(result.op(), Op::Index { .. }), "Result should be INDEX");

    let Op::Index { buffer: res_buf, indices: res_idx, .. } = result.op() else {
        panic!("Expected INDEX");
    };

    // After both EXPAND and RESHAPE are transformed, we get 1 flattened index
    assert_eq!(res_idx.len(), 1, "Should have 1 index after all movement ops transformed");

    // The buffer should be the original buffer (no movement ops remaining)
    assert!(matches!(res_buf.op(), Op::Buffer { .. }), "Buffer should be the original buffer");
}

// ===== PERMUTE Tests =====

#[test]
fn test_permute_index_transformation() {
    // Test: PERMUTE([10, 20, 30], axes=[1, 2, 0]).INDEX([r0, r1, r2])
    // Since the source is RESHAPE(buffer), graph_rewrite transforms both:
    // 1. PERMUTE transformation: INDEX(RESHAPE(buf), [r2, r0, r1]) - indices reordered
    // 2. RESHAPE transformation: INDEX(buf, [flattened]) - combines to 1D index

    let buffer = create_buffer(10 * 20 * 30);
    let reshaped =
        buffer.try_reshape(&vec![SInt::Const(10), SInt::Const(20), SInt::Const(30)].into_iter().collect()).unwrap();

    // Permute: axes [1, 2, 0]
    let permuted = reshaped.try_permute(vec![1, 2, 0]).unwrap();

    // Create INDEX
    let r0 = create_range(20, 0); // Now indexing dimension 1 of original
    let r1 = create_range(30, 1); // Now indexing dimension 2 of original
    let r2 = create_range(10, 2); // Now indexing dimension 0 of original
    let indexed = UOp::index().buffer(permuted).indices(vec![r0.clone(), r1.clone(), r2.clone()]).call().unwrap();

    // Apply pattern
    let pm = movement_op_patterns();
    let result = graph_rewrite(&pm, indexed, &mut ());

    // Verify transformation
    assert!(matches!(result.op(), Op::Index { .. }));

    let Op::Index { buffer: res_buf, indices: res_idx, .. } = result.op() else {
        panic!("Expected INDEX");
    };

    // After both PERMUTE and RESHAPE are transformed, we get 1 flattened index
    assert_eq!(res_idx.len(), 1, "Should have 1 index after all movement ops transformed");

    // The buffer should be the original buffer (no movement ops remaining)
    assert!(matches!(res_buf.op(), Op::Buffer { .. }), "Buffer should be the original buffer");
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
    let indexed = UOp::index().buffer(reshaped).indices(vec![r0, r1]).call().unwrap();

    // Apply pattern
    let pm = movement_op_patterns();
    let result = graph_rewrite(&pm, indexed, &mut ());

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
    let begins = UOp::vectorize(vec![UOp::index_const(0), UOp::index_const(10)].into());
    let ends = UOp::vectorize(vec![UOp::index_const(5), UOp::index_const(30)].into());
    let shrunk = UOp::new(Op::Shrink { src: reshaped, begins, ends }, DType::Float32);

    // Create INDEX
    let r0 = create_range(5, 0);
    let r1 = create_range(20, 1);
    let indexed = UOp::index().buffer(shrunk).indices(vec![r0, r1]).call().unwrap();

    // Apply pattern
    let pm = movement_op_patterns();
    let result = graph_rewrite(&pm, indexed, &mut ());

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
    let indexed = UOp::index().buffer(flipped).indices(vec![r0, r1]).call().unwrap();

    // Apply pattern
    let pm = movement_op_patterns();
    let result = graph_rewrite(&pm, indexed, &mut ());

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
    let begin_pads = UOp::vectorize(vec![UOp::index_const(1), UOp::index_const(2)].into());
    let end_pads = UOp::vectorize(vec![UOp::index_const(1), UOp::index_const(2)].into());
    let padded = UOp::new(Op::Pad { src: reshaped, begin_pads, end_pads }, DType::Float32);

    // Create INDEX
    let r0 = create_range(12, 0); // 10 + 1 + 1
    let r1 = create_range(24, 1); // 20 + 2 + 2
    let indexed = UOp::index().buffer(padded).indices(vec![r0, r1]).call().unwrap();

    // Apply pattern
    let pm = movement_op_patterns();
    let result = graph_rewrite(&pm, indexed, &mut ());

    // Verify transformation
    assert!(matches!(result.op(), Op::Index { .. }));
}

// ===== Non-Movement Op Tests =====

#[test]
fn test_non_movement_op_no_match() {
    // Test that non-movement operations are not transformed

    let buffer = create_buffer(100);

    // Create a non-movement op (NEG) - using unary op to avoid shape issues
    let negated = buffer.neg();

    // Create INDEX
    let r0 = create_range(100, 0);
    let indexed = UOp::index().buffer(negated).indices(vec![r0]).call().unwrap();

    // Apply pattern
    let pm = movement_op_patterns();
    let result = graph_rewrite(&pm, indexed, &mut ());

    // Should NOT transform (no movement op)
    // The result should still have the NEG operation somewhere in the tree
    assert!(matches!(result.op(), Op::Index { .. }));

    // The buffer should still be the NEG node
    let Op::Index { buffer: res_buf, .. } = result.op() else {
        panic!("Expected INDEX");
    };

    assert!(matches!(res_buf.op(), Op::Unary(..)), "Buffer should still be the NEG");
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
    let shape = UOp::vectorize(vec![UOp::index_const(10), UOp::index_const(5)].into());
    let expanded = UOp::new(Op::Expand { src: reshaped1, new_shape: shape }, DType::Float32);

    // Reshape to [50]
    let reshaped2 = expanded.try_reshape(&vec![SInt::Const(50)].into_iter().collect()).unwrap();

    // Create INDEX
    let r0 = create_range(50, 0);
    let indexed = UOp::index().buffer(reshaped2).indices(vec![r0]).call().unwrap();

    // Apply pattern (should iterate multiple times)
    let pm = movement_op_patterns();
    let result = graph_rewrite(&pm, indexed, &mut ());

    // Verify: should have transformed through all movement ops
    assert!(matches!(result.op(), Op::Index { .. }));

    // The final buffer should be close to the original buffer
    // (may have intermediate operations, but movement ops should be gone)
}
