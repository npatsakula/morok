//! Integration tests for multi-index linearization pass.
//!
//! Tests the public `pm_linearize_multi_index()` pattern matcher.
//! Implementation detail tests (helpers) are kept inline with the code.

use std::sync::Arc;

use morok_ir::{AxisId, AxisType, DType, Op, UOp};

use crate::passes::pm_linearize_multi_index;

/// Create a RANGE for testing.
fn make_range(size: i64, axis_id: usize) -> Arc<UOp> {
    let end = UOp::index_const(size);
    UOp::range_axis(end, AxisId::Renumbered(axis_id), AxisType::Loop)
}

/// Create a BUFFERIZE with given dimensions.
/// This creates a buffer with a proper multi-dimensional shape.
fn make_bufferize(dims: &[i64]) -> Arc<UOp> {
    // Create a dummy computation (const value)
    let compute = UOp::const_(DType::Float32, morok_ir::ConstValue::Float(0.0));

    // Create ranges for each dimension
    let ranges: Vec<Arc<UOp>> = dims
        .iter()
        .enumerate()
        .map(|(i, &size)| make_range(size, i))
        .collect();

    UOp::bufferize_global(compute, ranges)
}

#[test]
fn test_linearize_pattern_2d() {
    // Create a 4x8 buffer
    let buffer = make_bufferize(&[4, 8]);
    let i = make_range(4, 0);
    let j = make_range(8, 1);

    // Create INDEX(buffer, [i, j])
    let multi_index = UOp::index(buffer.clone(), vec![i.clone(), j.clone()]).unwrap();
    assert_eq!(multi_index.op().sources().len(), 3); // buffer, i, j

    // Apply linearization pattern
    let pattern = pm_linearize_multi_index();
    let result = crate::rewrite::graph_rewrite_bottom_up(&pattern, multi_index.clone(), &mut ());

    // Result should be INDEX(buffer, [linear])
    if let Op::Index { indices, .. } = result.op() {
        assert_eq!(indices.len(), 1, "Should have single linear index after linearization");
    } else {
        panic!("Expected INDEX op after linearization");
    }
}

#[test]
fn test_linearize_pattern_3d() {
    // Create a 2x3x4 buffer
    let buffer = make_bufferize(&[2, 3, 4]);
    let i = make_range(2, 0);
    let j = make_range(3, 1);
    let k = make_range(4, 2);

    let multi_index = UOp::index(buffer.clone(), vec![i, j, k]).unwrap();

    let pattern = pm_linearize_multi_index();
    let result = crate::rewrite::graph_rewrite_bottom_up(&pattern, multi_index.clone(), &mut ());

    if let Op::Index { indices, .. } = result.op() {
        assert_eq!(indices.len(), 1, "3D index should be linearized to 1D");
    } else {
        panic!("Expected INDEX op");
    }
}

#[test]
fn test_single_index_unchanged() {
    // Create a 1D buffer with size 10
    let buffer = make_bufferize(&[10]);
    let i = make_range(10, 0);

    let single_index = UOp::index(buffer.clone(), vec![i.clone()]).unwrap();

    let pattern = pm_linearize_multi_index();
    let result = crate::rewrite::graph_rewrite_bottom_up(&pattern, single_index.clone(), &mut ());

    // Single-index should be unchanged
    assert!(Arc::ptr_eq(&result, &single_index), "Single index should not be transformed");
}

#[test]
fn test_linearize_pattern_4d() {
    // Create a 2x3x4x5 buffer
    let buffer = make_bufferize(&[2, 3, 4, 5]);
    let i = make_range(2, 0);
    let j = make_range(3, 1);
    let k = make_range(4, 2);
    let l = make_range(5, 3);

    let multi_index = UOp::index(buffer.clone(), vec![i, j, k, l]).unwrap();

    let pattern = pm_linearize_multi_index();
    let result = crate::rewrite::graph_rewrite_bottom_up(&pattern, multi_index.clone(), &mut ());

    if let Op::Index { indices, .. } = result.op() {
        assert_eq!(indices.len(), 1, "4D index should be linearized to 1D");
    } else {
        panic!("Expected INDEX op");
    }
}

#[test]
fn test_unbounded_buffer_still_linearizes() {
    // Create a buffer with unbounded size (no concrete shape)
    // With index-based dimension extraction, linearization should still work
    // because dimensions come from the RANGE indices, not the buffer
    let ptr_dtype = DType::Float32.ptr(None, morok_dtype::AddrSpace::Global);
    let buffer = UOp::define_global(0, ptr_dtype);
    let i = make_range(4, 0);
    let j = make_range(8, 1);

    let multi_index = UOp::index(buffer.clone(), vec![i, j]).unwrap();

    let pattern = pm_linearize_multi_index();
    let result = crate::rewrite::graph_rewrite_bottom_up(&pattern, multi_index.clone(), &mut ());

    // Should be linearized - dimensions come from indices
    if let Op::Index { indices, .. } = result.op() {
        assert_eq!(indices.len(), 1, "Should have single linear index after linearization");
    } else {
        panic!("Expected INDEX op after linearization");
    }
}
