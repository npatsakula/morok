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

/// Create a test buffer.
fn make_buffer() -> Arc<UOp> {
    let ptr_dtype = DType::Float32.ptr(Some(1024), morok_dtype::AddrSpace::Global);
    UOp::define_global(0, ptr_dtype)
}

#[test]
fn test_linearize_pattern_2d() {
    let buffer = make_buffer();
    let i = make_range(4, 0); // Dimension 4
    let j = make_range(8, 1); // Dimension 8

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
    let buffer = make_buffer();
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
    let buffer = make_buffer();
    let i = make_range(10, 0);

    let single_index = UOp::index(buffer.clone(), vec![i.clone()]).unwrap();

    let pattern = pm_linearize_multi_index();
    let result = crate::rewrite::graph_rewrite_bottom_up(&pattern, single_index.clone(), &mut ());

    // Single-index should be unchanged
    assert!(Arc::ptr_eq(&result, &single_index), "Single index should not be transformed");
}

#[test]
fn test_linearize_pattern_4d() {
    let buffer = make_buffer();
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
