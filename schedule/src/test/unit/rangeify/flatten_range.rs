//! Tests for range flattening and canonicalization.
//!
//! Validates that flatten_range correctly unnests and canonicalizes RANGE operations
//! for kernel deduplication.

use std::rc::Rc;

use morok_ir::UOp;

use crate::rangeify::flatten_range::{flatten_range_impl, flatten_ranges};

#[test]
fn test_flatten_range_impl_non_supported_op() {
    // Operations that don't support flattening should return None
    let const_op = UOp::native_const(1.0f32);

    let result = flatten_range_impl(&const_op);
    assert!(result.is_none());
}

#[test]
fn test_flatten_range_impl_no_ranges() {
    // STORE operation with no ranges should return None
    let buffer = UOp::buffer_id(Some(0));
    let index = UOp::index_const(0);
    let value = UOp::native_const(1.0f32);
    let store = UOp::store(buffer, index, value);

    let result = flatten_range_impl(&store);
    assert!(result.is_none());
}

#[test]
fn test_flatten_ranges_identity() {
    // Graph with no nested ranges should return unchanged
    let computation = UOp::native_const(1.0f32);
    let flattened = flatten_ranges(&computation);

    // Should return identical graph (same pointer)
    assert!(Rc::ptr_eq(&flattened, &computation));
}
