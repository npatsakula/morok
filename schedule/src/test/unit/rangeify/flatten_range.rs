//! Tests for range flattening and canonicalization.
//!
//! Validates that flatten_range correctly unnests and canonicalizes RANGE operations
//! for kernel deduplication.

use std::rc::Rc;

use morok_dtype::DType;
use morok_ir::{AxisType, ConstValue, UOp};

use crate::rangeify::flatten_range::{flatten_range_impl, flatten_ranges, get_range_chain, get_range_parents};

#[test]
fn test_get_range_chain_empty() {
    // Non-RANGE operation should return empty chain
    let const_op = UOp::const_(DType::Index, ConstValue::Int(10));
    let chain = get_range_chain(&const_op);
    assert!(chain.is_empty());
}

#[test]
fn test_get_range_chain_single() {
    // Single RANGE should return itself
    let range = UOp::range_const(10, 0);

    let chain = get_range_chain(&range);
    assert_eq!(chain.len(), 1);
    assert!(Rc::ptr_eq(&chain[0], &range));
}

#[test]
fn test_get_range_chain_nested() {
    // Nested RANGEs should return all in order
    let const_end = UOp::const_(DType::Index, ConstValue::Int(10));
    let range1 = UOp::range_axis(const_end, 0, AxisType::Loop);
    let range2 = UOp::range_axis(range1.clone(), 1, AxisType::Loop);
    let range3 = UOp::range_axis(range2.clone(), 2, AxisType::Loop);

    let chain = get_range_chain(&range3);
    // Should return [range1, range2, range3]
    assert_eq!(chain.len(), 3);
    assert!(Rc::ptr_eq(&chain[2], &range3));
}

#[test]
fn test_get_range_parents_no_consumers() {
    // UOp with no consumers should return empty
    let uop = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let consumer_map = uop.get_consumer_map();

    let parents = get_range_parents(&uop, &consumer_map);
    assert!(parents.is_empty());
}

#[test]
fn test_get_range_parents_with_range_consumers() {
    // UOp consumed by RANGE should return that RANGE
    let end = UOp::const_(DType::Index, ConstValue::Int(10));
    let range = UOp::range_axis(end.clone(), 0, AxisType::Loop);

    // Build consumer map from range (which depends on end)
    let consumer_map = range.get_consumer_map();

    let parents = get_range_parents(&end, &consumer_map);
    assert_eq!(parents.len(), 1);
    assert!(Rc::ptr_eq(&parents[0], &range));
}

#[test]
fn test_flatten_range_impl_non_supported_op() {
    // Operations that don't support flattening should return None
    let const_op = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let consumer_map = const_op.get_consumer_map();

    let result = flatten_range_impl(&const_op, &consumer_map);
    assert!(result.is_none());
}

#[test]
fn test_flatten_range_impl_no_ranges() {
    // Supported operation with no ranges should return None
    let a = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let b = UOp::const_(DType::Float32, ConstValue::Float(2.0));
    let sum = a.try_add_op(&b).unwrap();

    let consumer_map = sum.get_consumer_map();
    let result = flatten_range_impl(&sum, &consumer_map);
    assert!(result.is_none());
}

#[test]
fn test_flatten_ranges_identity() {
    // Graph with no nested ranges should return unchanged
    let computation = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let flattened = flatten_ranges(&computation);

    // Should return identical graph (same pointer)
    assert!(Rc::ptr_eq(&flattened, &computation));
}
