//! Tests for dead loop elimination via symbolic simplification.
//!
//! Tests verify that dead ranges (vmax ≤ 0) are properly eliminated through 3 patterns:
//! 1. RANGE with vmax ≤ 0 → Const(0)
//! 2. END with dead ranges → remove dead ranges or unwrap entirely
//! 3. REDUCE with all dead ranges → identity element

use morok_dtype::DType;
use morok_ir::types::{ConstValue, ReduceOp};
use morok_ir::{Op, UOp};
use smallvec::smallvec;
use std::sync::Arc;

use crate::rewrite::graph_rewrite;

use super::helpers::{assert_const_value, assert_end_range_count, assert_end_unwrapped, get_matcher};

// ============================================================================
// Priority 1: Core Transformation Tests
// ============================================================================

// ----------------------------------------------------------------------------
// RANGE Elimination Tests
// ----------------------------------------------------------------------------

#[test]
fn test_range_zero_to_const() {
    // RANGE(0) → Const(0)
    let zero = UOp::native_const(0i32);
    let range = UOp::range(zero, 0);

    let matcher = get_matcher();
    let result = graph_rewrite(&matcher, range, &mut ());

    assert_const_value(&result, ConstValue::Int(0));
}

#[test]
fn test_range_negative_to_const() {
    // RANGE(-5) → Const(0)
    let neg_five = UOp::native_const(-5i32);
    let range = UOp::range(neg_five, 0);

    let matcher = get_matcher();
    let result = graph_rewrite(&matcher, range, &mut ());

    assert_const_value(&result, ConstValue::Int(0));
}

// ----------------------------------------------------------------------------
// END Cleanup Tests
// ----------------------------------------------------------------------------

#[test]
fn test_end_all_dead_ranges_unwrapped() {
    // END(store, [RANGE(0)]) → store
    let store = UOp::noop();
    let dead_range = UOp::range_const(0, 0);
    let end = UOp::end(Arc::clone(&store), smallvec![dead_range]);

    let matcher = get_matcher();
    let result = graph_rewrite(&matcher, end, &mut ());

    // Should unwrap to just the store
    let unwrapped = assert_end_unwrapped(&result);
    assert!(Arc::ptr_eq(&unwrapped, &store), "Expected END to unwrap to original store");
}

#[test]
fn test_end_partial_dead_ranges_removed() {
    // END(store, [RANGE(10), RANGE(0), RANGE(5)])
    // → END(store, [RANGE(10), RANGE(5)])
    let store = UOp::noop();
    let live1 = UOp::range_const(10, 0);
    let dead = UOp::range_const(0, 0);
    let live2 = UOp::range_const(5, 0);
    let end = UOp::end(Arc::clone(&store), smallvec![Arc::clone(&live1), dead, Arc::clone(&live2)]);

    let matcher = get_matcher();
    let result = graph_rewrite(&matcher, end, &mut ());

    // Should have exactly 2 ranges (dead one removed)
    let (computation, ranges) = assert_end_range_count(&result, 2);

    // Verify it's the original store
    assert!(Arc::ptr_eq(&computation, &store), "Expected same computation");

    // Verify the live ranges are preserved in order
    assert!(Arc::ptr_eq(&ranges[0], &live1), "Expected first live range preserved");
    assert!(Arc::ptr_eq(&ranges[1], &live2), "Expected second live range preserved");
}

// ----------------------------------------------------------------------------
// REDUCE Identity Tests
// ----------------------------------------------------------------------------

#[test]
fn test_reduce_add_empty_to_zero() {
    // REDUCE(x, [RANGE(0)], Add) → Const(0)
    let src = UOp::var("x", DType::Int32, 100);
    let dead_range = UOp::range_const(0, 0);
    let reduce = UOp::reduce(src, smallvec![dead_range], ReduceOp::Add);

    let matcher = get_matcher();
    let result = graph_rewrite(&matcher, reduce, &mut ());

    assert_const_value(&result, ConstValue::Int(0));
}

#[test]
fn test_reduce_mul_empty_to_one() {
    // REDUCE(x, [RANGE(-5)], Mul) → Const(1)
    let src = UOp::var("x", DType::Int32, 100);
    let dead_range = UOp::range_const(-5, 0);
    let reduce = UOp::reduce(src, smallvec![dead_range], ReduceOp::Mul);

    let matcher = get_matcher();
    let result = graph_rewrite(&matcher, reduce, &mut ());

    assert_const_value(&result, ConstValue::Int(1));
}

#[test]
fn test_reduce_max_empty_to_min() {
    // REDUCE(x, [RANGE(0)], Max) → Const(INT32_MIN)
    let src = UOp::var("x", DType::Int32, 100);
    let dead_range = UOp::range_const(0, 0);
    let reduce = UOp::reduce(src, smallvec![dead_range], ReduceOp::Max);

    let matcher = get_matcher();
    let result = graph_rewrite(&matcher, reduce, &mut ());

    assert_const_value(&result, ConstValue::Int(i32::MIN as i64));
}

// ============================================================================
// Priority 2: Edge Case Tests
// ============================================================================

// ----------------------------------------------------------------------------
// RANGE Edge Cases
// ----------------------------------------------------------------------------

#[test]
fn test_range_symbolic_dead() {
    // size ∈ [0,5], RANGE(size - 10) → Const(0)
    // vmax(size - 10) = 5 - 10 = -5 ≤ 0, so dead
    let size = UOp::var("size", DType::Int32, 5);
    let ten = UOp::native_const(10i32);
    let count = size.try_sub(&ten).expect("SUB should succeed");
    let range = UOp::range(count, 0);

    let matcher = get_matcher();
    let result = graph_rewrite(&matcher, range, &mut ());

    assert_const_value(&result, ConstValue::Int(0));
}

#[test]
fn test_range_boundary_vmax_zero() {
    // max(-10, 0) = 0, so RANGE has vmax = 0 (boundary)
    // RANGE(max(-10, 0)) → Const(0)
    let neg_ten = UOp::native_const(-10i32);
    let zero = UOp::native_const(0i32);
    let max_val = neg_ten.try_max(&zero).unwrap();
    let range = UOp::range(max_val, 0);

    let matcher = get_matcher();
    let result = graph_rewrite(&matcher, range, &mut ());

    assert_const_value(&result, ConstValue::Int(0));
}

// ----------------------------------------------------------------------------
// END Edge Cases
// ----------------------------------------------------------------------------

#[test]
fn test_end_empty_ranges_unchanged() {
    // END(store, []) should remain unchanged
    let store = UOp::noop();
    let end = UOp::end(Arc::clone(&store), smallvec![]);

    let matcher = get_matcher();
    let result = graph_rewrite(&matcher, end, &mut ());

    // Should remain an END with empty ranges
    match result.op() {
        Op::End { computation, ranges } => {
            assert!(Arc::ptr_eq(computation, &store), "Expected same computation");
            assert_eq!(ranges.len(), 0, "Expected empty ranges");
        }
        other => panic!("Expected END operation, got {:?}", other),
    }
}

#[test]
fn test_end_multiple_dead_ranges_unwrapped() {
    // END(store, [RANGE(0), RANGE(-5)]) → store
    let store = UOp::noop();
    let dead1 = UOp::range_const(0, 0);
    let dead2 = UOp::range_const(-5, 0);
    let end = UOp::end(Arc::clone(&store), smallvec![dead1, dead2]);

    let matcher = get_matcher();
    let result = graph_rewrite(&matcher, end, &mut ());

    // Should unwrap completely
    let unwrapped = assert_end_unwrapped(&result);
    assert!(Arc::ptr_eq(&unwrapped, &store), "Expected END to unwrap to original store");
}

// ----------------------------------------------------------------------------
// REDUCE Edge Cases
// ----------------------------------------------------------------------------

#[test]
fn test_reduce_multiple_dead_ranges() {
    // REDUCE(x, [RANGE(0), RANGE(-5)], Add) → Const(0)
    let src = UOp::var("x", DType::Int32, 100);
    let dead1 = UOp::range_const(0, 0);
    let dead2 = UOp::range_const(-5, 0);
    let reduce = UOp::reduce(src, smallvec![dead1, dead2], ReduceOp::Add);

    let matcher = get_matcher();
    let result = graph_rewrite(&matcher, reduce, &mut ());

    assert_const_value(&result, ConstValue::Int(0));
}
