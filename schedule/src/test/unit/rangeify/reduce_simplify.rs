//! Comprehensive tests for reduction simplification optimizations.
//!
//! Tests verify `reduce_unparented` and `reduce_collapse` optimizations:
//! - reduce_unparented: Remove ranges that don't appear in source (2-10x speedup)
//! - reduce_collapse: Lift range-independent computations outside reductions
//! - Helper functions: no_range(), range_size_as_i64()

use std::{f32::consts::PI, sync::Arc};

use morok_dtype::DType;
use morok_ir::{pattern::RewriteResult, AxisId, AxisType, BinaryOp, Op, ReduceOp, UOp};

use crate::rangeify::transforms::reduce_collapse as reduce_collapse_inner;

/// Test helper - thin wrapper around pattern matcher for reduce_unparented tests.
fn reduce_unparented(reduce: &Arc<UOp>) -> Option<Arc<UOp>> {
    match crate::rangeify::patterns::reduction_simplify_patterns().rewrite(reduce, &mut ()) {
        RewriteResult::Rewritten(r) => Some(r),
        _ => None,
    }
}

/// Test helper - wrapper for reduce_collapse that extracts src/ranges from REDUCE node.
fn reduce_collapse(reduce: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Reduce { src, ranges, .. } = reduce.op() else {
        return None;
    };
    reduce_collapse_inner(src, ranges)
}

// ===== Test Helper Functions =====

/// Check if graph contains any REDUCE operations
fn has_reduce_op(uop: &Arc<UOp>) -> bool {
    uop.toposort().iter().any(|n| matches!(n.op(), Op::Reduce { .. } | Op::ReduceAxis { .. }))
}

/// Check if graph contains any RANGE operations
fn has_ranges_in_graph(uop: &Arc<UOp>) -> bool {
    uop.toposort().iter().any(|n| matches!(n.op(), Op::Range { .. }))
}

// ===== reduce_unparented Tests =====

#[test]
fn test_reduce_unparented_add_basic() {
    // Input: REDUCE(CONST(5), [range(10)], ADD)
    // Expected: CONST(5) * 10
    let const_val = UOp::native_const(5i32);
    let range = UOp::range_axis(UOp::index_const(10), AxisId::Renumbered(0), AxisType::Reduce);
    let reduce = UOp::reduce(const_val, vec![range].into(), ReduceOp::Add);

    let result = reduce_unparented(&reduce).expect("Should simplify");

    // Verify result is MUL operation
    assert!(matches!(result.op(), Op::Binary(BinaryOp::Mul, _, _)));
}

#[test]
fn test_reduce_unparented_mul() {
    // Input: REDUCE(CONST(2), [range(3)], MUL)
    // Expected: CONST(2)^3
    let const_val = UOp::native_const(2i32);
    let range = UOp::range_axis(UOp::index_const(3), AxisId::Renumbered(0), AxisType::Reduce);
    let reduce = UOp::reduce(const_val, vec![range].into(), ReduceOp::Mul);

    let result = reduce_unparented(&reduce).expect("Should simplify");

    // Verify result is POW operation
    assert!(matches!(result.op(), Op::Binary(BinaryOp::Pow, _, _)));
}

#[test]
fn test_reduce_unparented_max() {
    // Input: REDUCE(CONST(42), [range(5)], MAX)
    // Expected: CONST(42)
    let const_val = UOp::native_const(42i32);
    let range = UOp::range_axis(UOp::index_const(5), AxisId::Renumbered(0), AxisType::Reduce);
    let reduce = UOp::reduce(const_val.clone(), vec![range].into(), ReduceOp::Max);

    let result = reduce_unparented(&reduce).expect("Should simplify");

    // Result should be the constant value itself
    assert!(Arc::ptr_eq(&result, &const_val));
}

#[test]
fn test_reduce_unparented_min() {
    // Input: REDUCE(CONST(42), [range(5)], MIN)
    // Expected: CONST(42)
    let const_val = UOp::native_const(42i32);
    let range = UOp::range_axis(UOp::index_const(5), AxisId::Renumbered(0), AxisType::Reduce);
    let reduce = UOp::reduce(const_val.clone(), vec![range].into(), ReduceOp::Min);

    let result = reduce_unparented(&reduce).expect("Should simplify");

    // Result should be the constant value itself
    assert!(Arc::ptr_eq(&result, &const_val));
}

#[test]
fn test_reduce_unparented_all_parented() {
    // Input: REDUCE(range, [range], ADD) - can't optimize
    let range = UOp::range_axis(UOp::index_const(10), AxisId::Renumbered(0), AxisType::Reduce);
    let reduce = UOp::reduce(Arc::clone(&range), vec![range].into(), ReduceOp::Add);

    let result = reduce_unparented(&reduce);

    // Should return None because range is parented
    assert!(result.is_none());
}

#[test]
fn test_reduce_unparented_mixed_ranges() {
    // Input: REDUCE(x + range_0, [range_0, range_1], ADD)
    // range_0 is parented, range_1 is unparented
    // Expected: REDUCE(x + range_0, [range_0], ADD) * 10
    let range_0 = UOp::range_axis(UOp::index_const(5), AxisId::Renumbered(0), AxisType::Reduce);
    let range_1 = UOp::range_axis(UOp::index_const(10), AxisId::Renumbered(1), AxisType::Reduce);

    let x = UOp::native_const(3i32);
    let src = x.try_add(&UOp::cast(range_0.clone(), DType::Int32)).unwrap();

    let reduce = UOp::reduce(src, vec![range_0.clone(), range_1].into(), ReduceOp::Add);

    let result = reduce_unparented(&reduce).expect("Should simplify");

    // Result should be: (... * 10)
    // Verify outer op is MUL
    assert!(matches!(result.op(), Op::Binary(BinaryOp::Mul, _, _)));

    // Verify inner REDUCE still has range_0
    if let Op::Binary(_, inner, _) = result.op() {
        if let Op::Reduce { ranges, .. } = inner.op() {
            assert_eq!(ranges.len(), 1);
            assert!(Arc::ptr_eq(&ranges[0], &range_0));
        } else {
            panic!("Expected REDUCE in inner op, got {:?}", inner.op());
        }
    }
}

#[test]
fn test_reduce_unparented_multiple_unparented() {
    // Input: REDUCE(CONST(5), [range(3), range(4)], ADD)
    // Expected: CONST(5) * 3 * 4
    let const_val = UOp::native_const(5i32);
    let range_0 = UOp::range_axis(UOp::index_const(3), AxisId::Renumbered(0), AxisType::Reduce);
    let range_1 = UOp::range_axis(UOp::index_const(4), AxisId::Renumbered(1), AxisType::Reduce);

    let reduce = UOp::reduce(const_val, vec![range_0, range_1].into(), ReduceOp::Add);

    let result = reduce_unparented(&reduce).expect("Should simplify");

    // Result should be nested MUL operations: (5 * 3) * 4
    // Top level should be MUL
    assert!(matches!(result.op(), Op::Binary(BinaryOp::Mul, _, _)));

    // Inner should also be MUL
    if let Op::Binary(_, inner, _) = result.op() {
        assert!(matches!(inner.op(), Op::Binary(BinaryOp::Mul, _, _)));
    }
}

#[test]
fn test_reduce_unparented_non_reduce_returns_none() {
    // Test that non-REDUCE operations return None
    let const_op = UOp::native_const(1.0f32);

    let result = reduce_unparented(&const_op);
    assert!(result.is_none());
}

// ===== reduce_collapse Tests =====

#[test]
fn test_reduce_collapse_basic() {
    // Input: REDUCE(const, [range], ADD) where const doesn't depend on range
    // Expected: After symbolic simplification, range dependency should be eliminated
    // Note: This is a simple case - reduce_unparented would also handle this
    let const_val = UOp::native_const(5i32);
    let range = UOp::range_axis(UOp::index_const(10), AxisId::Renumbered(0), AxisType::Reduce);
    let reduce = UOp::reduce(const_val.clone(), vec![range].into(), ReduceOp::Add);

    let result = reduce_collapse(&reduce).expect("reduce_collapse should succeed on constant");

    // Verify no range dependencies remain
    assert!(!has_ranges_in_graph(&result), "Result should have no range dependencies");

    // Verify REDUCE operation was eliminated
    assert!(!has_reduce_op(&result), "Result should not contain REDUCE operations");

    // Verify dtype preserved
    assert_eq!(result.dtype(), const_val.dtype(), "Should preserve dtype");
}

#[test]
fn test_reduce_collapse_with_range_dependency() {
    // Input: REDUCE(range + 1, [range], ADD)
    // This creates a true dependency on the range variable
    // Expected: reduce_collapse may succeed (substitution works), but won't eliminate the REDUCE
    let range = UOp::range_axis(UOp::index_const(10), AxisId::Renumbered(0), AxisType::Reduce);
    let one = UOp::native_const(1i32);
    let range_int = UOp::cast(range.clone(), DType::Int32);
    let src = range_int.try_add(&one).unwrap();

    let reduce = UOp::reduce(src, vec![range].into(), ReduceOp::Add);

    let result = reduce_collapse(&reduce);

    // With the fixed logic, this should now return None since the var dependency remains
    assert!(result.is_none(), "reduce_collapse should return None when range dependency can't be eliminated");
}

#[test]
fn test_reduce_collapse_non_reduce_returns_none() {
    // Test that non-REDUCE operations return None
    let const_op = UOp::native_const(1.0f32);

    let result = reduce_collapse(&const_op);
    assert!(result.is_none());
}

#[test]
fn test_reduce_collapse_empty_ranges() {
    // REDUCE with no ranges should return None
    let const_val = UOp::native_const(5i32);
    let reduce = UOp::reduce(const_val, vec![].into(), ReduceOp::Add);

    let result = reduce_collapse(&reduce);
    assert!(result.is_none(), "reduce_collapse should return None for empty ranges");
}

#[test]
fn test_reduce_collapse_multiple_ranges_all_independent() {
    // REDUCE(const, [range1, range2], ADD) where const doesn't depend on either range
    let const_val = UOp::native_const(5i32);
    let range1 = UOp::range_axis(UOp::index_const(10), AxisId::Renumbered(0), AxisType::Reduce);
    let range2 = UOp::range_axis(UOp::index_const(20), AxisId::Renumbered(1), AxisType::Reduce);

    let reduce = UOp::reduce(const_val.clone(), vec![range1, range2].into(), ReduceOp::Add);

    let result = reduce_collapse(&reduce);

    // Should successfully collapse since const has no range dependency
    assert!(result.is_some(), "reduce_collapse should succeed with multiple independent ranges");

    if let Some(res) = result {
        // Result should have no range dependencies
        assert!(crate::rangeify::indexing::no_range(&res), "Result should have no range dependencies");
    }
}

#[test]
fn test_reduce_collapse_algebraic_simplification() {
    // Test that reduce_collapse works with algebraic patterns
    // REDUCE(x + 0, [range], ADD) where x is constant
    let x = UOp::native_const(42i32);
    let zero = UOp::native_const(0i32);
    let x_plus_0 = x.try_add(&zero).unwrap();

    let range = UOp::range_axis(UOp::index_const(10), AxisId::Renumbered(0), AxisType::Reduce);

    let reduce = UOp::reduce(x_plus_0, vec![range].into(), ReduceOp::Add);

    let result = reduce_collapse(&reduce).expect("reduce_collapse should succeed after x+0 simplification");

    // Verify symbolic simplification eliminated both x+0 AND range dependency
    assert!(!has_ranges_in_graph(&result), "x+0 simplification should eliminate ranges");
    assert!(!has_reduce_op(&result), "Result should not contain REDUCE");

    // Verify result is simplified (no ADD operation for x+0)
    let has_add = result.toposort().iter().any(|n| matches!(n.op(), Op::Binary(BinaryOp::Add, _, _)));
    assert!(!has_add, "x+0 should be simplified away");
}

#[test]
fn test_reduce_collapse_multiplication_by_one() {
    // REDUCE(x * 1, [range], MUL) where x is constant
    let x = UOp::native_const(PI);
    let one = UOp::native_const(1.0f32);
    let x_times_1 = x.try_mul(&one).unwrap();

    let range = UOp::range_axis(UOp::index_const(5), AxisId::Renumbered(0), AxisType::Reduce);

    let reduce = UOp::reduce(x_times_1, vec![range].into(), ReduceOp::Mul);

    let result = reduce_collapse(&reduce).expect("reduce_collapse should succeed after x*1 simplification");

    // Verify symbolic simplification eliminated both x*1 AND range dependency
    assert!(!has_ranges_in_graph(&result), "x*1 simplification should eliminate ranges");
    assert!(!has_reduce_op(&result), "Result should not contain REDUCE");

    // Verify result is simplified (no MUL operation for x*1)
    let has_mul = result.toposort().iter().any(|n| matches!(n.op(), Op::Binary(BinaryOp::Mul, _, _)));
    assert!(!has_mul, "x*1 should be simplified away");
}

#[test]
fn test_reduce_collapse_preserves_dtype() {
    // Verify that reduce_collapse preserves data types correctly
    let const_val = UOp::native_const(2.5f64);
    let range = UOp::range_axis(UOp::index_const(100), AxisId::Renumbered(0), AxisType::Reduce);
    let reduce = UOp::reduce(const_val.clone(), vec![range].into(), ReduceOp::Add);

    let result = reduce_collapse(&reduce);

    assert!(result.is_some(), "reduce_collapse should succeed");

    if let Some(res) = result {
        // The result dtype should match the source (Float64 in this case)
        assert_eq!(res.dtype(), const_val.dtype(), "reduce_collapse should preserve dtype");
    }
}

#[test]
fn test_reduce_collapse_different_reduce_ops() {
    // Test reduce_collapse with different ReduceOp types
    let const_val = UOp::native_const(10i32);
    let range = UOp::range_axis(UOp::index_const(5), AxisId::Renumbered(0), AxisType::Reduce);

    // Test ADD
    let reduce_add = UOp::reduce(const_val.clone(), vec![range.clone()].into(), ReduceOp::Add);
    assert!(reduce_collapse(&reduce_add).is_some(), "reduce_collapse should work with ReduceOp::Add");

    // Test MUL
    let reduce_mul = UOp::reduce(const_val.clone(), vec![range.clone()].into(), ReduceOp::Mul);
    assert!(reduce_collapse(&reduce_mul).is_some(), "reduce_collapse should work with ReduceOp::Mul");

    // Test MAX
    let reduce_max = UOp::reduce(const_val.clone(), vec![range.clone()].into(), ReduceOp::Max);
    assert!(reduce_collapse(&reduce_max).is_some(), "reduce_collapse should work with ReduceOp::Max");

    // Test MIN
    let reduce_min = UOp::reduce(const_val, vec![range].into(), ReduceOp::Min);
    assert!(reduce_collapse(&reduce_min).is_some(), "reduce_collapse should work with ReduceOp::Min");
}

// ===== Helper Function Tests =====

#[test]
fn test_no_range_with_ranges() {
    // UOp with RANGE dependencies should return false
    let range = UOp::range_axis(UOp::index_const(10), AxisId::Renumbered(0), AxisType::Reduce);
    let const_5 = UOp::native_const(5i32);

    // Create expression that depends on range: range + 5
    let sum = UOp::cast(range.clone(), DType::Int32).try_add(&const_5).unwrap();

    // Should return false because sum depends on range
    assert!(!crate::rangeify::indexing::no_range(&sum));
}

#[test]
fn test_no_range_without_ranges() {
    // UOp without RANGE dependencies should return true
    let const_val = UOp::native_const(42i32);
    assert!(crate::rangeify::indexing::no_range(&const_val));

    // Arithmetic operations on constants also have no ranges
    let a = UOp::native_const(10i32);
    let b = UOp::native_const(20i32);
    let sum = a.try_add(&b).unwrap();
    assert!(crate::rangeify::indexing::no_range(&sum));
}

#[test]
fn test_range_size_extraction_constant() {
    // Extract size from constant RANGE
    let range = UOp::range_axis(UOp::index_const(100), AxisId::Renumbered(0), AxisType::Loop);

    assert_eq!(crate::rangeify::indexing::range_size_as_i64(&range), Some(100));

    // Test with different constant values
    let range_42 = UOp::range_axis(UOp::index_const(42), AxisId::Renumbered(1), AxisType::Reduce);

    assert_eq!(crate::rangeify::indexing::range_size_as_i64(&range_42), Some(42));
}

#[test]
fn test_range_size_extraction_symbolic() {
    // Symbolic RANGE should return None
    let symbolic_var = UOp::define_var("N".to_string(), 0, 1000);
    let range = UOp::range_axis(symbolic_var, AxisId::Renumbered(0), AxisType::Loop);

    assert_eq!(crate::rangeify::indexing::range_size_as_i64(&range), None);
}

#[test]
fn test_range_size_extraction_non_range() {
    // Non-RANGE UOp should return None
    let const_op = UOp::native_const(100i32);
    assert_eq!(crate::rangeify::indexing::range_size_as_i64(&const_op), None);

    // Binary operation also returns None
    let a = UOp::native_const(10i32);
    let b = UOp::native_const(20i32);
    let sum = a.try_add(&b).unwrap();
    assert_eq!(crate::rangeify::indexing::range_size_as_i64(&sum), None);
}
