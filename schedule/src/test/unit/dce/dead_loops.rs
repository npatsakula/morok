//! Tests for dead loop detection using range analysis.

use morok_dtype::DType;
use morok_ir::types::{BinaryOp, ConstValue};
use morok_ir::{Op, UOp};

use crate::rewrite::graph_rewrite;
use crate::symbolic::symbolic_simple;

#[test]
fn test_range_zero_iterations() {
    // RANGE(0) has no iterations
    let zero = UOp::const_(DType::Int32, ConstValue::Int(0));
    let range = UOp::range(zero, 0);

    // Currently, we only detect dead ranges but don't eliminate them directly
    // This test documents current behavior
    let matcher = symbolic_simple();
    let _result = graph_rewrite(&matcher, range.clone());

    // For now, RANGE is not eliminated (needs more infrastructure)
    // But is_empty_range() should detect it
    use crate::symbolic::dce::is_empty_range;
    assert!(is_empty_range(&range));
}

#[test]
fn test_range_negative_iterations() {
    // RANGE with negative end should be dead
    let neg_five = UOp::const_(DType::Int32, ConstValue::Int(-5));
    let range = UOp::range(neg_five, 0);

    use crate::symbolic::dce::is_empty_range;
    assert!(is_empty_range(&range));
}

#[test]
fn test_range_symbolic_dead() {
    // size in [0, 5], count = size - 10
    // count is always negative, so RANGE(count) is dead
    let size = UOp::var("size", DType::Int32, 0, 5);
    let ten = UOp::const_(DType::Int32, ConstValue::Int(10));
    let count = UOp::new(Op::Binary(BinaryOp::Sub, size, ten), DType::Int32); // 0..5 - 10 = -10..-5

    let range = UOp::range(count, 0);

    use crate::symbolic::dce::is_empty_range;
    assert!(is_empty_range(&range));
}

#[test]
fn test_range_positive_not_dead() {
    // RANGE(10) is not dead
    let range = UOp::range_const(10, 0);

    use crate::symbolic::dce::is_empty_range;
    assert!(!is_empty_range(&range));
}

#[test]
fn test_range_unknown_not_dead() {
    // size in [0, 100] - could be positive or zero
    let size = UOp::var("size", DType::Int32, 0, 100);
    let range = UOp::range(size, 0);

    use crate::symbolic::dce::is_empty_range;
    assert!(!is_empty_range(&range));
}

#[test]
fn test_max_with_zero_dead_range() {
    // max(0, negative) = 0, so RANGE is dead
    let neg_val = UOp::const_(DType::Int32, ConstValue::Int(-10));
    let zero = UOp::const_(DType::Int32, ConstValue::Int(0));
    let count = UOp::new(Op::Binary(BinaryOp::Max, neg_val, zero.clone()), DType::Int32); // max(-10, 0) = 0

    let range = UOp::range(count, 0);

    use crate::symbolic::dce::is_empty_range;
    assert!(is_empty_range(&range));
}

// Note: Full dead loop elimination would require removing dependent computations
// This is not yet implemented but the infrastructure is in place
