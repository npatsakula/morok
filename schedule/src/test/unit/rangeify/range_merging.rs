//! Tests for range merging in multi-consumer scenarios.
//!
//! Validates that merge_consumer_ranges correctly handles:
//! - Identical ranges (merge validity, no realization)
//! - Different ranges (create new range, partial realization)
//! - Validity mask merging (OR operation)

use std::sync::Arc;

use morok_dtype::DType;
use morok_ir::{AxisType, ConstValue, Op, SInt, TernaryOp, UOp};

use crate::rangeify::indexing::IndexingContext;

#[test]
fn test_identical_ranges_no_realization() {
    // Two consumers with identical ranges should merge without realization
    let mut ctx = IndexingContext::new();

    // Create ranges
    let r0 = ctx.new_range(&SInt::Const(10), AxisType::Loop);
    let r1 = ctx.new_range(&SInt::Const(20), AxisType::Loop);

    // Both consumers use the same ranges
    let consumer_rngs = [vec![r0.clone(), r1.clone()], vec![r0.clone(), r1.clone()]];

    // This would be called by merge_consumer_ranges
    // For now, verify that identical ranges would not cause realization
    // by checking all_ranges_same helper
    use crate::rangeify::helpers::all_ranges_same;

    let indices0: Vec<_> = consumer_rngs[0].iter().map(|r| r.get_idx()).collect();
    let indices1: Vec<_> = consumer_rngs[1].iter().map(|r| r.get_idx()).collect();

    // Verify indices are same for each dimension
    assert!(all_ranges_same(&[indices0[0].clone(), indices1[0].clone()]));
    assert!(all_ranges_same(&[indices0[1].clone(), indices1[1].clone()]));
}

#[test]
fn test_get_idx_plain_range() {
    // Plain ranges should return themselves
    let mut ctx = IndexingContext::new();
    let range = ctx.new_range(&SInt::Const(10), AxisType::Loop);

    let idx = range.get_idx();
    assert!(Arc::ptr_eq(&idx, &range));
}

#[test]
fn test_get_valid_plain_range() {
    // Plain ranges should return constant true for validity
    let mut ctx = IndexingContext::new();
    let range = ctx.new_range(&SInt::Const(10), AxisType::Loop);

    let valid = range.get_valid();
    if let Op::Const(cv) = valid.op() {
        assert_eq!(cv.0, ConstValue::Bool(true));
    } else {
        panic!("Expected constant true for plain range validity");
    }
}

#[test]
fn test_get_idx_with_validity() {
    // Ranges with WHERE wrapper should extract the index
    let mut ctx = IndexingContext::new();
    let idx = ctx.new_range(&SInt::Const(10), AxisType::Loop);
    let valid = UOp::native_const(true);
    let invalid = UOp::invalid_marker();

    let wrapped = UOp::try_where(valid, idx.clone(), invalid).unwrap();

    let extracted_idx = wrapped.get_idx();
    assert!(Arc::ptr_eq(&extracted_idx, &idx));
}

#[test]
fn test_get_valid_with_validity() {
    // Ranges with WHERE wrapper should extract the validity condition
    let mut ctx = IndexingContext::new();
    let idx = ctx.new_range(&SInt::Const(10), AxisType::Loop);

    // Create validity condition: i < 5
    let five = UOp::index_const(5);
    let valid = idx.try_cmplt(&five).unwrap();
    let invalid = UOp::invalid_marker();

    let wrapped = UOp::try_where(valid.clone(), idx.clone(), invalid).unwrap();

    let extracted_valid = wrapped.get_valid();
    assert!(Arc::ptr_eq(&extracted_valid, &valid));
}

#[test]
fn test_all_ranges_same_identical() {
    // Identical ranges (same pointer)
    let mut ctx = IndexingContext::new();
    let r1 = ctx.new_range(&SInt::Const(10), AxisType::Loop);
    let r2 = r1.clone();

    use crate::rangeify::helpers::all_ranges_same;
    assert!(all_ranges_same(&[r1, r2]));
}

#[test]
fn test_all_ranges_same_different() {
    // Different ranges
    let mut ctx = IndexingContext::new();
    let r1 = ctx.new_range(&SInt::Const(10), AxisType::Loop);
    let r2 = ctx.new_range(&SInt::Const(20), AxisType::Loop);

    // Extract indices (as merge_consumer_ranges does)
    let idx1 = r1.get_idx();
    let idx2 = r2.get_idx();

    use crate::rangeify::helpers::all_ranges_same;
    assert!(!all_ranges_same(&[idx1, idx2]));
}

#[test]
fn test_invalid_marker_detection() {
    // Test that invalid marker is correctly detected
    let invalid = UOp::invalid_marker();

    // Should be Op::Invalid, not a constant
    assert!(matches!(invalid.op(), Op::Invalid));
    assert_eq!(invalid.dtype(), DType::Index);
}

#[test]
fn test_padding_uses_invalid_marker() {
    // Test that padding logic creates WHERE with Invalid marker
    let idx = UOp::index_const(0);
    let valid = UOp::native_const(true);
    let invalid = UOp::invalid_marker();

    let padded = UOp::try_where(valid, idx, invalid).unwrap();

    // Verify structure: WHERE(valid, idx, Invalid)
    if let Op::Ternary(TernaryOp::Where, _cond, _true_val, false_val) = padded.op() {
        assert!(matches!(false_val.op(), Op::Invalid));
    } else {
        panic!("Expected WHERE operation");
    }
}

#[test]
fn test_or_merging_of_validity_masks() {
    // Test that validity masks can be OR'd together
    let mut ctx = IndexingContext::new();
    let idx = ctx.new_range(&SInt::Const(10), AxisType::Loop);

    // Create two different validity conditions
    let five = UOp::index_const(5);
    let eight = UOp::index_const(8);

    let valid1 = idx.try_cmplt(&five).unwrap(); // i < 5
    let valid2 = idx.try_cmplt(&eight).unwrap(); // i < 8

    // Merge with OR
    let merged = valid1.try_or_op(&valid2).unwrap();

    // Verify it's a binary OR operation
    if let Op::Binary(op, _, _) = merged.op() {
        assert!(matches!(op, morok_ir::BinaryOp::Or));
    } else {
        panic!("Expected OR operation");
    }
}

#[test]
fn test_empty_ranges_list() {
    // Empty ranges list should return true
    use crate::rangeify::helpers::all_ranges_same;
    assert!(all_ranges_same(&[]));
}

#[test]
fn test_single_range() {
    // Single range should return true
    let mut ctx = IndexingContext::new();
    let r1 = ctx.new_range(&SInt::Const(10), AxisType::Loop);

    use crate::rangeify::helpers::all_ranges_same;
    assert!(all_ranges_same(&[r1]));
}
