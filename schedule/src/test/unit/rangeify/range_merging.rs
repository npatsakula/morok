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

// ===== Direct merge_consumer_ranges Tests =====

/// Helper to create a BUFFER with shape (size,)
fn create_buffer_with_size(size: usize) -> Arc<UOp> {
    // Op::Buffer has shape (size,) - see shape.rs line 591
    UOp::new_buffer(morok_dtype::DeviceSpec::Cpu, size, DType::Float32)
}

/// Helper to create a 2D shaped UOp via RESHAPE
fn create_reshaped_2d(sizes: &[usize]) -> Arc<UOp> {
    let src = create_buffer_with_size(sizes.iter().product());
    let new_shape = UOp::vectorize(sizes.iter().map(|&s| UOp::index_const(s as i64)).collect());
    UOp::new(Op::Reshape { src, new_shape }, DType::Float32)
}

#[test]
fn test_merge_consumer_ranges_identical_1d() {
    // Two consumers with identical ranges should merge without realization
    use crate::rangeify::merge_consumer_ranges;

    let mut ctx = IndexingContext::new();

    // Create a BUFFER with shape (100,)
    let buffer = create_buffer_with_size(100);

    // Create identical ranges for both consumers
    let r0 = ctx.new_range(&SInt::Const(100), AxisType::Loop);

    let consumer_rngs = vec![vec![r0.clone()], vec![r0.clone()]];

    // Merge ranges
    let merged = merge_consumer_ranges(&buffer, &consumer_rngs, &mut ctx).unwrap();

    // Should return 1 range (one per dimension)
    assert_eq!(merged.len(), 1, "Should have 1 merged range");

    // Merged range should be identical to input (no realization needed)
    assert!(Arc::ptr_eq(&merged[0], &r0), "Range should be unchanged");

    // Should NOT mark for realization
    assert!(
        !ctx.realize_map.contains_key(&morok_ir::UOpKey(buffer.clone())),
        "Identical ranges should NOT require realization"
    );
}

#[test]
fn test_merge_consumer_ranges_different_1d() {
    // Consumers with different ranges should trigger realization
    use crate::rangeify::merge_consumer_ranges;

    let mut ctx = IndexingContext::new();

    // Create a BUFFER with shape (100,)
    let buffer = create_buffer_with_size(100);

    // Create different ranges for consumers (same size, different IDs)
    let r0_a = ctx.new_range(&SInt::Const(100), AxisType::Loop);
    let r0_b = ctx.new_range(&SInt::Const(100), AxisType::Loop);

    let consumer_rngs = vec![vec![r0_a.clone()], vec![r0_b.clone()]];

    // Merge ranges
    let merged = merge_consumer_ranges(&buffer, &consumer_rngs, &mut ctx).unwrap();

    // Should return 1 range
    assert_eq!(merged.len(), 1, "Should have 1 merged range");

    // Merged range should be NEW (not the original ones)
    assert!(!Arc::ptr_eq(&merged[0], &r0_a), "Different ranges should create new range");
    assert!(!Arc::ptr_eq(&merged[0], &r0_b), "Different ranges should create new range");

    // Should mark for realization (because ranges differ)
    let realize_info = ctx.realize_map.get(&morok_ir::UOpKey(buffer.clone()));
    assert!(realize_info.is_some(), "Different ranges should require realization");
}

#[test]
fn test_merge_consumer_ranges_2d_partial_overlap() {
    // One dimension same, one different
    use crate::rangeify::merge_consumer_ranges;

    let mut ctx = IndexingContext::new();

    // Create a 2D shaped UOp (10, 20)
    let reshaped = create_reshaped_2d(&[10, 20]);

    // First dimension: same range, second dimension: different ranges
    let r0 = ctx.new_range(&SInt::Const(10), AxisType::Loop);
    let r1_a = ctx.new_range(&SInt::Const(20), AxisType::Loop);
    let r1_b = ctx.new_range(&SInt::Const(20), AxisType::Loop);

    let consumer_rngs = vec![vec![r0.clone(), r1_a.clone()], vec![r0.clone(), r1_b.clone()]];

    // Merge ranges
    let merged = merge_consumer_ranges(&reshaped, &consumer_rngs, &mut ctx).unwrap();

    // Should return 2 ranges
    assert_eq!(merged.len(), 2, "Should have 2 merged ranges");

    // First range should be unchanged (identical)
    assert!(Arc::ptr_eq(&merged[0], &r0), "Identical first dimension should be unchanged");

    // Second range should be NEW (different)
    assert!(!Arc::ptr_eq(&merged[1], &r1_a), "Different second dimension should create new range");

    // Should mark only dimension 1 for realization
    let realize_info = ctx.realize_map.get(&morok_ir::UOpKey(reshaped.clone()));
    assert!(realize_info.is_some(), "Should mark for realization");
    if let Some(Some(axes)) = realize_info {
        assert_eq!(axes, &[1], "Only dimension 1 should need realization");
    }
}

#[test]
fn test_merge_consumer_ranges_with_validity() {
    // Ranges with validity masks should OR the masks
    use crate::rangeify::merge_consumer_ranges;

    let mut ctx = IndexingContext::new();

    // Create a BUFFER with shape (10,)
    let buffer = create_buffer_with_size(10);

    // Create same index but different validity masks
    let idx = ctx.new_range(&SInt::Const(10), AxisType::Loop);

    // Consumer 1: valid when i < 5
    let five = UOp::index_const(5);
    let valid1 = idx.try_cmplt(&five).unwrap();
    let invalid = UOp::invalid_marker();
    let r0_a = UOp::try_where(valid1.clone(), idx.clone(), invalid.clone()).unwrap();

    // Consumer 2: valid when i < 8
    let eight = UOp::index_const(8);
    let valid2 = idx.try_cmplt(&eight).unwrap();
    let r0_b = UOp::try_where(valid2.clone(), idx.clone(), invalid).unwrap();

    let consumer_rngs = vec![vec![r0_a.clone()], vec![r0_b.clone()]];

    // Merge ranges
    let merged = merge_consumer_ranges(&buffer, &consumer_rngs, &mut ctx).unwrap();

    // Should return 1 range
    assert_eq!(merged.len(), 1, "Should have 1 merged range");

    // Merged range should have WHERE structure with OR'd validity
    if let Op::Ternary(TernaryOp::Where, merged_valid, merged_idx, _) = merged[0].op() {
        // Merged index should be the original idx
        assert!(Arc::ptr_eq(merged_idx, &idx), "Merged index should be unchanged");

        // Merged validity should be OR of both conditions
        if let Op::Binary(op, _, _) = merged_valid.op() {
            assert!(matches!(op, morok_ir::BinaryOp::Or), "Validity should be OR'd");
        } else {
            panic!("Expected OR operation in merged validity, got {:?}", merged_valid.op());
        }
    } else {
        panic!("Expected WHERE operation in merged range, got {:?}", merged[0].op());
    }
}

#[test]
fn test_merge_consumer_ranges_empty() {
    // No consumers - should handle gracefully
    use crate::rangeify::merge_consumer_ranges;

    let mut ctx = IndexingContext::new();

    // Create a BUFFER with shape (10,)
    let buffer = create_buffer_with_size(10);

    let consumer_rngs: Vec<Vec<Arc<UOp>>> = vec![];

    // Merge ranges
    let merged = merge_consumer_ranges(&buffer, &consumer_rngs, &mut ctx).unwrap();

    // Should create new ranges for all dimensions
    assert_eq!(merged.len(), 1, "Should create 1 range for 1-dim buffer");

    // Should mark for realization (no consumer ranges means new ranges needed)
    let realize_info = ctx.realize_map.get(&morok_ir::UOpKey(buffer.clone()));
    assert!(realize_info.is_some(), "Should mark for realization");
}
