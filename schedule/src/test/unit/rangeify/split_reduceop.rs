#![allow(clippy::identity_op)]

//! Tests for split_reduceop two-stage reduction optimization.
//!
//! Verifies that large reductions are correctly split into two stages:
//! - Configuration and thresholds
//! - Helper functions (collect_range_ids)
//! - Integration tests for actual splitting behavior

use std::rc::Rc;

use morok_device::DeviceSpec;
use morok_dtype::DType;
use morok_ir::{Op, SInt, UOp};
use smallvec::SmallVec;

use crate::rangeify::split_reduceop::{SplitReduceOpConfig, collect_range_ids, split_reduceop};

// ===== Configuration Tests =====

#[test]
fn test_config_default() {
    let config = SplitReduceOpConfig::default();
    assert_eq!(config.split_threshold, 32768);
    assert_eq!(config.output_size_bits, 22);
    assert_eq!(config.max_divisor, 256);
    assert_eq!(config.min_divisor, 8);
    assert!(config.enabled);
}

#[test]
fn test_config_max_output_size() {
    let config = SplitReduceOpConfig::default();
    assert_eq!(config.max_output_size(), 4_194_304); // 2^22
}

#[test]
fn test_config_custom() {
    let config = SplitReduceOpConfig { split_threshold: 65536, output_size_bits: 20, ..Default::default() };
    assert_eq!(config.split_threshold, 65536);
    assert_eq!(config.output_size_bits, 20);
    assert_eq!(config.max_output_size(), 1_048_576); // 2^20
}

// ===== Helper Function Tests =====

#[test]
fn test_collect_range_ids_empty() {
    let const_val = UOp::native_const(1.0f32);
    let ids = collect_range_ids(&const_val);
    assert_eq!(ids, Vec::<usize>::new());
}

#[test]
fn test_collect_range_ids_single() {
    let range = UOp::range_const(10, 0);
    let ids = collect_range_ids(&range);
    assert_eq!(ids, vec![0]);
}

#[test]
fn test_collect_range_ids_multiple() {
    let r0 = UOp::range_const(10, 0);
    let r1 = UOp::range_const(5, 1);
    let r2 = UOp::range_const(3, 2);
    let add = r0.try_add(&r1).unwrap();
    let mul = add.try_mul(&r2).unwrap();
    let ids = collect_range_ids(&mul);
    assert_eq!(ids, vec![0, 1, 2]);
}

#[test]
fn test_collect_range_ids_unsorted() {
    // Create ranges out of order
    let r2 = UOp::range_const(3, 2);
    let r0 = UOp::range_const(10, 0);
    let r1 = UOp::range_const(5, 1);
    let expr = r2.try_add(&r0).unwrap().try_add(&r1).unwrap();
    let ids = collect_range_ids(&expr);
    assert_eq!(ids, vec![0, 1, 2]); // Should be sorted
}

// ===== Integration Tests =====

/// Helper to create a test tensor with given shape for reduce testing
fn create_test_tensor(shape: &[usize]) -> Rc<UOp> {
    let total_size: usize = shape.iter().product();
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, total_size, DType::Float32);

    // If shape is not 1D, we need to reshape
    if shape.len() == 1 {
        buffer
    } else {
        let shape_sint: SmallVec<[SInt; 4]> = shape.iter().map(|&s| SInt::Const(s)).collect();
        buffer.try_reshape(&shape_sint).unwrap()
    }
}

#[test]
fn test_split_reduceop_disabled() {
    let config = SplitReduceOpConfig { enabled: false, ..Default::default() };

    let tensor = create_test_tensor(&[100000]);
    let reduce = tensor.try_reduce_axis(morok_ir::ReduceOp::Add, vec![0]).unwrap();

    // Should return None when disabled
    assert!(split_reduceop(&reduce, &config).is_none());
}

#[test]
fn test_split_reduceop_below_threshold() {
    let config = SplitReduceOpConfig::default();

    // Small reduction: 1000 elements < threshold (32768)
    let tensor = create_test_tensor(&[1000]);
    let reduce = tensor.try_reduce_axis(morok_ir::ReduceOp::Add, vec![0]).unwrap();

    // Should NOT split (below threshold)
    assert!(split_reduceop(&reduce, &config).is_none());
}

#[test]
fn test_split_reduceop_basic_split() {
    let config = SplitReduceOpConfig::default();

    // Large reduction: 100000 > threshold (32768)
    let tensor = create_test_tensor(&[100000]);
    let reduce = tensor.try_reduce_axis(morok_ir::ReduceOp::Add, vec![0]).unwrap();

    // Should split
    let result = split_reduceop(&reduce, &config);
    assert!(result.is_some(), "Should split large reduction");

    let transformed = result.unwrap();

    // Verify output shape matches original
    let original_shape = reduce.shape().unwrap().unwrap();
    let result_shape = transformed.shape().unwrap().unwrap();
    assert_eq!(result_shape.len(), original_shape.len());
}

#[test]
fn test_split_reduceop_preserves_reduce_op() {
    let config = SplitReduceOpConfig::default();

    for reduce_op in
        [morok_ir::ReduceOp::Add, morok_ir::ReduceOp::Mul, morok_ir::ReduceOp::Max, morok_ir::ReduceOp::Min]
    {
        let tensor = create_test_tensor(&[100000]);
        let reduce = tensor.try_reduce_axis(reduce_op, vec![0]).unwrap();

        let result = split_reduceop(&reduce, &config);
        assert!(result.is_some(), "Should split large reduction for {:?}", reduce_op);

        let transformed = result.unwrap();

        // Verify reduce op appears in the transformation
        let has_reduce_op = transformed.toposort().iter().any(|node| {
            matches!(
                node.op(),
                Op::ReduceAxis { reduce_op: op, .. } if *op == reduce_op
            )
        });
        assert!(has_reduce_op, "Reduce op {:?} should be preserved", reduce_op);
    }
}

#[test]
fn test_split_reduceop_has_contiguous() {
    let config = SplitReduceOpConfig::default();

    let tensor = create_test_tensor(&[100000]);
    let reduce = tensor.try_reduce_axis(morok_ir::ReduceOp::Add, vec![0]).unwrap();

    let result = split_reduceop(&reduce, &config);
    assert!(result.is_some(), "Should split large reduction");

    let transformed = result.unwrap();

    // Verify CONTIGUOUS appears (materializes intermediate)
    let has_contiguous = transformed.toposort().iter().any(|node| matches!(node.op(), Op::Contiguous { .. }));
    assert!(has_contiguous, "Should have CONTIGUOUS for intermediate materialization");
}

#[test]
fn test_split_reduceop_multidim_below_threshold() {
    let config = SplitReduceOpConfig::default();

    // (1000, 1000) tensor, reduce on axis 1
    // Ratio: 1,000,000 / 1000 = 1000 < 32768, won't split
    let tensor = create_test_tensor(&[1000, 1000]);
    let reduce = tensor.try_reduce_axis(morok_ir::ReduceOp::Add, vec![1]).unwrap();

    let result = split_reduceop(&reduce, &config);
    assert!(result.is_none(), "Should NOT split - ratio too low");
}

#[test]
fn test_split_reduceop_multidim_above_threshold() {
    let config = SplitReduceOpConfig::default();

    // (1000, 100000) tensor, reduce on axis 1
    // Ratio: 100,000,000 / 1000 = 100,000 > 32768
    let tensor = create_test_tensor(&[1000, 100000]);
    let reduce = tensor.try_reduce_axis(morok_ir::ReduceOp::Add, vec![1]).unwrap();

    let result = split_reduceop(&reduce, &config);
    assert!(result.is_some(), "Should split large multidim reduction");

    // Verify output shape matches
    let transformed = result.unwrap();
    let original_shape = reduce.shape().unwrap().unwrap();
    let result_shape = transformed.shape().unwrap().unwrap();
    assert_eq!(result_shape.len(), original_shape.len());
}

// ===== Movement Pattern Integration Tests =====

#[test]
fn test_split_with_expand_detects_broadcast() {
    // Test that EXPAND operations are correctly detected as broadcast dimensions
    // and NOT included in split candidates

    let config = SplitReduceOpConfig::default();

    // Create tensor: [100, 1, 1000] expanded to [100, 500, 1000]
    use morok_ir::Op;
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 100 * 1 * 1000, DType::Float32);
    let reshaped =
        buffer.try_reshape(&vec![SInt::Const(100), SInt::Const(1), SInt::Const(1000)].into_iter().collect()).unwrap();

    let expand_shape =
        UOp::vectorize(vec![UOp::index_const(100), UOp::index_const(500), UOp::index_const(1000)].into());
    let expanded = UOp::new(Op::Expand { src: reshaped, new_shape: expand_shape }, DType::Float32);

    // Reduce on axis 1 (the expanded dimension)
    // Total: 100 * 500 * 1000 = 50,000,000
    // After reduction: 100 * 1000 = 100,000
    // Ratio: 50,000,000 / 100,000 = 500 > 32,768 ✓
    let reduce = expanded.try_reduce_axis(morok_ir::ReduceOp::Add, vec![1]).unwrap();

    // Attempt to split
    let result = split_reduceop(&reduce, &config);

    // Should NOT split because axis 1 is expanded (broadcast)
    // The split algorithm should detect that dimension 1 is expanded and skip it
    assert!(result.is_none(), "Should NOT split - axis 1 is broadcast (expanded)");
}

#[test]
fn test_split_with_nested_movement_ops() {
    // Test that nested movement operations are correctly handled

    let config = SplitReduceOpConfig::default();

    // Create: buffer → RESHAPE → EXPAND → RESHAPE → reduce
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 50 * 1, DType::Float32);

    // Reshape to [50, 1]
    let reshaped1 = buffer.try_reshape(&vec![SInt::Const(50), SInt::Const(1)].into_iter().collect()).unwrap();

    // Expand to [50, 1000] (axis 1 broadcast)
    let expand_shape = UOp::vectorize(vec![UOp::index_const(50), UOp::index_const(1000)].into());
    let expanded = UOp::new(morok_ir::Op::Expand { src: reshaped1, new_shape: expand_shape }, DType::Float32);

    // Reshape to [50000] (flatten)
    let reshaped2 = expanded.try_reshape(&vec![SInt::Const(50000)].into_iter().collect()).unwrap();

    // Reduce on axis 0
    // Total: 50,000
    // After reduction: 1
    // Ratio: 50,000 / 1 = 50,000 > 32,768 ✓
    let reduce = reshaped2.try_reduce_axis(morok_ir::ReduceOp::Add, vec![0]).unwrap();

    // Attempt to split
    let result = split_reduceop(&reduce, &config);

    // With movement patterns working correctly, this should be able to split
    // (The patterns should push RESHAPE and EXPAND through INDEX correctly)
    assert!(result.is_some(), "Should split - movement patterns should allow split on nested operations");

    let transformed = result.unwrap();
    // Verify it has the expected structure (CONTIGUOUS in the middle)
    let has_contiguous = transformed.toposort().iter().any(|node| matches!(node.op(), morok_ir::Op::Contiguous { .. }));
    assert!(has_contiguous, "Split result should have CONTIGUOUS");
}

#[test]
fn test_split_skips_expanded_dimensions() {
    // End-to-end test: verify split_reduceop correctly avoids splitting broadcast dimensions

    let config = SplitReduceOpConfig::default();

    // Create tensor: [100, 1, 100000] expanded to [100, 50, 100000], reduce on axis 2
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 100 * 1 * 100000, DType::Float32);
    let reshaped =
        buffer.try_reshape(&vec![SInt::Const(100), SInt::Const(1), SInt::Const(100000)].into_iter().collect()).unwrap();

    let expand_shape =
        UOp::vectorize(vec![UOp::index_const(100), UOp::index_const(50), UOp::index_const(100000)].into());
    let expanded = UOp::new(morok_ir::Op::Expand { src: reshaped, new_shape: expand_shape }, DType::Float32);

    // Reduce on axis 2
    // Total: 100 * 50 * 100000 = 500,000,000
    // After reduction: 100 * 50 = 5,000
    // Ratio: 500,000,000 / 5,000 = 100,000 > 32,768 ✓
    let reduce = expanded.try_reduce_axis(morok_ir::ReduceOp::Add, vec![2]).unwrap();

    let result = split_reduceop(&reduce, &config);

    // Should be able to split on axis 2 (it's NOT expanded)
    // The split should correctly identify that only axis 1 is expanded
    assert!(result.is_some(), "Should split on axis 2 - it's not expanded (only axis 1 is expanded)");

    let transformed = result.unwrap();
    // Verify split happened
    let has_contiguous = transformed.toposort().iter().any(|node| matches!(node.op(), morok_ir::Op::Contiguous { .. }));
    assert!(has_contiguous, "Should have split successfully");
}
