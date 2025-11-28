use crate::reduce::AxisSpec;
use crate::*;
use morok_dtype::DType;

#[test]
fn test_axis_spec_all() {
    let axes = Tensor::resolve_axis_spec(&AxisSpec::All, 3).unwrap();
    assert_eq!(axes, vec![0, 1, 2]);
}

#[test]
fn test_axis_spec_single() {
    // Positive index
    let axes = Tensor::resolve_axis_spec(&AxisSpec::Single(1), 3).unwrap();
    assert_eq!(axes, vec![1]);

    // Negative index
    let axes = Tensor::resolve_axis_spec(&AxisSpec::Single(-1), 3).unwrap();
    assert_eq!(axes, vec![2]);

    // Out of bounds
    assert!(Tensor::resolve_axis_spec(&AxisSpec::Single(5), 3).is_err());
    assert!(Tensor::resolve_axis_spec(&AxisSpec::Single(-5), 3).is_err());
}

#[test]
fn test_axis_spec_multiple() {
    // Multiple axes
    let axes = Tensor::resolve_axis_spec(&AxisSpec::Multiple(vec![0, 2]), 3).unwrap();
    assert_eq!(axes, vec![0, 2]);

    // With negatives
    let axes = Tensor::resolve_axis_spec(&AxisSpec::Multiple(vec![0, -1]), 3).unwrap();
    assert_eq!(axes, vec![0, 2]);

    // Deduplication
    let axes = Tensor::resolve_axis_spec(&AxisSpec::Multiple(vec![1, 1, 1]), 3).unwrap();
    assert_eq!(axes, vec![1]);
}

#[test]
fn test_sum_acc_dtype() {
    assert_eq!(Tensor::sum_acc_dtype(&DType::Bool), DType::Int32);
    assert_eq!(Tensor::sum_acc_dtype(&DType::Int8), DType::Int32);
    assert_eq!(Tensor::sum_acc_dtype(&DType::Int16), DType::Int32);
    assert_eq!(Tensor::sum_acc_dtype(&DType::Int32), DType::Int32);
    assert_eq!(Tensor::sum_acc_dtype(&DType::Int64), DType::Int64);

    assert_eq!(Tensor::sum_acc_dtype(&DType::UInt8), DType::UInt32);
    assert_eq!(Tensor::sum_acc_dtype(&DType::UInt16), DType::UInt32);
    assert_eq!(Tensor::sum_acc_dtype(&DType::UInt32), DType::UInt32);
    assert_eq!(Tensor::sum_acc_dtype(&DType::UInt64), DType::UInt64);

    assert_eq!(Tensor::sum_acc_dtype(&DType::Float16), DType::Float32);
    assert_eq!(Tensor::sum_acc_dtype(&DType::BFloat16), DType::Float32);
    assert_eq!(Tensor::sum_acc_dtype(&DType::Float32), DType::Float32);
    assert_eq!(Tensor::sum_acc_dtype(&DType::Float64), DType::Float64);
}

// ========== Argmax Tests ==========

#[test]
fn test_argmax_1d_basic() {
    let t = Tensor::from_slice([1.0f32, 3.0, 2.0, 5.0, 4.0]);
    let result = t.argmax(Some(0)).unwrap();
    // Max value 5.0 is at index 3
    let result_shape = result.shape().unwrap();
    assert_eq!(result_shape.len(), 0); // Scalar result
    assert_eq!(result.uop.dtype(), DType::Int32);
}

#[test]
fn test_argmax_1d_ties_first_occurrence() {
    let t = Tensor::from_slice([1.0f32, 5.0, 3.0, 5.0, 2.0]);
    let result = t.argmax(Some(0)).unwrap();
    // Two maxima at indices 1 and 3, should return 1 (first)
    assert_eq!(result.uop.dtype(), DType::Int32);
}

#[test]
fn test_argmax_2d_axis0() {
    let t = Tensor::from_slice([1.0f32, 3.0, 2.0, 4.0, 2.0, 5.0]).try_reshape(&[2, 3]).unwrap();
    // Shape [2, 3]: [[1.0, 3.0, 2.0], [4.0, 2.0, 5.0]]
    let result = t.argmax(Some(0)).unwrap();
    // Expected: [1, 0, 1] (max per column)
    let result_shape = result.shape().unwrap();
    assert_eq!(result_shape.len(), 1);
    assert_eq!(result_shape[0].as_const().unwrap(), 3);
}

#[test]
fn test_argmax_2d_axis1() {
    let t = Tensor::from_slice([1.0f32, 3.0, 2.0, 4.0, 2.0, 5.0]).try_reshape(&[2, 3]).unwrap();
    let result = t.argmax(Some(1)).unwrap();
    // Expected: [1, 2] (max per row)
    let result_shape = result.shape().unwrap();
    assert_eq!(result_shape.len(), 1);
    assert_eq!(result_shape[0].as_const().unwrap(), 2);
}

#[test]
fn test_argmax_flatten() {
    let t = Tensor::from_slice([1.0f32, 3.0, 2.0, 4.0, 2.0, 5.0]).try_reshape(&[2, 3]).unwrap();
    let result = t.argmax(None).unwrap();
    // Flattened: [1, 3, 2, 4, 2, 5], max 5.0 at index 5
    assert_eq!(result.shape().unwrap().len(), 0); // Scalar
}

#[test]
fn test_argmax_keepdim() {
    let t = Tensor::from_slice([1.0f32, 3.0, 2.0, 4.0, 2.0, 5.0]).try_reshape(&[2, 3]).unwrap();
    let result = t.argmax_with().axis(Some(1)).keepdim(true).call().unwrap();
    // Shape should be [2, 1] instead of [2]
    let result_shape = result.shape().unwrap();
    assert_eq!(result_shape.len(), 2);
    assert_eq!(result_shape[0].as_const().unwrap(), 2);
    assert_eq!(result_shape[1].as_const().unwrap(), 1);
}

#[test]
fn test_argmax_negative_axis() {
    let t = Tensor::from_slice([1.0f32, 3.0, 2.0, 4.0, 2.0, 5.0]).try_reshape(&[2, 3]).unwrap();
    let result = t.argmax(Some(-1)).unwrap();
    // -1 should resolve to axis 1
    let result_shape = result.shape().unwrap();
    assert_eq!(result_shape.len(), 1);
    assert_eq!(result_shape[0].as_const().unwrap(), 2);
}

// ========== Argmin Tests ==========

#[test]
fn test_argmin_1d_basic() {
    let t = Tensor::from_slice([5.0f32, 3.0, 1.0, 4.0, 2.0]);
    let result = t.argmin(Some(0)).unwrap();
    // Min value 1.0 is at index 2
    assert_eq!(result.uop.dtype(), DType::Int32);
}

#[test]
fn test_argmin_float() {
    let t = Tensor::from_slice([1.5f32, -2.3, 0.5, 1.0]);
    let result = t.argmin(Some(0)).unwrap();
    // Min is -2.3 at index 1
    assert_eq!(result.uop.dtype(), DType::Int32);
}

#[test]
fn test_argmin_int() {
    let t = Tensor::from_slice([5i32, 3, 1, 4, 2]);
    let result = t.argmin(Some(0)).unwrap();
    // Min is 1 at index 2
    assert_eq!(result.uop.dtype(), DType::Int32);
}

#[test]
fn test_argmin_bool() {
    let t = Tensor::from_slice([true, false, true]);
    let result = t.argmin(Some(0)).unwrap();
    // Min (false) at index 1
    assert_eq!(result.uop.dtype(), DType::Int32);
}

// ========== Any Tests ==========

#[test]
fn test_any_all_true() {
    let t = Tensor::from_slice([true, true, true]);
    let result = t.any(()).unwrap();
    // Should be true
    assert_eq!(result.uop.dtype(), DType::Bool);
}

#[test]
fn test_any_all_false() {
    let t = Tensor::from_slice([false, false, false]);
    let result = t.any(()).unwrap();
    // Should be false
    assert_eq!(result.uop.dtype(), DType::Bool);
}

#[test]
fn test_any_mixed() {
    let t = Tensor::from_slice([false, true, false]);
    let result = t.any(()).unwrap();
    // Should be true
    assert_eq!(result.uop.dtype(), DType::Bool);
}

#[test]
fn test_any_numeric() {
    let t = Tensor::from_slice([0.0f32, 1.0, 0.0]);
    let result = t.any(()).unwrap();
    // Non-zero treated as true, should be true
    assert_eq!(result.uop.dtype(), DType::Bool);
}

#[test]
fn test_any_2d_axis0() {
    let t = Tensor::from_slice([true, false, false, false]).try_reshape(&[2, 2]).unwrap();
    let result = t.any(0).unwrap();
    // Expected: [true, false]
    let result_shape = result.shape().unwrap();
    assert_eq!(result_shape.len(), 1);
    assert_eq!(result_shape[0].as_const().unwrap(), 2);
}

#[test]
fn test_any_2d_axis1() {
    let t = Tensor::from_slice([true, false, false, false]).try_reshape(&[2, 2]).unwrap();
    let result = t.any(1).unwrap();
    // Expected: [true, false]
    let result_shape = result.shape().unwrap();
    assert_eq!(result_shape.len(), 1);
    assert_eq!(result_shape[0].as_const().unwrap(), 2);
}

#[test]
fn test_any_keepdim() {
    let t = Tensor::from_slice([true, false, false, true]).try_reshape(&[2, 2]).unwrap();
    let result = t.any_with().axes(0).keepdim(true).call().unwrap();
    // Shape should be [1, 2]
    let result_shape = result.shape().unwrap();
    assert_eq!(result_shape.len(), 2);
    assert_eq!(result_shape[0].as_const().unwrap(), 1);
    assert_eq!(result_shape[1].as_const().unwrap(), 2);
}

// ========== All Tests ==========

#[test]
fn test_all_all_true() {
    let t = Tensor::from_slice([true, true, true]);
    let result = t.all(()).unwrap();
    // Should be true
    assert_eq!(result.uop.dtype(), DType::Bool);
}

#[test]
fn test_all_one_false() {
    let t = Tensor::from_slice([true, false, true]);
    let result = t.all(()).unwrap();
    // Should be false
    assert_eq!(result.uop.dtype(), DType::Bool);
}

#[test]
fn test_all_numeric() {
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let result = t.all(()).unwrap();
    // All non-zero, should be true
    assert_eq!(result.uop.dtype(), DType::Bool);
}

#[test]
fn test_all_numeric_with_zero() {
    let t = Tensor::from_slice([1.0f32, 0.0, 3.0]);
    let result = t.all(()).unwrap();
    // Has zero, should be false
    assert_eq!(result.uop.dtype(), DType::Bool);
}

#[test]
fn test_all_2d_multiaxis() {
    let t = Tensor::from_slice([true, true, true, true]).try_reshape(&[2, 2]).unwrap();
    let result = t.all(()).unwrap();
    // All true across all axes
    assert_eq!(result.uop.dtype(), DType::Bool);
}

// ========== Edge Cases ==========

#[test]
fn test_argmax_single_element() {
    let t = Tensor::from_slice([42.0f32]);
    let result = t.argmax(Some(0)).unwrap();
    // Only element, index should be 0
    assert_eq!(result.uop.dtype(), DType::Int32);
}

#[test]
fn test_argmax_all_equal() {
    let t = Tensor::from_slice([5.0f32, 5.0, 5.0, 5.0]);
    let result = t.argmax(Some(0)).unwrap();
    // All equal, should return first index (0)
    assert_eq!(result.uop.dtype(), DType::Int32);
}
