use crate::reduce::AxisSpec;
use crate::test::helpers::*;
use crate::*;
use morok_dtype::DType;
use tracing_test::traced_test;

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
    assert_eq!(result.uop().dtype(), DType::Int32);
}

#[test]
fn test_argmax_1d_ties_first_occurrence() {
    let t = Tensor::from_slice([1.0f32, 5.0, 3.0, 5.0, 2.0]);
    let result = t.argmax(Some(0)).unwrap();
    // Two maxima at indices 1 and 3, should return 1 (first)
    assert_eq!(result.uop().dtype(), DType::Int32);
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
    assert_eq!(result.uop().dtype(), DType::Int32);
}

#[test]
fn test_argmin_float() {
    let t = Tensor::from_slice([1.5f32, -2.3, 0.5, 1.0]);
    let result = t.argmin(Some(0)).unwrap();
    // Min is -2.3 at index 1
    assert_eq!(result.uop().dtype(), DType::Int32);
}

#[test]
fn test_argmin_int() {
    let t = Tensor::from_slice([5i32, 3, 1, 4, 2]);
    let result = t.argmin(Some(0)).unwrap();
    // Min is 1 at index 2
    assert_eq!(result.uop().dtype(), DType::Int32);
}

#[test]
fn test_argmin_bool() {
    let t = Tensor::from_slice([true, false, true]);
    let result = t.argmin(Some(0)).unwrap();
    // Min (false) at index 1
    assert_eq!(result.uop().dtype(), DType::Int32);
}

// ========== Any Tests ==========

#[test]
fn test_any_all_true() {
    let t = Tensor::from_slice([true, true, true]);
    let result = t.any(()).unwrap();
    // Should be true
    assert_eq!(result.uop().dtype(), DType::Bool);
}

#[test]
fn test_any_all_false() {
    let t = Tensor::from_slice([false, false, false]);
    let result = t.any(()).unwrap();
    // Should be false
    assert_eq!(result.uop().dtype(), DType::Bool);
}

#[test]
fn test_any_mixed() {
    let t = Tensor::from_slice([false, true, false]);
    let result = t.any(()).unwrap();
    // Should be true
    assert_eq!(result.uop().dtype(), DType::Bool);
}

#[test]
fn test_any_numeric() {
    let t = Tensor::from_slice([0.0f32, 1.0, 0.0]);
    let result = t.any(()).unwrap();
    // Non-zero treated as true, should be true
    assert_eq!(result.uop().dtype(), DType::Bool);
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
    assert_eq!(result.uop().dtype(), DType::Bool);
}

#[test]
fn test_all_one_false() {
    let t = Tensor::from_slice([true, false, true]);
    let result = t.all(()).unwrap();
    // Should be false
    assert_eq!(result.uop().dtype(), DType::Bool);
}

#[test]
fn test_all_numeric() {
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let result = t.all(()).unwrap();
    // All non-zero, should be true
    assert_eq!(result.uop().dtype(), DType::Bool);
}

#[test]
fn test_all_numeric_with_zero() {
    let t = Tensor::from_slice([1.0f32, 0.0, 3.0]);
    let result = t.all(()).unwrap();
    // Has zero, should be false
    assert_eq!(result.uop().dtype(), DType::Bool);
}

#[test]
fn test_all_2d_multiaxis() {
    let t = Tensor::from_slice([true, true, true, true]).try_reshape(&[2, 2]).unwrap();
    let result = t.all(()).unwrap();
    // All true across all axes
    assert_eq!(result.uop().dtype(), DType::Bool);
}

// ========== Edge Cases ==========

#[test]
fn test_argmax_single_element() {
    let t = Tensor::from_slice([42.0f32]);
    let result = t.argmax(Some(0)).unwrap();
    // Only element, index should be 0
    assert_eq!(result.uop().dtype(), DType::Int32);
}

#[test]
fn test_argmax_all_equal() {
    let t = Tensor::from_slice([5.0f32, 5.0, 5.0, 5.0]);
    let result = t.argmax(Some(0)).unwrap();
    // All equal, should return first index (0)
    assert_eq!(result.uop().dtype(), DType::Int32);
}

// ============================================================================
// VALUE-VERIFYING TESTS (ported from Tinygrad test_ops.py)
// ============================================================================

// ========== Sum Tests (from Tinygrad test_ops.py:1344-1380) ==========

#[test]
fn test_sum_1d_value() {
    let _guard = test_setup();
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);
    let result = realize_f32(t.sum(()).unwrap());
    assert_close_f32(&result, &[10.0], 1e-6);
}

#[test]
fn test_sum_2d_full_value() {
    let _guard = test_setup();
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).try_reshape(&[2, 3]).unwrap();
    let result = realize_f32(t.sum(()).unwrap());
    assert_close_f32(&result, &[21.0], 1e-6);
}

#[test]
fn test_sum_axis0_value() {
    let _guard = test_setup();
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).try_reshape(&[2, 3]).unwrap();
    let result = realize_f32(t.sum(0).unwrap());
    // [[1, 2, 3], [4, 5, 6]] -> [1+4, 2+5, 3+6] = [5, 7, 9]
    assert_close_f32(&result, &[5.0, 7.0, 9.0], 1e-6);
}

#[test]
fn test_sum_axis1_value() {
    let _guard = test_setup();
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).try_reshape(&[2, 3]).unwrap();
    let result = realize_f32(t.sum(1).unwrap());
    // [[1, 2, 3], [4, 5, 6]] -> [1+2+3, 4+5+6] = [6, 15]
    assert_close_f32(&result, &[6.0, 15.0], 1e-6);
}

#[test]
fn test_sum_keepdim_value() {
    let _guard = test_setup();
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]).try_reshape(&[2, 2]).unwrap();
    let result = t.sum_with().axes(1).keepdim(true).call().unwrap();
    let arr = realize_f32(result);
    assert_eq!(arr.shape(), &[2, 1]);
    // [[1, 2], [3, 4]] -> [[3], [7]]
    assert_close_f32(&arr, &[3.0, 7.0], 1e-6);
}

#[test]
fn test_sum_single_element_value() {
    let _guard = test_setup();
    let t = Tensor::from_slice([42.0f32]);
    let result = realize_f32(t.sum(()).unwrap());
    assert_close_f32(&result, &[42.0], 1e-6);
}

#[test]
fn test_sum_negative_values() {
    let _guard = test_setup();
    let t = Tensor::from_slice([-1.0f32, -2.0, 3.0, 4.0]);
    let result = realize_f32(t.sum(()).unwrap());
    assert_close_f32(&result, &[4.0], 1e-6);
}

// ========== Max Tests (from Tinygrad test_ops.py:1405-1416) ==========

#[test]
fn test_max_1d_value() {
    let _guard = test_setup();
    let t = Tensor::from_slice([1.0f32, 5.0, 3.0, 2.0]);
    let result = realize_f32(t.max(()).unwrap());
    assert_close_f32(&result, &[5.0], 1e-6);
}

#[test]
fn test_max_2d_axis1_value() {
    let _guard = test_setup();
    let t = Tensor::from_slice([1.0f32, 5.0, 3.0, 2.0, 8.0, 4.0]).try_reshape(&[2, 3]).unwrap();
    let result = realize_f32(t.max(1).unwrap());
    // [[1, 5, 3], [2, 8, 4]] -> [5, 8]
    assert_close_f32(&result, &[5.0, 8.0], 1e-6);
}

#[test]
fn test_max_2d_axis0_value() {
    let _guard = test_setup();
    let t = Tensor::from_slice([1.0f32, 5.0, 3.0, 2.0, 8.0, 4.0]).try_reshape(&[2, 3]).unwrap();
    let result = realize_f32(t.max(0).unwrap());
    // [[1, 5, 3], [2, 8, 4]] -> [2, 8, 4]
    assert_close_f32(&result, &[2.0, 8.0, 4.0], 1e-6);
}

#[test]
fn test_max_negative_values() {
    let _guard = test_setup();
    let t = Tensor::from_slice([-5.0f32, -1.0, -3.0, -2.0]);
    let result = realize_f32(t.max(()).unwrap());
    assert_close_f32(&result, &[-1.0], 1e-6);
}

// ========== Min Tests (from Tinygrad test_ops.py:1394-1403) ==========

#[test]
fn test_min_1d_value() {
    let _guard = test_setup();
    let t = Tensor::from_slice([5.0f32, 1.0, 3.0, 2.0]);
    let result = realize_f32(t.min(()).unwrap());
    assert_close_f32(&result, &[1.0], 1e-6);
}

#[test]
fn test_min_2d_axis1_value() {
    let _guard = test_setup();
    let t = Tensor::from_slice([1.0f32, 5.0, 3.0, 2.0, 8.0, 4.0]).try_reshape(&[2, 3]).unwrap();
    let result = realize_f32(t.min(1).unwrap());
    // [[1, 5, 3], [2, 8, 4]] -> [1, 2]
    assert_close_f32(&result, &[1.0, 2.0], 1e-6);
}

#[test]
fn test_min_negative_value() {
    let _guard = test_setup();
    let t = Tensor::from_slice([-1.0f32, -5.0, -3.0]);
    let result = realize_f32(t.min(()).unwrap());
    assert_close_f32(&result, &[-5.0], 1e-6);
}

// ========== Argmax Tests (from Tinygrad test_ops.py:1087-1105) ==========

#[test]
#[traced_test]
fn test_argmax_debug_steps() {
    let _guard = test_setup();

    // Simplest test: compare two tensors
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let b = Tensor::from_slice([1.0f32, 5.0, 3.0]);
    let eq = a.try_eq(&b).unwrap();
    let eq_result = eq.realize().unwrap().to_ndarray::<bool>().unwrap();
    println!("Simple eq test: a=[1,2,3], b=[1,5,3], a==b={:?}", eq_result.as_slice().unwrap());
    assert_eq!(eq_result.as_slice().unwrap(), &[true, false, true], "Simple eq failed");

    // Test with broadcast
    let c = Tensor::from_slice([1.0f32, 2.0, 3.0, 2.0]);
    let two = Tensor::from_slice([2.0f32]);
    let two_broadcast = two.try_expand(&[4]).unwrap();
    println!("two_broadcast shape={:?}", two_broadcast.uop().shape());
    let eq2 = c.try_eq(&two_broadcast).unwrap();
    let eq2_result = eq2.realize().unwrap().to_ndarray::<bool>().unwrap();
    println!(
        "Broadcast eq: c=[1,2,3,2], two_broadcast=[2,2,2,2], c==two_broadcast={:?}",
        eq2_result.as_slice().unwrap()
    );
    // Expected: [false, true, false, true] (positions 1 and 3 equal 2)
    assert_eq!(eq2_result.as_slice().unwrap(), &[false, true, false, true], "Broadcast eq failed");

    // Test expand of reduction result
    let d = Tensor::from_slice([1.0f32, 5.0, 3.0, 2.0]);
    let d_max = d.max_with().axes(0).keepdim(true).call().unwrap();
    println!("d_max shape={:?}", d_max.uop().shape());

    // Realize d_max first - IMPORTANT: use the realized tensor for subsequent ops
    // In Rust, tensor.clone().realize() creates a new independent tensor.
    // The original tensor's uop is unchanged, so we must use the realized result.
    let d_max_realized = d_max.realize().unwrap();
    let d_max_value = d_max_realized.clone().to_ndarray::<f32>().unwrap();
    println!("DEBUG: d_max realized value = {:?}", d_max_value.as_slice().unwrap());

    // Use the REALIZED d_max for expansion
    let d_max_expanded = d_max_realized.try_expand(&[4]).unwrap();
    println!("d_max_expanded shape={:?}", d_max_expanded.uop().shape());

    let eq3 = d.try_eq(&d_max_expanded).unwrap();
    eprintln!("=== EQ3 AST ===\n{}", eq3.uop().tree_full());
    let eq3_result = eq3.realize().unwrap().to_ndarray::<bool>().unwrap();
    println!("Reduction expand eq: d=[1,5,3,2], d_max=5, eq={:?}", eq3_result.as_slice().unwrap());
    // Expected: [false, true, false, false] (only position 1 equals 5)
    assert_eq!(eq3_result.as_slice().unwrap(), &[false, true, false, false], "Reduction expand eq failed");
}

#[test]
fn test_argmax_full_steps() {
    let _guard = test_setup();
    let t = Tensor::from_slice([1.0f32, 3.0, 2.0, 5.0, 4.0]);

    // Step 1: max value along axis 0
    let max_vals = t.max_with().axes(0).keepdim(true).call().unwrap();

    // Step 2: expand max to original shape
    let max_broadcast = max_vals.try_expand(&[5]).unwrap();
    println!("max_broadcast shape={:?}", max_broadcast.uop().shape());

    // Step 3: mask where values == max
    let mask = t.try_eq(&max_broadcast).unwrap();
    let mask_realized = mask.clone().realize().unwrap().to_ndarray::<bool>().unwrap();
    println!("Mask (eq max): {:?}", mask_realized.as_slice().unwrap());
    assert_eq!(mask_realized.as_slice().unwrap(), &[false, false, false, true, false], "Mask mismatch");

    // Step 4: Create descending indices [5, 4, 3, 2, 1]
    let axis_size = 5;
    let indices = Tensor::arange(axis_size as i64, Some(0), Some(-1)).unwrap();
    let indices_realized = indices.clone().realize().unwrap().to_ndarray::<i32>().unwrap();
    println!("Step 4 - Descending indices: {:?}", indices_realized.as_slice().unwrap());
    assert_eq!(indices_realized.as_slice().unwrap(), &[5i32, 4, 3, 2, 1], "Indices mismatch");

    // Step 5: Cast to int32
    let mask_int = mask.cast(DType::Int32).unwrap();
    let mask_int_realized = realize_i32(mask_int.clone());
    println!("Step 5a - Mask as int32: {:?}", mask_int_realized.as_slice().unwrap());
    // Expected: [0, 0, 0, 1, 0]
    assert_eq!(mask_int_realized.as_slice().unwrap(), &[0, 0, 0, 1, 0], "Mask int mismatch");

    let indices_i32 = indices.cast(DType::Int32).unwrap();
    let indices_i32_realized = realize_i32(indices_i32.clone());
    println!("Step 5b - Indices as int32: {:?}", indices_i32_realized.as_slice().unwrap());
    assert_eq!(indices_i32_realized.as_slice().unwrap(), &[5, 4, 3, 2, 1], "Indices int32 mismatch");

    // Step 6: Multiply mask by indices
    let masked_indices = mask_int.try_mul(&indices_i32).unwrap();
    let masked_realized = realize_i32(masked_indices.clone());
    println!("Step 6 - Masked indices (mask * indices): {:?}", masked_realized.as_slice().unwrap());
    // Expected: [0, 0, 0, 2, 0]
    assert_eq!(masked_realized.as_slice().unwrap(), &[0, 0, 0, 2, 0], "Masked indices mismatch");

    // Step 7: max of masked indices
    let max_idx = masked_indices.max_with().axes(0).keepdim(false).call().unwrap();
    let max_idx_realized = realize_i32(max_idx.clone());
    println!("Step 7 - Max of masked indices: {:?}", max_idx_realized.as_slice().unwrap());
    // Expected: [2]
    assert_eq!(max_idx_realized.as_slice().unwrap(), &[2], "Max idx mismatch");

    // Step 8: n - max_idx
    let n_tensor = Tensor::from_slice([axis_size]);
    let n_scalar = n_tensor.try_reshape(&[]).unwrap();
    let result = n_scalar.try_sub(&max_idx).unwrap();
    let result_realized = realize_i32(result);
    println!("Step 8 - Final result (N - max_idx): {:?}", result_realized.as_slice().unwrap());
    // Expected: [3]
    assert_eq!(result_realized.as_slice().unwrap(), &[3], "Final result mismatch");
}

#[test]
#[tracing_test::traced_test]
fn test_argmax_value_1d() {
    let _guard = test_setup();
    let t = Tensor::from_slice([1.0f32, 3.0, 2.0, 5.0, 4.0]);
    let result = realize_i32(t.argmax(Some(0)).unwrap());
    // Max 5.0 at index 3
    assert_eq_i32(&result, &[3]);
}

#[test]
fn test_argmax_ties_first_value() {
    let _guard = test_setup();
    // Tinygrad test: [2, 2] -> should return 0 (first occurrence)
    let t = Tensor::from_slice([2.0f32, 2.0]);
    let result = realize_i32(t.argmax(Some(0)).unwrap());
    assert_eq_i32(&result, &[0]);
}

#[test]
fn test_argmax_2d_axis0_value() {
    let _guard = test_setup();
    let t = Tensor::from_slice([1.0f32, 3.0, 2.0, 4.0, 2.0, 5.0]).try_reshape(&[2, 3]).unwrap();
    let result = realize_i32(t.argmax(Some(0)).unwrap());
    // [[1, 3, 2], [4, 2, 5]]
    // Max per column: [4>1, 3>2, 5>2] -> indices [1, 0, 1]
    assert_eq_i32(&result, &[1, 0, 1]);
}

#[test]
fn test_argmax_2d_axis1_value() {
    let _guard = test_setup();
    let t = Tensor::from_slice([1.0f32, 3.0, 2.0, 4.0, 2.0, 5.0]).try_reshape(&[2, 3]).unwrap();
    let result = realize_i32(t.argmax(Some(1)).unwrap());
    // [[1, 3, 2], [4, 2, 5]]
    // Max per row: [3 at idx 1, 5 at idx 2] -> [1, 2]
    assert_eq_i32(&result, &[1, 2]);
}

#[test]
fn test_argmax_flatten_value() {
    let _guard = test_setup();
    let t = Tensor::from_slice([1.0f32, 3.0, 2.0, 4.0, 2.0, 5.0]).try_reshape(&[2, 3]).unwrap();
    let result = realize_i32(t.argmax(None).unwrap());
    // Flattened: [1, 3, 2, 4, 2, 5], max 5.0 at index 5
    assert_eq_i32(&result, &[5]);
}

// ========== Argmin Tests (from Tinygrad test_ops.py:1106-1122) ==========

#[test]
#[tracing_test::traced_test]
fn test_argmin_value_1d() {
    let _guard = test_setup();
    let t = Tensor::from_slice([5.0f32, 3.0, 1.0, 4.0, 2.0]);
    let result = realize_i32(t.argmin(Some(0)).unwrap());
    // Min 1.0 at index 2
    assert_eq_i32(&result, &[2]);
}

#[test]
fn test_argmin_negative_value() {
    let _guard = test_setup();
    let t = Tensor::from_slice([1.5f32, -2.3, 0.5, 1.0]);
    let result = realize_i32(t.argmin(Some(0)).unwrap());
    // Min -2.3 at index 1
    assert_eq_i32(&result, &[1]);
}

#[test]
fn test_argmin_2d_axis0_value() {
    let _guard = test_setup();
    let t = Tensor::from_slice([1.0f32, 3.0, 2.0, 4.0, 2.0, 5.0]).try_reshape(&[2, 3]).unwrap();
    let result = realize_i32(t.argmin(Some(0)).unwrap());
    // [[1, 3, 2], [4, 2, 5]]
    // Min per column: [1<4, 2<3, 2<5] -> indices [0, 1, 0]
    assert_eq_i32(&result, &[0, 1, 0]);
}

#[test]
fn test_argmin_2d_axis1_value() {
    let _guard = test_setup();
    let t = Tensor::from_slice([1.0f32, 3.0, 2.0, 4.0, 2.0, 5.0]).try_reshape(&[2, 3]).unwrap();
    let result = realize_i32(t.argmin(Some(1)).unwrap());
    // [[1, 3, 2], [4, 2, 5]]
    // Min per row: [1 at idx 0, 2 at idx 1] -> [0, 1]
    assert_eq_i32(&result, &[0, 1]);
}

// ========== Mean Tests (from Tinygrad test_ops.py:1465-1471) ==========

#[test]
fn test_mean_1d_value() {
    let _guard = test_setup();
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);
    let result = realize_f32(t.mean(()).unwrap());
    assert_close_f32(&result, &[2.5], 1e-6);
}

#[test]
fn test_mean_2d_axis1_value() {
    let _guard = test_setup();
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).try_reshape(&[2, 3]).unwrap();
    let result = realize_f32(t.mean(1).unwrap());
    // [[1, 2, 3], [4, 5, 6]] -> [mean(1,2,3), mean(4,5,6)] = [2, 5]
    assert_close_f32(&result, &[2.0, 5.0], 1e-6);
}

#[test]
fn test_mean_2d_axis0_value() {
    let _guard = test_setup();
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).try_reshape(&[2, 3]).unwrap();
    let result = realize_f32(t.mean(0).unwrap());
    // [[1, 2, 3], [4, 5, 6]] -> [mean(1,4), mean(2,5), mean(3,6)] = [2.5, 3.5, 4.5]
    assert_close_f32(&result, &[2.5, 3.5, 4.5], 1e-6);
}

// ========== Any Tests (from Tinygrad test_ops.py:1423-1432) ==========

#[test]
fn test_any_value_all_true() {
    let _guard = test_setup();
    let t = Tensor::from_slice([true, true, true]);
    let result = t.any(()).unwrap().realize().unwrap().to_ndarray::<bool>().unwrap();
    assert!(result[[]]);
}

#[test]
fn test_any_value_one_true() {
    let _guard = test_setup();
    let t = Tensor::from_slice([false, true, false]);
    let result = t.any(()).unwrap().realize().unwrap().to_ndarray::<bool>().unwrap();
    assert!(result[[]]);
}

#[test]
fn test_any_value_all_false() {
    let _guard = test_setup();
    let t = Tensor::from_slice([false, false, false]);
    let result = t.any(()).unwrap().realize().unwrap().to_ndarray::<bool>().unwrap();
    assert!(!result[[]]);
}

#[test]
fn test_any_2d_axis0_value() {
    let _guard = test_setup();
    let t = Tensor::from_slice([true, false, false, true]).try_reshape(&[2, 2]).unwrap();
    let result = t.any(0).unwrap().realize().unwrap().to_ndarray::<bool>().unwrap();
    // [[true, false], [false, true]] -> any along axis 0 -> [true, true]
    assert_eq!(result.as_slice().unwrap(), &[true, true]);
}

#[test]
fn test_any_2d_axis1_value() {
    let _guard = test_setup();
    let t = Tensor::from_slice([true, false, false, false]).try_reshape(&[2, 2]).unwrap();
    let result = t.any(1).unwrap().realize().unwrap().to_ndarray::<bool>().unwrap();
    // [[true, false], [false, false]] -> any along axis 1 -> [true, false]
    assert_eq!(result.as_slice().unwrap(), &[true, false]);
}

// ========== All Tests (from Tinygrad test_ops.py:1435-1444) ==========

#[test]
fn test_all_value_all_true() {
    let _guard = test_setup();
    let t = Tensor::from_slice([true, true, true]);
    let result = t.all(()).unwrap().realize().unwrap().to_ndarray::<bool>().unwrap();
    assert!(result[[]]);
}

#[test]
fn test_all_value_one_false() {
    let _guard = test_setup();
    let t = Tensor::from_slice([true, false, true]);
    let result = t.all(()).unwrap().realize().unwrap().to_ndarray::<bool>().unwrap();
    assert!(!result[[]]);
}

#[test]
fn test_all_value_all_false() {
    let _guard = test_setup();
    let t = Tensor::from_slice([false, false, false]);
    let result = t.all(()).unwrap().realize().unwrap().to_ndarray::<bool>().unwrap();
    assert!(!result[[]]);
}

#[test]
fn test_all_2d_axis0_value() {
    let _guard = test_setup();
    let t = Tensor::from_slice([true, true, false, true]).try_reshape(&[2, 2]).unwrap();
    let result = t.all(0).unwrap().realize().unwrap().to_ndarray::<bool>().unwrap();
    // [[true, true], [false, true]] -> all along axis 0 -> [false, true]
    assert_eq!(result.as_slice().unwrap(), &[false, true]);
}

#[test]
fn test_all_2d_axis1_value() {
    let _guard = test_setup();
    let t = Tensor::from_slice([true, true, true, false]).try_reshape(&[2, 2]).unwrap();
    let result = t.all(1).unwrap().realize().unwrap().to_ndarray::<bool>().unwrap();
    // [[true, true], [true, false]] -> all along axis 1 -> [true, false]
    assert_eq!(result.as_slice().unwrap(), &[true, false]);
}

// ========== Edge Cases (from Tinygrad) ==========

#[test]
fn test_argmax_single_element_value() {
    let _guard = test_setup();
    let t = Tensor::from_slice([42.0f32]);
    let result = realize_i32(t.argmax(Some(0)).unwrap());
    // Only element, index should be 0
    assert_eq_i32(&result, &[0]);
}

#[test]
fn test_argmax_all_equal_value() {
    let _guard = test_setup();
    let t = Tensor::from_slice([5.0f32, 5.0, 5.0, 5.0]);
    let result = realize_i32(t.argmax(Some(0)).unwrap());
    // All equal, should return first index (0)
    assert_eq_i32(&result, &[0]);
}

#[test]
fn test_argmin_single_element_value() {
    let _guard = test_setup();
    let t = Tensor::from_slice([42.0f32]);
    let result = realize_i32(t.argmin(Some(0)).unwrap());
    // Only element, index should be 0
    assert_eq_i32(&result, &[0]);
}

#[test]
fn test_max_single_element_value() {
    let _guard = test_setup();
    let t = Tensor::from_slice([42.0f32]);
    let result = realize_f32(t.max(()).unwrap());
    assert_close_f32(&result, &[42.0], 1e-6);
}

#[test]
fn test_min_single_element_value() {
    let _guard = test_setup();
    let t = Tensor::from_slice([42.0f32]);
    let result = realize_f32(t.min(()).unwrap());
    assert_close_f32(&result, &[42.0], 1e-6);
}

#[test]
fn test_debug_argmin_intermediate() {
    let _guard = test_setup();
    let values = Tensor::from_slice([5.0f32, 3.0, 1.0, 4.0, 2.0]);

    // Test neg first - does it produce correct values?
    let inverted = -values.clone();
    let inv_arr = realize_f32(inverted);
    println!("Negated values: {:?}", inv_arr.as_slice().unwrap());
    // Expected: [-5.0, -3.0, -1.0, -4.0, -2.0]
    assert_close_f32(&inv_arr, &[-5.0, -3.0, -1.0, -4.0, -2.0], 1e-6);

    // Test max of explicit negated (should be -1.0)
    let inverted2 = Tensor::from_slice([-5.0f32, -3.0, -1.0, -4.0, -2.0]);
    let max_inv = inverted2.max_with().axes(0).keepdim(false).call().unwrap();
    let max_arr = realize_f32(max_inv);
    println!("Max of explicit negated: {:?}", max_arr.as_slice().unwrap());
    assert_close_f32(&max_arr, &[-1.0], 1e-6);

    // Test argmax of explicit negated values
    let inverted3 = Tensor::from_slice([-5.0f32, -3.0, -1.0, -4.0, -2.0]);
    let argmax_inv = inverted3.argmax(0).unwrap();
    let argmax_arr = realize_i32(argmax_inv);
    println!("Argmax of explicit negated: {:?}", argmax_arr.as_slice().unwrap());
    assert_eq!(argmax_arr.as_slice().unwrap()[0], 2); // -1.0 is at index 2
}

#[test]
#[traced_test]
fn test_debug_lazy_neg_max() {
    let _guard = test_setup();
    let values = Tensor::from_slice([5.0f32, 3.0, 1.0, 4.0, 2.0]);

    // Test max of lazy negated values
    let inverted = -values.clone(); // lazy neg
    let max_lazy = inverted.max(()).unwrap();
    let max_arr = realize_f32(max_lazy);
    println!("Max of LAZY negated: {:?}", max_arr.as_slice().unwrap());
    // Should be -1.0 (max of [-5, -3, -1, -4, -2])
    assert_close_f32(&max_arr, &[-1.0], 1e-6);
}

#[test]
#[traced_test]
fn test_debug_lazy_neg_argmax() {
    let _guard = test_setup();
    let values = Tensor::from_slice([5.0f32, 3.0, 1.0, 4.0, 2.0]);

    // Chain neg and argmax LAZILY (like argmin does internally)
    let inverted = -values; // lazy neg
    let argmax_lazy = inverted.argmax(0).unwrap(); // lazy argmax of lazy neg
    let argmax_arr = realize_i32(argmax_lazy);
    println!("Argmax of LAZY negated: {:?}", argmax_arr.as_slice().unwrap());
    // Should be 2 (index of -1.0, which is the max of [-5, -3, -1, -4, -2])
    assert_eq!(argmax_arr.as_slice().unwrap()[0], 2);
}
