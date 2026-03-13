use crate::test::helpers::*;

#[test]
fn test_slice_step_2() {
    let registry = OpRegistry::new();
    let data = Tensor::from_slice([0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    let starts = Tensor::from_slice([0i64]);
    let ends = Tensor::from_slice([10i64]);
    let axes = Tensor::from_slice([0i64]);
    let steps = Tensor::from_slice([2i64]);
    let inputs = vec![Some(data), Some(starts), Some(ends), Some(axes), Some(steps)];
    let node = NodeProto::default();

    let result = registry.dispatch_multi("Slice", "", &inputs, &node, i64::MAX).unwrap();
    let vals = result[0].to_vec::<f32>().unwrap();
    assert_eq!(vals, vec![0.0, 2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn test_slice_step_3() {
    let registry = OpRegistry::new();
    let data = Tensor::from_slice([0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    let starts = Tensor::from_slice([0i64]);
    let ends = Tensor::from_slice([10i64]);
    let axes = Tensor::from_slice([0i64]);
    let steps = Tensor::from_slice([3i64]);
    let inputs = vec![Some(data), Some(starts), Some(ends), Some(axes), Some(steps)];
    let node = NodeProto::default();

    let result = registry.dispatch_multi("Slice", "", &inputs, &node, i64::MAX).unwrap();
    let vals = result[0].to_vec::<f32>().unwrap();
    assert_eq!(vals, vec![0.0, 3.0, 6.0, 9.0]);
}

#[test]
fn test_slice_neg_step_2() {
    let registry = OpRegistry::new();
    let data = Tensor::from_slice([0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let starts = Tensor::from_slice([5i64]);
    let ends = Tensor::from_slice([0i64]);
    let axes = Tensor::from_slice([0i64]);
    let steps = Tensor::from_slice([-2i64]);
    let inputs = vec![Some(data), Some(starts), Some(ends), Some(axes), Some(steps)];
    let node = NodeProto::default();

    let result = registry.dispatch_multi("Slice", "", &inputs, &node, i64::MAX).unwrap();
    let vals = result[0].to_vec::<f32>().unwrap();
    assert_eq!(vals, vec![5.0, 3.0, 1.0]);
}

#[test]
fn test_slice_full_reverse() {
    let registry = OpRegistry::new();
    let data = Tensor::from_slice([0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let starts = Tensor::from_slice([5i64]);
    let ends = Tensor::from_slice([-100i64]);
    let axes = Tensor::from_slice([0i64]);
    let steps = Tensor::from_slice([-1i64]);
    let inputs = vec![Some(data), Some(starts), Some(ends), Some(axes), Some(steps)];
    let node = NodeProto::default();

    let result = registry.dispatch_multi("Slice", "", &inputs, &node, i64::MAX).unwrap();
    let vals = result[0].to_vec::<f32>().unwrap();
    assert_eq!(vals, vec![5.0, 4.0, 3.0, 2.0, 1.0, 0.0]);
}

#[test]
fn test_slice_large_start_neg_step() {
    let registry = OpRegistry::new();
    let data = Tensor::from_slice([0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let starts = Tensor::from_slice([100i64]);
    let ends = Tensor::from_slice([0i64]);
    let axes = Tensor::from_slice([0i64]);
    let steps = Tensor::from_slice([-1i64]);
    let inputs = vec![Some(data), Some(starts), Some(ends), Some(axes), Some(steps)];
    let node = NodeProto::default();

    let result = registry.dispatch_multi("Slice", "", &inputs, &node, i64::MAX).unwrap();
    let vals = result[0].to_vec::<f32>().unwrap();
    assert_eq!(vals, vec![5.0, 4.0, 3.0, 2.0, 1.0]);
}

#[test]
fn test_slice_step_zero_errors() {
    let registry = OpRegistry::new();
    let data = Tensor::from_slice([0.0f32, 1.0, 2.0]);
    let starts = Tensor::from_slice([0i64]);
    let ends = Tensor::from_slice([3i64]);
    let axes = Tensor::from_slice([0i64]);
    let steps = Tensor::from_slice([0i64]);
    let inputs = vec![Some(data), Some(starts), Some(ends), Some(axes), Some(steps)];
    let node = NodeProto::default();

    assert!(registry.dispatch_multi("Slice", "", &inputs, &node, i64::MAX).is_err());
}
