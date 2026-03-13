use crate::test::helpers::*;
use ndarray::{Array2, Array3, array};

#[test]
fn test_registry_transpose() {
    let registry = OpRegistry::new();
    let x = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let mut node = NodeProto::default();
    node.attribute.push(make_attr_ints("perm", &[1, 0]));

    let result = registry.dispatch("Transpose", "", &[x], &node);
    let result = result.unwrap().realize().unwrap();
    assert!(result.buffer().is_some());
}

#[test]
fn test_registry_flatten() {
    let registry = OpRegistry::new();
    let x = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let node = NodeProto::default(); // axis defaults to 1

    let result = registry.dispatch("Flatten", "", &[x], &node).unwrap();
    let realized = result.realize().unwrap();
    assert!(realized.buffer().is_some());
}

#[test]
fn test_reshape_allowzero_copy() {
    let registry = OpRegistry::new();
    let data = Tensor::from_ndarray(&Array3::from_elem((2, 3, 4), 1.0f32));
    let shape_tensor = Tensor::from_slice([0i64, 3, -1]);
    let inputs = vec![Some(data), Some(shape_tensor)];
    let node = NodeProto::default(); // allowzero defaults to 0

    let result = registry.dispatch_multi("Reshape", "", &inputs, &node, i64::MAX).unwrap();
    let s = result[0].shape().unwrap();
    assert_eq!(s.iter().map(|d| d.as_const().unwrap()).collect::<Vec<_>>(), vec![2, 3, 4]);
}

#[test]
fn test_expand_broadcast() {
    let registry = OpRegistry::new();
    let data = Tensor::from_ndarray(&array![[1.0f32], [2.0], [3.0]]);
    let target = Tensor::from_slice([2i64, 3, 4]);
    let inputs = vec![Some(data), Some(target)];
    let node = NodeProto::default();

    let result = registry.dispatch_multi("Expand", "", &inputs, &node, i64::MAX).unwrap();
    let s = result[0].shape().unwrap();
    assert_eq!(s.iter().map(|d| d.as_const().unwrap()).collect::<Vec<_>>(), vec![2, 3, 4]);
}

#[test]
fn test_flatten_axis_variants() {
    let registry = OpRegistry::new();

    for (axis, expected_shape) in [(0, vec![1, 24]), (1, vec![2, 12]), (2, vec![6, 4])] {
        let x = Tensor::from_ndarray(&Array3::from_elem((2, 3, 4), 1.0f32));
        let mut node = NodeProto::default();
        node.attribute.push(make_attr_int("axis", axis));

        let result = registry.dispatch("Flatten", "", &[x], &node).unwrap();
        let s = result.shape().unwrap();
        let dims: Vec<usize> = s.iter().map(|d| d.as_const().unwrap()).collect();
        assert_eq!(dims, expected_shape, "Flatten axis={axis}");
    }
}

#[test]
fn test_flatten_negative_axis() {
    let registry = OpRegistry::new();
    let x = Tensor::from_ndarray(&Array3::from_elem((2, 3, 4), 1.0f32));
    let mut node = NodeProto::default();
    node.attribute.push(make_attr_int("axis", -1));

    let result = registry.dispatch("Flatten", "", &[x], &node).unwrap();
    let s = result.shape().unwrap();
    let dims: Vec<usize> = s.iter().map(|d| d.as_const().unwrap()).collect();
    assert_eq!(dims, vec![6, 4]); // axis=-1 -> axis=2 -> pre=2*3=6, post=4
}

#[test]
fn test_pad_with_axes() {
    let registry = OpRegistry::new();
    let data = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let pads = Tensor::from_slice([1i64, 1]); // 1 before, 1 after
    let axes = Tensor::from_slice([1i64]); // only pad axis 1
    let inputs = vec![Some(data), Some(pads), None, Some(axes)];
    let node = NodeProto::default();

    let result = registry.dispatch_multi("Pad", "", &inputs, &node, i64::MAX).unwrap();
    let s = result[0].shape().unwrap();
    assert_eq!(s.iter().map(|d| d.as_const().unwrap()).collect::<Vec<_>>(), vec![2, 5]);
}

#[test]
fn test_shape_start_end() {
    let registry = OpRegistry::new();
    let arr = Array3::from_shape_vec((2, 3, 4), (1..=24).map(|v| v as f32).collect()).unwrap();
    let x = Tensor::from_ndarray(&arr);
    let mut node = NodeProto::default();
    node.attribute.push(make_attr_int("start", 1));
    node.attribute.push(make_attr_int("end", 3));

    let result = registry.dispatch("Shape", "", &[x], &node).unwrap();
    let vals = result.to_vec::<i64>().unwrap();
    assert_eq!(vals, vec![3, 4]);
}

#[test]
fn test_shape_negative_start() {
    let registry = OpRegistry::new();
    let x = Tensor::from_ndarray(&Array3::from_elem((2, 3, 4), 1.0f32));
    let mut node = NodeProto::default();
    node.attribute.push(make_attr_int("start", -1));

    let result = registry.dispatch("Shape", "", &[x], &node).unwrap();
    let vals = result.to_vec::<i64>().unwrap();
    assert_eq!(vals, vec![4]);
}

#[test]
fn test_shape_start_gt_end() {
    let registry = OpRegistry::new();
    let x = Tensor::from_ndarray(&Array3::from_elem((2, 3, 4), 1.0f32));
    let inputs = vec![Some(x)];
    let mut node = NodeProto::default();
    node.attribute.push(make_attr_int("start", 2));
    node.attribute.push(make_attr_int("end", 1));

    let result = registry.dispatch_multi("Shape", "", &inputs, &node, i64::MAX).unwrap();
    let vals = result[0].to_vec::<i64>().unwrap();
    assert!(vals.is_empty());
}

#[test]
fn test_split_remainder_distribution() {
    let registry = OpRegistry::new();
    let data = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    let inputs = vec![Some(data)];

    let mut node = NodeProto::default();
    node.attribute.push(make_attr_int("num_outputs", 3));

    let result = registry.dispatch_multi("Split", "", &inputs, &node, i64::MAX).unwrap();
    assert_eq!(result.len(), 3);
    let shapes: Vec<usize> = result.iter().map(|t| t.shape().unwrap()[0].as_const().unwrap()).collect();
    assert_eq!(shapes, vec![3, 2, 2]);
}

#[test]
fn test_dropout_mask_shape() {
    let registry = OpRegistry::new();
    let x = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let inputs = vec![Some(x)];
    let node = NodeProto::default();

    let result = registry.dispatch_multi("Dropout", "", &inputs, &node, i64::MAX).unwrap();
    assert_eq!(result.len(), 2);
    let mask_shape = result[1].shape().unwrap();
    assert_eq!(mask_shape.len(), 2);
    assert_eq!(mask_shape[0].as_const().unwrap(), 2);
    assert_eq!(mask_shape[1].as_const().unwrap(), 3);
    assert!(result[1].to_vec::<bool>().unwrap().iter().all(|&v| v));
}

#[test]
fn test_constant_of_shape_empty() {
    let registry = OpRegistry::new();
    let shape = Tensor::from_slice([0i64]);
    let node = NodeProto::default();

    let result = registry.dispatch("ConstantOfShape", "", &[shape], &node).unwrap();
    let s = result.shape().unwrap();
    assert_eq!(s.len(), 1);
    assert_eq!(s[0].as_const().unwrap(), 0);
    assert_eq!(result.to_vec::<f32>().unwrap().len(), 0);
}

#[test]
fn test_eye_like() {
    let registry = OpRegistry::new();
    let x = Tensor::from_ndarray(&Array2::<f32>::zeros((3, 3)));
    let node = NodeProto::default();
    let result = registry.dispatch("EyeLike", "", &[x], &node).unwrap();
    let s = result.shape().unwrap();
    assert_eq!(s.iter().map(|d| d.as_const().unwrap()).collect::<Vec<_>>(), vec![3, 3]);
    assert_eq!(result.to_vec::<f32>().unwrap(), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
}

#[test]
fn test_center_crop_pad_crop() {
    let registry = OpRegistry::new();
    let x = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let shape = Tensor::from_slice([2i64, 2]);
    let inputs = vec![Some(x), Some(shape)];
    let node = NodeProto::default();
    let result = registry.dispatch_multi("CenterCropPad", "", &inputs, &node, i64::MAX).unwrap();
    let s = result[0].shape().unwrap();
    assert_eq!(s.iter().map(|d| d.as_const().unwrap()).collect::<Vec<_>>(), vec![2, 2]);
}

#[test]
fn test_compress() {
    let registry = OpRegistry::new();
    let data = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);
    let condition = Tensor::from_slice([0i64, 1, 1, 0]);
    let inputs = vec![Some(data), Some(condition)];
    let node = NodeProto::default();
    let result = registry.dispatch_multi("Compress", "", &inputs, &node, i64::MAX).unwrap();
    assert_eq!(result[0].to_vec::<f32>().unwrap(), vec![2.0, 3.0]);
}
