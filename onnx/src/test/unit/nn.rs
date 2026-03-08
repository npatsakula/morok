use crate::test::helpers::*;

#[test]
fn test_registry_matmul() {
    let registry = OpRegistry::new();
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]).try_reshape(&[2, 2]).unwrap();
    let b = Tensor::from_slice([5.0f32, 6.0, 7.0, 8.0]).try_reshape(&[2, 2]).unwrap();
    let node = NodeProto::default();

    let result = registry.dispatch("MatMul", "", &[a, b], &node);
    let result = result.unwrap().realize().unwrap();
    assert!(result.buffer().is_some());
}

#[test]
fn test_conv_basic() {
    let registry = OpRegistry::new();
    let x_data: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let x = Tensor::from_slice(&x_data).try_reshape(&[1, 1, 4, 4]).unwrap();
    let w = Tensor::from_slice([1.0f32; 9]).try_reshape(&[1, 1, 3, 3]).unwrap();
    let inputs = vec![Some(x), Some(w)];
    let mut node = NodeProto::default();
    node.attribute.push(make_attr_ints("kernel_shape", &[3, 3]));

    let result = registry.dispatch_multi("Conv", "", &inputs, &node, i64::MAX).unwrap();
    let arr = result[0].to_ndarray::<f32>().unwrap();
    assert_eq!(arr.shape(), &[1, 1, 2, 2]);
    assert!((arr[[0, 0, 0, 0]] - 45.0).abs() < 1e-4);
}

#[test]
fn test_conv_auto_pad_same_upper() {
    let registry = OpRegistry::new();
    let x = Tensor::from_slice([1.0f32; 9]).try_reshape(&[1, 1, 3, 3]).unwrap();
    let w = Tensor::from_slice([1.0f32; 9]).try_reshape(&[1, 1, 3, 3]).unwrap();
    let inputs = vec![Some(x), Some(w)];
    let mut node = NodeProto::default();
    node.attribute.push(make_attr_ints("kernel_shape", &[3, 3]));
    node.attribute.push(make_attr_string("auto_pad", "SAME_UPPER"));

    let result = registry.dispatch_multi("Conv", "", &inputs, &node, i64::MAX).unwrap();
    let s = result[0].shape().unwrap();
    assert_eq!(s.iter().map(|d| d.as_const().unwrap()).collect::<Vec<_>>(), vec![1, 1, 3, 3]);
}

#[test]
fn test_conv_transpose() {
    let registry = OpRegistry::new();
    let x = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]).try_reshape(&[1, 1, 2, 2]).unwrap();
    let w = Tensor::from_slice([2.0f32]).try_reshape(&[1, 1, 1, 1]).unwrap();
    let inputs = vec![Some(x), Some(w)];
    let mut node = NodeProto::default();
    node.attribute.push(make_attr_ints("kernel_shape", &[1, 1]));

    let result = registry.dispatch_multi("ConvTranspose", "", &inputs, &node, i64::MAX).unwrap();
    let arr = result[0].to_ndarray::<f32>().unwrap();
    assert_eq!(arr.shape(), &[1, 1, 2, 2]);
    assert!((arr[[0, 0, 0, 0]] - 2.0).abs() < 1e-4);
    assert!((arr[[0, 0, 1, 1]] - 8.0).abs() < 1e-4);
}

#[test]
fn test_average_pool() {
    let registry = OpRegistry::new();
    let x_data: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let x = Tensor::from_slice(&x_data).try_reshape(&[1, 1, 4, 4]).unwrap();
    let inputs = vec![Some(x)];
    let mut node = NodeProto::default();
    node.attribute.push(make_attr_ints("kernel_shape", &[2, 2]));
    node.attribute.push(make_attr_ints("strides", &[2, 2]));

    let result = registry.dispatch_multi("AveragePool", "", &inputs, &node, i64::MAX).unwrap();
    let arr = result[0].to_ndarray::<f32>().unwrap();
    assert_eq!(arr.shape(), &[1, 1, 2, 2]);
    assert!((arr[[0, 0, 0, 0]] - 2.5).abs() < 1e-4);
}

#[test]
fn test_average_pool_ceil() {
    let registry = OpRegistry::new();
    let x = Tensor::from_slice([0.0f32; 49]).try_reshape(&[1, 1, 7, 7]).unwrap();
    let inputs = vec![Some(x)];
    let mut node = NodeProto::default();
    node.attribute.push(make_attr_ints("kernel_shape", &[2, 2]));
    node.attribute.push(make_attr_ints("strides", &[3, 3]));
    node.attribute.push(make_attr_int("ceil_mode", 1));

    let result = registry.dispatch_multi("AveragePool", "", &inputs, &node, i64::MAX).unwrap();
    let s = result[0].shape().unwrap();
    assert_eq!(s.iter().map(|d| d.as_const().unwrap()).collect::<Vec<_>>(), vec![1, 1, 3, 3]);
}

#[test]
fn test_max_pool() {
    let registry = OpRegistry::new();
    let x_data: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let x = Tensor::from_slice(&x_data).try_reshape(&[1, 1, 4, 4]).unwrap();
    let inputs = vec![Some(x)];
    let mut node = NodeProto::default();
    node.attribute.push(make_attr_ints("kernel_shape", &[2, 2]));
    node.attribute.push(make_attr_ints("strides", &[2, 2]));

    let result = registry.dispatch_multi("MaxPool", "", &inputs, &node, i64::MAX).unwrap();
    assert_eq!(result.len(), 2);
    let vals = result[0].to_ndarray::<f32>().unwrap();
    assert_eq!(vals.shape(), &[1, 1, 2, 2]);
    assert!((vals[[0, 0, 0, 0]] - 5.0).abs() < 1e-4);
}

#[test]
fn test_max_pool_indices() {
    let registry = OpRegistry::new();
    let x_data: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let x = Tensor::from_slice(&x_data).try_reshape(&[1, 1, 4, 4]).unwrap();
    let inputs = vec![Some(x)];
    let mut node = NodeProto::default();
    node.attribute.push(make_attr_ints("kernel_shape", &[2, 2]));
    node.attribute.push(make_attr_ints("strides", &[2, 2]));

    let result = registry.dispatch_multi("MaxPool", "", &inputs, &node, i64::MAX).unwrap();
    assert_eq!(result.len(), 2);
    let idx = result[1].to_ndarray::<i64>().unwrap();
    assert_eq!(idx.shape(), &[1, 1, 2, 2]);
    assert_eq!(idx[[0, 0, 0, 0]], 5);
}

#[test]
fn test_global_average_pool() {
    let registry = OpRegistry::new();
    let x_data: Vec<f32> = (0..18).map(|v| v as f32).collect();
    let x = Tensor::from_slice(&x_data).try_reshape(&[1, 2, 3, 3]).unwrap();
    let inputs = vec![Some(x)];
    let node = NodeProto::default();

    let result = registry.dispatch_multi("GlobalAveragePool", "", &inputs, &node, i64::MAX).unwrap();
    let s = result[0].shape().unwrap();
    assert_eq!(s.iter().map(|d| d.as_const().unwrap()).collect::<Vec<_>>(), vec![1, 2, 1, 1]);
    let arr = result[0].to_ndarray::<f32>().unwrap();
    assert!((arr[[0, 0, 0, 0]] - 4.0).abs() < 1e-4);
    assert!((arr[[0, 1, 0, 0]] - 13.0).abs() < 1e-4);
}

#[test]
fn test_global_max_pool() {
    let registry = OpRegistry::new();
    let x_data: Vec<f32> = (0..18).map(|v| v as f32).collect();
    let x = Tensor::from_slice(&x_data).try_reshape(&[1, 2, 3, 3]).unwrap();
    let inputs = vec![Some(x)];
    let node = NodeProto::default();

    let result = registry.dispatch_multi("GlobalMaxPool", "", &inputs, &node, i64::MAX).unwrap();
    let s = result[0].shape().unwrap();
    assert_eq!(s.iter().map(|d| d.as_const().unwrap()).collect::<Vec<_>>(), vec![1, 2, 1, 1]);
    let arr = result[0].to_ndarray::<f32>().unwrap();
    assert!((arr[[0, 0, 0, 0]] - 8.0).abs() < 1e-4);
    assert!((arr[[0, 1, 0, 0]] - 17.0).abs() < 1e-4);
}

#[test]
fn test_layer_norm() {
    let registry = OpRegistry::new();
    let x = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).try_reshape(&[2, 4]).unwrap();
    let scale = Tensor::from_slice([1.0f32; 4]);
    let bias = Tensor::from_slice([0.0f32; 4]);
    let inputs = vec![Some(x), Some(scale), Some(bias)];
    let mut node = NodeProto::default();
    node.attribute.push(make_attr_int("axis", -1));

    let result = registry.dispatch_multi("LayerNormalization", "", &inputs, &node, i64::MAX).unwrap();
    assert_eq!(result.len(), 3);
    let arr = result[0].to_ndarray::<f32>().unwrap();
    assert_eq!(arr.shape(), &[2, 4]);
    for row in 0..2 {
        let row_data: Vec<f32> = (0..4).map(|c| arr[[row, c]]).collect();
        let mean: f32 = row_data.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-4, "row {row} mean should be ~0, got {mean}");
    }
}

#[test]
fn test_group_norm() {
    let registry = OpRegistry::new();
    let x_data: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let x = Tensor::from_slice(&x_data).try_reshape(&[1, 4, 2, 2]).unwrap();
    let scale = Tensor::from_slice([1.0f32; 4]);
    let bias = Tensor::from_slice([0.0f32; 4]);
    let inputs = vec![Some(x), Some(scale), Some(bias)];
    let mut node = NodeProto::default();
    node.attribute.push(make_attr_int("num_groups", 2));

    let result = registry.dispatch_multi("GroupNormalization", "", &inputs, &node, i64::MAX).unwrap();
    let arr = result[0].to_ndarray::<f32>().unwrap();
    assert_eq!(arr.shape(), &[1, 4, 2, 2]);
}

#[test]
fn test_instance_norm() {
    let registry = OpRegistry::new();
    let x_data: Vec<f32> = (0..18).map(|v| v as f32).collect();
    let x = Tensor::from_slice(&x_data).try_reshape(&[1, 2, 3, 3]).unwrap();
    let scale = Tensor::from_slice([1.0f32; 2]);
    let bias = Tensor::from_slice([0.0f32; 2]);
    let inputs = vec![Some(x), Some(scale), Some(bias)];
    let node = NodeProto::default();

    let result = registry.dispatch_multi("InstanceNormalization", "", &inputs, &node, i64::MAX).unwrap();
    let arr = result[0].to_ndarray::<f32>().unwrap();
    assert_eq!(arr.shape(), &[1, 2, 3, 3]);
}
