use crate::test::helpers::*;
use ndarray::{Array4, array};

morok_tensor::codegen_tests! {
    fn test_registry_matmul(config) {
        let registry = OpRegistry::new();
        let a = Tensor::from_ndarray(&array![[1.0f32, 2.0], [3.0, 4.0]]);
        let b = Tensor::from_ndarray(&array![[5.0f32, 6.0], [7.0, 8.0]]);
        let node = NodeProto::default();

        let result = registry.dispatch("MatMul", "", &[a, b], &node);
        let result = result.unwrap().realize_with(&config).unwrap();
        assert!(result.buffer().is_some());
    }

    fn test_conv_basic(config) {
        let registry = OpRegistry::new();
        let x = Tensor::from_ndarray(&Array4::from_shape_vec((1, 1, 4, 4), (0..16).map(|v| v as f32).collect()).unwrap());
        let w = Tensor::from_ndarray(&Array4::from_elem((1, 1, 3, 3), 1.0f32));
        let inputs = vec![Some(x), Some(w)];
        let mut node = NodeProto::default();
        node.attribute.push(make_attr_ints("kernel_shape", &[3, 3]));

        let result = registry.dispatch_multi("Conv", "", &inputs, &node, i64::MAX).unwrap();
        let result0 = result[0].contiguous().realize_with(&config).unwrap();
        let dims: Vec<usize> = result0.shape().unwrap().iter().map(|d| d.as_const().unwrap()).collect();
        assert_eq!(dims, [1, 1, 2, 2]);
        let view = result0.array_view::<f32>().unwrap();
        assert!((view[[0, 0, 0, 0]] - 45.0).abs() < 1e-4);
    }

    fn test_conv_transpose(config) {
        let registry = OpRegistry::new();
        let x = Tensor::from_ndarray(&Array4::from_shape_vec((1, 1, 2, 2), vec![1.0f32, 2.0, 3.0, 4.0]).unwrap());
        let w = Tensor::from_ndarray(&Array4::from_shape_vec((1, 1, 1, 1), vec![2.0f32]).unwrap());
        let inputs = vec![Some(x), Some(w)];
        let mut node = NodeProto::default();
        node.attribute.push(make_attr_ints("kernel_shape", &[1, 1]));

        let result = registry.dispatch_multi("ConvTranspose", "", &inputs, &node, i64::MAX).unwrap();
        let result0 = result[0].contiguous().realize_with(&config).unwrap();
        let dims: Vec<usize> = result0.shape().unwrap().iter().map(|d| d.as_const().unwrap()).collect();
        assert_eq!(dims, [1, 1, 2, 2]);
        let view = result0.array_view::<f32>().unwrap();
        assert!((view[[0, 0, 0, 0]] - 2.0).abs() < 1e-4);
        assert!((view[[0, 0, 1, 1]] - 8.0).abs() < 1e-4);
    }

    fn test_average_pool(config) {
        let registry = OpRegistry::new();
        let x = Tensor::from_ndarray(&Array4::from_shape_vec((1, 1, 4, 4), (0..16).map(|v| v as f32).collect()).unwrap());
        let inputs = vec![Some(x)];
        let mut node = NodeProto::default();
        node.attribute.push(make_attr_ints("kernel_shape", &[2, 2]));
        node.attribute.push(make_attr_ints("strides", &[2, 2]));

        let result = registry.dispatch_multi("AveragePool", "", &inputs, &node, i64::MAX).unwrap();
        let result0 = result[0].contiguous().realize_with(&config).unwrap();
        let dims: Vec<usize> = result0.shape().unwrap().iter().map(|d| d.as_const().unwrap()).collect();
        assert_eq!(dims, [1, 1, 2, 2]);
        let view = result0.array_view::<f32>().unwrap();
        assert!((view[[0, 0, 0, 0]] - 2.5).abs() < 1e-4);
    }

    fn test_max_pool(config) {
        let registry = OpRegistry::new();
        let x = Tensor::from_ndarray(&Array4::from_shape_vec((1, 1, 4, 4), (0..16).map(|v| v as f32).collect()).unwrap());
        let inputs = vec![Some(x)];
        let mut node = NodeProto::default();
        node.attribute.push(make_attr_ints("kernel_shape", &[2, 2]));
        node.attribute.push(make_attr_ints("strides", &[2, 2]));

        let result = registry.dispatch_multi("MaxPool", "", &inputs, &node, i64::MAX).unwrap();
        assert_eq!(result.len(), 2);
        let result0 = result[0].contiguous().realize_with(&config).unwrap();
        let dims: Vec<usize> = result0.shape().unwrap().iter().map(|d| d.as_const().unwrap()).collect();
        assert_eq!(dims, [1, 1, 2, 2]);
        let view = result0.array_view::<f32>().unwrap();
        assert!((view[[0, 0, 0, 0]] - 5.0).abs() < 1e-4);
    }

    fn test_max_pool_indices(config) {
        let registry = OpRegistry::new();
        let x = Tensor::from_ndarray(&Array4::from_shape_vec((1, 1, 4, 4), (0..16).map(|v| v as f32).collect()).unwrap());
        let inputs = vec![Some(x)];
        let mut node = NodeProto::default();
        node.attribute.push(make_attr_ints("kernel_shape", &[2, 2]));
        node.attribute.push(make_attr_ints("strides", &[2, 2]));

        let result = registry.dispatch_multi("MaxPool", "", &inputs, &node, i64::MAX).unwrap();
        assert_eq!(result.len(), 2);
        let result1 = result[1].contiguous().realize_with(&config).unwrap();
        let dims: Vec<usize> = result1.shape().unwrap().iter().map(|d| d.as_const().unwrap()).collect();
        assert_eq!(dims, [1, 1, 2, 2]);
        let view = result1.array_view::<i64>().unwrap();
        assert_eq!(view[[0, 0, 0, 0]], 5);
    }

    fn test_global_average_pool(config) {
        let registry = OpRegistry::new();
        let x = Tensor::from_ndarray(&Array4::from_shape_vec((1, 2, 3, 3), (0..18).map(|v| v as f32).collect()).unwrap());
        let inputs = vec![Some(x)];
        let node = NodeProto::default();

        let result = registry.dispatch_multi("GlobalAveragePool", "", &inputs, &node, i64::MAX).unwrap();
        let result0 = result[0].contiguous().realize_with(&config).unwrap();
        let dims: Vec<usize> = result0.shape().unwrap().iter().map(|d| d.as_const().unwrap()).collect();
        assert_eq!(dims, [1, 2, 1, 1]);
        let view = result0.array_view::<f32>().unwrap();
        assert!((view[[0, 0, 0, 0]] - 4.0).abs() < 1e-4);
        assert!((view[[0, 1, 0, 0]] - 13.0).abs() < 1e-4);
    }

    fn test_global_max_pool(config) {
        let registry = OpRegistry::new();
        let x = Tensor::from_ndarray(&Array4::from_shape_vec((1, 2, 3, 3), (0..18).map(|v| v as f32).collect()).unwrap());
        let inputs = vec![Some(x)];
        let node = NodeProto::default();

        let result = registry.dispatch_multi("GlobalMaxPool", "", &inputs, &node, i64::MAX).unwrap();
        let result0 = result[0].contiguous().realize_with(&config).unwrap();
        let dims: Vec<usize> = result0.shape().unwrap().iter().map(|d| d.as_const().unwrap()).collect();
        assert_eq!(dims, [1, 2, 1, 1]);
        let view = result0.array_view::<f32>().unwrap();
        assert!((view[[0, 0, 0, 0]] - 8.0).abs() < 1e-4);
        assert!((view[[0, 1, 0, 0]] - 17.0).abs() < 1e-4);
    }

    fn test_layer_norm(config) {
        let registry = OpRegistry::new();
        let x = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]);
        let scale = Tensor::from_slice([1.0f32; 4]);
        let bias = Tensor::from_slice([0.0f32; 4]);
        let inputs = vec![Some(x), Some(scale), Some(bias)];
        let mut node = NodeProto::default();
        node.attribute.push(make_attr_int("axis", -1));

        let result = registry.dispatch_multi("LayerNormalization", "", &inputs, &node, i64::MAX).unwrap();
        assert_eq!(result.len(), 3);
        let result0 = result[0].contiguous().realize_with(&config).unwrap();
        let dims: Vec<usize> = result0.shape().unwrap().iter().map(|d| d.as_const().unwrap()).collect();
        assert_eq!(dims, [2, 4]);
        let view = result0.array_view::<f32>().unwrap();
        for row in 0..2 {
            let row_data: Vec<f32> = (0..4).map(|c| view[[row, c]]).collect();
            let mean: f32 = row_data.iter().sum::<f32>() / 4.0;
            assert!(mean.abs() < 1e-4, "row {row} mean should be ~0, got {mean}");
        }
    }
}

#[test]
fn test_conv_auto_pad_same_upper() {
    let registry = OpRegistry::new();
    let x = Tensor::from_ndarray(&Array4::from_elem((1, 1, 3, 3), 1.0f32));
    let w = Tensor::from_ndarray(&Array4::from_elem((1, 1, 3, 3), 1.0f32));
    let inputs = vec![Some(x), Some(w)];
    let mut node = NodeProto::default();
    node.attribute.push(make_attr_ints("kernel_shape", &[3, 3]));
    node.attribute.push(make_attr_string("auto_pad", "SAME_UPPER"));

    let result = registry.dispatch_multi("Conv", "", &inputs, &node, i64::MAX).unwrap();
    let s = result[0].shape().unwrap();
    assert_eq!(s.iter().map(|d| d.as_const().unwrap()).collect::<Vec<_>>(), vec![1, 1, 3, 3]);
}

#[test]
fn test_average_pool_ceil() {
    let registry = OpRegistry::new();
    let x = Tensor::from_ndarray(&Array4::<f32>::zeros((1, 1, 7, 7)));
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
fn test_group_norm() {
    let registry = OpRegistry::new();
    let x = Tensor::from_ndarray(&Array4::from_shape_vec((1, 4, 2, 2), (0..16).map(|v| v as f32).collect()).unwrap());
    let scale = Tensor::from_slice([1.0f32; 4]);
    let bias = Tensor::from_slice([0.0f32; 4]);
    let inputs = vec![Some(x), Some(scale), Some(bias)];
    let mut node = NodeProto::default();
    node.attribute.push(make_attr_int("num_groups", 2));

    let result = registry.dispatch_multi("GroupNormalization", "", &inputs, &node, i64::MAX).unwrap();
    let dims: Vec<usize> = result[0].shape().unwrap().iter().map(|d| d.as_const().unwrap()).collect();
    assert_eq!(dims, [1, 4, 2, 2]);
}

#[test]
fn test_instance_norm() {
    let registry = OpRegistry::new();
    let x = Tensor::from_ndarray(&Array4::from_shape_vec((1, 2, 3, 3), (0..18).map(|v| v as f32).collect()).unwrap());
    let scale = Tensor::from_slice([1.0f32; 2]);
    let bias = Tensor::from_slice([0.0f32; 2]);
    let inputs = vec![Some(x), Some(scale), Some(bias)];
    let node = NodeProto::default();

    let result = registry.dispatch_multi("InstanceNormalization", "", &inputs, &node, i64::MAX).unwrap();
    let dims: Vec<usize> = result[0].shape().unwrap().iter().map(|d| d.as_const().unwrap()).collect();
    assert_eq!(dims, [1, 2, 3, 3]);
}
