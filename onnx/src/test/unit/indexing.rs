use crate::test::helpers::*;
use ndarray::array;

morok_tensor::codegen_tests! {
    fn test_registry_gather(config) {
        let registry = OpRegistry::new();
        let data = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0]);
        let indices = Tensor::from_slice([0i64, 2, 4]);
        let node = NodeProto::default();

        let result = registry.dispatch("Gather", "", &[data, indices], &node).unwrap().realize_with(&config).unwrap();
        assert!(result.buffer().is_some());
    }

    fn test_registry_gather_axis1(config) {
        let registry = OpRegistry::new();
        let data = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let indices = Tensor::from_ndarray(&array![[0i64, 2], [1, 0]]);
        let mut node = NodeProto::default();
        node.attribute.push(make_attr_int("axis", 1));

        let result = registry.dispatch("Gather", "", &[data, indices], &node).unwrap().realize_with(&config).unwrap();
        assert!(result.buffer().is_some());
    }

    fn test_gather_negative_indices(config) {
        let registry = OpRegistry::new();
        let data = Tensor::from_slice([10.0f32, 20.0, 30.0, 40.0, 50.0]);
        let indices = Tensor::from_slice([0i64, -1, 2, -2]);
        let node = NodeProto::default();

        let result = registry.dispatch("Gather", "", &[data, indices], &node).unwrap();
        assert_eq!(result.realize_with(&config).unwrap().to_vec::<f32>().unwrap(), vec![10.0, 50.0, 30.0, 40.0]);
    }

    fn test_gather_elements(config) {
        let registry = OpRegistry::new();
        let data = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        let indices = Tensor::from_ndarray(&array![[1i64, 2, 0], [2, 0, 0], [0, 1, 1]]);
        let mut node = NodeProto::default();
        node.attribute.push(make_attr_int("axis", 1));
        let inputs = vec![Some(data), Some(indices)];

        let result = registry.dispatch_multi("GatherElements", "", &inputs, &node, i64::MAX).unwrap();
        assert_eq!(result[0].clone().realize_with(&config).unwrap().to_vec::<f32>().unwrap(), vec![2.0, 3.0, 1.0, 6.0, 4.0, 4.0, 7.0, 8.0, 8.0]);
    }

    fn test_gather_elements_negative_indices(config) {
        let registry = OpRegistry::new();
        let data = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let indices = Tensor::from_ndarray(&array![[-1i64], [0]]);
        let mut node = NodeProto::default();
        node.attribute.push(make_attr_int("axis", 1));
        let inputs = vec![Some(data), Some(indices)];

        let result = registry.dispatch_multi("GatherElements", "", &inputs, &node, i64::MAX).unwrap();
        assert_eq!(result[0].clone().realize_with(&config).unwrap().to_vec::<f32>().unwrap(), vec![3.0, 4.0]);
    }

    fn test_trilu_upper(config) {
        let registry = OpRegistry::new();
        let x = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        let inputs = vec![Some(x)];
        let mut node = NodeProto::default();
        node.attribute.push(make_attr_int("upper", 1));

        let result = registry.dispatch_multi("Trilu", "", &inputs, &node, i64::MAX).unwrap();
        assert_eq!(result[0].clone().realize_with(&config).unwrap().to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0]);
    }

    fn test_trilu_lower(config) {
        let registry = OpRegistry::new();
        let x = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        let inputs = vec![Some(x)];
        let mut node = NodeProto::default();
        node.attribute.push(make_attr_int("upper", 0));

        let result = registry.dispatch_multi("Trilu", "", &inputs, &node, i64::MAX).unwrap();
        assert_eq!(result[0].clone().realize_with(&config).unwrap().to_vec::<f32>().unwrap(), vec![1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0]);
    }

    fn test_trilu_with_k(config) {
        let registry = OpRegistry::new();
        let x = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        let k = Tensor::from_slice([1i64]);
        let inputs = vec![Some(x), Some(k)];
        let node = NodeProto::default(); // upper=1 by default

        let result = registry.dispatch_multi("Trilu", "", &inputs, &node, i64::MAX).unwrap();
        assert_eq!(result[0].clone().realize_with(&config).unwrap().to_vec::<f32>().unwrap(), vec![0.0, 2.0, 3.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0]);
    }

    fn test_cumsum(config) {
        let registry = OpRegistry::new();
        let x = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);
        let axis = Tensor::from_slice([0i64]);
        let inputs = vec![Some(x), Some(axis)];
        let node = NodeProto::default();

        let result = registry.dispatch_multi("CumSum", "", &inputs, &node, i64::MAX).unwrap();
        assert_eq!(result[0].clone().realize_with(&config).unwrap().to_vec::<f32>().unwrap(), vec![1.0, 3.0, 6.0, 10.0]);
    }

    fn test_cumsum_exclusive_reverse(config) {
        let registry = OpRegistry::new();
        let x = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);
        let axis = Tensor::from_slice([0i64]);
        let inputs = vec![Some(x), Some(axis)];
        let mut node = NodeProto::default();
        node.attribute.push(make_attr_int("exclusive", 1));
        node.attribute.push(make_attr_int("reverse", 1));

        let result = registry.dispatch_multi("CumSum", "", &inputs, &node, i64::MAX).unwrap();
        assert_eq!(result[0].clone().realize_with(&config).unwrap().to_vec::<f32>().unwrap(), vec![9.0, 7.0, 4.0, 0.0]);
    }

    fn test_one_hot(config) {
        let registry = OpRegistry::new();
        let indices = Tensor::from_slice([0i64, 1, 2]);
        let depth = Tensor::from_slice([3i64]);
        let values = Tensor::from_slice([0.0f32, 1.0]); // off=0, on=1
        let inputs = vec![Some(indices), Some(depth), Some(values)];
        let mut node = NodeProto::default();
        node.attribute.push(make_attr_int("axis", -1));

        let result = registry.dispatch_multi("OneHot", "", &inputs, &node, i64::MAX).unwrap();
        assert_eq!(result[0].clone().realize_with(&config).unwrap().to_vec::<f32>().unwrap(), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }
}
