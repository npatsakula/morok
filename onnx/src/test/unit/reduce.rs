use crate::test::helpers::*;

#[test]
fn test_registry_reduce_sum() {
    let registry = OpRegistry::new();
    let x = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]).try_reshape(&[2, 2]).unwrap();
    let node = NodeProto::default();

    let result = registry.dispatch("ReduceSum", "", &[x], &node);
    let result = result.unwrap().realize().unwrap();
    assert!(result.buffer().is_some());
}

#[test]
fn test_registry_reduce_max() {
    let registry = OpRegistry::new();
    let x = Tensor::from_slice([1.0f32, 4.0, 2.0, 3.0]).try_reshape(&[2, 2]).unwrap();
    let node = NodeProto::default();

    let result = registry.dispatch("ReduceMax", "", &[x], &node).unwrap().realize().unwrap();
    assert!(result.buffer().is_some());
}

#[test]
fn test_registry_reduce_with_keepdims() {
    let registry = OpRegistry::new();
    let x = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]).try_reshape(&[2, 2]).unwrap();

    let mut node = NodeProto::default();
    node.attribute.push(make_attr_ints("axes", &[1]));
    node.attribute.push(make_attr_int("keepdims", 1));

    // Use opset 12 so axes come from attributes (opset >=13 reads axes from input[1])
    let inputs = vec![Some(x)];
    let result = registry.dispatch_multi("ReduceSum", "", &inputs, &node, 12).unwrap();
    let result = result[0].clone().realize().unwrap();
    assert!(result.buffer().is_some());
}

#[test]
fn test_reduce_log_sum_exp() {
    let registry = OpRegistry::new();
    let x = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]).try_reshape(&[2, 2]).unwrap();

    let mut node = NodeProto::default();
    node.attribute.push(make_attr_ints("axes", &[1]));
    node.attribute.push(make_attr_int("keepdims", 1));

    // Use opset 12 so axes come from attributes
    let inputs = vec![Some(x)];
    let result = registry.dispatch_multi("ReduceLogSumExp", "", &inputs, &node, 12).unwrap();
    let arr = result[0].to_ndarray::<f32>().unwrap();
    let vals: Vec<f32> = arr.iter().copied().collect();
    // log(exp(1)+exp(2)) ~ 2.3133, log(exp(3)+exp(4)) ~ 4.3133
    assert!((vals[0] - 2.3133).abs() < 0.01, "got {}", vals[0]);
    assert!((vals[1] - 4.3133).abs() < 0.01, "got {}", vals[1]);
}

#[test]
fn test_argmax_select_last() {
    let registry = OpRegistry::new();
    let x = Tensor::from_slice([1.0f32, 3.0, 2.0, 3.0]);
    let mut node = NodeProto::default();
    node.attribute.push(make_attr_int("select_last_index", 1));

    let result = registry.dispatch("ArgMax", "", &[x], &node).unwrap();
    let arr = result.to_ndarray::<i64>().unwrap();
    let vals: Vec<i64> = arr.iter().copied().collect();
    assert_eq!(vals, vec![3], "ArgMax select_last should return 3");
}

#[test]
fn test_argmin_select_last() {
    let registry = OpRegistry::new();
    let x = Tensor::from_slice([3.0f32, 1.0, 2.0, 1.0]);
    let mut node = NodeProto::default();
    node.attribute.push(make_attr_int("select_last_index", 1));

    let result = registry.dispatch("ArgMin", "", &[x], &node).unwrap();
    let arr = result.to_ndarray::<i64>().unwrap();
    let vals: Vec<i64> = arr.iter().copied().collect();
    assert_eq!(vals, vec![3], "ArgMin select_last should return 3");
}

#[test]
fn test_argmax_cast_int64() {
    let registry = OpRegistry::new();
    let x = Tensor::from_slice([1.0f32, 3.0, 2.0]);
    let node = NodeProto::default();

    let result = registry.dispatch("ArgMax", "", &[x], &node).unwrap();
    assert_eq!(result.uop().dtype(), DType::Int64, "ArgMax should always return Int64");
}
