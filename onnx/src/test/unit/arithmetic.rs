use crate::test::helpers::*;

#[test]
fn test_registry_add() {
    let registry = OpRegistry::new();
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let b = Tensor::from_slice([4.0f32, 5.0, 6.0]);
    let node = NodeProto::default();

    let result = registry.dispatch("Add", "", &[a, b], &node);
    let result = result.unwrap().realize().unwrap();
    assert!(result.buffer().is_some());
}

#[test]
fn test_registry_abs() {
    let registry = OpRegistry::new();
    let x = Tensor::from_slice([-2.0f32, -1.0, 0.0, 1.0, 2.0]);
    let node = NodeProto::default();

    let result = registry.dispatch("Abs", "", &[x], &node).unwrap().realize().unwrap();
    assert!(result.buffer().is_some());
}

#[test]
fn test_registry_equal() {
    let registry = OpRegistry::new();
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let b = Tensor::from_slice([1.0f32, 0.0, 3.0]);
    let node = NodeProto::default();

    let result = registry.dispatch("Equal", "", &[a, b], &node).unwrap().realize().unwrap();
    assert!(result.buffer().is_some());
}

#[test]
fn test_registry_where() {
    let registry = OpRegistry::new();
    let condition = Tensor::from_slice([true, false, true]);
    let x = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let y = Tensor::from_slice([10.0f32, 20.0, 30.0]);
    let node = NodeProto::default();

    let result = registry.dispatch("Where", "", &[condition, x, y], &node).unwrap().realize().unwrap();
    assert!(result.buffer().is_some());
}

#[test]
fn test_registry_math_ops() {
    let registry = OpRegistry::new();
    let x = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let node = NodeProto::default();

    for op in ["Exp", "Log", "Ceil", "Floor", "Round", "Sign", "Reciprocal", "Sin", "Cos", "Tan"] {
        let result = registry.dispatch(op, "", std::slice::from_ref(&x), &node);
        assert!(result.is_ok(), "Operator {op} failed: {:?}", result.err());
    }
}

#[test]
fn test_registry_comparison_ops() {
    let registry = OpRegistry::new();
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let b = Tensor::from_slice([2.0f32, 2.0, 1.0]);
    let node = NodeProto::default();

    for op in ["Less", "LessOrEqual", "Greater", "GreaterOrEqual"] {
        let result = registry.dispatch(op, "", &[a.clone(), b.clone()], &node);
        assert!(result.is_ok(), "Operator {op} failed: {:?}", result.err());
    }
}

#[test]
fn test_max_variadic_3_inputs() {
    let registry = OpRegistry::new();
    let a = Tensor::from_slice([1.0f32, 5.0]);
    let b = Tensor::from_slice([3.0f32, 2.0]);
    let c = Tensor::from_slice([2.0f32, 4.0]);
    let inputs = vec![Some(a), Some(b), Some(c)];
    let node = NodeProto::default();

    let result = registry.dispatch_multi("Max", "", &inputs, &node, i64::MAX).unwrap();
    let arr = result[0].to_ndarray::<f32>().unwrap();
    let vals: Vec<f32> = arr.iter().copied().collect();
    assert_eq!(vals, vec![3.0, 5.0]);
}

#[test]
fn test_max_single_input() {
    let registry = OpRegistry::new();
    let a = Tensor::from_slice([7.0f32, 3.0]);
    let inputs = vec![Some(a)];
    let node = NodeProto::default();

    let result = registry.dispatch_multi("Max", "", &inputs, &node, i64::MAX).unwrap();
    let arr = result[0].to_ndarray::<f32>().unwrap();
    let vals: Vec<f32> = arr.iter().copied().collect();
    assert_eq!(vals, vec![7.0, 3.0]);
}

#[test]
fn test_min_variadic_3_inputs() {
    let registry = OpRegistry::new();
    let a = Tensor::from_slice([3.0f32, 1.0]);
    let b = Tensor::from_slice([1.0f32, 5.0]);
    let c = Tensor::from_slice([2.0f32, 3.0]);
    let inputs = vec![Some(a), Some(b), Some(c)];
    let node = NodeProto::default();

    let result = registry.dispatch_multi("Min", "", &inputs, &node, i64::MAX).unwrap();
    let arr = result[0].to_ndarray::<f32>().unwrap();
    let vals: Vec<f32> = arr.iter().copied().collect();
    assert_eq!(vals, vec![1.0, 1.0]);
}

#[test]
fn test_range_float() {
    let registry = OpRegistry::new();
    let start = Tensor::from_slice([0.0f32]);
    let limit = Tensor::from_slice([5.5f32]);
    let delta = Tensor::from_slice([1.5f32]);
    let node = NodeProto::default();

    let result = registry.dispatch("Range", "", &[start, limit, delta], &node).unwrap();
    let arr = result.to_ndarray::<f32>().unwrap();
    let vals: Vec<f32> = arr.iter().copied().collect();
    assert_eq!(vals, vec![0.0, 1.5, 3.0, 4.5]);
}

#[test]
fn test_range_integer_regression() {
    let registry = OpRegistry::new();
    let start = Tensor::from_slice([0i32]);
    let limit = Tensor::from_slice([5i32]);
    let delta = Tensor::from_slice([1i32]);
    let node = NodeProto::default();

    let result = registry.dispatch("Range", "", &[start, limit, delta], &node).unwrap();
    let arr = result.to_ndarray::<i32>().unwrap();
    let vals: Vec<i32> = arr.iter().copied().collect();
    assert_eq!(vals, vec![0, 1, 2, 3, 4]);
}

#[test]
fn test_cast_fallback() {
    let registry = OpRegistry::new();
    let x = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let mut node = NodeProto::default();
    node.attribute.push(make_attr_int("to", 999)); // invalid dtype code

    // Should not crash — falls back to Float32
    let result = registry.dispatch("Cast", "", &[x], &node);
    assert!(result.is_ok(), "Cast with invalid dtype should fallback, not crash");
}

#[test]
fn test_and() {
    let registry = OpRegistry::new();
    let a = Tensor::from_slice([true, true, false, false]);
    let b = Tensor::from_slice([true, false, true, false]);
    let node = NodeProto::default();

    let result = registry.dispatch("And", "", &[a, b], &node).unwrap().realize().unwrap();
    let arr = result.to_ndarray::<bool>().unwrap();
    let vals: Vec<bool> = arr.iter().copied().collect();
    assert_eq!(vals, vec![true, false, false, false]);
}

#[test]
fn test_or() {
    let registry = OpRegistry::new();
    let a = Tensor::from_slice([true, true, false, false]);
    let b = Tensor::from_slice([true, false, true, false]);
    let node = NodeProto::default();

    let result = registry.dispatch("Or", "", &[a, b], &node).unwrap().realize().unwrap();
    let arr = result.to_ndarray::<bool>().unwrap();
    let vals: Vec<bool> = arr.iter().copied().collect();
    assert_eq!(vals, vec![true, true, true, false]);
}

#[test]
fn test_xor() {
    let registry = OpRegistry::new();
    let a = Tensor::from_slice([true, true, false, false]);
    let b = Tensor::from_slice([true, false, true, false]);
    let node = NodeProto::default();

    let result = registry.dispatch("Xor", "", &[a, b], &node).unwrap().realize().unwrap();
    let arr = result.to_ndarray::<bool>().unwrap();
    let vals: Vec<bool> = arr.iter().copied().collect();
    assert_eq!(vals, vec![false, true, true, false]);
}

#[test]
fn test_isnan() {
    let registry = OpRegistry::new();
    let x = Tensor::from_slice([1.0f32, f32::NAN, 3.0]);
    let node = NodeProto::default();

    let result = registry.dispatch("IsNaN", "", &[x], &node).unwrap().realize().unwrap();
    let arr = result.to_ndarray::<bool>().unwrap();
    let vals: Vec<bool> = arr.iter().copied().collect();
    assert_eq!(vals, vec![false, true, false]);
}

#[test]
fn test_isinf() {
    let registry = OpRegistry::new();
    let x = Tensor::from_slice([1.0f32, f32::INFINITY, f32::NEG_INFINITY]);
    let node = NodeProto::default();

    let result = registry.dispatch("IsInf", "", &[x], &node).unwrap().realize().unwrap();
    assert!(result.buffer().is_some());
}

#[test]
fn test_shrink() {
    let registry = OpRegistry::new();
    let x = Tensor::from_slice([-2.0f32, -0.3, 0.0, 0.3, 2.0]);
    let node = NodeProto::default();

    let result = registry.dispatch("Shrink", "", &[x], &node).unwrap().realize().unwrap();
    let arr = result.to_ndarray::<f32>().unwrap();
    let vals: Vec<f32> = arr.iter().copied().collect();
    let expected = [-2.0f32, 0.0, 0.0, 0.0, 2.0];
    for (a, b) in vals.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-4, "expected {b}, got {a}");
    }
}
