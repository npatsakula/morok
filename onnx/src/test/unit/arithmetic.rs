use crate::test::helpers::*;

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
fn test_cast_fallback() {
    let registry = OpRegistry::new();
    let x = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let mut node = NodeProto::default();
    node.attribute.push(make_attr_int("to", 999)); // invalid dtype code

    // Should not crash — falls back to Float32
    let result = registry.dispatch("Cast", "", &[x], &node);
    assert!(result.is_ok(), "Cast with invalid dtype should fallback, not crash");
}

svod_tensor::codegen_tests! {
    fn test_registry_add(config) {
        let registry = OpRegistry::new();
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let b = Tensor::from_slice([4.0f32, 5.0, 6.0]);
        let node = NodeProto::default();

        let result = registry.dispatch("Add", "", &[a, b], &node).unwrap();
        result.realize_with(&config).unwrap();
        assert!(result.buffer().is_some());
    }

    fn test_registry_abs(config) {
        let registry = OpRegistry::new();
        let x = Tensor::from_slice([-2.0f32, -1.0, 0.0, 1.0, 2.0]);
        let node = NodeProto::default();

        let result = registry.dispatch("Abs", "", &[x], &node).unwrap();
        result.realize_with(&config).unwrap();
        assert!(result.buffer().is_some());
    }

    fn test_registry_equal(config) {
        let registry = OpRegistry::new();
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let b = Tensor::from_slice([1.0f32, 0.0, 3.0]);
        let node = NodeProto::default();

        let result = registry.dispatch("Equal", "", &[a, b], &node).unwrap();
        result.realize_with(&config).unwrap();
        assert!(result.buffer().is_some());
    }

    fn test_registry_where(config) {
        let registry = OpRegistry::new();
        let condition = Tensor::from_slice([true, false, true]);
        let x = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let y = Tensor::from_slice([10.0f32, 20.0, 30.0]);
        let node = NodeProto::default();

        let result = registry.dispatch("Where", "", &[condition, x, y], &node).unwrap();
        result.realize_with(&config).unwrap();
        assert!(result.buffer().is_some());
    }

    fn test_max_variadic_3_inputs(config) {
        let registry = OpRegistry::new();
        let a = Tensor::from_slice([1.0f32, 5.0]);
        let b = Tensor::from_slice([3.0f32, 2.0]);
        let c = Tensor::from_slice([2.0f32, 4.0]);
        let inputs = vec![Some(a), Some(b), Some(c)];
        let node = NodeProto::default();

        let result = registry.dispatch_multi("Max", "", &inputs, &node, i64::MAX).unwrap();
        let r = result[0].clone();
        r.realize_with(&config).unwrap();
        let vals = r.as_vec::<f32>().unwrap();
        assert_eq!(vals, vec![3.0, 5.0]);
    }

    fn test_max_single_input(config) {
        let registry = OpRegistry::new();
        let a = Tensor::from_slice([7.0f32, 3.0]);
        let inputs = vec![Some(a)];
        let node = NodeProto::default();

        let result = registry.dispatch_multi("Max", "", &inputs, &node, i64::MAX).unwrap();
        let r = result[0].clone();
        r.realize_with(&config).unwrap();
        let vals = r.as_vec::<f32>().unwrap();
        assert_eq!(vals, vec![7.0, 3.0]);
    }

    fn test_min_variadic_3_inputs(config) {
        let registry = OpRegistry::new();
        let a = Tensor::from_slice([3.0f32, 1.0]);
        let b = Tensor::from_slice([1.0f32, 5.0]);
        let c = Tensor::from_slice([2.0f32, 3.0]);
        let inputs = vec![Some(a), Some(b), Some(c)];
        let node = NodeProto::default();

        let result = registry.dispatch_multi("Min", "", &inputs, &node, i64::MAX).unwrap();
        let r = result[0].clone();
        r.realize_with(&config).unwrap();
        let vals = r.as_vec::<f32>().unwrap();
        assert_eq!(vals, vec![1.0, 1.0]);
    }

    fn test_range_float(config) {
        let registry = OpRegistry::new();
        let start = Tensor::from_slice([0.0f32]);
        let limit = Tensor::from_slice([5.5f32]);
        let delta = Tensor::from_slice([1.5f32]);
        let node = NodeProto::default();

        let result = registry.dispatch("Range", "", &[start, limit, delta], &node).unwrap();
        result.realize_with(&config).unwrap();
        let vals = result.as_vec::<f32>().unwrap();
        assert_eq!(vals, vec![0.0, 1.5, 3.0, 4.5]);
    }

    fn test_range_integer_regression(config) {
        let registry = OpRegistry::new();
        let start = Tensor::from_slice([0i32]);
        let limit = Tensor::from_slice([5i32]);
        let delta = Tensor::from_slice([1i32]);
        let node = NodeProto::default();

        let result = registry.dispatch("Range", "", &[start, limit, delta], &node).unwrap();
        result.realize_with(&config).unwrap();
        let vals = result.as_vec::<i32>().unwrap();
        assert_eq!(vals, vec![0, 1, 2, 3, 4]);
    }

    fn test_mod_broadcasts_integer_operands(config) {
        // fmod=1 on ints used to bypass Tensor broadcasting by calling the raw UOp.
        let registry = OpRegistry::new();
        let xs = [3i32, -7, 5, 9, 11, -4, 6, 8, 2, 13, -1, 10];
        let ys = [2i32, 3, 4, 5];
        for (fmod, y_shape) in [(1i64, vec![4isize]), (1, vec![1, 4]), (0, vec![4])] {
            let x = Tensor::from_slice(xs).try_reshape([3isize, 4]).unwrap();
            let y = Tensor::from_slice(ys).try_reshape(y_shape).unwrap();
            let node = NodeProto { attribute: vec![make_attr_int("fmod", fmod)], ..Default::default() };

            let result = registry.dispatch("Mod", "", &[x, y], &node).unwrap();
            result.realize_with(&config).unwrap();
            assert_eq!(result.dims().unwrap(), vec![3, 4]);
            let expected = xs
                .iter()
                .enumerate()
                .map(|(i, x)| if fmod == 1 { x % ys[i % 4] } else { x.rem_euclid(ys[i % 4]) })
                .collect::<Vec<_>>();
            assert_eq!(result.as_vec::<i32>().unwrap(), expected, "fmod {fmod}");
        }

        // Rank-promoting broadcast: the result rank comes from the divisor.
        let node = NodeProto { attribute: vec![make_attr_int("fmod", 1)], ..Default::default() };
        let lhs = Tensor::from_slice(ys);
        let rhs = Tensor::from_slice(xs).try_reshape([3isize, 4]).unwrap();
        let result = registry.dispatch("Mod", "", &[lhs, rhs], &node).unwrap();
        result.realize_with(&config).unwrap();
        let expected = xs.iter().enumerate().map(|(i, x)| ys[i % 4] % x).collect::<Vec<_>>();
        assert_eq!(result.as_vec::<i32>().unwrap(), expected);
    }

    fn test_and(config) {
        let registry = OpRegistry::new();
        let a = Tensor::from_slice([true, true, false, false]);
        let b = Tensor::from_slice([true, false, true, false]);
        let node = NodeProto::default();

        let result = registry.dispatch("And", "", &[a, b], &node).unwrap();
        result.realize_with(&config).unwrap();
        let vals = result.as_vec::<bool>().unwrap();
        assert_eq!(vals, vec![true, false, false, false]);
    }

    fn test_or(config) {
        let registry = OpRegistry::new();
        let a = Tensor::from_slice([true, true, false, false]);
        let b = Tensor::from_slice([true, false, true, false]);
        let node = NodeProto::default();

        let result = registry.dispatch("Or", "", &[a, b], &node).unwrap();
        result.realize_with(&config).unwrap();
        let vals = result.as_vec::<bool>().unwrap();
        assert_eq!(vals, vec![true, true, true, false]);
    }

    fn test_xor(config) {
        let registry = OpRegistry::new();
        let a = Tensor::from_slice([true, true, false, false]);
        let b = Tensor::from_slice([true, false, true, false]);
        let node = NodeProto::default();

        let result = registry.dispatch("Xor", "", &[a, b], &node).unwrap();
        result.realize_with(&config).unwrap();
        let vals = result.as_vec::<bool>().unwrap();
        assert_eq!(vals, vec![false, true, true, false]);
    }

    fn test_isnan(config) {
        let registry = OpRegistry::new();
        let x = Tensor::from_slice([1.0f32, f32::NAN, 3.0]);
        let node = NodeProto::default();

        let result = registry.dispatch("IsNaN", "", &[x], &node).unwrap();
        result.realize_with(&config).unwrap();
        let vals = result.as_vec::<bool>().unwrap();
        assert_eq!(vals, vec![false, true, false]);
    }

    fn test_isinf(config) {
        let registry = OpRegistry::new();
        let x = Tensor::from_slice([1.0f32, f32::INFINITY, f32::NEG_INFINITY]);
        let node = NodeProto::default();

        let result = registry.dispatch("IsInf", "", &[x], &node).unwrap();
        result.realize_with(&config).unwrap();
        assert!(result.buffer().is_some());
    }

    fn test_shrink(config) {
        let registry = OpRegistry::new();
        let x = Tensor::from_slice([-2.0f32, -0.3, 0.0, 0.3, 2.0]);
        let node = NodeProto::default();

        let result = registry.dispatch("Shrink", "", &[x], &node).unwrap();
        result.realize_with(&config).unwrap();
        let vals = result.as_vec::<f32>().unwrap();
        let expected = [-2.0f32, 0.0, 0.0, 0.0, 2.0];
        for (a, b) in vals.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-4, "expected {b}, got {a}");
        }
    }
}
