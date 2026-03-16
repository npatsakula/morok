use crate::test::helpers::*;

morok_tensor::codegen_tests! {
    fn test_constant_value_float(config) {
        let registry = OpRegistry::new();
        let node = NodeProto {
            op_type: "Constant".to_string(),
            attribute: vec![make_attr_float("value_float", 3.125)],
            ..Default::default()
        };

        let result = registry.dispatch("Constant", "", &[], &node).unwrap();
        let realized = result.contiguous().realize_with(&config).unwrap();
        assert!(realized.buffer().is_some());
    }

    fn test_constant_value_floats(config) {
        let registry = OpRegistry::new();
        let node = NodeProto {
            op_type: "Constant".to_string(),
            attribute: vec![make_attr_floats("value_floats", &[1.0, 2.0, 3.0])],
            ..Default::default()
        };

        let result = registry.dispatch("Constant", "", &[], &node).unwrap();
        let realized = result.realize_with(&config).unwrap();
        assert!(realized.buffer().is_some());
    }

    fn test_constant_value_int(config) {
        let registry = OpRegistry::new();
        let node = NodeProto {
            op_type: "Constant".to_string(),
            attribute: vec![make_attr_int("value_int", 42)],
            ..Default::default()
        };

        let result = registry.dispatch("Constant", "", &[], &node).unwrap();
        let realized = result.contiguous().realize_with(&config).unwrap();
        assert!(realized.buffer().is_some());
    }

    fn test_constant_value_ints(config) {
        let registry = OpRegistry::new();
        let node = NodeProto {
            op_type: "Constant".to_string(),
            attribute: vec![make_attr_ints("value_ints", &[0, 1, 2, 3])],
            ..Default::default()
        };

        let result = registry.dispatch("Constant", "", &[], &node).unwrap();
        let realized = result.realize_with(&config).unwrap();
        assert!(realized.buffer().is_some());
    }

    fn test_constant_value_tensor(config) {
        let registry = OpRegistry::new();
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let tp = make_tensor_proto(raw, vec![2, 2], 1); // FLOAT
        let node = NodeProto {
            op_type: "Constant".to_string(),
            attribute: vec![make_attr_tensor("value", tp)],
            ..Default::default()
        };

        let result = registry.dispatch("Constant", "", &[], &node).unwrap();
        let realized = result.realize_with(&config).unwrap();
        assert!(realized.buffer().is_some());
    }
}
