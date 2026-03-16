use crate::test::helpers::*;
use ndarray::array;

#[test]
fn test_registry_log_softmax() {
    let registry = OpRegistry::new();
    let x = Tensor::from_ndarray(&array![[1.0f32, 2.0], [3.0, 4.0]]);
    let node = NodeProto::default();

    let result = registry.dispatch("LogSoftmax", "", &[x], &node);
    assert!(result.is_ok());
}

morok_tensor::codegen_tests! {
    fn test_registry_relu(config) {
        let registry = OpRegistry::new();
        let x = Tensor::from_slice([-2.0f32, -1.0, 0.0, 1.0, 2.0]);
        let node = NodeProto::default();

        let result = registry.dispatch("Relu", "", &[x], &node);
        let result = result.unwrap().realize_with(&config).unwrap();
        assert!(result.buffer().is_some());
    }

    fn test_registry_sigmoid(config) {
        let registry = OpRegistry::new();
        let x = Tensor::from_slice([-1.0f32, 0.0, 1.0]);
        let node = NodeProto::default();

        let result = registry.dispatch("Sigmoid", "", &[x], &node);
        let result = result.unwrap().realize_with(&config).unwrap();
        assert!(result.buffer().is_some());
    }

    fn test_gelu_exact(config) {
        let registry = OpRegistry::new();
        let x = Tensor::from_slice([0.0f32, 1.0, -1.0]);
        let mut node = NodeProto::default();
        node.attribute.push(make_attr_string("approximate", "none"));

        let result = registry.dispatch("Gelu", "", &[x], &node).unwrap();
        let vals = result.realize_with(&config).unwrap().to_vec::<f32>().unwrap();
        assert!((vals[0] - 0.0).abs() < 1e-4, "gelu(0) = {}", vals[0]);
        assert!((vals[1] - 0.8413).abs() < 1e-3, "gelu(1) = {}", vals[1]);
        assert!((vals[2] - (-0.1587)).abs() < 1e-3, "gelu(-1) = {}", vals[2]);
    }

    fn test_gelu_tanh_regression(config) {
        let registry = OpRegistry::new();
        let x = Tensor::from_slice([0.0f32, 1.0]);
        let mut node = NodeProto::default();
        node.attribute.push(make_attr_string("approximate", "tanh"));

        let result = registry.dispatch("Gelu", "", &[x], &node).unwrap();
        let vals = result.realize_with(&config).unwrap().to_vec::<f32>().unwrap();
        assert!((vals[0] - 0.0).abs() < 1e-4, "gelu_tanh(0) = {}", vals[0]);
        assert!((vals[1] - 0.8412).abs() < 1e-3, "gelu_tanh(1) = {}", vals[1]);
    }

    fn test_softplus(config) {
        let registry = OpRegistry::new();
        let x = Tensor::from_slice([0.0f32, 1.0, -1.0]);
        let node = NodeProto::default();

        let result = registry.dispatch("Softplus", "", &[x], &node).unwrap();
        let vals = result.realize_with(&config).unwrap().to_vec::<f32>().unwrap();
        let expected = [std::f32::consts::LN_2, 1.3133, 0.3133];
        for (i, (v, e)) in vals.iter().zip(expected.iter()).enumerate() {
            assert!((v - e).abs() < 1e-3, "softplus[{i}]: got {v}, expected {e}");
        }
    }

    fn test_mish(config) {
        let registry = OpRegistry::new();
        let x = Tensor::from_slice([0.0f32, 1.0, -1.0]);
        let node = NodeProto::default();

        let result = registry.dispatch("Mish", "", &[x], &node).unwrap();
        let vals = result.realize_with(&config).unwrap().to_vec::<f32>().unwrap();
        let expected = [0.0f32, 0.8651, -0.3034];
        for (i, (v, e)) in vals.iter().zip(expected.iter()).enumerate() {
            assert!((v - e).abs() < 1e-3, "mish[{i}]: got {v}, expected {e}");
        }
    }

    fn test_hardswish(config) {
        let registry = OpRegistry::new();
        let x = Tensor::from_slice([-4.0f32, 0.0, 3.0, 4.0]);
        let node = NodeProto::default();

        let result = registry.dispatch("HardSwish", "", &[x], &node).unwrap();
        let vals = result.realize_with(&config).unwrap().to_vec::<f32>().unwrap();
        let expected = [0.0f32, 0.0, 3.0, 4.0];
        for (i, (v, e)) in vals.iter().zip(expected.iter()).enumerate() {
            assert!((v - e).abs() < 1e-3, "hardswish[{i}]: got {v}, expected {e}");
        }
    }

    fn test_softsign(config) {
        let registry = OpRegistry::new();
        let x = Tensor::from_slice([-2.0f32, 0.0, 2.0]);
        let node = NodeProto::default();

        let result = registry.dispatch("Softsign", "", &[x], &node).unwrap();
        let vals = result.realize_with(&config).unwrap().to_vec::<f32>().unwrap();
        let expected = [-0.6667f32, 0.0, 0.6667];
        for (i, (v, e)) in vals.iter().zip(expected.iter()).enumerate() {
            assert!((v - e).abs() < 1e-3, "softsign[{i}]: got {v}, expected {e}");
        }
    }

    fn test_celu(config) {
        let registry = OpRegistry::new();
        let x = Tensor::from_slice([-1.0f32, 0.0, 1.0]);
        let mut node = NodeProto::default();
        node.attribute.push(make_attr_float("alpha", 1.0));

        let result = registry.dispatch("Celu", "", &[x], &node).unwrap();
        let vals = result.realize_with(&config).unwrap().to_vec::<f32>().unwrap();
        let expected = [-0.6321f32, 0.0, 1.0];
        for (i, (v, e)) in vals.iter().zip(expected.iter()).enumerate() {
            assert!((v - e).abs() < 1e-3, "celu[{i}]: got {v}, expected {e}");
        }
    }

    fn test_hardmax(config) {
        let registry = OpRegistry::new();
        let x = Tensor::from_ndarray(&array![[1.0f32, 3.0, 2.0], [5.0, 4.0, 0.0]]);
        let mut node = NodeProto::default();
        node.attribute.push(make_attr_int("axis", -1));

        let result = registry.dispatch("Hardmax", "", &[x], &node).unwrap();
        let vals = result.realize_with(&config).unwrap().to_vec::<f32>().unwrap();
        let expected = [0.0f32, 1.0, 0.0, 1.0, 0.0, 0.0];
        for (i, (v, e)) in vals.iter().zip(expected.iter()).enumerate() {
            assert!((v - e).abs() < 1e-6, "hardmax[{i}]: got {v}, expected {e}");
        }
    }

    fn test_binarizer(config) {
        let registry = OpRegistry::new();
        let x = Tensor::from_slice([-1.0f32, 0.0, 0.5, 1.0]);
        let mut node = NodeProto::default();
        node.attribute.push(make_attr_float("threshold", 0.0));

        let result = registry.dispatch("Binarizer", "", &[x], &node).unwrap();
        let vals = result.realize_with(&config).unwrap().to_vec::<f32>().unwrap();
        let expected = [0.0f32, 0.0, 1.0, 1.0];
        for (i, (v, e)) in vals.iter().zip(expected.iter()).enumerate() {
            assert!((v - e).abs() < 1e-6, "binarizer[{i}]: got {v}, expected {e}");
        }
    }

    fn test_swish_alpha(config) {
        let registry = OpRegistry::new();
        let x = Tensor::from_slice([0.0f32, 1.0]);
        let mut node = NodeProto::default();
        node.attribute.push(make_attr_float("alpha", 1.0));

        let result = registry.dispatch("Swish", "", &[x], &node).unwrap();
        let vals = result.realize_with(&config).unwrap().to_vec::<f32>().unwrap();
        let expected = [0.0f32, 0.7311];
        for (i, (v, e)) in vals.iter().zip(expected.iter()).enumerate() {
            assert!((v - e).abs() < 1e-3, "swish[{i}]: got {v}, expected {e}");
        }
    }

    fn test_bias_gelu(config) {
        let registry = OpRegistry::new();
        let x = Tensor::from_slice([0.0f32, 1.0]);
        let bias = Tensor::from_slice([0.5f32, 0.5]);
        let node = NodeProto::default();

        let result = registry.dispatch("BiasGelu", "", &[x, bias], &node).unwrap();
        let vals = result.realize_with(&config).unwrap().to_vec::<f32>().unwrap();
        assert_eq!(vals.len(), 2, "bias_gelu should produce 2 elements, got {}", vals.len());
    }

    fn test_fast_gelu(config) {
        let registry = OpRegistry::new();
        let x = Tensor::from_slice([0.0f32, 1.0]);
        let node = NodeProto::default();

        let result = registry.dispatch("FastGelu", "", &[x], &node).unwrap();
        let vals = result.realize_with(&config).unwrap().to_vec::<f32>().unwrap();
        let expected = [0.0f32, 0.8412];
        for (i, (v, e)) in vals.iter().zip(expected.iter()).enumerate() {
            assert!((v - e).abs() < 1e-3, "fast_gelu[{i}]: got {v}, expected {e}");
        }
    }
}
