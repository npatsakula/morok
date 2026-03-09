use std::collections::HashMap;
use std::path::{Path, PathBuf};

use prost::Message;

pub(crate) use crate::parser::onnx::{
    AttributeProto, GraphProto, ModelProto, NodeProto, StringStringEntryProto, TensorProto, ValueInfoProto,
    tensor_proto,
};
pub(crate) use crate::registry::*;
pub(crate) use morok_dtype::{DType, ScalarDType};
pub(crate) use morok_tensor::Tensor;

use crate::importer::OnnxImporter;

pub(crate) fn make_attr_int(name: &str, val: i64) -> AttributeProto {
    AttributeProto { name: name.to_string(), i: val, ..Default::default() }
}

pub(crate) fn make_attr_ints(name: &str, vals: &[i64]) -> AttributeProto {
    AttributeProto { name: name.to_string(), ints: vals.to_vec(), ..Default::default() }
}

pub(crate) fn make_attr_float(name: &str, val: f32) -> AttributeProto {
    AttributeProto { name: name.to_string(), f: val, ..Default::default() }
}

pub(crate) fn make_attr_string(name: &str, val: &str) -> AttributeProto {
    AttributeProto { name: name.to_string(), s: val.as_bytes().to_vec(), ..Default::default() }
}

pub(crate) fn make_attr_floats(name: &str, vals: &[f32]) -> AttributeProto {
    AttributeProto { name: name.to_string(), floats: vals.to_vec(), ..Default::default() }
}

pub(crate) fn make_attr_tensor(name: &str, tensor: TensorProto) -> AttributeProto {
    AttributeProto { name: name.to_string(), t: Some(tensor), ..Default::default() }
}

pub(crate) fn make_attr_graph(name: &str, graph: GraphProto) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: 5, // GRAPH
        g: Some(graph),
        ..Default::default()
    }
}

pub(crate) fn make_graph(
    nodes: Vec<NodeProto>,
    inputs: Vec<&str>,
    outputs: Vec<&str>,
    initializers: Vec<TensorProto>,
) -> GraphProto {
    GraphProto {
        node: nodes,
        input: inputs.iter().map(|n| ValueInfoProto { name: n.to_string(), ..Default::default() }).collect(),
        output: outputs.iter().map(|n| ValueInfoProto { name: n.to_string(), ..Default::default() }).collect(),
        initializer: initializers,
        ..Default::default()
    }
}

pub(crate) fn make_tensor_proto(raw_data: Vec<u8>, dims: Vec<i64>, dtype: i32) -> TensorProto {
    TensorProto { data_type: dtype, dims, raw_data, ..Default::default() }
}

fn make_initializer(name: &str, data_type: i32, dims: Vec<i64>, raw_data: Vec<u8>) -> (ValueInfoProto, TensorProto) {
    let input = ValueInfoProto { name: name.to_string(), ..Default::default() };
    let init = TensorProto { name: name.to_string(), data_type, dims, raw_data, ..Default::default() };
    (input, init)
}

pub(crate) fn make_minimal_model() -> ModelProto {
    let (input, init) = make_initializer(
        "input",
        tensor_proto::DataType::Float as i32,
        vec![3],
        [1.0f32, 2.0, 3.0].iter().flat_map(|v| v.to_le_bytes()).collect(),
    );

    let node = NodeProto {
        op_type: "Identity".to_string(),
        input: vec!["input".to_string()],
        output: vec!["output".to_string()],
        ..Default::default()
    };

    ModelProto {
        graph: Some(GraphProto {
            name: "test_graph".to_string(),
            input: vec![input],
            output: vec![ValueInfoProto { name: "output".to_string(), ..Default::default() }],
            initializer: vec![init],
            node: vec![node],
            ..Default::default()
        }),
        ..Default::default()
    }
}

pub(crate) fn make_multi_output_model() -> ModelProto {
    let (input, init) = make_initializer(
        "input",
        tensor_proto::DataType::Float as i32,
        vec![3],
        [1.0f32, 2.0, 3.0].iter().flat_map(|v| v.to_le_bytes()).collect(),
    );

    let node1 = NodeProto {
        op_type: "Identity".to_string(),
        input: vec!["input".to_string()],
        output: vec!["out1".to_string()],
        ..Default::default()
    };
    let node2 = NodeProto {
        op_type: "Identity".to_string(),
        input: vec!["input".to_string()],
        output: vec!["out2".to_string()],
        ..Default::default()
    };

    ModelProto {
        graph: Some(GraphProto {
            name: "multi_output_test".to_string(),
            input: vec![input],
            output: ["out1", "out2"]
                .iter()
                .map(|n| ValueInfoProto { name: n.to_string(), ..Default::default() })
                .collect(),
            initializer: vec![init],
            node: vec![node1, node2],
            ..Default::default()
        }),
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// ONNX node conformance test infrastructure
// ---------------------------------------------------------------------------

macro_rules! assert_float_close {
    ($actual:expr, $expected:expr, $name:expr, $rtol:expr, $atol:expr, $ty:ty) => {{
        let a_shape = $actual.shape().unwrap();
        let e_shape = $expected.shape().unwrap();
        assert_eq!(a_shape, e_shape, "Shape mismatch on output '{}'", $name);
        let a = $actual.to_vec::<$ty>().unwrap();
        let e = $expected.to_vec::<$ty>().unwrap();
        for (idx, (av, ev)) in a.iter().zip(e.iter()).enumerate() {
            let av = *av as f64;
            let ev = *ev as f64;
            if av.is_nan() && ev.is_nan() {
                continue;
            }
            if av == ev {
                continue; // handles +/-Inf and exact matches
            }
            let diff = (av - ev).abs();
            let tol = $atol + $rtol * ev.abs();
            assert!(
                diff <= tol,
                "Output '{}' element {}: actual={}, expected={}, diff={}, tol={}",
                $name,
                idx,
                av,
                ev,
                diff,
                tol
            );
        }
    }};
}

macro_rules! assert_int_exact {
    ($actual:expr, $expected:expr, $name:expr, $ty:ty) => {{
        let a_shape = $actual.shape().unwrap();
        let e_shape = $expected.shape().unwrap();
        assert_eq!(a_shape, e_shape, "Shape mismatch on output '{}'", $name);
        let a = $actual.to_vec::<$ty>().unwrap();
        let e = $expected.to_vec::<$ty>().unwrap();
        assert_eq!(a, e, "Value mismatch on output '{}'", $name);
    }};
}

fn assert_tensors_close(actual: &Tensor, expected: &Tensor, label: &str) {
    let expected_dtype = expected.uop().dtype();

    // Cast actual to match expected dtype if they differ
    let actual_cast;
    let actual = if actual.uop().dtype() != expected_dtype {
        actual_cast = actual.cast(expected_dtype.clone()).unwrap_or_else(|e| {
            panic!("Output '{label}': dtype cast failed ({:?} -> {expected_dtype:?}): {e}", actual.uop().dtype())
        });
        &actual_cast
    } else {
        actual
    };

    match expected_dtype.base() {
        ScalarDType::Float32 => assert_float_close!(actual, expected, label, 1e-3, 1e-7, f32),
        ScalarDType::Float64 => assert_float_close!(actual, expected, label, 1e-3, 1e-7, f64),
        ScalarDType::Float16 | ScalarDType::BFloat16 | ScalarDType::FP8E4M3 | ScalarDType::FP8E5M2 => {
            let f32_dtype = DType::Scalar(ScalarDType::Float32);
            let a = actual.cast(f32_dtype.clone()).unwrap();
            let e = expected.cast(f32_dtype).unwrap();
            assert_float_close!(&a, &e, label, 1e-2, 1e-3, f32);
        }
        ScalarDType::Int8 => assert_int_exact!(actual, expected, label, i8),
        ScalarDType::Int16 => assert_int_exact!(actual, expected, label, i16),
        ScalarDType::Int32 => assert_int_exact!(actual, expected, label, i32),
        ScalarDType::Int64 => assert_int_exact!(actual, expected, label, i64),
        ScalarDType::UInt8 => assert_int_exact!(actual, expected, label, u8),
        ScalarDType::UInt16 => assert_int_exact!(actual, expected, label, u16),
        ScalarDType::UInt32 => assert_int_exact!(actual, expected, label, u32),
        ScalarDType::UInt64 => assert_int_exact!(actual, expected, label, u64),
        ScalarDType::Bool => assert_int_exact!(actual, expected, label, bool),
        other => panic!("Unsupported dtype for comparison: {other:?}"),
    }
}

fn sorted_dirs(parent: &Path, prefix: &str) -> Vec<PathBuf> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(parent)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_type().map(|ft| ft.is_dir()).unwrap_or(false) && e.file_name().to_string_lossy().starts_with(prefix)
        })
        .map(|e| e.path())
        .collect();
    entries.sort();
    entries
}

pub(crate) fn run_onnx_node_test(test_dir: &str) {
    let test_dir = Path::new(test_dir);
    let test_name = test_dir.file_name().unwrap().to_string_lossy();

    // 1. Load and decode model
    let model_bytes = std::fs::read(test_dir.join("model.onnx"))
        .unwrap_or_else(|e| panic!("{test_name}: failed to read model.onnx: {e}"));
    let model = ModelProto::decode(model_bytes.as_slice())
        .unwrap_or_else(|e| panic!("{test_name}: failed to decode model: {e}"));

    // 2. Extract input/output names from raw proto (before prepare filters out initializers)
    let proto_graph = model.graph.as_ref().unwrap_or_else(|| panic!("{test_name}: model has no graph"));
    let input_names: Vec<String> = proto_graph.input.iter().map(|i| i.name.clone()).collect();
    let output_names: Vec<String> = proto_graph.output.iter().map(|o| o.name.clone()).collect();

    // 3. Prepare graph
    let importer = OnnxImporter::new();
    let graph = importer.prepare(model).unwrap_or_else(|e| panic!("{test_name}: prepare failed: {e}"));

    // 4. Run each test data set
    for set_dir in sorted_dirs(test_dir, "test_data_set_") {
        let set_name = set_dir.file_name().unwrap().to_string_lossy();

        // Load inputs
        let mut inputs = HashMap::new();
        for (i, name) in input_names.iter().enumerate() {
            let pb_path = set_dir.join(format!("input_{i}.pb"));
            if !pb_path.exists() {
                break;
            }
            let pb_bytes = std::fs::read(&pb_path)
                .unwrap_or_else(|e| panic!("{test_name}/{set_name}: failed to read input_{i}.pb: {e}"));
            let tensor_proto = TensorProto::decode(pb_bytes.as_slice())
                .unwrap_or_else(|e| panic!("{test_name}/{set_name}: failed to decode input_{i}.pb: {e}"));
            let tensor = tensor_from_proto_ext(&tensor_proto, None)
                .unwrap_or_else(|e| panic!("{test_name}/{set_name}: input '{name}': {e}"));
            inputs.insert(name.clone(), tensor);
        }

        // Execute via trace_external (inputs override auto-resolved placeholders)
        let (_, outputs) = importer
            .trace_external(&graph, inputs)
            .unwrap_or_else(|e| panic!("{test_name}/{set_name}: execution failed: {e}"));

        // Load expected outputs and compare
        for (i, name) in output_names.iter().enumerate() {
            let pb_path = set_dir.join(format!("output_{i}.pb"));
            if !pb_path.exists() {
                break;
            }
            let pb_bytes = std::fs::read(&pb_path)
                .unwrap_or_else(|e| panic!("{test_name}/{set_name}: failed to read output_{i}.pb: {e}"));
            let tensor_proto = TensorProto::decode(pb_bytes.as_slice())
                .unwrap_or_else(|e| panic!("{test_name}/{set_name}: failed to decode output_{i}.pb: {e}"));
            let expected = tensor_from_proto_ext(&tensor_proto, None)
                .unwrap_or_else(|e| panic!("{test_name}/{set_name}: expected output '{name}': {e}"));
            let actual = outputs.get(name).unwrap_or_else(|| panic!("{test_name}/{set_name}: missing output '{name}'"));
            assert_tensors_close(actual, &expected, &format!("{test_name}/{set_name}:{name}"));
        }
    }
}
