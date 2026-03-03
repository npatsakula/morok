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
    let mut attr = AttributeProto::default();
    attr.name = name.to_string();
    attr.i = val;
    attr
}

pub(crate) fn make_attr_ints(name: &str, vals: &[i64]) -> AttributeProto {
    let mut attr = AttributeProto::default();
    attr.name = name.to_string();
    attr.ints = vals.to_vec();
    attr
}

pub(crate) fn make_attr_float(name: &str, val: f32) -> AttributeProto {
    let mut attr = AttributeProto::default();
    attr.name = name.to_string();
    attr.f = val;
    attr
}

pub(crate) fn make_attr_string(name: &str, val: &str) -> AttributeProto {
    let mut attr = AttributeProto::default();
    attr.name = name.to_string();
    attr.s = val.as_bytes().to_vec();
    attr
}

pub(crate) fn make_attr_floats(name: &str, vals: &[f32]) -> AttributeProto {
    let mut attr = AttributeProto::default();
    attr.name = name.to_string();
    attr.floats = vals.to_vec();
    attr
}

pub(crate) fn make_attr_tensor(name: &str, tensor: TensorProto) -> AttributeProto {
    let mut attr = AttributeProto::default();
    attr.name = name.to_string();
    attr.t = Some(tensor);
    attr
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
    let mut tensor = TensorProto::default();
    tensor.data_type = dtype;
    tensor.dims = dims;
    tensor.raw_data = raw_data;
    tensor
}

pub(crate) fn make_minimal_model() -> ModelProto {
    let mut model = ModelProto::default();
    let mut graph = GraphProto::default();
    graph.name = "test_graph".to_string();

    let mut input = ValueInfoProto::default();
    input.name = "input".to_string();
    graph.input.push(input);

    let mut init = TensorProto::default();
    init.name = "input".to_string();
    init.data_type = tensor_proto::DataType::Float as i32;
    init.dims = vec![3];
    init.raw_data = [1.0f32, 2.0, 3.0].iter().flat_map(|v| v.to_le_bytes()).collect();
    graph.initializer.push(init);

    let mut output = ValueInfoProto::default();
    output.name = "output".to_string();
    graph.output.push(output);

    let mut node = NodeProto::default();
    node.op_type = "Identity".to_string();
    node.input.push("input".to_string());
    node.output.push("output".to_string());
    graph.node.push(node);

    model.graph = Some(graph);
    model
}

pub(crate) fn make_multi_output_model() -> ModelProto {
    let mut model = ModelProto::default();
    let mut graph = GraphProto::default();
    graph.name = "multi_output_test".to_string();

    let mut input = ValueInfoProto::default();
    input.name = "input".to_string();
    graph.input.push(input);

    let mut init = TensorProto::default();
    init.name = "input".to_string();
    init.data_type = tensor_proto::DataType::Float as i32;
    init.dims = vec![3];
    init.raw_data = [1.0f32, 2.0, 3.0].iter().flat_map(|v| v.to_le_bytes()).collect();
    graph.initializer.push(init);

    for name in ["out1", "out2"] {
        let mut output = ValueInfoProto::default();
        output.name = name.to_string();
        graph.output.push(output);
    }

    let mut node1 = NodeProto::default();
    node1.op_type = "Identity".to_string();
    node1.input.push("input".to_string());
    node1.output.push("out1".to_string());
    graph.node.push(node1);

    let mut node2 = NodeProto::default();
    node2.op_type = "Identity".to_string();
    node2.input.push("input".to_string());
    node2.output.push("out2".to_string());
    graph.node.push(node2);

    model.graph = Some(graph);
    model
}

// ---------------------------------------------------------------------------
// ONNX node conformance test infrastructure
// ---------------------------------------------------------------------------

macro_rules! assert_float_close {
    ($actual:expr, $expected:expr, $name:expr, $rtol:expr, $atol:expr, $ty:ty) => {{
        let a = $actual.to_ndarray::<$ty>().unwrap();
        let e = $expected.to_ndarray::<$ty>().unwrap();
        assert_eq!(a.shape(), e.shape(), "Shape mismatch on output '{}'", $name);
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
        let a = $actual.to_ndarray::<$ty>().unwrap();
        let e = $expected.to_ndarray::<$ty>().unwrap();
        assert_eq!(a.shape(), e.shape(), "Shape mismatch on output '{}'", $name);
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

        // Execute
        let outputs = importer
            .execute(&graph, inputs)
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
