use std::collections::HashMap;
use std::path::{Path, PathBuf};

use prost::Message;

pub(crate) use crate::parser::onnx::{
    AttributeProto, GraphProto, ModelProto, NodeProto, StringStringEntryProto, TensorProto, ValueInfoProto,
    tensor_proto, type_proto,
};
pub(crate) use crate::registry::*;
pub(crate) use svod_dtype::{DType, ScalarDType};
pub(crate) use svod_tensor::{PrepareConfig, Tensor};

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
    TensorProto { data_type: dtype, dims, raw_data: raw_data.into(), ..Default::default() }
}

fn make_initializer(name: &str, data_type: i32, dims: Vec<i64>, raw_data: Vec<u8>) -> (ValueInfoProto, TensorProto) {
    let input = ValueInfoProto { name: name.to_string(), ..Default::default() };
    let init = TensorProto { name: name.to_string(), data_type, dims, raw_data: raw_data.into(), ..Default::default() };
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
// ONNX light model test infrastructure
// ---------------------------------------------------------------------------

pub(crate) fn run_onnx_light_test(model_path: &str, output_pb_path: &str, config: &PrepareConfig) {
    let model_path = Path::new(model_path);
    let test_name = model_path.file_stem().unwrap().to_string_lossy();

    // 1+2. Import model via file path — enables DISK-backed weight loading
    // (Bytes zero-copy decoding + DISK tensor for lazy weight views)
    let mut importer = OnnxImporter::new();
    let result = importer.import(model_path, &[]).unwrap_or_else(|e| panic!("{test_name}: import failed: {e}"));

    // 3. Assign deterministic inputs: arange(n)/n (matches ONNX backend test runner)
    for (name, input_tensor) in &result.inputs {
        let shape: Vec<usize> = input_tensor
            .shape()
            .unwrap_or_else(|e| panic!("{test_name}: input '{name}' shape: {e}"))
            .iter()
            .map(|d| d.as_const().unwrap_or_else(|| panic!("{test_name}: dynamic dim in '{name}'")))
            .collect();
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
        let bytes: &[u8] = bytemuck::cast_slice(&data);
        let real_tensor = Tensor::from_raw_bytes(bytes, &shape, DType::Scalar(ScalarDType::Float32))
            .unwrap_or_else(|e| panic!("{test_name}: input '{name}': {e}"));
        input_tensor.assign(&real_tensor);
    }

    // 5. Batch-realize ALL outputs (matches Tinygrad: all outputs in one SINK)
    let mut outputs: Vec<(String, Tensor)> = result.outputs.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    {
        let mut refs: Vec<&mut Tensor> = outputs.iter_mut().map(|(_, t)| t).collect();
        Tensor::realize_batch_with(refs.iter_mut().map(|t| &mut **t), config)
            .unwrap_or_else(|e| panic!("{test_name}: realize failed: {e}"));
    }

    // 6. Load expected output and compare
    let output_name = result.outputs.keys().next().unwrap_or_else(|| panic!("{test_name}: no outputs")).clone();
    let pb_bytes =
        std::fs::read(output_pb_path).unwrap_or_else(|e| panic!("{test_name}: failed to read expected output: {e}"));
    let tensor_proto = TensorProto::decode(pb_bytes.as_slice())
        .unwrap_or_else(|e| panic!("{test_name}: failed to decode expected output: {e}"));
    let expected = tensor_from_proto_ext(&tensor_proto, None)
        .unwrap_or_else(|e| panic!("{test_name}: expected output conversion: {e}"));

    let mut actual = outputs
        .iter_mut()
        .find(|(k, _)| *k == output_name)
        .unwrap_or_else(|| panic!("{test_name}: missing output '{output_name}'"))
        .1
        .clone();
    assert_tensors_close(&mut actual, &expected, &test_name, config);
}

// ---------------------------------------------------------------------------
// ONNX node conformance test infrastructure
// ---------------------------------------------------------------------------

macro_rules! assert_float_close {
    ($actual:expr, $expected:expr, $name:expr, $rtol:expr, $atol:expr, $ty:ty) => {{
        let a_shape = $actual.shape().unwrap();
        let e_shape = $expected.shape().unwrap();
        assert_eq!(a_shape, e_shape, "Shape mismatch on output '{}'", $name);
        let a = $actual.as_vec::<$ty>().unwrap();
        let e = $expected.as_vec::<$ty>().unwrap();
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
        let a = $actual.as_vec::<$ty>().unwrap();
        let e = $expected.as_vec::<$ty>().unwrap();
        assert_eq!(a, e, "Value mismatch on output '{}'", $name);
    }};
}

fn assert_tensors_close(actual: &mut Tensor, expected: &Tensor, label: &str, config: &PrepareConfig) {
    let expected_dtype = expected.uop().dtype();

    // Cast actual to match expected dtype if they differ
    let mut actual_cast;
    let actual: &mut Tensor = if actual.uop().dtype() != expected_dtype {
        actual_cast = actual.cast(expected_dtype.clone()).unwrap_or_else(|e| {
            panic!("Output '{label}': dtype cast failed ({:?} -> {expected_dtype:?}): {e}", actual.uop().dtype())
        });
        actual_cast.realize_with(config).unwrap_or_else(|e| panic!("Output '{label}': realize after cast failed: {e}"));
        &mut actual_cast
    } else {
        actual
    };

    match expected_dtype.base() {
        ScalarDType::Float32 => assert_float_close!(actual, expected, label, 1e-3, 1e-7, f32),
        ScalarDType::Float64 => assert_float_close!(actual, expected, label, 1e-3, 1e-7, f64),
        ScalarDType::Float16 | ScalarDType::BFloat16 | ScalarDType::FP8E4M3 | ScalarDType::FP8E5M2 => {
            let f32_dtype = DType::Scalar(ScalarDType::Float32);
            let mut a = actual.cast(f32_dtype.clone()).unwrap();
            a.realize_with(config).unwrap();
            let mut e = expected.cast(f32_dtype).unwrap();
            e.realize_with(config).unwrap();
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

/// Whether the model touches ONNX `DOUBLE` anywhere: graph I/O, initializers,
/// constant payloads, `Cast`/`EyeLike`-style dtype attributes, type attributes,
/// and subgraphs.
fn graph_uses_double(graph: &GraphProto) -> bool {
    const DOUBLE: i32 = tensor_proto::DataType::Double as i32;
    let value_is_double = |info: &ValueInfoProto| {
        matches!(
            info.r#type.as_ref().and_then(|ty| ty.value.as_ref()),
            Some(type_proto::Value::TensorType(tensor)) if tensor.elem_type == DOUBLE
        )
    };
    graph.input.iter().chain(&graph.output).chain(&graph.value_info).any(value_is_double)
        || graph.initializer.iter().any(|init| init.data_type == DOUBLE)
        || graph.node.iter().flat_map(|node| &node.attribute).any(|attr| {
            attr.t.as_ref().is_some_and(|tensor| tensor.data_type == DOUBLE)
                || ((attr.name == "to" || attr.name == "dtype") && attr.i == i64::from(DOUBLE))
                || matches!(
                    attr.tp.as_ref().and_then(|ty| ty.value.as_ref()),
                    Some(type_proto::Value::TensorType(tensor)) if tensor.elem_type == DOUBLE
                )
                || attr.g.as_ref().is_some_and(graph_uses_double)
        })
}

/// `DOUBLE` models cannot run on a device without f64 storage (Metal, WebGPU);
/// they skip there, as tinygrad's ONNX suite does, instead of failing.
fn skipped_without_f64(test_name: &str, model: &ModelProto) -> bool {
    let device = svod_tensor::default_device();
    if svod_tensor::device_supports_storage_dtype(&device, ScalarDType::Float64) {
        return false;
    }
    let uses_double = model.graph.as_ref().is_some_and(graph_uses_double);
    if uses_double {
        eprintln!("{test_name}: skipped, {device:?} has no Float64 storage");
    }
    uses_double
}

pub(crate) fn run_onnx_node_test(test_dir: &str, config: &PrepareConfig) {
    let test_dir = Path::new(test_dir);
    let test_name = test_dir.file_name().unwrap().to_string_lossy();

    // 1. Load and decode model
    let model_bytes = std::fs::read(test_dir.join("model.onnx"))
        .unwrap_or_else(|e| panic!("{test_name}: failed to read model.onnx: {e}"));
    let model = ModelProto::decode(model_bytes.as_slice())
        .unwrap_or_else(|e| panic!("{test_name}: failed to decode model: {e}"));
    if skipped_without_f64(&test_name, &model) {
        return;
    }

    // 2. Extract input/output names from raw proto (before prepare filters out initializers)
    let proto_graph = model.graph.as_ref().unwrap_or_else(|| panic!("{test_name}: model has no graph"));
    let input_names: Vec<String> = proto_graph.input.iter().map(|i| i.name.clone()).collect();
    let output_names: Vec<String> = proto_graph.output.iter().map(|o| o.name.clone()).collect();

    let importer = OnnxImporter::new();

    // 3. Run each test data set
    for set_dir in sorted_dirs(test_dir, "test_data_set_") {
        let set_name = set_dir.file_name().unwrap().to_string_lossy();

        // Load test inputs
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

        // Import with concrete inputs (some ops read values at trace time)
        let result = importer
            .import_model_with_inputs(model.clone(), inputs, &[])
            .unwrap_or_else(|e| panic!("{test_name}/{set_name}: import failed: {e}"));

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
            let actual =
                result.outputs.get(name).unwrap_or_else(|| panic!("{test_name}/{set_name}: missing output '{name}'"));
            let mut actual = actual.clone();
            actual.realize_with(config).unwrap_or_else(|e| panic!("{test_name}/{set_name}: realize failed: {e}"));
            assert_tensors_close(&mut actual, &expected, &format!("{test_name}/{set_name}:{name}"), config);
        }
    }
}
