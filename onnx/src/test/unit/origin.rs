//! Origin capture in the importer: every node of an imported graph, including the
//! nodes of an `If` subgraph, must own the UOps it builds.

use std::collections::BTreeSet;

use svod_ir::origin::{self, OriginFrame};
use svod_tensor::Tensor;

use super::importer::make_if_model;
use crate::importer::OnnxImporter;
use crate::parser::onnx::{GraphProto, ModelProto, NodeProto, TensorProto, ValueInfoProto, tensor_proto};

/// Every graph path in the tensor's cone: the `Onnx` and `Label` frames of each
/// origin chain, joined. Call frames are the file:line layer beneath a node and
/// are dropped here.
fn graph_paths(tensor: &Tensor) -> BTreeSet<String> {
    let mut paths = BTreeSet::new();
    let leaves: BTreeSet<_> = tensor.uop().toposort().iter().filter_map(|node| node.origin()).collect();
    for leaf in leaves {
        let mut path = String::new();
        for frame in origin::chain(leaf).into_iter().filter_map(origin::get).map(|origin| origin.frame) {
            if matches!(frame, OriginFrame::Call { .. }) {
                continue;
            }
            if !path.is_empty() {
                path.push('.');
            }
            path.push_str(&frame.to_string());
            paths.insert(path.clone());
        }
    }
    paths
}

/// Every `Onnx` frame in the tensor's cone, as `(index, name, op_type, version)`.
fn onnx_frames(tensor: &Tensor) -> BTreeSet<(u32, Option<String>, String, i64)> {
    tensor
        .uop()
        .toposort()
        .iter()
        .filter_map(|node| node.origin())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .flat_map(origin::chain)
        .filter_map(origin::get)
        .filter_map(|origin| match origin.frame {
            OriginFrame::Onnx { index, name, op_type, version, .. } => {
                Some((index, name.as_deref().map(str::to_owned), op_type.to_string(), version))
            }
            _ => None,
        })
        .collect()
}

/// `a * b + c` over three initializers, so the graph has two compute nodes with
/// known indices and one named node.
fn make_chain_model() -> ModelProto {
    let initializer = |name: &str, values: [f32; 4]| TensorProto {
        name: name.to_string(),
        data_type: tensor_proto::DataType::Float as i32,
        dims: vec![2, 2],
        raw_data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ..Default::default()
    };
    let node = |name: &str, op_type: &str, inputs: [&str; 2], output: &str| NodeProto {
        name: name.to_string(),
        op_type: op_type.to_string(),
        input: inputs.iter().map(|i| i.to_string()).collect(),
        output: vec![output.to_string()],
        ..Default::default()
    };

    ModelProto {
        graph: Some(GraphProto {
            name: "chain".to_string(),
            input: ["a", "b", "c"]
                .iter()
                .map(|n| ValueInfoProto { name: n.to_string(), ..Default::default() })
                .collect(),
            initializer: vec![
                initializer("a", [1.0, 2.0, 3.0, 4.0]),
                initializer("b", [5.0, 6.0, 7.0, 8.0]),
                initializer("c", [9.0, 10.0, 11.0, 12.0]),
            ],
            output: vec![ValueInfoProto { name: "out".to_string(), ..Default::default() }],
            node: vec![node("scale", "Mul", ["a", "b"], "prod"), node("", "Add", ["prod", "c"], "out")],
            ..Default::default()
        }),
        ..Default::default()
    }
}

#[test]
fn nodes_own_the_uops_they_build() {
    let _capture = origin::capture_for_thread(true);
    let outputs = OnnxImporter::new().import_model(make_chain_model(), &[]).unwrap().outputs;
    let out = outputs.get("out").unwrap();

    // Node index is the position in `graph.node`; the name is carried when set,
    // and the version is the resolved opset (1 with no `opset_import`).
    assert_eq!(
        onnx_frames(out).into_iter().collect::<Vec<_>>(),
        vec![(0, Some("scale".to_string()), "Mul".to_string(), 1), (1, None, "Add".to_string(), 1)]
    );

    // Weights are built outside any node and keep their own root.
    assert!(graph_paths(out).contains("initializer"));
}

#[test]
fn capture_is_off_by_default() {
    let _capture = origin::capture_for_thread(false);
    let outputs = OnnxImporter::new().import_model(make_chain_model(), &[]).unwrap().outputs;
    assert!(graph_paths(outputs.get("out").unwrap()).is_empty());
}

#[test]
fn subgraph_nodes_chain_to_the_enclosing_node() {
    let _capture = origin::capture_for_thread(true);
    let outputs = OnnxImporter::new().import_model(make_if_model(true, &[1.0, 2.0, 3.0]), &[]).unwrap().outputs;
    let paths = graph_paths(outputs.get("output").unwrap());

    // The If node is index 0 of the outer graph; the branch attribute is a
    // segment under it, and the branch's own node indices restart at zero.
    // Both branches are live in the merged graph and must stay distinguishable.
    for expected in [
        "#0:If",
        "#0:If.then_branch.#0:Constant",
        "#0:If.then_branch.#1:Add",
        "#0:If.else_branch.#1:Add",
        "initializer",
    ] {
        assert!(paths.contains(expected), "missing {expected} in {paths:?}");
    }
}
