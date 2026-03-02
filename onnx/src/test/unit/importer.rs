use std::collections::HashMap;

use crate::importer::{DimValue, InputSpec, OnnxImporter};
use crate::parser::onnx::{GraphProto, ModelProto, NodeProto, TensorProto, ValueInfoProto, tensor_proto};
use crate::test::helpers::*;

#[test]
fn test_importer_creation() {
    let _importer = OnnxImporter::new();
}

#[test]
fn test_prepare_minimal_model() {
    let importer = OnnxImporter::new();
    let model = make_minimal_model();
    let graph = importer.prepare(model).unwrap();

    assert!(graph.inputs.is_empty());
    assert_eq!(graph.outputs.len(), 1);
    assert_eq!(graph.outputs[0], "output");
    assert!(graph.initializers.contains_key("input"));
}

#[test]
fn test_execute_minimal_model() {
    let importer = OnnxImporter::new();
    let model = make_minimal_model();
    let graph = importer.prepare(model).unwrap();

    let outputs = importer.execute(&graph, HashMap::new()).unwrap();

    assert_eq!(outputs.len(), 1);
    assert!(outputs.contains_key("output"));
}

#[test]
fn test_import_model_minimal() {
    let mut importer = OnnxImporter::new();
    let model = make_minimal_model();
    let outputs = importer.import_model(model).unwrap();

    assert_eq!(outputs.len(), 1);
    assert!(outputs.contains_key("output"));
}

#[test]
fn test_multi_output_model() {
    let importer = OnnxImporter::new();
    let model = make_multi_output_model();
    let graph = importer.prepare(model).unwrap();

    assert_eq!(graph.outputs.len(), 2);
    assert!(graph.outputs.contains(&"out1".to_string()));
    assert!(graph.outputs.contains(&"out2".to_string()));

    let outputs = importer.execute(&graph, HashMap::new()).unwrap();
    assert_eq!(outputs.len(), 2);
    assert!(outputs.contains_key("out1"));
    assert!(outputs.contains_key("out2"));
}

#[test]
fn test_import_empty_model() {
    let importer = OnnxImporter::new();
    let model = ModelProto::default();
    let result = importer.prepare(model);
    assert!(result.is_err());
}

#[test]
fn test_import_with_add() {
    let mut importer = OnnxImporter::new();

    let mut model = ModelProto::default();
    let mut graph = GraphProto::default();
    graph.name = "add_test".to_string();

    for (name, values) in [("a", vec![1.0f32, 2.0, 3.0]), ("b", vec![4.0f32, 5.0, 6.0])] {
        let mut input = ValueInfoProto::default();
        input.name = name.to_string();
        graph.input.push(input);

        let mut init = TensorProto::default();
        init.name = name.to_string();
        init.data_type = tensor_proto::DataType::Float as i32;
        init.dims = vec![3];
        init.raw_data = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        graph.initializer.push(init);
    }

    let mut output = ValueInfoProto::default();
    output.name = "output".to_string();
    graph.output.push(output);

    let mut node = NodeProto::default();
    node.op_type = "Add".to_string();
    node.input.push("a".to_string());
    node.input.push("b".to_string());
    node.output.push("output".to_string());
    graph.node.push(node);

    model.graph = Some(graph);

    let outputs = importer.import_model(model).unwrap();
    assert_eq!(outputs.len(), 1);
    assert!(outputs.contains_key("output"));
}

#[test]
fn test_import_with_matmul() {
    let mut importer = OnnxImporter::new();

    let mut model = ModelProto::default();
    let mut graph = GraphProto::default();
    graph.name = "matmul_test".to_string();

    for (name, values) in [("a", vec![1.0f32, 2.0, 3.0, 4.0]), ("b", vec![5.0f32, 6.0, 7.0, 8.0])] {
        let mut input = ValueInfoProto::default();
        input.name = name.to_string();
        graph.input.push(input);

        let mut init = TensorProto::default();
        init.name = name.to_string();
        init.data_type = tensor_proto::DataType::Float as i32;
        init.dims = vec![2, 2];
        init.raw_data = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        graph.initializer.push(init);
    }

    let mut output = ValueInfoProto::default();
    output.name = "output".to_string();
    graph.output.push(output);

    let mut node = NodeProto::default();
    node.op_type = "MatMul".to_string();
    node.input.push("a".to_string());
    node.input.push("b".to_string());
    node.output.push("output".to_string());
    graph.node.push(node);

    model.graph = Some(graph);

    let outputs = importer.import_model(model).unwrap();
    assert_eq!(outputs.len(), 1);
    assert!(outputs.contains_key("output"));
}

#[test]
fn test_input_spec_static_shape() {
    let spec = InputSpec::new(
        vec![DimValue::Static(2), DimValue::Static(3)],
        DType::Scalar(morok_dtype::ScalarDType::Float32),
        false,
    );

    assert!(spec.is_static());
    assert_eq!(spec.static_shape(), Some(vec![2, 3]));
}

#[test]
fn test_input_spec_dynamic_shape() {
    let spec = InputSpec::new(
        vec![DimValue::Dynamic("batch".to_string()), DimValue::Static(3)],
        DType::Scalar(morok_dtype::ScalarDType::Float32),
        false,
    );

    assert!(!spec.is_static());
    assert_eq!(spec.static_shape(), None);
}

/// Build a minimal model with an If node.
///
/// Graph: condition (bool scalar initializer) -> If -> output
/// The then_branch adds `x + 10.0`, the else_branch adds `x + 20.0`.
fn make_if_model(cond_value: bool, x_values: &[f32]) -> ModelProto {
    let mut model = ModelProto::default();
    let mut graph = GraphProto::default();
    graph.name = "if_test".to_string();

    // Initializer: condition (bool scalar)
    let mut cond_input = ValueInfoProto::default();
    cond_input.name = "condition".to_string();
    graph.input.push(cond_input);

    let mut cond_init = TensorProto::default();
    cond_init.name = "condition".to_string();
    cond_init.data_type = tensor_proto::DataType::Bool as i32;
    cond_init.dims = vec![];
    cond_init.raw_data = vec![cond_value as u8];
    graph.initializer.push(cond_init);

    // Initializer: x (float tensor)
    let mut x_input = ValueInfoProto::default();
    x_input.name = "x".to_string();
    graph.input.push(x_input);

    let mut x_init = TensorProto::default();
    x_init.name = "x".to_string();
    x_init.data_type = tensor_proto::DataType::Float as i32;
    x_init.dims = vec![x_values.len() as i64];
    x_init.raw_data = x_values.iter().flat_map(|v| v.to_le_bytes()).collect();
    graph.initializer.push(x_init);

    // Then branch: output = x + 10
    let then_add_const = {
        let mut n = NodeProto::default();
        n.op_type = "Constant".to_string();
        n.output.push("ten".to_string());
        n.attribute.push(make_attr_tensor(
            "value",
            make_tensor_proto(10.0f32.to_le_bytes().to_vec(), vec![1], tensor_proto::DataType::Float as i32),
        ));
        n
    };
    let then_add = {
        let mut n = NodeProto::default();
        n.op_type = "Add".to_string();
        n.input.push("x".to_string());
        n.input.push("ten".to_string());
        n.output.push("then_out".to_string());
        n
    };
    let then_graph = make_graph(vec![then_add_const, then_add], vec!["x"], vec!["then_out"], vec![]);

    // Else branch: output = x + 20
    let else_add_const = {
        let mut n = NodeProto::default();
        n.op_type = "Constant".to_string();
        n.output.push("twenty".to_string());
        n.attribute.push(make_attr_tensor(
            "value",
            make_tensor_proto(20.0f32.to_le_bytes().to_vec(), vec![1], tensor_proto::DataType::Float as i32),
        ));
        n
    };
    let else_add = {
        let mut n = NodeProto::default();
        n.op_type = "Add".to_string();
        n.input.push("x".to_string());
        n.input.push("twenty".to_string());
        n.output.push("else_out".to_string());
        n
    };
    let else_graph = make_graph(vec![else_add_const, else_add], vec!["x"], vec!["else_out"], vec![]);

    // If node
    let mut if_node = NodeProto::default();
    if_node.op_type = "If".to_string();
    if_node.input.push("condition".to_string());
    if_node.output.push("output".to_string());
    if_node.attribute.push(make_attr_graph("then_branch", then_graph));
    if_node.attribute.push(make_attr_graph("else_branch", else_graph));
    graph.node.push(if_node);

    let mut output = ValueInfoProto::default();
    output.name = "output".to_string();
    graph.output.push(output);

    model.graph = Some(graph);
    model
}

#[test]
fn test_if_true_condition() {
    let mut importer = OnnxImporter::new();
    let model = make_if_model(true, &[1.0, 2.0, 3.0]);
    let outputs = importer.import_model(model).unwrap();

    let result = outputs.get("output").unwrap();
    let arr = result.to_ndarray::<f32>().unwrap();
    let vals: Vec<f32> = arr.iter().copied().collect();
    assert_eq!(vals, vec![11.0, 12.0, 13.0]); // x + 10
}

#[test]
fn test_if_false_condition() {
    let mut importer = OnnxImporter::new();
    let model = make_if_model(false, &[1.0, 2.0, 3.0]);
    let outputs = importer.import_model(model).unwrap();

    let result = outputs.get("output").unwrap();
    let arr = result.to_ndarray::<f32>().unwrap();
    let vals: Vec<f32> = arr.iter().copied().collect();
    assert_eq!(vals, vec![21.0, 22.0, 23.0]); // x + 20
}

#[test]
fn test_if_where_path() {
    // Both branches produce same shape -> lazy where_ selection
    let mut importer = OnnxImporter::new();
    let model = make_if_model(true, &[5.0, 6.0]);
    let outputs = importer.import_model(model).unwrap();

    let result = outputs.get("output").unwrap();
    let arr = result.to_ndarray::<f32>().unwrap();
    let vals: Vec<f32> = arr.iter().copied().collect();
    // condition=true -> then_branch (x + 10) = [15, 16]
    assert_eq!(vals, vec![15.0, 16.0]);
}

#[test]
fn test_if_shape_mismatch_fallback() {
    // Build a model where then_branch outputs shape [3] and else_branch outputs shape [2]
    let mut model = ModelProto::default();
    let mut graph = GraphProto::default();
    graph.name = "if_mismatch_test".to_string();

    // Condition: true
    let mut cond_input = ValueInfoProto::default();
    cond_input.name = "condition".to_string();
    graph.input.push(cond_input);
    let mut cond_init = TensorProto::default();
    cond_init.name = "condition".to_string();
    cond_init.data_type = tensor_proto::DataType::Bool as i32;
    cond_init.dims = vec![];
    cond_init.raw_data = vec![1u8]; // true
    graph.initializer.push(cond_init);

    // Then branch: produces [1, 2, 3] (shape [3])
    let then_const = {
        let mut n = NodeProto::default();
        n.op_type = "Constant".to_string();
        n.output.push("then_out".to_string());
        n.attribute.push(make_attr_tensor(
            "value",
            make_tensor_proto(
                [1.0f32, 2.0, 3.0].iter().flat_map(|v| v.to_le_bytes()).collect(),
                vec![3],
                tensor_proto::DataType::Float as i32,
            ),
        ));
        n
    };
    let then_graph = make_graph(vec![then_const], vec![], vec!["then_out"], vec![]);

    // Else branch: produces [10, 20] (shape [2])
    let else_const = {
        let mut n = NodeProto::default();
        n.op_type = "Constant".to_string();
        n.output.push("else_out".to_string());
        n.attribute.push(make_attr_tensor(
            "value",
            make_tensor_proto(
                [10.0f32, 20.0].iter().flat_map(|v| v.to_le_bytes()).collect(),
                vec![2],
                tensor_proto::DataType::Float as i32,
            ),
        ));
        n
    };
    let else_graph = make_graph(vec![else_const], vec![], vec!["else_out"], vec![]);

    let mut if_node = NodeProto::default();
    if_node.op_type = "If".to_string();
    if_node.input.push("condition".to_string());
    if_node.output.push("output".to_string());
    if_node.attribute.push(make_attr_graph("then_branch", then_graph));
    if_node.attribute.push(make_attr_graph("else_branch", else_graph));
    graph.node.push(if_node);

    let mut output = ValueInfoProto::default();
    output.name = "output".to_string();
    graph.output.push(output);

    model.graph = Some(graph);

    let mut importer = OnnxImporter::new();
    let outputs = importer.import_model(model).unwrap();

    let result = outputs.get("output").unwrap();
    let arr = result.to_ndarray::<f32>().unwrap();
    let vals: Vec<f32> = arr.iter().copied().collect();
    // condition=true, shapes differ -> eager fallback selects then_branch
    assert_eq!(vals, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_if_with_parent_scope() {
    // Subgraph references a tensor ("parent_val") from the parent graph scope
    let mut model = ModelProto::default();
    let mut graph = GraphProto::default();
    graph.name = "if_parent_scope_test".to_string();

    // Condition: true
    let mut cond_input = ValueInfoProto::default();
    cond_input.name = "condition".to_string();
    graph.input.push(cond_input);
    let mut cond_init = TensorProto::default();
    cond_init.name = "condition".to_string();
    cond_init.data_type = tensor_proto::DataType::Bool as i32;
    cond_init.dims = vec![];
    cond_init.raw_data = vec![1u8];
    graph.initializer.push(cond_init);

    // Parent value: [100, 200, 300]
    let mut pv_input = ValueInfoProto::default();
    pv_input.name = "parent_val".to_string();
    graph.input.push(pv_input);
    let mut pv_init = TensorProto::default();
    pv_init.name = "parent_val".to_string();
    pv_init.data_type = tensor_proto::DataType::Float as i32;
    pv_init.dims = vec![3];
    pv_init.raw_data = [100.0f32, 200.0, 300.0].iter().flat_map(|v| v.to_le_bytes()).collect();
    graph.initializer.push(pv_init);

    // Then branch: Identity of parent_val (references parent scope)
    let then_id = {
        let mut n = NodeProto::default();
        n.op_type = "Identity".to_string();
        n.input.push("parent_val".to_string());
        n.output.push("then_out".to_string());
        n
    };
    let then_graph = make_graph(vec![then_id], vec![], vec!["then_out"], vec![]);

    // Else branch: Constant [0]
    let else_const = {
        let mut n = NodeProto::default();
        n.op_type = "Constant".to_string();
        n.output.push("else_out".to_string());
        n.attribute.push(make_attr_tensor(
            "value",
            make_tensor_proto(
                [0.0f32, 0.0, 0.0].iter().flat_map(|v| v.to_le_bytes()).collect(),
                vec![3],
                tensor_proto::DataType::Float as i32,
            ),
        ));
        n
    };
    let else_graph = make_graph(vec![else_const], vec![], vec!["else_out"], vec![]);

    let mut if_node = NodeProto::default();
    if_node.op_type = "If".to_string();
    if_node.input.push("condition".to_string());
    if_node.output.push("output".to_string());
    if_node.attribute.push(make_attr_graph("then_branch", then_graph));
    if_node.attribute.push(make_attr_graph("else_branch", else_graph));
    graph.node.push(if_node);

    let mut output = ValueInfoProto::default();
    output.name = "output".to_string();
    graph.output.push(output);

    model.graph = Some(graph);

    let mut importer = OnnxImporter::new();
    let outputs = importer.import_model(model).unwrap();

    let result = outputs.get("output").unwrap();
    let arr = result.to_ndarray::<f32>().unwrap();
    let vals: Vec<f32> = arr.iter().copied().collect();
    // condition=true, then_branch uses parent_val -> [100, 200, 300]
    assert_eq!(vals, vec![100.0, 200.0, 300.0]);
}

#[test]
fn test_if_nested() {
    // Outer If (cond=true) -> then_branch contains inner If (cond=false)
    // Inner If: then_branch = x + 1, else_branch = x + 2
    // Since outer=true, inner=false => result should be x + 2
    let mut model = ModelProto::default();
    let mut graph = GraphProto::default();
    graph.name = "nested_if_test".to_string();

    // Initializers
    for (name, val) in [("outer_cond", true), ("inner_cond", false)] {
        let mut inp = ValueInfoProto::default();
        inp.name = name.to_string();
        graph.input.push(inp);
        let mut init = TensorProto::default();
        init.name = name.to_string();
        init.data_type = tensor_proto::DataType::Bool as i32;
        init.dims = vec![];
        init.raw_data = vec![val as u8];
        graph.initializer.push(init);
    }

    let mut x_input = ValueInfoProto::default();
    x_input.name = "x".to_string();
    graph.input.push(x_input);
    let mut x_init = TensorProto::default();
    x_init.name = "x".to_string();
    x_init.data_type = tensor_proto::DataType::Float as i32;
    x_init.dims = vec![2];
    x_init.raw_data = [10.0f32, 20.0].iter().flat_map(|v| v.to_le_bytes()).collect();
    graph.initializer.push(x_init);

    // Inner then: x + 1
    let inner_then = {
        let c = {
            let mut n = NodeProto::default();
            n.op_type = "Constant".to_string();
            n.output.push("one".to_string());
            n.attribute.push(make_attr_tensor(
                "value",
                make_tensor_proto(1.0f32.to_le_bytes().to_vec(), vec![1], tensor_proto::DataType::Float as i32),
            ));
            n
        };
        let add = {
            let mut n = NodeProto::default();
            n.op_type = "Add".to_string();
            n.input.push("x".to_string());
            n.input.push("one".to_string());
            n.output.push("inner_then_out".to_string());
            n
        };
        make_graph(vec![c, add], vec!["x"], vec!["inner_then_out"], vec![])
    };

    // Inner else: x + 2
    let inner_else = {
        let c = {
            let mut n = NodeProto::default();
            n.op_type = "Constant".to_string();
            n.output.push("two".to_string());
            n.attribute.push(make_attr_tensor(
                "value",
                make_tensor_proto(2.0f32.to_le_bytes().to_vec(), vec![1], tensor_proto::DataType::Float as i32),
            ));
            n
        };
        let add = {
            let mut n = NodeProto::default();
            n.op_type = "Add".to_string();
            n.input.push("x".to_string());
            n.input.push("two".to_string());
            n.output.push("inner_else_out".to_string());
            n
        };
        make_graph(vec![c, add], vec!["x"], vec!["inner_else_out"], vec![])
    };

    // Inner If node (inside outer then_branch)
    let inner_if = {
        let mut n = NodeProto::default();
        n.op_type = "If".to_string();
        n.input.push("inner_cond".to_string()); // references parent scope
        n.output.push("outer_then_out".to_string());
        n.attribute.push(make_attr_graph("then_branch", inner_then));
        n.attribute.push(make_attr_graph("else_branch", inner_else));
        n
    };
    let outer_then = make_graph(vec![inner_if], vec![], vec!["outer_then_out"], vec![]);

    // Outer else: constant [0, 0]
    let outer_else = {
        let c = {
            let mut n = NodeProto::default();
            n.op_type = "Constant".to_string();
            n.output.push("outer_else_out".to_string());
            n.attribute.push(make_attr_tensor(
                "value",
                make_tensor_proto(
                    [0.0f32, 0.0].iter().flat_map(|v| v.to_le_bytes()).collect(),
                    vec![2],
                    tensor_proto::DataType::Float as i32,
                ),
            ));
            n
        };
        make_graph(vec![c], vec![], vec!["outer_else_out"], vec![])
    };

    // Outer If node
    let mut outer_if = NodeProto::default();
    outer_if.op_type = "If".to_string();
    outer_if.input.push("outer_cond".to_string());
    outer_if.output.push("output".to_string());
    outer_if.attribute.push(make_attr_graph("then_branch", outer_then));
    outer_if.attribute.push(make_attr_graph("else_branch", outer_else));
    graph.node.push(outer_if);

    let mut output = ValueInfoProto::default();
    output.name = "output".to_string();
    graph.output.push(output);

    model.graph = Some(graph);

    let mut importer = OnnxImporter::new();
    let outputs = importer.import_model(model).unwrap();

    let result = outputs.get("output").unwrap();
    let arr = result.to_ndarray::<f32>().unwrap();
    let vals: Vec<f32> = arr.iter().copied().collect();
    // outer_cond=true -> outer then_branch executes
    // inner_cond=false -> inner else_branch executes (x + 2)
    // x = [10, 20], so result = [12, 22]
    assert_eq!(vals, vec![12.0, 22.0]);
}

#[test]
fn test_load_silero_vad_prepare() {
    use prost::Message;
    use std::fs::File;
    use std::io::BufReader;

    let file = File::open("audio.onnx").expect("audio.onnx not found");
    let mut reader = BufReader::new(file);
    let mut bytes = Vec::new();
    std::io::Read::read_to_end(&mut reader, &mut bytes).unwrap();

    let model = ModelProto::decode(&bytes[..]).expect("Failed to decode ONNX");

    let importer = OnnxImporter::new();
    let graph = importer.prepare(model).expect("Failed to prepare graph");

    println!("Inputs: {:?}", graph.input_names());
    println!("Outputs: {:?}", graph.output_names());
    println!("Num initializers: {}", graph.initializers.len());
    println!("Num nodes: {}", graph.nodes.len());

    let mut op_counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for node in &graph.nodes {
        *op_counts.entry(node.op_type.as_str()).or_insert(0) += 1;
    }

    let mut ops: Vec<_> = op_counts.into_iter().collect();
    ops.sort_by(|a, b| b.1.cmp(&a.1));

    println!("\nOperators (sorted by frequency):");
    for (op, count) in ops {
        println!("  {:4}x  {}", count, op);
    }
}
