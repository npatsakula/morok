pub(crate) use crate::parser::onnx::{
    AttributeProto, GraphProto, ModelProto, NodeProto, StringStringEntryProto, TensorProto, ValueInfoProto,
    tensor_proto,
};
pub(crate) use crate::registry::*;
pub(crate) use morok_dtype::{DType, ScalarDType};
pub(crate) use morok_tensor::Tensor;

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
