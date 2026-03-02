//! ONNX model importer - converts ONNX protobuf to Morok Tensors.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use morok_dtype::DType;
use morok_tensor::Tensor;
use prost::Message;
use snafu::ResultExt;

use crate::error::{EmptyModelSnafu, IoSnafu, MissingInputSnafu, ProtobufDecodeSnafu, Result};
use crate::parser::onnx::{GraphProto, ModelProto, NodeProto, ValueInfoProto};
use crate::registry::{
    OpRegistry, OpSetId, convert_onnx_dtype, onnx_opset_version, tensor_from_proto_ext, tensor_to_bool_scalar,
};

/// Dimension value - either static (known size) or dynamic (named, e.g., batch dim).
#[derive(Debug, Clone, PartialEq)]
pub enum DimValue {
    /// Static dimension with known size
    Static(usize),
    /// Dynamic dimension with symbolic name (e.g., "batch_size")
    Dynamic(String),
}

/// Input specification extracted from ONNX ValueInfoProto.
#[derive(Debug, Clone)]
pub struct InputSpec {
    /// Shape dimensions (can be static or dynamic)
    pub shape: Vec<DimValue>,
    /// Data type
    pub dtype: DType,
    /// Whether this input is optional
    pub optional: bool,
}

impl InputSpec {
    /// Create a new input spec.
    pub fn new(shape: Vec<DimValue>, dtype: DType, optional: bool) -> Self {
        Self { shape, dtype, optional }
    }

    /// Check if all dimensions are static (no dynamic dims).
    pub fn is_static(&self) -> bool {
        self.shape.iter().all(|d| matches!(d, DimValue::Static(_)))
    }

    /// Get static shape if all dimensions are known.
    pub fn static_shape(&self) -> Option<Vec<usize>> {
        self.shape
            .iter()
            .map(|d| match d {
                DimValue::Static(s) => Some(*s),
                DimValue::Dynamic(_) => None,
            })
            .collect()
    }
}

/// A prepared subgraph for control-flow operators (If, Loop, Scan).
#[allow(dead_code)]
pub(crate) struct SubGraph {
    pub initializers: HashMap<String, Tensor>,
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
    pub nodes: Vec<NodeProto>,
    /// Nested subgraph attributes within this subgraph's nodes.
    pub subgraphs: HashMap<(usize, String), SubGraph>,
}

/// Prepared ONNX graph - structure extracted, ready for execution.
pub struct OnnxGraph {
    /// Input specifications (name -> spec)
    pub inputs: HashMap<String, InputSpec>,
    /// Output names in order
    pub outputs: Vec<String>,
    /// Initializers/weights (name -> tensor)
    pub initializers: HashMap<String, Tensor>,
    /// Nodes in topological execution order
    pub nodes: Vec<NodeProto>,
    /// Opset versions
    pub opsets: Vec<OpSetId>,
    /// Pre-parsed subgraph attributes: (node_index, attr_name) -> SubGraph
    pub(crate) subgraphs: HashMap<(usize, String), SubGraph>,
}

impl OnnxGraph {
    /// Get the list of input names.
    pub fn input_names(&self) -> Vec<&str> {
        self.inputs.keys().map(|s| s.as_str()).collect()
    }

    /// Get the list of output names.
    pub fn output_names(&self) -> &[String] {
        &self.outputs
    }

    /// Check if an input is optional.
    pub fn is_input_optional(&self, name: &str) -> bool {
        self.inputs.get(name).map(|s| s.optional).unwrap_or(false)
    }
}

/// ONNX model importer.
///
/// Converts ONNX models to Morok Tensors using a two-phase approach:
/// 1. `prepare()` - Extract graph structure without executing
/// 2. `execute()` - Run graph with provided inputs
pub struct OnnxImporter {
    /// Operator registry for dispatch
    registry: OpRegistry,
    /// Directory containing the model file (for external data loading)
    model_dir: Option<std::path::PathBuf>,
}

impl OnnxImporter {
    /// Create a new ONNX importer.
    pub fn new() -> Self {
        Self { registry: OpRegistry::new(), model_dir: None }
    }

    /// Import ONNX model from file path (convenience method for all-initializer models).
    pub fn import_path<P: AsRef<Path>>(&mut self, path: P) -> Result<HashMap<String, Tensor>> {
        self.model_dir = path.as_ref().parent().map(|p| p.to_path_buf());
        let file = File::open(path.as_ref()).context(IoSnafu)?;
        let mut reader = BufReader::new(file);
        self.import_reader(&mut reader)
    }

    /// Import ONNX model from a reader.
    pub fn import_reader<R: Read>(&mut self, reader: &mut R) -> Result<HashMap<String, Tensor>> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).context(IoSnafu)?;
        self.import_bytes(&bytes)
    }

    /// Import from bytes.
    pub fn import_bytes(&mut self, bytes: &[u8]) -> Result<HashMap<String, Tensor>> {
        let model = ModelProto::decode(bytes).context(ProtobufDecodeSnafu)?;
        self.import_model(model)
    }

    /// Import from parsed ModelProto (convenience for all-initializer models).
    ///
    /// For models with runtime inputs, use `prepare()` + `execute()` instead.
    pub fn import_model(&mut self, model: ModelProto) -> Result<HashMap<String, Tensor>> {
        let graph = self.prepare(model)?;

        // If all inputs have initializers, we can execute directly
        let has_runtime_inputs = graph.inputs.keys().any(|name| !graph.initializers.contains_key(name));

        if has_runtime_inputs {
            // Need user inputs - return empty map, user should use execute()
            // Actually, for backward compat, let's try with empty inputs and see what happens
            // This will error on missing inputs, which is expected
            let inputs = HashMap::new();
            self.execute_with_initializers(&graph, inputs)
        } else {
            // All inputs are initializers - can execute with empty runtime inputs
            let inputs = HashMap::new();
            self.execute_with_initializers(&graph, inputs)
        }
    }

    /// Phase 1: Extract graph structure from ONNX model.
    ///
    /// Returns an `OnnxGraph` that can be executed multiple times with different inputs.
    pub fn prepare(&self, model: ModelProto) -> Result<OnnxGraph> {
        let proto_graph = model.graph.ok_or_else(|| EmptyModelSnafu.build())?;

        // Collect opsets
        let opsets: Vec<OpSetId> =
            model.opset_import.iter().map(|op| OpSetId { domain: op.domain.clone(), version: op.version }).collect();

        // Build initializer map (weights/constants)
        let mut initializers: HashMap<String, Tensor> = HashMap::new();
        let initializer_names: Vec<String> = proto_graph.initializer.iter().map(|i| i.name.clone()).collect();

        for init in &proto_graph.initializer {
            if !init.name.is_empty() {
                let tensor = tensor_from_proto_ext(init, self.model_dir.as_deref())?;
                initializers.insert(init.name.clone(), tensor);
            }
        }

        // Extract input specs (excluding initializers)
        let mut inputs: HashMap<String, InputSpec> = HashMap::new();
        for input in &proto_graph.input {
            if input.name.is_empty() {
                continue;
            }
            // Skip if this input is an initializer
            if initializer_names.contains(&input.name) {
                continue;
            }
            // Extract input spec from ValueInfoProto
            if let Some(spec) = self.extract_input_spec(input)? {
                inputs.insert(input.name.clone(), spec);
            }
        }

        // Collect output names
        let outputs: Vec<String> =
            proto_graph.output.iter().filter(|o| !o.name.is_empty()).map(|o| o.name.clone()).collect();

        // Collect nodes
        let nodes = proto_graph.node;

        // Pre-parse subgraph attributes for control-flow ops (If, Loop, Scan)
        let mut subgraphs: HashMap<(usize, String), SubGraph> = HashMap::new();
        for (idx, node) in nodes.iter().enumerate() {
            for attr in &node.attribute {
                if let Some(ref graph) = attr.g {
                    let sub = self.prepare_subgraph(graph)?;
                    subgraphs.insert((idx, attr.name.clone()), sub);
                }
            }
        }

        Ok(OnnxGraph { inputs, outputs, initializers, nodes, opsets, subgraphs })
    }

    /// Extract InputSpec from ValueInfoProto.
    fn extract_input_spec(&self, input: &ValueInfoProto) -> Result<Option<InputSpec>> {
        // ValueInfoProto has a `type` field (TypeProto) - use r#type for keyword
        // TypeProto has a oneof `value` containing tensor_type, sequence_type, etc.
        use crate::parser::onnx::tensor_shape_proto::dimension::Value as DimValueProto;
        use crate::parser::onnx::type_proto::Value;

        let type_proto = match input.r#type.as_ref() {
            Some(t) => t,
            None => return Ok(None),
        };

        // Access the oneof value
        let tensor_type = match &type_proto.value {
            Some(Value::TensorType(tt)) => tt,
            _ => return Ok(None), // Not a tensor type
        };

        // Extract dtype
        let dtype = convert_onnx_dtype(tensor_type.elem_type)?;

        // Extract shape from tensor_type.shape
        let shape: Vec<DimValue> = tensor_type
            .shape
            .as_ref()
            .map(|s| {
                s.dim
                    .iter()
                    .map(|d| match &d.value {
                        Some(DimValueProto::DimValue(v)) => {
                            if *v > 0 {
                                DimValue::Static(*v as usize)
                            } else {
                                DimValue::Dynamic(String::new())
                            }
                        }
                        Some(DimValueProto::DimParam(name)) => DimValue::Dynamic(name.clone()),
                        None => DimValue::Dynamic(String::new()),
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(Some(InputSpec::new(shape, dtype, false)))
    }

    /// Phase 2: Execute the graph with provided inputs.
    ///
    /// Returns a HashMap mapping output names to their tensor values.
    pub fn execute(&self, graph: &OnnxGraph, inputs: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
        self.execute_with_initializers(graph, inputs)
    }

    /// Execute graph with initializers merged into values.
    fn execute_with_initializers(
        &self,
        graph: &OnnxGraph,
        inputs: HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        // Merge initializers into values
        let mut values: HashMap<String, Tensor> = graph.initializers.clone();

        // Add user inputs
        for (name, tensor) in inputs {
            values.insert(name, tensor);
        }

        // Validate required inputs are present
        for (name, spec) in &graph.inputs {
            if !values.contains_key(name) && !spec.optional {
                return MissingInputSnafu { node: "graph_input".to_string(), input: name.clone() }.fail();
            }
        }

        // Resolve the default (ai.onnx) opset version
        let opset_version = onnx_opset_version(&graph.opsets, "");

        // Process nodes in order (ONNX guarantees topological order)
        for (node_index, node) in graph.nodes.iter().enumerate() {
            let node_opset = if node.domain.is_empty() || node.domain == "ai.onnx" {
                opset_version
            } else {
                onnx_opset_version(&graph.opsets, &node.domain)
            };

            if node.op_type == "If" {
                self.process_if_node(node_index, node, &mut values, node_opset, &graph.subgraphs)?;
            } else {
                self.process_node(node, &mut values, node_opset)?;
            }
        }

        // Collect outputs by name
        let outputs: HashMap<String, Tensor> =
            graph.outputs.iter().filter_map(|name| values.get(name).cloned().map(|t| (name.clone(), t))).collect();

        Ok(outputs)
    }

    /// Prepare a subgraph from a GraphProto embedded in a node attribute.
    /// Recursively extracts nested subgraph attributes (e.g., If inside If).
    fn prepare_subgraph(&self, graph: &GraphProto) -> Result<SubGraph> {
        let mut initializers: HashMap<String, Tensor> = HashMap::new();
        for init in &graph.initializer {
            if !init.name.is_empty() {
                let tensor = tensor_from_proto_ext(init, self.model_dir.as_deref())?;
                initializers.insert(init.name.clone(), tensor);
            }
        }

        let input_names: Vec<String> = graph
            .input
            .iter()
            .filter(|i| !i.name.is_empty() && !initializers.contains_key(&i.name))
            .map(|i| i.name.clone())
            .collect();

        let output_names: Vec<String> =
            graph.output.iter().filter(|o| !o.name.is_empty()).map(|o| o.name.clone()).collect();

        let nodes = graph.node.clone();

        // Recursively extract nested subgraph attributes
        let mut subgraphs: HashMap<(usize, String), SubGraph> = HashMap::new();
        for (idx, node) in nodes.iter().enumerate() {
            for attr in &node.attribute {
                if let Some(ref nested_graph) = attr.g {
                    let sub = self.prepare_subgraph(nested_graph)?;
                    subgraphs.insert((idx, attr.name.clone()), sub);
                }
            }
        }

        Ok(SubGraph { initializers, input_names, output_names, nodes, subgraphs })
    }

    /// Execute a pre-parsed subgraph with access to parent scope values.
    fn execute_subgraph(
        &self,
        subgraph: &SubGraph,
        parent_values: &HashMap<String, Tensor>,
        opset_version: i64,
    ) -> Result<Vec<Tensor>> {
        // Start with subgraph's own initializers
        let mut values: HashMap<String, Tensor> = subgraph.initializers.clone();

        // Merge parent scope (subgraph initializers take precedence)
        for (k, v) in parent_values {
            values.entry(k.clone()).or_insert_with(|| v.clone());
        }

        // Execute subgraph nodes (subgraphs inherit the parent's opset).
        // Control-flow ops (If) are intercepted here, matching execute_with_initializers.
        for (node_index, node) in subgraph.nodes.iter().enumerate() {
            if node.op_type == "If" {
                self.process_if_node(node_index, node, &mut values, opset_version, &subgraph.subgraphs)?;
            } else {
                self.process_node(node, &mut values, opset_version)?;
            }
        }

        // Collect declared outputs in order
        subgraph
            .output_names
            .iter()
            .map(|name| {
                values.get(name).cloned().ok_or_else(|| crate::Error::IrConstruction {
                    details: format!("subgraph output '{name}' not found in values"),
                })
            })
            .collect()
    }

    /// Process an ONNX If node: execute both branches, select outputs.
    fn process_if_node(
        &self,
        node_index: usize,
        node: &NodeProto,
        values: &mut HashMap<String, Tensor>,
        opset_version: i64,
        subgraphs: &HashMap<(usize, String), SubGraph>,
    ) -> Result<()> {
        let condition = values
            .get(&node.input[0])
            .ok_or_else(|| crate::Error::MissingInput { node: node.name.clone(), input: node.input[0].clone() })?
            .clone();

        let then_branch = subgraphs
            .get(&(node_index, "then_branch".to_string()))
            .ok_or_else(|| crate::Error::IrConstruction { details: "If node missing then_branch attribute".into() })?;
        let else_branch = subgraphs
            .get(&(node_index, "else_branch".to_string()))
            .ok_or_else(|| crate::Error::IrConstruction { details: "If node missing else_branch attribute".into() })?;

        // Execute both branches (lazy — builds tensor graphs without computing)
        let then_outputs = self.execute_subgraph(then_branch, values, opset_version)?;
        let else_outputs = self.execute_subgraph(else_branch, values, opset_version)?;

        if then_outputs.len() != else_outputs.len() {
            return Err(crate::Error::IrConstruction {
                details: format!(
                    "If: then_branch ({}) and else_branch ({}) must produce the same number of outputs",
                    then_outputs.len(),
                    else_outputs.len()
                ),
            });
        }

        // Select outputs: lazy where_ if shapes match, eager fallback otherwise
        let selected: Vec<Tensor> = {
            let shapes_match = then_outputs
                .iter()
                .zip(else_outputs.iter())
                .all(|(t, e)| t.shape().ok() == e.shape().ok() && t.shape().is_ok());

            if shapes_match {
                then_outputs
                    .iter()
                    .zip(else_outputs.iter())
                    .map(|(t, e)| t.where_(&condition, e).map_err(Into::into))
                    .collect::<Result<_>>()?
            } else {
                let cond = tensor_to_bool_scalar(&condition)?;
                if cond { then_outputs } else { else_outputs }
            }
        };

        for (i, output_name) in node.output.iter().enumerate() {
            if !output_name.is_empty()
                && let Some(tensor) = selected.get(i)
            {
                values.insert(output_name.clone(), tensor.clone());
            }
        }

        Ok(())
    }

    /// Process a single ONNX node.
    fn process_node(&self, node: &NodeProto, values: &mut HashMap<String, Tensor>, opset_version: i64) -> Result<()> {
        let op_type = &node.op_type;
        let domain = &node.domain;
        let node_name = if node.name.is_empty() { "unnamed" } else { &node.name };

        // Collect input Tensors, preserving positional indices for optional inputs.
        // Empty input names in ONNX mean "optional, not provided" — we represent
        // these as None to keep correct positional indexing for operators like Clip.
        let mut inputs: Vec<Option<Tensor>> = Vec::new();
        for input_name in &node.input {
            if input_name.is_empty() {
                inputs.push(None);
            } else {
                match values.get(input_name) {
                    Some(tensor) => inputs.push(Some(tensor.clone())),
                    None => {
                        return Err(crate::Error::MissingInput {
                            node: node_name.to_string(),
                            input: input_name.clone(),
                        });
                    }
                }
            }
        }

        // Dispatch to operator registry - may return multiple outputs
        let outputs = self.registry.dispatch_multi(op_type, domain, &inputs, node, opset_version)?;

        // Register outputs by name
        for (i, output_name) in node.output.iter().enumerate() {
            if let Some(output_tensor) = outputs.get(i) {
                values.insert(output_name.clone(), output_tensor.clone());
            }
        }

        Ok(())
    }
}

impl Default for OnnxImporter {
    fn default() -> Self {
        Self::new()
    }
}
