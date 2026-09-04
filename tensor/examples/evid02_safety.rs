use std::collections::HashMap;
use std::sync::Arc;

use serde_json::{Value, json};
use svod_dtype::{AmdArch, DType, DeviceSpec};
use svod_ir::ops;
use svod_ir::{ConstValue, Op, UOp};
use svod_schedule::{
    HeuristicsConfig, OptStrategy, OptimizerConfig, OptimizerRenderer, TcOptLevel, TcSelect,
    optimize_kernel_with_config_and_final_rewrite,
};
use svod_tensor::Tensor;

const PIN: &str = "8c8b43de62515abe6c820b1de5aa26b30f48e43a";

fn dtype_name(dtype: &DType) -> String {
    format!("{:?}", dtype.base()).to_ascii_lowercase()
}

fn op_name(op: &Op) -> String {
    match op {
        Op::Binary(kind, ..) => format!("{kind:?}").to_ascii_uppercase(),
        Op::Ternary(kind, ..) => format!("{kind:?}").to_ascii_uppercase(),
        Op::Unary(kind, ..) => format!("{kind:?}").to_ascii_uppercase(),
        Op::EndIf(..) => "ENDIF".into(),
        Op::Wmma(..) => "WMMA".into(),
        other => other.as_ref().to_ascii_uppercase(),
    }
}

fn node_arg(node: &Arc<UOp>) -> Value {
    match node.op() {
        Op::Const(value) => match value.0 {
            ConstValue::Int(value) => json!({"kind": "int", "value": value}),
            ConstValue::UInt(value) => json!({"kind": "int", "value": value}),
            ConstValue::Float(value) => json!({"kind": "float", "value": value}),
            ConstValue::Bool(value) => json!({"kind": "bool", "value": value}),
            ConstValue::Invalid => json!({"kind": "invalid"}),
        },
        Op::Param(ops::Param { arg, .. }) => json!({"slot": arg.slot}),
        Op::Special(ops::Special { name, .. }) => json!({"name": name}),
        Op::Wmma(ops::Wmma { metadata, .. }) => json!({
            "dims": [metadata.dims.0, metadata.dims.1, metadata.dims.2],
            "input_dtype": dtype_name(&metadata.dtype_in),
            "device": metadata.device.canonical(),
            "threads": metadata.threads,
            "upcast_axes": metadata.upcast_axes,
        }),
        _ => Value::Null,
    }
}

fn graph(name: &str, root: &Arc<UOp>) -> Value {
    let nodes = root.toposort();
    let ids: HashMap<u64, usize> = nodes.iter().enumerate().map(|(id, node)| (node.id, id)).collect();
    let table = nodes
        .iter()
        .enumerate()
        .map(|(id, node)| {
            let shape = node.shape().expect("EVID-02 node shape").map(|shape| {
                shape
                    .iter()
                    .map(|extent| extent.as_const().expect("EVID-02 requires constant node shapes"))
                    .collect::<Vec<_>>()
            });
            json!({
                "id": id,
                "op": op_name(node.op()),
                "dtype": dtype_name(&node.dtype()),
                "shape": shape,
                "src": node.op().sources().iter().map(|source| ids[&source.id]).collect::<Vec<_>>(),
                "arg": node_arg(node),
            })
        })
        .collect::<Vec<_>>();
    json!({"name": name, "root": ids[&root.id], "nodes": table})
}

fn main() {
    let a = Tensor::empty(&[5, 16], DType::Float16);
    let b = Tensor::empty(&[16, 16], DType::Float16);
    let c = a.matmul_with().other(&b).dtype(DType::Float32).call().expect("tensor matmul");
    let rangeified = svod_schedule::rangeify_with_map(UOp::sink(vec![c.uop().contiguous()])).expect("rangeify");
    let (kernel_graph, _) = svod_schedule::try_get_kernel_graph(rangeified.sink).expect("split kernels");
    let pre = svod_tensor::schedule::create_pre_schedule(kernel_graph).expect("schedule");
    assert_eq!(pre.items.len(), 1);
    let heuristics = HeuristicsConfig::builder()
        .tc_opt(TcOptLevel::Padded)
        .tc_select(TcSelect::Index(0))
        .matvec_enabled(false)
        .build();
    let config = OptimizerConfig::builder().strategy(OptStrategy::Heuristic).heuristics(heuristics).build();
    let renderer = OptimizerRenderer::for_amd_arch(AmdArch::Gfx1151).with_rewrite_capabilities(
        svod_ir::RendererOps::all(),
        None,
        None,
    );
    let (final_rewrite, optimized) =
        optimize_kernel_with_config_and_final_rewrite(pre.items[0].ast.clone(), &renderer, &config).expect("optimize");
    let program = svod_codegen::program_pipeline::program_from_sink(optimized, DeviceSpec::Amd { device_id: 0 })
        .expect("PROGRAM");
    let linearized = svod_codegen::program_pipeline::do_linearize(&program).expect("linearize");
    let linear = linearized.toposort().into_iter().find(|node| matches!(node.op(), Op::Linear(..))).expect("LINEAR");
    let document = json!({
        "schema_version": 2,
        "evidence": "EVID-02",
        "reference": PIN,
        "fixture": {"m": 5, "k": 16, "n": 16, "input_dtype": "float16", "accumulator_dtype": "float32", "target": "gfx1151"},
        "stages": [graph("late-final-rewrite", &final_rewrite), graph("linearized", &linear)],
    });
    println!("{}", serde_json::to_string_pretty(&document).unwrap());
}
