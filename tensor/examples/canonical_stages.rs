use std::collections::HashMap;
use std::sync::Arc;

use svod_dtype::{DType, DeviceSpec};
use svod_ir::ops;
use svod_ir::{ConstValue, Op, UOp};
use svod_schedule::optimizer::{OptStrategy, OptimizerConfig, Renderer};

fn tensor_graph(multi_output: bool) -> Arc<UOp> {
    let input = UOp::param(0, 64, DType::Float32, Some(DeviceSpec::Cpu));
    let add = input.try_add(&input.const_like(1.0f32)).expect("fixture elementwise add");
    let add = if multi_output {
        add
    } else {
        let variable = UOp::variable("schedule_n".into(), 1, 8, DType::Int32);
        let bound = variable.bind(UOp::const_(DType::Int32, ConstValue::Int(4)));
        add.try_add(&bound.cast(DType::Float32)).expect("fixture bound variable add")
    };
    if multi_output {
        let mul = input.try_mul(&input.const_like(2.0f32)).expect("fixture elementwise multiply");
        UOp::sink(vec![add.contiguous(), mul.contiguous()])
    } else {
        let add = add.contiguous();
        let mul = add.try_mul(&add.const_like(2.0f32)).expect("fixture chained multiply");
        UOp::sink(vec![mul.contiguous()])
    }
}

fn kernel_body(root: &Arc<UOp>, index: usize) -> Arc<UOp> {
    root.toposort_call_aware(false)
        .into_iter()
        .filter_map(|node| match node.op() {
            Op::Call(ops::Call { body, .. }) => Some(body.clone()),
            _ => None,
        })
        .nth(index)
        .expect("production fixture must callify at least one kernel")
}

fn finish_if_requested(requested: &str, stage: &str, path: &str) -> bool {
    if requested != stage {
        return false;
    }
    let json = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("requested stage {stage:?} was not captured at {path}: {error}"));
    print!("{json}");
    true
}

fn canonical_slot(slot: usize) -> i128 {
    if slot == usize::MAX { -1 } else { slot as i128 }
}

fn canonical_schedule_slot(buffer: &Arc<UOp>, slot: usize) -> i128 {
    if buffer.tag().as_ref().is_some_and(|tags| tags.contains(&svod_ir::uop::canonical::TAG_SCHEDULE_LOCAL_BUFFER)) {
        // Tinygrad's scheduled BUFFER descriptors are allocation-local; the
        // argument/global slot fields carry ordering and identity separately.
        0
    } else {
        canonical_slot(slot)
    }
}

fn schedule_buffer(source: &Arc<UOp>, argument_index: usize) -> serde_json::Value {
    let buffer = source.buf_uop();
    let (origin, slot) = match buffer.op() {
        Op::Param(ops::Param { arg, .. }) => ("PARAM", arg.slot),
        Op::Buffer(ops::Buffer { arg, .. }) => ("BUFFER", arg.slot),
        other => panic!("scheduled buffer argument must resolve to PARAM or BUFFER, got {other:?}"),
    };
    serde_json::json!({
        "argument_index": argument_index,
        "global_slot": argument_index,
        "buffer_slot": canonical_schedule_slot(&buffer, slot),
        "origin": origin,
    })
}

fn ast_output_slots(ast: &Arc<UOp>) -> Vec<i128> {
    let mut slots: Vec<_> = ast
        .toposort()
        .into_iter()
        .filter_map(|node| {
            let Op::Store(ops::Store { index, .. }) = node.op() else { return None };
            let buffer = match index.op() {
                Op::Index(ops::Index { buffer, .. }) => buffer.buf_uop(),
                _ => index.buf_uop(),
            };
            match buffer.op() {
                Op::Param(ops::Param { arg, .. }) | Op::Buffer(ops::Buffer { arg, .. }) => {
                    Some(canonical_slot(arg.slot))
                }
                _ => None,
            }
        })
        .collect();
    slots.sort_unstable();
    slots.dedup();
    slots
}

fn current_var_vals(root: &Arc<UOp>) -> HashMap<String, i64> {
    root.toposort()
        .into_iter()
        .filter_map(|node| {
            let Op::Bind(ops::Bind { var, value }) = node.op() else { return None };
            let Op::Param(ops::Param { arg, .. }) = var.op() else { return None };
            let name = arg.name.as_ref()?;
            let Op::Const(value) = value.op() else { return None };
            Some((name.clone(), value.0.try_int()?))
        })
        .collect()
}

fn capture_schedule(
    path: &str,
    pre: &svod_tensor::schedule::PreSchedule,
    var_vals: &HashMap<String, i64>,
) -> serde_json::Value {
    let descriptors: HashMap<_, _> =
        pre.items.iter().enumerate().map(|(index, item)| (item.kernel.id, index)).collect();
    let mut latest_item = HashMap::new();
    let mut item_buffers = Vec::with_capacity(pre.invocations.len());
    let mut items = Vec::with_capacity(pre.invocations.len());

    for (order, invocation) in pre.invocations.iter().enumerate() {
        let callable_index = descriptors[&invocation.kernel_id];
        let descriptor = &pre.items[callable_index];
        let buffers: Vec<_> = descriptor
            .sources
            .iter()
            .filter(|source| !matches!(source.op(), Op::Bind(..)))
            .enumerate()
            .map(|(argument_index, source)| schedule_buffer(source, argument_index))
            .collect();
        let buffer_ids: Vec<_> = descriptor
            .sources
            .iter()
            .filter(|source| !matches!(source.op(), Op::Bind(..)))
            .map(|source| source.buf_uop().id)
            .collect();
        let mut dependencies: Vec<_> = descriptor
            .dependencies
            .iter()
            .map(|dependency| {
                *latest_item
                    .get(dependency)
                    .unwrap_or_else(|| panic!("schedule dependency {dependency} does not precede item {order}"))
            })
            .collect();
        dependencies.sort_unstable();
        dependencies.dedup();
        let ast = svod_ir::CanonicalGraph::from_root("kernel_ast", &descriptor.ast).expect("canonical scheduled AST");
        let ast = serde_json::to_value(ast).expect("canonical scheduled AST JSON");
        let variables: HashMap<_, _> = ast["nodes"]
            .as_array()
            .expect("canonical AST nodes")
            .iter()
            .filter_map(|node| {
                let arg = &node["arg"];
                if node["op"] == "PARAM" && arg["kind"] == "param" && arg["address_space"].is_null() {
                    let name = arg["name"].as_str()?;
                    return Some((
                        name.to_string(),
                        serde_json::json!({"kind": "param", "slot": arg["slot"], "name": name, "dtype": node["dtype"]}),
                    ));
                }
                if node["op"] == "DEFINE_VAR" && arg["kind"] == "define_var" {
                    let name = arg["name"].as_str()?;
                    return Some((
                        name.to_string(),
                        serde_json::json!({"kind": "define_var", "slot": null, "name": name, "dtype": node["dtype"]}),
                    ));
                }
                None
            })
            .collect();
        let mut bindings = Vec::new();
        for (name, value, schedule_loop) in var_vals
            .iter()
            .map(|(name, value)| (name, value, false))
            .chain(invocation.fixedvars.iter().map(|(name, value)| (name, value, true)))
        {
            let Some(identity) = variables.get(name) else { continue };
            let mut binding = identity.clone();
            binding["value"] = (*value).into();
            binding["schedule_loop"] = schedule_loop.into();
            bindings.push(binding);
        }
        bindings.sort_by(|left, right| {
            (left["name"].as_str(), left["kind"].as_str(), left["slot"].as_i64()).cmp(&(
                right["name"].as_str(),
                right["kind"].as_str(),
                right["slot"].as_i64(),
            ))
        });
        items.push(serde_json::json!({
            "order": order,
            "callable_index": callable_index,
            "ast": ast,
            "buffers": buffers,
            "output_slots": ast_output_slots(&descriptor.ast),
            "dependencies": dependencies,
            "bindings": bindings,
        }));
        item_buffers.push(buffer_ids);
        latest_item.insert(invocation.kernel_id, order);
    }

    let output_slots: Vec<_> = pre
        .output_buffer_uops
        .iter()
        .map(|output| {
            let output_id = output.buf_uop().id;
            item_buffers
                .iter()
                .enumerate()
                .rev()
                .find_map(|(item, buffers)| {
                    buffers.iter().position(|buffer_id| *buffer_id == output_id).map(|buffer| {
                        serde_json::json!({
                            "item": item,
                            "buffer": buffer,
                        })
                    })
                })
                .unwrap_or_else(|| panic!("output buffer {output_id} is absent from scheduled arguments"))
        })
        .collect();
    let document = serde_json::json!({
        "schema_version": svod_ir::uop::canonical::CANONICAL_SCHEMA_VERSION,
        "stage": "scheduled",
        "items": items,
        "output_slots": output_slots,
    });
    let json = serde_json::to_string_pretty(&document).expect("canonical schedule JSON");
    std::fs::write(path, format!("{json}\n"))
        .unwrap_or_else(|error| panic!("writing canonical schedule to {path} failed: {error}"));
    document
}

fn main() {
    let requested = std::env::args().nth(1).expect("usage: canonical_stages <stage>");
    let body_index =
        std::env::args().nth(2).map(|value| value.parse().expect("kernel index must be an integer")).unwrap_or(0);
    let capture_path = std::env::var("SVOD_CAPTURE_CANONICAL_PATH")
        .expect("canonical stage runner requires SVOD_CAPTURE_CANONICAL_PATH");

    let multi_output = requested == "multi_output_callified";
    let tensor = tensor_graph(multi_output);
    let var_vals = current_var_vals(&tensor);
    svod_ir::dump_canonical_stage("tensor", &tensor);
    if finish_if_requested(&requested, "tensor", &capture_path) {
        return;
    }

    let rangeified = svod_schedule::rangeify_with_map(tensor).expect("production rangeify").sink;
    if finish_if_requested(&requested, "rangeified", &capture_path) {
        return;
    }
    let (kernel_graph, _) = svod_schedule::try_get_kernel_graph(rangeified).expect("production kernel splitting");
    if multi_output {
        let calls = kernel_graph
            .toposort_call_aware(false)
            .into_iter()
            .filter(|node| matches!(node.op(), Op::Call(..)))
            .count();
        assert!(calls >= 2, "multi-output production fixture must callify both outputs");
        assert!(finish_if_requested(&requested, "multi_output_callified", &capture_path));
        return;
    }
    if finish_if_requested(&requested, "kernel_ast", &capture_path) {
        return;
    }
    let pre_schedule = svod_tensor::schedule::create_pre_schedule(kernel_graph.clone()).expect("production scheduler");
    if requested == "scheduled" {
        assert_eq!(pre_schedule.invocations.len(), 2, "schedule evidence must preserve both chained kernels");
        let second = pre_schedule
            .items
            .iter()
            .find(|item| item.kernel.id == pre_schedule.invocations[1].kernel_id)
            .expect("second schedule descriptor");
        assert_eq!(second.dependencies, vec![pre_schedule.invocations[0].kernel_id]);
        let schedule = capture_schedule(&capture_path, &pre_schedule, &var_vals);
        assert_eq!(
            schedule["items"][0]["bindings"],
            serde_json::json!([{
                "kind": "param", "slot": -1, "name": "schedule_n",
                "dtype": {"kind": "scalar", "name": "int32"}, "value": 4, "schedule_loop": false,
            }])
        );
        assert_eq!(schedule["items"][1]["bindings"], serde_json::json!([]));
        assert_eq!(schedule["items"][1]["dependencies"], serde_json::json!([0]));
        assert_eq!(schedule["output_slots"], serde_json::json!([{"item": 1, "buffer": 0}]));
        assert!(finish_if_requested(&requested, "scheduled", &capture_path));
        return;
    }

    let mut ast = kernel_body(&kernel_graph, body_index);
    if let Op::Sink(ops::Sink { sources, info }) = ast.op() {
        ast = UOp::sink_with_info(sources.iter().cloned().collect(), info.clone().unwrap_or_default());
    }
    assert!(!pre_schedule.items.is_empty());

    let renderer = Renderer::tinygrad_base_cpu();
    let config = OptimizerConfig { strategy: OptStrategy::Heuristic, ..Default::default() };
    let optimized =
        svod_schedule::optimizer::optimize_kernel_with_config(ast, &renderer, &config).expect("production optimizer");
    assert!(
        optimized
            .toposort_call_aware(false)
            .iter()
            .any(|node| matches!(node.op(), Op::Param(ops::Param { arg, .. }) if arg.slot == usize::MAX)),
        "canonical production fixture must reach PROGRAM with an unnumbered PARAM"
    );
    if matches!(requested.as_str(), "optimized" | "postrange" | "expanded" | "coalesced" | "gated") {
        assert!(finish_if_requested(&requested, &requested, &capture_path));
        return;
    }
    let program = svod_codegen::program_pipeline::program_from_sink(optimized, DeviceSpec::Cpu)
        .expect("production PROGRAM boundary");
    assert!(
        program
            .toposort_call_aware(false)
            .iter()
            .all(|node| !matches!(node.op(), Op::Param(ops::Param { arg, .. }) if arg.slot == usize::MAX)),
        "PROGRAM boundary must number every outer executable PARAM"
    );
    if finish_if_requested(&requested, "program", &capture_path) {
        return;
    }
    let _linearized = svod_codegen::program_pipeline::do_linearize(&program).expect("production linearization");
    assert!(finish_if_requested(&requested, "linearized", &capture_path));
}
