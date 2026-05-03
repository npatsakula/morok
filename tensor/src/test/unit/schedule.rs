use std::collections::HashMap;

use morok_device::Buffer;
use morok_ir::{AxisId, AxisType, CallInfo, DType, DeviceSpec, Op, UOp};
use smallvec::SmallVec;

use crate::schedule::{
    BoundRangeRef, InputBuffers, LinearSchedOp, PreSchedule, PreScheduleItem, ScheduleItem, create_schedule,
    instantiate_schedule,
};

fn cpu_buffer(numel: usize) -> Buffer {
    let alloc = morok_device::registry::cpu().expect("cpu allocator");
    Buffer::new(alloc, DType::Float32, vec![numel], Default::default())
}

fn assert_ir_construction_error_contains(err: crate::Error, needle: &str) {
    match err {
        crate::Error::IrConstruction { details } => {
            assert!(details.contains(needle), "expected IrConstruction details to contain '{needle}', got '{details}'");
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn test_schedule_item_creation() {
    let body = UOp::sink(vec![UOp::native_const(1.0f32)]);
    let call = body.call(SmallVec::new(), CallInfo::default());

    let item = ScheduleItem {
        kernel: call,
        ast: body,
        buffers: vec![],
        buffer_uop_ids: vec![],
        fixedvars: HashMap::new(),
        dependencies: vec![],
        instance_dependencies: vec![],
        alias_registered_ids: vec![],
    };

    assert!(matches!(item.kernel.op(), Op::Call { .. }));
}

#[test]
fn test_create_schedule_after_uses_canonical_buffer_id() {
    let buffer_uop = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let after = buffer_uop.after(SmallVec::new());
    let body = UOp::sink(vec![UOp::native_const(0.0f32)]);
    let call = body.call(SmallVec::from_vec(vec![after.clone()]), CallInfo::default());
    let transformed = UOp::sink(vec![call]);

    let mut input_buffers = InputBuffers::new();
    let input = cpu_buffer(4);
    input_buffers.insert(buffer_uop.id, input.clone());

    let result = create_schedule(transformed, &input_buffers, &HashMap::new()).expect("create schedule");
    let item = &result.items[0];

    assert_eq!(item.buffer_uop_ids, vec![buffer_uop.id]);
    assert_eq!(item.buffers.len(), 1);
    assert_eq!(item.buffers[0].id(), input.id());
    assert!(item.alias_registered_ids.contains(&after.id));
}

#[test]
fn test_create_schedule_mselect_uses_canonical_buffer_id() {
    let buffer_uop = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let mstack = UOp::mstack(SmallVec::from_vec(vec![buffer_uop.clone()]));
    let mselect = mstack.mselect(0);
    let body = UOp::sink(vec![UOp::native_const(0.0f32)]);
    let call = body.call(SmallVec::from_vec(vec![mselect.clone()]), CallInfo::default());
    let transformed = UOp::sink(vec![call]);

    let mut input_buffers = InputBuffers::new();
    let input = cpu_buffer(4);
    input_buffers.insert(buffer_uop.id, input.clone());

    let result = create_schedule(transformed, &input_buffers, &HashMap::new()).expect("create schedule");
    let item = &result.items[0];

    assert_eq!(item.buffer_uop_ids, vec![buffer_uop.id]);
    assert_eq!(item.buffers.len(), 1);
    assert_eq!(item.buffers[0].id(), input.id());
    assert!(item.alias_registered_ids.contains(&mselect.id));
}

#[test]
fn test_create_schedule_preserves_kernel_dependencies() {
    let buffer_uop = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);

    let body1 = UOp::sink(vec![UOp::native_const(1.0f32)]);
    let kernel1 = body1.call(SmallVec::from_vec(vec![buffer_uop.clone()]), CallInfo::default());

    let after = buffer_uop.after(SmallVec::from_vec(vec![kernel1.clone()]));

    let body2 = UOp::sink(vec![UOp::native_const(2.0f32)]);
    let kernel2 = body2.call(SmallVec::from_vec(vec![after]), CallInfo::default());

    let transformed = UOp::sink(vec![kernel1.clone(), kernel2.clone()]);

    let mut input_buffers = InputBuffers::new();
    input_buffers.insert(buffer_uop.id, cpu_buffer(4));

    let result = create_schedule(transformed, &input_buffers, &HashMap::new()).expect("create schedule");

    assert_eq!(result.items.len(), 2);
    let k1_item = result.items.iter().find(|it| it.kernel.id == kernel1.id).expect("k1 item");
    let k2_item = result.items.iter().find(|it| it.kernel.id == kernel2.id).expect("k2 item");

    assert!(k1_item.dependencies.is_empty());
    assert_eq!(k2_item.dependencies, vec![kernel1.id]);
}

#[test]
fn test_create_schedule_supports_call_wrapper() {
    let buffer_uop = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let body = UOp::sink(vec![UOp::native_const(1.0f32)]);
    let call = body.call(SmallVec::from_vec(vec![buffer_uop.clone()]), CallInfo::default());
    let transformed = UOp::sink(vec![call.clone()]);

    let mut input_buffers = InputBuffers::new();
    input_buffers.insert(buffer_uop.id, cpu_buffer(4));

    let result = create_schedule(transformed, &input_buffers, &HashMap::new()).expect("create schedule");

    assert_eq!(result.items.len(), 1);
    assert_eq!(result.items[0].kernel.id, call.id);
    assert!(matches!(result.items[0].kernel.op(), Op::Call { .. }));
}

#[test]
fn test_create_schedule_preserves_call_dependencies_after_call() {
    let buffer_uop = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);

    let producer_body = UOp::sink(vec![UOp::native_const(1.0f32)]);
    let producer = producer_body.call(SmallVec::from_vec(vec![buffer_uop.clone()]), CallInfo::default());

    let after = buffer_uop.after(SmallVec::from_vec(vec![producer.clone()]));

    let consumer_body = UOp::sink(vec![UOp::native_const(2.0f32)]);
    let consumer = consumer_body.call(SmallVec::from_vec(vec![after]), CallInfo::default());

    let transformed = UOp::sink(vec![producer.clone(), consumer.clone()]);

    let mut input_buffers = InputBuffers::new();
    input_buffers.insert(buffer_uop.id, cpu_buffer(4));

    let result = create_schedule(transformed, &input_buffers, &HashMap::new()).expect("create schedule");
    assert_eq!(result.items.len(), 2);

    let producer_item = result.items.iter().find(|it| it.kernel.id == producer.id).expect("producer item");
    let consumer_item = result.items.iter().find(|it| it.kernel.id == consumer.id).expect("consumer item");

    assert!(producer_item.dependencies.is_empty());
    assert_eq!(consumer_item.dependencies, vec![producer.id]);
}

#[test]
fn test_create_schedule_preserves_call_dependencies_end_call() {
    let buffer_uop = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);

    let producer_body = UOp::sink(vec![UOp::native_const(1.0f32)]);
    let producer = producer_body.call(SmallVec::from_vec(vec![buffer_uop.clone()]), CallInfo::default());
    let end_call = producer.end(SmallVec::from_vec(vec![UOp::range_const(1, 0)]));
    let after = buffer_uop.after(SmallVec::from_vec(vec![end_call]));

    let consumer_body = UOp::sink(vec![UOp::native_const(2.0f32)]);
    let consumer = consumer_body.call(SmallVec::from_vec(vec![after]), CallInfo::default());

    let transformed = UOp::sink(vec![producer.clone(), consumer.clone()]);

    let mut input_buffers = InputBuffers::new();
    input_buffers.insert(buffer_uop.id, cpu_buffer(4));

    let result = create_schedule(transformed, &input_buffers, &HashMap::new()).expect("create schedule");
    let consumer_item = result.items.iter().find(|it| it.kernel.id == consumer.id).expect("consumer item");

    assert_eq!(consumer_item.dependencies, vec![producer.id]);
}

#[test]
fn test_create_schedule_preserves_call_arg_order() {
    let a_uop = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let b_uop = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);

    let body = UOp::sink(vec![UOp::native_const(3.0f32)]);
    let call = body.call(SmallVec::from_vec(vec![b_uop.clone(), a_uop.clone()]), CallInfo::default());
    let transformed = UOp::sink(vec![call]);

    let input_a = cpu_buffer(4);
    let input_b = cpu_buffer(4);

    let mut input_buffers = InputBuffers::new();
    input_buffers.insert(a_uop.id, input_a.clone());
    input_buffers.insert(b_uop.id, input_b.clone());

    let result = create_schedule(transformed, &input_buffers, &HashMap::new()).expect("create schedule");
    let item = &result.items[0];

    assert_eq!(item.buffer_uop_ids, vec![b_uop.id, a_uop.id]);
    assert_eq!(item.buffers.len(), 2);
    assert_eq!(item.buffers[0].id(), input_b.id());
    assert_eq!(item.buffers[1].id(), input_a.id());
}

#[test]
fn test_create_schedule_unrolls_call_bound_ranges() {
    let buffer_uop = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let outer_range = UOp::range_axis(UOp::index_const(3), morok_ir::AxisId::Renumbered(0), morok_ir::AxisType::Outer);
    let bind = UOp::define_var("outer_i".to_string(), 0, 2).bind(outer_range.clone());
    let body = UOp::sink(vec![UOp::native_const(1.0f32)]);
    let call = body.call(SmallVec::from_vec(vec![buffer_uop.clone(), bind]), CallInfo::default());
    let end_call = call.end(SmallVec::from_vec(vec![outer_range]));
    let transformed = UOp::sink(vec![call, end_call]);

    let mut input_buffers = InputBuffers::new();
    input_buffers.insert(buffer_uop.id, cpu_buffer(4));

    let schedule_result = create_schedule(transformed, &input_buffers, &HashMap::new()).expect("create schedule");
    assert_eq!(schedule_result.items.len(), 3);
    assert!(schedule_result.items.iter().all(|it| matches!(it.kernel.op(), Op::Call { .. })));

    let mut fixed: Vec<i64> =
        schedule_result.items.iter().map(|it| *it.fixedvars.get("outer_i").expect("outer_i fixedvar")).collect();
    fixed.sort_unstable();
    assert_eq!(fixed, vec![0, 1, 2]);
}

#[test]
fn test_create_schedule_collects_all_after_callable_dependencies() {
    let in_a = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let in_b = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let passthrough = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);

    let p1 =
        UOp::sink(vec![UOp::native_const(1.0f32)]).call(SmallVec::from_vec(vec![in_a.clone()]), CallInfo::default());
    let p2 =
        UOp::sink(vec![UOp::native_const(2.0f32)]).call(SmallVec::from_vec(vec![in_b.clone()]), CallInfo::default());

    let after = passthrough.after(SmallVec::from_vec(vec![p1.clone(), p2.clone()]));
    let consumer =
        UOp::sink(vec![UOp::native_const(3.0f32)]).call(SmallVec::from_vec(vec![after]), CallInfo::default());

    let transformed = UOp::sink(vec![p1.clone(), p2.clone(), consumer.clone()]);

    let mut input_buffers = InputBuffers::new();
    input_buffers.insert(in_a.id, cpu_buffer(4));
    input_buffers.insert(in_b.id, cpu_buffer(4));
    input_buffers.insert(passthrough.id, cpu_buffer(4));

    let result = create_schedule(transformed, &input_buffers, &HashMap::new()).expect("create schedule");
    let consumer_item = result.items.iter().find(|it| it.kernel.id == consumer.id).expect("consumer item");

    let mut expected = vec![p1.id, p2.id];
    expected.sort_unstable();
    assert_eq!(consumer_item.dependencies, expected);
}

#[test]
fn test_create_schedule_collects_end_call_and_call_dependencies() {
    let in_a = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let in_b = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let passthrough = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);

    let p1 =
        UOp::sink(vec![UOp::native_const(1.0f32)]).call(SmallVec::from_vec(vec![in_a.clone()]), CallInfo::default());
    let p1_end = p1.end(SmallVec::from_vec(vec![UOp::range_const(1, 0)]));
    let p2 =
        UOp::sink(vec![UOp::native_const(2.0f32)]).call(SmallVec::from_vec(vec![in_b.clone()]), CallInfo::default());

    let after = passthrough.after(SmallVec::from_vec(vec![p1_end, p2.clone()]));
    let consumer =
        UOp::sink(vec![UOp::native_const(3.0f32)]).call(SmallVec::from_vec(vec![after]), CallInfo::default());

    let transformed = UOp::sink(vec![p1.clone(), p2.clone(), consumer.clone()]);

    let mut input_buffers = InputBuffers::new();
    input_buffers.insert(in_a.id, cpu_buffer(4));
    input_buffers.insert(in_b.id, cpu_buffer(4));
    input_buffers.insert(passthrough.id, cpu_buffer(4));

    let result = create_schedule(transformed, &input_buffers, &HashMap::new()).expect("create schedule");
    let consumer_item = result.items.iter().find(|it| it.kernel.id == consumer.id).expect("consumer item");

    let mut expected = vec![p1.id, p2.id];
    expected.sort_unstable();
    assert_eq!(consumer_item.dependencies, expected);
}

#[test]
fn test_create_schedule_accepts_bind_callable_source() {
    let buffer_uop = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let bind = UOp::define_var("N".to_string(), 0, 4).bind(UOp::range_const(4, 0));
    let body = UOp::sink(vec![UOp::native_const(1.0f32)]);
    let call = body.call(SmallVec::from_vec(vec![buffer_uop.clone(), bind]), CallInfo::default());
    let transformed = UOp::sink(vec![call]);

    let mut input_buffers = InputBuffers::new();
    let input = cpu_buffer(4);
    input_buffers.insert(buffer_uop.id, input.clone());

    let result = create_schedule(transformed, &input_buffers, &HashMap::new()).expect("create schedule");
    let item = &result.items[0];

    assert_eq!(item.buffer_uop_ids, vec![buffer_uop.id]);
    assert_eq!(item.buffers.len(), 1);
    assert_eq!(item.buffers[0].id(), input.id());
}

#[test]
fn test_create_schedule_collects_nested_after_dependencies() {
    let input = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let passthrough = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);

    let producer =
        UOp::sink(vec![UOp::native_const(1.0f32)]).call(SmallVec::from_vec(vec![input.clone()]), CallInfo::default());
    let nested_after = passthrough.after(SmallVec::from_vec(vec![producer.clone()]));
    let after = passthrough.after(SmallVec::from_vec(vec![nested_after]));
    let consumer =
        UOp::sink(vec![UOp::native_const(2.0f32)]).call(SmallVec::from_vec(vec![after]), CallInfo::default());

    let transformed = UOp::sink(vec![producer.clone(), consumer.clone()]);

    let mut input_buffers = InputBuffers::new();
    input_buffers.insert(input.id, cpu_buffer(4));
    input_buffers.insert(passthrough.id, cpu_buffer(4));

    let result = create_schedule(transformed, &input_buffers, &HashMap::new()).expect("create schedule");
    let consumer_item = result.items.iter().find(|it| it.kernel.id == consumer.id).expect("consumer item");

    assert_eq!(consumer_item.dependencies, vec![producer.id]);
}

#[test]
fn test_create_schedule_rejects_non_callable_after_dependency() {
    let passthrough = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let invalid_dep = UOp::native_const(1.0f32);
    let after = passthrough.after(SmallVec::from_vec(vec![invalid_dep]));
    let consumer =
        UOp::sink(vec![UOp::native_const(2.0f32)]).call(SmallVec::from_vec(vec![after]), CallInfo::default());
    let transformed = UOp::sink(vec![consumer]);

    let mut input_buffers = InputBuffers::new();
    input_buffers.insert(passthrough.id, cpu_buffer(4));

    let err = match create_schedule(transformed, &input_buffers, &HashMap::new()) {
        Ok(_) => panic!("invalid AFTER dep should fail"),
        Err(err) => err,
    };
    match err {
        crate::Error::IrConstruction { details } => {
            assert!(details.contains("AFTER dependency must be CALL/END(CALL)/STORE/AFTER"));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn test_create_schedule_accepts_store_in_after_dependencies() {
    let input = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let passthrough = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);

    let producer =
        UOp::sink(vec![UOp::native_const(1.0f32)]).call(SmallVec::from_vec(vec![input.clone()]), CallInfo::default());

    let store_idx = UOp::index()
        .buffer(passthrough.clone())
        .indices(vec![UOp::index_const(0)])
        .ptr(true)
        .call()
        .expect("store index");
    let side_store = store_idx.store(UOp::native_const(0.0f32));

    let after = passthrough.after(SmallVec::from_vec(vec![producer.clone(), side_store]));
    let consumer =
        UOp::sink(vec![UOp::native_const(2.0f32)]).call(SmallVec::from_vec(vec![after]), CallInfo::default());

    let transformed = UOp::sink(vec![producer.clone(), consumer.clone()]);

    let mut input_buffers = InputBuffers::new();
    input_buffers.insert(input.id, cpu_buffer(4));
    input_buffers.insert(passthrough.id, cpu_buffer(4));

    let result = create_schedule(transformed, &input_buffers, &HashMap::new()).expect("create schedule");
    let consumer_item = result.items.iter().find(|it| it.kernel.id == consumer.id).expect("consumer item");

    assert_eq!(consumer_item.dependencies, vec![producer.id]);
}

#[test]
fn test_create_schedule_avoids_false_dep_from_shared_passthrough_identity() {
    let in_a = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let in_b = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let passthrough = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);

    let p1 =
        UOp::sink(vec![UOp::native_const(1.0f32)]).call(SmallVec::from_vec(vec![in_a.clone()]), CallInfo::default());
    let p2 =
        UOp::sink(vec![UOp::native_const(2.0f32)]).call(SmallVec::from_vec(vec![in_b.clone()]), CallInfo::default());

    let c1 = UOp::sink(vec![UOp::native_const(3.0f32)])
        .call(SmallVec::from_vec(vec![passthrough.after(SmallVec::from_vec(vec![p1.clone()]))]), CallInfo::default());
    let c2 = UOp::sink(vec![UOp::native_const(4.0f32)])
        .call(SmallVec::from_vec(vec![passthrough.after(SmallVec::from_vec(vec![p2.clone()]))]), CallInfo::default());

    let transformed = UOp::sink(vec![p1.clone(), p2.clone(), c1.clone(), c2.clone()]);

    let mut input_buffers = InputBuffers::new();
    input_buffers.insert(in_a.id, cpu_buffer(4));
    input_buffers.insert(in_b.id, cpu_buffer(4));
    input_buffers.insert(passthrough.id, cpu_buffer(4));

    let result = create_schedule(transformed, &input_buffers, &HashMap::new()).expect("create schedule");
    let c1_item = result.items.iter().find(|it| it.kernel.id == c1.id).expect("c1 item");
    let c2_item = result.items.iter().find(|it| it.kernel.id == c2.id).expect("c2 item");

    assert_eq!(c1_item.dependencies, vec![p1.id]);
    assert_eq!(c2_item.dependencies, vec![p2.id]);
}

#[test]
fn test_create_schedule_preserves_ordering_only_dep_for_void_custom_call() {
    let input = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let producer_passthrough = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let ordering_passthrough = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);

    let producer =
        UOp::sink(vec![UOp::native_const(1.0f32)]).call(SmallVec::from_vec(vec![input.clone()]), CallInfo::default());
    let producer_after = producer_passthrough.after(SmallVec::from_vec(vec![producer.clone()]));

    let custom = UOp::custom_function(morok_ir::CustomFunctionKind::EncDec, SmallVec::new())
        .call(SmallVec::new(), CallInfo::default());

    let ordering_after = ordering_passthrough.after(SmallVec::from_vec(vec![custom.clone(), producer_after]));
    let transformed = UOp::sink(vec![producer.clone(), custom.clone(), ordering_after]);

    let mut input_buffers = InputBuffers::new();
    input_buffers.insert(input.id, cpu_buffer(4));
    input_buffers.insert(producer_passthrough.id, cpu_buffer(4));
    input_buffers.insert(ordering_passthrough.id, cpu_buffer(4));

    let result = create_schedule(transformed, &input_buffers, &HashMap::new()).expect("create schedule");
    let custom_item = result.items.iter().find(|it| it.kernel.id == custom.id).expect("custom item");

    assert_eq!(custom_item.dependencies, vec![producer.id]);
}

#[test]
fn test_create_schedule_nested_after_mstack_mselect_dependencies_consistent() {
    let in_a = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let in_b = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let out_a = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let out_b = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);

    let p1 =
        UOp::sink(vec![UOp::native_const(1.0f32)]).call(SmallVec::from_vec(vec![in_a.clone()]), CallInfo::default());
    let p2 =
        UOp::sink(vec![UOp::native_const(2.0f32)]).call(SmallVec::from_vec(vec![in_b.clone()]), CallInfo::default());

    let a1 = out_a.after(SmallVec::from_vec(vec![p1.clone()]));
    let a2 = out_b.after(SmallVec::from_vec(vec![p2.clone()]));
    let stacked = UOp::mstack(SmallVec::from_vec(vec![a1, a2.clone()]));
    let selected = stacked.mselect(0);

    let consumer =
        UOp::sink(vec![UOp::native_const(3.0f32)]).call(SmallVec::from_vec(vec![selected]), CallInfo::default());
    let transformed = UOp::sink(vec![p1.clone(), p2.clone(), consumer.clone()]);

    let mut input_buffers = InputBuffers::new();
    input_buffers.insert(in_a.id, cpu_buffer(4));
    input_buffers.insert(in_b.id, cpu_buffer(4));
    input_buffers.insert(out_a.id, cpu_buffer(4));
    input_buffers.insert(out_b.id, cpu_buffer(4));

    let result = create_schedule(transformed, &input_buffers, &HashMap::new()).expect("create schedule");
    let consumer_item = result.items.iter().find(|it| it.kernel.id == consumer.id).expect("consumer item");

    let mut expected = vec![p1.id, p2.id];
    expected.sort_unstable();
    assert_eq!(consumer_item.dependencies, expected);
}

#[test]
fn test_create_schedule_rejects_outer_range_missing_end() {
    let buffer_uop = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let outer_range = UOp::range_outer_const(3, 0);
    let bind = UOp::define_var("outer_i".to_string(), 0, 2).bind(outer_range);
    let body = UOp::sink(vec![UOp::native_const(1.0f32)]);
    let call = body.call(SmallVec::from_vec(vec![buffer_uop.clone(), bind]), CallInfo::default());
    let transformed = UOp::sink(vec![call]);

    let mut input_buffers = InputBuffers::new();
    input_buffers.insert(buffer_uop.id, cpu_buffer(4));

    let err = match create_schedule(transformed, &input_buffers, &HashMap::new()) {
        Ok(_) => panic!("outer RANGE without END should fail"),
        Err(err) => err,
    };
    assert_ir_construction_error_contains(err, "missing END in strict scheduler");
}

#[test]
fn test_create_schedule_rejects_non_concrete_outer_range_bounds() {
    let buffer_uop = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let malformed_outer_range = UOp::new(
        Op::Range {
            end: UOp::native_const(3.0f32),
            axis_id: AxisId::Renumbered(0),
            axis_type: AxisType::Outer,
            deps: SmallVec::new(),
        },
        DType::Float32,
    );
    let bind = UOp::define_var("outer_i".to_string(), 0, 2).bind(malformed_outer_range.clone());
    let body = UOp::sink(vec![UOp::native_const(1.0f32)]);
    let call = body.call(SmallVec::from_vec(vec![buffer_uop.clone(), bind]), CallInfo::default());
    let end_call = call.end(SmallVec::from_vec(vec![malformed_outer_range]));
    let transformed = UOp::sink(vec![call, end_call]);

    let mut input_buffers = InputBuffers::new();
    input_buffers.insert(buffer_uop.id, cpu_buffer(4));

    let err = match create_schedule(transformed, &input_buffers, &HashMap::new()) {
        Ok(_) => panic!("non-concrete OUTER RANGE bounds should fail"),
        Err(err) => err,
    };
    assert_ir_construction_error_contains(err, "OUTER RANGE vmax must be concrete integer");
}

#[test]
fn test_instantiate_schedule_rejects_bound_range_missing_from_linear_schedule() {
    let buffer_uop = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let body = UOp::sink(vec![UOp::native_const(1.0f32)]);
    let call = body.call(SmallVec::from_vec(vec![buffer_uop.clone()]), CallInfo::default());

    let declared_range = UOp::range_outer_const(1, 0);
    let missing_range = UOp::range_outer_const(1, 1);
    let pre_schedule = PreSchedule {
        items: vec![PreScheduleItem {
            kernel: call.clone(),
            ast: body,
            sources: vec![buffer_uop.clone()],
            dependencies: vec![],
            bound_ranges: vec![BoundRangeRef { var_name: "outer_i".to_string(), range_uop: missing_range }],
        }],
        linear_ops: vec![
            LinearSchedOp::Range { range: declared_range.clone() },
            LinearSchedOp::Call { kernel_id: call.id },
            LinearSchedOp::End { range: declared_range, kernel_id: call.id },
        ],
        output_buffer_uops: vec![buffer_uop.clone()],
    };

    let mut input_buffers = InputBuffers::new();
    input_buffers.insert(buffer_uop.id, cpu_buffer(4));

    let err = match instantiate_schedule(&pre_schedule, &input_buffers, &HashMap::new()) {
        Ok(_) => panic!("bound range missing from linear schedule should fail"),
        Err(err) => err,
    };
    assert_ir_construction_error_contains(err, "missing from linear schedule");
}

#[test]
fn test_instantiate_schedule_rejects_inactive_outer_range_binding() {
    let buffer_uop = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let body = UOp::sink(vec![UOp::native_const(1.0f32)]);
    let call = body.call(SmallVec::from_vec(vec![buffer_uop.clone()]), CallInfo::default());

    let outer_range = UOp::range_outer_const(2, 0);
    let pre_schedule = PreSchedule {
        items: vec![PreScheduleItem {
            kernel: call.clone(),
            ast: body,
            sources: vec![buffer_uop.clone()],
            dependencies: vec![],
            bound_ranges: vec![BoundRangeRef { var_name: "outer_i".to_string(), range_uop: outer_range.clone() }],
        }],
        linear_ops: vec![
            LinearSchedOp::Call { kernel_id: call.id },
            LinearSchedOp::Range { range: outer_range.clone() },
            LinearSchedOp::End { range: outer_range, kernel_id: call.id },
        ],
        output_buffer_uops: vec![buffer_uop.clone()],
    };

    let mut input_buffers = InputBuffers::new();
    input_buffers.insert(buffer_uop.id, cpu_buffer(4));

    let err = match instantiate_schedule(&pre_schedule, &input_buffers, &HashMap::new()) {
        Ok(_) => panic!("inactive OUTER RANGE binding should fail"),
        Err(err) => err,
    };
    assert_ir_construction_error_contains(err, "references inactive OUTER RANGE");
}

#[test]
fn test_instantiate_schedule_rejects_unknown_kernel_id_in_linear_call() {
    let buffer_uop = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let body = UOp::sink(vec![UOp::native_const(1.0f32)]);
    let call = body.call(SmallVec::from_vec(vec![buffer_uop.clone()]), CallInfo::default());

    let pre_schedule = PreSchedule {
        items: vec![PreScheduleItem {
            kernel: call,
            ast: body,
            sources: vec![buffer_uop.clone()],
            dependencies: vec![],
            bound_ranges: vec![],
        }],
        linear_ops: vec![LinearSchedOp::Call { kernel_id: u64::MAX - 1 }],
        output_buffer_uops: vec![buffer_uop.clone()],
    };

    let mut input_buffers = InputBuffers::new();
    input_buffers.insert(buffer_uop.id, cpu_buffer(4));

    let err = match instantiate_schedule(&pre_schedule, &input_buffers, &HashMap::new()) {
        Ok(_) => panic!("linear CALL with unknown kernel id should fail"),
        Err(err) => err,
    };
    assert_ir_construction_error_contains(err, "linear CALL references unknown kernel id");
}

#[test]
fn test_instantiate_schedule_rejects_unknown_kernel_id_in_linear_end() {
    let buffer_uop = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let body = UOp::sink(vec![UOp::native_const(1.0f32)]);
    let call = body.call(SmallVec::from_vec(vec![buffer_uop.clone()]), CallInfo::default());

    let outer_range = UOp::range_outer_const(1, 0);
    let pre_schedule = PreSchedule {
        items: vec![PreScheduleItem {
            kernel: call,
            ast: body,
            sources: vec![buffer_uop.clone()],
            dependencies: vec![],
            bound_ranges: vec![],
        }],
        linear_ops: vec![
            LinearSchedOp::Range { range: outer_range.clone() },
            LinearSchedOp::End { range: outer_range, kernel_id: u64::MAX - 2 },
        ],
        output_buffer_uops: vec![buffer_uop.clone()],
    };

    let mut input_buffers = InputBuffers::new();
    input_buffers.insert(buffer_uop.id, cpu_buffer(4));

    let err = match instantiate_schedule(&pre_schedule, &input_buffers, &HashMap::new()) {
        Ok(_) => panic!("linear END with unknown kernel id should fail"),
        Err(err) => err,
    };
    assert_ir_construction_error_contains(err, "linear END references unknown CALL id");
}
