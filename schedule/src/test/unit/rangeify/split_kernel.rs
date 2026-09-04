//! `split_store`: which STOREs become their own kernel, and what the resulting
//! CALL carries.

use std::sync::Arc;

use smallvec::smallvec;
use svod_dtype::{AddrSpace, DType, DeviceSpec};
use svod_ir::{AxisId, AxisType, Op, UOp};
use test_case::test_case;

use super::helpers::extract_kernel;
use crate::rangeify::kernel::{split_store, try_get_kernel_graph};
use svod_ir::ops;

fn call_split_store(x: &Arc<UOp>) -> Option<Arc<UOp>> {
    split_store(&mut Vec::new(), x)
}

fn buffer(size: usize) -> Arc<UOp> {
    UOp::new_buffer(DeviceSpec::Cpu, size, DType::Float32)
}

fn index_at_zero(buffer: Arc<UOp>) -> Arc<UOp> {
    UOp::index().buffer(buffer).indices(vec![UOp::index_const(0)]).call().expect("index")
}

fn store_of(value: Arc<UOp>) -> Arc<UOp> {
    index_at_zero(buffer(100)).store(value)
}

/// Ranges closed by the END the CALL body wraps.
fn closed_range_count(uop: &Arc<UOp>) -> usize {
    match uop.op() {
        Op::End(ops::End { ranges, .. }) => ranges.as_slice().len(),
        Op::Sink(ops::Sink { sources, .. }) => sources.iter().map(closed_range_count).sum(),
        _ => 0,
    }
}

// ===== what a plain STORE lowers to =====

/// A STORE with no open ranges splits straight into `CALL(SINK(STORE))`: the
/// destination BUFFER becomes a body-local PARAM and the CALL binds it.
#[test_case(DType::Float32, UOp::native_const(1.0f32) ; "float const")]
#[test_case(DType::Int32, UOp::native_const(1i32) ; "int const")]
#[test_case(DType::Bool, UOp::native_const(true) ; "bool const")]
#[test_case(DType::Float32, UOp::native_const(1.0f32).try_add(&UOp::native_const(2.0f32)).expect("add") ; "arithmetic")]
fn a_closed_store_becomes_a_call_over_a_param(dtype: DType, value: Arc<UOp>) {
    let store = index_at_zero(UOp::new_buffer(DeviceSpec::Cpu, 100, dtype)).store(Arc::clone(&value));
    assert!(store.in_scope_ranges().is_empty(), "the fixture must have no open ranges");

    let kernel = call_split_store(&store).expect("a closed STORE splits");
    let Op::Call(ops::Call { body, args, .. }) = kernel.op() else { panic!("expected CALL, got {}", kernel.tree()) };
    assert!(!args.is_empty(), "the CALL must bind the destination buffer");

    let Op::Sink(ops::Sink { sources, .. }) = body.op() else { panic!("expected SINK body, got {}", body.tree()) };
    let [stored] = sources.as_slice() else { panic!("expected one STORE in the body") };
    let Op::Store(ops::Store { index, value: stored_value, .. }) = stored.op() else { panic!("expected STORE") };
    let Op::Index(ops::Index { buffer, .. }) = index.op() else { panic!("expected INDEX in the STORE") };
    assert!(
        matches!(buffer.op(), Op::Param(ops::Param { arg, .. }) if arg.device == Some(DeviceSpec::Cpu)),
        "the body reaches storage through a codegen PARAM, got {}",
        buffer.tree()
    );
    assert!(Arc::ptr_eq(stored_value, &value));
}

#[test]
fn two_stores_split_into_two_distinct_kernels() {
    let first = call_split_store(&store_of(UOp::native_const(1.0f32))).expect("first splits");
    let second = call_split_store(&store_of(UOp::native_const(2.0f32))).expect("second splits");
    assert!(!Arc::ptr_eq(&first, &second));
}

#[test]
fn only_a_store_splits() {
    assert!(call_split_store(&UOp::native_const(1.0f32)).is_none());
    assert!(call_split_store(&UOp::noop().end(smallvec![UOp::range_const(10, 0)])).is_none(), "END(NOOP) is a marker");
}

// ===== END(STORE): closed ranges make a kernel whatever their axis type =====

fn range(end: i64, axis_id: usize, axis_type: AxisType) -> Arc<UOp> {
    UOp::range_axis(UOp::index_const(end), AxisId::Renumbered(axis_id), axis_type)
}

fn one_weak() -> Vec<Arc<UOp>> {
    vec![UOp::range_const(10, 0)]
}

fn two_weak() -> Vec<Arc<UOp>> {
    vec![UOp::range_const(4, 0), UOp::range_const(8, 1)]
}

fn one_loop() -> Vec<Arc<UOp>> {
    vec![range(10, 0, AxisType::Loop)]
}

fn weak_then_loop() -> Vec<Arc<UOp>> {
    vec![UOp::range_const(4, 0), range(8, 1, AxisType::Loop)]
}

fn loop_then_weak() -> Vec<Arc<UOp>> {
    vec![range(8, 1, AxisType::Loop), UOp::range_const(4, 0)]
}

/// END closes its ranges, so the STORE under it always splits and the CALL body
/// keeps every closed range — order and axis type do not gate the split.
#[test_case(super::one_weak, 1 ; "one weak range")]
#[test_case(super::two_weak, 2 ; "two weak ranges")]
#[test_case(super::one_loop, 1 ; "one loop range")]
#[test_case(super::weak_then_loop, 2 ; "weak before loop")]
#[test_case(super::loop_then_weak, 2 ; "loop before weak")]
fn an_end_over_a_store_keeps_every_closed_range(ranges: fn() -> Vec<Arc<UOp>>, expected: usize) {
    let end = store_of(UOp::native_const(1.0f32)).end(ranges().into());

    let kernel = call_split_store(&end).expect("END(STORE) splits");
    let Op::Call(ops::Call { body, args, .. }) = kernel.op() else { panic!("expected CALL, got {}", kernel.tree()) };
    assert!(!args.is_empty(), "the CALL must bind the destination buffer");
    assert_eq!(closed_range_count(body), expected);
}

// ===== open ranges gate the split =====

/// An open computational loop means the STORE is interior — it belongs to the
/// enclosing kernel. A DEVICE range is a launch lane, not a loop, so it does not
/// block the split; mixing one with a real loop does.
#[test_case(AxisType::Weak, false ; "open weak range")]
#[test_case(AxisType::Loop, false ; "open loop range")]
#[test_case(AxisType::Device, true ; "open device lane")]
fn an_open_range_blocks_the_split_unless_it_is_a_launch_lane(axis_type: AxisType, splits: bool) {
    let r = range(4, 0, axis_type);
    let store =
        UOp::index().buffer(buffer(64)).indices(vec![r]).call().expect("index").store(UOp::native_const(1.0f32));

    assert!(!store.in_scope_ranges().is_empty(), "the fixture must have an open range");
    assert_eq!(call_split_store(&store).is_some(), splits);
}

#[test]
fn a_device_lane_crossed_with_a_loop_still_blocks_the_split() {
    let device = range(2, 0, AxisType::Device);
    let inner = range(2, 1, AxisType::Loop);
    let flat = device.mul(&UOp::index_const(2)).add(&inner);
    let store =
        UOp::index().buffer(buffer(4)).indices(vec![flat]).call().expect("index").store(UOp::native_const(1.0f32));

    assert!(call_split_store(&store).is_none());
}

// ===== COPY kernels =====

fn bare_copy(copy: Arc<UOp>) -> Arc<UOp> {
    index_at_zero(UOp::new_buffer(DeviceSpec::Cuda { device_id: 0 }, 100, DType::Float32)).store(copy)
}

fn copy_under_an_end(copy: Arc<UOp>) -> Arc<UOp> {
    bare_copy(copy).end(smallvec![UOp::range_const(10, 0)])
}

fn double_copy(copy: Arc<UOp>) -> Arc<UOp> {
    index_at_zero(buffer(100)).store(copy.copy_to_device(DeviceSpec::Cpu))
}

/// A COPY stored to a buffer becomes the kernel body directly — no SINK wrapper —
/// so mixed-op runtime lowering can still recover it, even under an END.
#[test_case(super::bare_copy ; "copy stored directly")]
#[test_case(super::copy_under_an_end ; "copy under an end")]
#[test_case(super::double_copy ; "copy of a copy")]
fn a_stored_copy_becomes_the_kernel_body(build: fn(Arc<UOp>) -> Arc<UOp>) {
    let copy = buffer(100).copy_to_device(DeviceSpec::Cuda { device_id: 0 });

    let kernel = call_split_store(&build(copy)).expect("a stored COPY splits");
    let Op::Call(ops::Call { body, .. }) = kernel.op() else { panic!("expected CALL, got {}", kernel.tree()) };
    let body = match body.op() {
        Op::End(ops::End { computation, .. }) => computation,
        _ => body,
    };
    assert!(matches!(body.op(), Op::Copy(..)), "expected a COPY body, got {}", body.tree());
}

#[test]
fn a_cross_device_copy_survives_the_whole_kernel_graph() {
    let copy = buffer(16).copy_to_device(DeviceSpec::Cuda { device_id: 0 });
    let out = UOp::new_buffer(DeviceSpec::Cuda { device_id: 0 }, 16, DType::Float32);
    let root = UOp::sink(vec![index_at_zero(out).store(copy)]);

    let (graph, _ctx) = try_get_kernel_graph(root).expect("cross-device COPY must be allowed");
    let kernel = extract_kernel(&graph).expect("copy call");
    let Op::Call(ops::Call { body, .. }) = kernel.op() else { panic!("expected CALL") };
    assert!(matches!(body.op(), Op::Copy(..)), "expected a direct COPY body, got {}", body.tree());
}

/// A BIND argument may carry device-owned storage for its value; that storage is
/// not the kernel's, so it must not enter device validation.
#[test]
fn bind_args_do_not_participate_in_kernel_device_validation() {
    let cuda_param = UOp::param(1, 1, DType::Index, Some(DeviceSpec::Cuda { device_id: 0 }));
    let bound = UOp::define_var("i".to_string(), 0, 15).bind(cuda_param);
    let index = UOp::index().buffer(buffer(16)).indices(vec![bound]).call().expect("index");
    let root = UOp::sink(vec![index.store(UOp::native_const(1.0f32))]);

    let (graph, _ctx) = try_get_kernel_graph(root).expect("BIND args must not fail device validation");
    assert!(extract_kernel(&graph).is_some());
}

// ===== the CALL argument tuple =====

/// Storage identities are sparse by construction. Only globals and scalar
/// bindings become CALL positions — local and register allocations stay inside
/// the body — and the body's PARAM slots are renumbered dense to match.
#[test]
fn only_globals_and_scalar_bindings_become_dense_call_positions() {
    let output = UOp::buffer(41, 4, DType::Float32, AddrSpace::Global, Some(DeviceSpec::Cpu));
    let local = UOp::buffer(700, 4, DType::Float32, AddrSpace::Local, None);
    let local_peer = UOp::buffer(701, 4, DType::Float32, AddrSpace::Local, None);
    let reg = UOp::buffer(800, 1, DType::Float32, AddrSpace::Reg, None);
    let input = UOp::buffer(990, 4, DType::Float32, AddrSpace::Global, Some(DeviceSpec::Cpu));
    let scalar = UOp::variable("N".to_string(), 1, 4, DType::Float32).bind(UOp::native_const(2.0f32));

    let local_stack = UOp::new(
        Op::MStack(ops::MStack { buffers: smallvec![local.clone(), local_peer] }),
        DType::Float32.ptr(Some(8), AddrSpace::Local).expect("local ptr"),
    )
    .after(smallvec![UOp::noop()]);
    let load = |buf| UOp::load().index(index_at_zero(buf)).call();
    let value = load(local_stack)
        .try_add(&load(reg.clone().after(smallvec![UOp::noop()])))
        .expect("add")
        .try_add(&load(input.clone()))
        .expect("add")
        .try_add(&scalar)
        .expect("add");
    let call = call_split_store(&index_at_zero(output.clone()).store(value)).expect("STORE should split");

    let Op::Call(ops::Call { body, args, .. }) = call.op() else { panic!("expected CALL") };
    assert_eq!(args.as_slice().len(), 3, "two globals followed by the scalar binding");
    assert!(args.iter().any(|arg| Arc::ptr_eq(arg, &output)));
    assert!(args.iter().any(|arg| Arc::ptr_eq(arg, &input)));

    let last_arg = args.last().expect("scalar binding");
    let Op::Bind(ops::Bind { var: call_var, value: call_value }) = last_arg.op() else {
        panic!("the last CALL arg must be the scalar BIND")
    };
    let Op::Bind(ops::Bind { var: body_var, value: body_value }) = scalar.op() else { unreachable!() };
    assert!(Arc::ptr_eq(call_value, body_value));
    assert!(!Arc::ptr_eq(call_var, body_var), "CALL binding must not alias the body-local PARAM");
    let (Op::Param(ops::Param { arg: call_arg, .. }), Op::Param(ops::Param { arg: body_arg, .. })) =
        (call_var.op(), body_var.op())
    else {
        panic!("scalar binding must retain PARAM semantics")
    };
    assert_eq!(call_arg, body_arg, "boundary identity must not change scalar metadata");
    assert_eq!(call_var.dtype(), body_var.dtype());

    let mut global_slots: Vec<usize> = body
        .toposort()
        .into_iter()
        .filter_map(|u| match u.op() {
            Op::Param(ops::Param { arg, .. }) if arg.addrspace == Some(AddrSpace::Global) => Some(arg.slot),
            _ => None,
        })
        .collect();
    global_slots.sort_unstable();
    global_slots.dedup();
    assert_eq!(global_slots, vec![0, 1], "PARAM slots must be dense CALL positions");

    let program_info = svod_ir::ProgramInfo::from_sink(body, DeviceSpec::Cpu);
    assert_eq!(program_info.globals, vec![0, 1], "PROGRAM globals are direct CALL tuple positions");
    assert_eq!(program_info.vars.len(), 1, "scalar variables are PROGRAM values, not globals");
    for (addrspace, slot) in [(AddrSpace::Local, 700), (AddrSpace::Reg, 800)] {
        assert!(
            body.toposort().iter().any(
                |u| matches!(u.op(), Op::Buffer(ops::Buffer { arg, .. }) if arg.addrspace == Some(addrspace) && arg.slot == slot)
            ),
            "{addrspace:?} slot {slot} must stay inside the body"
        );
    }
}
