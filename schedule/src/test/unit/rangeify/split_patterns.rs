//! `to_param_patterns`: the rewrite `split_store` runs over a kernel body to turn
//! storage into codegen PARAMs, unbind scalars, canonicalise range ids, and peel
//! AFTER ordering wrappers.

use std::sync::Arc;

use smallvec::smallvec;
use svod_dtype::{AddrSpace, DType};
use svod_ir::{AxisId, AxisType, ConstValue, Op, UOp};
use test_case::test_case;

use crate::rangeify::{RangeifyBufferContext, patterns::to_param_patterns};
use svod_ir::ops;

fn apply(uop: &Arc<UOp>, ctx: &mut RangeifyBufferContext) -> Option<Arc<UOp>> {
    match to_param_patterns().rewrite(uop, ctx) {
        svod_ir::pattern::RewriteResult::Rewritten(result) => Some(result),
        _ => None,
    }
}

fn range(end: i64, axis_id: AxisId, axis_type: AxisType) -> Arc<UOp> {
    UOp::range_axis(UOp::index_const(end), axis_id, axis_type)
}

// ===== storage becomes a dense codegen PARAM =====

/// Each BUFFER becomes the next PARAM slot and is mapped to it, so later reads of
/// the same BUFFER reuse the slot.
#[test]
fn buffers_are_numbered_into_dense_param_slots() {
    let mut ctx = RangeifyBufferContext::new();

    for slot in 0..2 {
        let buffer = UOp::new_buffer(svod_device::DeviceSpec::Cpu, 100 * (slot + 1), DType::Float32);
        let param = apply(&buffer, &mut ctx).expect("a BUFFER becomes a PARAM");

        assert!(matches!(param.op(), Op::Param(ops::Param { arg, .. })
            if arg.slot == slot && arg.device == Some(svod_device::DeviceSpec::Cpu)));
        assert_eq!(ctx.global_counter, slot + 1);
        assert!(Arc::ptr_eq(ctx.get_buffer(&buffer).expect("mapped"), &param));
    }
}

/// A bound scalar becomes a scalar PARAM and its value moves into `ctx.vars`, to
/// be passed at launch.
#[test]
fn a_bound_variable_becomes_a_scalar_param_and_a_launch_value() {
    let mut ctx = RangeifyBufferContext::new();
    let var = UOp::variable("x".to_string(), 0, 10, DType::WeakInt);
    let bind = var.bind(UOp::const_(DType::WeakInt, ConstValue::Int(5)));

    let param = apply(&bind, &mut ctx).expect("a BIND unbinds");

    assert!(matches!(param.op(), Op::Param(ops::Param { arg, .. }) if arg.addrspace.is_none()));
    let (_, bound) = ctx.vars.get("x").expect("the value is recorded for launch");
    assert_eq!(*bound, Some(5));
}

// ===== range canonicalisation =====

/// Unrenumbered ranges get sequential canonical ids, keeping their axis type and
/// extent node; an already-renumbered range is left alone.
#[test]
fn unrenumbered_ranges_are_numbered_sequentially() {
    let mut ctx = RangeifyBufferContext::new();
    let axis_types = [AxisType::Loop, AxisType::Loop, AxisType::Reduce];

    for (i, axis_type) in axis_types.into_iter().enumerate() {
        let original = range(10 * (i as i64 + 1), AxisId::Unrenumbered(i + 5), axis_type);
        let renumbered = apply(&original, &mut ctx).expect("an unrenumbered range is renumbered");

        let Op::Range(ops::Range { axis_id, axis_type: kept, end, .. }) = renumbered.op() else {
            panic!("expected RANGE, got {}", renumbered.tree())
        };
        assert_eq!(*axis_id, AxisId::Renumbered(i));
        assert_eq!(*kept, axis_type, "renumbering must not change the axis type");
        let Op::Range(ops::Range { end: original_end, .. }) = original.op() else { unreachable!() };
        assert!(Arc::ptr_eq(end, original_end), "the extent node is preserved");
    }
    assert_eq!(ctx.range_counter, axis_types.len());
}

/// Nothing to do: an already-canonical range with a non-zero extent, and a bare
/// CONST, are both left as they are.
#[test_case(range(10, AxisId::Renumbered(5), AxisType::Loop) ; "already renumbered")]
#[test_case(range(10, AxisId::Renumbered(0), AxisType::Loop) ; "renumbered with a non-zero extent")]
#[test_case(UOp::native_const(42i32) ; "a bare const")]
fn canonical_nodes_are_left_alone(uop: Arc<UOp>) {
    assert!(apply(&uop, &mut RangeifyBufferContext::new()).is_none());
}

/// An empty range materialises as index 0. It still consumes its canonical id,
/// and the constant stays weak until target-width lowering.
#[test]
fn a_zero_extent_range_collapses_to_a_weak_zero() {
    let mut ctx = RangeifyBufferContext::new();
    let empty = range(0, AxisId::Renumbered(0), AxisType::Loop);

    let collapsed = apply(&empty, &mut ctx).expect("an empty range collapses");

    assert!(matches!(collapsed.op(), Op::Const(v) if v.0 == ConstValue::Int(0)));
    assert_eq!(collapsed.dtype(), DType::WeakInt);
    assert_eq!(ctx.range_counter, 1, "a materialized RANGE must still consume its canonical id");
}

// ===== AFTER peeling and buffer tracking =====

fn global_param() -> Arc<UOp> {
    let dtype = DType::Float32.ptr(Some(1024), AddrSpace::Global).expect("global ptr");
    UOp::param(11, 1024, DType::Scalar(dtype.base()), None)
}

fn local_buffer() -> Arc<UOp> {
    UOp::buffer(1, 1024, DType::Float32, AddrSpace::Local, None)
}

fn unique_buffer() -> Arc<UOp> {
    UOp::buffer_id(Some(0))
}

/// AFTER is an ordering wrapper: the pattern unwraps it to the storage it passes
/// through. Global storage is recorded in the buffer map so later readers pick up
/// the same ordering edge; local storage is kernel-scoped and synchronised by
/// BARRIER instead, so it must not be tracked.
#[test_case(super::global_param, true ; "global param is tracked")]
#[test_case(super::unique_buffer, true ; "unique buffer is tracked")]
#[test_case(super::local_buffer, false ; "local buffer is not tracked")]
fn after_unwraps_to_its_storage_and_tracks_only_global(build: fn() -> Arc<UOp>, tracked: bool) {
    let mut ctx = RangeifyBufferContext::new();
    let storage = build();
    let after = storage.clone().after(smallvec![UOp::noop()]);

    let unwrapped = apply(&after, &mut ctx).expect("AFTER unwraps");

    assert!(Arc::ptr_eq(&unwrapped, &storage));
    assert_eq!(ctx.has_buffer(&storage), tracked);
    if tracked {
        assert!(Arc::ptr_eq(ctx.get_buffer(&storage).expect("tracked"), &after), "the map points at the AFTER");
    }
}

fn mstack(first: Arc<UOp>, second: Arc<UOp>) -> Arc<UOp> {
    let dtype = first.dtype();
    UOp::new(Op::MStack(ops::MStack { buffers: smallvec![first, second] }), dtype)
}

fn mselect(buffer: Arc<UOp>) -> Arc<UOp> {
    let dtype = buffer.dtype();
    UOp::new(Op::MSelect(ops::MSelect { buffer, device_index: 0 }), dtype)
}

/// Multi-device wrappers resolve to a single representative buffer — the first of
/// an MSTACK, the selected one of an MSELECT — and that buffer, not the wrapper,
/// is what the AFTER is recorded against.
#[test]
fn after_sees_through_multi_device_wrappers() {
    for wrap in [
        mstack(UOp::buffer_id(Some(1)), UOp::buffer_id(Some(2))),
        mselect(UOp::buffer_id(Some(1))),
        mstack(local_buffer(), UOp::buffer(2, 1024, DType::Float32, AddrSpace::Local, None)),
        mselect(local_buffer()),
    ] {
        let mut ctx = RangeifyBufferContext::new();
        let representative = match wrap.op() {
            Op::MStack(ops::MStack { buffers }) => buffers[0].clone(),
            Op::MSelect(ops::MSelect { buffer, .. }) => buffer.clone(),
            _ => unreachable!(),
        };
        let is_local = matches!(representative.op(), Op::Buffer(ops::Buffer { arg, .. }) if arg.addrspace == Some(AddrSpace::Local));
        let after = wrap.after(smallvec![UOp::noop()]);

        let unwrapped = apply(&after, &mut ctx).expect("AFTER unwraps the wrapper");

        assert!(Arc::ptr_eq(&unwrapped, &representative));
        assert_eq!(ctx.has_buffer(&representative), !is_local);
        if !is_local {
            assert!(Arc::ptr_eq(ctx.get_buffer(&representative).expect("tracked"), &after));
        }
    }
}

#[test]
#[ignore = "MSTACK/AFTER handling not fully implemented yet"]
fn an_after_with_no_deps_still_tracks_its_mstack() {
    let mut ctx = RangeifyBufferContext::new();
    let buf1 = UOp::buffer_id(Some(1));
    let stack =
        UOp::new(Op::MStack(ops::MStack { buffers: smallvec![buf1.clone(), UOp::buffer_id(Some(2))] }), DType::Float32);
    let after = stack.clone().after(smallvec::SmallVec::new());

    let unwrapped = apply(&after, &mut ctx).expect("AFTER unwraps");

    assert!(Arc::ptr_eq(&unwrapped, &buf1));
    assert!(ctx.buffer_map.contains_key(&svod_ir::UOpKey(stack)));
}
