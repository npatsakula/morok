//! Tests for `pm_generate_realize_map` rows ported from tinygrad's
//! `tinygrad/schedule/indexing.py:37-56`.

use std::sync::Arc;

use smallvec::smallvec;
use svod_device::DeviceSpec;
use svod_dtype::DType;
use svod_ir::{AxisId, AxisType, CallInfo, Op, UOp};

use crate::rangeify::IndexingContext;
use crate::rangeify::indexing::pm_generate_realize_map;

fn run(root: Arc<UOp>, ctx: &mut IndexingContext) {
    crate::rewrite::graph_rewrite_bottom_up_preserve_calls(pm_generate_realize_map(), root, ctx);
}

fn buffer(size: usize) -> Arc<UOp> {
    UOp::new_buffer(DeviceSpec::Cpu, size, DType::Float32)
}

/// `CALL(SINK, args...)` — a hand-written kernel over `args`.
fn call_with(args: Vec<Arc<UOp>>) -> Arc<UOp> {
    let body = UOp::sink(vec![UOp::param(0, 4, DType::Float32, Some(DeviceSpec::Cpu))]);
    UOp::new(Op::Call { body, args: args.into_iter().collect(), info: CallInfo::default().into() }, DType::Void)
}

/// `STORE(INDEX(dest, r), value)` — `dest` is indexed without any movement op.
fn store_of(value: Arc<UOp>) -> Arc<UOp> {
    let r = UOp::range_axis(UOp::index_const(4), AxisId::Renumbered(0), AxisType::Loop);
    let index = UOp::index().buffer(buffer(4)).indices(vec![r]).call().expect("INDEX");
    index.store(value)
}

#[test]
fn custom_kernel_source_is_realized_and_pinned() {
    let arg = UOp::native_const(1.0f32).try_add(&UOp::native_const(2.0f32)).expect("add");
    let mut ctx = IndexingContext::new();

    run(call_with(vec![Arc::clone(&arg)]), &mut ctx);

    assert!(ctx.should_realize(&arg), "a CALL input that is not already a buffer must be realized");
    assert!(ctx.is_non_removable_realize(&arg), "the kernel reads it through a PARAM slot, so it must stay a buffer");
}

#[test]
fn custom_kernel_source_realizes_through_reshapes() {
    // `while s.op is Ops.RESHAPE: s = s.src[0]` — the compute is realized, not
    // the view of it.
    let compute = UOp::native_const(1.0f32).try_add(&UOp::native_const(2.0f32)).expect("add");
    let view = compute.try_reshape(&smallvec![1usize.into()]).expect("reshape");
    let mut ctx = IndexingContext::new();

    run(call_with(vec![Arc::clone(&view)]), &mut ctx);

    assert!(ctx.should_realize(&compute), "the reshaped-through source is the one realized");
    assert!(!ctx.should_realize(&view), "the RESHAPE view itself needs no buffer");
}

#[test]
fn custom_kernel_buffer_source_is_left_alone() {
    let buf = buffer(4);
    let mut ctx = IndexingContext::new();

    run(call_with(vec![Arc::clone(&buf)]), &mut ctx);

    assert!(!ctx.should_realize(&buf), "an ALWAYS_CONTIGUOUS CALL input is already a buffer");
}

#[test]
fn slice_source_of_store_loses_its_realize_entry() {
    let slice = buffer(8).contiguous_slice(4, 0, DType::Float32);
    let store = store_of(Arc::clone(&slice));

    let mut ctx = IndexingContext::new();
    ctx.mark_realize_pending(&slice);
    run(store, &mut ctx);

    assert!(!ctx.should_realize(&slice), "the store target already is the output buffer");
}

#[test]
fn slice_source_keeps_its_realize_entry_behind_a_movement_op() {
    let slice = buffer(8).contiguous_slice(4, 0, DType::Float32);
    let r = UOp::range_axis(UOp::index_const(4), AxisId::Renumbered(0), AxisType::Loop);
    let dest = buffer(4)
        .try_reshape(&smallvec![2usize.into(), 2usize.into()])
        .expect("reshape")
        .try_permute(vec![1, 0])
        .expect("permute");
    assert!(matches!(dest.op(), Op::Permute { .. }), "the test needs a real PERMUTE on the destination");
    let index = UOp::index().buffer(dest).indices(vec![r]).call().expect("INDEX");
    let store = index.store(Arc::clone(&slice));

    let mut ctx = IndexingContext::new();
    ctx.mark_realize_pending(&slice);
    run(store, &mut ctx);

    assert!(ctx.should_realize(&slice), "a moved destination does not line up with the SLICE");
}
