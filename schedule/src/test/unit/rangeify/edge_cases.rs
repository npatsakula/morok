//! Degenerate iteration spaces: no ranges at all, and a zero-sized range.

use std::sync::Arc;

use smallvec::SmallVec;
use svod_dtype::DType;
use svod_ir::{AddrSpace, BufferizeOpts, Op, UOp};

use crate::rangeify::{RangeifyBufferContext, bufferize_to_store, try_get_kernel_graph};

use super::helpers::extract_kernel;

fn scalar_stage() -> Arc<UOp> {
    let opts = BufferizeOpts { device: None, local_axis: None, addrspace: AddrSpace::Global, removable: true };
    UOp::new(
        Op::Stage { compute: UOp::native_const(42.0f32), ranges: SmallVec::new(), opts: opts.into() },
        DType::Float32,
    )
}

#[test]
fn rangeless_stage_stores_without_an_end_wrapper() {
    let stage = scalar_stage();
    let result = bufferize_to_store(&stage, &mut RangeifyBufferContext::new()).expect("scalar STAGE converts");

    let Op::After { passthrough, deps } = result.op() else { panic!("expected AFTER, got {}", result.tree()) };
    assert!(matches!(passthrough.op(), Op::Buffer { .. }));
    let [store] = deps.as_slice() else { panic!("expected exactly one dep") };
    let Op::Store { value, .. } = store.op() else { panic!("no ranges means STORE is not wrapped in END") };
    assert!(matches!(value.op(), Op::Const(_)));
}

#[test]
fn rangeless_stage_still_produces_a_kernel() {
    let (result, _ctx) = try_get_kernel_graph(scalar_stage()).expect("kernel split");
    assert!(matches!(extract_kernel(&result).expect("CALL").op(), Op::Call { .. }));
}

/// Tinygrad's `assert size > 0`: an empty range cannot back a buffer.
#[test]
#[should_panic(expected = "Cannot allocate buffer: range vmax resolved to")]
fn zero_sized_range_cannot_be_allocated() {
    let opts = BufferizeOpts { device: None, local_axis: None, addrspace: AddrSpace::Global, removable: true };
    let ranges = smallvec::smallvec![UOp::range_const(0, 0)];
    let stage = UOp::new(Op::Stage { compute: UOp::native_const(1.0f32), ranges, opts: opts.into() }, DType::Float32);

    bufferize_to_store(&stage, &mut RangeifyBufferContext::new());
}

/// Tinygrad-aligned: closing zero ranges is the identity, not a fresh END node.
#[test]
fn end_over_no_ranges_returns_the_computation() {
    let store = UOp::noop();
    assert!(Arc::ptr_eq(&store.clone().end(SmallVec::new()), &store));
}
