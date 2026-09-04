//! `bufferize_to_store`: `STAGE(compute, ranges)` becomes
//! `AFTER(BUFFER, [END(STORE(INDEX(BUFFER, ..), compute), ranges)])`.
//!
//! The BUFFER → PARAM conversion happens later, in `split_store`.

use std::sync::Arc;

use svod_ir::{Op, UOp};

use crate::rangeify::{RangeifyBufferContext, bufferize_to_store};
use svod_ir::ops;

#[test]
fn a_staged_compute_becomes_a_buffer_backed_store() {
    let mut ctx = RangeifyBufferContext::new();
    let compute = UOp::native_const(42.0f32);
    let range = UOp::range_const(10, 0);
    let stage = UOp::stage_global(Arc::clone(&compute), vec![Arc::clone(&range)]);

    let result = bufferize_to_store(&stage, &mut ctx).expect("a global STAGE converts");

    let Op::After(ops::After { passthrough, deps }) = result.op() else {
        panic!("expected AFTER, got {}", result.tree())
    };
    assert!(matches!(passthrough.op(), Op::Buffer(..)), "the passthrough is the allocated BUFFER");
    let [dep] = deps.as_slice() else { panic!("expected exactly one dep") };

    let Op::End(ops::End { computation, ranges }) = dep.op() else { panic!("expected END, got {}", dep.tree()) };
    assert_eq!(ranges.as_slice().len(), 1);
    assert!(Arc::ptr_eq(&ranges[0], &range), "END closes the STAGE's own range");

    let Op::Store(ops::Store { index, value, gate }) = computation.op() else {
        panic!("expected STORE inside the END")
    };
    assert!(gate.is_none());
    assert!(Arc::ptr_eq(value, &compute));
    let Op::Index(ops::Index { buffer, .. }) = index.op() else { panic!("expected INDEX, got {}", index.tree()) };
    assert!(Arc::ptr_eq(buffer, passthrough), "the STORE writes the buffer the AFTER passes through");

    let tracked = ctx.get_buffer(&stage).expect("the STAGE is tracked");
    assert!(Arc::ptr_eq(tracked, &result), "the context maps the STAGE to the whole AFTER");
    assert_eq!(ctx.local_counter, 0, "a global BUFFER does not consume a local slot");
}

/// Multi-range STAGEs are lowered upstream; reaching one here is a bug, not a
/// case to linearise.
#[test]
#[should_panic(expected = "unexpected multi-range")]
fn a_multi_range_stage_is_rejected() {
    let ranges = vec![UOp::range_const(4, 0), UOp::range_const(8, 1)];
    let stage = UOp::stage_global(UOp::native_const(100i32), ranges);

    bufferize_to_store(&stage, &mut RangeifyBufferContext::new());
}

#[test]
fn only_a_stage_converts() {
    let mut ctx = RangeifyBufferContext::new();
    assert!(bufferize_to_store(&UOp::native_const(1.0f32), &mut ctx).is_none());
}
