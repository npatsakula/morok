//! End-to-end behaviour of `run_rangeify` / `rangeify` / `try_get_kernel_graph`.

use std::{f32::consts::PI, sync::Arc};

use smallvec::smallvec;
use svod_device::DeviceSpec;
use svod_dtype::DType;
use svod_ir::{AxisId, AxisType, CallInfo, ConstValue, Op, ReduceOp, SInt, UOp};
use test_case::test_case;

use crate::rangeify::{rangeify, run_rangeify, try_get_kernel_graph};
use svod_ir::ops;

struct NoRewrite;

impl svod_ir::Matcher<()> for NoRewrite {
    fn rewrite(&self, _uop: &Arc<UOp>, _ctx: &mut ()) -> svod_ir::RewriteResult {
        svod_ir::RewriteResult::NoMatch
    }
}

struct StripDetach;

impl svod_ir::Matcher<()> for StripDetach {
    fn rewrite(&self, uop: &Arc<UOp>, _ctx: &mut ()) -> svod_ir::RewriteResult {
        match uop.op() {
            Op::Detach(ops::Detach { src }) => svod_ir::RewriteResult::Rewritten(src.clone()),
            _ => svod_ir::RewriteResult::NoMatch,
        }
    }
}

fn rangeify_unwrap(uop: Arc<UOp>) -> Arc<UOp> {
    rangeify(uop).expect("rangeify").0
}

fn store_at_index(value: Arc<UOp>) -> Arc<UOp> {
    UOp::index_const(0).store(value)
}

// ===== run_rangeify =====

#[test]
fn tensor_reduce_becomes_a_loop_reduce_over_ranges() {
    let source = UOp::new_buffer(DeviceSpec::Cpu, 6, DType::Float32)
        .try_reshape(&smallvec![SInt::Const(2), SInt::Const(3)])
        .expect("reshape");
    let tensor_reduce = source.try_reduce_axis(ReduceOp::Add, vec![1]).expect("reduce axis");
    let (rangeified, _ctx) = run_rangeify(UOp::sink(vec![tensor_reduce.contiguous()])).expect("run_rangeify");

    let reductions: Vec<_> =
        rangeified.toposort().into_iter().filter(|node| matches!(node.op(), Op::Reduce(..))).collect();
    assert!(!reductions.is_empty());
    assert!(
        reductions
            .iter()
            .all(|node| matches!(node.op(), Op::Reduce(ops::Reduce { ranges, num_axes: 0, .. }) if !ranges.is_empty())),
        "every REDUCE must carry explicit ranges and no tensor axes"
    );
}

#[test_case(&[(0, 0), (0, 0)]; "no padding")]
#[test_case(&[(1, 1), (2, 2)]; "symmetric")]
#[test_case(&[(3, 0), (0, 5)]; "one sided")]
fn run_rangeify_lowers_every_pad(pads: &[(usize, usize)]) {
    let dims: smallvec::SmallVec<[SInt; 4]> = smallvec![SInt::Const(4), SInt::Const(5)];
    let source = UOp::new_buffer(DeviceSpec::Cpu, 20, DType::Float32).try_reshape(&dims).expect("reshape");
    let padded = source.try_pad(&pads.iter().map(|&(lo, hi)| (lo.into(), hi.into())).collect::<Vec<_>>()).expect("pad");

    let (rangeified, _ctx) = run_rangeify(UOp::sink(vec![padded.contiguous()])).expect("run_rangeify");
    assert!(
        !rangeified.toposort().iter().any(|node| matches!(node.op(), Op::Pad(..) | Op::ReduceAxis(..))),
        "rangeify must lower every PAD:\n{}",
        rangeified.tree()
    );
}

#[test]
fn run_rangeify_preserves_call_and_function_bodies_by_default() {
    let p0 = UOp::param(0, 8, DType::Float32, None);
    let reduced = p0.try_reduce_axis(ReduceOp::Add, vec![0]).expect("reduce axis should construct");
    let arg = UOp::new_buffer(DeviceSpec::Cpu, 8, DType::Float32);

    let call = reduced.call(smallvec![arg.clone()], CallInfo::default());
    let (rangeified_call, _ctx) = run_rangeify(call).expect("run_rangeify should preserve call bodies");
    let Op::Call(ops::Call { body: call_body, .. }) = rangeified_call.op() else {
        panic!("expected CALL root after run_rangeify")
    };
    assert!(
        call_body.toposort().iter().any(|u| matches!(u.op(), Op::Reduce(ops::Reduce { num_axes: 1, .. }))),
        "run_rangeify should not rewrite CALL body by default"
    );

    let function = reduced.function(smallvec![arg], CallInfo::default());
    let (rangeified_function, _ctx) = run_rangeify(function).expect("run_rangeify should preserve function bodies");
    let Op::Function(ops::Function { body: function_body, .. }) = rangeified_function.op() else {
        panic!("expected FUNCTION root after run_rangeify")
    };
    assert!(
        function_body.toposort().iter().any(|u| matches!(u.op(), Op::Reduce(ops::Reduce { num_axes: 1, .. }))),
        "run_rangeify should not rewrite FUNCTION body by default"
    );
}

#[test]
fn only_an_explicit_full_traversal_rewrites_inside_call_and_function_bodies() {
    let detached = UOp::native_const(1.0f32).detach();
    let arg = UOp::native_const(2.0f32);
    let has_detach = |body: &Arc<UOp>| body.toposort().iter().any(|u| matches!(u.op(), Op::Detach(..)));

    for opaque in [
        detached.call(smallvec![arg.clone()], CallInfo::default()),
        detached.function(smallvec![arg], CallInfo::default()),
    ] {
        let preserved =
            svod_ir::rewrite::graph_rewrite_with_bpm_preserve_calls(&NoRewrite, &StripDetach, opaque, &mut ());
        let body = match preserved.op() {
            Op::Call(ops::Call { body, .. }) | Op::Function(ops::Function { body, .. }) => body,
            op => panic!("expected an opaque root, got {op:?}"),
        };
        assert!(has_detach(body), "preserve-calls rewrite must leave the body alone");

        let full = svod_ir::rewrite::graph_rewrite_with_bpm(&NoRewrite, &StripDetach, Arc::clone(&preserved), &mut ());
        let body = match full.op() {
            Op::Call(ops::Call { body, .. }) | Op::Function(ops::Function { body, .. }) => body,
            op => panic!("expected an opaque root, got {op:?}"),
        };
        assert!(!has_detach(body), "an explicit full rewrite must reach into the body");
    }
}

// ===== Full pipeline =====

/// DETACH and CONTIGUOUS_BACKWARD are stripped by `earliest_rewrites`, ahead of
/// `run_rangeify`. Pattern-level coverage lives in `patterns.rs`.
#[test]
fn autograd_markers_are_gone_after_the_full_pipeline() {
    let x = UOp::native_const(1.0f32);
    for marked in [x.detach(), x.contiguous_backward()] {
        assert!(Arc::ptr_eq(&rangeify_unwrap(marked), &x));
    }
}

#[test_case(store_at_index(UOp::native_const(1.0f32)) ; "store")]
#[test_case(store_at_index(UOp::native_const(2.0f32)).end(smallvec![UOp::range_axis(UOp::index_const(100), AxisId::Renumbered(0), AxisType::Loop)]) ; "end of store")]
#[test_case(store_at_index(UOp::native_const(2.0f32).try_add(&UOp::native_const(3.0f32)).expect("add")) ; "store of arithmetic")]
fn rangeify_then_kernel_split_produces_a_void_kernel_graph(root: Arc<UOp>) {
    let (kernel, _ctx) = try_get_kernel_graph(rangeify_unwrap(root)).expect("kernel split");
    assert_eq!(kernel.dtype(), DType::Void);
}

/// LOADs feeding a STORE stay inside one kernel; the CALL owns every buffer.
#[test_case(1 ; "one load")]
#[test_case(2 ; "two loads")]
fn loads_feeding_one_store_split_into_one_kernel(loads: usize) {
    let buffer = || UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let at = UOp::index_const(0);
    let load =
        |buf| UOp::load().index(UOp::index().buffer(buf).indices(vec![at.clone()]).call().expect("index")).call();

    let value = (1..loads).fold(load(buffer()), |acc, _| acc.try_add(&load(buffer())).expect("add"));
    let store = UOp::index().buffer(buffer()).indices(vec![at]).call().expect("index").store(value);

    let (result, _ctx) = try_get_kernel_graph(store).expect("kernel split");
    assert_eq!(super::helpers::count_kernels(&result), 1);
}

/// Reductions over a range-independent source lose the loop entirely: ADD scales
/// the value by the extent, MAX leaves it untouched. MIN is deliberately absent
/// upstream — see `reduce_simplify::min_is_not_an_unparented_fold`.
#[test_case(ReduceOp::Add, 10, ConstValue::Int(50) ; "add scales by the extent")]
#[test_case(ReduceOp::Max, 5, ConstValue::Int(5) ; "max is idempotent")]
fn unparented_reductions_collapse_to_a_constant(op: ReduceOp, extent: i64, expected: ConstValue) {
    let range = UOp::range_axis(UOp::index_const(extent), AxisId::Renumbered(0), AxisType::Reduce);
    let reduce = UOp::native_const(5i32).reduce(vec![range].into(), op);

    let result = rangeify_unwrap(reduce);
    assert!(matches!(result.op(), Op::Const(c) if c.0 == expected), "got {}", result.tree());
}

/// A source that reads the range cannot be collapsed — the REDUCE survives.
#[test]
fn range_dependent_reductions_are_kept() {
    let range = UOp::range_axis(UOp::index_const(10), AxisId::Renumbered(0), AxisType::Reduce);
    let src = range.cast(DType::Int32).try_add(&UOp::native_const(1i32)).expect("add");
    let reduce = src.reduce(vec![range].into(), ReduceOp::Add);

    let result = rangeify_unwrap(reduce);
    assert!(result.toposort().iter().any(|n| matches!(n.op(), Op::Reduce(..))), "got {}", result.tree());
}

/// `split_reduceop` materialises an intermediate (a CONTIGUOUS) only once the
/// reduced extent passes its threshold. Threshold arithmetic: `split_reduceop.rs`.
#[test_case(1_000, false ; "below threshold")]
#[test_case(100_000, true ; "above threshold")]
fn large_reductions_are_split_in_two_stages(size: usize, split: bool) {
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, size, DType::Float32);
    let reduce = buffer.try_reduce_axis(ReduceOp::Add, vec![0]).expect("reduce axis");

    let rangeified = rangeify_unwrap(reduce);
    let has_contiguous = rangeified.toposort().iter().any(|node| matches!(node.op(), Op::Contiguous(..)));
    assert_eq!(has_contiguous, split);
    assert_eq!(rangeified.dtype(), DType::Float32);
}

#[test]
fn a_multi_range_reduction_survives_the_pipeline() {
    let ranges = (0..2)
        .map(|i| UOp::range_axis(UOp::index_const(8 >> i), AxisId::Renumbered(i), AxisType::Reduce))
        .collect::<Vec<_>>();
    let (rangeified, _ctx) =
        run_rangeify(UOp::native_const(PI).reduce(ranges.into(), ReduceOp::Add)).expect("run_rangeify");

    assert_eq!(rangeified.dtype(), DType::Float32);
}
