//! Kernel counts the rangeify + kernel-split pipeline produces for the graph
//! shapes fusion has to get right.

use std::sync::Arc;

use smallvec::smallvec;
use svod_dtype::DType;
use svod_ir::{Op, ReduceOp, UOp};
use test_case::test_case;

use crate::rangeify::{rangeify_with_map, try_get_kernel_graph};

use super::helpers::count_kernels;
use svod_ir::ops;

fn buffer(size: usize) -> Arc<UOp> {
    UOp::new_buffer(svod_dtype::DeviceSpec::Cpu, size, DType::Float32)
}

fn reshape_2d(src: Arc<UOp>, rows: i64, cols: i64) -> Arc<UOp> {
    let new_shape = UOp::stack(smallvec![UOp::index_const(rows), UOp::index_const(cols)]);
    UOp::new(Op::Reshape(ops::Reshape { src, new_shape }), DType::Float32)
}

fn binop() -> Arc<UOp> {
    UOp::sink(vec![buffer(100).try_add(&buffer(100)).expect("add")])
}

fn binop_chain() -> Arc<UOp> {
    let sum = buffer(100).try_add(&buffer(100)).expect("add");
    UOp::sink(vec![sum.try_add(&buffer(100)).expect("add")])
}

fn binop_then_reshape() -> Arc<UOp> {
    UOp::sink(vec![reshape_2d(buffer(100).try_add(&buffer(100)).expect("add"), 10, 10)])
}

fn binop_then_permute() -> Arc<UOp> {
    let reshaped = reshape_2d(buffer(100).try_add(&buffer(100)).expect("add"), 10, 10);
    UOp::sink(vec![UOp::new(Op::Permute(ops::Permute { src: reshaped, axes: vec![1, 0] }), DType::Float32)])
}

fn reduce() -> Arc<UOp> {
    let reshaped = reshape_2d(buffer(100), 10, 10);
    UOp::sink(vec![reshaped.try_reduce_axis(ReduceOp::Add, vec![1]).expect("reduce")])
}

fn binop_then_reduce() -> Arc<UOp> {
    let reshaped = reshape_2d(buffer(100).try_add(&buffer(100)).expect("add"), 10, 10);
    UOp::sink(vec![reshaped.try_reduce_axis(ReduceOp::Add, vec![1]).expect("reduce")])
}

/// CONTIGUOUS is a realize point, so the ADD it wraps cannot fuse into the sink.
fn contiguous_between_binops() -> Arc<UOp> {
    let inner = buffer(100).try_add(&buffer(100)).expect("add").contiguous();
    UOp::sink(vec![inner.try_mul(&buffer(100)).expect("mul")])
}

/// Two outputs reading one shared ADD: with no CONTIGUOUS to pin it, the ADD is
/// cheap enough to inline into both kernels rather than buffer it.
fn two_outputs_share_an_add() -> Arc<UOp> {
    let shared = buffer(100).try_add(&buffer(100)).expect("add");
    UOp::sink(vec![shared.try_mul(&buffer(100)).expect("mul"), shared.try_mul(&buffer(100)).expect("mul")])
}

fn single_const() -> Arc<UOp> {
    UOp::sink(vec![UOp::native_const(1.0f32)])
}

fn empty_sink() -> Arc<UOp> {
    UOp::sink(vec![])
}

#[test_case(super::binop, 1 ; "a + b is one kernel")]
#[test_case(super::binop_chain, 1 ; "a + b + c is one kernel")]
#[test_case(super::binop_then_reshape, 1 ; "reshape does not break fusion")]
#[test_case(super::binop_then_permute, 1 ; "permute does not break fusion")]
#[test_case(super::reduce, 1 ; "reduce is one kernel")]
#[test_case(super::binop_then_reduce, 1 ; "elementwise fuses into the reduce")]
#[test_case(super::contiguous_between_binops, 2 ; "contiguous forces a second kernel")]
#[test_case(super::two_outputs_share_an_add, 2 ; "shared add is inlined into both outputs")]
#[test_case(super::single_const, 1 ; "a bare const still writes its output")]
#[test_case(super::empty_sink, 0 ; "empty sink launches nothing")]
fn kernel_count(build: fn() -> Arc<UOp>, expected: usize) {
    let built = build();
    let Op::Sink(ops::Sink { sources, .. }) = built.op() else { panic!("builders return SINKs") };
    let root = UOp::sink(sources.iter().map(|source| source.contiguous()).collect());
    let rangeified = rangeify_with_map(root).expect("rangeify");
    let (kernels, _) = try_get_kernel_graph(rangeified.sink).expect("kernel graph");
    assert_eq!(count_kernels(&kernels), expected);
}
