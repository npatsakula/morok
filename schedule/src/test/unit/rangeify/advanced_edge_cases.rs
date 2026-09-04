//! Graph shapes `rangeify` must accept without losing the compute.

use std::sync::Arc;

use svod_ir::{DType, Op, SInt, UOp, shape::Shape};
use test_case::test_case;

use crate::rangeify::indexing::is_dead_axis;
use crate::rangeify::transforms::rangeify;

use super::helpers::{create_bufferize, create_const, create_range, create_range_symbolic};
use svod_ir::ops;

fn symbolic_range_size() -> Arc<UOp> {
    create_bufferize(UOp::native_const(1.0f32), vec![create_range_symbolic(UOp::var("size", DType::Index, 0, 1024), 0)])
}

fn symbolic_range_sizes() -> Arc<UOp> {
    let ranges = (0..2)
        .map(|i| create_range_symbolic(UOp::var(format!("size{i}").as_str(), DType::Index, 0, 1024), i))
        .collect();
    create_bufferize(UOp::native_const(2.0f32), ranges)
}

fn symbolic_range_arithmetic() -> Arc<UOp> {
    let n = UOp::variable("n".into(), 0, 512, DType::Int32);
    let size = n.try_mul(&create_const(2)).expect("mul");
    create_bufferize(UOp::native_const(3.0f32), vec![create_range_symbolic(size, 0)])
}

fn mixed_const_and_symbolic_ranges() -> Arc<UOp> {
    let symbolic = create_range_symbolic(UOp::param(0, 1, DType::Index, None), 1);
    create_bufferize(UOp::native_const(1.0f32), vec![create_range(10, 0), symbolic])
}

/// `STAGE(STAGE(STAGE(x, r0), r1), r2)` — each level buffers a different extent.
fn nested_bufferize() -> Arc<UOp> {
    (0..3)
        .fold(UOp::native_const(1.0f32), |inner, i| create_bufferize(inner, vec![create_range(5 * (i as i64 + 1), i)]))
}

/// One STAGE read by two independent consumers.
fn bufferize_with_two_consumers() -> Arc<UOp> {
    let buf = create_bufferize(UOp::native_const(1.0f32), vec![create_range(10, 0)]);
    let buf_shape = buf.shape().expect("shape").expect("static shape");
    let ones: Shape = buf_shape.iter().map(|_| SInt::Const(1)).collect();
    let broadcast =
        |v: f32| UOp::native_const(v).try_reshape(&ones).expect("reshape").try_expand(buf_shape).expect("expand");
    UOp::sink(vec![buf.try_add(&broadcast(2.0)).expect("add"), buf.try_mul(&broadcast(3.0)).expect("mul")])
}

/// One compute staged twice with different iteration spaces.
fn compute_bufferized_twice() -> Arc<UOp> {
    let compute = UOp::native_const(1.0f32);
    UOp::sink(vec![
        create_bufferize(compute.clone(), vec![create_range(10, 0)]),
        create_bufferize(compute, vec![create_range(20, 1)]),
    ])
}

/// A permuted view of a buffer: the index expressions rangeify builds for it are
/// what the symbolic simplification in step 8 has to survive.
fn permuted_buffer() -> Arc<UOp> {
    let src = UOp::new_buffer(svod_device::DeviceSpec::Cpu, 6, DType::Float32);
    let reshaped = src.try_reshape(&smallvec::smallvec![SInt::Const(2), SInt::Const(3)]).expect("reshape");
    reshaped.try_permute(vec![1, 0]).expect("permute")
}

fn index_over_all_stage_ranges() -> Arc<UOp> {
    let ranges = vec![create_range(10, 0), create_range(20, 1), create_range(5, 2)];
    let staged = create_bufferize(UOp::native_const(1.0f32), ranges.clone());
    UOp::new(Op::Index(ops::Index { buffer: staged, indices: ranges.into() }), DType::Float32)
}

#[test_case(super::symbolic_range_size ; "symbolic range size")]
#[test_case(super::symbolic_range_sizes ; "two symbolic range sizes")]
#[test_case(super::symbolic_range_arithmetic ; "symbolic range size from arithmetic")]
#[test_case(super::mixed_const_and_symbolic_ranges ; "const and symbolic ranges mixed")]
#[test_case(super::nested_bufferize ; "three nested stages")]
#[test_case(super::bufferize_with_two_consumers ; "stage read by two consumers")]
#[test_case(super::compute_bufferized_twice ; "compute staged twice")]
#[test_case(super::index_over_all_stage_ranges ; "index over every stage range")]
#[test_case(super::permuted_buffer ; "permuted buffer view")]
fn rangeify_accepts(build: fn() -> Arc<UOp>) {
    let root = build();
    let (result, _ctx) = rangeify(Arc::clone(&root)).expect("rangeify");
    assert_eq!(result.dtype(), root.dtype());
}

/// `is_dead_axis` is `vmax < 1`: extent 0 and 1 collapse, extent 2 and up survive.
#[test_case(0, true ; "empty range")]
#[test_case(1, true ; "singleton range")]
#[test_case(2, false ; "two element range")]
#[test_case(10, false ; "ten element range")]
fn dead_axis_by_extent(extent: i64, dead: bool) {
    assert_eq!(is_dead_axis(&create_range(extent, 0)), dead);
}

#[test]
fn only_ranges_can_be_dead_axes() {
    let c = UOp::index_const(0);
    assert!(!is_dead_axis(&c));
    assert!(!is_dead_axis(&c.try_add(&c).expect("add")));
}
