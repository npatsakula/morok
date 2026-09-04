//! End-to-end `devectorize()`.

use std::sync::Arc;

use svod_dtype::{AddrSpace, DType, ScalarDType};
use svod_ir::types::ConstValue;
use svod_ir::uop::cached_property::CachedProperty;
use svod_ir::uop::properties::InScopeRangesProperty;
use svod_ir::{AxisId, AxisType, Op, UOp};
use test_case::test_case;

use super::helpers::*;
use svod_ir::ops;

/// A shaped memory read of `n` lanes becomes `n` scalar LOADs under one STACK,
/// whatever the offsets or element type.
#[test_case(ScalarDType::Float32, &[0, 1, 2, 3]; "contiguous")]
#[test_case(ScalarDType::Float32, &[0, 1, 2, 3, 4, 5, 6, 7]; "eight wide output upcast")]
#[test_case(ScalarDType::Float32, &[0, 2, 4, 6]; "strided")]
#[test_case(ScalarDType::Float32, &[3, 4, 5, 6]; "unaligned start")]
#[test_case(ScalarDType::Float32, &[0, 1, 2]; "three lanes")]
#[test_case(ScalarDType::Float32, &[0, 1, 2, 3, 4]; "five lanes")]
#[test_case(ScalarDType::Float32, &[9000, 9001, 9002, 9003]; "large offset")]
#[test_case(ScalarDType::Float16, &[0, 1, 2, 3]; "half precision")]
#[test_case(ScalarDType::Int8, &[0, 1, 2, 3]; "int8")]
#[test_case(ScalarDType::UInt8, &[0, 1, 2, 3]; "uint8")]
#[test_case(ScalarDType::Int32, &[0, 1, 2, 3]; "int32")]
fn shaped_load_becomes_one_scalar_load_per_lane(scalar: ScalarDType, offsets: &[i64]) {
    let index = create_vector_index(create_buffer_typed(16384, scalar), offsets.iter().copied());
    let result = apply_devectorize(&UOp::load().index(index).call());

    assert_vcount(&result, offsets.len());
    assert_eq!(count_loads(&result), offsets.len());
    let Op::Stack(ops::Stack { sources }) = result.op() else { panic!("expected a STACK of lanes: {}", result.tree()) };
    assert_eq!(sources.len(), offsets.len());
    assert!(sources.iter().all(|lane| lane.dtype() == DType::Scalar(scalar)));
}

/// Wide vectors are scalarized the same way, without a width cap.
#[test_case(32; "vec32")]
#[test_case(64; "vec64")]
fn wide_shaped_load_is_fully_scalarized(width: usize) {
    let index = create_vector_index_iota(create_buffer(16384), width);
    let result = apply_devectorize(&UOp::load().index(index).call());

    assert_vcount(&result, width);
    assert_eq!(count_loads(&result), width);
}

/// `c[0..4] = a[0..4] + b[0..4]` scalarizes on both sides, and no memory op keeps
/// a vector dtype.
#[test]
fn shaped_elementwise_kernel_scalarizes_loads_and_stores() {
    let load = |buffer| UOp::load().index(create_vector_index_iota(buffer, 4)).call();
    let sum = load(create_buffer(64)).add(&load(create_buffer(64)));
    let result = apply_devectorize(&create_vector_index_iota(create_buffer(64), 4).store(sum));

    assert_eq!(count_loads(&result), 8);
    assert_eq!(count_stores(&result), 4);
    assert!(
        !result
            .toposort()
            .iter()
            .any(|node| { matches!(node.op(), Op::Load(..) | Op::Store(..)) && node.dtype().vcount() > 1 })
    );
}

#[test]
fn sink_scalarizes_every_shaped_store() {
    let store = |value| create_vector_index_iota(create_buffer(64), 4).store(value);
    let sink = UOp::sink(vec![store(create_vector_float_iota(4)), store(create_vector_float_values(vec![9.0; 4]))]);

    assert_eq!(count_stores(&apply_devectorize(&sink)), 8);
}

/// A loop-dependent address (`range * 4 + lane`) scalarizes like a constant one.
#[test]
fn loop_dependent_shaped_load_is_scalarized() {
    let buffer = UOp::param(20000, 256, DType::Float32, None);
    let base = create_range(64, 0, AxisType::Loop).mul(&UOp::index_const(4));
    let offsets = UOp::stack((0..4).map(|lane| base.add(&UOp::index_const(lane))).collect());
    let index = UOp::new(Op::Index(ops::Index { buffer, indices: smallvec::smallvec![offsets] }), DType::Float32);

    let result = apply_devectorize(&UOp::load().index(index).call());

    assert_vcount(&result, 4);
    assert_eq!(count_loads(&result), 4);
}

/// A shaped STORE into a register file keeps every lane inside the enclosing loop.
#[test]
fn shaped_register_store_preserves_outer_range() {
    let outer = UOp::range_axis(UOp::index_const(4), AxisId::Unrenumbered(0), AxisType::Loop);
    let register = UOp::buffer(0, 2, DType::Float32, AddrSpace::Reg, None);
    let zeros = UOp::stack(vec![create_float_const(0.0), create_float_const(0.0)].into());

    let result = apply_devectorize(&register.after(vec![outer.clone()].into()).store(zeros));

    let stores = result.toposort().into_iter().filter(|node| matches!(node.op(), Op::Store(..)));
    let stores = stores.collect::<Vec<_>>();
    assert_eq!(stores.len(), 2);
    assert!(stores.iter().all(|store| InScopeRangesProperty::get(store).iter().any(|range| *range == outer.id)));
}

/// A memory address stays `INDEX(PARAM, flat_offset)` with a scalar offset; only a
/// value-space INDEX (into a STACK) keeps a shape.
#[test]
fn flat_2d_memory_index_and_shaped_value_index_remain_distinct() {
    let buffer = UOp::param(22000, 64, DType::Float32, None);
    let row = UOp::range_const(8, 22001);
    let row_offset = row.mul(&UOp::index_const(8));
    let offsets = UOp::stack((0..4).map(|lane| row_offset.add(&UOp::index_const(lane))).collect());
    let memory_index = UOp::index().buffer(buffer.clone()).indices(vec![offsets]).call().unwrap();
    let result = apply_devectorize(&UOp::load().index(memory_index).call());

    for node in result.toposort().into_iter().filter(|node| matches!(node.op(), Op::Index(..))) {
        let Op::Index(ops::Index { buffer: address, indices }) = node.op() else { unreachable!() };
        if address.addrspace().is_some() {
            assert!(
                Arc::ptr_eq(address, &buffer),
                "memory lane must remain INDEX(PARAM, flat_offset):\n{}",
                node.tree()
            );
            assert_eq!(indices.len(), 1);
            assert!(indices[0].shape().unwrap().unwrap().is_empty());
        }
    }

    let shaped = UOp::stack((0i32..4).map(UOp::native_const).collect())
        .try_reshape(&smallvec::smallvec![svod_ir::SInt::Const(2), svod_ir::SInt::Const(2)])
        .unwrap();
    let shaped_index =
        UOp::index().buffer(shaped.clone()).indices(vec![row.mod_(&UOp::index_const(2))]).call().unwrap();
    assert!(shaped_index.addrspace().is_none());
    assert_eq!(shaped_index.shape().unwrap().unwrap().as_slice(), &[svod_ir::SInt::Const(2)]);
    assert!(matches!(shaped_index.op(), Op::Index(ops::Index { buffer: source, .. }) if Arc::ptr_eq(source, &shaped)));
}

/// `devectorize` is a single `graph_rewrite` (tinygrad `codegen/__init__.py:333`), so it
/// must reach a fixed point in one pass.
#[test]
fn test_devectorize_is_idempotent() {
    let buffer = create_buffer(64);
    let load = UOp::load().index(create_vector_index_iota(buffer.clone(), 4)).call();
    let store = create_vector_index_iota(buffer, 4).store(create_vector_float_iota(4));
    for root in [load, store] {
        let once = apply_devectorize(&root);
        assert!(Arc::ptr_eq(&apply_devectorize(&once), &once), "{}", once.tree());
    }
}

/// The devectorizer's index grouping relies on `is_increasing` to tell a monotone
/// address apart from an arbitrary one.
#[test]
fn is_increasing_tracks_monotone_index_expressions() {
    let range = create_range(16, 0, AxisType::Loop);
    let weak = |value| UOp::const_(DType::WeakInt, ConstValue::Int(value));
    assert!(range.is_increasing());
    assert!(UOp::const_(DType::Int32, ConstValue::Int(5)).is_increasing());
    assert!(range.try_add(&weak(5)).unwrap().is_increasing());
    assert!(range.try_mul(&weak(4)).unwrap().is_increasing());
    assert!(
        !UOp::var("x", DType::Int32, 0, 100)
            .try_mul(&UOp::const_(DType::Int32, ConstValue::Int(-1)))
            .unwrap()
            .is_increasing()
    );
}

/// A scalar access is already devectorized and must survive untouched.
#[test]
fn scalar_memory_ops_pass_through() {
    let index = create_index(create_buffer(64), 5);
    assert_is_index(&apply_devectorize(&index));
    let load = apply_devectorize(&UOp::load().index(index).call());
    assert_is_load(&load);
    assert_eq!(load.dtype(), DType::Float32);
}

/// Devectorize runs tinygrad's `symbolic_simple` tier, which does not flatten SINK.
#[test]
fn sink_structure_is_preserved() {
    assert!(
        matches!(apply_devectorize(&UOp::sink(vec![])).op(), Op::Sink(ops::Sink { sources, .. }) if sources.is_empty())
    );
    let result = apply_devectorize(&UOp::sink(vec![UOp::noop()]));
    assert!(
        matches!(result.op(), Op::Sink(ops::Sink { sources, .. }) if sources.len() == 1 && matches!(sources[0].op(), Op::Noop)),
        "devectorize must not run the larger sym cleanup tier: {}",
        result.tree()
    );
}
