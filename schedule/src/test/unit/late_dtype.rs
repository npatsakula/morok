use std::sync::Arc;

use svod_dtype::{AddrSpace, DType, ScalarDType};
use svod_ir::{ConstValue, Op, ReduceOp, UOp};

use crate::late::demote_unsupported_floats;
use crate::optimizer::Renderer;

fn element(buffer: Arc<UOp>, lane: i64) -> Arc<UOp> {
    UOp::index().buffer(buffer).indices(vec![UOp::const_(DType::Index, ConstValue::Int(lane))]).call().unwrap()
}

fn has_dtype(root: &Arc<UOp>, scalar: ScalarDType) -> bool {
    root.toposort().iter().any(|node| node.dtype().base() == scalar)
}

/// `out = f32((f64(in) * 0.5) + 1.25)` — the linspace shape.
fn internal_f64_sink() -> Arc<UOp> {
    let input = UOp::load().index(element(UOp::param(1, 4, DType::Float32, None), 0)).call();
    let scaled = input
        .cast(DType::Float64)
        .try_mul(&UOp::const_(DType::Float64, ConstValue::Float(0.5)))
        .unwrap()
        .try_add(&UOp::const_(DType::Float64, ConstValue::Float(1.25)))
        .unwrap();
    UOp::sink(vec![element(UOp::param(0, 4, DType::Float32, None), 0).store(scaled.cast(DType::Float32))])
}

#[test]
fn internal_f64_computes_in_f32_when_unsupported() {
    let sink = internal_f64_sink();
    assert!(has_dtype(&sink, ScalarDType::Float64));
    let demoted = demote_unsupported_floats(sink, &Renderer::metal());
    assert!(!has_dtype(&demoted, ScalarDType::Float64), "{}", demoted.tree());
    let consts: Vec<_> = demoted
        .toposort()
        .into_iter()
        .filter(|node| matches!(node.op(), Op::Const(value) if matches!(value.0, ConstValue::Float(_))))
        .collect();
    assert_eq!(consts.len(), 2, "{}", demoted.tree());
    assert!(consts.iter().all(|node| node.dtype() == DType::Float32));
}

#[test]
fn supported_renderers_are_untouched() {
    let sink = internal_f64_sink();
    let same = demote_unsupported_floats(sink.clone(), &Renderer::cpu());
    assert!(Arc::ptr_eq(&sink, &same));
}

#[test]
fn external_f64_storage_keeps_its_dtype() {
    let input = UOp::param(1, 4, DType::Float64, None);
    let load = UOp::load().index(element(input.clone(), 0)).call();
    let doubled = load.try_mul(&UOp::const_(DType::Float64, ConstValue::Float(2.0))).unwrap();
    let sink = UOp::sink(vec![element(UOp::param(0, 4, DType::Float32, None), 0).store(doubled.cast(DType::Float32))]);
    let demoted = demote_unsupported_floats(sink, &Renderer::metal());
    let nodes = demoted.toposort();
    let param = nodes.iter().find(|node| matches!(node.op(), Op::Param(..)) && node.dtype() == DType::Float64);
    assert!(param.is_some(), "external storage must keep Float64:\n{}", demoted.tree());
    let load = nodes.iter().find(|node| matches!(node.op(), Op::Load(..))).expect("load survives");
    assert_eq!(load.dtype(), DType::Float64);
    // The arithmetic itself runs in f32 on a converted load.
    let mul = nodes.iter().find(|node| matches!(node.op(), Op::Binary(svod_ir::BinaryOp::Mul, ..))).expect("mul");
    assert_eq!(mul.dtype(), DType::Float32, "{}", demoted.tree());
}

#[test]
fn scratch_buffers_and_reductions_are_demoted() {
    let local = UOp::buffer(3, 16, DType::Float64, AddrSpace::Local, None);
    let value = UOp::load().index(element(UOp::param(1, 16, DType::Float32, None), 0)).call().cast(DType::Float64);
    let fill = element(local.clone(), 0).store(value);
    let range = UOp::range_axis_dtype(
        UOp::const_(DType::Int32, ConstValue::Int(16)),
        svod_ir::AxisId::Renumbered(0),
        svod_ir::AxisType::Reduce,
        DType::Int32,
    );
    let partial = UOp::load().index(UOp::index().buffer(local).indices(vec![range.clone()]).call().unwrap()).call();
    let sum = partial.reduce(smallvec::smallvec![range.clone()], ReduceOp::Add);
    let out = element(UOp::param(0, 1, DType::Float32, None), 0).store(sum.cast(DType::Float32));
    let sink = UOp::sink(vec![fill, sum.end(smallvec::smallvec![range]), out]);

    let demoted = demote_unsupported_floats(sink, &Renderer::metal());
    assert!(!has_dtype(&demoted, ScalarDType::Float64), "{}", demoted.tree());
    let local = demoted
        .toposort()
        .into_iter()
        .find(|node| matches!(node.op(), Op::Buffer(svod_ir::ops::Buffer { arg, .. }) if arg.addrspace == Some(AddrSpace::Local)))
        .expect("local buffer");
    let Op::Buffer(svod_ir::ops::Buffer { arg, .. }) = local.op() else { unreachable!() };
    assert_eq!(arg.dtype, DType::Float32);
}

#[test]
fn vector_f64_becomes_vector_f32() {
    let lanes = UOp::vconst(vec![ConstValue::Float(1.0), ConstValue::Float(2.0)], DType::Float64);
    let sink = UOp::sink(vec![lanes.try_add(&lanes).unwrap()]);
    let demoted = demote_unsupported_floats(sink, &Renderer::metal());
    assert!(!has_dtype(&demoted, ScalarDType::Float64), "{}", demoted.tree());
    assert!(demoted.toposort().iter().any(|node| node.dtype() == DType::Float32.vec(2).unwrap()), "{}", demoted.tree());
}

/// A gated load from f64 storage keeps its `alt` in f64: the load's dtype is
/// pinned by its address and a demoted `alt` would make the node ill-formed.
#[test]
fn gated_external_load_keeps_its_alt_dtype() {
    let input = UOp::param(1, 4, DType::Float64, None);
    let zero = UOp::const_(DType::Float64, ConstValue::Float(0.0));
    let gate = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let load = UOp::new(
        Op::Load(svod_ir::ops::Load { index: element(input, 0), alt: Some(zero.clone()), gate: Some(gate) }),
        DType::Float64,
    );
    // The same constant also feeds internal math, which still runs in f32.
    let internal = zero.try_add(&UOp::const_(DType::Float64, ConstValue::Float(1.0))).unwrap();
    let value = load.try_add(&internal).unwrap().cast(DType::Float32);
    let sink = UOp::sink(vec![element(UOp::param(0, 4, DType::Float32, None), 0).store(value)]);

    let demoted = demote_unsupported_floats(sink, &Renderer::metal());
    let nodes = demoted.toposort();
    let load = nodes.iter().find(|node| matches!(node.op(), Op::Load(..))).expect("load survives");
    let Op::Load(svod_ir::ops::Load { alt: Some(alt), .. }) = load.op() else { unreachable!() };
    assert_eq!((load.dtype(), alt.dtype()), (DType::Float64, DType::Float64), "{}", demoted.tree());
    let add = nodes
        .iter()
        .find(|node| matches!(node.op(), Op::Binary(svod_ir::BinaryOp::Add, ..)) && node.dtype() == DType::Float32)
        .expect("internal add runs in f32");
    assert!(add.op().sources().iter().all(|source| source.dtype() == DType::Float32), "{}", demoted.tree());
}
