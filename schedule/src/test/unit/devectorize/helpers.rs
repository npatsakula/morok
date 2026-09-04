//! Builders and assertion helpers shared by the devectorizer tests.

use std::sync::Arc;

use smallvec::SmallVec;
use svod_dtype::{AddrSpace, DType, ScalarDType};
use svod_ir::types::ConstValue;
use svod_ir::{AxisId, AxisType, Op, ReduceOp, UOp};

use crate::devectorize::{bool_storage_patterns, devectorize, no_vectorized_alu};
use crate::optimizer::Renderer;
use crate::rewrite::graph_rewrite;
use svod_ir::ops;

pub fn apply_devectorize(uop: &Arc<UOp>) -> Arc<UOp> {
    devectorize(uop, &Renderer::cpu())
}

/// Bool LOAD/STORE -> uint8 storage only.
pub fn apply_bool_storage(uop: &Arc<UOp>) -> Arc<UOp> {
    graph_rewrite(bool_storage_patterns(), uop.clone(), &mut ())
}

pub fn apply_no_vectorized_alu(uop: &Arc<UOp>) -> Arc<UOp> {
    graph_rewrite(no_vectorized_alu(), uop.clone(), &mut ())
}

/// REDUCE -> accumulator (`reduce_to_acc`).
pub fn apply_pm_reduce(uop: &Arc<UOp>) -> Arc<UOp> {
    use crate::devectorize::{ReduceContext, pm_reduce};
    let mut ctx = ReduceContext::default();
    graph_rewrite(&pm_reduce(), uop.clone(), &mut ctx)
}

pub fn create_buffer(size: usize) -> Arc<UOp> {
    create_buffer_typed(size, ScalarDType::Float32)
}

pub fn create_buffer_typed(size: usize, scalar: ScalarDType) -> Arc<UOp> {
    UOp::new_buffer(svod_dtype::DeviceSpec::Cpu, size, DType::Scalar(scalar))
}

pub fn create_bool_buffer(size: usize) -> Arc<UOp> {
    create_buffer_typed(size, ScalarDType::Bool)
}

/// `INDEX(buffer, [idx])` with a scalar index.
pub fn create_index(buffer: Arc<UOp>, idx: i64) -> Arc<UOp> {
    let idx_uop = UOp::const_(DType::Index, ConstValue::Int(idx));
    UOp::index().buffer(buffer).indices(vec![idx_uop]).call().unwrap()
}

/// Convert a BUFFER to a codegen PARAM. In production this happens during kernel
/// splitting; the shaped-INDEX rules only fire on PARAM.
pub fn buffer_to_define(buffer: &Arc<UOp>) -> Arc<UOp> {
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let size = buffer.buffer_size().unwrap_or(1024);
    UOp::param(id, size, buffer.dtype(), None)
}

/// `INDEX(PARAM, STACK(offsets))` — the shaped memory address the devectorizer splits.
pub fn create_vector_index(buffer: Arc<UOp>, offsets: impl IntoIterator<Item = i64>) -> Arc<UOp> {
    let indices: SmallVec<[Arc<UOp>; 4]> =
        offsets.into_iter().map(|offset| UOp::const_(DType::Index, ConstValue::Int(offset))).collect();
    let idx_dtype = buffer.dtype().base();
    let define = buffer_to_define(&buffer);
    UOp::new(
        Op::Index(ops::Index { buffer: define, indices: smallvec::smallvec![UOp::stack(indices)] }),
        DType::Scalar(idx_dtype),
    )
}

pub fn create_vector_index_iota(buffer: Arc<UOp>, count: usize) -> Arc<UOp> {
    create_vector_index(buffer, 0..count as i64)
}

pub fn create_load(index: Arc<UOp>) -> Arc<UOp> {
    UOp::load().index(index).call()
}

pub fn create_store(index: Arc<UOp>, value: Arc<UOp>) -> Arc<UOp> {
    index.store(value)
}

pub fn create_float_const(value: f64) -> Arc<UOp> {
    UOp::const_(DType::Float32, ConstValue::Float(value))
}

pub fn create_bool_const(value: bool) -> Arc<UOp> {
    UOp::const_(DType::Bool, ConstValue::Bool(value))
}

pub fn create_vector_float_iota(count: usize) -> Arc<UOp> {
    create_vector_float_values((0..count).map(|i| i as f64).collect())
}

pub fn create_vector_float_values(values: Vec<f64>) -> Arc<UOp> {
    UOp::stack(values.into_iter().map(|v| UOp::const_(DType::Float32, ConstValue::Float(v))).collect())
}

pub fn create_vector_bool(values: Vec<bool>) -> Arc<UOp> {
    UOp::stack(values.into_iter().map(|v| UOp::const_(DType::Bool, ConstValue::Bool(v))).collect())
}

pub fn create_reduce(src: Arc<UOp>, ranges: Vec<Arc<UOp>>, reduce_op: ReduceOp) -> Arc<UOp> {
    src.reduce(ranges.into_iter().collect(), reduce_op)
}

/// Parallel axes carry `Index`; sequential ones carry `WeakInt`, as the schedulers build them.
pub fn create_range(end: i64, axis_id: usize, axis_type: AxisType) -> Arc<UOp> {
    let dtype = match axis_type {
        AxisType::Global | AxisType::Local => DType::Index,
        _ => DType::WeakInt,
    };
    UOp::range_axis(UOp::const_(dtype, ConstValue::Int(end)), AxisId::Renumbered(axis_id), axis_type)
}

pub fn create_range_reduce(end: i64, axis_id: usize) -> Arc<UOp> {
    create_range(end, axis_id, AxisType::Reduce)
}

/// The number of scalar elements `uop` carries: from its shape when it has one,
/// from its mechanical vector width otherwise.
pub fn assert_vcount(uop: &Arc<UOp>, expected: usize) {
    let count = uop
        .shape()
        .ok()
        .flatten()
        .and_then(|shape| shape.iter().try_fold(1usize, |product, dim| Some(product * dim.as_const()?)))
        .unwrap_or_else(|| uop.dtype().vcount());
    assert_eq!(count, expected, "element count mismatch: expected {expected}, got {count}");
}

pub fn assert_is_load(uop: &Arc<UOp>) {
    assert!(matches!(uop.op(), Op::Load(..)), "Expected LOAD, got {:?}", uop.op());
}

pub fn assert_is_index(uop: &Arc<UOp>) {
    assert!(matches!(uop.op(), Op::Index(..)), "Expected INDEX, got {:?}", uop.op());
}

pub fn count_ops<F>(uop: &Arc<UOp>, predicate: F) -> usize
where
    F: Fn(&Arc<UOp>) -> bool,
{
    uop.toposort().iter().filter(|node| predicate(node)).count()
}

pub fn count_loads(uop: &Arc<UOp>) -> usize {
    count_ops(uop, |u| matches!(u.op(), Op::Load(..)))
}

pub fn count_stores(uop: &Arc<UOp>) -> usize {
    count_ops(uop, |u| matches!(u.op(), Op::Store(..)))
}

pub fn count_define_regs(uop: &Arc<UOp>) -> usize {
    count_ops(uop, |u| matches!(u.op(), Op::Buffer(ops::Buffer { arg, .. }) if arg.addrspace == Some(AddrSpace::Reg)))
}

pub fn count_ends(uop: &Arc<UOp>) -> usize {
    count_ops(uop, |u| matches!(u.op(), Op::End(..)))
}
