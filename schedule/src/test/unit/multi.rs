use std::sync::Arc;

use smallvec::smallvec;
use svod_device::DeviceSpec;
use svod_dtype::DType;
use svod_ir::{BinaryOp, Error, Op, ReduceOp, SInt, UOp};
use test_case::test_case;

use crate::multi::{lower_allreduce_pm, multi_pm, validate_no_unresolved_allreduce, validate_supported_subset};
use crate::optimizer::apply_pre_optimization;
use crate::rangeify::rangeify_with_map;
use crate::rewrite::graph_rewrite;
use svod_ir::ops;

fn buffer(size: usize) -> Arc<UOp> {
    UOp::new_buffer(DeviceSpec::Cpu, size, DType::Float32)
}

/// An 8-element buffer viewed as `[2, 4]`, so both axes can carry a layout.
fn matrix() -> Arc<UOp> {
    buffer(8).try_reshape(&smallvec![SInt::Const(2), SInt::Const(4)]).unwrap()
}

fn sharded(axis: usize) -> Arc<UOp> {
    UOp::multi(matrix(), axis)
}

fn add(lhs: Arc<UOp>, rhs: Arc<UOp>) -> Arc<UOp> {
    UOp::new(Op::Binary(BinaryOp::Add, lhs, rhs), DType::Float32)
}

#[test]
fn mselect_mstack_selects_the_requested_shard() {
    let shard1 = buffer(8);
    let selected = UOp::mstack(smallvec![buffer(8), shard1.clone()]).mselect(1);
    let result = graph_rewrite(&multi_pm(), selected, &mut ());
    assert!(Arc::ptr_eq(&result, &shard1));
}

#[test]
fn pre_optimization_does_not_repeat_multi_rewrite() {
    let stacked = UOp::mstack(smallvec![buffer(6), buffer(6)]);
    let reshaped = stacked.try_reshape(&smallvec![SInt::Const(2), SInt::Const(3)]).unwrap();
    let result = apply_pre_optimization(reshaped.mselect(1)).unwrap();

    assert!(matches!(result.op(), Op::MSelect(..)), "the per-kernel optimizer must not rerun multi_pm");
}

#[test]
fn same_axis_alu_runs_per_shard() {
    let local0 = buffer(8);
    let local1 = buffer(8);
    let result = graph_rewrite(&multi_pm(), add(UOp::multi(local0.clone(), 0), UOp::multi(local1.clone(), 0)), &mut ());

    let Op::Multi(ops::Multi { src, axis: 0 }) = result.op() else { panic!("expected MULTI, got {:?}", result.op()) };
    assert!(matches!(src.op(), Op::Binary(BinaryOp::Add, a, b) if Arc::ptr_eq(a, &local0) && Arc::ptr_eq(b, &local1)));
}

#[test]
fn a_scalar_operand_needs_no_layout_of_its_own() {
    let local = buffer(8);
    let scalar = UOp::native_const(2.0f32);
    let result = graph_rewrite(&multi_pm(), add(UOp::multi(local.clone(), 0), scalar.clone()), &mut ());

    assert!(matches!(result.op(), Op::Multi(ops::Multi { src, axis: 0 })
        if matches!(src.op(), Op::Binary(BinaryOp::Add, lhs, rhs)
            if Arc::ptr_eq(lhs, &local) && Arc::ptr_eq(rhs, &scalar))));
    validate_supported_subset(&result).unwrap();
}

#[test]
fn permute_remaps_the_shard_axis() {
    let local = buffer(6).try_reshape(&smallvec![SInt::Const(2), SInt::Const(3)]).unwrap();
    let permute =
        UOp::new(Op::Permute(ops::Permute { src: UOp::multi(local.clone(), 0), axes: vec![1, 0] }), DType::Float32);
    let result = graph_rewrite(&multi_pm(), permute, &mut ());

    assert!(matches!(result.op(), Op::Multi(ops::Multi { src, axis: 1 })
        if matches!(src.op(), Op::Permute(ops::Permute { src: inner, axes }) if Arc::ptr_eq(inner, &local) && axes == &[1, 0])));
}

#[test]
fn a_reduce_over_another_axis_keeps_the_shard_layout() {
    let local = buffer(8);
    let reduce = UOp::multi(local.clone(), 1).reduce_with_num_axes(smallvec![], ReduceOp::Add, 1);
    let result = graph_rewrite(&multi_pm(), reduce, &mut ());

    assert!(matches!(result.op(), Op::Multi(ops::Multi { src, axis: 0 })
        if matches!(src.op(), Op::Reduce(ops::Reduce { src: inner, num_axes: 1, .. }) if Arc::ptr_eq(inner, &local))));
}

#[test]
fn a_non_sharded_reduce_axis_runs_per_shard_before_rangeify() {
    let local = matrix();
    let reduced = UOp::multi(local.clone(), 1).try_reduce_axis(ReduceOp::Add, vec![0]).unwrap();
    let rewritten = graph_rewrite(&multi_pm(), reduced.clone(), &mut ());

    assert!(matches!(rewritten.op(), Op::Multi(ops::Multi { src, axis: 0 })
        if matches!(src.op(), Op::Reduce(ops::Reduce { src: inner, ranges, num_axes: 1, .. })
            if Arc::ptr_eq(inner, &local) && ranges.is_empty())));
    validate_supported_subset(&rewritten).unwrap();

    let rangeified = rangeify_with_map(UOp::sink(vec![reduced])).unwrap();
    let topo = rangeified.sink.toposort();
    assert!(topo.iter().any(|node| matches!(node.op(), Op::Multi(ops::Multi { axis: 0, .. }))));
    assert!(
        topo.iter()
            .all(|node| !matches!(node.op(), Op::Reduce(ops::Reduce { src, .. }) if matches!(src.op(), Op::Multi(..))))
    );
}

/// Forms `multi_pm` must leave alone: either single-device, or missing the
/// resharding metadata that would let it push the rewrite through.
#[test_case(add(UOp::multi(buffer(8), 0), UOp::multi(buffer(8), 1)); "mixed shard axes")]
#[test_case(UOp::new(Op::Reshape(ops::Reshape { src: UOp::multi(buffer(8), 0), new_shape: UOp::index_const(8) }), DType::Float32); "reshape without a shard count")]
#[test_case(UOp::mstack(smallvec![buffer(8), buffer(8)]).mselect(2); "mselect out of range")]
#[test_case(add(buffer(8), buffer(8)); "single-device graph")]
fn multi_pm_leaves_unsupported_forms_alone(node: Arc<UOp>) {
    let result = graph_rewrite(&multi_pm(), node.clone(), &mut ());
    assert!(Arc::ptr_eq(&result, &node), "rewrote into {}", result.tree());
}

fn heterogeneous_shard_reduce() -> Arc<UOp> {
    let float = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let integer = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Int32);
    UOp::multi(UOp::mstack(smallvec![float, integer]), 0).try_reduce_axis(ReduceOp::Add, vec![0]).unwrap()
}

#[test_case(add(sharded(0), sharded(1)), |e| matches!(e, Error::MultiAxisMismatch { .. }); "mixed shard axes")]
#[test_case(UOp::multi(sharded(0), 0), |e| matches!(e, Error::MultiNested { .. }); "nested multi")]
#[test_case(
    UOp::new(Op::Reshape(ops::Reshape { src: sharded(0), new_shape: UOp::index_const(8) }), DType::Float32),
    |e| matches!(e, Error::MultiMovementUnsupported { operation: "RESHAPE", .. }); "reshape across the shard boundary")]
#[test_case(
    UOp::new(Op::Flip(ops::Flip { src: sharded(0), axes: vec![true, false] }), DType::Float32),
    |e| matches!(e, Error::MultiMovementUnsupported { operation: "FLIP", axis: 0, .. }); "flip of the shard axis")]
#[test_case(add(sharded(0), matrix()), |e| matches!(e, Error::MultiLayoutMissing { axis: 0 , .. }); "operand without a layout")]
#[test_case(
    sharded(0).try_reduce_axis(ReduceOp::Add, vec![0]).unwrap(),
    |e| matches!(e, Error::MultiReductionAcrossShardAxis { axis: 0 }); "sum across the shard axis without explicit shards")]
#[test_case(
    UOp::multi(UOp::mstack(smallvec![buffer(4), buffer(4)]), 0).try_reduce_axis(ReduceOp::Mul, vec![0]).unwrap(),
    |e| matches!(e, Error::MultiReductionAcrossShardAxis { axis: 0 }); "product is not a supported collective")]
#[test_case(heterogeneous_shard_reduce(), |e| e.to_string().contains("identical dtype and shape"); "shards of different dtypes")]
fn rangeify_rejects_unsupported_multi_forms_with_typed_errors(node: Arc<UOp>, expected: fn(&Error) -> bool) {
    let err = rangeify_with_map(UOp::sink(vec![node])).err().expect("unsupported MULTI form");
    assert!(expected(&err), "unexpected error: {err:?}");
}

#[test]
fn rangeify_runs_multi_before_tagging() {
    let result = rangeify_with_map(UOp::sink(vec![add(UOp::multi(buffer(8), 0), UOp::multi(buffer(8), 0))])).unwrap();

    assert!(result.uop_list.iter().all(|node| {
        !matches!(node.op(), Op::Binary(..))
            || node.op().sources().iter().all(|source| !matches!(source.op(), Op::Multi(..)))
    }));
    assert!(result.sink.toposort().iter().any(|node| matches!(node.op(), Op::Multi(..))));
}

#[test]
fn rangeify_resolves_mselect_before_movement_lowering() {
    let shard1 = buffer(6);
    let stacked = UOp::mstack(smallvec![buffer(6), shard1.clone()]);
    let reshaped = stacked.try_reshape(&smallvec![SInt::Const(2), SInt::Const(3)]).unwrap();
    let result = rangeify_with_map(UOp::sink(vec![reshaped.mselect(1)])).unwrap();

    assert!(result.uop_list.iter().any(|node| Arc::ptr_eq(node, &shard1)));
    assert!(result.sink.toposort().iter().all(|node| !matches!(node.op(), Op::MSelect(..))));
}

#[test]
fn independent_outputs_may_have_different_single_axis_layouts() {
    validate_supported_subset(&UOp::sink(vec![sharded(0), sharded(1)])).unwrap();
}

/// A reduction over the shard axis becomes a per-shard local reduce feeding one
/// ALLREDUCE; a non-leading shard axis is permuted to the front first.
#[test_case(0, ReduceOp::Add, false; "leading shard axis")]
#[test_case(1, ReduceOp::Add, true; "non-leading shard axis")]
#[test_case(0, ReduceOp::Max, false; "max collective")]
fn shard_axis_reduce_emits_local_reduce_then_allreduce(axis: usize, reduce_op: ReduceOp, permuted: bool) {
    let shard0 = matrix();
    let shard1 = matrix();
    let shards = UOp::mstack(smallvec![shard0.clone(), shard1.clone()]);
    let reduced = UOp::multi(shards, axis).try_reduce_axis(reduce_op, vec![axis]).unwrap();
    let rewritten = graph_rewrite(&multi_pm(), reduced.clone(), &mut ());

    let Op::AllReduce(ops::AllReduce { src, reduce_op: collective, .. }) = rewritten.op() else {
        panic!("expected ALLREDUCE, got {:?}", rewritten.op());
    };
    assert_eq!(collective, &reduce_op);
    let Op::MStack(ops::MStack { buffers }) = src.op() else { panic!("expected local reduction MSTACK") };
    assert_eq!(buffers.len(), 2);
    for (local, shard) in buffers.iter().zip([shard0, shard1]) {
        let Op::Reduce(ops::Reduce { src, ranges, num_axes: 1, .. }) = local.op() else {
            panic!("expected tensor REDUCE")
        };
        assert!(ranges.is_empty());
        if permuted {
            assert!(
                matches!(src.op(), Op::Permute(ops::Permute { src, axes }) if Arc::ptr_eq(src, &shard) && axes == &[1, 0])
            );
        } else {
            assert!(Arc::ptr_eq(src, &shard));
        }
    }
    validate_supported_subset(&rewritten).unwrap();

    let rangeified = rangeify_with_map(UOp::sink(vec![reduced])).unwrap();
    assert!(
        rangeified
            .sink
            .toposort_call_aware(true)
            .iter()
            .all(|node| !matches!(node.op(), Op::Reduce(ops::Reduce { num_axes, .. }) if *num_axes != 0))
    );
}

#[test]
fn reduced_precision_cast_is_restored_around_collective() {
    let low0 = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float16);
    let low1 = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float16);
    let shards = UOp::mstack(smallvec![low0.cast(DType::Float32), low1.cast(DType::Float32)]);
    let reduced = UOp::multi(shards, 0).try_reduce_axis(ReduceOp::Add, vec![0]).unwrap();
    let rewritten = graph_rewrite(&multi_pm(), reduced, &mut ());

    let Op::Cast(ops::Cast { src: collective, dtype: DType::Scalar(svod_dtype::ScalarDType::Float32) }) =
        rewritten.op()
    else {
        panic!("expected widened result cast, got {:?}", rewritten.op());
    };
    let Op::AllReduce(ops::AllReduce { src, .. }) = collective.op() else { panic!("expected ALLREDUCE") };
    let Op::MStack(ops::MStack { buffers }) = src.op() else { panic!("expected MSTACK") };
    assert!(buffers.iter().all(|local| local.dtype() == DType::Float16));
}

#[test]
fn allreduce_lowers_to_opaque_host_call_before_program_codegen() {
    let local0 = buffer(4).try_reduce_axis(ReduceOp::Add, vec![0]).unwrap();
    let local1 = buffer(4).try_reduce_axis(ReduceOp::Add, vec![0]).unwrap();
    let allreduce = UOp::allreduce(UOp::mstack(smallvec![local0, local1]), DeviceSpec::Cpu, ReduceOp::Add);
    let lowered = graph_rewrite(&lower_allreduce_pm(), allreduce.clone(), &mut ());
    validate_no_unresolved_allreduce(&lowered).unwrap();

    let Op::After(ops::After { deps, .. }) = lowered.op() else { panic!("expected AFTER output") };
    let Op::Call(ops::Call { body, args, .. }) = deps[0].op() else { panic!("expected host collective CALL") };
    assert!(matches!(
        body.op(),
        Op::CustomFunction(ops::CustomFunction {
            kind: svod_ir::CustomFunctionKind::AllReduce { reduce_op: ReduceOp::Add },
            ..
        })
    ));
    assert_eq!(args.len(), 3, "output plus two explicit shard buffers");
    assert!(matches!(args[0].op(), Op::Contiguous(..)));
    assert!(Arc::ptr_eq(&args[0], &args[1]), "collective output must alias materialized shard zero");
    assert!(matches!(body.op(), Op::CustomFunction(ops::CustomFunction { attrs, .. }) if attrs.len() == args.len()));
    assert!(lowered.toposort_call_aware(true).iter().all(|node| !matches!(node.op(), Op::AllReduce(..))));

    let rangeified = rangeify_with_map(UOp::sink(vec![allreduce])).unwrap();
    assert!(rangeified.sink.toposort_call_aware(true).iter().all(|node| !matches!(node.op(), Op::AllReduce(..))));
    assert!(rangeified.sink.toposort().iter().any(|node| matches!(
        node.op(),
        Op::CustomFunction(ops::CustomFunction { kind: svod_ir::CustomFunctionKind::AllReduce { .. }, .. })
    )));
}
