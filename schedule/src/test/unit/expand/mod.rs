use smallvec::smallvec;
use svod_dtype::DType;
use svod_ir::{AxisId, AxisType, Op, ReduceOp, RendererDevice, SInt, UOp, WmmaMetadata, WmmaUpcastAxes};

use crate::devectorize::pm_expand_broadcast;
use crate::expand::{build_range_map, pm_group_for_reduce, pre_expand};
use crate::rewrite::graph_rewrite;
use svod_ir::ops;

fn wmma_metadata(name: &str, upcast_axes: Option<WmmaUpcastAxes>) -> WmmaMetadata {
    WmmaMetadata {
        name: name.into(),
        dims: (16, 16, 16),
        dtype_in: DType::Float32,
        dtype_out: DType::Float32,
        device: RendererDevice::Cpu,
        threads: 32,
        upcast_axes,
        reduce_axes: vec![],
        tile_grid: (1, 1),
    }
}

/// A STACK of `count` distinct constants reshaped to `shape`.
fn shaped(count: usize, shape: &[usize]) -> std::sync::Arc<UOp> {
    UOp::stack((0..count).map(|value| UOp::native_const(value as f32)).collect())
        .try_reshape(&shape.iter().copied().map(SInt::Const).collect())
        .unwrap()
}

#[test]
fn range_map_uses_kernel_coordinate_order() {
    let first = UOp::range_axis(UOp::index_const(2), AxisId::Renumbered(7), AxisType::Upcast);
    let second = UOp::range_axis(UOp::index_const(3), AxisId::Renumbered(9), AxisType::Unroll);
    let sink = UOp::sink(vec![first, second]);
    let map = build_range_map(&sink);
    assert_eq!(map[&AxisId::Renumbered(7)], 0);
    assert_eq!(map[&AxisId::Renumbered(9)], 1);
}

#[test]
fn upcast_and_unroll_ranges_become_shaped_coordinates() {
    let first = UOp::range_axis(UOp::index_const(2), AxisId::Renumbered(7), AxisType::Upcast);
    let second = UOp::range_axis(UOp::index_const(3), AxisId::Renumbered(9), AxisType::Unroll);
    let result = pre_expand(&UOp::sink(vec![first, second]));
    let Op::Sink(ops::Sink { sources, .. }) = result.op() else { panic!("expected SINK") };
    assert_eq!(sources[0].shape().unwrap().unwrap().as_slice(), &[SInt::Const(2), SInt::Const(1)]);
    assert_eq!(sources[1].shape().unwrap().unwrap().as_slice(), &[SInt::Const(1), SInt::Const(3)]);
}

#[test]
fn expansion_runs_movement_cleanup_in_the_same_fixpoint() {
    let range = UOp::range_axis(UOp::index_const(4), AxisId::Renumbered(7), AxisType::Upcast);
    let result = pre_expand(&UOp::sink(vec![range]));
    let Op::Sink(ops::Sink { sources, .. }) = result.op() else { panic!("expected SINK") };
    assert!(matches!(sources[0].op(), Op::Stack(..)), "{}", sources[0].tree());

    let buffer = UOp::param(0, 4, DType::Float32, None);
    let indexed = (0..4)
        .map(|index| UOp::index().buffer(buffer.clone()).indices(vec![UOp::index_const(index)]).call().unwrap())
        .collect();
    let result = pre_expand(&UOp::sink(vec![UOp::stack(indexed)]));
    let Op::Sink(ops::Sink { sources, .. }) = result.op() else { panic!("expected SINK") };
    assert!(std::sync::Arc::ptr_eq(&sources[0], &buffer), "{}", sources[0].tree());
}

#[test]
fn wmma_shapes_operands_independently_and_reconstructs_output() {
    let first = UOp::range_axis(UOp::index_const(2), AxisId::Renumbered(7), AxisType::Upcast);
    let second = UOp::range_axis(UOp::index_const(3), AxisId::Renumbered(9), AxisType::Upcast);
    let a_lanes = shaped(2, &[2, 1]);
    let b_lanes = shaped(3, &[1, 3]);
    let metadata = wmma_metadata(
        "test",
        Some(WmmaUpcastAxes {
            a: vec![(AxisId::Renumbered(7), 2)],
            b: vec![(AxisId::Renumbered(9), 3)],
            c: vec![(AxisId::Renumbered(7), 2)],
        }),
    );
    let accumulator = UOp::stack(smallvec![UOp::native_const(0.0f32); 2]);
    let result = pre_expand(&UOp::sink(vec![first, second, UOp::wmma(a_lanes, b_lanes, accumulator, metadata)]));
    let Op::Sink(ops::Sink { sources, .. }) = result.op() else { panic!("expected SINK") };
    assert_eq!(sources[2].shape().unwrap().unwrap().as_slice(), &[SInt::Const(2), SInt::Const(1)]);
    let expanded = sources[2].toposort().into_iter().find(|u| matches!(u.op(), Op::Wmma(..))).unwrap();
    let Op::Wmma(ops::Wmma { a, b, metadata, .. }) = expanded.op() else { unreachable!() };
    assert_eq!(a.shape().unwrap().unwrap().as_slice(), &[SInt::Const(1), SInt::Const(2)]);
    assert_eq!(b.shape().unwrap().unwrap().as_slice(), &[SInt::Const(1), SInt::Const(3)]);
    assert!(metadata.upcast_axes.is_none());
}

#[test]
fn nested_split_axis_survives_wmma_contract_and_output_unroll() {
    let nested = AxisId::Renumbered(7).child(1).child(0);
    let scalar = AxisId::Renumbered(7);
    let nested_range = UOp::range_axis(UOp::index_const(2), nested.clone(), AxisType::Upcast);
    let scalar_range = UOp::range_axis(UOp::index_const(3), scalar, AxisType::Upcast);
    let lanes = shaped(6, &[2, 3]);
    let metadata = wmma_metadata(
        "nested-test",
        Some(WmmaUpcastAxes { a: vec![(nested.clone(), 2)], b: vec![(nested.clone(), 2)], c: vec![(nested, 2)] }),
    );
    let accumulator = UOp::stack(smallvec![UOp::native_const(0.0f32); 2]);
    let result = pre_expand(&UOp::sink(vec![
        nested_range,
        scalar_range,
        UOp::wmma(lanes.clone(), lanes, accumulator, metadata),
    ]));
    let Op::Sink(ops::Sink { sources, .. }) = result.op() else { panic!("expected SINK") };
    assert_eq!(sources[2].shape().unwrap().unwrap().as_slice(), &[SInt::Const(2), SInt::Const(3)]);
    let expanded = sources[2].toposort().into_iter().find(|u| matches!(u.op(), Op::Wmma(..))).unwrap();
    let Op::Wmma(ops::Wmma { a, b, metadata, .. }) = expanded.op() else { unreachable!() };
    assert_eq!(a.shape().unwrap().unwrap().as_slice(), &[SInt::Const(3), SInt::Const(2)]);
    assert_eq!(b.shape().unwrap().unwrap().as_slice(), &[SInt::Const(3), SInt::Const(2)]);
    assert!(metadata.upcast_axes.is_none());
}

#[test]
fn wmma_broadcast_stacks_fragments_before_reshape() {
    let metadata = wmma_metadata("broadcast-test", None);
    let wmma = UOp::wmma(shaped(64, &[4, 1, 16]), shaped(64, &[1, 4, 16]), shaped(8, &[8]), metadata);
    let result = graph_rewrite(pm_expand_broadcast(), wmma, &mut ());

    assert_eq!(result.shape().unwrap().unwrap().as_slice(), &[SInt::Const(4), SInt::Const(4), SInt::Const(8)]);
    let Op::Reshape(ops::Reshape { src, .. }) = result.op() else { panic!("expected STACK reshape") };
    assert!(matches!(src.op(), Op::Stack(ops::Stack { sources }) if sources.len() == 16));
}

#[test]
fn pre_expansion_wmma_accepts_scalar_inputs() {
    let metadata = wmma_metadata("scalar-input-test", None);
    let accumulator = UOp::stack(smallvec![UOp::native_const(0.0f32); 8]);
    let wmma = UOp::wmma(UOp::native_const(1.0f32), UOp::native_const(2.0f32), accumulator, metadata);
    assert_eq!(wmma.shape().unwrap().unwrap().as_slice(), &[SInt::Const(8)]);
}

#[test]
fn shaped_reduce_axes_are_expanded_before_reduction_lowering() {
    let upcast = UOp::range_axis(UOp::index_const(4), AxisId::Renumbered(2), AxisType::Unroll);
    let loop_range = UOp::range_axis(UOp::index_const(8), AxisId::Renumbered(3), AxisType::Reduce);
    let source = upcast.cast(DType::Float32);
    let reduce = source.reduce(smallvec![loop_range, upcast], ReduceOp::Add);
    let result = pre_expand(&reduce);
    assert_eq!(result.shape().unwrap().unwrap().as_slice(), &[SInt::Const(1)]);
    assert!(result.toposort().iter().any(|node| matches!(node.op(), Op::Reduce(ops::Reduce { num_axes: 1, .. }))));
}

#[test]
fn grouped_reduce_loop_keeps_nested_axis_identity_and_range_dependencies() {
    let ordering = UOp::range_axis(UOp::index_const(2), AxisId::Renumbered(3), AxisType::Loop);
    let ended = UOp::new(Op::Noop, DType::Void).end(smallvec![ordering]);
    let after = UOp::index_const(1).after(smallvec![ended]);
    let grouped_axis = AxisId::Renumbered(7).child(1).child(0);
    let grouped = UOp::new(
        Op::Range(ops::Range {
            end: UOp::index_const(4),
            axis_id: grouped_axis.clone(),
            axis_type: AxisType::GroupReduce,
            deps: smallvec![after.clone()],
        }),
        DType::WeakInt,
    );
    let reduce = grouped.cast(DType::Float32).reduce(smallvec![grouped], ReduceOp::Add);

    let lowered = graph_rewrite(pm_group_for_reduce(), reduce, &mut ());
    assert!(lowered.toposort().iter().any(|node| {
        matches!(node.op(), Op::Stage(ops::Stage { opts, .. }) if opts.local_axis.as_ref() == Some(&grouped_axis))
    }));
    let loop_range = lowered
        .toposort()
        .into_iter()
        .find(|node| {
            matches!(
                node.op(),
                Op::Range(ops::Range { axis_id, axis_type: AxisType::Reduce, .. })
                    if axis_id == &grouped_axis.group_reduce_loop()
            )
        })
        .expect("derived grouped-reduce loop");
    let Op::Range(ops::Range { deps, .. }) = loop_range.op() else { unreachable!() };
    assert_eq!(deps.len(), 1);
    assert!(std::sync::Arc::ptr_eq(&deps[0], &after));
    assert_eq!(grouped_axis.group_reduce_loop().path(), &[7, 1, 0, 2]);
}

#[test]
fn grouped_reduce_loop_does_not_collide_with_offset_axis_or_range_map_parent() {
    let grouped_axis = AxisId::Renumbered(0);
    let grouped = UOp::range_axis(UOp::index_const(4), grouped_axis.clone(), AxisType::GroupReduce);
    let old_offset_collision = UOp::range_axis(UOp::index_const(4), AxisId::Renumbered(100), AxisType::Upcast);
    let split_outer = UOp::range_axis(UOp::index_const(4), grouped_axis.child(0), AxisType::Upcast);
    let split_inner = UOp::range_axis(UOp::index_const(4), grouped_axis.child(1), AxisType::Upcast);
    let derived = UOp::range_axis(UOp::index_const(4), grouped_axis.group_reduce_loop(), AxisType::Upcast);
    let map = build_range_map(&UOp::sink(vec![old_offset_collision, split_outer, split_inner, derived]));

    assert_eq!(map.len(), 4);
    assert!(map.contains_key(&AxisId::Renumbered(100)));
    assert!(map.contains_key(&grouped_axis.child(0)));
    assert!(map.contains_key(&grouped_axis.child(1)));
    assert!(map.contains_key(&grouped_axis.group_reduce_loop()));

    let reduce = grouped.cast(DType::Float32).reduce(smallvec![grouped], ReduceOp::Add);
    let lowered = graph_rewrite(pm_group_for_reduce(), reduce, &mut ());
    assert!(lowered.toposort().iter().any(|node| {
        matches!(node.op(), Op::Range(ops::Range { axis_id, axis_type: AxisType::Reduce, .. })
            if axis_id == &grouped_axis.group_reduce_loop())
    }));
    assert!(!lowered.toposort().iter().any(|node| {
        matches!(node.op(), Op::Range(ops::Range { axis_id: AxisId::Renumbered(100), axis_type: AxisType::Reduce, .. }))
    }));
}
