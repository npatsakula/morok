use std::sync::Arc;

use svod_dtype::{AddrSpace, DType};
use svod_ir::{AxisId, AxisType, BinaryOp, ConstValue, Op, ParamArg, UOp};

use test_case::test_case;

use crate::rangeify::{SplitRangesContext, pm_flatten_range, pm_split_ranges};
use crate::rewrite::graph_rewrite;

fn split_modulo_range(axis_type: AxisType) -> (Arc<UOp>, Arc<UOp>, Arc<UOp>) {
    let range = UOp::range_axis(UOp::index_const(8), AxisId::Renumbered(0), axis_type);
    let sink = UOp::sink(vec![range.mod_(&UOp::index_const(2)).end(smallvec::smallvec![range.clone()])]);
    let result = graph_rewrite(&pm_split_ranges(), sink.clone(), &mut SplitRangesContext::default());
    (range, sink, result)
}

/// `r % k` splits `r` into an outer and an inner range of the same axis type.
/// WARP and DEVICE are launch lanes with fixed extents and are never split.
#[test_case(AxisType::Global, true ; "global")]
#[test_case(AxisType::Local, true ; "local")]
#[test_case(AxisType::Weak, true ; "weak")]
#[test_case(AxisType::Loop, true ; "loop axis")]
#[test_case(AxisType::Reduce, true ; "reduce")]
#[test_case(AxisType::GroupReduce, true ; "group reduce")]
#[test_case(AxisType::Upcast, true ; "upcast")]
#[test_case(AxisType::Warp, false ; "warp")]
#[test_case(AxisType::Device, false ; "device")]
fn modulo_splits_every_axis_type_but_the_launch_lanes(axis_type: AxisType, splits: bool) {
    let (original, sink, result) = split_modulo_range(axis_type);
    let ranges: Vec<_> = result.toposort().into_iter().filter(|uop| matches!(uop.op(), Op::Range { .. })).collect();

    if !splits {
        assert!(Arc::ptr_eq(&result, &sink));
        assert_eq!(ranges.len(), 1);
        assert!(Arc::ptr_eq(&ranges[0], &original));
        return;
    }

    assert_eq!(ranges.len(), 2);
    assert!(
        ranges.iter().all(|range| matches!(range.op(), Op::Range { axis_type: split, .. } if *split == axis_type)),
        "the split children inherit the axis type"
    );
    assert!(!ranges.iter().any(|range| Arc::ptr_eq(range, &original)));
}

#[test]
fn an_image_dtype_index_is_never_split() {
    let range = UOp::range_axis(UOp::index_const(8), AxisId::Renumbered(0), AxisType::Loop);
    let image = UOp::new(Op::Noop, DType::Image { kind: svod_dtype::ImageKind::Float, shape: vec![4, 2, 4] });
    let address = UOp::new(
        Op::Index { buffer: image, indices: smallvec::smallvec![range.clone()] },
        DType::Image { kind: svod_dtype::ImageKind::Float, shape: vec![4, 2, 4] },
    );
    let store = UOp::new(Op::Store { index: address, value: UOp::native_const(1.0f32), gate: None }, DType::Void);
    let sink = UOp::sink(vec![range.mod_(&UOp::index_const(2)).end(smallvec::smallvec![range.clone()]), store]);

    let result = graph_rewrite(&pm_split_ranges(), sink.clone(), &mut SplitRangesContext::default());
    assert!(Arc::ptr_eq(&result, &sink), "image coordinates must survive range splitting");
}

#[test]
fn split_simplifies_substituted_parent_in_preopt_composition() {
    let range = UOp::range_axis(UOp::index_const(8), AxisId::Renumbered(0), AxisType::Loop);
    let ended = range.mod_(&UOp::index_const(2)).end(smallvec::smallvec![range.clone()]);
    let sink = UOp::sink(vec![ended]);
    let matcher = pm_split_ranges() + pm_flatten_range().with_context::<SplitRangesContext>();

    let result = graph_rewrite(&matcher, sink, &mut SplitRangesContext::default());
    let Op::Sink { sources, .. } = result.op() else { panic!("expected SINK") };
    let Op::End { computation, .. } = sources[0].op() else { panic!("expected END") };

    assert!(
        matches!(computation.op(), Op::Range { end, .. }
            if matches!(end.op(), Op::Const(value) if value.0 == ConstValue::Int(2))),
        "the substituted (outer*2+inner)%2 parent must simplify to the inner range immediately: {}",
        result.tree()
    );
    assert_eq!(
        result.toposort().into_iter().filter(|uop| matches!(uop.op(), Op::Range { .. })).count(),
        2,
        "the END dependency must retain both split ranges"
    );
}

#[test]
fn negative_divisor_split_is_simplified_immediately() {
    let range = UOp::range_axis(UOp::index_const(8), AxisId::Renumbered(7), AxisType::Loop);
    let divisor = UOp::index_const(-2);
    let sink = UOp::sink(vec![range.mod_(&divisor).end(smallvec::smallvec![range.clone()])]);

    let result = graph_rewrite(&pm_split_ranges(), sink, &mut SplitRangesContext::default());
    let Op::Sink { sources, .. } = result.op() else { panic!("expected SINK") };
    let Op::End { computation, ranges } = sources[0].op() else { panic!("expected END") };
    assert!(matches!(computation.op(), Op::Const(value) if value.0 == ConstValue::Int(0)));
    let [split] = ranges.as_slice() else { panic!("expected one substituted END range") };
    assert!(matches!(split.op(), Op::Const(value) if value.0 == ConstValue::Int(0)));
}

#[test]
fn zero_and_nondivisible_moduli_do_not_split_ranges() {
    for divisor in [0, 3] {
        let range = UOp::range_axis(UOp::index_const(8), AxisId::Renumbered(0), AxisType::Loop);
        // Build the zero case directly: the checked arithmetic constructor
        // correctly rejects division by zero before the rewrite can inspect it.
        let modulo = UOp::new(Op::Binary(BinaryOp::FloorMod, range.clone(), UOp::index_const(divisor)), DType::Index);
        let sink = UOp::sink(vec![modulo]);
        let result = graph_rewrite(&pm_split_ranges(), sink.clone(), &mut SplitRangesContext::default());

        assert!(Arc::ptr_eq(&result, &sink), "divisor {divisor} must not split");
    }
}

#[test]
fn image_store_modulo_split_preserves_structural_coordinates() {
    let shape = svod_ir::shape::shape_to_uop(&smallvec::smallvec![2usize.into(), 1usize.into(), 4usize.into()]);
    let image = UOp::new(
        Op::Param { shape, arg: ParamArg::buffer(0, DType::Float32, AddrSpace::Global, None).into() },
        DType::Float32,
    );
    let range = UOp::range_axis(UOp::index_const(8), AxisId::Renumbered(0), AxisType::Loop);
    let four = UOp::index_const(4);
    let index = UOp::index()
        .buffer(image)
        .indices(vec![range.floor_div(&four), range.mod_(&four)])
        .call()
        .expect("structural image index");
    let sink = UOp::sink(vec![index.store(UOp::const_(DType::Float32, 1.0f32.into()))]);

    let result = graph_rewrite(&pm_split_ranges(), sink, &mut SplitRangesContext::default());
    let ranges: Vec<_> = result.toposort().into_iter().filter(|uop| matches!(uop.op(), Op::Range { .. })).collect();
    assert_eq!(ranges.len(), 2, "image stores must not suppress Tinygrad's modulo split: {}", result.tree());

    let outer = ranges
        .iter()
        .find(|range| matches!(range.op(), Op::Range { end, .. } if matches!(end.op(), Op::Const(value) if value.0 == ConstValue::Int(2))))
        .expect("outer range");
    let inner = ranges
        .iter()
        .find(|range| matches!(range.op(), Op::Range { end, .. } if matches!(end.op(), Op::Const(value) if value.0 == ConstValue::Int(4))))
        .expect("inner range");
    let store = result.toposort().into_iter().find(|uop| matches!(uop.op(), Op::Store { .. })).expect("image store");
    let Op::Store { index, .. } = store.op() else { unreachable!() };
    let Op::Index { indices, .. } = index.op() else { panic!("expected image INDEX") };
    assert_eq!(indices.len(), 2);
    assert!(Arc::ptr_eq(&indices[0], outer), "image y coordinate must be the split outer range");
    assert!(Arc::ptr_eq(&indices[1], inner), "image x coordinate must be the split inner range");
}

fn sorted_range_ids(root: &Arc<UOp>) -> Vec<AxisId> {
    let mut ids: Vec<_> = root
        .toposort()
        .into_iter()
        .filter_map(|u| match u.op() {
            Op::Range { axis_id, .. } => Some(axis_id.clone()),
            _ => None,
        })
        .collect();
    ids.sort();
    ids
}

#[test]
fn repeated_splits_append_children_to_each_original_axis() {
    let r0 = UOp::range_axis(UOp::index_const(12), AxisId::Renumbered(3), AxisType::Loop);
    let r1 = UOp::range_axis(UOp::index_const(10), AxisId::Renumbered(7), AxisType::Reduce);
    let sink = UOp::sink(vec![
        r0.mod_(&UOp::index_const(4)).end(smallvec::smallvec![r0]),
        r1.mod_(&UOp::index_const(5)).end(smallvec::smallvec![r1]),
    ]);

    let result = graph_rewrite(&pm_split_ranges(), sink, &mut SplitRangesContext::default());
    assert_eq!(
        sorted_range_ids(&result),
        vec![
            AxisId::Renumbered(3).child(0),
            AxisId::Renumbered(3).child(1),
            AxisId::Renumbered(7).child(0),
            AxisId::Renumbered(7).child(1),
        ]
    );
}

#[test]
fn nested_split_appends_to_the_existing_target_axis_path() {
    let parent = AxisId::Renumbered(5).child(1);
    let range = UOp::range_axis(UOp::index_const(12), parent.clone(), AxisType::Upcast);
    let sink = UOp::sink(vec![range.mod_(&UOp::index_const(3)).end(smallvec::smallvec![range])]);

    let result = graph_rewrite(&pm_split_ranges(), sink, &mut SplitRangesContext::default());
    assert_eq!(sorted_range_ids(&result), vec![parent.child(0), parent.child(1)]);
}
