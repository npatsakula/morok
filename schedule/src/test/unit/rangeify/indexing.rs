//! `IndexingContext`: range allocation, the realize map, and the helpers
//! `transform_single_source` uses to line consumer ranges up with a source.

use std::sync::Arc;

use smallvec::smallvec;
use svod_ir::{AxisId, AxisType, ConstValue, DType, Op, SInt, UOp};

use crate::rangeify::{
    IndexingContext,
    indexing::{broadcast_ranges, data_sources},
};

fn var() -> Arc<UOp> {
    UOp::var("x", DType::Float32, 0, i64::MAX)
}

/// Ranges are numbered sequentially as `AxisId::Unrenumbered`, keep their extent
/// (constant or symbolic), and a size-1 axis short-circuits to CONST 0 without
/// consuming an id.
#[test]
fn ranges_are_numbered_sequentially_and_size_one_axes_are_free() {
    let mut ctx = IndexingContext::new();
    assert_eq!(ctx.range_counter(), 0);

    for (i, extent) in [10i64, 20, 0, 1 << 30].into_iter().enumerate() {
        let range = ctx.new_range(&SInt::Const(extent as usize), AxisType::Loop);
        assert!(matches!(range.op(), Op::Range { axis_id, .. } if *axis_id == AxisId::Unrenumbered(i)));
        assert!(
            matches!(range.op(), Op::Range { end, .. } if matches!(end.op(), Op::Const(c) if c.0 == ConstValue::Int(extent)))
        );
        assert_eq!(ctx.range_counter(), i + 1);
    }

    let collapsed = ctx.new_range(&SInt::Const(1), AxisType::Loop);
    assert!(matches!(collapsed.op(), Op::Const(_)), "a singleton axis is index 0, not a loop");
    assert_eq!(ctx.range_counter(), 4, "and it consumes no axis id");

    let n = UOp::define_var("n".to_string(), 0, i64::MAX);
    let symbolic = ctx.new_range(&SInt::Symbolic(n.clone()), AxisType::Loop);
    assert!(matches!(symbolic.op(), Op::Range { end, .. } if Arc::ptr_eq(end, &n)));

    let reduce = ctx.new_range(&SInt::Const(10), AxisType::Reduce);
    assert!(matches!(reduce.op(), Op::Range { axis_type: AxisType::Reduce, .. }));
}

#[test]
fn separate_contexts_number_their_ranges_independently() {
    let (mut first, mut second) = (IndexingContext::new(), IndexingContext::new());
    first.new_range(&SInt::Const(10), AxisType::Loop);
    first.new_range(&SInt::Const(20), AxisType::Loop);

    let range = second.new_range(&SInt::Const(30), AxisType::Loop);
    assert!(matches!(range.op(), Op::Range { axis_id: AxisId::Unrenumbered(0), .. }));
}

#[test]
fn input_and_output_ranges_are_stored_and_read_back_per_uop() {
    let mut ctx = IndexingContext::new();
    let x = var();
    let r0 = ctx.new_range(&SInt::Const(10), AxisType::Loop);
    let r1 = ctx.new_range(&SInt::Const(20), AxisType::Loop);

    assert!(ctx.get_ranges(&x).is_none());
    ctx.set_ranges(&x, vec![r0.clone(), r1.clone()], vec![r0.clone()]);

    let (inputs, outputs) = ctx.get_ranges(&x).expect("ranges were set");
    assert!(inputs.iter().zip([&r0, &r1]).all(|(a, b)| Arc::ptr_eq(a, b)));
    assert!(outputs.len() == 1 && Arc::ptr_eq(&outputs[0], &r0));
}

/// `mark_realize_all` realizes every axis (no axis list); `mark_realize` records
/// exactly the axes given.
#[test]
fn the_realize_map_distinguishes_all_axes_from_named_axes() {
    let mut ctx = IndexingContext::new();
    let x = var();

    assert!(!ctx.should_realize(&x));
    assert!(ctx.get_realize_axes(&x).is_none());

    ctx.mark_realize_all(&x).expect("mark all");
    assert!(ctx.should_realize(&x));

    ctx.mark_realize(&x, vec![0, 2]);
    assert_eq!(ctx.get_realize_axes(&x).expect("axes"), &[0, 2]);
}

/// Index coordinates and AFTER ordering deps are not data — only the buffer is.
#[test]
fn data_sources_skips_index_coordinates_and_after_deps() {
    let buffer = UOp::new_buffer(svod_device::DeviceSpec::Cpu, 8, DType::Float32);
    let range = UOp::range_axis(UOp::index_const(8), AxisId::Renumbered(0), AxisType::Loop);
    let index = UOp::index().buffer(buffer.clone()).indices(vec![range]).call().expect("index");
    let after = buffer.after(smallvec![UOp::noop()]);

    for node in [index, after] {
        let sources = data_sources(&node);
        assert_eq!(sources.len(), 1);
        assert!(Arc::ptr_eq(&sources[0], &buffer));
    }
}

/// A rank-0 source keeps the consumer's range verbatim; an expanded singleton
/// axis is pinned to index 0 instead.
#[test]
fn broadcast_ranges_zeroes_only_the_expanded_axes() {
    let range = UOp::range_axis(UOp::index_const(4), AxisId::Renumbered(0), AxisType::Loop);

    let scalar = var();
    let consumer = scalar.try_add(&UOp::var("other", DType::Float32, 0, 4)).expect("add");
    let mapped = broadcast_ranges(&consumer, &scalar, std::slice::from_ref(&range));
    assert!(mapped.len() == 1 && Arc::ptr_eq(&mapped[0], &range));

    let source = UOp::const_(DType::Float32, 1.0f32.into()).try_reshape(&smallvec![SInt::Const(1)]).expect("reshape");
    let expanded = source.try_expand(&smallvec![SInt::Const(4)]).expect("expand");
    let consumer = expanded.try_add(&expanded).expect("add");
    let mapped = broadcast_ranges(&consumer, &source, &[range]);
    assert!(mapped.len() == 1 && matches!(mapped[0].op(), Op::Const(_)));
}

/// An image buffer addresses two coordinates; every other dtype linearises to
/// one. `transform_single_source` has to pick per dtype.
#[test]
fn image_buffers_keep_two_index_addresses() {
    let ranges = [
        UOp::range_axis(UOp::index_const(2), AxisId::Renumbered(0), AxisType::Loop),
        UOp::range_axis(UOp::index_const(8), AxisId::Renumbered(1), AxisType::Loop),
    ];
    let shape = svod_ir::shape::shape_to_uop(&smallvec![2usize.into(), 8usize.into()]);
    let image = DType::Image { kind: svod_dtype::ImageKind::Float, shape: vec![2, 8, 4] };

    for (dtype, expected_indices) in [(image, 2), (DType::Float32, 1)] {
        let arg =
            svod_ir::ParamArg::buffer(0, dtype.clone(), svod_dtype::AddrSpace::Global, Some(svod_ir::DeviceSpec::Cpu));
        let buffer = UOp::new(Op::Buffer { shape: shape.clone(), arg: arg.into() }, dtype);
        let indexed = crate::rangeify::transforms::transform_single_source(
            &UOp::sink(vec![]),
            &buffer,
            &ranges,
            &mut IndexingContext::new(),
        );
        assert!(matches!(indexed.op(), Op::Index { indices, .. } if indices.len() == expected_indices));
    }
}

/// `apply_movement_op` and `_apply_reshape` are `@functools.cache` upstream
/// (tinygrad/schedule/indexing.py:158,171): process-global and keyed on the inputs, so
/// a second call with the same op, input shape and range tuple never rebuilds the
/// index chain — it hands back the very nodes the first call produced.
#[test]
fn equal_movement_inputs_reuse_the_cached_index_chain() {
    // Prime extents so no other test shares these inputs in the process-global cache.
    let rngs = vec![UOp::range_const(13, 0), UOp::range_const(11, 1)];
    let in_shape = [SInt::Const(11), SInt::Const(13)];
    let out_shape = svod_ir::shape::shape_to_uop(&smallvec![SInt::Const(13), SInt::Const(11)]);
    let reshape = UOp::new(Op::Reshape { src: UOp::index_const(0), new_shape: out_shape }, DType::Float32);
    let holds = || crate::rangeify::indexing::movement_cache_holds(reshape.op(), &in_shape, &rngs);

    assert!(!holds(), "these inputs must be new");
    let first = crate::rangeify::apply_movement_op(reshape.op(), &in_shape, &rngs);
    assert!(holds(), "the first call memoises the inputs");
    let second = crate::rangeify::apply_movement_op(reshape.op(), &in_shape, &rngs);

    assert_eq!(first.len(), in_shape.len());
    assert!(first.iter().zip(&second).all(|(a, b)| Arc::ptr_eq(a, b)), "a hit returns the cached nodes");
}
