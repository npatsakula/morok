//! Movement ops (RESHAPE, PERMUTE, EXPAND, PAD, SHRINK, FLIP) folded into the
//! INDEX that reads them.

use std::sync::Arc;

use smallvec::smallvec;
use svod_device::DeviceSpec;
use svod_dtype::DType;
use svod_ir::{AxisId, AxisType, Op, SInt, UOp};
use test_case::test_case;

use crate::rangeify::patterns::movement_op_patterns;
use svod_ir::ops;

/// A movement chain and the ranges an INDEX reads it with.
type Access = (Arc<UOp>, Vec<Arc<UOp>>);
use crate::rewrite::{graph_rewrite, graph_rewrite_bottom_up};

fn buffer(size: usize) -> Arc<UOp> {
    UOp::new_buffer(DeviceSpec::Cpu, size, DType::Float32)
}

fn range(size: i64, axis_id: usize) -> Arc<UOp> {
    UOp::range_axis(UOp::index_const(size), AxisId::Renumbered(axis_id), AxisType::Loop)
}

fn reshaped(buffer: Arc<UOp>, dims: &[i64]) -> Arc<UOp> {
    buffer.try_reshape(&dims.iter().map(|&d| SInt::Const(d as usize)).collect()).expect("reshape")
}

/// `[200] -> [10, 20]`, read with one range per dim.
fn reshape() -> Access {
    (reshaped(buffer(200), &[10, 20]), vec![range(10, 0), range(20, 1)])
}

/// `[10, 1, 20] -> expand -> [10, 5, 20]`; the broadcast index becomes 0.
fn expand() -> Access {
    let src = reshaped(buffer(200), &[10, 1, 20]);
    let new_shape = UOp::stack(smallvec![UOp::index_const(10), UOp::index_const(5), UOp::index_const(20)]);
    (
        UOp::new(Op::Expand(ops::Expand { src, new_shape }), DType::Float32),
        vec![range(10, 0), range(5, 1), range(20, 2)],
    )
}

/// `[10, 20, 30]` permuted to `[20, 30, 10]`; the indices are reordered.
fn permute() -> Access {
    let src = reshaped(buffer(6000), &[10, 20, 30]);
    (src.try_permute(vec![1, 2, 0]).expect("permute"), vec![range(20, 0), range(30, 1), range(10, 2)])
}

/// `[10, 40]` shrunk to `[0:5, 10:30]`; the indices are offset.
fn shrink() -> Access {
    let src = reshaped(buffer(400), &[10, 40]);
    let offsets = UOp::stack(smallvec![UOp::index_const(0), UOp::index_const(10)]);
    let sizes = UOp::stack(smallvec![UOp::index_const(5), UOp::index_const(20)]);
    (UOp::new(Op::Shrink(ops::Shrink { src, offsets, sizes }), DType::Float32), vec![range(5, 0), range(20, 1)])
}

/// `[10, 20]` with the second axis reversed; index 1 becomes `19 - r1`.
fn flip() -> Access {
    let src = reshaped(buffer(200), &[10, 20]);
    (UOp::new(Op::Flip(ops::Flip { src, axes: vec![false, true] }), DType::Float32), vec![range(10, 0), range(20, 1)])
}

/// `[10, 20]` padded by `(1,1)` and `(2,2)`; the indices are offset and gated.
fn pad() -> Access {
    let src = reshaped(buffer(200), &[10, 20]);
    let begin_pads = UOp::stack(smallvec![UOp::index_const(1), UOp::index_const(2)]);
    let end_pads = UOp::stack(smallvec![UOp::index_const(1), UOp::index_const(2)]);
    (UOp::new(Op::Pad(ops::Pad { src, begin_pads, end_pads }), DType::Float32), vec![range(12, 0), range(24, 1)])
}

/// `RESHAPE(EXPAND(RESHAPE(buffer)))` read with one flat range — the rewrite has
/// to reach a fixed point across all three.
fn nested() -> Access {
    let src = reshaped(buffer(10), &[10, 1]);
    let new_shape = UOp::stack(smallvec![UOp::index_const(10), UOp::index_const(5)]);
    let expanded = UOp::new(Op::Expand(ops::Expand { src, new_shape }), DType::Float32);
    (expanded.try_reshape(&smallvec![SInt::Const(50)]).expect("reshape"), vec![range(50, 0)])
}

/// Every movement chain collapses to a single flat index straight off the
/// original BUFFER, whatever gating or offsetting the op contributes.
#[test_case(super::reshape ; "reshape")]
#[test_case(super::expand ; "expand")]
#[test_case(super::permute ; "permute")]
#[test_case(super::shrink ; "shrink")]
#[test_case(super::flip ; "flip")]
#[test_case(super::pad ; "pad")]
#[test_case(super::nested ; "reshape of expand of reshape")]
fn movement_chains_flatten_into_the_buffer_index(build: fn() -> Access) {
    let (movement, ranges) = build();
    let indexed = UOp::index().buffer(movement).indices(ranges).call().expect("index");

    let result = graph_rewrite(&movement_op_patterns(), indexed, &mut ());

    assert_eq!(result.dtype(), DType::Float32);
    let Op::Index(ops::Index { buffer, indices, .. }) = result.op() else {
        panic!("expected INDEX, got {}", result.tree())
    };
    assert_eq!(indices.len(), 1, "movement ops flatten to one index: {}", result.tree());
    assert!(matches!(buffer.op(), Op::Buffer(..)), "no movement op may survive: {}", result.tree());
}

#[test]
fn a_non_movement_source_is_left_under_the_index() {
    let sqrt = buffer(100).try_sqrt().expect("sqrt");
    let indexed = UOp::index().buffer(Arc::clone(&sqrt)).indices(vec![range(100, 0)]).call().expect("index");

    let result = graph_rewrite(&movement_op_patterns(), indexed, &mut ());

    let Op::Index(ops::Index { buffer, .. }) = result.op() else { panic!("expected INDEX") };
    assert!(Arc::ptr_eq(buffer, &sqrt));
}

/// A movement op with no INDEX/AFTER/END consumer has nothing to fold into.
#[test]
fn a_bare_movement_chain_is_untouched() {
    let expanded = reshaped(buffer(10), &[10, 1])
        .try_expand(&smallvec![SInt::Const(10), SInt::Const(4)])
        .expect("expand")
        .try_permute(vec![1, 0])
        .expect("permute");

    assert!(Arc::ptr_eq(&graph_rewrite_bottom_up(&movement_op_patterns(), expanded.clone(), &mut ()), &expanded));
}

/// A partial index — fewer indices than dims — only folds when the movement is a
/// RESHAPE whose trailing dims line up; PERMUTE and EXPAND never do.
#[test]
fn a_partial_index_is_left_alone() {
    let mismatched_reshape = reshaped(reshaped(buffer(12), &[2, 6]), &[2, 3, 2]);
    let permuted = reshaped(buffer(6), &[2, 3]).try_permute(vec![1, 0]).expect("permute");
    let expanded = reshaped(buffer(2), &[2, 1]).try_expand(&smallvec![SInt::Const(2), SInt::Const(3)]).expect("expand");

    for movement in [mismatched_reshape, permuted, expanded] {
        let indexed = UOp::index().buffer(Arc::clone(&movement)).indices(vec![range(2, 0)]).call().expect("index");
        let result = graph_rewrite_bottom_up(&movement_op_patterns(), indexed.clone(), &mut ());

        assert!(Arc::ptr_eq(&result, &indexed), "{}", result.tree());
    }
}

// ===== AFTER boundaries =====

/// AFTER is an ordering edge, not data: INDEX pushes through it and keeps both
/// the passthrough buffer and the dep.
#[test]
fn index_pushes_through_an_after_without_losing_its_dep() {
    let buffer = buffer(8);
    let range = range(8, 0);
    let indexed = UOp::index().buffer(Arc::clone(&buffer)).indices(vec![Arc::clone(&range)]).call().expect("index");
    let dep = UOp::noop();

    let result = graph_rewrite(&movement_op_patterns(), indexed.after(smallvec![Arc::clone(&dep)]), &mut ());

    let Op::Index(ops::Index { buffer: result_buffer, indices, .. }) = result.op() else { panic!("expected INDEX") };
    let Op::After(ops::After { passthrough, deps }) = result_buffer.op() else { panic!("expected INDEX(AFTER(..))") };
    assert!(Arc::ptr_eq(passthrough, &buffer));
    assert_eq!(deps.as_slice().len(), 1);
    assert!(Arc::ptr_eq(&deps[0], &dep));
    assert!(Arc::ptr_eq(&indices[0], &range));
}

/// Moving a movement op outside an AFTER leaves the tag on the AFTER; the
/// rebuilt movement node is fresh and untagged.
#[test]
fn movement_through_after_keeps_the_tag_on_the_after() {
    let buffer = buffer(20);
    let store = buffer.store(UOp::native_const(1.0f32));
    let after = reshaped(Arc::clone(&buffer), &[4, 5]).after(smallvec![store]).rtag(Some(smallvec![7]));

    let result = graph_rewrite(&movement_op_patterns(), after, &mut ());

    let Op::Reshape(ops::Reshape { src: inner, .. }) = result.op() else {
        panic!("expected RESHAPE outside, got {}", result.tree())
    };
    assert!(matches!(inner.op(), Op::After(..)));
    assert_eq!(inner.tag().as_deref(), Some([7usize].as_slice()));
    assert!(result.tag().is_none());
}
