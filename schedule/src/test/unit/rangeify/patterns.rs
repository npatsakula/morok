//! Rangeify pattern matchers: `early_rewrites`, `dead_axis_removal`, and the
//! movement-op removal folded into `apply_rangeify_patterns`.
//!
//! `buffer_folding` rows live in `buffer_folding.rs`.

use std::sync::Arc;

use smallvec::smallvec;
use svod_dtype::DType;
use svod_ir::{AxisId, AxisType, BinaryOp, BufferizeOpts, DeviceSpec, Op, ReduceOp, SInt, UOp};
use test_case::test_case;

use crate::pattern::RewriteResult;
use crate::rangeify::IndexingContext;
use crate::rangeify::patterns;
use crate::rewrite::graph_rewrite;
use svod_ir::ops;

fn rewritten(result: RewriteResult) -> Arc<UOp> {
    match result {
        RewriteResult::Rewritten(uop) => uop,
        other => panic!("expected Rewritten, got {other:?}"),
    }
}

// ===== early_rewrites =====

/// Autograd and same-device-copy markers are erased, returning their source
/// verbatim (tinygrad rangeify.py:153 for the COPY row).
#[test]
fn markers_are_replaced_by_their_source() {
    let x = UOp::native_const(42.0f32);
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let same_device_copy = buffer.copy(DeviceSpec::Cpu).rtag(Some(smallvec![3]));

    for (marked, source) in [
        (x.detach(), &x),
        (x.contiguous_backward(), &x),
        (x.detach().detach(), &x.detach()),
        (same_device_copy, &buffer),
    ] {
        assert!(Arc::ptr_eq(&rewritten(patterns::early_rewrites().rewrite(&marked, &mut ())), source));
    }
}

#[test]
fn early_rewrites_leaves_plain_compute_alone() {
    let a = UOp::native_const(1.0f32);
    for untouched in [a.clone(), a.try_add(&a).expect("add")] {
        assert!(matches!(patterns::early_rewrites().rewrite(&untouched, &mut ()), RewriteResult::NoMatch));
    }
}

/// A widening integer cast of a product moves onto the operands, so the
/// product is formed at the accumulator's width instead of wrapping.
#[test_case(DType::Int8, DType::Int32, true; "int8 product to int32")]
#[test_case(DType::UInt8, DType::UInt32, true; "uint8 product to uint32")]
#[test_case(DType::Int8, DType::UInt16, false; "sign change stays")]
#[test_case(DType::Float16, DType::Float32, false; "float product stays")]
#[test_case(DType::Int32, DType::Int8, false; "narrowing stays")]
fn widening_integer_cast_moves_onto_the_product_operands(stored: DType, wide: DType, rewritten_: bool) {
    let a = UOp::new_buffer(DeviceSpec::Cpu, 4, stored.clone());
    let b = UOp::new_buffer(DeviceSpec::Cpu, 4, stored);
    let cast = a.try_mul(&b).expect("mul").cast(wide.clone());
    match patterns::early_rewrites().rewrite(&cast, &mut ()) {
        RewriteResult::Rewritten(out) => {
            assert!(rewritten_, "unexpected rewrite of {}", out.op().as_ref());
            let Op::Binary(BinaryOp::Mul, x, y) = out.op() else { panic!("expected MUL, got {}", out.op().as_ref()) };
            assert!(matches!(x.op(), Op::Cast(..)) && matches!(y.op(), Op::Cast(..)));
            assert_eq!((out.dtype(), x.dtype(), y.dtype()), (wide.clone(), wide.clone(), wide));
        }
        RewriteResult::NoMatch => assert!(!rewritten_),
        other => panic!("unexpected {other:?}"),
    }
}

/// A reduction over an empty axis folds to its identity broadcast over the
/// surviving shape — not to a bare scalar.
#[test]
fn empty_reduction_folds_to_a_shaped_identity() {
    let source = UOp::new_buffer(DeviceSpec::Cpu, 0, DType::Float32)
        .try_reshape(&smallvec![SInt::Const(0), SInt::Const(3)])
        .expect("reshape");
    let reduce = source.try_reduce_axis(ReduceOp::Add, vec![0]).expect("reduce axis");

    let identity = rewritten(patterns::early_rewrites().rewrite(&reduce, &mut ()));
    assert_eq!(identity.shape().expect("shape").expect("static").as_slice(), &[SInt::Const(3)]);
    assert!(matches!(
        identity.op(),
        Op::Expand(ops::Expand { src, .. }) if matches!(src.op(), Op::Const(value) if value.0.try_float() == Some(0.0))
    ));
}

/// `[4] -> reshape -> expand([4,8]) -> to(Amd)`: without materialising the view
/// the transfer is sized by the `[4]` base and the destination under-allocated.
/// A pure reshape is a contiguous view of the same element count, so it passes.
#[test]
fn a_copy_source_is_materialised_only_when_the_view_resizes_it() {
    let source = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let amd = DeviceSpec::Amd { device_id: 0 };

    let expanded = source
        .try_reshape(&smallvec![SInt::Const(4), SInt::Const(1)])
        .expect("reshape")
        .try_expand(&smallvec![SInt::Const(4), SInt::Const(8)])
        .expect("expand");
    let rewritten = graph_rewrite(&patterns::early_rewrites(), expanded.copy_to_device(amd.clone()), &mut ());
    let Op::Copy(ops::Copy { src, .. }) = rewritten.op() else { panic!("expected COPY, got {}", rewritten.tree()) };
    assert!(matches!(src.op(), Op::Contiguous(..)), "resized copy source must be materialised");

    let flat = source.try_reshape(&smallvec![SInt::Const(2), SInt::Const(2)]).expect("reshape").copy_to_device(amd);
    assert!(Arc::ptr_eq(&graph_rewrite(&patterns::early_rewrites(), flat.clone(), &mut ()), &flat));
}

// ===== dead_axis_removal =====

/// A range the compute does not read is dead. The STAGE is kept (it still has to
/// become a STORE) but shrunk to zero ranges and re-broadcast through
/// RESHAPE/EXPAND — an identity EXPAND is elided at construction.
#[test_case(&[1] ; "one dead axis")]
#[test_case(&[10, 1] ; "live extent, still unread")]
#[test_case(&[10, 20] ; "two unread axes")]
fn unread_ranges_are_stripped_from_the_stage(extents: &[i64]) {
    let ranges = extents
        .iter()
        .enumerate()
        .map(|(i, &end)| UOp::range_axis(UOp::index_const(end), AxisId::Renumbered(i), AxisType::Loop))
        .collect();
    let stage = UOp::stage(UOp::native_const(1.0f32), ranges, BufferizeOpts::local());

    let result = rewritten(patterns::dead_axis_removal().rewrite(&stage, &mut ()));
    let reshape = match result.op() {
        Op::Expand(ops::Expand { src, .. }) => src,
        Op::Reshape(..) => &result,
        _ => panic!("expected EXPAND or RESHAPE, got {}", result.tree()),
    };
    let Op::Reshape(ops::Reshape { src: shrunk, .. }) = reshape.op() else {
        panic!("expected RESHAPE, got {}", result.tree())
    };
    assert!(
        matches!(shrunk.op(), Op::Stage(ops::Stage { ranges, .. }) if ranges.is_empty()),
        "the STAGE must survive with no ranges, got {}",
        result.tree()
    );
}

/// A COPY destination is sized by the transfer, so a dead axis must not shrink
/// it — the guard `remove_bufferize` also applies (tinygrad rangeify.py:198,227).
#[test]
fn always_run_sources_keep_their_dead_axes() {
    let source = UOp::native_const(1.0f32).copy(DeviceSpec::Cpu);
    let dead_range = UOp::range_axis(UOp::index_const(1), AxisId::Renumbered(0), AxisType::Loop);
    let stage = UOp::stage(source, vec![dead_range], BufferizeOpts::local());

    assert!(matches!(patterns::dead_axis_removal().rewrite(&stage, &mut ()), RewriteResult::NoMatch));
}

// ===== movement-op removal =====

fn permute(src: Arc<UOp>) -> Arc<UOp> {
    UOp::new(Op::Permute(ops::Permute { src, axes: vec![1, 0] }), DType::Float32)
}

fn reshape(src: Arc<UOp>) -> Arc<UOp> {
    let new_shape = UOp::stack(smallvec![UOp::index_const(4), UOp::index_const(8)]);
    UOp::new(Op::Reshape(ops::Reshape { src, new_shape }), DType::Float32)
}

fn expand(src: Arc<UOp>) -> Arc<UOp> {
    let new_shape = UOp::stack(smallvec![UOp::index_const(4), UOp::index_const(8)]);
    UOp::new(Op::Expand(ops::Expand { src, new_shape }), DType::Float32)
}

/// Once ranges are assigned the movement op has been absorbed into the index
/// expression and collapses to its source.
#[test_case(super::permute ; "permute")]
#[test_case(super::reshape ; "reshape")]
#[test_case(super::expand ; "expand")]
fn a_ranged_movement_op_collapses_to_its_source(build: fn(Arc<UOp>) -> Arc<UOp>) {
    let src = UOp::native_const(1.0f32);
    let movement = build(Arc::clone(&src));
    let range = UOp::range_axis(UOp::index_const(4), AxisId::Renumbered(0), AxisType::Loop);

    let mut ctx = IndexingContext::new();
    ctx.set_ranges(&movement, vec![range.clone()], vec![range]);

    assert!(Arc::ptr_eq(&rewritten(patterns::apply_rangeify_patterns().rewrite(&movement, &mut ctx)), &src));
}

/// Without ranges there is nothing to fold the movement into, and a non-movement
/// op never matches at all.
#[test]
fn nothing_is_removed_before_ranges_are_assigned() {
    let src = UOp::native_const(1.0f32);
    let mut ctx = IndexingContext::new();

    for uop in [permute(Arc::clone(&src)), src.try_sqrt().expect("sqrt")] {
        assert!(matches!(patterns::apply_rangeify_patterns().rewrite(&uop, &mut ctx), RewriteResult::NoMatch));
    }
}
