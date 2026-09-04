//! Regressions and corner cases of the devectorize pass.

use std::sync::Arc;

use svod_dtype::{AddrSpace, DType, ScalarDType};

use svod_ir::{Op, UOp};

use super::helpers::*;
use svod_ir::ops;

/// Register reads sharing a range collapse into one END; an unrelated END over the
/// same range is left alone.
#[test]
fn register_reads_merge_only_their_shared_range_ends() {
    let range = UOp::range_const(4, 0);
    let make_register_end = |slot, value| {
        let buffer = UOp::buffer(slot, 1, DType::Int32, AddrSpace::Reg, None);
        let index = UOp::index().buffer(buffer.clone()).indices(vec![UOp::index_const(0)]).call().unwrap();
        let end = index.store(UOp::native_const(value)).end(smallvec::smallvec![range.clone()]);
        let load_index = UOp::index()
            .buffer(buffer.after(smallvec::smallvec![end.clone()]))
            .indices(vec![UOp::index_const(0)])
            .call()
            .unwrap();
        (UOp::load().index(load_index).call(), end)
    };
    let (left, left_end) = make_register_end(0, 1i32);
    let (right, right_end) = make_register_end(1, 2i32);

    let unrelated = create_index(create_buffer_typed(1, ScalarDType::Int32), 0)
        .store(UOp::native_const(3i32))
        .end(smallvec::smallvec![range.clone()]);
    let result = crate::devectorize::merge_register_read_ends(UOp::sink(vec![left, right, unrelated.clone()]));

    let matching: Vec<_> = result
        .toposort()
        .into_iter()
        .filter(
            |node| matches!(node.op(), Op::End(ops::End { ranges, .. }) if ranges.len() == 1 && Arc::ptr_eq(&ranges[0], &range)),
        )
        .collect();
    assert_eq!(matching.len(), 2);
    assert!(matching.iter().any(|node| Arc::ptr_eq(node, &unrelated)));
    assert!(matching.iter().any(|node| matches!(node.op(), Op::End(ops::End { computation, .. })
        if matches!(computation.op(), Op::Group(ops::Group { sources }) if sources.len() == 2))));
    assert!(!result.toposort().iter().any(|node| Arc::ptr_eq(node, &left_end) || Arc::ptr_eq(node, &right_end)));
}

/// `index_axes` records the selected positions verbatim, in order — the shaped
/// INDEX the devectorizer then splits lane by lane.
#[test]
fn shaped_index_keeps_its_selected_positions() {
    let indexed = create_vector_float_iota(8).index_axes(vec![1, 3, 5, 7]);

    let Op::Index(ops::Index { indices, .. }) = indexed.op() else { panic!("expected INDEX") };
    let Op::Stack(ops::Stack { sources }) = indices[0].op() else { panic!("expected a shaped index") };
    let positions: Vec<_> = sources
        .iter()
        .map(|source| match source.op() {
            Op::Const(value) => value.0.try_int().expect("integer position"),
            other => panic!("expected a constant position, got {other:?}"),
        })
        .collect();
    assert_eq!(positions, vec![1, 3, 5, 7]);
}

/// A zero-sized trailing dimension has no elements to chunk: reject instead of panicking.
#[test_case::test_case(0, &[4, 0], false; "trailing zero dim")]
#[test_case::test_case(4, &[2, 2], true; "square")]
#[test_case::test_case(4, &[4, 0], false; "zero dim with elements")]
#[test_case::test_case(3, &[2, 2], false; "not divisible")]
fn stack_with_shape_rejects_unchunkable_shapes(count: usize, dims: &[usize], expected: bool) {
    let elements: Vec<_> = (0..count).map(|i| UOp::native_const(i as i32)).collect();
    let shape: Vec<_> = dims.iter().map(|&d| svod_ir::SInt::Const(d)).collect();
    assert_eq!(crate::devectorize::stack_with_shape(elements, &shape).is_some(), expected);
}
