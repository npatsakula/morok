//! Target shaped INDEX and movement cleanup tests for `devectorizer2`.

use smallvec::smallvec;
use svod_dtype::DType;
use svod_ir::{ConstValue, Op, UOp};

use super::helpers::create_buffer;
use crate::devectorize::devectorize_patterns;
use crate::rewrite::graph_rewrite;
use svod_ir::ops;

fn rewrite(uop: std::sync::Arc<UOp>) -> std::sync::Arc<UOp> {
    graph_rewrite(devectorize_patterns(), uop, &mut ())
}

#[test]
fn stacked_index_becomes_stack_of_scalar_indices() {
    let buffer = super::helpers::buffer_to_define(&create_buffer(8));
    let indices = UOp::stack(smallvec![UOp::index_const(1), UOp::index_const(3)]);
    let indexed = UOp::index().buffer(buffer).indices(vec![indices]).call().unwrap();

    let result = rewrite(indexed);
    let Op::Stack(ops::Stack { sources }) = result.op() else { panic!("expected STACK: {}", result.tree()) };
    assert_eq!(sources.len(), 2);
    assert!(sources.iter().all(|source| source.dtype() == DType::Float32));
    assert!(
        sources.iter().all(|source| matches!(source.op(), Op::Index(ops::Index { indices, .. }) if indices.len() == 1))
    );
}

#[test]
fn reshaped_index_is_fully_consumed_by_shaped_indexing() {
    let buffer = super::helpers::buffer_to_define(&create_buffer(8));
    let indices = UOp::stack(smallvec![UOp::index_const(0), UOp::index_const(1)]);
    let reshaped = indices.try_reshape(&smallvec![1usize.into(), 2usize.into()]).unwrap();
    let shaped = UOp::index().buffer(buffer).indices(vec![reshaped]).call().unwrap();
    let scalar = UOp::index().buffer(shaped).indices(vec![UOp::index_const(0), UOp::index_const(1)]).call().unwrap();

    let result = rewrite(scalar);
    assert!(matches!(result.op(), Op::Index(ops::Index { indices, .. }) if indices.len() == 1));
    assert!(!result.toposort().iter().any(|node| node.op().is_movement()));
}

#[test]
fn scalar_expand_becomes_stack() {
    let scalar = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let expanded = UOp::new(
        Op::Expand(ops::Expand {
            src: scalar,
            new_shape: svod_ir::shape::shape_to_uop(&smallvec::smallvec![4usize.into()]),
        }),
        DType::Float32,
    );
    let result = rewrite(expanded);

    assert!(matches!(result.op(), Op::Stack(ops::Stack { sources }) if sources.len() == 4));
    assert_eq!(result.dtype(), DType::Float32);
}

#[test]
fn singleton_reshape_to_scalar_becomes_index() {
    let shaped = UOp::stack(smallvec![UOp::const_(DType::Float32, ConstValue::Float(2.0))]);
    let scalar = shaped.try_reshape(&smallvec::smallvec![]).unwrap();
    let result = rewrite(scalar);

    assert!(matches!(result.op(), Op::Const(_)), "constant INDEX into STACK should clean up: {}", result.tree());
}

#[test]
fn void_reshape_is_removed() {
    let store = UOp::noop().with_dtype(DType::Void);
    let reshape = UOp::new(
        Op::Reshape(ops::Reshape { src: store.clone(), new_shape: svod_ir::shape::shape_to_uop(&smallvec![]) }),
        DType::Void,
    );
    assert!(std::sync::Arc::ptr_eq(&rewrite(reshape), &store));
}

#[test]
fn scalar_index_into_stack_folds_without_reconstructing_storage() {
    let values = UOp::stack(smallvec![
        UOp::const_(DType::Float32, ConstValue::Float(1.0)),
        UOp::const_(DType::Float32, ConstValue::Float(2.0)),
    ]);
    let rebuilt = UOp::stack(smallvec![
        UOp::new(
            Op::Index(ops::Index { buffer: values.clone(), indices: smallvec![UOp::index_const(0)] }),
            DType::Float32
        ),
        UOp::new(
            Op::Index(ops::Index { buffer: values.clone(), indices: smallvec![UOp::index_const(1)] }),
            DType::Float32
        ),
    ]);
    let rebuilt = rewrite(rebuilt);
    assert!(matches!(rebuilt.op(), Op::Stack(..)));

    let selected =
        UOp::new(Op::Index(ops::Index { buffer: values, indices: smallvec![UOp::index_const(1)] }), DType::Float32);
    assert!(matches!(rewrite(selected).op(), Op::Const(value) if value.0 == ConstValue::Float(2.0)));
}

#[test]
fn adjacent_movement_ops_are_cleaned_in_upstream_order() {
    let values = UOp::stack(smallvec![
        UOp::const_(DType::Float32, ConstValue::Float(1.0)),
        UOp::const_(DType::Float32, ConstValue::Float(2.0)),
    ]);
    let nested = UOp::new(
        Op::Reshape(ops::Reshape {
            src: UOp::new(
                Op::Reshape(ops::Reshape {
                    src: values.clone(),
                    new_shape: svod_ir::shape::shape_to_uop(&smallvec![1usize.into(), 2usize.into()]),
                }),
                DType::Float32,
            ),
            new_shape: svod_ir::shape::shape_to_uop(&smallvec![2usize.into()]),
        }),
        DType::Float32,
    );
    let reshaped = rewrite(nested);
    assert_eq!(reshaped.shape().unwrap().unwrap().as_slice(), &[2usize.into()]);
    assert!(!reshaped.toposort().iter().any(|node| node.op().is_movement()));

    let permuted = UOp::new(
        Op::Permute(ops::Permute {
            src: UOp::new(Op::Permute(ops::Permute { src: values.clone(), axes: vec![0] }), DType::Float32),
            axes: vec![0],
        }),
        DType::Float32,
    );
    let permuted = rewrite(permuted);
    assert_eq!(permuted.shape().unwrap().unwrap().as_slice(), &[2usize.into()]);
    assert!(!permuted.toposort().iter().any(|node| node.op().is_movement()));
}

#[test]
fn child_singleton_reshape_is_visible_to_parent_expand() {
    let values = UOp::stack(smallvec![
        UOp::const_(DType::Float32, ConstValue::Float(1.0)),
        UOp::const_(DType::Float32, ConstValue::Float(2.0)),
    ]);
    let singleton = values.try_reshape(&smallvec![1usize.into(), 2usize.into()]).unwrap();
    let expanded = singleton.try_expand(&smallvec![3usize.into(), 2usize.into()]).unwrap();

    let result = rewrite(expanded);
    let Op::Stack(ops::Stack { sources }) = result.op() else { panic!("expected outer STACK: {}", result.tree()) };
    assert_eq!(sources.len(), 3);
    assert!(sources.iter().all(|source| std::sync::Arc::ptr_eq(source, &values)));
    assert!(!result.toposort().iter().any(|node| matches!(node.op(), Op::Reshape(..) | Op::Expand(..))));
}

#[test]
fn child_stack_broadcast_is_visible_to_mixed_alu_parent() {
    let scalar = UOp::const_(DType::Float32, ConstValue::Float(2.0));
    let singleton = UOp::stack(smallvec![scalar]);
    let broadcast = singleton.try_expand(&smallvec![4usize.into()]).unwrap();
    let vector = UOp::vconst(vec![ConstValue::Float(1.0); 4], DType::Float32);
    let add = UOp::new(svod_ir::Op::Binary(svod_ir::BinaryOp::Add, broadcast, vector), DType::Float32.vec(4).unwrap());

    let result = rewrite(add);
    assert!(
        matches!(result.op(), Op::Stack(ops::Stack { sources }) if sources.len() == 4),
        "expected scalar STACK ALU: {}",
        result.tree()
    );
    assert!(!result.toposort().iter().any(|node| {
        matches!(node.op(), Op::Binary(..) | Op::Ternary(..))
            && node.op().sources().iter().any(|source| matches!(source.op(), Op::Stack(..)))
            && (node.dtype().vcount() > 1 || node.op().sources().iter().any(|source| source.dtype().vcount() > 1))
    }));
}
