//! Direct tests for the boundary between pre-gater valid indices and post-gater memory ops.

use std::sync::Arc;

use svod_dtype::DType;
use svod_ir::{ConstValue, Op, UOp};

use super::helpers::{create_buffer, create_buffer_typed};
use crate::rewrite::graph_rewrite;
use svod_ir::ops;

fn apply_gater(root: &Arc<UOp>) -> Arc<UOp> {
    graph_rewrite(&crate::late::pm_move_gates_from_index(), root.clone(), &mut ())
}

fn valid_index(buffer: Arc<UOp>, index: Arc<UOp>, gate: Arc<UOp>) -> Arc<UOp> {
    UOp::index().buffer(buffer).indices(vec![index.valid(gate)]).call().unwrap()
}

#[test]
fn test_move_load_gate_from_index() {
    let buffer = create_buffer(16);
    let index = UOp::index_const(3);
    let gate = UOp::var("gate", DType::Bool, 0, 1);
    let load = UOp::load().index(valid_index(buffer, index.clone(), gate.clone())).call();

    let result = apply_gater(&load);
    let Op::Load(ops::Load { index: result_index, alt: Some(alt), gate: Some(result_gate) }) = result.op() else {
        panic!("valid-index LOAD must become a gated LOAD with an alt");
    };
    assert!(Arc::ptr_eq(result_gate, &gate));
    assert!(matches!(alt.op(), Op::Const(value) if value.0 == ConstValue::Float(0.0)));
    assert!(matches!(result_index.op(), Op::Index(ops::Index { indices, .. }) if Arc::ptr_eq(&indices[0], &index)));
}

#[test]
fn test_shaped_load_gate_uses_post_movement_stack_alt() {
    let buffer = create_buffer(16);
    let indices = UOp::stack((0..4).map(UOp::index_const).collect());
    let gate = UOp::stack((0..4).map(|_| UOp::const_(DType::Bool, ConstValue::Bool(true))).collect());
    let load = UOp::load().index(UOp::index().buffer(buffer).indices(vec![indices.valid(gate)]).call().unwrap()).call();

    let result = apply_gater(&load);
    let Op::Load(ops::Load { alt: Some(alt), .. }) = result.op() else { panic!("expected gated LOAD") };
    assert!(matches!(alt.op(), Op::Stack(ops::Stack { sources }) if sources.len() == 4));
    assert!(!alt.toposort().iter().any(|node| node.op().is_movement()));
}

#[test]
fn test_two_index_gate_needs_an_image_buffer() {
    let buffer = create_buffer_typed(16, svod_dtype::ScalarDType::Int32);
    let gate = UOp::var("gate", DType::Bool, 0, 1);
    let indices = vec![UOp::index_const(1).valid(gate.clone()), UOp::index_const(2).valid(gate)];
    let index = UOp::index().buffer(buffer).indices(indices).call().unwrap();
    let load = UOp::load().index(index).call();

    // The image rules hard-code the two-coordinate form; a plain Int32 two-index
    // access must fall through to the generic rule and keep its own dtype.
    let result = apply_gater(&load);
    let Op::Load(ops::Load { index, alt: Some(_), gate: Some(_) }) = result.op() else {
        panic!("expected a gated LOAD")
    };
    assert_eq!(index.dtype(), DType::Int32);
    let Op::Index(ops::Index { indices, .. }) = index.op() else { panic!("expected INDEX") };
    assert_eq!(indices.len(), 2);
    assert!(
        matches!(indices[1].op(), Op::Ternary(svod_ir::TernaryOp::Where, ..)),
        "the generic rule only lifts the first index's gate; the image rule would lift both"
    );
}

#[test]
fn test_move_store_gate_from_index() {
    let buffer = create_buffer(16);
    let index = UOp::index_const(3);
    let gate = UOp::var("gate", DType::Bool, 0, 1);
    let store = valid_index(buffer, index.clone(), gate.clone()).store(UOp::native_const(2.0f32));

    let result = apply_gater(&store);
    let Op::Store(ops::Store { index: result_index, gate: Some(result_gate), .. }) = result.op() else {
        panic!("valid-index STORE must become a gated STORE");
    };
    assert!(Arc::ptr_eq(result_gate, &gate));
    assert!(matches!(result_index.op(), Op::Index(ops::Index { indices, .. }) if Arc::ptr_eq(&indices[0], &index)));
}

#[test]
fn test_where_after_gated_load_becomes_load_alt() {
    let buffer = create_buffer(16);
    let index = UOp::index_const(3);
    let gate = UOp::var("gate", DType::Bool, 0, 1);
    let load = UOp::load().index(valid_index(buffer, index, gate.clone())).call();
    let alt = UOp::native_const(7.0f32);
    let where_ = UOp::try_where(gate.clone(), load, alt.clone()).unwrap();

    let result = apply_gater(&where_);
    let Op::Load(ops::Load { alt: Some(result_alt), gate: Some(result_gate), .. }) = result.op() else {
        panic!("WHERE around a matching gated LOAD must become its alt");
    };
    assert!(Arc::ptr_eq(result_gate, &gate));
    assert!(Arc::ptr_eq(result_alt, &alt));
}

/// tinygrad folds a fully-invalid memory access away: the LOAD becomes its alt (zero
/// when it has none) with the shape preserved, and the STORE becomes a noop.
#[test]
fn symbolic_simple_folds_fully_invalid_memory_accesses() {
    let buffer = create_buffer(16);
    let fold = |root| graph_rewrite(crate::symbolic::symbolic_simple(), root, &mut ());
    let invalid_index = |lanes: Vec<Arc<UOp>>| {
        let indices = if lanes.len() == 1 { lanes[0].clone() } else { UOp::stack(lanes.into()) };
        UOp::index().buffer(buffer.clone()).indices(vec![indices]).call().expect("legal before late lowering")
    };

    let scalar = invalid_index(vec![UOp::invalid_marker()]);
    let load = fold(UOp::load().index(scalar.clone()).call());
    assert!(matches!(load.op(), Op::Const(value) if value.0 == ConstValue::Float(0.0)));
    let store = fold(scalar.store(UOp::native_const(2.0f32)));
    assert!(matches!(store.op(), Op::Noop));
    assert!(![&load, &store].iter().any(|root| root.toposort().iter().any(UOp::is_invalid_marker)));

    let shaped = invalid_index(vec![UOp::invalid_marker(), UOp::invalid_marker()]);
    let load = fold(UOp::load().index(shaped.clone()).call());
    assert_eq!(load.dtype(), DType::Float32);
    assert_eq!(load.shape().unwrap().unwrap().as_slice(), &[svod_ir::SInt::Const(2)]);
    assert!(matches!(load.op(), Op::Stack(ops::Stack { sources })
        if sources.iter().all(|lane| matches!(lane.op(), Op::Const(value) if value.0 == ConstValue::Float(0.0)))));

    let alt = UOp::stack(vec![UOp::native_const(7.0f32), UOp::native_const(7.0f32)].into());
    let gated =
        UOp::load().index(shaped).alt(alt.clone()).gate(UOp::const_(DType::Bool, ConstValue::Bool(true))).call();
    assert!(Arc::ptr_eq(&fold(gated), &alt), "an invalid gated LOAD returns its existing alt");
}

#[test]
fn test_invalid_memory_fold_is_not_overbroad() {
    let buffer = create_buffer(16);
    let mixed_lanes = UOp::stack(vec![UOp::invalid_marker(), UOp::index_const(1)].into());
    // A mixed vector index only exists transiently during expansion, so construct
    // that IR directly rather than passing it through the public INDEX validator.
    let mixed_index = UOp::new(
        Op::Index(ops::Index { buffer: buffer.clone().broadcast(2), indices: vec![mixed_lanes].into() }),
        DType::Float32.vec(2).unwrap(),
    );
    let mixed_load = UOp::load().index(mixed_index).call();
    let result = graph_rewrite(crate::symbolic::symbolic_simple(), mixed_load, &mut ());
    assert!(matches!(result.op(), Op::Load(..)), "partially invalid vector indices must survive for lane lowering");

    let second_dimension_invalid =
        UOp::index().buffer(buffer.clone()).indices(vec![UOp::index_const(0), UOp::invalid_marker()]).call().unwrap();
    let load = UOp::load().index(second_dimension_invalid).call();
    let result = graph_rewrite(crate::symbolic::symbolic_simple(), load, &mut ());
    assert!(matches!(result.op(), Op::Load(..)), "only Tinygrad's first validity index is folded");

    let invalid_index = UOp::index().buffer(buffer).indices(vec![UOp::invalid_marker()]).call().unwrap();
    let store = invalid_index.store_gated(UOp::native_const(2.0f32), UOp::const_(DType::Bool, ConstValue::Bool(true)));
    let result = graph_rewrite(crate::symbolic::symbolic_simple(), store, &mut ());
    assert!(
        matches!(result.op(), Op::Store(ops::Store { gate: Some(_), .. })),
        "Tinygrad's invalid STORE fold excludes gated stores"
    );
}

#[test]
fn test_final_decomposition_removes_vectorized_data_invalid() {
    let gate = UOp::var("gate", DType::Bool, 0, 1);
    let invalid_data = UOp::try_where(gate, UOp::native_const(3.0f32), UOp::invalid_marker()).unwrap();
    let value = UOp::stack(
        vec![invalid_data, UOp::native_const(4.0f32), UOp::native_const(5.0f32), UOp::native_const(6.0f32)].into(),
    );
    let index = UOp::index().buffer(create_buffer(16)).indices(vec![UOp::index_const(0)]).call().unwrap();
    let store = UOp::new(Op::Store(ops::Store { index, value, gate: None }), DType::Void);

    let result = graph_rewrite(crate::optimizer::final_rewrite_patterns(), store, &mut ());

    assert!(
        !result.toposort().iter().any(UOp::is_invalid_marker),
        "final decomposition must replace data Invalid before rendering:\n{}",
        result.tree()
    );
}

#[test]
fn test_final_rewrite_does_not_reintroduce_invalid_memory_index() {
    let buffer = create_buffer(16);
    let clean_index = UOp::index_const(3);
    let gate = UOp::var("gate", DType::Bool, 0, 1);
    let load = UOp::load().index(valid_index(buffer, clean_index.clone(), gate.clone())).call();

    let gated = apply_gater(&load);
    assert!(!gated.toposort().iter().any(UOp::is_invalid_marker));

    let result = graph_rewrite(crate::optimizer::final_rewrite_patterns(), gated, &mut ());
    assert!(
        !result.toposort().iter().any(UOp::is_invalid_marker),
        "stage 20 must contain no Invalid marker:\n{}",
        result.tree()
    );
    let Op::Load(ops::Load { index, alt: Some(_), gate: Some(result_gate) }) = result.op() else {
        panic!("final rewrite must preserve the gated LOAD");
    };
    assert!(Arc::ptr_eq(result_gate, &gate));
    assert!(matches!(index.op(), Op::Index(ops::Index { indices, .. }) if Arc::ptr_eq(&indices[0], &clean_index)));
}
