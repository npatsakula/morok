//! White-box tests over `crate::optimizer::implicit_barriers`: RAW/WAR barrier
//! inference and its interaction with renderer-supplied rewrite capabilities.

use crate::optimizer::early_decomposition_patterns;
use crate::optimizer::implicit_barriers::pm_implicit_barriers;
use smallvec::{SmallVec, smallvec};
use std::sync::Arc;
use svod_dtype::{AddrSpace, DType};
use svod_ir::ops;
use svod_ir::rewrite::graph_rewrite;
use svod_ir::{AxisId, AxisType, BinaryOp, ConstValue, Op, ParamArg, RendererOps, UOp};

fn buffer(slot: usize, addrspace: AddrSpace) -> Arc<UOp> {
    UOp::new(
        Op::Param(ops::Param {
            shape: svod_ir::shape::shape_to_uop(&smallvec::smallvec![8usize.into()]),
            arg: ParamArg::buffer(slot, DType::Float32, addrspace, None).into(),
        }),
        DType::Float32,
    )
}

fn index(buffer: Arc<UOp>, offset: Arc<UOp>) -> Arc<UOp> {
    UOp::index().buffer(buffer).indices(vec![offset]).call().unwrap()
}

fn rewrite(root: Arc<UOp>) -> Arc<UOp> {
    graph_rewrite(pm_implicit_barriers(), root, &mut ())
}

#[test]
fn renderer_extra_matcher_local_dependency_precedes_barrier_inference() {
    let extra = crate::patterns! {
        Noop() => || {
            let local = buffer(0, AddrSpace::Local);
            let store = index(local.clone(), UOp::index_const(0)).store_value(UOp::native_const(1.0f32));
            Some(local.after(smallvec![store]))
        },
    };
    let renderer = crate::optimizer::Renderer::cpu().with_rewrite_capabilities(RendererOps::all(), None, Some(extra));
    let rewritten = graph_rewrite(renderer.extra_matcher().unwrap(), UOp::new(Op::Noop, DType::Void), &mut ());
    let result = crate::optimizer::finish_final_rewrite(rewritten);

    assert!(
        matches!(result.op(), Op::After(ops::After { deps, .. })
        if matches!(deps.as_slice(), [barrier] if matches!(barrier.op(), Op::Barrier(..)))),
        "{}",
        result.tree()
    );
}

#[test]
fn renderer_supported_ops_control_decomposition() {
    let x = UOp::const_(DType::UInt64, ConstValue::UInt(1));
    let key = UOp::const_(DType::UInt64, ConstValue::UInt(2));
    let threefry = UOp::new(Op::Binary(BinaryOp::Threefry, x, key), DType::UInt64);

    let supported = RendererOps::all();
    let unchanged = graph_rewrite(&early_decomposition_patterns(&supported), threefry.clone(), &mut ());
    assert!(matches!(unchanged.op(), Op::Binary(BinaryOp::Threefry, ..)));

    let mut unsupported = RendererOps::all();
    unsupported.binary.remove(&BinaryOp::Threefry);
    let decomposed = graph_rewrite(&early_decomposition_patterns(&unsupported), threefry, &mut ());
    assert!(!decomposed.toposort().iter().any(|uop| matches!(uop.op(), Op::Binary(BinaryOp::Threefry, ..))));

    let erf = UOp::native_const(0.5f32).erf().unwrap();
    let native = graph_rewrite(&early_decomposition_patterns(&supported), erf.clone(), &mut ());
    assert!(matches!(native.op(), Op::Unary(svod_ir::UnaryOp::Erf, _)));

    unsupported.unary.remove(&svod_ir::UnaryOp::Erf);
    let decomposed = graph_rewrite(&early_decomposition_patterns(&unsupported), erf, &mut ());
    assert!(!decomposed.toposort().iter().any(|uop| matches!(uop.op(), Op::Unary(svod_ir::UnaryOp::Erf, _))));
}

#[test]
fn local_after_store_gets_raw_barrier() {
    let local = buffer(0, AddrSpace::Local);
    let store = index(local.clone(), UOp::index_const(0)).store_value(UOp::native_const(1.0f32));
    let result = rewrite(local.after(smallvec![store.clone()]));

    let Op::After(ops::After { passthrough, deps }) = result.op() else { panic!("expected AFTER") };
    assert!(Arc::ptr_eq(passthrough, &local));
    assert!(matches!(deps.as_slice(), [barrier]
        if matches!(barrier.op(), Op::Barrier(ops::Barrier { src, deps }) if Arc::ptr_eq(src, &store) && deps.is_empty())));
}

#[test]
fn global_after_store_does_not_get_barrier() {
    let global = buffer(0, AddrSpace::Global);
    let store = index(global.clone(), UOp::index_const(0)).store_value(UOp::native_const(1.0f32));
    let result = rewrite(global.after(smallvec![store.clone()]));

    assert!(matches!(result.op(), Op::After(ops::After { deps, .. })
        if matches!(deps.as_slice(), [dep] if Arc::ptr_eq(dep, &store))));
}

#[test]
fn local_store_and_load_get_war_barrier_for_all_loop_axes() {
    for (slot, axis_type) in [AxisType::Reduce, AxisType::Weak, AxisType::Loop].into_iter().enumerate() {
        let local = buffer(slot, AddrSpace::Local);
        let range = UOp::range_axis(UOp::index_const(4), AxisId::Renumbered(slot), axis_type);
        let load = UOp::load().index(index(local.clone(), range.clone())).call();
        let store = index(local, range.clone()).store_value(load.clone());
        let result = rewrite(store.end(smallvec![range.clone()]));

        let Op::End(ops::End { computation, ranges }) = result.op() else { panic!("expected END") };
        assert!(matches!(computation.op(), Op::Barrier(ops::Barrier { src, deps })
            if Arc::ptr_eq(src, &store) && matches!(deps.as_slice(), [dep] if Arc::ptr_eq(dep, &load))));
        assert!(matches!(ranges.as_slice(), [closed] if Arc::ptr_eq(closed, &range)));
    }
}

#[test]
fn end_computation_load_participates_in_war_detection() {
    let local = buffer(0, AddrSpace::Local);
    let range = UOp::range_axis(UOp::index_const(4), AxisId::Renumbered(0), AxisType::Weak);
    let store = index(local.clone(), range.clone()).store_value(UOp::native_const(1.0f32));
    let load = UOp::load().index(index(local.after(smallvec![store]), range.clone())).call();
    let result = rewrite(load.end(smallvec![range]));

    assert!(
        matches!(result.op(), Op::End(ops::End { computation, .. })
        if matches!(computation.op(), Op::Barrier(ops::Barrier { src, deps })
            if matches!(src.op(), Op::Load(..))
                && matches!(deps.as_slice(), [dep] if Arc::ptr_eq(dep, src)))),
        "{}",
        result.tree()
    );
}

/// A WAR barrier is only needed for a local buffer read and written across at least
/// two iterations of the same range.
#[test_case::test_case(AddrSpace::Local, 0; "range with no second iteration")]
#[test_case::test_case(AddrSpace::Global, 4; "global memory is not thread-shared")]
fn no_war_barrier_without_a_local_cross_iteration_hazard(addrspace: AddrSpace, extent: i64) {
    let memory = buffer(0, addrspace);
    let range = UOp::range_axis(UOp::index_const(extent), AxisId::Renumbered(0), AxisType::Weak);
    let load = UOp::load().index(index(memory.clone(), range.clone())).call();
    let store = index(memory, range.clone()).store_value(load);
    let result = rewrite(store.clone().end(smallvec![range]));

    assert!(matches!(result.op(), Op::End(ops::End { computation, .. }) if Arc::ptr_eq(computation, &store)));
}

#[test]
fn unrelated_global_load_does_not_match_local_store() {
    let local = buffer(0, AddrSpace::Local);
    let global = buffer(1, AddrSpace::Global);
    let range = UOp::range_axis(UOp::index_const(4), AxisId::Renumbered(0), AxisType::Weak);
    let store = index(local, range.clone()).store_value(UOp::native_const(1.0f32));
    let load = UOp::load().index(index(global, range.clone())).call();
    let computation = UOp::sink(vec![store, load]);
    let result = rewrite(computation.clone().end(smallvec![range]));

    assert!(
        matches!(result.op(), Op::End(ops::End { computation: rewritten, .. }) if Arc::ptr_eq(rewritten, &computation))
    );
}

#[test]
fn existing_barrier_is_not_reinferred() {
    let local = buffer(0, AddrSpace::Local);
    let store = index(local.clone(), UOp::index_const(0)).store_value(UOp::native_const(1.0f32));
    let explicit = store.barrier(SmallVec::new());
    let result = rewrite(local.after(smallvec![explicit.clone()]));

    assert!(matches!(result.op(), Op::After(ops::After { deps, .. })
        if matches!(deps.as_slice(), [dep] if Arc::ptr_eq(dep, &explicit))));
}
