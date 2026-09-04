//! Kernel-shape pin for weak-dtype constant folding.
//!
//! resnet50 layer4 3x3 conv (`r_16_32_7_7_512_3_3`, the worst devectorize blow-up in
//! the model): input [1,512,7,7], weight [512,512,3,3], pad 1. Without the weak arm of
//! `fold_const_alu` every folded index constant survives devectorize.

use crate::optimizer::{Opt, Renderer, Scheduler, apply_opt, apply_pre_optimization};
use crate::rewrite::graph_rewrite;
use smallvec::smallvec;
use std::sync::Arc;
use svod_dtype::DType;
use svod_ir::{AxisId, AxisType, ReduceOp, UOp};

const C: i64 = 512;
const H: i64 = 7;
const K: i64 = 3;

/// Loop axes (co, oy, ox) then reduce axes (ci, ky, kx): full shape [512,7,7,512,3,3].
fn conv_kernel_sink() -> Arc<UOp> {
    let k = UOp::index_const;
    let reduce_axis =
        |end: i64, id: usize| UOp::range_axis(UOp::index_const(end), AxisId::Renumbered(id), AxisType::Reduce);
    let (co, oy, ox) = (UOp::range_const(C, 0), UOp::range_const(H, 1), UOp::range_const(H, 2));
    let (ci, ky, kx) = (reduce_axis(C, 3), reduce_axis(K, 4), reduce_axis(K, 5));

    let read = |slot: usize, len: i64, index: Arc<UOp>| {
        let buffer = UOp::param(slot, len as usize, DType::Float32, None);
        UOp::load().index(UOp::index().buffer(buffer).indices(vec![index]).call().unwrap()).call()
    };

    // pad 1: iy = oy + ky - 1, ix = ox + kx - 1, both gated to [0, H).
    let (iy, ix) = (oy.add(&ky).sub(&k(1)), ox.add(&kx).sub(&k(1)));
    let gate = iy.ge(&k(0)).and_(&iy.lt(&k(H))).and_(&ix.ge(&k(0))).and_(&ix.lt(&k(H)));
    let x = read(1, C * H * H, ci.mul(&k(H * H)).add(&iy.mul(&k(H))).add(&ix).valid(gate));
    let w = read(2, C * C * K * K, co.mul(&k(C * K * K)).add(&ci.mul(&k(K * K))).add(&ky.mul(&k(K))).add(&kx));

    let acc = x.try_mul(&w).unwrap().reduce(smallvec![ci, ky, kx], ReduceOp::Add);
    let out = UOp::param(0, (C * H * H) as usize, DType::Float32, None);
    let out_index = co.mul(&k(H * H)).add(&oy.mul(&k(H))).add(&ox);
    let store = UOp::index().buffer(out).indices(vec![out_index]).call().unwrap().store(acc);
    UOp::sink(vec![store.end(smallvec![co, oy, ox])])
}

/// `apply_post_optimization_configured_with_capture` stages 08 -> 14 (optimizer/mod.rs).
/// Stage 11 (local buffers) is inert here: the kernel holds no STAGE or movement op.
fn through_devectorize(ast: Arc<UOp>, renderer: &Renderer) -> Arc<UOp> {
    let ast = graph_rewrite(&*crate::optimizer::POST_OPT_SYM, ast, &mut ());
    let ast = crate::expand::pre_expand(&ast);
    let reduce = crate::devectorize::movement_cleanup_patterns().with_context::<crate::devectorize::ReduceContext>()
        + crate::devectorize::pm_reduce_local();
    let ast = graph_rewrite(&reduce, ast, &mut crate::devectorize::ReduceContext::default());
    let ast = graph_rewrite(&crate::gpudims::pm_lower_device_ranges(), ast, &mut ());
    let ast = graph_rewrite(
        &crate::gpudims::pm_add_gpudims(),
        ast,
        &mut crate::gpudims::GpuDimsContext::from(renderer.clone()),
    );
    let loads = crate::symbolic::patterns::symbolic_simple().clone()
        + crate::devectorize::pm_expand_broadcast().clone()
        + crate::devectorize::pm_add_loads().clone();
    crate::devectorize::devectorize(&graph_rewrite(&loads, ast, &mut ()), renderer)
}

#[test]
fn weak_folding_keeps_the_resnet_conv_devectorize_bounded() {
    let mut renderer = Renderer::cpu().with_rewrite_capabilities(svod_ir::RendererOps::all(), None, None);
    // Renderer::cpu() caps threads at the host core count; the pinned opts need 32.
    renderer.global_max = Some(vec![64]);

    let mut scheduler = Scheduler::new(apply_pre_optimization(conv_kernel_sink()).unwrap(), renderer.clone());
    scheduler.convert_loop_to_global().unwrap();
    assert_eq!(scheduler.full_shape(), vec![C, H, H, C, K, K]);

    for opt in [Opt::upcast(2, 7), Opt::upcast(1, 7), Opt::unroll(2, 0), Opt::unroll(1, 0), Opt::thread(0, 32)] {
        apply_opt(&mut scheduler, &opt, true).unwrap_or_else(|e| panic!("{opt:?} failed: {e}"));
    }
    assert_eq!(scheduler.full_shape(), vec![16, 32, 7, 7, 512, 3, 3]);

    // 6862 nodes before the weak arm of `fold_const_alu`, 4301 after (tinygrad: 2480).
    let devectorized = through_devectorize(scheduler.get_optimized_ast(None), &renderer);
    assert!(devectorized.node_count() < 5000, "devectorize blew up: {} nodes", devectorized.node_count());
}
