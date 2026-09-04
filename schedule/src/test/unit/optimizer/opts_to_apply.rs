//! Tests for the `KernelInfo.opts_to_apply` optimization-control mechanism
//! (the Svod port of tinygrad's `opts_to_apply`).

use smallvec::smallvec;
use svod_dtype::DType;
use svod_ir::{AxisType, ConstValue, KernelInfo, Op, Opt, UOp};

use crate::optimizer::config::{OptStrategy, OptimizerConfig};
use crate::optimizer::{Renderer, optimize_kernel_with_config};
use svod_ir::ops;

/// Build a hand-ranged `out[i] = in[i] + 1` SINK over PARAM buffers, marked
/// with the given `opts_to_apply`. Mirrors a `Tensor::custom_kernel` body.
fn hand_ranged_sink(n: i64, opts_to_apply: Option<Vec<Opt>>) -> std::sync::Arc<UOp> {
    let out_buf = UOp::param(0, n as usize, DType::Float32, None);
    let in_buf = UOp::param(1, n as usize, DType::Float32, None);
    let i = UOp::range_const(n, 0);
    let in_idx = UOp::index().buffer(in_buf.clone()).indices(vec![i.clone()]).call().unwrap();
    let loaded = UOp::load().index(in_idx).call();
    let one = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let val = loaded.try_add(&one).unwrap();
    let out_idx = UOp::index().buffer(out_buf).indices(vec![i.clone()]).call().unwrap();
    let store = out_idx.store(val).end(smallvec![i]);
    UOp::sink_with_info(vec![store], KernelInfo { opts_to_apply, ..Default::default() })
}

fn count_axis_type(ast: &std::sync::Arc<UOp>, axis_type: AxisType) -> usize {
    ast.toposort()
        .iter()
        .filter(|u| matches!(u.op(), Op::Range(ops::Range { axis_type: at, .. }) if *at == axis_type))
        .count()
}

/// `opts_to_apply = Some(vec![])` (the tinygrad `()` analog): the optimizer must
/// apply ZERO opts — no heuristic default-upcast, and no beam search either — so the
/// manual Weak range survives and no Upcast axis is introduced.
#[test_case::test_case(OptStrategy::Heuristic; "over the heuristic path")]
#[test_case::test_case(OptStrategy::Beam { width: 1 }; "over the beam path")]
fn test_opts_to_apply_empty_applies_no_opts(strategy: OptStrategy) {
    let sink = hand_ranged_sink(8, Some(vec![]));
    let config = OptimizerConfig { strategy, ..Default::default() };
    let renderer = Renderer::cpu().with_rewrite_capabilities(svod_ir::RendererOps::all(), None, None);
    let optimized = optimize_kernel_with_config(sink, &renderer, &config).expect("optimize");

    assert_eq!(count_axis_type(&optimized, AxisType::Upcast), 0, "opts_to_apply=() must not introduce an Upcast axis");
    assert!(count_axis_type(&optimized, AxisType::Weak) >= 1, "the manual Weak range must survive untouched");
}

/// The `Op::Special` (gidx/lidx) hand-lowered bypass is gone: a SINK carrying
/// SPECIAL ops plus `opts_to_apply = Some(vec![])` runs the shared pipeline and
/// still applies zero schedule opts.
#[test]
fn test_opts_to_apply_empty_with_special_uses_the_shared_pipeline() {
    let out_buf = UOp::param(0, 8, DType::Float32, None);
    let in_buf = UOp::param(1, 8, DType::Float32, None);
    let gidx = UOp::special(UOp::index_const(8), "gidx0".into());
    let in_idx = UOp::index().buffer(in_buf).indices(vec![gidx.clone()]).call().unwrap();
    let value = UOp::load().index(in_idx).call().try_add(&UOp::const_(DType::Float32, ConstValue::Float(1.0))).unwrap();
    let out_idx = UOp::index().buffer(out_buf).indices(vec![gidx]).call().unwrap();
    let sink = UOp::sink_with_info(
        vec![out_idx.store(value)],
        KernelInfo { opts_to_apply: Some(vec![]), ..Default::default() },
    );

    let renderer = Renderer::cpu().with_rewrite_capabilities(svod_ir::RendererOps::all(), None, None);
    let config = OptimizerConfig { strategy: OptStrategy::Heuristic, ..Default::default() };
    let optimized = optimize_kernel_with_config(sink, &renderer, &config).expect("optimize");

    assert_eq!(count_axis_type(&optimized, AxisType::Upcast), 0);
    assert!(optimized.toposort().iter().any(|u| matches!(u.op(), Op::Special(..))), "{}", optimized.tree());
}
