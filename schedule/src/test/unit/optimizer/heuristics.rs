use std::sync::Arc;

use svod_dtype::{AddrSpace, DType, DeviceSpec};
use svod_ir::{AxisId, AxisType, Op, ParamArg, ReduceOp, UOp};
use test_case::test_case;

use crate::optimizer::config::{HeuristicsConfig, TcOpt};
use crate::optimizer::heuristics::{
    apply_default_upcast, apply_image_upcasts, apply_matvec_fast_path, try_tensor_cores,
};
use crate::optimizer::{OptOps, Renderer, Scheduler};
use svod_ir::ops;

fn create_matvec_like_pattern(rows: i64, cols: i64) -> Arc<UOp> {
    let row = UOp::range_axis(UOp::index_const(rows), AxisId::Renumbered(0), AxisType::Global);
    let reduce = UOp::range_axis(UOp::index_const(cols), AxisId::Renumbered(1), AxisType::Reduce);

    let a_buf = UOp::new_buffer(DeviceSpec::Cpu, (rows * cols) as usize, DType::Float32);
    let b_buf = UOp::new_buffer(DeviceSpec::Cpu, (rows * cols) as usize, DType::Float32);

    let idx_expr = row.try_add(&reduce).expect("index add should succeed");
    let a = UOp::index().buffer(a_buf).indices(vec![idx_expr.clone()]).call().expect("A index should build");
    let b = UOp::index().buffer(b_buf).indices(vec![idx_expr]).call().expect("B index should build");

    let mul = a.try_mul(&b).expect("mul should succeed");
    let red = mul.reduce(vec![reduce].into(), ReduceOp::Add);
    UOp::sink(vec![red, row])
}

fn create_tc_retry_pattern() -> Arc<UOp> {
    let m_range = UOp::range_axis(UOp::index_const(16), AxisId::Renumbered(0), AxisType::Global);
    let n_good_range = UOp::range_axis(UOp::index_const(16), AxisId::Renumbered(1), AxisType::Global);
    let k_range = UOp::range_axis(UOp::index_const(16), AxisId::Renumbered(2), AxisType::Reduce);
    let n_bad_range = UOp::range_axis(UOp::index_const(15), AxisId::Renumbered(3), AxisType::Global);

    let a_buf = UOp::new_buffer(DeviceSpec::Cpu, 4096, DType::Float32);
    let b_buf = UOp::new_buffer(DeviceSpec::Cpu, 4096, DType::Float32);

    let a_idx = m_range.try_add(&k_range).expect("A index should build");
    let b_idx = k_range.try_add(&n_bad_range).and_then(|x| x.try_add(&n_good_range)).expect("B index should build");

    let a_val = UOp::index().buffer(a_buf).indices(vec![a_idx]).call().expect("A load should build");
    let b_val = UOp::index().buffer(b_buf).indices(vec![b_idx]).call().expect("B load should build");

    let mul = a_val.try_mul(&b_val).expect("mul should succeed");
    let red = mul.reduce(vec![k_range].into(), ReduceOp::Add);
    UOp::sink(vec![red, m_range, n_good_range, n_bad_range])
}

/// The matvec fast path applies GROUP + LOCAL + UPCAST in one shot, unless
/// `matvec_enabled` turns it off.
#[test_case(true; "enabled")]
#[test_case(false; "disabled by config")]
fn test_apply_matvec_fast_path(enabled: bool) {
    let mut scheduler = Scheduler::new(create_matvec_like_pattern(64, 128), Renderer::cuda());
    let config = HeuristicsConfig::builder().matvec_enabled(enabled).build();

    assert_eq!(apply_matvec_fast_path(&mut scheduler, &config), enabled);
    for axis in [AxisType::GroupReduce, AxisType::Local, AxisType::Upcast] {
        assert_eq!(!scheduler.axes_of(&[axis]).is_empty(), enabled, "{axis:?}");
    }
}

#[test_case(DType::Image { kind: svod_dtype::ImageKind::Float, shape: vec![2, 8, 4] }, true; "image buffer")]
#[test_case(DType::Float32, false; "plain rank three tensor")]
fn test_apply_image_upcasts_non_stub_behavior(dtype: DType, expected: bool) {
    let g = UOp::range_axis(UOp::index_const(8), AxisId::Renumbered(0), AxisType::Global);
    let shape = svod_ir::shape::shape_to_uop(&smallvec::smallvec![2usize.into(), 8usize.into(), 4usize.into()]);
    let arg = ParamArg::buffer(0, dtype.clone(), AddrSpace::Global, Some(DeviceSpec::Cpu));
    let img = UOp::new(Op::Buffer(ops::Buffer { shape, arg: arg.into() }), dtype);
    let indexed = UOp::index().buffer(img).indices(vec![g.clone()]).call().expect("image index should build");
    let sink = UOp::sink(vec![indexed, g]);

    let mut scheduler = Scheduler::new(sink, Renderer::cpu());
    assert_eq!(apply_image_upcasts(&mut scheduler), expected);
    assert_eq!(scheduler.axes_of(&[AxisType::Upcast]).len(), usize::from(expected));
}

#[test]
fn test_try_tensor_cores_retries_axis_choices() {
    // Use a non-AMX renderer so a successful trial commits — on AMX
    // `try_tensor_cores` discards the TC'd copy (see
    // `test_try_tensor_cores_amx_discards_trial`).
    let sink = create_tc_retry_pattern();
    let mut scheduler = Scheduler::new(sink, Renderer::metal());

    let config = HeuristicsConfig::builder().tc_opt(TcOpt::Relaxed).build();
    let applied = try_tensor_cores(&mut scheduler, &config);
    assert!(applied, "try_tensor_cores should recover with a later axis choice");

    let tc_opt = scheduler.applied_opts.iter().find(|opt| opt.op == OptOps::TC).expect("TC opt should be recorded");
    assert_eq!(tc_opt.axis, Some(1), "retry should commit the passing axis choice");
}

#[test]
fn test_try_tensor_cores_amx_discards_trial() {
    // On AMX, even when TC would apply successfully, the trial copy is
    // discarded so the heuristic falls through to the regular
    // UPCAST/THREAD/LOCAL chain on the untouched scheduler.
    let sink = create_tc_retry_pattern();
    let mut scheduler = Scheduler::new(sink, Renderer::apple_amx());
    let snapshot_opts_before = scheduler.applied_opts.clone();

    let config = HeuristicsConfig::builder().tc_opt(TcOpt::Relaxed).build();
    let applied = try_tensor_cores(&mut scheduler, &config);
    assert!(!applied, "try_tensor_cores must return false on AMX (TC is intentionally discarded)");
    assert_eq!(
        scheduler.applied_opts, snapshot_opts_before,
        "AMX path must leave the scheduler's applied_opts untouched (no TC commit)"
    );
}

/// Elementwise SINK with one WEAK axis plus an optional extra axis of `extra`
/// type, so `apply_default_upcast`'s gate and axis pick can be exercised.
fn create_default_upcast_pattern(size: i64, extra: Option<(i64, AxisType)>) -> Arc<UOp> {
    let weak = UOp::range_axis(UOp::index_const(size), AxisId::Renumbered(0), AxisType::Weak);
    let buf = UOp::new_buffer(DeviceSpec::Cpu, size as usize * 64, DType::Float32);
    let (idx, mut sink_srcs) = match extra {
        Some((extra_size, axis_type)) => {
            let other = UOp::range_axis(UOp::index_const(extra_size), AxisId::Renumbered(1), axis_type);
            (weak.try_add(&other).expect("index add"), vec![weak.clone(), other])
        }
        None => (weak.clone(), vec![weak.clone()]),
    };
    let val = UOp::index().buffer(buf).indices(vec![idx]).call().expect("index should build");
    let doubled = val.try_add(&val).expect("add should succeed");
    sink_srcs.insert(0, doubled);
    UOp::sink(sink_srcs)
}

#[test_case(16, None, true; "divisible weak axis upcasts")]
#[test_case(6, None, false; "size not divisible by four")]
#[test_case(1, None, false; "size one axis is not upcastable")]
#[test_case(16, Some((4, AxisType::Unroll)), false; "unrolled kernel skips the fallback")]
#[test_case(16, Some((4, AxisType::Upcast)), false; "already upcast kernel skips the fallback")]
#[test_case(16, Some((8, AxisType::Reduce)), true; "reduce axis does not block the fallback")]
fn default_upcast_follows_tinygrad_gate(size: i64, extra: Option<(i64, AxisType)>, expected: bool) {
    let pre_existing = usize::from(matches!(extra, Some((_, AxisType::Upcast))));
    let mut scheduler = Scheduler::new(create_default_upcast_pattern(size, extra), Renderer::cpu());

    assert_eq!(apply_default_upcast(&mut scheduler), expected);
    assert_eq!(
        scheduler.axes_of(&[AxisType::Upcast]).len(),
        pre_existing + usize::from(expected),
        "UPCAST axis count after the fallback"
    );
}

#[test]
fn default_upcast_picks_the_innermost_upcastable_axis() {
    // Tinygrad takes `k.upcastable_dims[-1]`; both axes qualify here, and only
    // the trailing one must be split.
    let sink = create_default_upcast_pattern(16, Some((8, AxisType::Global)));
    let mut scheduler = Scheduler::new(sink, Renderer::cpu());
    let innermost = *scheduler.upcastable_dims().last().expect("two upcastable dims");

    assert!(apply_default_upcast(&mut scheduler));
    let opt = scheduler.applied_opts.iter().find(|opt| opt.op == OptOps::UPCAST).expect("UPCAST recorded");
    assert_eq!(opt.axis, Some(innermost));
}
