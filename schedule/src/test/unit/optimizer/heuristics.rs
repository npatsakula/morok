use std::sync::Arc;

use morok_dtype::{DType, DeviceSpec, ImageKind};
use morok_ir::{AxisId, AxisType, ReduceOp, UOp};

use crate::optimizer::config::{HeuristicsConfig, TcOpt};
use crate::optimizer::heuristics::{apply_image_upcasts, apply_matvec_fast_path, try_tensor_cores};
use crate::optimizer::{OptOps, Renderer, Scheduler};

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

#[test]
fn test_apply_matvec_fast_path_applies_group_local_upcast() {
    let sink = create_matvec_like_pattern(64, 128);
    let mut scheduler = Scheduler::new(sink, Renderer::cuda());

    let config = HeuristicsConfig::default();
    let applied = apply_matvec_fast_path(&mut scheduler, &config);
    assert!(applied, "matvec fast-path should apply on matching pattern");

    assert!(!scheduler.axes_of(&[AxisType::GroupReduce]).is_empty(), "GROUP should be applied");
    assert!(!scheduler.axes_of(&[AxisType::Local]).is_empty(), "LOCAL should be applied");
    assert!(!scheduler.axes_of(&[AxisType::Upcast]).is_empty(), "UPCAST should be applied");
}

#[test]
fn test_apply_matvec_fast_path_respects_disable_flag() {
    let sink = create_matvec_like_pattern(64, 128);
    let mut scheduler = Scheduler::new(sink, Renderer::cuda());

    let config = HeuristicsConfig::builder().matvec_enabled(false).build();
    let applied = apply_matvec_fast_path(&mut scheduler, &config);
    assert!(!applied, "matvec fast-path should be disabled by config");
}

#[test]
fn test_apply_image_upcasts_non_stub_behavior() {
    let g = UOp::range_axis(UOp::index_const(8), AxisId::Renumbered(0), AxisType::Global);
    let img = UOp::new_buffer(DeviceSpec::Cpu, 64, DType::Image { kind: ImageKind::Float, shape: vec![8, 8] });
    let indexed = UOp::index().buffer(img).indices(vec![g.clone()]).call().expect("image index should build");
    let sink = UOp::sink(vec![indexed, g]);

    let mut scheduler = Scheduler::new(sink, Renderer::cpu());
    let applied = apply_image_upcasts(&mut scheduler);
    assert!(applied, "image upcast should apply for axis divisible by 4");
    assert_eq!(scheduler.axes_of(&[AxisType::Upcast]).len(), 1);
}

#[test]
fn test_try_tensor_cores_retries_axis_choices() {
    let sink = create_tc_retry_pattern();
    let mut scheduler = Scheduler::new(sink, Renderer::apple_amx());

    let config = HeuristicsConfig::builder().tc_opt(TcOpt::Relaxed).build();
    let applied = try_tensor_cores(&mut scheduler, &config);
    assert!(applied, "try_tensor_cores should recover with a later axis choice");

    let tc_opt = scheduler.applied_opts.iter().find(|opt| opt.op == OptOps::TC).expect("TC opt should be recorded");
    assert_eq!(tc_opt.axis, Some(1), "retry should commit the passing axis choice");
}
