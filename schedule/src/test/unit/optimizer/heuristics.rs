use std::sync::Arc;

use morok_dtype::{DType, DeviceSpec, ImageKind};
use morok_ir::{AxisId, AxisType, ReduceOp, UOp};

use crate::optimizer::config::HeuristicsConfig;
use crate::optimizer::heuristics::{apply_image_upcasts, apply_matvec_fast_path};
use crate::optimizer::{Renderer, Scheduler};

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
