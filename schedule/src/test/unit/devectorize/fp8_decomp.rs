use std::sync::Arc;

use morok_dtype::{DType, ScalarDType};
use morok_ir::types::ConstValue;
use morok_ir::{Op, UOp};
use smallvec::smallvec;

use super::helpers::{apply_pm_render, create_bool_const, create_buffer_typed};

#[test]
fn test_fp8_decomp_preserves_alt_on_gated_load() {
    let buffer = create_buffer_typed(64, ScalarDType::FP8E5M2);
    let idx = UOp::const_(DType::Index, ConstValue::Int(0));
    let gate = create_bool_const(false);
    let gated_index = UOp::new(
        Op::Index { buffer: buffer.clone(), indices: smallvec![idx], gate: Some(gate) },
        DType::Scalar(ScalarDType::FP8E5M2),
    );
    let load = UOp::load().buffer(buffer).index(gated_index).dtype(DType::Scalar(ScalarDType::FP8E5M2)).call();

    let rendered = apply_pm_render(&load);
    assert_eq!(count_gated_loads_without_alt(&rendered), 0, "pm_render must attach alt values for gated FP8 loads");

    let mut ctx = crate::devectorize::Fp8DecompCtx { from: ScalarDType::FP8E5M2, to: ScalarDType::Float16 };
    let decomposed = morok_ir::rewrite::graph_rewrite_with_bpm(
        &crate::devectorize::pm_float_decomp(),
        &crate::devectorize::pm_float_decomp_store(),
        rendered,
        &mut ctx,
    );

    assert!(count_gated_loads(&decomposed) > 0, "expected at least one gated load after FP8 decomposition");
    assert_eq!(count_gated_loads_without_alt(&decomposed), 0, "FP8 decomposition must preserve alt on gated loads");
}

fn count_gated_loads(root: &Arc<UOp>) -> usize {
    root.toposort()
        .into_iter()
        .filter(|node| matches!(node.op(), Op::Load { index, .. } if index_has_gate(index)))
        .count()
}

fn count_gated_loads_without_alt(root: &Arc<UOp>) -> usize {
    root.toposort()
        .into_iter()
        .filter(|node| matches!(node.op(), Op::Load { index, alt: None, .. } if index_has_gate(index)))
        .count()
}

fn index_has_gate(index: &Arc<UOp>) -> bool {
    match index.op() {
        Op::Index { gate: Some(_), .. } => true,
        Op::Cast { src, .. } => index_has_gate(src),
        _ => false,
    }
}
