//! MLIR renderer tests.

use morok_dtype::DType;
use morok_ir::{ConstValue, UOp};
use smallvec::smallvec;

use crate::mlir::render;

fn render_linearized(root: &std::sync::Arc<UOp>, name: Option<&str>) -> crate::Result<crate::RenderedKernel> {
    let linear = UOp::linear(morok_schedule::linearize_with_cfg(root.clone()).into());
    render(&linear, name)
}

#[test]
#[should_panic(expected = "multi-index INDEX must be linearized before MLIR codegen")]
fn test_multi_index_requires_linearization() {
    let ptr_dtype = DType::Float32.ptr(None, morok_dtype::AddrSpace::Global);
    let buffer = UOp::param(0, 1024, ptr_dtype, None);
    let i = UOp::const_(DType::Index, ConstValue::Int(1));
    let j = UOp::const_(DType::Index, ConstValue::Int(2));
    let index = UOp::index().buffer(buffer).indices(vec![i, j]).call().unwrap();
    let sink = UOp::sink(vec![index]);

    let linear = UOp::linear(sink.toposort().into());
    let _ = render(&linear, Some("test_multi_index_requires_linearization"));
}

#[test]
fn test_custom_is_explicitly_unsupported() {
    let one = UOp::const_(DType::Int32, ConstValue::Int(1));
    let custom = UOp::custom(smallvec![one], "({0} + 3)".to_string(), DType::Int32);
    let sink = UOp::sink(vec![custom]);

    let err = render_linearized(&sink, Some("test_custom_unsupported")).expect_err("MLIR backend should reject CUSTOM");
    match err {
        crate::Error::MlirError { reason } => {
            assert!(reason.contains("CUSTOM is not supported"), "unexpected error: {reason}");
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[test]
fn test_gated_load_requires_alt() {
    let ptr_dtype = DType::Float32.ptr(None, morok_dtype::AddrSpace::Global);
    let buffer = UOp::param(0, 1024, ptr_dtype, None);
    let idx = UOp::const_(DType::Index, ConstValue::Int(1));
    let gate = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let gated_index = UOp::index().buffer(buffer.clone()).indices(vec![idx]).gate(gate).call().unwrap();
    let load = UOp::load().buffer(buffer).index(gated_index).call();
    let sink = UOp::sink(vec![load]);

    let err = render_linearized(&sink, Some("test_gated_load_requires_alt"))
        .expect_err("MLIR backend should reject gated load without alt");
    match err {
        crate::Error::MlirError { reason } => {
            assert!(reason.contains("gated LOAD without alt"), "unexpected error: {reason}");
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}
