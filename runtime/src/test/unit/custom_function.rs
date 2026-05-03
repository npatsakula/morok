use super::*;

use morok_dtype::DType;
use morok_ir::CustomFunctionKind;

#[test]
fn test_encdec_returns_typed_unsupported() {
    let alloc = morok_device::registry::cpu().expect("cpu allocator");
    let dst = Buffer::new(alloc.clone(), DType::Float32, vec![4], Default::default());
    let src = Buffer::new(alloc, DType::Float32, vec![4], Default::default());

    let attr = morok_ir::UOp::index_const(7);
    let mut bufs = vec![dst, src];
    let err = run_custom_function(&CustomFunctionKind::EncDec, &[attr], &mut bufs, &HashMap::new())
        .expect_err("encdec should report unsupported runtime behavior");

    match err {
        crate::Error::Unsupported { kind, reason } => {
            assert_eq!(kind, "EncDec");
            assert!(reason.contains("attrs=1"), "unexpected reason: {reason}");
        }
        other => panic!("expected Error::Unsupported, got {other:?}"),
    }
}

#[test]
fn test_graph_returns_typed_unsupported() {
    let mut no_buffers = Vec::<Buffer>::new();
    let err = run_custom_function(&CustomFunctionKind::Graph, &[], &mut no_buffers, &HashMap::new())
        .expect_err("graph should report unsupported runtime behavior");
    assert!(matches!(err, crate::Error::Unsupported { kind, .. } if kind == "Graph"));
}

#[test]
fn test_encdec_unsupported_does_not_require_buffers_first() {
    let mut no_buffers = Vec::<Buffer>::new();
    let err = run_custom_function(&CustomFunctionKind::EncDec, &[], &mut no_buffers, &HashMap::new())
        .expect_err("encdec should fail as unsupported");
    assert!(matches!(err, crate::Error::Unsupported { .. }));
}
