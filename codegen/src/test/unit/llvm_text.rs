use super::*;
use morok_dtype::{AddrSpace, DType};
use morok_ir::{BinaryOp, Op};

#[test]
fn test_simple_add() {
    let a = UOp::param(0, 1, DType::Float32.ptr(Some(1), AddrSpace::Global), None);
    let b = UOp::param(1, 1, DType::Float32.ptr(Some(1), AddrSpace::Global), None);
    let out = UOp::param(2, 1, DType::Float32.ptr(Some(1), AddrSpace::Global), None);

    let idx = UOp::index_const(0);
    let a_idx = UOp::index().buffer(a.clone()).indices(vec![idx.clone()]).call().unwrap();
    let b_idx = UOp::index().buffer(b.clone()).indices(vec![idx.clone()]).call().unwrap();
    let out_idx = UOp::index().buffer(out.clone()).indices(vec![idx.clone()]).call().unwrap();

    let a_load = UOp::load().buffer(a.clone()).index(a_idx).call();
    let b_load = UOp::load().buffer(b.clone()).index(b_idx).call();

    let add = UOp::new(Op::Binary(BinaryOp::Add, a_load, b_load), DType::Float32);

    let store = out_idx.store(add);
    let sink = UOp::sink(vec![store]);
    let linear = UOp::linear(morok_schedule::linearize_with_cfg(sink.clone()).into());

    let result = render(&linear, Some("test_add")).unwrap();
    println!("{}", result.code);

    assert!(result.code.contains("define void @test_add("));
    assert!(result.code.contains("noalias align 32"));
    assert!(!result.code.contains("_inner"));
    assert!(!result.code.contains("ptr %args"));
    assert!(result.code.contains("fadd"));
    assert!(result.code.contains("load"));
    assert!(result.code.contains("store"));
}
