//! `bool_storage_patterns`: bool LOAD/STORE go through uint8 storage so LLVM never
//! sees an `i1` with garbage high bits (tinygrad's PTX/NIR bool rules).

use svod_dtype::{DType, ScalarDType};
use svod_ir::types::ConstValue;
use svod_ir::{Op, UOp};
use test_case::test_case;

use super::helpers::*;
use svod_ir::ops;

/// A bool LOAD becomes `CAST(LOAD<uint8>, bool)`; every other element type is left alone.
#[test_case(ScalarDType::Bool; "bool loads through uint8")]
#[test_case(ScalarDType::Float32; "float32 untouched")]
#[test_case(ScalarDType::Int32; "int32 untouched")]
fn load_uses_uint8_storage_only_for_bool(scalar: ScalarDType) {
    let load = create_load(create_index(create_buffer_typed(64, scalar), 0));
    let result = apply_bool_storage(&load);

    if scalar != ScalarDType::Bool {
        assert_is_load(&result);
        assert_eq!(result.dtype(), DType::Scalar(scalar));
        return;
    }
    let Op::Cast(ops::Cast { src, dtype }) = result.op() else { panic!("expected CAST(LOAD), got {}", result.tree()) };
    assert_eq!(*dtype, DType::Bool);
    assert_is_load(src);
    assert_eq!(src.dtype(), DType::UInt8);
}

/// A bool STORE casts its value to uint8 first; other element types keep theirs.
#[test_case(create_bool_const(true), ScalarDType::Bool, ScalarDType::UInt8; "bool stores as uint8")]
#[test_case(create_vector_bool(vec![true, false, true, false]), ScalarDType::Bool, ScalarDType::UInt8; "shaped bool stores as uint8")]
#[test_case(create_float_const(3.0), ScalarDType::Float32, ScalarDType::Float32; "float32 untouched")]
fn store_uses_uint8_storage_only_for_bool(value: std::sync::Arc<UOp>, buffer: ScalarDType, expected: ScalarDType) {
    let store = create_store(create_index(create_buffer_typed(64, buffer), 0), value);
    let result = apply_bool_storage(&store);

    let Op::Store(ops::Store { value, .. }) = result.op() else { panic!("expected STORE, got {}", result.tree()) };
    assert_eq!(value.dtype().base(), expected, "{}", result.tree());
}

/// An Invalid store value has no bool storage form yet; it is left for the final
/// decomposition pass.
#[test]
fn invalid_bool_store_is_left_for_final_cleanup() {
    let store = create_store(create_index(create_bool_buffer(1), 0), UOp::invalid_marker());
    assert!(std::sync::Arc::ptr_eq(&apply_bool_storage(&store), &store));
}

/// The gate and its alt survive the storage rewrite, with the alt widened to uint8.
#[test]
fn gated_bool_load_keeps_gate_and_converts_alt() {
    let index = UOp::index()
        .buffer(create_bool_buffer(64))
        .indices(vec![UOp::const_(DType::Index, ConstValue::Int(0))])
        .call()
        .unwrap();
    let load = UOp::load().index(index).alt(create_bool_const(true)).gate(create_bool_const(false)).call();

    let result = apply_bool_storage(&load);

    let Op::Cast(ops::Cast { src, .. }) = result.op() else { panic!("expected CAST(LOAD), got {}", result.tree()) };
    let Op::Load(ops::Load { alt: Some(alt), gate: Some(_), .. }) = src.op() else {
        panic!("the late LOAD gate and alt must both survive: {}", src.tree())
    };
    assert_eq!(alt.dtype(), DType::UInt8);
}

/// The full pass reaches the same bool storage form, and lowers BITCAST to CAST on
/// the way (no backend renders a bool bitcast).
#[test]
fn devectorize_lowers_bool_loads_and_bitcasts() {
    let load = apply_devectorize(&create_load(create_index(create_bool_buffer(64), 0)));
    assert!(matches!(load.op(), Op::Cast(ops::Cast { src, .. }) if src.dtype() == DType::UInt8), "{}", load.tree());
    assert_eq!(load.dtype(), DType::Bool);

    let bitcast =
        UOp::new(Op::BitCast(ops::BitCast { src: create_bool_const(true), dtype: DType::UInt8 }), DType::UInt8);
    let result = apply_devectorize(&bitcast);
    assert!(!result.toposort().iter().any(|uop| matches!(uop.op(), Op::BitCast(..))));
    assert_eq!(result.dtype(), DType::UInt8);
}
