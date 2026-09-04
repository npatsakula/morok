//! Rule-level tests for Tinygrad's `spec_program` at pinned commit 8c8b43de.
//!
//! `spec_tensor` is used where useful to prove that a rejection comes from the
//! program rule rather than an inherited `spec_shared` rule.

use std::sync::Arc;

use smallvec::smallvec;
use svod_dtype::{AddrSpace, DType, DeviceSpec};
use svod_ir::types::ConstValue;
use svod_ir::{BinaryOp, ConstValueHash, Op, ParamArg, ReduceOp, UOp};
use test_case::test_case;

use crate::optimizer::apply_pre_optimization;
use crate::spec::{
    SpecError, spec_hcq, spec_program, spec_tensor, type_verify, verify_kernel_graph, verify_no_legacy_index_dtype,
};

fn global_param(slot: usize) -> Arc<UOp> {
    UOp::new(
        Op::Param {
            shape: UOp::stack(smallvec![]),
            arg: ParamArg::buffer(slot, DType::Float32, AddrSpace::Global, None).into(),
        },
        DType::Float32,
    )
}

fn int_const(dtype: DType, value: i64) -> Arc<UOp> {
    UOp::const_(dtype, ConstValue::Int(value))
}

fn integer_const(dtype: DType, value: u64) -> Arc<UOp> {
    let constant = if dtype.is_unsigned() { ConstValue::UInt(value) } else { ConstValue::Int(value as i64) };
    UOp::const_(dtype, constant)
}

fn verify_program_err(root: &Arc<UOp>) -> String {
    type_verify(root, &spec_program()).expect_err("expected spec_program rejection").to_string()
}

fn structured_buffer(addrspace: AddrSpace, device: Option<DeviceSpec>) -> Arc<UOp> {
    UOp::new(
        Op::Buffer {
            shape: int_const(DType::Int32, 4),
            arg: ParamArg::buffer(3, DType::Float32, addrspace, device).into(),
        },
        DType::Float32,
    )
}

fn float_const(dtype: DType, value: f64) -> Arc<UOp> {
    UOp::new(Op::Const(ConstValueHash(ConstValue::Float(value))), dtype)
}

fn float_vconst(dtype: DType, values: [f64; 2]) -> Arc<UOp> {
    UOp::new(
        Op::VConst { values: values.into_iter().map(ConstValue::Float).collect() },
        dtype.vec(2).expect("two-lane vector"),
    )
}

fn special(dtype: DType, end: i64) -> Arc<UOp> {
    UOp::new(Op::Special { end: int_const(dtype.clone(), end), name: "lidx0".to_string() }, dtype)
}

fn shaped_stack(dtype: DType) -> Arc<UOp> {
    UOp::stack((0..4).map(|value| UOp::const_(dtype.clone(), ConstValue::Float(value as f64))).collect())
}

fn if_over(dedup_source: Arc<UOp>) -> Arc<UOp> {
    let condition = UOp::const_(DType::Bool, ConstValue::Bool(true));
    UOp::new(Op::If { condition, body: smallvec![dedup_source] }, DType::Void)
}

fn scalar_index() -> Arc<UOp> {
    UOp::index().buffer(global_param(0)).indices(vec![int_const(DType::Int32, 0)]).call().unwrap()
}

/// The `END(END(x, range), backedge)` shape that `split_ends` leaves behind.
fn split_ends_backedge() -> Arc<UOp> {
    let range = UOp::range_axis_dtype(
        int_const(DType::Int32, 4),
        svod_ir::AxisId::Renumbered(0),
        svod_ir::types::AxisType::Loop,
        DType::Int32,
    );
    UOp::noop().end(smallvec![range]).end(smallvec![UOp::const_(DType::Bool, ConstValue::Bool(true))])
}

#[test_case(structured_buffer(AddrSpace::Local, None); "local buffer")]
#[test_case(structured_buffer(AddrSpace::Reg, None); "reg buffer")]
#[test_case(int_const(DType::Int32, 1); "concrete int const")]
#[test_case(float_const(DType::Float32, 1.0); "concrete float const")]
#[test_case(float_const(DType::Float32, f64::NAN); "canonical nan")]
#[test_case(float_vconst(DType::Float32, [f64::NAN, 1.0]); "canonical nan lane in a vector")]
#[test_case(special(DType::Int32, 8); "int32 special")]
#[test_case(shaped_stack(DType::BFloat16); "devectorized shaped stack")]
#[test_case(shaped_stack(DType::Float32).index_axes(vec![2]); "shaped index")]
#[test_case(UOp::endif(if_over(scalar_index())); "if closed by endif")]
#[test_case(split_ends_backedge(); "backedge end left by split_ends")]
// spec.py:207-208 places the special SHRINK rule before the general movement rejection.
#[test_case(UOp::new(
    Op::Shrink { src: global_param(0), offsets: int_const(DType::Int32, 0), sizes: int_const(DType::Int32, 1) },
    DType::Float32,
); "special shrink wins over the movement rejection")]
fn spec_program_accepts(node: Arc<UOp>) {
    type_verify(&UOp::sink(vec![node]), &spec_program()).expect("spec_program should accept");
}

#[test_case(structured_buffer(AddrSpace::Global, Some(DeviceSpec::Cpu)), "structured REG/LOCAL allocation"; "global buffer")]
#[test_case(UOp::native_const(1.0f32).reduce_with_num_axes(smallvec![], ReduceOp::Add, 1), "must be rangeified"; "tensor-form reduce")]
#[test_case(UOp::const_(DType::Index, ConstValue::Int(1)), "legacy Index dtype must be lowered"; "legacy index dtype")]
#[test_case(float_const(DType::Float32, 1.0 + 2f64.powi(-24)), "not canonical for its dtype"; "unrepresentable scalar")]
#[test_case(float_vconst(DType::Float32, [1.0, 1.0 + 2f64.powi(-24)]), "not canonical for its dtype"; "unrepresentable lane")]
#[test_case(float_const(DType::Float32, f64::from_bits(0x7ff8_0000_0000_0001)), "not canonical for its dtype"; "nan with a payload")]
#[test_case(if_over(int_const(DType::Int32, 0)), "CAST/INDEX/SHRINK dedup source"; "if over a non-index dedup source")]
#[test_case(UOp::new(Op::EndIf { if_op: int_const(DType::Int32, 0) }, DType::Void), "ENDIF must be void and close an IF"; "endif without an if")]
#[test_case(special(DType::Int64, 8), "must be int32 after index lowering"; "int64 special")]
#[test_case(UOp::new(Op::Multi { src: int_const(DType::Int32, 0), axis: 0 }, DType::Int32), "no matching rule"; "op outside the whitelist")]
fn spec_program_rejects(node: Arc<UOp>, expected: &str) {
    let err = verify_program_err(&UOp::sink(vec![node]));
    assert!(err.contains(expected), "unexpected error: {err}");
}

/// Forms the tensor graph may still carry but a program may not.
#[test_case(UOp::const_(DType::WeakInt, ConstValue::Int(1)), "weak dtype must be lowered"; "weakint")]
#[test_case(UOp::const_(DType::WeakFloat, ConstValue::Float(1.0)), "weak dtype must be lowered"; "weakfloat")]
#[test_case(UOp::new(
    Op::Reshape { src: int_const(DType::Int32, 0), new_shape: int_const(DType::Int32, 1) },
    DType::Int32,
), "movement op must be lowered away"; "movement op")]
#[test_case(UOp::invalid_marker(), "Invalid constant must be folded out"; "invalid marker")]
fn spec_tensor_accepts_what_spec_program_rejects(node: Arc<UOp>, expected: &str) {
    let sink = UOp::sink(vec![node]);
    type_verify(&sink, &spec_tensor()).expect("legal in spec_shared/spec_tensor");
    let err = verify_program_err(&sink);
    assert!(err.contains(expected), "unexpected error: {err}");
}

fn raw_index(index: Arc<UOp>) -> Arc<UOp> {
    UOp::new(Op::Index { buffer: global_param(0), indices: smallvec![index] }, DType::Float32)
}

#[test_case(UOp::const_(DType::WeakInt, ConstValue::Int(0)); "weakint")]
#[test_case(int_const(DType::Int32, 1); "int32")]
#[test_case(UOp::vconst(vec![ConstValue::Int(0), ConstValue::Int(1)], DType::Int32); "int32 vector")]
#[test_case(UOp::invalid_marker(); "invalid marker")]
#[test_case(UOp::vconst(vec![ConstValue::Invalid; 4], DType::Bool); "vector of invalid")]
#[test_case(UOp::stack(smallvec![UOp::invalid_marker(); 4]); "stack of invalid markers")]
fn spec_shared_accepts_index_value(index: Arc<UOp>) {
    type_verify(&UOp::sink(vec![raw_index(index)]), &spec_tensor()).expect("legal INDEX address operand");
}

#[test_case(UOp::const_(DType::WeakFloat, ConstValue::Float(0.0)); "weakfloat")]
#[test_case(UOp::const_(DType::Bool, ConstValue::Bool(false)); "bool")]
#[test_case(UOp::vconst(vec![ConstValue::Bool(false), ConstValue::Bool(true)], DType::Bool); "bool vector")]
fn spec_shared_rejects_non_integer_index_value(index: Arc<UOp>) {
    let err = type_verify(&UOp::sink(vec![raw_index(index)]), &spec_tensor())
        .expect_err("non-integer INDEX address operand")
        .to_string();
    assert!(err.contains("non-integer value reached a memory INDEX operand"), "unexpected rejection: {err}");
}

#[test]
fn spec_shared_rejects_index_vector_mixing_bool_and_int_lanes() {
    let mixed =
        UOp::new(Op::VConst { values: vec![ConstValue::Bool(false), ConstValue::Int(0)] }, DType::Bool.vec(2).unwrap());
    assert!(type_verify(&UOp::sink(vec![raw_index(mixed)]), &spec_tensor()).is_err());
}

/// Pinned shift matrix: only a scalar `uint32` count may differ from the left
/// operand's dtype.
#[test]
fn spec_shift_dtype_matrix_matches_pinned_uint32_exception() {
    let integer_dtypes = [
        DType::Int8,
        DType::UInt8,
        DType::Int16,
        DType::UInt16,
        DType::Int32,
        DType::UInt32,
        DType::Int64,
        DType::UInt64,
    ];

    for lhs_dtype in integer_dtypes {
        let lhs = integer_const(lhs_dtype.clone(), 8);
        for op in [BinaryOp::Shl, BinaryOp::Shr] {
            let same = UOp::new(Op::Binary(op, lhs.clone(), integer_const(lhs_dtype.clone(), 1)), lhs_dtype.clone());
            assert!(type_verify(&UOp::sink(vec![same]), &spec_program()).is_ok(), "{op:?} {lhs_dtype:?}");

            let uint32 = UOp::new(Op::Binary(op, lhs.clone(), UOp::native_const(1u32)), lhs_dtype.clone());
            assert!(type_verify(&UOp::sink(vec![uint32]), &spec_program()).is_ok(), "{op:?} {lhs_dtype:?} << u32");

            let weak = UOp::new(Op::Binary(op, lhs.clone(), UOp::index_const(1)), lhs_dtype.clone());
            assert!(type_verify(&UOp::sink(vec![weak.clone()]), &spec_tensor()).is_ok());
            assert!(type_verify(&UOp::sink(vec![weak]), &spec_program()).is_err(), "weak count must commit first");

            let unrelated_dtype = if lhs_dtype == DType::Int8 { DType::Int16 } else { DType::Int8 };
            let unrelated = UOp::new(Op::Binary(op, lhs.clone(), integer_const(unrelated_dtype, 1)), lhs_dtype.clone());
            assert!(type_verify(&UOp::sink(vec![unrelated]), &spec_program()).is_err(), "{op:?} {lhs_dtype:?}");
        }
    }

    let lhs = UOp::vconst(vec![ConstValue::Int(8), ConstValue::Int(16)], DType::Int16);
    let count = UOp::vconst(vec![ConstValue::UInt(1), ConstValue::UInt(2)], DType::UInt32);
    for op in [BinaryOp::Shl, BinaryOp::Shr] {
        let vector_count = UOp::new(Op::Binary(op, lhs.clone(), count.clone()), DType::Int16.vec(2).unwrap());
        assert!(
            type_verify(&UOp::sink(vec![vector_count]), &spec_program()).is_err(),
            "vector u32 is not the exception"
        );
    }
}

#[test]
fn spec_tensor_accepts_structured_global_buffer() {
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    assert!(type_verify(&UOp::sink(vec![buffer]), &spec_tensor()).is_ok());
}

#[test]
fn spec_hcq_accepts_exact_getaddr_and_rejects_non_storage_source() {
    let address = UOp::new(Op::GetAddr { src: global_param(0), device: DeviceSpec::Cpu }, DType::UInt64);
    assert!(type_verify(&UOp::sink(vec![address]), &spec_hcq()).is_ok());

    let invalid = UOp::new(Op::GetAddr { src: UOp::native_const(1u32), device: DeviceSpec::Cpu }, DType::UInt64);
    assert!(type_verify(&UOp::sink(vec![invalid]), &spec_hcq()).is_err());
}

fn kernel_param(slot: usize, dtype: DType) -> Arc<UOp> {
    UOp::new(
        Op::Param {
            shape: UOp::stack(smallvec![]),
            arg: ParamArg::buffer(slot, dtype.clone(), AddrSpace::Global, None).into(),
        },
        dtype,
    )
}

fn kernel_call(formals: Vec<Arc<UOp>>, args: Vec<Arc<UOp>>) -> Arc<UOp> {
    UOp::sink(formals).call(args.into(), svod_ir::CallInfo::default())
}

fn cpu_buffer(dtype: DType) -> Arc<UOp> {
    UOp::new_buffer(DeviceSpec::Cpu, 4, dtype)
}

fn device_mstack() -> Arc<UOp> {
    let cuda = UOp::new_buffer(DeviceSpec::Cuda { device_id: 0 }, 4, DType::Float32);
    UOp::mstack(smallvec![cpu_buffer(DType::Float32), cuda])
}

/// One CALL feeding two buffers through AFTER.
fn multi_output_call_graph() -> Arc<UOp> {
    let call = kernel_call(vec![kernel_param(0, DType::Float32)], vec![cpu_buffer(DType::Float32)]);
    let out0 = cpu_buffer(DType::Float32).after(smallvec![call.clone()]);
    let out1 = cpu_buffer(DType::Float32).after(smallvec![call]);
    UOp::sink(vec![out0, out1])
}

/// A cross-device COPY is the one non-SINK body a CALL may wrap.
fn cross_device_copy_call() -> Arc<UOp> {
    let copy = kernel_param(0, DType::Float32).copy_to_device(DeviceSpec::Cuda { device_id: 0 });
    UOp::sink(vec![copy.call(smallvec![cpu_buffer(DType::Float32)], svod_ir::CallInfo::default())])
}

#[test_case(multi_output_call_graph(); "one call feeding two outputs")]
#[test_case(UOp::sink(vec![device_mstack().mselect(1)]); "concrete-device mstack layout")]
#[test_case(cross_device_copy_call(); "cross-device copy call body")]
fn spec_kernel_graph_accepts(sink: Arc<UOp>) {
    verify_kernel_graph(&sink).expect("valid kernel graph");
}

#[test_case(
    UOp::sink(vec![UOp::native_const(0i32).call(smallvec![], svod_ir::CallInfo::default())]),
    "supported opaque body"; "const call body")]
#[test_case(
    UOp::sink(vec![kernel_call(
        vec![kernel_param(0, DType::Float32), kernel_param(1, DType::Int32)],
        vec![cpu_buffer(DType::Int32), cpu_buffer(DType::Float32)],
    )]),
    "positional arguments"; "call arguments swapped against their slots")]
#[test_case(UOp::sink(vec![device_mstack().mselect(2)]), "in-range MSTACK"; "mselect out of range")]
#[test_case(
    UOp::sink(vec![UOp::mstack(smallvec![cpu_buffer(DType::Float32), kernel_param(0, DType::Float32)])]),
    "MSTACK"; "mstack mixing device and device-free sources")]
#[test_case(
    UOp::sink(vec![cpu_buffer(DType::Float32).copy_to_device(DeviceSpec::Cuda { device_id: 0 })]),
    "no matching rule"; "bare copy in the outer graph")]
fn spec_kernel_graph_rejects(sink: Arc<UOp>, expected: &str) {
    let err = verify_kernel_graph(&sink).expect_err("invalid kernel graph");
    assert!(err.to_string().contains(expected), "unexpected error: {err}");
}

#[test]
fn verification_errors_locate_the_offending_node() {
    let malformed = cpu_buffer(DType::Float32).after(smallvec![UOp::native_const(1i32)]);
    let SpecError::Verification { boundary, uop_id, source_path, reason, .. } =
        verify_kernel_graph(&UOp::sink(vec![malformed.clone()])).expect_err("AFTER dependency must be callable");
    assert_eq!(boundary, "kernel graph");
    assert_eq!(uop_id, malformed.id);
    assert_eq!(source_path, vec![0]);
    assert!(reason.contains("CALL/AFTER dependencies"), "unexpected reason: {reason}");

    let stale = UOp::new(Op::Noop, DType::Index);
    let SpecError::Verification { boundary, uop_id, source_path, reason, .. } =
        verify_no_legacy_index_dtype(&UOp::sink(vec![stale.clone()])).expect_err("stale Index dtype");
    assert_eq!(boundary, "post-index-lowering");
    assert_eq!(uop_id, stale.id);
    assert_eq!(source_path, vec![0]);
    assert_eq!(reason, "legacy Index dtype must be lowered before a program");
}

#[test_case(
    UOp::new(
        Op::Binary(BinaryOp::Add, UOp::native_const(1i32), UOp::const_(DType::Float32, ConstValue::Float(1.0))),
        DType::Int32,
    ),
    "binary operand/result dtype mismatch"; "mixed alu dtype")]
#[test_case(
    structured_buffer(AddrSpace::Global, Some(DeviceSpec::Cpu)),
    "tensor BUFFER must be structured GLOBAL storage"; "global buffer with a non-weakint shape")]
#[test_case(UOp::native_const(1i32).mselect(0), "MSELECT requires"; "mselect of a non-multi source")]
#[test_case(
    UOp::new(
        Op::ReduceAxis {
            src: UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32),
            reduce_op: ReduceOp::Add,
            axes: vec![0],
        },
        DType::Float32,
    ),
    "no matching rule"; "legacy reduce_axis")]
#[test_case(
    UOp::multi(UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32), 1),
    "MULTI must"; "multi axis outside its source shape")]
fn preoptimization_rejects_at_the_tensor_boundary(node: Arc<UOp>, expected: &str) {
    let err = apply_pre_optimization(UOp::sink(vec![node])).expect_err("malformed tensor graph");
    assert!(err.to_string().contains(expected), "unexpected error: {err}");
}

#[test]
fn preoptimization_accepts_a_hand_authored_custom_kernel() {
    let index = UOp::index().buffer(global_param(0)).indices(vec![int_const(DType::Int32, 0)]).call().unwrap();
    let loaded = UOp::load().index(index.clone()).call();
    let custom = UOp::custom(smallvec![loaded], "({0} + 1.0f)".to_string(), DType::Float32);

    assert!(apply_pre_optimization(UOp::sink(vec![index.store(custom)])).is_ok());
}
