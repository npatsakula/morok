//! Weak-dtype lowering: every weak node must commit to a concrete width before codegen,
//! and the commit must be driven by the consumer, not by the weak node itself.

use std::sync::Arc;

use svod_dtype::{AddrSpace, DType};
use svod_ir::{BinaryOp, ConstValue, ConstValueHash, Op, ParamArg, TernaryOp, UOp};
use test_case::test_case;

use crate::rewrite::graph_rewrite;
use crate::symbolic::index_lowering::{WeakMemo, pm_commit_weak, pm_lower_index_dtype, pm_lower_weak};
use svod_ir::ops;

fn lower_weak(graph: Arc<UOp>) -> Arc<UOp> {
    graph_rewrite(&pm_lower_weak(), graph, &mut ())
}

fn lower_index(graph: Arc<UOp>) -> Arc<UOp> {
    graph_rewrite(&pm_lower_index_dtype(), graph, &mut WeakMemo::default())
}

fn assert_no_weak(graph: &Arc<UOp>) {
    assert!(graph.toposort().iter().all(|node| !node.dtype().is_weak()), "weak dtype survived:\n{}", graph.tree());
}

fn weak_int(value: i64) -> Arc<UOp> {
    UOp::const_(DType::WeakInt, ConstValue::Int(value))
}

fn float_buffer(slot: usize, len: usize) -> Arc<UOp> {
    UOp::param(slot, len, DType::Float32, None)
}

fn index_of(buffer: Arc<UOp>, index: Arc<UOp>) -> Arc<UOp> {
    UOp::index().buffer(buffer).indices(vec![index]).call().unwrap()
}

/// A weak leaf is wrapped in a CAST back to its weak dtype; the CAST source carries the
/// committed width, chosen mechanically from the value.
#[test_case(weak_int(42), DType::WeakInt, DType::Int32; "int fits the default width")]
#[test_case(weak_int(i64::MAX / 2), DType::WeakInt, DType::Int64; "int needs the wide default")]
#[test_case(UOp::const_(DType::WeakFloat, ConstValue::Float(1.5)), DType::WeakFloat, DType::Float32; "float default")]
#[test_case(UOp::native_const(7i32).cast(DType::WeakInt).cast(DType::WeakFloat), DType::WeakFloat, DType::Float32;
    "stacked weak casts resolve at the outer default")]
fn weak_leaf_commits_to_the_default_width(weak: Arc<UOp>, expected_weak: DType, expected_source: DType) {
    let lowered = lower_weak(weak);
    let Op::Cast(ops::Cast { src, dtype }) = lowered.op() else {
        panic!("expected a weak cast, got {}", lowered.tree())
    };
    assert_eq!(*dtype, expected_weak);
    assert_eq!(src.dtype(), expected_source);
}

#[test_case(DType::WeakInt, vec![ConstValue::Int(1); 4], DType::Int32; "int lanes")]
#[test_case(DType::WeakFloat, vec![ConstValue::Float(1.5); 8], DType::Float32; "float lanes")]
fn weak_vconst_commits_lanewise_to_the_default_width(dtype: DType, values: Vec<ConstValue>, expected: DType) {
    let lanes = values.len();
    let lowered = lower_weak(UOp::vconst(values, dtype.clone()));
    let Op::Cast(ops::Cast { src, dtype: cast_dtype }) = lowered.op() else { panic!("expected a weak vector cast") };
    assert_eq!(*cast_dtype, dtype.vec(lanes).unwrap());
    assert_eq!(src.dtype(), expected.vec(lanes).unwrap());
}

/// The commit rounds each lane to the committed width before any consumer can read it —
/// an `f64` midpoint that `f32` cannot represent must already read as exactly `1.0` — and
/// an Invalid lane passes through untouched so the gater still sees it.
#[test]
fn weak_vconst_commit_rounds_lanes_and_keeps_invalid() {
    let midpoint = 1.0 + 2f64.powi(-24);
    let weak =
        UOp::vconst(vec![ConstValue::Float(midpoint), ConstValue::Invalid, ConstValue::Float(2.0)], DType::WeakFloat);

    let lowered = lower_index(UOp::sink(vec![weak]));

    let Op::Sink(ops::Sink { sources, .. }) = lowered.op() else { panic!("expected a sink") };
    assert_eq!(sources[0].dtype(), DType::Float32.vec(3).unwrap());
    assert!(matches!(sources[0].op(), Op::VConst(ops::VConst { values })
        if values == &vec![ConstValue::Float(1.0), ConstValue::Invalid, ConstValue::Float(2.0)]));
}

/// A weak source under a concrete consumer commits to the consumer's dtype. A bare weak
/// constant is rewritten in place; a `shape_to_uop` extent is a shaped node and keeps a
/// CAST, but neither may leave anything weak behind.
#[test_case(weak_int(1), DType::Int64, true; "weak constant")]
#[test_case(svod_ir::shape::shape_to_uop(&[2usize.into(), 3usize.into()].into_iter().collect()), DType::Int32, false;
    "shape extent")]
fn weak_operand_commits_to_its_concrete_consumer(weak: Arc<UOp>, target: DType, commits_in_place: bool) {
    assert!(weak.dtype().is_weak());
    let concrete = UOp::variable("idx".into(), 0, 31, target.clone());

    let lowered = lower_index(UOp::new(Op::Binary(BinaryOp::Add, concrete, weak), target.clone()));

    let Op::Binary(BinaryOp::Add, lhs, rhs) = lowered.op() else { panic!("expected an add") };
    assert_eq!(lowered.dtype(), target);
    assert_eq!(lhs.dtype(), target);
    assert_eq!(rhs.dtype(), target);
    if commits_in_place {
        assert!(matches!(rhs.op(), Op::Const(_)), "expected no residual cast: {}", lowered.tree());
    }
    assert_no_weak(&lowered);
}

/// A comparison has no weak result to drive the commit, so the node itself must be lowered
/// and both operands unified at the width the widest of them needs.
#[test]
fn comparison_lowers_whole_node_to_unify_operand_width() {
    let comparison = UOp::new(Op::Binary(BinaryOp::Lt, weak_int(i32::MAX as i64 + 1), weak_int(1)), DType::Bool);

    let lowered = lower_index(comparison);

    let Op::Binary(BinaryOp::Lt, lhs, rhs) = lowered.op() else { panic!("expected a comparison") };
    assert_eq!(lowered.dtype(), DType::Bool);
    assert_eq!(lhs.dtype(), DType::Int64);
    assert_eq!(rhs.dtype(), DType::Int64);
}

/// A shift takes its result dtype from the left operand, so committing the left operand
/// re-derives the whole node — the shift count follows it rather than the other way round.
#[test]
fn committing_a_shift_lhs_rederives_the_result_dtype() {
    let shift = UOp::new(Op::Binary(BinaryOp::Shl, weak_int(1), UOp::native_const(2i64)), DType::WeakInt);

    let lowered = lower_index(shift);

    let Op::Binary(BinaryOp::Shl, lhs, _) = lowered.op() else { panic!("expected a shift") };
    assert_eq!(lhs.dtype(), DType::Int64);
    assert_eq!(lowered.dtype(), DType::Int64);
}

#[test_case(DType::Int8; "i8")]
#[test_case(DType::UInt8; "u8")]
#[test_case(DType::Int16; "i16")]
#[test_case(DType::UInt16; "u16")]
#[test_case(DType::Int32; "i32")]
#[test_case(DType::UInt32; "u32")]
#[test_case(DType::Int64; "i64")]
#[test_case(DType::UInt64; "u64")]
fn weak_shift_counts_commit_to_the_integer_lhs_width(dtype: DType) {
    let value = if dtype.is_unsigned() { ConstValue::UInt(8) } else { ConstValue::Int(8) };
    let lhs = UOp::const_(dtype.clone(), value);

    for op in [BinaryOp::Shl, BinaryOp::Shr] {
        let shift = UOp::new(Op::Binary(op, lhs.clone(), UOp::index_const(1)), dtype.clone());
        let lowered = graph_rewrite(&pm_commit_weak(), shift, &mut ());

        let Op::Binary(actual, actual_lhs, actual_rhs) = lowered.op() else { panic!("expected a shift") };
        assert_eq!(*actual, op);
        assert_eq!(lowered.dtype(), dtype);
        assert_eq!(actual_lhs.dtype(), dtype);
        assert_eq!(actual_rhs.dtype(), dtype);
        assert_no_weak(&lowered);
    }
}

/// A concrete CAST is a floor, not a ceiling: it fixes the result width but the weak
/// operands underneath still commit to whatever width their own values need.
#[test]
fn concrete_cast_is_a_width_floor() {
    let weak_add = UOp::new(Op::Binary(BinaryOp::Add, weak_int(i32::MAX as i64 + 1), weak_int(1)), DType::WeakInt);

    let lowered = lower_index(weak_add.cast(DType::Int32));

    let Op::Cast(ops::Cast { src, dtype }) = lowered.op() else { panic!("expected a concrete cast") };
    assert_eq!(*dtype, DType::Int32);
    assert!(src.op().sources().iter().all(|source| source.dtype() == DType::Int64));
}

fn weak_bitwise_index(combine: fn(&Arc<UOp>, &Arc<UOp>) -> Arc<UOp>) -> Arc<UOp> {
    let expression = combine(&UOp::index_const(12), &UOp::index_const(3));
    assert_eq!(expression.dtype(), DType::WeakInt, "graph construction must preserve mathematical integers");
    index_of(float_buffer(0, 16), expression)
}

/// A lane read out of a hardware vector of weak constants.
fn weak_lane_extraction() -> Arc<UOp> {
    let lanes = UOp::vconst((0..4).map(ConstValue::Int).collect(), DType::WeakInt);
    UOp::sink(vec![index_of(lanes, UOp::index_const(2))])
}

/// A shaped vector (STACK) added to a hardware vector (VCONST) of weak constants.
fn mixed_vector_add() -> Arc<UOp> {
    let shaped = UOp::stack((0..8).map(|value| UOp::native_const(value as i64)).collect());
    let hardware = UOp::vconst(vec![ConstValue::Int(1); 8], DType::WeakInt);
    UOp::sink(vec![UOp::new(Op::Binary(BinaryOp::Add, shaped, hardware), DType::WeakInt.vec(8).unwrap())])
}

#[test_case(weak_bitwise_index(|value, operand| value.try_shr_op(operand).unwrap()); "shifted index")]
#[test_case(weak_bitwise_index(|value, operand| value.try_and_op(operand).unwrap()); "masked index")]
#[test_case(weak_bitwise_index(|value, operand| value.try_xor_op(operand).unwrap()); "xored index")]
#[test_case(weak_lane_extraction(); "lane extracted from a hardware vector")]
#[test_case(mixed_vector_add(); "shaped vector added to a hardware vector")]
fn no_weak_dtype_reaches_the_program_boundary(graph: Arc<UOp>) {
    assert_no_weak(&lower_index(graph));
}

/// A weak PARAM that is an ALU value is lowered; one that addresses memory keeps its weak
/// dtype, because the buffer element type is what decides its width later.
#[test]
fn only_the_alu_weak_param_is_lowered() {
    let shape = svod_ir::shape::shape_to_uop(&[1usize.into()].into_iter().collect());
    let make_param = |addrspace| {
        let arg = ParamArg {
            slot: 0,
            dtype: DType::WeakInt,
            vmin_vmax: Some((ConstValueHash(ConstValue::Int(0)), ConstValueHash(ConstValue::Int(7)))),
            multiple_of: None,
            name: None,
            addrspace,
            axis: None,
            device: None,
            volatile: false,
        };
        UOp::new(Op::Param(ops::Param { shape: shape.clone(), arg: arg.into() }), DType::WeakInt)
    };

    let alu = lower_weak(make_param(None));
    assert!(
        matches!(alu.op(), Op::Cast(ops::Cast { src, dtype }) if *dtype == DType::WeakInt && !src.dtype().is_weak()),
        "{}",
        alu.tree()
    );

    let buffer = lower_weak(make_param(Some(AddrSpace::Global)));
    assert!(matches!(buffer.op(), Op::Param(ops::Param { arg, .. }) if arg.dtype == DType::WeakInt));
}

/// Extracting one element of a shaped LOAD must not collapse the LOAD to a scalar: the
/// shaped LOAD stays under the extracting INDEX so the whole vector is still read.
#[test]
fn lowering_a_weak_index_preserves_the_shaped_load_under_extraction() {
    let offsets = UOp::stack((0..8).map(weak_int).collect());
    let load = UOp::load().index(index_of(UOp::param(0, 64, DType::BFloat16, None), offsets)).call();
    let lane = index_of(load, UOp::index_const(3));

    let matcher = crate::symbolic::patterns::symbolic_simple().with_context::<WeakMemo>() + pm_lower_index_dtype();
    let lowered = graph_rewrite(&matcher, lane, &mut WeakMemo::default());

    let shaped_load = lowered
        .toposort()
        .into_iter()
        .find(|node| matches!(node.op(), Op::Load(..)))
        .expect("shaped LOAD must remain under extraction");
    assert_eq!(shaped_load.dtype(), DType::BFloat16);
    assert_eq!(shaped_load.shape().unwrap().unwrap().as_slice(), &[svod_ir::SInt::Const(8)]);
    assert!(matches!(lowered.op(), Op::Index(ops::Index { buffer, .. }) if Arc::ptr_eq(buffer, &shaped_load)));
    assert_no_weak(&lowered);
}

/// Weak lowering commits the value but must leave the INVALID marker alone — it is the
/// gate, not a number, and rewriting it would turn a skipped access into a real one.
#[test]
fn weak_lowering_preserves_the_invalid_marker() {
    let gate = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let lowered = lower_weak(weak_int(7).valid(gate.clone()));

    let Op::Cast(ops::Cast { src, dtype }) = lowered.op() else { panic!("expected a weak cast") };
    assert_eq!(*dtype, DType::WeakInt);
    let Op::Ternary(TernaryOp::Where, condition, value, invalid) = src.op() else { panic!("expected a WHERE") };
    assert!(Arc::ptr_eq(condition, &gate));
    assert_eq!(value.dtype(), DType::Int32);
    assert!(UOp::is_invalid_marker(invalid));

    let invalid = UOp::invalid_marker();
    let vector = UOp::stack([weak_int(7), invalid.clone()].into_iter().collect());
    let lowered = lower_weak(vector);

    assert_eq!(lowered.dtype(), DType::Int32);
    assert_eq!(lowered.shape().unwrap().unwrap().as_slice(), &[svod_ir::SInt::Const(2)]);
    let Op::Stack(ops::Stack { sources: lanes }) = lowered.op() else { panic!("expected a STACK") };
    assert_eq!(lanes[0].dtype(), DType::Int32);
    assert!(Arc::ptr_eq(&lanes[1], &invalid));
}

/// Rewriting an address INVALID to 0 would turn a skipped access into an unconditional
/// read of element 0, so `pm_remove_invalid` must leave gated addresses to index lowering.
#[test]
fn invalid_removal_leaves_gated_addresses_alone() {
    let address = UOp::var("i", DType::Index, 0, 16).valid(UOp::var("gate", DType::Bool, 0, 1));
    let stacked = UOp::new(
        Op::Stack(ops::Stack { sources: [address.clone(), UOp::invalid_marker()].into_iter().collect() }),
        DType::Index,
    );

    for gated in [address, stacked] {
        let result = graph_rewrite(crate::symbolic::patterns::pm_remove_invalid(), gated.clone(), &mut ());
        assert!(Arc::ptr_eq(&result, &gated));
    }
}

/// A STORE takes its value dtype from the destination buffer and must not adapt, replace
/// or re-index the address it was given.
#[test]
fn store_commits_its_weak_value_to_the_destination() {
    let index = index_of(float_buffer(0, 16), UOp::native_const(0i32));
    let store = index.store(UOp::const_(DType::WeakFloat, ConstValue::Float(1.0)));

    let lowered = graph_rewrite(&pm_commit_weak(), store, &mut ());

    let Op::Store(ops::Store { index: lowered_index, value, .. }) = lowered.op() else { panic!("expected a store") };
    assert_eq!(value.dtype(), DType::Float32);
    assert_eq!(index.dtype(), DType::Float32, "INDEX exposes the adopted buffer dtype");
    assert!(Arc::ptr_eq(lowered_index, &index));
}

/// A 64-bit address may only narrow when the buffer is small enough that every in-range
/// index fits in 32 bits; the gate survives narrowing either way.
#[test_case(16, true; "index fits in 32 bits")]
#[test_case(i32::MAX as usize + 2, false; "buffer needs the full 64-bit range")]
fn gated_long_index_narrows_only_for_small_buffers(size: usize, narrowed: bool) {
    let index = index_of(
        float_buffer(0, size),
        UOp::variable("idx".into(), 0, i64::MAX / 2, DType::Int64)
            .valid(UOp::const_(DType::Bool, ConstValue::Bool(true))),
    );

    let lowered = lower_index(index);

    let Op::Index(ops::Index { indices, .. }) = lowered.op() else { panic!("expected an index") };
    let Op::Ternary(TernaryOp::Where, _, idx, invalid) = indices[0].op() else { panic!("expected a gated index") };
    assert_eq!(idx.dtype() == DType::Int32, narrowed);
    assert!(UOp::is_invalid_marker(invalid));
}

/// `lower_weak_srcs` (tinygrad/uop/weak.py:29-40) keeps a `ctx` dict keyed by source:
/// `if (r:=ctx.get(s)) is None: r = graph_rewrite(s, pm_lower_weak)`. The memo lives for
/// one `to_program` (`ctx={}`, codegen/__init__.py:349), so a source shared by several
/// consumers is rewritten once, not once per edge.
#[test]
fn shared_weak_sources_are_lowered_once_per_pass() {
    let weak_index =
        |offset: i64| UOp::new(Op::Binary(BinaryOp::Add, UOp::range_const(64, 0), weak_int(offset)), DType::WeakInt);
    let shared = weak_index(3);
    let sink = UOp::sink(vec![
        index_of(float_buffer(0, 64), shared.clone()),
        index_of(float_buffer(1, 64), shared),
        index_of(float_buffer(2, 64), weak_index(5)),
    ]);

    let mut memo = WeakMemo::default();
    graph_rewrite(&pm_lower_index_dtype(), sink, &mut memo);

    // Six weak edges reach a non-weak consumer here: three INDEX indices and the shared
    // WeakInt extent of the three PARAM shapes. They collapse to three rewrites.
    assert_eq!(memo.len(), 3, "one entry per distinct weak source, not per consumer edge");
}
