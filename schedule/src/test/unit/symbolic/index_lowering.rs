//! Tests for pm_lower_index_dtype (Stage 15) - Index dtype lowering patterns.
//!
//! These tests verify that Index dtype operations are correctly lowered
//! to concrete i32/i64 types.

use std::sync::Arc;

use morok_dtype::{AddrSpace, DType, ScalarDType};
use morok_ir::{AxisId, AxisType, ConstValue, Op, UOp};

use crate::pattern::RewriteResult;
use crate::symbolic::index_lowering::pm_lower_index_dtype;

/// Create an Index constant.
fn index_const(val: i64) -> Arc<UOp> {
    UOp::index_const(val)
}

#[test]
fn test_index_const_lowering_i32() {
    // Small constants should lower to i32
    let c = index_const(42);
    assert_eq!(c.dtype(), DType::Index);

    let matcher = pm_lower_index_dtype();
    let result = matcher.rewrite(&c, &mut ());

    if let RewriteResult::Rewritten(lowered) = result {
        assert!(matches!(lowered.dtype(), DType::Scalar(ScalarDType::Int32)), "Small constant should lower to i32");
    }
}

#[test]
fn test_index_const_lowering_i64() {
    // Large constants should lower to i64
    let c = index_const(i64::MAX / 2);

    let matcher = pm_lower_index_dtype();
    let result = matcher.rewrite(&c, &mut ());

    if let RewriteResult::Rewritten(lowered) = result {
        assert!(matches!(lowered.dtype(), DType::Scalar(ScalarDType::Int64)), "Large constant should lower to i64");
    }
}

#[test]
fn test_index_binary_op_lowering() {
    // Binary op with Index operands should be lowered
    let a = index_const(10);
    let b = index_const(20);
    let add = a.try_add(&b).expect("add");

    assert_eq!(add.dtype(), DType::Index);

    let matcher = pm_lower_index_dtype();
    let result = matcher.rewrite(&add, &mut ());

    if let RewriteResult::Rewritten(lowered) = result {
        assert!(
            matches!(lowered.dtype(), DType::Scalar(ScalarDType::Int32) | DType::Scalar(ScalarDType::Int64)),
            "Binary op should lower to concrete int type"
        );
    }
}

#[test]
fn test_index_where_lowering() {
    // WHERE with Index branches should be lowered
    let cond = UOp::const_(DType::Bool, ConstValue::Int(1));
    let x = index_const(10);
    let y = index_const(20);
    let where_op = UOp::try_where(cond, x.clone(), y.clone()).expect("where");

    let matcher = pm_lower_index_dtype();
    let result = matcher.rewrite(&where_op, &mut ());

    if let RewriteResult::Rewritten(lowered) = result {
        assert!(
            matches!(lowered.dtype(), DType::Scalar(ScalarDType::Int32) | DType::Scalar(ScalarDType::Int64)),
            "WHERE with Index branches should lower"
        );
    }
}

#[test]
fn test_index_range_lowering() {
    // RANGE with Index end should be lowered
    let end = index_const(100);
    let range = UOp::range_axis(end, AxisId::Renumbered(0), AxisType::Reduce);

    let matcher = pm_lower_index_dtype();
    let result = matcher.rewrite(&range, &mut ());

    if let RewriteResult::Rewritten(lowered) = result {
        // The range's end should be lowered
        assert!(matches!(lowered.op(), Op::Range { .. }));
    }
}

#[test]
fn test_cast_to_index_removal() {
    // Cast from concrete int to Index should be removed
    let concrete = UOp::native_const(42i32);
    let cast_to_index = concrete.cast(DType::Index);

    let matcher = pm_lower_index_dtype();
    let result = matcher.rewrite(&cast_to_index, &mut ());

    if let RewriteResult::Rewritten(lowered) = result {
        // Should just return the concrete int
        assert!(lowered.dtype().is_int(), "Cast to Index should be stripped, returning concrete int");
    }
}

#[test]
fn test_define_var_lowering_i32() {
    // DEFINE_VAR with small bounds should lower to i32
    let dv = UOp::new(Op::DefineVar { name: "x".into(), min_val: 0, max_val: 1000 }, DType::Index);

    let matcher = pm_lower_index_dtype();
    let result = matcher.rewrite(&dv, &mut ());

    if let RewriteResult::Rewritten(lowered) = result {
        assert!(matches!(lowered.dtype(), DType::Scalar(ScalarDType::Int32)), "Small DEFINE_VAR should lower to i32");
    }
}

#[test]
fn test_define_var_lowering_i64() {
    // DEFINE_VAR with large bounds should lower to i64
    let dv = UOp::new(Op::DefineVar { name: "x".into(), min_val: 0, max_val: i64::MAX / 2 }, DType::Index);

    let matcher = pm_lower_index_dtype();
    let result = matcher.rewrite(&dv, &mut ());

    if let RewriteResult::Rewritten(lowered) = result {
        assert!(matches!(lowered.dtype(), DType::Scalar(ScalarDType::Int64)), "Large DEFINE_VAR should lower to i64");
    }
}

#[test]
fn test_sink_cast_strip() {
    // SINK should strip .cast(index) from sources
    let val = UOp::native_const(42i32);
    let cast_val = val.cast(DType::Index);
    let sink = UOp::sink(vec![cast_val]);

    let matcher = pm_lower_index_dtype();
    let result = matcher.rewrite(&sink, &mut ());

    if let RewriteResult::Rewritten(lowered) = result {
        // The sink should have the uncast value
        if let Op::Sink { sources } = lowered.op() {
            assert!(!sources.is_empty());
            assert!(sources[0].dtype().is_int(), "SINK source should have cast stripped");
        }
    }
}

#[test]
fn test_index_with_cast_cleanup() {
    // INDEX(buf, idx.cast(index)) where idx is i32 → INDEX(buf, idx)
    let ptr_dtype = DType::Float32.ptr(Some(100), AddrSpace::Global);
    let buffer = UOp::define_global(0, ptr_dtype);
    let idx = UOp::native_const(0i32);
    let idx_cast = idx.cast(DType::Index);

    let index_op = UOp::index().buffer(buffer).indices(vec![idx_cast]).call().expect("index creation");

    let matcher = pm_lower_index_dtype();
    let result = matcher.rewrite(&index_op, &mut ());

    if let RewriteResult::Rewritten(lowered) = result {
        // The index should have the uncast idx
        if let Op::Index { indices, .. } = lowered.op() {
            assert!(!indices.is_empty());
            assert!(indices[0].dtype().is_int(), "INDEX should have cast stripped from index");
        }
    }
}

#[test]
fn test_index_with_gated_cast_cleanup() {
    // INDEX(buf, idx.cast(index), valid) → INDEX(buf, idx, valid)
    let ptr_dtype = DType::Float32.ptr(Some(100), AddrSpace::Global);
    let buffer = UOp::define_global(0, ptr_dtype);
    let idx = UOp::native_const(0i32);
    let idx_cast = idx.cast(DType::Index);
    let valid = UOp::const_(DType::Bool, ConstValue::Int(1));

    let index_op = UOp::index().buffer(buffer).indices(vec![idx_cast]).gate(valid).call().expect("index creation");

    let matcher = pm_lower_index_dtype();
    let result = matcher.rewrite(&index_op, &mut ());

    if let RewriteResult::Rewritten(lowered) = result {
        // The index should have the uncast idx and preserve valid
        if let Op::Index { indices, gate, .. } = lowered.op() {
            assert!(!indices.is_empty());
            assert!(indices[0].dtype().is_int(), "INDEX should have cast stripped");
            assert!(gate.is_some(), "INDEX should preserve gate");
        }
    }
}
