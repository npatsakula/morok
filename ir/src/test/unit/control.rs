//! Control flow operation tests.
//!
//! Tests control flow operations: If, EndIf, Range, End, Barrier.

use std::{f32::consts::PI, f64::consts::E, sync::Arc};

use smallvec::{SmallVec, smallvec};

use svod_dtype::DType;

use crate::ops;
use crate::{AxisId, AxisType, ConstValue, Op, UOp};

type Body = SmallVec<[Arc<UOp>; 4]>;

fn empty_body() -> Body {
    smallvec![]
}

fn single_statement() -> Body {
    smallvec![UOp::native_const(100i32)]
}

fn many_statements() -> Body {
    smallvec![UOp::native_const(1i32), UOp::native_const(2i32), UOp::native_const(3i32)]
}

fn nested_if() -> Body {
    smallvec![UOp::if_(UOp::native_const(false), single_statement())]
}

fn range_body() -> Body {
    smallvec![UOp::range(UOp::native_const(10i32), 0)]
}

// =========================================================================
// If / EndIf
// =========================================================================

/// IF and its ENDIF are void whatever the body holds.
#[test]
fn if_and_endif_are_void() {
    let condition = UOp::native_const(5i32).try_cmplt(&UOp::native_const(10i32)).unwrap();
    assert_eq!(condition.dtype(), DType::Bool);

    for body in [empty_body(), single_statement(), many_statements(), nested_if(), range_body()] {
        let if_op = UOp::if_(condition.clone(), body);
        assert_eq!(if_op.dtype(), DType::Void);
        assert_eq!(UOp::endif(if_op).dtype(), DType::Void);
    }
}

// =========================================================================
// Range / Special
// =========================================================================

/// RANGE, SPECIAL and DEFINE_VAR all take their dtype from the loop extent, which stays
/// weak until lowering — independently of axis type or of how large the extent is.
#[test]
fn loop_index_dtype_follows_the_weak_extent() {
    let end = UOp::native_const(10i32);
    let axis_types = [
        AxisType::Global,
        AxisType::Warp,
        AxisType::Local,
        AxisType::Weak,
        AxisType::Loop,
        AxisType::GroupReduce,
        AxisType::Reduce,
        AxisType::Upcast,
        AxisType::Unroll,
        AxisType::Thread,
    ];
    for (index, axis_type) in axis_types.into_iter().enumerate() {
        let range = UOp::range_axis(end.clone(), AxisId::Renumbered(index), axis_type);
        assert_eq!(range.dtype(), DType::WeakInt, "{axis_type:?}");
    }

    assert_eq!(UOp::range_const(10, 0).dtype(), DType::WeakInt);
    assert_eq!(UOp::define_var("small".to_string(), 0, 10).dtype(), DType::WeakInt);
    assert_eq!(UOp::define_var("large".to_string(), 0, i64::MAX).dtype(), DType::WeakInt);
    assert_eq!(UOp::special(UOp::index_const(8), "gidx0".to_string()).dtype(), DType::WeakInt);
}

#[test]
fn range_and_special_explicit_dtype_preserve_concrete_end() {
    let end = UOp::native_const(8i32);
    let range = UOp::range_axis_dtype(end.clone(), AxisId::Renumbered(0), AxisType::Global, DType::Int32);
    let special = UOp::special_dtype(end.clone(), "gidx0".to_string(), DType::Int32);

    assert_eq!(range.dtype(), DType::Int32);
    assert_eq!(special.dtype(), DType::Int32);
    assert!(matches!(range.op(), Op::Range(ops::Range { end: range_end, .. }) if Arc::ptr_eq(range_end, &end)));
    assert!(
        matches!(special.op(), Op::Special(ops::Special { end: special_end, .. }) if Arc::ptr_eq(special_end, &end))
    );
}

/// `Weak` sorts with `Loop` but keeps its own serialized name.
#[test]
fn weak_axis_priority_letter_and_serialization() {
    assert_eq!(AxisType::Weak.priority(), -1);
    assert_eq!(AxisType::Weak.letter(), 'L');
    assert_eq!(AxisType::Weak.cmp(&AxisType::Loop), std::cmp::Ordering::Equal);
    assert_eq!(serde_json::to_string(&AxisType::Weak).unwrap(), "\"Weak\"");
    assert_eq!(serde_json::from_str::<AxisType>("\"Weak\"").unwrap(), AxisType::Weak);
}

#[test]
fn nested_axis_ids_round_trip_order_and_render_their_full_path() {
    let outer = AxisId::Renumbered(2).child(0);
    let inner = AxisId::Renumbered(2).child(1).child(0);

    assert_eq!(serde_json::from_str::<AxisId>(&serde_json::to_string(&inner).unwrap()).unwrap(), inner);
    assert_eq!(outer.path(), &[2, 0]);
    assert_eq!(inner.path(), &[2, 1, 0]);
    assert!(outer < inner);

    let grouped = AxisId::Renumbered(2).child(1).group_reduce_loop();
    assert!(UOp::range_axis(UOp::index_const(8), grouped, AxisType::Reduce).tree().contains("RANGE(R2_1_2, Reduce)"));
}

// =========================================================================
// End / Barrier
// =========================================================================

#[test]
fn end_is_void() {
    let range = UOp::range_axis(UOp::native_const(10i32), AxisId::Renumbered(0), AxisType::Global);
    assert_eq!(UOp::noop().end(smallvec![range]).dtype(), DType::Void);
}

/// BARRIER is void whatever source dtype it sequences and however many deps it carries.
#[test]
fn barrier_is_void_for_every_source_dtype_and_dep_count() {
    for (dtype, value) in [
        (DType::Int8, ConstValue::Int(1)),
        (DType::Int32, ConstValue::Int(100)),
        (DType::Float32, ConstValue::Float(PI as f64)),
        (DType::Float64, ConstValue::Float(E)),
        (DType::UInt32, ConstValue::UInt(42)),
    ] {
        let src = UOp::const_(dtype, value);
        for deps in [empty_body(), single_statement(), many_statements()] {
            assert_eq!(src.barrier(deps).dtype(), DType::Void);
        }
    }
}
