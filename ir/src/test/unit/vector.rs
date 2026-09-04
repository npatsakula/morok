//! Vector operation tests.
//!
//! Tests shaped STACK and late vector operations.

use std::sync::Arc;

use smallvec::smallvec;
use test_case::test_case;

use svod_dtype::DType;

use crate::ops;
use crate::{ConstValue, Op, UOp};

fn lane(dtype: DType, value: i64) -> Arc<UOp> {
    UOp::const_(dtype, ConstValue::Int(value))
}

fn sources(stack: &Arc<UOp>) -> &[Arc<UOp>] {
    let Op::Stack(ops::Stack { sources }) = stack.op() else { panic!("expected STACK, got {:?}", stack.op()) };
    sources
}

// =========================================================================
// Lane dtype promotion
// =========================================================================

/// Lanes are promoted to the STACK's own dtype by an inserted CAST; already-promoted lanes
/// are left untouched (pointer-identical).
#[test]
fn stack_casts_only_the_lanes_that_need_promoting() {
    let weak = UOp::const_(DType::WeakInt, ConstValue::Int(1));
    let strong = lane(DType::Int16, 2);
    let stack = UOp::stack(smallvec![weak.clone(), strong.clone()]);

    assert_eq!(stack.dtype(), DType::Int16);
    assert!(
        matches!(sources(&stack)[0].op(), Op::Cast(ops::Cast { src, dtype }) if Arc::ptr_eq(src, &weak) && *dtype == DType::Int16)
    );
    assert!(Arc::ptr_eq(&sources(&stack)[1], &strong));
}

#[test_case(DType::WeakInt, DType::Int16, DType::Int16; "weak int joins the strong lane")]
#[test_case(DType::Int8, DType::UInt16, DType::Int32; "mixed integer widths widen to int32")]
#[test_case(DType::Int16, DType::Float32, DType::Float32; "integer lane joins the float lane")]
fn stack_promotes_every_lane_to_the_stack_dtype(lhs: DType, rhs: DType, expected: DType) {
    let stack = UOp::stack(smallvec![lane(lhs, 1), lane(rhs, 2)]);

    assert_eq!(stack.dtype(), expected);
    assert!(sources(&stack).iter().all(|source| source.dtype() == expected));
}

/// Promotion of shaped lanes must not disturb their shapes, and must not paper over lanes
/// whose shapes disagree — those still produce an unshaped STACK.
#[test]
fn stack_casts_shaped_lanes_without_touching_their_shapes() {
    let weak_row = UOp::stack(smallvec![lane(DType::WeakInt, 1), lane(DType::WeakInt, 2)]);
    let strong_row = UOp::stack(smallvec![lane(DType::Int16, 3), lane(DType::Int16, 4)]);
    let matrix = UOp::stack(smallvec![weak_row, strong_row.clone()]);

    assert_eq!(matrix.dtype(), DType::Int16);
    assert_eq!(matrix.shape().unwrap().unwrap().as_slice(), &[2usize.into(), 2usize.into()]);
    assert!(sources(&matrix).iter().all(|source| source.dtype() == DType::Int16));

    let short = UOp::stack(smallvec![lane(DType::WeakInt, 1)]);
    let mismatched = UOp::stack(smallvec![short, strong_row]);
    assert_eq!(mismatched.dtype(), DType::Int16);
    assert_eq!(mismatched.shape().unwrap(), None, "casts must not hide mismatched lane shapes");
    assert!(sources(&mismatched).iter().all(|source| source.dtype() == DType::Int16));
}

#[test]
fn stack_reconstruction_recasts_rewritten_lanes() {
    let original = UOp::stack(smallvec![lane(DType::Int16, 1), lane(DType::Int16, 2)]);
    let weak = UOp::const_(DType::WeakInt, ConstValue::Int(3));
    let float = UOp::const_(DType::Float32, ConstValue::Float(4.0));
    let rebuilt = original.with_sources(vec![weak, float]);

    assert_eq!(rebuilt.dtype(), DType::Float32);
    assert!(sources(&rebuilt).iter().all(|source| source.dtype() == DType::Float32));
}

/// INVALID is polymorphic, so it never picks up a promoting CAST, whether it is a bare
/// marker lane or a shaped one.
#[test]
fn stack_keeps_invalid_lanes_uncast() {
    let invalid = UOp::invalid_marker();
    let value = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let stack = UOp::stack(smallvec![invalid.clone(), value.clone()]);
    assert_eq!(stack.dtype(), DType::Float32);
    assert!(Arc::ptr_eq(&sources(&stack)[0], &invalid));
    assert!(Arc::ptr_eq(&sources(&stack)[1], &value));

    let shaped_invalid = invalid.try_reshape(&smallvec![1usize.into()]).unwrap();
    let shaped = UOp::stack(smallvec![shaped_invalid.clone(), UOp::stack(smallvec![value])]);
    assert_eq!(shaped.dtype(), DType::Float32);
    assert_eq!(shaped.shape().unwrap().unwrap().as_slice(), &[2usize.into(), 1usize.into()]);
    assert!(Arc::ptr_eq(&sources(&shaped)[0], &shaped_invalid));

    assert!(UOp::is_invalid_marker(&UOp::stack(smallvec![UOp::invalid_marker(), UOp::invalid_marker()])));
}

// =========================================================================
// Shape and indexing
// =========================================================================

/// STACK has the scalar lane dtype and adds a leading axis of the lane count, over scalar
/// and shaped lanes alike.
#[test]
fn stack_adds_a_leading_axis_of_the_lane_count() {
    let row = UOp::stack(smallvec![UOp::native_const(1i32), UOp::native_const(2i32)]);
    assert_eq!(row.dtype(), DType::Int32);
    assert_eq!(row.shape().unwrap().unwrap().as_slice(), &[2usize.into()]);

    let matrix = UOp::stack(smallvec![row.clone(), row]);
    assert_eq!(matrix.dtype(), DType::Int32);
    assert_eq!(matrix.shape().unwrap().unwrap().as_slice(), &[2usize.into(), 2usize.into()]);
}

#[test]
fn stack_constant_index_returns_lane() {
    let second = UOp::native_const(22i32);
    let stack = UOp::stack(smallvec![UOp::native_const(11i32), second.clone()]);
    let selected = UOp::index().buffer(stack).indices(vec![UOp::index_const(1)]).call().unwrap();
    assert!(Arc::ptr_eq(&selected, &second));
}

/// A shaped index adds its target axis before the source's trailing axes.
#[test]
fn shaped_index_selects_multiple_positions() {
    let vector = UOp::stack((10..50).step_by(10).map(UOp::native_const).collect());
    let result = vector.index_axes(vec![0, 2]);

    assert_eq!(result.dtype(), DType::Int32);
    assert_eq!(result.shape().unwrap().unwrap().as_slice(), &[2usize.into()]);
}

#[test]
fn stack_reconstruction_preserves_hash_cons_identity() {
    let stack = UOp::stack(smallvec![UOp::native_const(1i32), UOp::native_const(2i32)]);
    let rebuilt = stack.with_sources(stack.op().sources().into_vec());
    assert!(Arc::ptr_eq(&stack, &rebuilt));
    assert!(matches!(rebuilt.op(), Op::Stack(..)));
}
