//! Control flow operation tests.
//!
//! Tests control flow operations: If, EndIf, Range, End, Barrier.

use std::{f32::consts::PI, f64::consts::E};

use smallvec::smallvec;

use morok_dtype::DType;

use crate::{AxisId, AxisType, ConstValue, UOp};

// =========================================================================
// Basic If/EndIf Tests
// =========================================================================

#[test]
fn test_if_basic_construction() {
    let condition = UOp::native_const(true);

    let if_op = UOp::if_(condition.clone(), smallvec![UOp::native_const(1.0f32)]);

    assert_eq!(if_op.dtype(), DType::Void);
}

#[test]
fn test_if_with_comparison_condition() {
    // Create comparison condition: a < b
    let condition = UOp::native_const(5i32).try_cmplt(&UOp::native_const(10i32)).unwrap();
    assert_eq!(condition.dtype(), DType::Bool);

    let body_stmt = UOp::native_const(42.0f32);

    let if_op = UOp::if_(condition, smallvec![body_stmt]);

    assert_eq!(if_op.dtype(), DType::Void);
}

#[test]
fn test_if_empty_body() {
    let condition = UOp::native_const(false);

    let if_op = UOp::if_(condition, smallvec![]);

    assert_eq!(if_op.dtype(), DType::Void);
}

#[test]
fn test_if_single_statement_body() {
    let condition = UOp::native_const(true);
    let stmt = UOp::native_const(100i32);

    let if_op = UOp::if_(condition, smallvec![stmt]);

    assert_eq!(if_op.dtype(), DType::Void);
}

#[test]
fn test_if_multiple_statements() {
    let condition = UOp::native_const(true);
    let stmt1 = UOp::native_const(1i32);
    let stmt2 = UOp::native_const(2i32);
    let stmt3 = UOp::native_const(3i32);

    let if_op = UOp::if_(condition, smallvec![stmt1, stmt2, stmt3]);

    assert_eq!(if_op.dtype(), DType::Void);
}

#[test]
fn test_endif_basic() {
    let condition = UOp::native_const(true);
    let body_stmt = UOp::native_const(1.0f32);

    let if_op = UOp::if_(condition, smallvec![body_stmt]);

    let endif = UOp::endif(if_op.clone());

    assert_eq!(endif.dtype(), DType::Void);
}

#[test]
fn test_if_returns_void() {
    let condition = UOp::native_const(true);
    let body = smallvec![UOp::native_const(42i32)];

    let if_op = UOp::if_(condition, body);

    // Verify If has DType::Void
    assert_eq!(if_op.dtype(), DType::Void);
}

// =========================================================================
// Range Operation Tests
// =========================================================================

#[test]
fn test_range_global_axis() {
    let end = UOp::native_const(10i32);

    let range_op = UOp::range_axis(end, AxisId::Renumbered(0), AxisType::Global);

    assert_eq!(range_op.dtype(), DType::Index);
}

#[test]
fn test_range_warp_axis() {
    let end = UOp::native_const(32i32);

    let range_op = UOp::range_axis(end, AxisId::Renumbered(1), AxisType::Warp);

    assert_eq!(range_op.dtype(), DType::Index);
}

#[test]
fn test_range_local_axis() {
    let end = UOp::native_const(256i32);
    let range_op = UOp::range_axis(end, AxisId::Renumbered(0), AxisType::Local);
    assert_eq!(range_op.dtype(), DType::Index);
}

#[test]
fn test_range_loop_axis() {
    let end = UOp::native_const(100i32);
    let range_op = UOp::range(end, 2);
    assert_eq!(range_op.dtype(), DType::Index);
}

#[test]
fn test_range_reduce_axis() {
    let end = UOp::native_const(1024i32);
    let range_op = UOp::range_axis(end, AxisId::Renumbered(0), AxisType::Reduce);
    assert_eq!(range_op.dtype(), DType::Index);
}

#[test]
fn test_range_unroll_axis() {
    let end = UOp::native_const(4i32);
    let range_op = UOp::range_axis(end, AxisId::Renumbered(3), AxisType::Unroll);
    assert_eq!(range_op.dtype(), DType::Index);
}

#[test]
fn test_range_thread_axis() {
    let end = UOp::native_const(8i32);
    let range_op = UOp::range_axis(end, AxisId::Renumbered(1), AxisType::Thread);
    assert_eq!(range_op.dtype(), DType::Index);
}

#[test]
fn test_range_dtype_is_index() {
    // Verify all Range operations return DType::Index
    let end = UOp::native_const(10i32);
    let axis_types = vec![
        AxisType::Global,
        AxisType::Warp,
        AxisType::Local,
        AxisType::Loop,
        AxisType::GroupReduce,
        AxisType::Reduce,
        AxisType::Upcast,
        AxisType::Unroll,
        AxisType::Thread,
    ];

    for (idx, axis_type) in axis_types.into_iter().enumerate() {
        let range_op = UOp::range_axis(end.clone(), AxisId::Renumbered(idx), axis_type);

        assert_eq!(range_op.dtype(), DType::Index);
    }
}

// =========================================================================
// End Operation Tests
// =========================================================================

#[test]
fn test_end_of_range() {
    let end_val = UOp::native_const(10i32);
    let range_op = UOp::range(end_val, 0);

    // Create a simple computation (NOOP)
    let computation = UOp::noop();

    // Create END with computation and ranges
    let end_op = computation.end(smallvec![range_op]);

    assert_eq!(end_op.dtype(), DType::Void);
}

#[test]
fn test_end_preserves_dtype() {
    // End operation should have DType::Void
    let end_val = UOp::native_const(5i32);
    let range_op = UOp::range_axis(end_val, AxisId::Renumbered(0), AxisType::Global);

    // Create a simple computation (NOOP)
    let computation = UOp::noop();

    // Create END with computation and ranges
    let end_op = computation.end(smallvec![range_op]);

    assert_eq!(end_op.dtype(), DType::Void);
}

#[test]
fn test_end_returns_void() {
    let end_val = UOp::native_const(100i32);
    let range_op = UOp::range_axis(end_val, AxisId::Renumbered(1), AxisType::Reduce);

    // Create a simple computation (NOOP)
    let computation = UOp::noop();

    // Create END with computation and ranges
    let end_op = computation.end(smallvec![range_op]);

    // Verify End has DType::Void
    assert_eq!(end_op.dtype(), DType::Void);
}

// =========================================================================
// Barrier Tests
// =========================================================================

#[test]
fn test_barrier_basic() {
    let src = UOp::native_const(1.0f32);

    let barrier = src.barrier(smallvec![]);

    // Barrier preserves src dtype
    assert_eq!(barrier.dtype(), DType::Float32);
}

#[test]
fn test_barrier_with_single_dep() {
    let src = UOp::native_const(42i32);
    let dep = UOp::native_const(PI);

    let barrier = src.barrier(smallvec![dep]);

    assert_eq!(barrier.dtype(), DType::Int32);
}

#[test]
fn test_barrier_with_multiple_deps() {
    let src = UOp::native_const(E);
    let dep1 = UOp::native_const(1i32);
    let dep2 = UOp::native_const(2i32);
    let dep3 = UOp::native_const(3i32);

    let barrier = src.barrier(smallvec![dep1, dep2, dep3]);

    assert_eq!(barrier.dtype(), DType::Float64);
}

#[test]
fn test_barrier_preserves_dtype() {
    // Test that Barrier preserves various dtypes
    let dtypes = vec![
        (DType::Int8, ConstValue::Int(1)),
        (DType::Int32, ConstValue::Int(100)),
        (DType::Float32, ConstValue::Float(PI as f64)),
        (DType::Float64, ConstValue::Float(E)),
        (DType::UInt32, ConstValue::UInt(42)),
    ];

    for (dtype, value) in dtypes {
        let src = UOp::const_(dtype.clone(), value);
        let barrier = src.barrier(smallvec![]);

        assert_eq!(barrier.dtype(), dtype);
    }
}

// =========================================================================
// Nested Control Flow
// =========================================================================

#[test]
fn test_nested_if() {
    let outer_cond = UOp::native_const(true);
    let inner_cond = UOp::native_const(false);

    let inner_body = UOp::native_const(42i32);

    let inner_if = UOp::if_(inner_cond, smallvec![inner_body]);

    // Outer if contains inner if in its body
    let outer_if = UOp::if_(outer_cond, smallvec![inner_if]);

    assert_eq!(outer_if.dtype(), DType::Void);
}

#[test]
fn test_range_inside_if() {
    let condition = UOp::native_const(true);
    let range_end = UOp::native_const(10i32);

    let range_op = UOp::range(range_end, 0);

    let if_op = UOp::if_(condition, smallvec![range_op]);

    assert_eq!(if_op.dtype(), DType::Void);
}

#[test]
fn test_multiple_sequential_ranges() {
    let end1 = UOp::native_const(10i32);
    let end2 = UOp::native_const(20i32);
    let end3 = UOp::native_const(30i32);

    let range1 = UOp::range_axis(end1, AxisId::Renumbered(0), AxisType::Global);

    let range2 = UOp::range_axis(end2, AxisId::Renumbered(1), AxisType::Local);

    let range3 = UOp::range(end3, 2);

    // All ranges should be valid
    assert_eq!(range1.dtype(), DType::Index);
    assert_eq!(range2.dtype(), DType::Index);
    assert_eq!(range3.dtype(), DType::Index);
}

// =========================================================================
// Dtype Validation
// =========================================================================

#[test]
fn test_if_dtype_is_void() {
    let condition = UOp::native_const(true);
    let body = smallvec![UOp::native_const(1.0f32)];

    let if_op = UOp::if_(condition, body);

    // Confirm If dtype
    assert_eq!(if_op.dtype(), DType::Void);
}

#[test]
fn test_endif_dtype_is_void() {
    let condition = UOp::native_const(true);
    let if_op = UOp::if_(condition, smallvec![]);

    let endif = UOp::endif(if_op);

    // Confirm EndIf dtype
    assert_eq!(endif.dtype(), DType::Void);
}

#[test]
fn test_range_confirms_index_dtype() {
    let end = UOp::native_const(100i32);

    let range_op = UOp::range(end, 0);

    // Confirm Range dtype
    assert_eq!(range_op.dtype(), DType::Index);
}

#[test]
fn test_end_dtype_is_void() {
    let end_val = UOp::native_const(10i32);
    let range_op = UOp::range_axis(end_val, AxisId::Renumbered(0), AxisType::Global);

    // Create a simple computation (NOOP)
    let computation = UOp::noop();

    // Create END with computation and ranges
    let end_op = computation.end(smallvec![range_op]);

    // Confirm End dtype
    assert_eq!(end_op.dtype(), DType::Void);
}

#[test]
fn test_barrier_dtype_preservation() {
    // Test that Barrier preserves src dtype across different types
    let int_src = UOp::native_const(42i32);
    let int_barrier = int_src.barrier(smallvec![]);
    assert_eq!(int_barrier.dtype(), DType::Int32);

    let float_src = UOp::native_const(PI);
    let float_barrier = float_src.barrier(smallvec![]);
    assert_eq!(float_barrier.dtype(), DType::Float32);
}
